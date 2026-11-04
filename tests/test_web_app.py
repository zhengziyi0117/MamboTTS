import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import wave

from fastapi.testclient import TestClient

from web_app import create_app


class FakeEngine:
    def __init__(self):
        self.online = True
        self.fail = False
        self.gate = None
        self.started = threading.Event()
        self.last_options = {}
        self.progress = None

    def get_progress(self, job_id):
        return self.progress

    def is_api_running(self, **kwargs):
        return self.online

    def generate_speech(self, text, path, **kwargs):
        self.last_options = kwargs
        self.started.set()
        if self.gate:
            self.gate.wait(timeout=5)
        if self.fail:
            Path(path).write_bytes(b"partial response")
            return False, "测试引擎合成失败"
        with wave.open(path, "wb") as output:
            output.setparams((1, 2, 32000, 0, "NONE", "not compressed"))
            output.writeframes(b"\x00\x10" * 3200)
        return True, "OK"


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = FakeEngine()
        self.app = create_app(self.temp.name, client=self.engine)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        if self.engine.gate:
            self.engine.gate.set()
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def generate(self, text="测试配音"):
        response = self.client.post("/api/generate", json={"text": text, "speed": 1})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["id"]

    def wait_job(self, job_id):
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"done", "error"}:
                return job
            time.sleep(0.01)
        self.fail("Synthesis job did not finish")

    def test_page_and_offline_engine(self):
        self.assertIn("配音工作台", self.client.get("/").text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.engine.online = False
        self.assertEqual(self.client.get("/api/status").json()["state"], "offline")
        self.assertEqual(self.client.post("/api/generate", json={"text": "测试"}).status_code, 503)

    def test_generation_history_playback_download_and_range(self):
        job = self.wait_job(self.generate())
        self.assertEqual(job["status"], "done")
        record = job["record"]
        self.assertEqual(self.engine.last_options["segmentation"], "sentence")
        self.assertEqual(record["segmentation"], "sentence")
        self.assertEqual(self.client.get("/api/history").json()[0]["id"], record["id"])
        response = self.client.get(record["audio_url"])
        self.assertEqual(response.content[:4], b"RIFF")
        self.assertIn("inline", response.headers["content-disposition"])
        response = self.client.get(record["download_url"])
        self.assertIn("attachment", response.headers["content-disposition"])
        response = self.client.get(record["audio_url"], headers={"Range": "bytes=0-43"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(len(response.content), 44)
        restored = create_app(self.temp.name, client=self.engine)
        self.assertEqual(restored.state.jobs.history()[0]["id"], record["id"])

    def test_invalid_input_is_rejected(self):
        for data in ({"text": " "}, {"text": "x" * 2001}, {"text": "x", "speed": 0},
                     {"text": "x", "speed": "NaN"}, {"text": "x", "language": "unknown"},
                     {"text": "，。！？"}, {"text": "x", "segmentation": "unknown"}):
            with self.subTest(data=str(data)[:80]):
                self.assertEqual(self.client.post("/api/generate", json=data).status_code, 422)

    def test_phrasing_choice_is_used_and_persisted_with_actual_segments(self):
        text = "城市一夜冰封，我才明白那不是bug"
        response = self.client.post("/api/generate", json={"text": text, "segmentation": "punctuation"})
        self.assertEqual(response.status_code, 202)
        job = self.wait_job(response.json()["id"])
        self.assertEqual(job["status"], "done")
        self.assertEqual(self.engine.last_options["segmentation"], "punctuation")
        self.assertEqual(job["record"]["segments"], ["城市一夜冰封，", "我才明白那不是bug"])
        restored = create_app(self.temp.name, client=self.engine)
        self.assertEqual(restored.state.jobs.history()[0]["segmentation"], "punctuation")

    def test_failed_audio_is_cleaned_and_not_in_history(self):
        self.engine.fail = True
        job = self.wait_job(self.generate())
        self.assertEqual(job["status"], "error")
        self.assertIn("合成失败", job["error"])
        self.assertEqual(self.client.get("/api/history").json(), [])
        self.assertEqual(list(Path(self.temp.name).glob("*.wav")), [])

    def test_queue_has_a_bound_and_never_runs_two_jobs_at_once(self):
        self.engine.gate = threading.Event()
        self.generate()
        self.assertTrue(self.engine.started.wait(timeout=1))
        self.generate(); self.generate(); self.generate()
        response = self.client.post("/api/generate", json={"text": "第五个任务"})
        self.assertEqual(response.status_code, 429)
        statuses = [job["status"] for job in self.app.state.jobs.jobs.values()]
        self.assertEqual(statuses.count("running"), 1)

    def test_live_progress_queue_position_and_final_timings(self):
        self.engine.gate = threading.Event()
        self.engine.progress = {"stage": "semantic", "total_segments": 2,
                                "completed_segments": 1, "current_tokens": 64,
                                "timings": {"text": 0.12, "semantic": 1.3}}
        first = self.generate()
        self.assertTrue(self.engine.started.wait(timeout=1))
        self.assertEqual(self.engine.last_options["request_id"], first)
        second = self.generate()
        running = self.client.get(f"/api/jobs/{first}").json()
        self.assertEqual(running["progress"]["completed_segments"], 1)
        self.assertEqual(running["progress"]["current_tokens"], 64)
        queued = self.client.get(f"/api/jobs/{second}").json()
        self.assertEqual(queued["status"], "queued")
        self.assertEqual(queued["queue_ahead"], 1)
        self.engine.gate.set()
        final = self.wait_job(first)
        self.assertEqual(final["record"]["timings"]["semantic"], 1.3)

    def test_unknown_audio_and_path_traversal_are_not_served(self):
        self.assertEqual(self.client.get("/api/audio/" + "a" * 32).status_code, 404)
        self.assertEqual(self.client.get("/api/audio/%2e%2e%2fconfig.json").status_code, 404)
        self.assertEqual(self.client.get("/api/jobs/missing").status_code, 404)

    def test_invalid_history_does_not_prevent_startup(self):
        path = Path(self.temp.name) / "web-history.json"
        path.write_text(json.dumps([{"id": "../secret", "text": "bad"}]))
        restored = create_app(self.temp.name, client=self.engine)
        self.assertEqual(restored.state.jobs.history(), [])

    def test_web_shutdown_stops_only_the_engine_it_started(self):
        engine = FakeEngine()
        engine.online = False
        app = create_app(self.temp.name, client=engine,
                         engine_command=[sys.executable, "-c", "import time; time.sleep(60)"])
        with TestClient(app):
            process = app.state.engine
            self.assertIsNotNone(process)
            self.assertIsNone(process.poll())
        self.assertIsNotNone(process.poll())

        engine.online = True
        reused = create_app(self.temp.name, client=engine,
                            engine_command=[sys.executable, "-c", "raise RuntimeError('must not start')"])
        with TestClient(reused):
            self.assertIsNone(reused.state.engine)
        self.assertTrue(engine.online)


if __name__ == "__main__":
    unittest.main()
