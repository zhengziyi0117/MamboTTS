from concurrent.futures import ThreadPoolExecutor
import threading
import unittest

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from engine_progress import EngineProgress, install_progress_api


class EngineProgressTests(unittest.TestCase):
    def test_http_job_context_reaches_stream_worker_and_progress_is_live(self):
        app = FastAPI()
        progress = EngineProgress()
        gate = threading.Event()
        entered = threading.Event()

        @app.post("/")
        async def generate():
            def audio():
                progress.update("semantic", total_segments=2, current_tokens=32)
                progress.segment_done()
                entered.set()
                gate.wait(timeout=3)
                progress.update("vocoder")
                progress.segment_done()
                yield b"audio"
            return StreamingResponse(audio())

        install_progress_api(app, progress)
        job_id = "a" * 32
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(client.post, "/", headers={"X-Mambo-Job-ID": job_id})
            try:
                self.assertTrue(entered.wait(timeout=2))
                live = client.get(f"/mambo/progress/{job_id}").json()["progress"]
                self.assertEqual(live["stage"], "semantic")
                self.assertEqual(live["completed_segments"], 1)
                self.assertEqual(live["current_tokens"], 32)
            finally:
                gate.set()
            self.assertEqual(future.result(timeout=3).content, b"audio")
            final = client.get(f"/mambo/progress/{job_id}").json()["progress"]
            self.assertEqual(final["stage"], "done")
            self.assertEqual(final["completed_segments"], 2)
            self.assertIsNone(client.get("/mambo/progress/unknown").json()["progress"])
        self.assertIsNone(progress.current.get())

    def test_http_error_is_not_reported_as_completed(self):
        app = FastAPI()
        progress = EngineProgress()

        @app.post("/")
        async def generate():
            raise RuntimeError("model failed")

        install_progress_api(app, progress)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/", headers={"X-Mambo-Job-ID": "b" * 32})
            self.assertEqual(response.status_code, 500)
            failed = client.get("/mambo/progress/" + "b" * 32).json()["progress"]
            self.assertEqual(failed["stage"], "error")
            self.assertEqual(failed["completed_segments"], 0)

    def test_incomplete_audio_is_not_reported_as_completed_even_with_http_200(self):
        app = FastAPI()
        progress = EngineProgress()

        @app.post("/")
        async def generate():
            progress.update("vocoder", total_segments=2)
            progress.segment_done()
            return "partial audio"

        install_progress_api(app, progress)
        with TestClient(app) as client:
            response = client.post("/", headers={"X-Mambo-Job-ID": "c" * 32})
            self.assertEqual(response.status_code, 200)
            partial = progress.get("c" * 32)
            self.assertEqual(partial["stage"], "error")
            self.assertEqual(partial["completed_segments"], 1)


if __name__ == "__main__":
    unittest.main()
