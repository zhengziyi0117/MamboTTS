"""手动启动的 MamboTTS Web 工作台；Ctrl+C 同时关闭它自己启动的引擎。"""
import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from engine_contract import API_PORT
from file_naming import filename_from_text
from inference import TTSClient
from run_engine import engine_env, preflight_check
from text_processing import SEGMENTATION_MODES, split_speech_text

BASE = Path(__file__).resolve().parent
logger = logging.getLogger("mambotts.web")
RECORD_ID = re.compile(r"^[0-9a-f]{32}$")


class GenerateRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    text: str = Field(min_length=1, max_length=2000)
    speed: float = Field(default=1.0, ge=0.1, le=3.0)
    language: str = "zh"
    segmentation: str = "sentence"

    @field_validator("text")
    @classmethod
    def trim_text(cls, value):
        if not value.strip() or not split_speech_text(value):
            raise ValueError("请输入配音文案")
        return value.strip()

    @field_validator("language")
    @classmethod
    def check_language(cls, value):
        if value not in {"zh", "en", "ja", "auto"}:
            raise ValueError("不支持的语言")
        return value

    @field_validator("segmentation")
    @classmethod
    def check_segmentation(cls, value):
        if value not in SEGMENTATION_MODES:
            raise ValueError("不支持的断句方式")
        return value


class SpeechJobs:
    """单个推理队列，避免多个浏览器同时占满核显内存。"""
    def __init__(self, output_dir, client):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "web-history.json"
        self.client = client
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.queue = queue.Queue()
        self.jobs = {}
        self.records = self._load_history()
        self.worker = threading.Thread(target=self._work, name="mambo-web-synthesis", daemon=True)

    def _load_history(self):
        try:
            records = json.loads(self.history_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                return []
        except (OSError, ValueError):
            return []
        return [r for r in records if isinstance(r, dict)
                and isinstance(r.get("id"), str) and RECORD_ID.fullmatch(r["id"])
                and isinstance(r.get("text"), str)
                and self.audio_path(r["id"]).is_file()][:50]

    def audio_path(self, record_id):
        if not RECORD_ID.fullmatch(record_id):
            raise ValueError("无效的音频编号")
        path = self.output_dir / f"{record_id}.wav"
        if not path.resolve().is_relative_to(self.output_dir):
            raise ValueError("音频路径超出输出目录")
        return path

    def public_record(self, record):
        return {**record, "audio_url": f"/api/audio/{record['id']}",
                "download_url": f"/api/audio/{record['id']}?download=1",
                "name": filename_from_text(record["text"])}

    def history(self):
        with self.lock:
            return [self.public_record(r) for r in self.records if self.audio_path(r["id"]).is_file()]

    def submit(self, request):
        with self.lock:
            if self.stopping.is_set():
                raise HTTPException(503, "工作台正在关闭")
            active = sum(j["status"] in {"queued", "running"} for j in self.jobs.values())
            if active >= 4:
                raise HTTPException(429, "当前生成任务较多，请稍后重试")
            # 保留最近任务供页面恢复轮询，运行中的任务不会被清理。
            if len(self.jobs) >= 100:
                completed = [key for key, j in self.jobs.items() if j["status"] in {"done", "error"}]
                for key in completed[:50]:
                    del self.jobs[key]
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {"id": job_id, "status": "queued", "elapsed": 0}
            self.queue.put((job_id, request))
            return dict(self.jobs[job_id])

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, "任务不存在，请重新生成")
            job = dict(self.jobs[job_id])
            start = job.pop("started", None)
            if start is not None and job["status"] == "running":
                job["elapsed"] = round(time.monotonic() - start, 1)
            if job["status"] == "queued":
                preceding = list(self.jobs)[:list(self.jobs).index(job_id)]
                job["queue_ahead"] = sum(self.jobs[key]["status"] in {"queued", "running"}
                                         for key in preceding)
        if job["status"] == "running":
            job["progress"] = self._engine_progress(job_id)
        return job

    def _engine_progress(self, job_id):
        getter = getattr(self.client, "get_progress", None)
        return getter(job_id) if getter else None

    def _work(self):
        while not self.stopping.is_set():
            try:
                job_id, request = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.monotonic()
            path = self.audio_path(job_id)
            with self.lock:
                self.jobs[job_id].update(status="running", started=started)
            try:
                ok, message = self.client.generate_speech(
                    request.text, str(path), text_lang=request.language, speed=request.speed,
                    segmentation=request.segmentation, request_id=job_id)
                if not ok:
                    raise RuntimeError(message)
                with wave.open(str(path), "rb") as audio:
                    duration = audio.getnframes() / audio.getframerate()
                    if duration <= 0:
                        raise RuntimeError("引擎返回了空音频，请重新生成")
                record = {"id": job_id, "text": request.text, "speed": request.speed,
                          "segmentation": request.segmentation,
                          "segments": split_speech_text(request.text, request.segmentation),
                          "language": request.language, "duration": round(duration, 2),
                          "elapsed": round(time.monotonic() - started, 2),
                          "created_at": datetime.now(timezone.utc).isoformat()}
                progress = self._engine_progress(job_id)
                if progress and progress.get("timings"):
                    record["timings"] = progress["timings"]
                with self.lock:
                    records = [record, *self.records][:50]
                    tmp = self.history_path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
                    tmp.replace(self.history_path)
                    self.records = records
                    self.jobs[job_id].update(status="done", elapsed=record["elapsed"],
                                             record=self.public_record(record))
            except Exception as exc:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("无法移除失败的音频文件: %s", path)
                logger.warning("合成失败: %s", exc)
                with self.lock:
                    self.jobs[job_id].update(status="error", error=str(exc) or "返回音频无效，请重试",
                                             elapsed=round(time.monotonic() - started, 2))
            finally:
                self.queue.task_done()


def stop_owned_engine(proc):
    if proc is None:
        return
    try:
        if os.name == "nt":
            if proc.poll() is None:
                proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except ProcessLookupError:
        proc.wait()
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait(timeout=3)


def create_app(output_dir=None, client=None, engine_command=None):
    client = client or TTSClient()
    jobs = SpeechJobs(output_dir or BASE / "outputs" / "web", client)

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = None
        if engine_command and not await asyncio.to_thread(client.is_api_running):
            app.state.engine = subprocess.Popen(engine_command, cwd=BASE,
                                                start_new_session=os.name != "nt", env=engine_env())
        jobs.worker.start()
        try:
            yield
        finally:
            jobs.stopping.set()
            await asyncio.to_thread(stop_owned_engine, app.state.engine)
            await asyncio.to_thread(jobs.worker.join, 2)

    app = FastAPI(title="MamboTTS Web", lifespan=lifespan)
    app.state.jobs = jobs
    app.mount("/static", StaticFiles(directory=BASE / "web"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(BASE / "web" / "index.html")

    @app.get("/api/status")
    def status():
        online = client.is_api_running(timeout=1)
        engine = app.state.engine
        code = engine.poll() if engine is not None else None
        state = "online" if online else "loading" if engine is not None and code is None else "offline"
        with jobs.lock:
            pending = sum(j["status"] in {"queued", "running"} for j in jobs.jobs.values())
        return {"state": state, "pending": pending}

    @app.get("/api/history")
    def history():
        return jobs.history()

    @app.post("/api/generate", status_code=202)
    def generate(request: GenerateRequest):
        if not client.is_api_running(timeout=1):
            raise HTTPException(503, "引擎尚未就绪，请稍候；如果一直离线，请查看启动终端的日志")
        return jobs.submit(request)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return jobs.get(job_id)

    @app.get("/api/audio/{record_id}")
    def audio(record_id: str, download: bool = False):
        with jobs.lock:
            record = next((r for r in jobs.records if r["id"] == record_id), None)
        if record is None:
            raise HTTPException(404, "音频不存在")
        try:
            path = jobs.audio_path(record_id)
        except ValueError:
            raise HTTPException(404, "音频不存在")
        if not path.is_file():
            raise HTTPException(404, "音频文件已被移除")
        return FileResponse(path, media_type="audio/wav", filename=filename_from_text(record["text"]),
                            content_disposition_type="attachment" if download else "inline")

    return app


def main():
    parser = argparse.ArgumentParser(description="MamboTTS Web：生成、试听与下载配音")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--engine-port", type=int, default=API_PORT)
    parser.add_argument("--device", choices=("auto", "cpu", "rocm", "cuda"),
                        default=os.environ.get("MAMBO_DEVICE", "auto"))
    parser.add_argument("--precision", choices=("auto", "full", "half"),
                        default=os.environ.get("MAMBO_PRECISION", "auto"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--no-engine", action="store_true", help="连接已手动启动的引擎")
    parser.add_argument("--check", action="store_true", help="只检查引擎和设备后退出")
    args = parser.parse_args()
    if not all(1 <= p <= 65535 for p in (args.port, args.engine_port)):
        parser.error("端口须为 1–65535")
    if args.port == args.engine_port:
        parser.error("Web 端口和引擎端口须不同")
    command = [sys.executable, "-u", str(BASE / "run_engine.py"), "--host", "127.0.0.1",
               "--port", str(args.engine_port), "--device", args.device, "--precision", args.precision]
    if args.threads is not None:
        if args.threads < 1:
            parser.error("线程数须为正整数")
        command.extend(("--threads", str(args.threads)))
    if args.check:
        return subprocess.call([*command, "--check"], cwd=BASE)
    if not args.no_engine:
        missing = preflight_check()
        if missing:
            parser.error("引擎文件不完整，请先运行 ./install.sh：\n" + "\n".join(missing))
    client = TTSClient(f"http://127.0.0.1:{args.engine_port}")
    app = create_app(client=client, engine_command=None if args.no_engine else command)
    import uvicorn
    print(f"[Web] 在浏览器打开 http://127.0.0.1:{args.port}", flush=True)
    if args.host == "0.0.0.0":
        print(f"[Web] 局域网访问：http://<小主机 IP>:{args.port}", flush=True)
    print("[Web] 按 Ctrl+C 退出，随之关闭本次启动的引擎。", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", timeout_graceful_shutdown=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
