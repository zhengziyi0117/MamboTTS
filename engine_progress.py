"""来自引擎实际执行阶段的进度；不根据时间猜百分比。"""
from contextvars import ContextVar
import re
import threading
import time

JOB_ID = re.compile(r"^[0-9a-f]{32}$")


class EngineProgress:
    def __init__(self):
        self.current = ContextVar("mambo_progress_job", default=None)
        self.lock = threading.RLock()
        self.jobs = {}

    def start(self, job_id):
        now = time.monotonic()
        with self.lock:
            if len(self.jobs) >= 100:
                finished = [key for key, value in self.jobs.items() if value["stage"] in {"done", "error"}]
                for key in finished[:50]:
                    del self.jobs[key]
            if len(self.jobs) >= 100:
                return False
            self.jobs[job_id] = {"stage": "preparing", "completed_segments": 0,
                                 "total_segments": 0, "current_tokens": 0,
                                 "timings": {}, "_started": now, "_phase_started": now}
            return True

    def update(self, stage=None, **fields):
        job_id = self.current.get()
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            now = time.monotonic()
            if stage is not None and stage != job["stage"]:
                previous = job["stage"]
                job["timings"][previous] = job["timings"].get(previous, 0) + now - job["_phase_started"]
                job["stage"] = stage
                job["_phase_started"] = now
            job.update(fields)

    def segment_done(self):
        with self.lock:
            job = self.jobs.get(self.current.get())
            if job is not None:
                job["completed_segments"] = min(job["total_segments"], job["completed_segments"] + 1)

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            timings = dict(job["timings"])
            now = time.monotonic()
            if job["stage"] not in {"done", "error"}:
                timings[job["stage"]] = timings.get(job["stage"], 0) + now - job["_phase_started"]
            return {key: value for key, value in job.items() if not key.startswith("_") and key != "timings"} | {
                "timings": {key: round(value, 3) for key, value in timings.items()}}


def install_progress_api(app, progress):
    class ProgressMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            job_id = None
            if scope["type"] == "http" and scope["path"] == "/":
                headers = dict(scope.get("headers", []))
                candidate = headers.get(b"x-mambo-job-id", b"").decode("ascii", errors="ignore")
                if JOB_ID.fullmatch(candidate) and progress.start(candidate):
                    job_id = candidate
            if job_id is None:
                return await self.app(scope, receive, send)
            token = progress.current.set(job_id)
            failed = False

            async def observe(message):
                nonlocal failed
                if message["type"] == "http.response.start" and message["status"] >= 400:
                    failed = True
                await send(message)

            try:
                await self.app(scope, receive, observe)
            except BaseException:
                failed = True
                raise
            finally:
                final = progress.get(job_id)
                complete = final["total_segments"] > 0 and final["completed_segments"] == final["total_segments"]
                progress.update("done" if not failed and complete else "error")
                progress.current.reset(token)

    app.add_middleware(ProgressMiddleware)

    @app.get("/mambo/progress/{job_id}", include_in_schema=False)
    async def synthesis_progress(job_id: str):
        return {"progress": progress.get(job_id) if JOB_ID.fullmatch(job_id) else None}
