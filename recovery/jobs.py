"""Running a counterfactual experiment from the dashboard, without wrecking it.

The obvious implementation -- call the experiment in a thread -- is wrong here,
and the reason is worth writing down. The whole simulation runs off Python's
global `random`, and every run reseeds it. The gateway is already running a live
simulation on a background thread, so an in-process experiment would reset the
RNG underneath it: the live feed would jump, and both results would be built on
a stream neither of them controlled.

So a job is a subprocess. A separate interpreter has its own `random` module and
cannot reach the gateway's, which also means a crash in an experiment cannot take
the web server with it.

One job at a time. A run is CPU-bound for tens of seconds and the box is already
carrying a live simulation; letting a dashboard queue up ten of them would starve
the thing people are watching.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field

# A dashboard control must not be able to start an unbounded run.
MIN_SESSIONS = 1000
MAX_SESSIONS = 20000
DEFAULT_SESSIONS = 3000

# A run that has not finished by now is wedged; the slot is released so the
# dashboard does not lock up permanently on one bad job.
JOB_TIMEOUT_SECONDS = 900

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Job:
    job_id: str
    params: dict
    status: str = "running"          # running | done | error
    result: dict | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return round(time.time() - self.started_at, 1)

    def summary(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "params": self.params,
            "elapsed_seconds": self.elapsed,
            "result": self.result,
            "error": self.error,
        }


class JobRunner:
    """Owns at most one running counterfactual, plus the finished ones."""

    def __init__(self, python_executable=None, cwd=None):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._active: str | None = None
        self._python = python_executable or sys.executable
        self._cwd = cwd or REPO_ROOT

    # -- accounting -----------------------------------------------------------
    def get(self, job_id) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None

    def _release(self, job_id):
        with self._lock:
            if self._active == job_id:
                self._active = None

    # -- submission -----------------------------------------------------------
    def submit(self, discount=0.15, latency=0.5, sessions=DEFAULT_SESSIONS,
               triggers=None) -> dict:
        """Start a run, or refuse politely if one is already going."""
        if not 0.0 <= discount <= 0.95:
            return {"accepted": False, "reason": "discount must be between 0 and 0.95"}
        if latency < 0:
            return {"accepted": False, "reason": "latency must not be negative"}

        sessions = max(MIN_SESSIONS, min(int(sessions), MAX_SESSIONS))

        with self._lock:
            if self._active is not None:
                return {"accepted": False, "reason": "a run is already in progress",
                        "job_id": self._active}
            job = Job(job_id=uuid.uuid4().hex[:12],
                      params={"discount": discount, "latency": latency,
                              "sessions": sessions,
                              "triggers": sorted(triggers) if triggers else None})
            self._jobs[job.job_id] = job
            self._active = job.job_id

        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return {"accepted": True, "job_id": job.job_id, "params": job.params}

    def _command(self, job, out_path):
        command = [
            self._python, "-m", "sim.experiment",
            "--sessions", str(job.params["sessions"]),
            "--discount", str(job.params["discount"]),
            "--latency", str(job.params["latency"]),
            "--budget-aware",
            "--out", out_path,
            # A dashboard experiment must not write into the real audit trail;
            # the published record is not a scratchpad.
            "--audit", os.path.join(tempfile.gettempdir(), f"cf_{job.job_id}.db"),
        ]
        if job.params["triggers"]:
            command += ["--triggers", ",".join(job.params["triggers"])]
        return command

    def _run(self, job):
        out_path = os.path.join(tempfile.gettempdir(), f"cf_{job.job_id}.json")
        try:
            completed = subprocess.run(
                self._command(job, out_path),
                cwd=self._cwd, capture_output=True, text=True,
                timeout=JOB_TIMEOUT_SECONDS,
            )
            if completed.returncode != 0:
                job.status = "error"
                job.error = (completed.stderr or "experiment failed").strip()[-600:]
                return

            with open(out_path, encoding="utf-8") as handle:
                job.result = json.load(handle)
            job.status = "done"
        except subprocess.TimeoutExpired:
            job.status = "error"
            job.error = f"timed out after {JOB_TIMEOUT_SECONDS}s"
        except Exception as exc:                      # noqa: BLE001 - reported, not raised
            job.status = "error"
            job.error = str(exc)
        finally:
            self._release(job.job_id)
            for path in (out_path,
                         os.path.join(tempfile.gettempdir(), f"cf_{job.job_id}.db")):
                try:
                    os.remove(path)
                except OSError:
                    pass
