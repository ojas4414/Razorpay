"""The counterfactual job runner.

Two properties here are not cosmetic. A dashboard run must go to a scratch
database rather than the published audit trail -- the record is not a
scratchpad. And a run must happen in a subprocess, because the simulation
reseeds the global RNG and doing that in-process would reset the stream the
gateway's live simulation thread is running on.

The subprocess itself is faked: these tests check the contract around it -- what
is launched, what happens on failure -- not that the experiment produces a
particular number, which is what the experiment's own tests are for.
"""
import json
import os
import subprocess
import time

import pytest

from recovery.jobs import (
    JOB_TIMEOUT_SECONDS,
    MAX_SESSIONS,
    MIN_SESSIONS,
    Job,
    JobRunner,
)


def wait_for(runner, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = runner.get(job_id)
        if job and job.status != "running":
            return job
        time.sleep(0.02)
    return runner.get(job_id)


class TestValidation:
    def test_rejects_an_impossible_discount(self):
        runner = JobRunner()
        assert runner.submit(discount=1.4)["accepted"] is False
        assert runner.submit(discount=-0.2)["accepted"] is False

    def test_rejects_negative_latency(self):
        assert JobRunner().submit(latency=-1)["accepted"] is False

    def test_clamps_session_count(self, monkeypatch):
        """A slider must not be able to start an unbounded run."""
        runner = JobRunner()
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_result())
        low = runner.submit(sessions=5)
        assert low["params"]["sessions"] == MIN_SESSIONS
        wait_for(runner, low["job_id"])
        high = runner.submit(sessions=10_000_000)
        assert high["params"]["sessions"] == MAX_SESSIONS


class TestConcurrency:
    def test_only_one_run_at_a_time(self, monkeypatch):
        """A run is CPU-bound and the box is already carrying a live simulation."""
        def slow(*args, **kwargs):
            time.sleep(0.4)
            return _ok_result()

        monkeypatch.setattr(subprocess, "run", slow)
        runner = JobRunner()
        first = runner.submit()
        assert first["accepted"] is True
        second = runner.submit()
        assert second["accepted"] is False
        assert "already in progress" in second["reason"]
        assert second["job_id"] == first["job_id"]

    def test_slot_is_released_when_a_run_finishes(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_result())
        runner = JobRunner()
        first = runner.submit()
        wait_for(runner, first["job_id"])
        assert runner.busy is False
        assert runner.submit()["accepted"] is True

    def test_slot_is_released_after_a_failure(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: _fail_result("boom"))
        runner = JobRunner()
        job = runner.submit()
        wait_for(runner, job["job_id"])
        assert runner.busy is False


class TestCommand:
    def test_runs_the_experiment_in_a_subprocess(self):
        """In-process would reset the RNG the live simulation is using."""
        runner = JobRunner()
        job = Job(job_id="abc123", params={"discount": 0.2, "latency": 1.5,
                                           "sessions": 4000, "triggers": None})
        command = runner._command(job, "/tmp/out.json")
        assert "-m" in command and "sim.experiment" in command
        assert "--discount" in command and "0.2" in command
        assert "--sessions" in command and "4000" in command
        assert "--budget-aware" in command

    def test_never_writes_to_the_published_audit_trail(self):
        """A dashboard experiment is not allowed to touch the real record."""
        runner = JobRunner()
        job = Job(job_id="abc123", params={"discount": 0.15, "latency": 0.5,
                                           "sessions": 3000, "triggers": None})
        command = runner._command(job, "/tmp/out.json")
        audit_path = command[command.index("--audit") + 1]
        assert "recovery_audit.db" not in audit_path
        assert "abc123" in audit_path

    def test_passes_targeting_through(self):
        runner = JobRunner()
        job = Job(job_id="x", params={"discount": 0.15, "latency": 0.5,
                                      "sessions": 3000,
                                      "triggers": ["price_shock"]})
        command = runner._command(job, "/tmp/out.json")
        assert "--triggers" in command
        assert "price_shock" in command


class TestLifecycle:
    def test_a_successful_run_carries_its_result(self, monkeypatch, tmp_path):
        payload = {"test": {"delta": 0.03, "significant": True},
                   "margin": {"net": 500.0}}

        def fake_run(command, **kwargs):
            out = command[command.index("--out") + 1]
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            return _ok_result()

        monkeypatch.setattr(subprocess, "run", fake_run)
        runner = JobRunner()
        submitted = runner.submit(discount=0.2)
        job = wait_for(runner, submitted["job_id"])
        assert job.status == "done"
        assert job.result["test"]["delta"] == 0.03

    def test_a_failed_run_reports_stderr_rather_than_raising(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: _fail_result("traceback: exploded"))
        runner = JobRunner()
        job = wait_for(runner, runner.submit()["job_id"])
        assert job.status == "error"
        assert "exploded" in job.error

    def test_a_hung_run_times_out(self, monkeypatch):
        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="x", timeout=JOB_TIMEOUT_SECONDS)

        monkeypatch.setattr(subprocess, "run", timeout)
        runner = JobRunner()
        job = wait_for(runner, runner.submit()["job_id"])
        assert job.status == "error"
        assert "timed out" in job.error

    def test_unknown_job_is_none(self):
        assert JobRunner().get("nope") is None

    def test_summary_is_serialisable(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_result())
        runner = JobRunner()
        job = wait_for(runner, runner.submit()["job_id"])
        json.dumps(job.summary())


# -- helpers ------------------------------------------------------------------

class _Completed:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def _ok_result():
    return _Completed(0)


def _fail_result(stderr):
    return _Completed(1, stderr)


class TestGridSelfBuild:
    """The dashboard must never tell a person to go and run a command."""

    def test_existing_sweep_is_reported_available(self, tmp_path):
        path = tmp_path / "grid.json"
        path.write_text("{}", encoding="utf-8")
        assert JobRunner().ensure_grid(str(path))["available"] is True

    def test_missing_sweep_starts_a_build(self, tmp_path, monkeypatch):
        started = []

        def fake_run(command, **kwargs):
            started.append(command)
            time.sleep(0.3)
            return _ok_result()

        monkeypatch.setattr(subprocess, "run", fake_run)
        runner = JobRunner()
        first = runner.ensure_grid(str(tmp_path / "grid.json"))
        assert first["available"] is False
        assert first["status"] == "building"
        time.sleep(0.05)
        assert any("sim.precompute_grid" in c for c in started[0])

    def test_a_second_request_does_not_start_a_second_build(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(1)
            time.sleep(0.4)
            return _ok_result()

        monkeypatch.setattr(subprocess, "run", fake_run)
        runner = JobRunner()
        path = str(tmp_path / "grid.json")
        runner.ensure_grid(path)
        time.sleep(0.05)
        second = runner.ensure_grid(path)
        assert second["status"] == "building"
        time.sleep(0.6)
        assert len(calls) == 1

    def test_a_failed_build_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: _fail_result("sweep exploded"))
        runner = JobRunner()
        path = str(tmp_path / "grid.json")
        runner.ensure_grid(path)
        time.sleep(0.3)
        status = runner.ensure_grid(path)
        assert status["status"] == "error"
        assert "exploded" in status["error"]

    def test_build_uses_the_requested_session_count(self, tmp_path, monkeypatch):
        captured = []
        monkeypatch.setattr(subprocess, "run",
                            lambda c, **k: (captured.append(c), _ok_result())[1])
        JobRunner().ensure_grid(str(tmp_path / "g.json"), sessions=2500)
        time.sleep(0.2)
        assert "2500" in captured[0]
