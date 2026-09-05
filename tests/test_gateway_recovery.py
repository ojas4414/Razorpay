"""The gateway's recovery endpoints.

These serve the dashboard from the same functions the MCP tools call, so the
page and an agent asking in words cannot report different numbers. The tests
check the HTTP shape the dashboard actually consumes -- a rename here would
break the page silently, the way the budget_sensitivity shape change did.
"""
import pytest
from fastapi.testclient import TestClient

from core.session import Outcome, Reason, Session
from recovery.audit import AuditLog
from recovery.detector import Flag
from recovery.intervention import InterventionPolicy

EXPERIMENT = "ab-gateway-test"


def flagged(session_id, arm="treatment", converted=False, treated=False):
    s = Session(agent_id=1, persona="average_buyer", started_at=0.0)
    s.session_id = session_id
    s.add_to_cart("P1", 250.0)
    s.arm = arm
    s.flag = Flag(session_id=session_id, sim_time=3.0, stage="cart",
                  trigger="price_shock",
                  features={"dwell": 4.0, "revisits": 1, "cart_value": 250.0})
    if treated:
        policy = InterventionPolicy(discount_rate=0.15, dispatch_latency=0.5)
        policy.dispatch(s, 3.0)
        policy.apply(s, 3.5)
    if converted:
        s.end(Outcome.CONVERTED, Reason.PURCHASED, 8.0, revenue=s.checkout_total())
    else:
        s.quit_driver = "price"
        s.recoverable = True
        s.end(Outcome.ABANDONED, Reason.FEE_SHOCK, 8.0)
    return s


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "audit.db")
    audit = AuditLog(db)
    audit.record_many(
        [flagged(f"t{i}", "treatment", converted=i < 5, treated=True) for i in range(40)]
        + [flagged(f"c{i}", "control", converted=i < 2) for i in range(40)],
        EXPERIMENT, seed=42,
    )
    audit.close()

    monkeypatch.setenv("FF_AUDIT_DB", db)
    monkeypatch.setenv("FF_LIVE_MODE", "off")     # no simulation thread in tests
    import backend.gateway.main as gateway

    monkeypatch.setattr(gateway, "AUDIT_DB", db)
    with TestClient(gateway.app) as c:
        yield c


class TestExperimentsEndpoint:
    def test_lists_experiments(self, client):
        body = client.get("/recovery/experiments").json()
        assert body["latest"] == EXPERIMENT
        assert body["experiments"][0]["rows"] == 80


class TestExperimentEndpoint:
    def test_returns_the_shape_the_dashboard_reads(self, client):
        body = client.get("/recovery/experiment").json()
        assert body["found"] is True
        for key in ("treat_rate", "control_rate", "delta", "ci_low", "ci_high",
                    "z", "significant", "mde", "treat_n", "control_n"):
            assert key in body["result"], key
        for key in ("gross_recovered", "discount_cost", "net", "margin_positive"):
            assert key in body["margin"], key

    def test_recomputes_the_arms_from_the_log(self, client):
        body = client.get(f"/recovery/experiment?experiment_id={EXPERIMENT}").json()
        assert body["arms"]["treatment"] == {"n": 40, "converted": 5}
        assert body["arms"]["control"] == {"n": 40, "converted": 2}

    def test_carries_the_synthetic_caveat(self, client):
        """The page prints this; it must not quietly disappear."""
        assert "not a prediction" in client.get("/recovery/experiment").json()["caveat"]


class TestDriversEndpoint:
    def test_returns_shares_and_the_ceiling(self, client):
        body = client.get("/recovery/drivers").json()
        assert body["found"] is True
        assert body["by_driver"]["price"]["share"] == 1.0
        assert body["recoverable_share"] == 1.0


class TestDecisionsEndpoint:
    def test_returns_recent_decisions_newest_first(self, client):
        body = client.get("/recovery/decisions?limit=10").json()
        assert body["experiment_id"] == EXPERIMENT
        assert len(body["decisions"]) == 10
        first = body["decisions"][0]
        for key in ("session_id", "arm", "trigger", "cart_value", "outcome",
                    "reason_code", "recoverable", "landed"):
            assert key in first, key

    def test_limit_is_bounded(self, client):
        """A page cannot ask for the whole table by accident."""
        assert len(client.get("/recovery/decisions?limit=99999").json()["decisions"]) <= 200

    def test_empty_log_returns_an_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FF_LIVE_MODE", "off")
        import backend.gateway.main as gateway

        monkeypatch.setattr(gateway, "AUDIT_DB", str(tmp_path / "empty.db"))
        with TestClient(gateway.app) as c:
            body = c.get("/recovery/decisions").json()
        assert body["decisions"] == []


class TestDecisionEndpoint:
    def test_explains_one_session(self, client):
        body = client.get("/recovery/decision/t0").json()
        assert body["found"] is True
        assert body["why_flagged"]["trigger"] == "price_shock"
        assert body["why_flagged"]["evidence"]["dwell"] == 4.0
        assert "15% discount" in body["offer"]

    def test_unknown_session_answers_rather_than_500s(self, client):
        response = client.get("/recovery/decision/nope")
        assert response.status_code == 200
        assert response.json()["found"] is False


class TestStaticPages:
    def test_recovery_dashboard_is_served(self, client):
        response = client.get("/dashboard/recovery.html")
        assert response.status_code == 200
        assert "Revenue Recovery" in response.text

    def test_root_redirects_to_it(self, client):
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 307)
        assert "recovery.html" in response.headers["location"]


class TestGridEndpoint:
    def test_missing_grid_starts_a_build_rather_than_instructing_the_user(
            self, client, monkeypatch, tmp_path):
        """The page must never tell a person to go and run a command.

        subprocess is faked so the test does not launch a real twelve-cell
        sweep; what is checked is that the endpoint reports work in progress.
        """
        import subprocess
        import time

        import backend.gateway.main as gateway

        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: type("R", (), {"returncode": 0,
                                                           "stderr": "",
                                                           "stdout": ""})())
        monkeypatch.setattr(gateway, "GRID_PATH", str(tmp_path / "absent.json"))
        body = client.get("/recovery/grid").json()
        assert body["available"] is False
        assert body["status"] == "building"
        assert "message" not in body           # no instruction to the reader
        time.sleep(0.1)

    def test_serves_the_sweep_when_present(self, client, monkeypatch, tmp_path):
        import json

        import backend.gateway.main as gateway

        grid = tmp_path / "grid.json"
        grid.write_text(json.dumps({
            "sessions_per_cell": 5000,
            "break_even_discount": 0.22,
            "by_discount": [{"discount": 0.15, "delta": 0.03,
                             "margin": {"net": 100.0, "margin_positive": True}}],
            "by_latency": [{"latency": 0.5, "issued": 10, "missed": 1}],
        }), encoding="utf-8")
        monkeypatch.setattr(gateway, "GRID_PATH", str(grid))
        body = client.get("/recovery/grid").json()
        assert body["break_even_discount"] == 0.22
        assert body["by_discount"][0]["discount"] == 0.15


class TestCounterfactualEndpoint:
    def test_rejects_an_impossible_discount(self, client):
        response = client.post("/recovery/counterfactual", json={"discount": 2.0})
        assert response.json()["accepted"] is False

    def test_polling_an_unknown_job_is_answered(self, client):
        assert client.get("/recovery/counterfactual/nope").json()["status"] == "unknown"

    def test_accepts_and_returns_a_job_id(self, client, monkeypatch):
        """The run itself is faked; the endpoint contract is what is checked."""
        import subprocess

        class _Done:
            returncode, stderr, stdout = 0, "", ""

        def fake_run(command, **kwargs):
            import json as _json

            with open(command[command.index("--out") + 1], "w", encoding="utf-8") as h:
                _json.dump({"test": {"delta": 0.02}, "margin": {"net": 1.0}}, h)
            return _Done()

        monkeypatch.setattr(subprocess, "run", fake_run)
        body = client.post("/recovery/counterfactual",
                           json={"discount": 0.2, "latency": 1.0, "sessions": 1000}).json()
        assert body["accepted"] is True
        assert body["params"]["discount"] == 0.2

        import time

        for _ in range(200):
            polled = client.get(f"/recovery/counterfactual/{body['job_id']}").json()
            if polled["status"] != "running":
                break
            time.sleep(0.02)
        assert polled["status"] == "done"
        assert polled["result"]["test"]["delta"] == 0.02
