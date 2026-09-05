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
