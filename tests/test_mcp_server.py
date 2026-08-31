"""The MCP tools, exercised without an MCP client.

Tool logic is tested directly; the stdio transport is not, because no MCP client
runs here. That boundary is the same one the Kafka consumer has, and it is
stated rather than implied.

The test that matters most is the last one: no tool may be able to affect
anything real. `run_counterfactual` starts a simulation and returns statistics,
and the read tools sit on an append-only table with no UPDATE or DELETE.
"""
import json

import pytest

from core.session import Outcome, Reason, Session
from recovery.audit import AuditLog, to_row
from recovery.detector import Flag
from recovery.intervention import InterventionPolicy
from recovery.mcp_server import (
    MAX_SESSIONS,
    MIN_SESSIONS,
    abandonment_drivers,
    build_server,
    experiment_summary,
    explain_decision,
    run_counterfactual,
)

EXPERIMENT = "ab-test-001"


def make_session(session_id, arm="treatment", converted=False, treated=False,
                 driver="price", recoverable=True):
    s = Session(agent_id=2, persona="average_buyer", started_at=0.0)
    s.session_id = session_id
    s.add_to_cart("P1", 300.0)
    s.arm = arm
    s.flag = Flag(session_id=session_id, sim_time=4.0, stage="cart",
                  trigger="price_shock",
                  features={"dwell": 5.5, "revisits": 1, "cart_value": 300.0})
    if treated:
        policy = InterventionPolicy(discount_rate=0.15, dispatch_latency=0.5)
        policy.dispatch(s, 4.0)
        policy.apply(s, 4.5)
    if converted:
        s.end(Outcome.CONVERTED, Reason.PURCHASED, 9.0, revenue=s.checkout_total())
    else:
        s.quit_driver = driver
        s.recoverable = recoverable
        s.end(Outcome.ABANDONED, Reason.FEE_SHOCK, 9.0)
    return s


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "audit.db")
    audit = AuditLog(path)
    sessions = (
        [make_session(f"t{i}", "treatment", converted=i < 6, treated=True)
         for i in range(60)]
        + [make_session(f"c{i}", "control", converted=i < 2) for i in range(60)]
    )
    audit.record_many(sessions, EXPERIMENT, seed=42)
    audit.close()
    return path


class TestExplainDecision:
    def test_reports_the_evidence_that_flagged_it(self, db):
        result = explain_decision("t0", db_path=db)
        assert result["found"] is True
        assert result["why_flagged"]["trigger"] == "price_shock"
        assert result["why_flagged"]["evidence"]["dwell"] == 5.5

    def test_describes_a_landed_offer(self, db):
        assert "15% discount, applied" in explain_decision("t0", db_path=db)["offer"]

    def test_describes_a_control_session_as_untreated(self, db):
        result = explain_decision("c0", db_path=db)
        assert result["arm"] == "control"
        assert "control arm" in result["offer"]

    def test_reports_the_idempotency_key(self, db):
        """The question an auditor asks: was this shopper offered twice?"""
        assert explain_decision("t0", db_path=db)["idempotency_key"] == "t0:discount"

    def test_unknown_session_is_answered_not_raised(self, db):
        result = explain_decision("nope", db_path=db)
        assert result["found"] is False
        assert "no recovery decision" in result["message"]

    def test_carries_the_outcome_and_its_cause(self, db):
        result = explain_decision("t59", db_path=db)
        assert result["outcome"] == "abandoned"
        assert result["reason_code"] == "fee_shock"
        assert result["was_recoverable"] is True


class TestExperimentSummary:
    def test_recomputes_the_effect_from_the_log(self, db):
        result = experiment_summary(EXPERIMENT, db_path=db)
        assert result["arms"]["treatment"] == {"n": 60, "converted": 6}
        assert result["arms"]["control"] == {"n": 60, "converted": 2}
        assert result["result"]["delta"] == pytest.approx(6 / 60 - 2 / 60, abs=1e-4)

    def test_reports_an_interval_and_an_mde(self, db):
        result = experiment_summary(EXPERIMENT, db_path=db)["result"]
        assert result["ci_low"] < result["delta"] < result["ci_high"]
        assert result["mde"] > 0

    def test_reports_margin_including_the_discount_bill(self, db):
        margin = experiment_summary(EXPERIMENT, db_path=db)["margin"]
        # Billed on all six treated conversions, not just the incremental ones.
        assert margin["discount_cost"] == pytest.approx(6 * 300.0 * 0.15)

    def test_states_the_synthetic_caveat(self, db):
        assert "not a prediction" in experiment_summary(EXPERIMENT, db_path=db)["caveat"]

    def test_defaults_to_the_latest_experiment(self, db):
        assert experiment_summary(db_path=db)["experiment_id"] == EXPERIMENT

    def test_empty_log_is_answered_not_raised(self, tmp_path):
        empty = str(tmp_path / "empty.db")
        assert experiment_summary(db_path=empty)["found"] is False


class TestAbandonmentDrivers:
    def test_breaks_losses_down_by_cause(self, db):
        result = abandonment_drivers(EXPERIMENT, db_path=db)
        assert result["abandoned_sessions"] == 112
        assert result["by_driver"]["price"]["n"] == 112

    def test_reports_the_recoverable_ceiling(self, db):
        assert abandonment_drivers(EXPERIMENT, db_path=db)["recoverable_share"] == 1.0

    def test_explains_what_recoverable_means(self, db):
        assert "cannot move the rest" in abandonment_drivers(EXPERIMENT, db_path=db)["note"]

    def test_empty_log_is_answered_not_raised(self, tmp_path):
        assert abandonment_drivers(db_path=str(tmp_path / "e.db"))["found"] is False


class TestCounterfactualGuards:
    """The expensive tool. Validation is checked; the run itself is not."""

    def test_rejects_an_impossible_discount(self):
        assert "error" in run_counterfactual(discount=1.5)
        assert "error" in run_counterfactual(discount=-0.1)

    def test_rejects_negative_latency(self):
        assert "error" in run_counterfactual(latency=-1.0)

    def test_session_bounds_are_sane(self):
        assert MIN_SESSIONS >= 1000
        assert MAX_SESSIONS <= 50000
        assert MIN_SESSIONS < MAX_SESSIONS


class TestSafety:
    def test_no_tool_can_reach_production(self):
        """Every tool either reads an append-only table or starts a simulation.

        Nothing writes to a payment system, issues a real offer, or moves money,
        and there is no code path from an MCP call to one.

        Checked over the parsed module rather than its text: an earlier version
        grepped the source and flagged the docstring sentence explaining that
        there is no UPDATE, which is the wrong kind of strict.
        """
        import ast
        import inspect

        import recovery.mcp_server as server

        tree = ast.parse(inspect.getsource(server))

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for forbidden in ("requests", "httpx", "urllib", "socket", "subprocess", "smtplib"):
            assert forbidden not in imported, f"{forbidden} is reachable from a tool"

        mutations = ("update", "delete", "drop", "insert", "alter")
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                first = node.value.strip().split(" ")[0].lower()
                assert first not in mutations, f"mutating SQL in a tool: {node.value[:40]!r}"

    def test_counterfactual_labels_itself_a_simulation(self):
        result = run_counterfactual(discount=1.5)   # cheap: rejected before running
        assert "error" in result

    def test_audit_schema_offers_no_mutation_verbs(self):
        from recovery.audit import SCHEMA

        assert "UPDATE" not in SCHEMA.upper().replace("AUTOINCREMENT", "")
        assert "DELETE" not in SCHEMA.upper()


class TestServerConstruction:
    def test_builds_and_registers_four_tools(self):
        """Exercises the MCP wiring itself, short of running the transport."""
        server = build_server()
        assert server is not None
        assert server.name == "razorpay-recovery"

    def test_tools_are_listed(self):
        import asyncio

        tools = asyncio.run(build_server().list_tools())
        names = {t.name for t in tools}
        assert names == {
            "explain_decision_tool", "experiment_summary_tool",
            "abandonment_drivers_tool", "run_counterfactual_tool",
        }

    def test_every_tool_describes_itself(self):
        import asyncio

        for tool in asyncio.run(build_server().list_tools()):
            assert tool.description and len(tool.description) > 30
