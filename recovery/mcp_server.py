"""An MCP server over the recovery pipeline.

Four tools, answering the questions somebody actually asks about a discount
programme: why did this shopper get an offer, what did the experiment show, what
is losing us carts, and what would happen if we changed the offer.

The last one is the reason this exists rather than being a database wrapper. The
measurement harness underneath it works -- the A/A check comes back null and the
A/B does not -- so an agent can be asked "what about 20% off, or half the
dispatch latency" and get a measured delta with a confidence interval instead of
a guess. Wrapping SQLite reads in RPC would have added nothing.

**Nothing here touches production.** ``run_counterfactual`` starts a simulation
and returns statistics; there is no path from any tool to a live shopper, a real
offer, or money. The read tools are read-only by construction -- the audit trail
has no UPDATE or DELETE.

Tool logic is kept separate from the MCP wiring so it can be tested without a
client, the same split used for the Kafka consumer. What the tests do *not*
cover is the transport: no MCP client runs in CI, so framing and the stdio loop
are unverified here.

Run:  python -m recovery.mcp_server
Register with Claude Code:
  claude mcp add recovery -- python -m recovery.mcp_server
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recovery.analysis import ProportionTest
from recovery.audit import AuditLog

DEFAULT_DB = os.environ.get("FF_AUDIT_DB", "recovery_audit.db")

# A counterfactual runs a real simulation, so it is bounded. Below the floor the
# arms are too small to say anything; above the ceiling a tool call would sit
# there for many minutes, and the honest answer is to run the CLI instead.
MIN_SESSIONS = 2000
MAX_SESSIONS = 20000
DEFAULT_SESSIONS = 6000


def _open(db_path):
    return AuditLog(db_path or DEFAULT_DB)


def _resolve(audit, experiment_id):
    return experiment_id or audit.latest_experiment()


# -- tool logic, testable without an MCP client -------------------------------

def explain_decision(session_id: str, experiment_id=None, db_path=None) -> dict:
    """Why one session was flagged, what was offered, and what happened."""
    audit = _open(db_path)
    try:
        row = audit.find_session(session_id, experiment_id)
        if row is None:
            return {"found": False, "session_id": session_id,
                    "message": "no recovery decision recorded for this session"}

        features = json.loads(row["features_at_flag"])
        offered = row["intervention_kind"] is not None
        landed = row["applied_at_sim"] is not None

        if not offered:
            offer = "none — this session was assigned to the control arm"
        elif landed:
            offer = (f"{row['discount_rate']:.0%} discount, applied at "
                     f"t={row['applied_at_sim']:.1f}s")
        else:
            offer = ("dispatched but never landed — the session ended before the "
                     "offer arrived")

        return {
            "found": True,
            "session_id": row["session_id"],
            "experiment_id": row["experiment_id"],
            "persona": row["persona"],
            "arm": row["arm"],
            "flagged_at_sim": row["flagged_at_sim"],
            "why_flagged": {
                "trigger": row["flag_trigger"],
                "stage": row["flag_stage"],
                "evidence": features,
            },
            "offer": offer,
            "idempotency_key": row["idempotency_key"],
            "outcome": row["outcome"],
            "reason_code": row["reason_code"],
            "quit_driver": row["quit_driver"],
            "was_recoverable": bool(row["recoverable"]),
            "cart_value": row["cart_value"],
            "discount_amount": row["discount_amount"],
            "realized_value": row["realized_value"],
            "detector_version": row["detector_version"],
        }
    finally:
        audit.close()


def experiment_summary(experiment_id=None, db_path=None) -> dict:
    """Arms, effect, confidence interval and margin, recomputed from the log."""
    audit = _open(db_path)
    try:
        experiment_id = _resolve(audit, experiment_id)
        if experiment_id is None:
            return {"found": False, "message": "no experiments in the audit trail"}

        by_arm = audit.outcomes_by_arm(experiment_id)
        treatment = by_arm.get("treatment", {"n": 0, "converted": 0})
        control = by_arm.get("control", {"n": 0, "converted": 0})
        test = ProportionTest(
            treat_hits=treatment["converted"], treat_n=treatment["n"],
            control_hits=control["converted"], control_n=control["n"],
        )

        rows = audit.rows(experiment_id)
        treated_converted = [r for r in rows
                             if r["arm"] == "treatment" and r["outcome"] == "converted"]
        control_converted = [r for r in rows
                             if r["arm"] == "control" and r["outcome"] == "converted"]
        average_cart = (sum(r["cart_value"] for r in control_converted)
                        / len(control_converted)) if control_converted else 0.0
        incremental = test.delta * treatment["n"]
        gross = incremental * average_cart
        cost = sum(r["discount_amount"] for r in treated_converted)

        return {
            "found": True,
            "experiment_id": experiment_id,
            "arms": by_arm,
            "result": test.summary(),
            "reading": test.describe(),
            "margin": {
                "incremental_conversions": round(incremental, 2),
                "gross_recovered": round(gross, 2),
                "discount_cost": round(cost, 2),
                "net": round(gross - cost, 2),
                "margin_positive": (gross - cost) > 0,
            },
            "caveat": ("Synthetic order values. The method transfers; the "
                       "magnitudes are not a prediction about live traffic."),
        }
    finally:
        audit.close()


def abandonment_drivers(experiment_id=None, db_path=None) -> dict:
    """What is actually losing carts, and how much of it a discount could reach."""
    audit = _open(db_path)
    try:
        experiment_id = _resolve(audit, experiment_id)
        if experiment_id is None:
            return {"found": False, "message": "no experiments in the audit trail"}

        drivers = audit.driver_breakdown(experiment_id)
        total = sum(drivers.values())
        rows = audit.rows(experiment_id)
        abandoned = [r for r in rows if r["outcome"] == "abandoned"]
        recoverable = sum(1 for r in abandoned if r["recoverable"])

        return {
            "found": True,
            "experiment_id": experiment_id,
            "abandoned_sessions": total,
            "by_driver": {k: {"n": v, "share": round(v / total, 4) if total else 0.0}
                          for k, v in drivers.items()},
            "recoverable_share": round(recoverable / total, 4) if total else 0.0,
            "note": ("recoverable means price-driven: a discount can move those "
                     "and cannot move the rest, whatever its size. This is "
                     "counterfactual ground truth the simulation has and a live "
                     "system does not."),
        }
    finally:
        audit.close()


def run_counterfactual(discount: float = 0.15, latency: float = 0.5,
                       sessions: int = DEFAULT_SESSIONS, triggers=None,
                       db_path=None) -> dict:
    """Simulate a different offer and measure it. Does not touch production.

    Runs the real experiment: thresholds calibrated on a held-out seed, arms
    randomised at flag time, result reported with a confidence interval and the
    effect the sample could have detected.
    """
    if not 0.0 <= discount <= 0.95:
        return {"error": "discount must be between 0 and 0.95"}
    if latency < 0:
        return {"error": "latency must not be negative"}

    sessions = max(MIN_SESSIONS, min(int(sessions), MAX_SESSIONS))
    from sim.experiment import run as run_experiment

    trigger_set = frozenset(triggers) if triggers else None
    result = run_experiment(
        n_sessions=sessions, discount=discount, latency=latency,
        triggers=trigger_set, budget_aware=True,
        audit_path=db_path or DEFAULT_DB,
    )
    test = result["test"]
    return {
        "simulated": True,
        "sessions": sessions,
        "discount": discount,
        "dispatch_latency": latency,
        "triggers": sorted(trigger_set) if trigger_set else "all",
        "result": test,
        "margin": result["margin"],
        "significant": test["significant"],
        "caveat": ("A simulation, not a live experiment. No shopper was offered "
                   "anything and no money moved."),
    }


# -- MCP wiring ---------------------------------------------------------------

def build_server(db_path=None):
    """Construct the FastMCP server. Imported lazily so the tool logic above
    stays usable, and testable, without the MCP SDK installed."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("razorpay-recovery")

    @server.tool(description="Explain why one shopping session was flagged as at "
                             "risk, what offer it received, and how it ended.")
    def explain_decision_tool(session_id: str, experiment_id: str | None = None) -> dict:
        return explain_decision(session_id, experiment_id, db_path)

    @server.tool(description="Summarise a recovery experiment: arms, effect size, "
                             "95% confidence interval and net margin.")
    def experiment_summary_tool(experiment_id: str | None = None) -> dict:
        return experiment_summary(experiment_id, db_path)

    @server.tool(description="Break cart abandonment down by cause and report how "
                             "much of it a discount could reach.")
    def abandonment_drivers_tool(experiment_id: str | None = None) -> dict:
        return abandonment_drivers(experiment_id, db_path)

    @server.tool(description="Simulate a different discount rate, dispatch latency "
                             "or targeting rule and measure the result. Runs a "
                             "simulation only; it cannot affect live traffic. "
                             "Takes a few minutes at larger session counts.")
    def run_counterfactual_tool(discount: float = 0.15, latency: float = 0.5,
                                sessions: int = DEFAULT_SESSIONS,
                                triggers: list[str] | None = None) -> dict:
        return run_counterfactual(discount, latency, sessions, triggers, db_path)

    return server


def main():        # pragma: no cover - needs an MCP client on stdio
    build_server().run(transport="stdio")


if __name__ == "__main__":     # pragma: no cover
    main()
