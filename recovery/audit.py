"""Append-only audit trail for every recovery decision.

One row per flagged session, written once and never updated. A payments system
has to be able to answer "why did this shopper get an offer, and what happened"
long after the fact, and answer it from a record rather than by re-running
anything.

Three properties are deliberate.

**Append-only.** There is no UPDATE and no DELETE. A row is written when a
session closes and its story is complete; correcting a decision means writing a
new row, not editing history. The schema enforces this only by not offering the
verbs -- a real deployment would revoke them at the database.

**The features are kept, not just the verdict.** ``features_at_flag`` holds what
the detector actually saw. Storing only "flagged: true" makes a decision
unexplainable the moment anyone asks why, which is the question that matters
most when a discount goes to the wrong person.

**Reason codes, not prose.** Outcome and driver are symbols the analysis can
group by, so "why did this session fail to convert" is answerable in SQL.

Note what is *not* here: no product names, no free text, nothing identifying a
shopper beyond a synthetic agent id. Every detector feature is behavioural --
dwell, revisits, basket size -- which is a real design property rather than a
claim, and the reason this shape of record could survive a data-governance
review.
"""
import json
import sqlite3
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS recovery_audit (
    row_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id       TEXT    NOT NULL,
    seed                INTEGER,
    session_id          TEXT    NOT NULL,
    agent_id            INTEGER NOT NULL,
    persona             TEXT    NOT NULL,

    detector_version    TEXT    NOT NULL,
    flagged_at_sim      REAL    NOT NULL,
    flag_trigger        TEXT    NOT NULL,
    flag_stage          TEXT    NOT NULL,
    features_at_flag    TEXT    NOT NULL,

    arm                 TEXT    NOT NULL,
    intervention_kind   TEXT,
    idempotency_key     TEXT,
    discount_rate       REAL,
    dispatched_at_sim   REAL,
    applied_at_sim      REAL,
    intervention_missed INTEGER NOT NULL DEFAULT 0,

    outcome             TEXT,
    reason_code         TEXT,
    quit_driver         TEXT,
    recoverable         INTEGER NOT NULL DEFAULT 0,
    ended_at_sim        REAL,

    cart_value          REAL    NOT NULL,
    discount_amount     REAL    NOT NULL DEFAULT 0,
    realized_value      REAL    NOT NULL DEFAULT 0,

    UNIQUE (experiment_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_audit_experiment ON recovery_audit (experiment_id);
CREATE INDEX IF NOT EXISTS idx_audit_arm        ON recovery_audit (experiment_id, arm);
"""

DETECTOR_VERSION = "stall+hesitation+price_shock/v2"


@dataclass
class AuditLog:
    """A SQLite-backed append-only log. ``:memory:`` is fine for tests."""

    path: str = "recovery_audit.db"

    def __post_init__(self):
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- writing --------------------------------------------------------------
    def record(self, session, experiment_id: str, seed=None,
               detector_version: str = DETECTOR_VERSION) -> bool:
        """Write one closed, flagged session. Returns False if already present.

        Idempotent on (experiment_id, session_id): re-running a write does not
        duplicate a decision, which matters because the same guarantee is what
        the intervention itself relies on.
        """
        if session.flag is None:
            return False

        row = to_row(session, experiment_id, seed, detector_version)
        try:
            self._conn.execute(
                f"INSERT INTO recovery_audit ({','.join(row)}) "
                f"VALUES ({','.join(':' + k for k in row)})",
                row,
            )
        except sqlite3.IntegrityError:
            return False        # already recorded; history is not rewritten
        self._conn.commit()
        return True

    def record_many(self, sessions, experiment_id: str, seed=None) -> int:
        return sum(self.record(s, experiment_id, seed) for s in sessions)

    # -- reading --------------------------------------------------------------
    def count(self, experiment_id=None) -> int:
        if experiment_id is None:
            cursor = self._conn.execute("SELECT COUNT(*) FROM recovery_audit")
        else:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM recovery_audit WHERE experiment_id = ?",
                (experiment_id,),
            )
        return cursor.fetchone()[0]

    def rows(self, experiment_id=None):
        if experiment_id is None:
            cursor = self._conn.execute("SELECT * FROM recovery_audit ORDER BY row_id")
        else:
            cursor = self._conn.execute(
                "SELECT * FROM recovery_audit WHERE experiment_id = ? ORDER BY row_id",
                (experiment_id,),
            )
        return [dict(r) for r in cursor.fetchall()]

    def outcomes_by_arm(self, experiment_id: str) -> dict:
        """Conversions and totals per arm, straight from the log.

        The experiment's headline numbers are recomputable from the audit trail
        alone. If these disagree with the in-memory analysis, one of them is
        wrong and the disagreement is visible rather than silent.
        """
        cursor = self._conn.execute(
            "SELECT arm, COUNT(*) AS n, "
            "SUM(CASE WHEN outcome = 'converted' THEN 1 ELSE 0 END) AS converted "
            "FROM recovery_audit WHERE experiment_id = ? GROUP BY arm",
            (experiment_id,),
        )
        return {r["arm"]: {"n": r["n"], "converted": r["converted"]}
                for r in cursor.fetchall()}

    def driver_breakdown(self, experiment_id: str) -> dict:
        cursor = self._conn.execute(
            "SELECT quit_driver, COUNT(*) AS n FROM recovery_audit "
            "WHERE experiment_id = ? AND outcome = 'abandoned' "
            "GROUP BY quit_driver ORDER BY n DESC",
            (experiment_id,),
        )
        return {r["quit_driver"]: r["n"] for r in cursor.fetchall()}

    def close(self):
        self._conn.close()


def to_row(session, experiment_id: str, seed=None,
           detector_version: str = DETECTOR_VERSION) -> dict:
    """Flatten a closed session into an audit row.

    Kept separate from the database so the same record can be published to a
    stream without going through SQLite first.
    """
    flag = session.flag
    intervention = session.intervention
    discount_amount = round(session.cart_value * session.discount, 2)

    return {
        "experiment_id": experiment_id,
        "seed": seed,
        "session_id": session.session_id,
        "agent_id": session.agent_id,
        "persona": session.persona,
        "detector_version": detector_version,
        "flagged_at_sim": flag.sim_time,
        "flag_trigger": flag.trigger,
        "flag_stage": flag.stage,
        "features_at_flag": json.dumps(flag.features, sort_keys=True),
        "arm": session.arm or "unassigned",
        "intervention_kind": intervention.kind if intervention else None,
        "idempotency_key": intervention.idempotency_key if intervention else None,
        "discount_rate": intervention.magnitude if intervention else None,
        "dispatched_at_sim": intervention.dispatched_at if intervention else None,
        "applied_at_sim": intervention.applied_at if intervention else None,
        "intervention_missed": int(bool(intervention and intervention.missed)),
        "outcome": session.outcome,
        "reason_code": session.reason,
        "quit_driver": session.quit_driver,
        "recoverable": int(bool(session.recoverable)),
        "ended_at_sim": session.ended_at,
        "cart_value": session.cart_value,
        "discount_amount": discount_amount,
        "realized_value": session.revenue,
    }
