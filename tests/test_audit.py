"""The audit trail, and the stream consumer that feeds it.

An audit trail that is merely written is not worth much. What makes it useful is
that a decision can be reconstructed from it afterwards -- so the tests check
that the detector's features survive the round trip, that the experiment's
headline numbers are recomputable from the log alone, and that writing twice
does not rewrite history.

The consumer tests exercise `handle_message` and `consume` directly against an
in-memory store. They do not test Kafka: no broker runs here, so framing,
delivery and offsets are unverified. That boundary is stated rather than
papered over.
"""
import json

import pytest

from core.session import Outcome, Reason, Session
from recovery.audit import AuditLog, to_row
from recovery.detector import Flag
from recovery.intervention import InterventionPolicy
from recovery.stream import consume, handle_message, publish


def closed_session(session_id="s0000001", arm="treatment", converted=False,
                   cart_value=400.0, treated=False):
    s = Session(agent_id=3, persona="average_buyer", started_at=0.0)
    s.session_id = session_id
    s.add_to_cart("P1", cart_value)
    s.arm = arm
    s.flag = Flag(
        session_id=session_id, sim_time=5.0, stage="cart", trigger="price_shock",
        features={"dwell": 6.5, "revisits": 1, "cart_value": cart_value,
                  "persona": "average_buyer"},
    )
    if treated:
        policy = InterventionPolicy(discount_rate=0.15, dispatch_latency=0.5)
        policy.dispatch(s, 5.0)
        policy.apply(s, 5.5)
    if converted:
        s.end(Outcome.CONVERTED, Reason.PURCHASED, 9.0,
              revenue=s.effective_cart_value())
    else:
        s.quit_driver = "price"
        s.recoverable = True
        s.end(Outcome.ABANDONED, Reason.QUIT, 9.0)
    return s


@pytest.fixture
def audit():
    log = AuditLog(":memory:")
    yield log
    log.close()


class TestRecording:
    def test_writes_a_flagged_session(self, audit):
        assert audit.record(closed_session(), "exp1", seed=42) is True
        assert audit.count("exp1") == 1

    def test_ignores_sessions_that_were_never_flagged(self, audit):
        s = Session(agent_id=1, persona="average_buyer", started_at=0.0)
        s.end(Outcome.ABANDONED, Reason.QUIT, 4.0)
        assert audit.record(s, "exp1") is False
        assert audit.count() == 0

    def test_is_idempotent_per_experiment(self, audit):
        """Writing the same decision twice must not duplicate it -- the same
        guarantee the intervention itself relies on."""
        session = closed_session()
        assert audit.record(session, "exp1") is True
        assert audit.record(session, "exp1") is False
        assert audit.count("exp1") == 1

    def test_the_same_session_can_appear_in_two_experiments(self, audit):
        session = closed_session()
        audit.record(session, "exp1")
        audit.record(session, "exp2")
        assert audit.count() == 2

    def test_record_many_counts_writes(self, audit):
        sessions = [closed_session(f"s{i:07d}") for i in range(10)]
        assert audit.record_many(sessions, "exp1", seed=1) == 10


class TestExplainability:
    def test_detector_features_survive_the_round_trip(self, audit):
        """Storing only "flagged: true" makes a decision unexplainable the
        moment anyone asks why."""
        audit.record(closed_session(), "exp1")
        stored = json.loads(audit.rows("exp1")[0]["features_at_flag"])
        assert stored["dwell"] == 6.5
        assert stored["revisits"] == 1

    def test_records_the_trigger_and_the_reason(self, audit):
        audit.record(closed_session(), "exp1")
        row = audit.rows("exp1")[0]
        assert row["flag_trigger"] == "price_shock"
        assert row["reason_code"] == "quit"
        assert row["quit_driver"] == "price"
        assert row["recoverable"] == 1

    def test_records_the_idempotency_key_of_the_offer(self, audit):
        audit.record(closed_session(treated=True), "exp1")
        row = audit.rows("exp1")[0]
        assert row["idempotency_key"] == "s0000001:discount"
        assert row["discount_rate"] == 0.15
        assert row["applied_at_sim"] == 5.5

    def test_discount_amount_is_recorded_for_the_margin_sum(self, audit):
        audit.record(closed_session(converted=True, treated=True), "exp1")
        row = audit.rows("exp1")[0]
        assert row["discount_amount"] == pytest.approx(400.0 * 0.15)
        assert row["realized_value"] == pytest.approx(340.0)

    def test_carries_no_free_text_or_product_identifiers(self, audit):
        """Every detector feature is behavioural. Nothing here identifies a
        shopper or what they looked at."""
        audit.record(closed_session(), "exp1")
        columns = set(audit.rows("exp1")[0])
        for forbidden in ("product_id", "product", "name", "email", "notes"):
            assert forbidden not in columns


class TestRecomputation:
    def test_headline_numbers_come_back_out_of_the_log(self, audit):
        sessions = (
            [closed_session(f"t{i}", arm="treatment", converted=i < 3) for i in range(10)]
            + [closed_session(f"c{i}", arm="control", converted=i < 1) for i in range(10)]
        )
        audit.record_many(sessions, "exp1")
        by_arm = audit.outcomes_by_arm("exp1")
        assert by_arm["treatment"] == {"n": 10, "converted": 3}
        assert by_arm["control"] == {"n": 10, "converted": 1}

    def test_driver_breakdown_groups_abandonment(self, audit):
        audit.record_many([closed_session(f"s{i}") for i in range(5)], "exp1")
        assert audit.driver_breakdown("exp1") == {"price": 5}

    def test_experiments_are_isolated(self, audit):
        audit.record(closed_session("a"), "exp1")
        audit.record(closed_session("b"), "exp2")
        assert audit.count("exp1") == 1
        assert audit.count("exp2") == 1


class TestRowShape:
    def test_to_row_is_json_serialisable(self):
        """The same record goes to the stream without passing through SQLite."""
        json.dumps(to_row(closed_session(treated=True), "exp1", seed=42))

    def test_unassigned_arm_is_labelled_not_null(self):
        session = closed_session()
        session.arm = None
        assert to_row(session, "exp1")["arm"] == "unassigned"


class TestStreamConsumer:
    """The first consumer this project has had. Kafka itself is not tested."""

    def test_consumes_a_dict_message(self, audit):
        assert handle_message(to_row(closed_session(), "exp1"), audit) is True
        assert audit.count("exp1") == 1

    def test_consumes_raw_json_bytes(self, audit):
        payload = json.dumps(to_row(closed_session(), "exp1")).encode("utf-8")
        assert handle_message(payload, audit) is True
        assert audit.count("exp1") == 1

    def test_survives_a_malformed_message(self, audit):
        """A consumer that dies on one bad record stops recording everything
        after it."""
        assert handle_message(b"{not json", audit) is False
        assert handle_message({"nonsense": True}, audit) is False
        assert handle_message(None, audit) is False
        assert audit.count() == 0

    def test_rejects_a_duplicate_without_dying(self, audit):
        row = to_row(closed_session(), "exp1")
        assert handle_message(row, audit) is True
        assert handle_message(row, audit) is False
        assert audit.count("exp1") == 1

    def test_consume_drains_an_iterable(self, audit):
        rows = [to_row(closed_session(f"s{i:07d}"), "exp1") for i in range(8)]
        assert consume(rows, audit) == 8
        assert audit.count("exp1") == 8

    def test_consume_unwraps_messages_with_a_value_attribute(self, audit):
        class _Message:
            def __init__(self, value):
                self.value = value

        rows = [_Message(to_row(closed_session(f"s{i:07d}"), "exp1")) for i in range(4)]
        assert consume(rows, audit) == 4

    def test_consume_honours_a_limit(self, audit):
        rows = [to_row(closed_session(f"s{i:07d}"), "exp1") for i in range(10)]
        assert consume(rows, audit, limit=3) == 3

    def test_a_streamed_experiment_reconciles(self, audit):
        """End to end: publish-shaped records in, headline numbers out."""
        sessions = (
            [closed_session(f"t{i}", arm="treatment", converted=i < 2) for i in range(6)]
            + [closed_session(f"c{i}", arm="control", converted=False) for i in range(6)]
        )
        consume([to_row(s, "exp1") for s in sessions], audit)
        assert audit.outcomes_by_arm("exp1")["treatment"] == {"n": 6, "converted": 2}


class TestPublish:
    def test_sends_every_record(self):
        class _Producer:
            def __init__(self):
                self.sent = []

            def send(self, topic, value):
                self.sent.append((topic, value))

            def flush(self):
                pass

        producer = _Producer()
        rows = [to_row(closed_session(f"s{i:07d}"), "exp1") for i in range(5)]
        assert publish(rows, producer, topic="recovery_events") == 5
        assert producer.sent[0][0] == "recovery_events"

    def test_no_producer_is_not_an_error(self):
        """SQLite is the record; the stream is an optional mirror."""
        assert publish([{"session_id": "s1"}], None) == 0

    def test_a_dead_broker_stops_mirroring_quietly(self):
        class _Broken:
            def send(self, topic, value):
                raise RuntimeError("broker gone")

            def flush(self):
                raise RuntimeError("broker gone")

        rows = [to_row(closed_session(), "exp1")]
        assert publish(rows, _Broken()) == 0
