"""Publishing recovery decisions to Kafka, and consuming them back.

The repo has published to Kafka since the beginning and never read from it --
`agent_actions` had no consumer at all, so the stream was write-only and the
broker was decoration. The audit trail is the thing that gives it a job, and
this is the first consumer the project has had.

The split is deliberate. SQLite is the source of truth and is always written,
so `make reproduce` works with plain Python and no broker. Kafka is an optional
mirror for the docker-compose topology, where a separate process materialises
the same records -- which is the shape a real payments pipeline has: decisions
emitted as events, and a downstream service that durably records them.

`handle_message` is the whole of the consumer's logic and is exercised directly
by the tests. What the tests do *not* cover is the broker itself: no Kafka runs
in CI, so message framing, delivery and offset handling are unverified here.
Saying so is more useful than implying otherwise.
"""
import json
import os

from recovery.audit import AuditLog

DEFAULT_TOPIC = os.environ.get("FF_RECOVERY_TOPIC", "recovery_events")


def publish(records, producer, topic: str = DEFAULT_TOPIC) -> int:
    """Mirror audit rows onto the stream. Returns how many were sent.

    A dead broker must never take the simulation down with it, so a failure
    stops the mirroring and leaves SQLite -- the actual record -- untouched.
    """
    if producer is None:
        return 0

    sent = 0
    for record in records:
        try:
            producer.send(topic, value=record)
            sent += 1
        except Exception:
            break       # broker gone; SQLite still holds the truth
    try:
        producer.flush()
    except Exception:
        pass
    return sent


def handle_message(payload, audit: AuditLog, experiment_id=None) -> bool:
    """Write one streamed record into the audit store.

    Accepts the decoded dict or raw bytes. Returns False for a duplicate or a
    message that cannot be parsed -- a consumer that dies on one bad record is
    a consumer that stops recording everything after it.
    """
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return False
    if not isinstance(payload, dict) or "session_id" not in payload:
        return False

    row = dict(payload)
    if experiment_id is not None:
        row["experiment_id"] = experiment_id

    columns = ",".join(row)
    placeholders = ",".join(":" + key for key in row)
    try:
        audit._conn.execute(
            f"INSERT INTO recovery_audit ({columns}) VALUES ({placeholders})", row
        )
    except Exception:
        return False        # duplicate, or a column this schema does not have
    audit._conn.commit()
    return True


def consume(messages, audit: AuditLog, experiment_id=None, limit=None) -> int:
    """Drain an iterable of messages into the audit store.

    Takes any iterable rather than a KafkaConsumer so the loop can be tested
    without a broker; in the docker topology the caller passes the real
    consumer, whose messages carry `.value`.
    """
    written = 0
    for index, message in enumerate(messages):
        if limit is not None and index >= limit:
            break
        payload = getattr(message, "value", message)
        if handle_message(payload, audit, experiment_id):
            written += 1
    return written


def run_consumer(db_path="recovery_audit.db", topic: str = DEFAULT_TOPIC,
                 limit=None):      # pragma: no cover - needs a live broker
    """Entry point for the docker-compose topology.

    Untested here by construction: it needs a running broker, which CI does not
    have. The logic it delegates to is covered; the transport is not.
    """
    from kafka import KafkaConsumer

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS") or \
        f"{os.environ.get('KAFKA_HOST', 'localhost')}:9092"
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        consumer_timeout_ms=10000,
    )
    audit = AuditLog(db_path)
    try:
        written = consume(consumer, audit, limit=limit)
        print(f"[recovery] wrote {written} audit rows from {topic}")
        return written
    finally:
        audit.close()
        consumer.close()


if __name__ == "__main__":     # pragma: no cover
    run_consumer()
