"""Turning Razorpay webhook events into the sessions this pipeline understands.

The pipeline runs on simulated sessions. This is the shim that would let it run
on real ones: it verifies a webhook, normalises the payload, and maps it onto the
same funnel stages `core.session` already models.

**It is not connected to a Razorpay account and has never seen a live webhook.**
It is written against the documented payload shape and tested against captured
fixtures. Shipping it makes the integration path concrete rather than hand-waved,
and stating this plainly is the difference between a design and a claim.

What it does establish:

*Signature verification is not optional.* An unauthenticated webhook endpoint is
an endpoint anyone can use to inject events into a discount engine. The check is
constant-time, and unverified payloads never reach the mapping stage.

*Amounts arrive in paise.* Razorpay works in the smallest currency unit; treating
that value as rupees inflates every cart by 100x, which is precisely the kind of
error that survives a demo and surfaces in production.

*Not every event is a funnel stage.* Refunds, settlements and disputes are
recognised and deliberately ignored rather than being force-fitted onto a stage.

The event that matters most for recovery is `payment.failed`: the shopper reached
checkout, tried, and did not complete. That is the moment an intervention has
something to act on, and it is where a live deployment would hook in.
"""
import hashlib
import hmac
import json
from dataclasses import dataclass, field

# Documented Razorpay webhook events, mapped to the funnel stage each implies.
# Only the ones that mean something for checkout recovery are mapped; the rest
# are recognised so they can be ignored on purpose rather than by accident.
STAGE_BY_EVENT = {
    "order.paid": "checkout",
    "payment.captured": "checkout",
    "payment.authorized": "checkout",
    "payment.failed": "cart",          # reached checkout, did not complete
    "order.notification.delivered": None,
    "refund.created": None,
    "refund.processed": None,
    "payment.dispute.created": None,
    "settlement.processed": None,
}

# Events that mean a recovery opportunity: the shopper tried and did not finish.
RECOVERY_TRIGGERS = frozenset({"payment.failed"})

PAISE_PER_RUPEE = 100


@dataclass(frozen=True)
class CheckoutEvent:
    """One normalised event, in the vocabulary the pipeline already speaks."""

    event: str
    stage: str | None
    order_id: str | None
    payment_id: str | None
    amount: float                     # rupees, converted from paise
    currency: str
    status: str | None
    failure_reason: str | None
    created_at: int | None
    is_recovery_opportunity: bool
    raw_keys: tuple = field(default_factory=tuple)

    def to_session_update(self) -> dict:
        """The shape live/engine's session model consumes.

        Deliberately carries no customer identifier. Every feature the detector
        uses is behavioural, and keeping identity out of this boundary is what
        makes that property hold end to end rather than only inside the
        simulation.
        """
        return {
            "session_key": self.order_id or self.payment_id,
            "stage": self.stage,
            "cart_value": self.amount,
            "terminal": self.stage == "checkout" and self.status in ("captured", "paid"),
            "failed": self.event == "payment.failed",
            "reason_code": self.failure_reason,
        }


def verify_signature(body: bytes, signature: str, secret: str) -> bool:
    """Constant-time HMAC-SHA256 check, as Razorpay documents it.

    An unverified webhook endpoint lets anyone inject events into something that
    issues discounts, so this gates the mapping rather than sitting beside it.
    """
    if not signature or not secret:
        return False
    expected = hmac.new(secret.encode("utf-8"),
                        body if isinstance(body, bytes) else body.encode("utf-8"),
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _entity(payload: dict, name: str) -> dict:
    return (payload.get("payload", {}).get(name, {}) or {}).get("entity", {}) or {}


def parse_event(payload: dict) -> CheckoutEvent | None:
    """Normalise a webhook body. Returns None for anything unrecognised."""
    if not isinstance(payload, dict):
        return None
    event = payload.get("event")
    if not event or event not in STAGE_BY_EVENT:
        return None

    payment = _entity(payload, "payment")
    order = _entity(payload, "order")
    source = payment or order

    amount_paise = source.get("amount") or 0
    return CheckoutEvent(
        event=event,
        stage=STAGE_BY_EVENT[event],
        order_id=order.get("id") or payment.get("order_id"),
        payment_id=payment.get("id"),
        amount=round(amount_paise / PAISE_PER_RUPEE, 2),
        currency=source.get("currency", "INR"),
        status=source.get("status"),
        failure_reason=payment.get("error_reason") or payment.get("error_code"),
        created_at=payload.get("created_at"),
        is_recovery_opportunity=event in RECOVERY_TRIGGERS,
        raw_keys=tuple(sorted(payload.keys())),
    )


def handle_webhook(body: bytes, signature: str, secret: str):
    """Verify then parse. Returns (event, error) with exactly one of them set."""
    if not verify_signature(body, signature, secret):
        return None, "signature verification failed"
    try:
        payload = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
    except (ValueError, UnicodeDecodeError):
        return None, "payload is not valid JSON"

    event = parse_event(payload)
    if event is None:
        return None, f"unrecognised event: {payload.get('event')!r}"
    return event, None
