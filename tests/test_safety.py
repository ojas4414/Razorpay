"""Kill switch, spend caps, and the Razorpay ingestion shim.

These guard the two directions something like this can go wrong in production: a
bug that spends money faster than anyone notices, and an endpoint that lets a
stranger inject events into a discount engine.

The webhook tests run against fixtures shaped like the documented payload. No
live Razorpay account is involved, and that limit is stated in the module rather
than implied away.
"""
import hashlib
import hmac
import json

import pytest

from core.session import Session
from integrations.razorpay_webhooks import (
    PAISE_PER_RUPEE,
    RECOVERY_TRIGGERS,
    handle_webhook,
    parse_event,
    verify_signature,
)
from recovery.detector import Flag
from recovery.intervention import InterventionPolicy
from recovery.safety import KillSwitch, Refusal, SafetyGate, SpendGuard

SECRET = "whsec_test_secret"


def webhook_body(event="payment.failed", amount=249900, status="failed"):
    return json.dumps({
        "entity": "event", "event": event, "contains": ["payment"],
        "payload": {"payment": {"entity": {
            "id": "pay_123", "order_id": "order_9", "amount": amount,
            "currency": "INR", "status": status,
            "error_reason": "insufficient_funds"}}},
        "created_at": 1735689600,
    }).encode("utf-8")


def signed(body, secret=SECRET):
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def flagged_session(session_id="s1", cart_value=300.0):
    s = Session(agent_id=1, persona="average_buyer", started_at=0.0)
    s.session_id = session_id
    s.add_to_cart("P1", cart_value)
    s.flag = Flag(session_id=session_id, sim_time=1.0, stage="cart",
                  trigger="price_shock", features={})
    return s


class TestKillSwitch:
    def test_starts_disengaged(self):
        assert KillSwitch().engaged is False

    def test_records_who_and_why(self):
        """An off switch nobody can account for afterwards is half a control."""
        switch = KillSwitch()
        switch.engage("detector regression", by="ojas")
        assert switch.engaged is True
        assert switch.reason == "detector regression"
        assert switch.engaged_by == "ojas"
        assert switch.engaged_at is not None

    def test_releases(self):
        switch = KillSwitch()
        switch.engage("testing")
        switch.release()
        assert switch.engaged is False
        assert switch.engaged_at is None


class TestSpendGuard:
    def test_allows_within_limits(self):
        assert SpendGuard().check(100.0, 0.15) is None

    def test_refuses_an_absurd_discount(self):
        """A model asking for 90% off is a broken model, not an opportunity."""
        guard = SpendGuard(max_discount_rate=0.40)
        assert guard.check(10.0, 0.90) == Refusal.BUDGET_EXHAUSTED

    def test_caps_offer_volume(self):
        guard = SpendGuard(max_offers_per_window=3, max_spend_per_window=1e9)
        for _ in range(3):
            assert guard.check(1.0, 0.15) is None
            guard.record(1.0)
        assert guard.check(1.0, 0.15) == Refusal.RATE_LIMITED

    def test_caps_total_spend(self):
        """A flood of small discounts is still a denial of service on margin."""
        guard = SpendGuard(max_offers_per_window=10_000, max_spend_per_window=100.0)
        guard.record(90.0)
        assert guard.check(20.0, 0.15) == Refusal.BUDGET_EXHAUSTED
        assert guard.check(5.0, 0.15) is None

    def test_window_slides(self):
        guard = SpendGuard(max_offers_per_window=1, window_seconds=60)
        guard.record(1.0, now=1000.0)
        assert guard.check(1.0, 0.15, now=1030.0) == Refusal.RATE_LIMITED
        assert guard.check(1.0, 0.15, now=1100.0) is None

    def test_counts_every_refusal(self):
        guard = SpendGuard(max_offers_per_window=0)
        for _ in range(3):
            guard.check(1.0, 0.15)
        assert guard.status()["refusals"][Refusal.RATE_LIMITED] == 3


class TestSafetyGate:
    def test_kill_switch_beats_everything(self):
        gate = SafetyGate()
        gate.kill_switch.engage("stop")
        assert gate.allow(100.0, 0.15) == Refusal.KILLED

    def test_records_issued_spend(self):
        gate = SafetyGate()
        gate.issued(1000.0, 0.15)
        assert gate.guard.spend_in_window() == pytest.approx(150.0)

    def test_status_is_serialisable(self):
        json.dumps(SafetyGate().status())


class TestPolicyIntegration:
    def test_rate_limit_stops_the_policy_issuing(self):
        gate = SafetyGate()
        gate.guard.max_offers_per_window = 2
        policy = InterventionPolicy(discount_rate=0.15, safety=gate)
        for i in range(5):
            policy.dispatch(flagged_session(f"s{i}"), 1.0)
        assert policy.issued == 2
        assert policy.refused_by_safety == 3

    def test_kill_switch_stops_the_policy_issuing(self):
        gate = SafetyGate()
        gate.kill_switch.engage("incident")
        policy = InterventionPolicy(discount_rate=0.15, safety=gate)
        assert policy.dispatch(flagged_session(), 1.0) is None
        assert policy.refused_by_safety == 1

    def test_no_gate_means_no_restriction(self):
        """The simulation runs without safety wired; it must behave as before."""
        policy = InterventionPolicy(discount_rate=0.15)
        assert policy.dispatch(flagged_session(), 1.0) is not None

    def test_refusals_appear_in_stats(self):
        gate = SafetyGate()
        gate.kill_switch.engage("x")
        policy = InterventionPolicy(discount_rate=0.15, safety=gate)
        policy.dispatch(flagged_session(), 1.0)
        assert policy.stats()["refused_by_safety"] == 1


class TestWebhookVerification:
    def test_accepts_a_correct_signature(self):
        body = webhook_body()
        assert verify_signature(body, signed(body), SECRET) is True

    def test_rejects_a_forged_signature(self):
        """An unverified endpoint lets a stranger drive a discount engine."""
        assert verify_signature(webhook_body(), "deadbeef", SECRET) is False

    def test_rejects_a_missing_signature_or_secret(self):
        body = webhook_body()
        assert verify_signature(body, "", SECRET) is False
        assert verify_signature(body, signed(body), "") is False

    def test_a_tampered_body_fails(self):
        body = webhook_body()
        signature = signed(body)
        assert verify_signature(webhook_body(amount=1), signature, SECRET) is False


class TestWebhookParsing:
    def test_converts_paise_to_rupees(self):
        """Treating paise as rupees inflates every cart 100x -- the kind of bug
        that survives a demo and surfaces in production."""
        event = parse_event(json.loads(webhook_body(amount=249900)))
        assert event.amount == pytest.approx(249900 / PAISE_PER_RUPEE)
        assert event.amount == 2499.0

    def test_payment_failed_is_a_recovery_opportunity(self):
        event = parse_event(json.loads(webhook_body("payment.failed")))
        assert event.is_recovery_opportunity is True
        assert event.stage == "cart"
        assert "payment.failed" in RECOVERY_TRIGGERS

    def test_order_paid_is_a_conversion_not_an_opportunity(self):
        event = parse_event(json.loads(webhook_body("order.paid", status="paid")))
        assert event.stage == "checkout"
        assert event.is_recovery_opportunity is False

    def test_irrelevant_events_map_to_no_stage(self):
        """Refunds and settlements are ignored on purpose, not by accident."""
        for name in ("refund.created", "settlement.processed",
                     "payment.dispute.created"):
            assert parse_event(json.loads(webhook_body(name))).stage is None

    def test_unknown_events_are_refused(self):
        assert parse_event({"event": "something.invented"}) is None
        assert parse_event({}) is None
        assert parse_event("not a dict") is None

    def test_session_update_carries_no_customer_identity(self):
        """Every detector feature is behavioural; identity must not cross this
        boundary either."""
        update = parse_event(json.loads(webhook_body())).to_session_update()
        for forbidden in ("email", "contact", "customer_id", "name", "phone"):
            assert forbidden not in update


class TestWebhookEndToEnd:
    def test_verified_payload_becomes_an_event(self):
        body = webhook_body()
        event, error = handle_webhook(body, signed(body), SECRET)
        assert error is None
        assert event.order_id == "order_9"
        assert event.failure_reason == "insufficient_funds"

    def test_unverified_payload_never_reaches_parsing(self):
        event, error = handle_webhook(webhook_body(), "bad", SECRET)
        assert event is None
        assert "signature" in error

    def test_malformed_json_is_reported(self):
        body = b"{not json"
        event, error = handle_webhook(body, signed(body), SECRET)
        assert event is None
        assert "JSON" in error
