"""
Rule-based fallback analyzer (Alternative B).

Used when the LLM is unavailable, rate-limited, or returns invalid output.
Uses pre-processing signals and keyword heuristics for classification.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from app.preprocessing import PreprocessedInput
from app.schemas import TransactionEntry

# ── Keyword classifiers ──────────────────────────────────────────────────────

_CASE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("phishing_or_social_engineering", [
        "pin", "otp", "password", "passcode", "scam", "phishing", "fraud call",
        "fake call", "suspicious call", "social engineering", "verify your account",
    ]),
    ("wrong_transfer", [
        "wrong number", "wrong recipient", "wrong person", "accidentally sent",
        "sent to wrong", "mistaken transfer", "wrong account",
    ]),
    ("payment_failed", [
        "payment failed", "failed payment", "transaction failed", "deducted",
        "money taken", "balance deducted", "charged but failed", "did not go through",
    ]),
    ("refund_request", [
        "refund", "money back", "return my money", "cancelled order", "want back",
    ]),
    ("duplicate_payment", [
        "charged twice", "double charge", "duplicate", "paid twice", "two times",
    ]),
    ("merchant_settlement_delay", [
        "settlement", "merchant payment", "not received settlement", "sales payment",
    ]),
    ("agent_cash_in_issue", [
        "cash in", "cash-in", "cash deposit", "agent deposit", "gave cash to agent",
        "agent did not", "cash through agent",
    ]),
]

_DEPARTMENT_MAP: dict[str, str] = {
    "wrong_transfer": "dispute_resolution",
    "payment_failed": "payments_ops",
    "refund_request": "customer_support",
    "duplicate_payment": "payments_ops",
    "merchant_settlement_delay": "merchant_operations",
    "agent_cash_in_issue": "agent_operations",
    "phishing_or_social_engineering": "fraud_risk",
    "other": "customer_support",
}


def _classify_case_type(complaint: str, user_type: Optional[str]) -> str:
    """Classify case type from complaint keywords."""
    lower = complaint.lower()

    for case_type, keywords in _CASE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return case_type

    if user_type == "merchant" and "settlement" in lower:
        return "merchant_settlement_delay"

    return "other"


def _find_relevant_transaction(
    ctx: PreprocessedInput,
) -> Optional[TransactionEntry]:
    """Pick the most relevant transaction from pre-match signals."""
    if not ctx.transactions:
        return None

    candidate_id = ctx.match_signals.best_candidate_id
    if candidate_id:
        for tx in ctx.transactions:
            if tx.transaction_id == candidate_id:
                return tx

    if len(ctx.match_signals.amount_matches) == 1:
        tx_id = ctx.match_signals.amount_matches[0]
        for tx in ctx.transactions:
            if tx.transaction_id == tx_id:
                return tx

    if len(ctx.transactions) == 1:
        return ctx.transactions[0]

    return None


def _determine_verdict(
    case_type: str,
    tx: Optional[TransactionEntry],
    has_history: bool,
) -> str:
    """Determine evidence verdict from transaction status vs complaint type."""
    if not has_history or tx is None:
        return "insufficient_data"

    status = tx.status.value

    if case_type == "payment_failed":
        if status == "failed":
            return "consistent"
        if status == "completed":
            return "inconsistent"
        return "insufficient_data"

    if case_type == "wrong_transfer":
        return "consistent" if status == "completed" else "insufficient_data"

    if case_type in ("merchant_settlement_delay", "agent_cash_in_issue"):
        if status == "pending":
            return "consistent"
        if status == "completed":
            return "inconsistent"
        return "insufficient_data"

    if case_type == "duplicate_payment":
        return "consistent"

    if case_type == "refund_request":
        return "consistent" if status == "completed" else "insufficient_data"

    if case_type == "phishing_or_social_engineering":
        return "insufficient_data"

    return "consistent" if tx else "insufficient_data"


def _determine_severity(amount: float, case_type: str, verdict: str) -> str:
    """Assign severity based on amount and case type."""
    if case_type == "phishing_or_social_engineering":
        return "critical"
    if amount > 10000:
        return "high"
    if verdict == "inconsistent" and amount > 1000:
        return "high"
    if amount >= 1000:
        return "medium"
    return "low"


def rule_based_analyze(ctx: PreprocessedInput) -> dict[str, Any]:
    """Produce a structured response using deterministic rules.

    Args:
        ctx: Pre-processed ticket data.

    Returns:
        Dict matching the TicketResponse schema fields.
    """
    complaint = ctx.cleaned_complaint
    case_type = _classify_case_type(complaint, ctx.user_type)
    tx = _find_relevant_transaction(ctx)
    has_history = bool(ctx.transactions)
    verdict = _determine_verdict(case_type, tx, has_history)

    amount = tx.amount if tx else 0.0
    if not tx and ctx.transactions:
        amount = max(t.amount for t in ctx.transactions)

    severity = _determine_severity(amount, case_type, verdict)
    department = _DEPARTMENT_MAP.get(case_type, "customer_support")
    relevant_id = tx.transaction_id if tx else None

    # Build summaries
    if case_type == "payment_failed" and tx:
        agent_summary = (
            f"Customer reported payment failure for {tx.amount:.0f} BDT "
            f"({tx.transaction_id}, status={tx.status.value}). "
            f"Evidence is {verdict}."
        )
    elif case_type == "wrong_transfer" and tx:
        agent_summary = (
            f"Customer reports wrong transfer of {tx.amount:.0f} BDT "
            f"to {tx.counterparty} ({tx.transaction_id})."
        )
    elif case_type == "phishing_or_social_engineering":
        agent_summary = (
            "Customer reported a suspected phishing or social engineering attempt. "
            "No matching transaction identified."
        )
    elif not has_history:
        agent_summary = (
            "Customer complaint received with no transaction history provided. "
            "Manual investigation required."
        )
    else:
        agent_summary = (
            f"Complaint classified as {case_type}. "
            f"Relevant transaction: {relevant_id or 'none identified'}."
        )

    recommended_next_action = _recommended_action(case_type, tx, verdict)
    customer_reply = _customer_reply(case_type, ctx.detected_language, tx, verdict)

    human_review = _needs_human_review(case_type, verdict, severity, amount)

    reason_codes = [case_type, "rule_based_fallback"]
    if verdict == "consistent":
        reason_codes.append("transaction_match")
    elif verdict == "inconsistent":
        reason_codes.append("evidence_mismatch")
    else:
        reason_codes.append("no_matching_transaction")
    if amount > 10000:
        reason_codes.append("high_value")
    if ctx.injection_detected:
        reason_codes.append("injection_attempt")

    confidence_map = {
        "consistent": 0.72,
        "inconsistent": 0.68,
        "insufficient_data": 0.55,
    }

    return {
        "ticket_id": ctx.ticket_id,
        "relevant_transaction_id": relevant_id,
        "evidence_verdict": verdict,
        "case_type": case_type,
        "severity": severity,
        "department": department,
        "agent_summary": agent_summary,
        "recommended_next_action": recommended_next_action,
        "customer_reply": customer_reply,
        "human_review_required": human_review,
        "confidence": confidence_map.get(verdict, 0.55),
        "reason_codes": reason_codes[:10],
    }


def _recommended_action(
    case_type: str,
    tx: Optional[TransactionEntry],
    verdict: str,
) -> str:
    """Generate a safe recommended next action."""
    tx_ref = f" for {tx.transaction_id}" if tx else ""

    actions = {
        "payment_failed": (
            f"Investigate the failed payment{tx_ref}, verify balance deduction, "
            "and escalate to payments_ops if funds were debited."
        ),
        "wrong_transfer": (
            f"Open dispute case{tx_ref}, attempt recipient contact, "
            "and escalate to dispute_resolution for reversal authorization."
        ),
        "refund_request": (
            f"Verify original transaction{tx_ref} and route to dispute_resolution "
            "for refund eligibility review."
        ),
        "duplicate_payment": (
            "Compare duplicate transaction records and initiate duplicate "
            "charge investigation with payments_ops."
        ),
        "merchant_settlement_delay": (
            f"Check settlement status{tx_ref} with merchant_operations "
            "and provide expected timeline to merchant."
        ),
        "agent_cash_in_issue": (
            f"Verify agent cash-in record{tx_ref} with agent_operations "
            "and reconcile customer balance."
        ),
        "phishing_or_social_engineering": (
            "Escalate to fraud_risk immediately. Advise customer to never "
            "share credentials and to block suspicious numbers."
        ),
    }
    return actions.get(
        case_type,
        "Review complaint details and escalate to supervisor for manual investigation.",
    )


def _customer_reply(
    case_type: str,
    language: str,
    tx: Optional[TransactionEntry],
    verdict: str,
) -> str:
    """Generate a safe customer-facing reply."""
    if language == "bn":
        base = (
            "আপনার অভিযোগটি আমরা পেয়েছি এবং তদন্ত করা হচ্ছে। "
            "যেকোনো যোগ্য পরিমাণ যাচাইকরণ এবং আমাদের স্ট্যান্ডার্ড "
            "বিরোধ নিষ্পত্তি প্রক্রিয়া সাপেক্ষে অফিসিয়াল চ্যানেলের "
            "মাধ্যমে ফেরত দেওয়া হবে।"
        )
        if case_type == "phishing_or_social_engineering":
            return (
                "আপনার অভিযোগটি গুরুত্ব সহকারে গ্রহণ করা হয়েছে। "
                "অনুগ্রহ করে কখনোই আপনার PIN, OTP বা পাসওয়ার্ড কারো সাথে "
                "শেয়ার করবেন না। শুধুমাত্র আমাদের অফিসিয়াল সাপোর্ট "
                "চ্যানেলের মাধ্যমে যোগাযোগ করুন।"
            )
        return base

    if case_type == "phishing_or_social_engineering":
        return (
            "Thank you for reporting this. Never share your PIN, OTP, or password "
            "with anyone, including callers claiming to be from our team. "
            "Please contact us only through official support channels."
        )

    if tx and case_type == "payment_failed" and verdict == "consistent":
        return (
            f"Thank you for contacting us regarding your {tx.amount:.0f} BDT payment. "
            "We have identified the related transaction and are investigating. "
            "Any eligible amount will be returned through official channels, "
            "subject to verification and our standard dispute resolution process."
        )

    return (
        "Thank you for contacting us. Your complaint has been received and "
        "is being reviewed by our team. We will update you through official "
        "channels shortly."
    )


def _needs_human_review(
    case_type: str,
    verdict: str,
    severity: str,
    amount: float,
) -> bool:
    """Determine if human review is required."""
    if verdict == "inconsistent":
        return True
    if case_type in ("wrong_transfer", "phishing_or_social_engineering"):
        return True
    if case_type == "refund_request" and verdict == "consistent":
        return True
    if severity in ("critical", "high"):
        return True
    if amount > 10000:
        return True
    if verdict == "insufficient_data" and severity in ("medium", "high", "critical"):
        return True
    return False
