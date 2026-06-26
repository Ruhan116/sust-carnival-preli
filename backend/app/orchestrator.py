"""
Layer 2 + 6 — Request Orchestrator & Output Builder.

Coordinates the full pipeline:
  Pre-process → LLM → Safety → Schema → Output
"""

from __future__ import annotations

import logging
import random
from typing import Any

from app.llm_core import analyze_with_llm
from app.preprocessing import PreprocessedInput, preprocess
from app.safety_guardrails import run_safety_pipeline
from app.schemas import TicketRequest, TicketResponse, TransactionEntry

logger = logging.getLogger("queuestorm.orchestrator")


# ── Confidence derivation ────────────────────────────────────────────────────

def _derive_confidence(verdict: str, existing: Any | None) -> float:
    """Derive confidence if not provided or out of range."""
    if existing is not None:
        try:
            val = float(existing)
            if 0.0 <= val <= 1.0:
                return val
        except (TypeError, ValueError):
            pass

    # Default ranges based on verdict
    ranges = {
        "consistent": (0.88, 0.94),
        "inconsistent": (0.75, 0.85),
        "insufficient_data": (0.50, 0.65),
    }
    lo, hi = ranges.get(verdict, (0.50, 0.65))
    return round(random.uniform(lo, hi), 2)


# ── Human review enforcement ────────────────────────────────────────────────

def _enforce_human_review(
    data: dict,
    transactions: list[TransactionEntry],
) -> bool:
    """Determine if human_review_required should be True.

    Returns True if any of the mandatory conditions are met,
    regardless of what the LLM set.
    """
    verdict = data.get("evidence_verdict", "")
    case_type = data.get("case_type", "")
    severity = data.get("severity", "")

    # Always true conditions
    if verdict == "inconsistent":
        return True
    if verdict == "insufficient_data" and severity in ("medium", "high", "critical"):
        return True
    if case_type == "wrong_transfer":
        return True
    if case_type == "phishing_or_social_engineering":
        return True
    if case_type == "refund_request" and verdict == "consistent":
        return True
    if severity in ("critical", "high"):
        return True

    # High-value transactions (>10,000 BDT) always require human review
    for tx in transactions:
        if tx.amount > 10000:
            return True

    return data.get("human_review_required", False)


# ── Reason codes builder ────────────────────────────────────────────────────

def _build_reason_codes(data: dict, ctx: PreprocessedInput) -> list[str]:
    """Build reason_codes if not provided or supplement existing ones."""
    codes: list[str] = list(data.get("reason_codes", []) or [])

    # Add derived codes
    case_type = data.get("case_type", "")
    verdict = data.get("evidence_verdict", "")

    if case_type and case_type not in codes:
        codes.append(case_type)

    if verdict == "consistent" and "transaction_match" not in codes:
        codes.append("transaction_match")
    elif verdict == "inconsistent" and "evidence_mismatch" not in codes:
        codes.append("evidence_mismatch")
    elif verdict == "insufficient_data" and "no_matching_transaction" not in codes:
        codes.append("no_matching_transaction")

    if ctx.injection_detected and "injection_attempt" not in codes:
        codes.append("injection_attempt")

    if ctx.match_signals.amount_matches and "amount_match" not in codes:
        codes.append("amount_match")

    # Check for high-value flag
    for tx in ctx.transactions:
        if tx.amount > 10000:
            if "high_value" not in codes:
                codes.append("high_value")
            break

    return codes[:10]  # Cap at 10 codes


# ── Output assembly ──────────────────────────────────────────────────────────

def _finalize_response(data: dict, ctx: PreprocessedInput) -> TicketResponse:
    """Apply safety guardrails and output building to a raw analysis dict."""
    safety_result = run_safety_pipeline(
        data=data,
        request_ticket_id=ctx.ticket_id,
        transactions=ctx.transactions,
    )

    if safety_result.credential_violation:
        logger.warning("Credential safety violation detected for ticket %s", ctx.ticket_id)
    if safety_result.commitment_violation:
        logger.warning("Commitment safety violation detected for ticket %s", ctx.ticket_id)
    if safety_result.third_party_violation:
        logger.warning("Third-party safety violation detected for ticket %s", ctx.ticket_id)
    if safety_result.schema_errors:
        logger.warning(
            "Schema issues for ticket %s: %s",
            ctx.ticket_id,
            "; ".join(safety_result.schema_errors),
        )

    finalized = safety_result.data
    finalized["confidence"] = _derive_confidence(
        finalized.get("evidence_verdict", "insufficient_data"),
        finalized.get("confidence"),
    )
    finalized["reason_codes"] = _build_reason_codes(finalized, ctx)
    finalized["human_review_required"] = _enforce_human_review(
        finalized, ctx.transactions
    )
    finalized["ticket_id"] = ctx.ticket_id

    try:
        return TicketResponse(**finalized)
    except Exception as e:
        logger.error("Failed to build TicketResponse for %s: %s", ctx.ticket_id, e)
        return TicketResponse(
            ticket_id=ctx.ticket_id,
            relevant_transaction_id=None,
            evidence_verdict="insufficient_data",
            case_type="other",
            severity="medium",
            department="customer_support",
            agent_summary="Automated analysis encountered an error. Manual review required.",
            recommended_next_action="Escalate to supervisor for manual review.",
            customer_reply=(
                "Thank you for contacting us. Your complaint has been received and "
                "is being reviewed by our team. We will update you through official "
                "channels shortly."
            ),
            human_review_required=True,
            confidence=0.3,
            reason_codes=["analysis_error", "manual_review_needed"],
        )


# ── Main orchestrator ───────────────────────────────────────────────────────

async def process_ticket(request: TicketRequest) -> TicketResponse:
    """Full pipeline: preprocess → LLM → safety → output.

    Args:
        request: Validated ticket request from the API.

    Returns:
        TicketResponse ready to serialize to JSON.
    """
    # Layer 3: Pre-processing
    ctx = preprocess(request)
    logger.info(
        "Preprocessed ticket %s: lang=%s, injection=%s, tx_count=%d",
        ctx.ticket_id,
        ctx.detected_language,
        ctx.injection_detected,
        len(ctx.transactions),
    )

    # Layer 4: LLM call (falls back to rule-based internally on failure)
    llm_result = await analyze_with_llm(ctx)
    logger.info("Analysis returned result for ticket %s", ctx.ticket_id)

    return _finalize_response(llm_result, ctx)


# Public alias for use in main.py fallback path
finalize_response = _finalize_response
