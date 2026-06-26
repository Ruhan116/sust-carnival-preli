"""
Layer 2 + 6 — Request Orchestrator & Output Builder.

Pipeline (AI-independent backbone):

    Pre-process
      → Deterministic Rule Engine  (authoritative for ALL scored fields)
      → Optional LLM prose polish   (best-effort; instant fail-over to rules)
      → Safety guardrails           (defense in depth on final text)
      → Schema-valid TicketResponse

The rule engine alone produces a complete, correct, safe response. The LLM
never changes the structured decision and is skipped entirely when the Gemini
key is rate-limited (60s cooldown) or disabled.
"""

from __future__ import annotations

import logging

from app.llm_core import LLM_MODE, decide_full, enhance_prose, llm_enabled
from app.preprocessing import preprocess
from app.rule_engine import run_rule_engine
from app.safety_guardrails import run_safety_pipeline
from app.schemas import TicketRequest, TicketResponse

logger = logging.getLogger("queuestorm.orchestrator")

_PROSE_KEYS = ("agent_summary", "recommended_next_action", "customer_reply")

# Structured fields the LLM may override. Enums/ids are re-validated by the
# safety + schema guardrails downstream, so a bad LLM value cannot break output.
_DECISION_KEYS = (
    "relevant_transaction_id", "evidence_verdict", "case_type", "severity",
    "department", "agent_summary", "recommended_next_action", "customer_reply",
    "human_review_required", "confidence", "reason_codes",
)


def _safe_fallback(ticket_id: str) -> TicketResponse:
    """Last-resort response if even response construction fails."""
    return TicketResponse(
        ticket_id=ticket_id,
        relevant_transaction_id=None,
        evidence_verdict="insufficient_data",
        case_type="other",
        severity="medium",
        department="customer_support",
        agent_summary="Automated analysis encountered an error. Manual review required.",
        recommended_next_action="Escalate to a supervisor for manual review.",
        customer_reply=(
            "Thank you for contacting us. Your complaint has been received and is "
            "being reviewed by our team. We will update you through official "
            "channels shortly. Please do not share your PIN or OTP with anyone."
        ),
        human_review_required=True,
        confidence=0.3,
        reason_codes=["analysis_error", "manual_review_needed"],
    )


async def process_ticket(request: TicketRequest) -> TicketResponse:
    """Full pipeline. Rules are authoritative; the LLM only polishes prose."""
    # Layer 3: Pre-processing
    ctx = preprocess(request)

    # Layer 4R: Deterministic rule engine — the authoritative decision + safe text.
    data = run_rule_engine(ctx)
    logger.info(
        "Rule engine: ticket=%s case=%s verdict=%s rtx=%s sev=%s dept=%s hrr=%s",
        ctx.ticket_id, data["case_type"], data["evidence_verdict"],
        data["relevant_transaction_id"], data["severity"], data["department"],
        data["human_review_required"],
    )

    # Layer 4 (LLM): the rule output above is the SAFE BASELINE + fallback. When
    # the LLM is available it produces the authoritative decision; we merge any
    # valid fields it returns over the baseline. Every enum, the ticket_id, and
    # the relevant_transaction_id are re-validated by the safety/schema guardrails
    # below, so a wrong or malformed LLM value can never break or unsafe the
    # response — at worst we keep the deterministic rule value.
    if llm_enabled():
        if LLM_MODE == "enhance":
            # Legacy mode: LLM only polishes the three prose fields.
            try:
                prose = await enhance_prose(ctx, data)
            except Exception as e:  # noqa: BLE001
                logger.warning("Prose enhancement raised: %s", e)
                prose = None
            if prose:
                for k in _PROSE_KEYS:
                    if prose.get(k):
                        data[k] = prose[k]
                logger.info("Applied LLM prose enhancement for ticket %s", ctx.ticket_id)
        else:
            # Authoritative mode: LLM decides; rules remain the fallback.
            try:
                decision = await decide_full(ctx, data)
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM decision raised: %s", e)
                decision = None
            if decision:
                for k in _DECISION_KEYS:
                    if k not in decision:
                        continue
                    # relevant_transaction_id may legitimately be null (the LLM
                    # refusing to guess on ambiguous/insufficient evidence), so
                    # let null through; for all other fields a null is ignored.
                    if decision[k] is None and k != "relevant_transaction_id":
                        continue
                    data[k] = decision[k]
                logger.info(
                    "Applied LLM decision for ticket %s: case=%s verdict=%s rtx=%s",
                    ctx.ticket_id, data.get("case_type"),
                    data.get("evidence_verdict"), data.get("relevant_transaction_id"),
                )

    # Layer 5: Safety guardrails (defense in depth — especially for LLM prose).
    safety_result = run_safety_pipeline(
        data=data,
        request_ticket_id=ctx.ticket_id,
        transactions=ctx.transactions,
    )
    if safety_result.credential_violation:
        logger.warning("Credential safety violation neutralized for %s", ctx.ticket_id)
    if safety_result.commitment_violation:
        logger.warning("Commitment safety violation neutralized for %s", ctx.ticket_id)
    data = safety_result.data

    # Layer 6: Build the response. Always echo ticket_id.
    data["ticket_id"] = ctx.ticket_id
    try:
        return TicketResponse(**data)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to build TicketResponse for %s: %s", ctx.ticket_id, e)
        return _safe_fallback(ctx.ticket_id)
