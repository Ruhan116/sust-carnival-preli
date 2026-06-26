"""
Layer 5 — Safety Guardrails (Post-Generation).

Three deterministic code-level checks that run on the LLM output:
  5a. Credential Filter — blocks PIN/OTP/password requests
  5b. Unauthorized Commitment Filter — blocks refund confirmations
  5c. Schema Validator — enforces response structure correctness
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas import (
    CaseType,
    Department,
    EvidenceVerdict,
    Severity,
    TransactionEntry,
)


# ── 5a. Credential Filter ───────────────────────────────────────────────────

# A credential token that must never be *requested* from the customer.
_CREDENTIAL_NOUN = (
    r"(?:pin|otp|password|passcode|cvv|"
    r"one[\s-]?time[\s-]?password|security\s+code|full\s+card\s+number|"
    r"card\s+number)"
)

# A solicitation = an imperative/request verb targeting a credential noun,
# possibly with a few words in between ("provide your one-time password").
_SOLICIT_VERB = (
    r"(?:share|provide|enter|send|give|tell|type|input|submit|confirm|verify|"
    r"reveal|disclose|need|require|ask\s+for)"
)

# Negation that turns a solicitation into a SAFE reminder, e.g.
# "do NOT share your PIN", "we NEVER ask for your OTP", "without sharing your PIN".
_NEGATION = (
    r"(?:not|never|n't|no\b|without|avoid|refrain|cannot|can't|won't|"
    r"don't|do\s+not|will\s+never|do\s+we|would\s+we)"
)

_CREDENTIAL_REQUEST = re.compile(
    rf"{_SOLICIT_VERB}\b[^.?!]{{0,40}}?\b{_CREDENTIAL_NOUN}\b",
    re.IGNORECASE,
)
# Direct interrogative form: "what is your PIN?", "your OTP is ...?"
_CREDENTIAL_QUESTION = re.compile(
    rf"what\s+(?:is|are)\s+(?:your\s+)?{_CREDENTIAL_NOUN}\b",
    re.IGNORECASE,
)
_NEGATION_BEFORE = re.compile(rf"{_NEGATION}\b", re.IGNORECASE)

SAFE_CREDENTIAL_REPLACEMENT = (
    "For your security, please never share your PIN, OTP, or password with "
    "anyone, including our support team. Our team will review your case and "
    "contact you only through official channels."
)


def _is_genuine_request(text: str, match: re.Match) -> bool:
    """True only if the credential mention is an actual solicitation, i.e. it is
    NOT preceded by a negation within the same clause.

    This preserves the protective reminder "do not share your PIN or OTP", which
    the official sample answers use, while still catching real requests like
    "please share your OTP for verification".
    """
    clause_start = max(
        text.rfind(".", 0, match.start()),
        text.rfind("?", 0, match.start()),
        text.rfind("!", 0, match.start()),
    )
    window = text[clause_start + 1 : match.start()]
    return _NEGATION_BEFORE.search(window) is None


def filter_credential_requests(text: str) -> tuple[str, bool]:
    """Replace text only if it *requests* credentials from the customer.

    A defensive reminder ("never share your PIN or OTP") is safe and preserved.

    Returns:
        Tuple of (safe_text, violation_detected).
    """
    for m in _CREDENTIAL_REQUEST.finditer(text):
        if _is_genuine_request(text, m):
            return SAFE_CREDENTIAL_REPLACEMENT, True
    for m in _CREDENTIAL_QUESTION.finditer(text):
        if _is_genuine_request(text, m):
            return SAFE_CREDENTIAL_REPLACEMENT, True
    return text, False


# ── 5b. Unauthorized Commitment Filter ───────────────────────────────────────

COMMITMENT_PATTERNS = [
    r"we\s+will\s+refund",
    r"your\s+(money|amount|balance)\s+will\s+be\s+returned",
    r"we\s+(confirm|confirmed)\s+the\s+reversal",
    r"we\s+will\s+credit",
    r"refund\s+has\s+been\s+processed",
    r"reversal\s+is\s+(complete|done|processed)",
    r"we\s+guarantee\s+",
    r"you\s+will\s+receive\s+your\s+(money|refund|amount)",
    r"we\s+have\s+(already\s+)?refunded",
    r"we\s+have\s+(already\s+)?reversed",
    r"we\s+have\s+(already\s+)?credited",
    r"your\s+refund\s+(is|has\s+been)\s+(processed|completed|approved)",
    r"we\s+are\s+processing\s+your\s+refund",
    r"the\s+refund\s+will\s+be\s+(processed|completed)",
    r"we\s+will\s+(reverse|return|credit)\s+(your|the)",
]

_COMPILED_COMMITMENT = [re.compile(p, re.IGNORECASE) for p in COMMITMENT_PATTERNS]

SAFE_COMMITMENT_REPLACEMENT = (
    "Any eligible amount will be returned through official channels, "
    "subject to verification and our standard dispute resolution process."
)


def filter_unauthorized_commitments(text: str) -> tuple[str, bool]:
    """Scan text for unauthorized refund/reversal commitments.

    Returns:
        Tuple of (safe_text, violation_detected).
    """
    for pattern in _COMPILED_COMMITMENT:
        if pattern.search(text):
            # Replace matching sentences with safe language
            safe_text = text
            for p in _COMPILED_COMMITMENT:
                # Replace the sentence containing the commitment
                safe_text = p.sub(SAFE_COMMITMENT_REPLACEMENT, safe_text)
            return safe_text, True
    return text, False


# ── 5c. Schema Validator ────────────────────────────────────────────────────

VALID_CASE_TYPES = {e.value for e in CaseType}
VALID_SEVERITIES = {e.value for e in Severity}
VALID_DEPARTMENTS = {e.value for e in Department}
VALID_VERDICTS = {e.value for e in EvidenceVerdict}


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    fixed_data: dict | None = None


def validate_response_schema(
    data: dict,
    request_ticket_id: str,
    valid_transaction_ids: set[str],
) -> ValidationResult:
    """Validate LLM response against the required schema.

    Args:
        data: Parsed LLM response dict.
        request_ticket_id: The ticket_id from the request (must match).
        valid_transaction_ids: Set of transaction IDs from the input history.

    Returns:
        ValidationResult with errors and optionally fixed data.
    """
    result = ValidationResult()
    fixed = dict(data)

    # Required fields
    required_fields = [
        "ticket_id", "relevant_transaction_id", "evidence_verdict",
        "case_type", "severity", "department", "agent_summary",
        "recommended_next_action", "customer_reply", "human_review_required",
    ]

    for f in required_fields:
        if f not in data:
            result.is_valid = False
            result.errors.append(f"Missing required field: {f}")

    # ticket_id must match
    if data.get("ticket_id") != request_ticket_id:
        fixed["ticket_id"] = request_ticket_id
        result.errors.append("ticket_id mismatch — corrected")

    # Enum validations
    if data.get("case_type") not in VALID_CASE_TYPES:
        result.is_valid = False
        result.errors.append(f"Invalid case_type: {data.get('case_type')}")
        fixed["case_type"] = "other"

    if data.get("severity") not in VALID_SEVERITIES:
        result.is_valid = False
        result.errors.append(f"Invalid severity: {data.get('severity')}")
        fixed["severity"] = "medium"

    if data.get("department") not in VALID_DEPARTMENTS:
        result.is_valid = False
        result.errors.append(f"Invalid department: {data.get('department')}")
        fixed["department"] = "customer_support"

    if data.get("evidence_verdict") not in VALID_VERDICTS:
        result.is_valid = False
        result.errors.append(f"Invalid evidence_verdict: {data.get('evidence_verdict')}")
        fixed["evidence_verdict"] = "insufficient_data"

    # relevant_transaction_id must be null or a valid ID from history
    rtx = data.get("relevant_transaction_id")
    if rtx is not None and rtx not in valid_transaction_ids:
        fixed["relevant_transaction_id"] = None
        result.errors.append(
            f"relevant_transaction_id '{rtx}' not in transaction history — set to null"
        )

    # confidence must be 0.0–1.0 if present
    conf = data.get("confidence")
    if conf is not None:
        try:
            conf_val = float(conf)
            if not (0.0 <= conf_val <= 1.0):
                fixed["confidence"] = max(0.0, min(1.0, conf_val))
                result.errors.append("confidence out of range — clamped")
        except (TypeError, ValueError):
            fixed["confidence"] = None
            result.errors.append("confidence not a valid number — set to null")

    # human_review_required must be bool
    if not isinstance(data.get("human_review_required"), bool):
        fixed["human_review_required"] = True
        result.errors.append("human_review_required not a bool — defaulted to true")

    # Ensure string fields are non-empty strings
    for sf in ("agent_summary", "recommended_next_action", "customer_reply"):
        val = data.get(sf)
        if not isinstance(val, str) or not val.strip():
            fixed[sf] = _default_string(sf)
            result.errors.append(f"{sf} is empty or not a string — using default")

    result.fixed_data = fixed
    return result


def _default_string(field_name: str) -> str:
    """Return a safe default string for a required field."""
    defaults = {
        "agent_summary": "Complaint received. Requires manual investigation.",
        "recommended_next_action": "Escalate to supervisor for manual review.",
        "customer_reply": (
            "Thank you for contacting us. Your complaint has been received and is being "
            "reviewed by our team. We will update you through official channels shortly."
        ),
    }
    return defaults.get(field_name, "N/A")


# ── Combined Safety Pipeline ────────────────────────────────────────────────

@dataclass
class SafetyCheckResult:
    """Result of running all safety checks."""
    data: dict
    credential_violation: bool = False
    commitment_violation: bool = False
    schema_errors: list[str] = field(default_factory=list)


def run_safety_pipeline(
    data: dict,
    request_ticket_id: str,
    transactions: list[TransactionEntry],
) -> SafetyCheckResult:
    """Run all post-generation safety checks on LLM output.

    Args:
        data: Parsed LLM response dict.
        request_ticket_id: The ticket_id from the request.
        transactions: List of transaction entries from input.

    Returns:
        SafetyCheckResult with cleaned data and violation flags.
    """
    result = SafetyCheckResult(data=dict(data))

    # 5a. Credential filter on customer_reply
    if "customer_reply" in result.data:
        safe_reply, cred_violation = filter_credential_requests(
            result.data["customer_reply"]
        )
        if cred_violation:
            result.credential_violation = True
            result.data["customer_reply"] = safe_reply

    # 5b. Unauthorized commitment filter on customer_reply
    if "customer_reply" in result.data:
        safe_reply, commit_violation = filter_unauthorized_commitments(
            result.data["customer_reply"]
        )
        if commit_violation:
            result.commitment_violation = True
            result.data["customer_reply"] = safe_reply

    # 5b. Also check recommended_next_action
    if "recommended_next_action" in result.data:
        safe_action, commit_violation = filter_unauthorized_commitments(
            result.data["recommended_next_action"]
        )
        if commit_violation:
            result.commitment_violation = True
            result.data["recommended_next_action"] = safe_action

    # 5c. Schema validation
    valid_tx_ids = {tx.transaction_id for tx in transactions}
    validation = validate_response_schema(
        result.data, request_ticket_id, valid_tx_ids
    )
    result.schema_errors = validation.errors
    if validation.fixed_data:
        result.data = validation.fixed_data

    return result
