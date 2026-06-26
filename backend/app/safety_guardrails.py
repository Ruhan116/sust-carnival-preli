"""
Layer 5 — Safety Guardrails (Post-Generation).

Four deterministic code-level checks that run on the LLM output:
  5a. Credential Filter — blocks PIN/OTP/password requests
  5b. Unauthorized Commitment Filter — blocks refund confirmations
  5c. Third-Party Contact Filter — blocks unofficial contact instructions
  5d. Schema Validator — enforces response structure correctness
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

CREDENTIAL_PATTERNS = [
    r"\b(your\s+)?pin\b",
    r"\b(your\s+)?otp\b",
    r"\bone[\s-]time[\s-]password\b",
    r"\b(your\s+)?password\b",
    r"\b(your\s+)?passcode\b",
    r"\b(full\s+)?card\s+number\b",
    r"\bcvv\b",
    r"\bsecurity\s+code\b",
    r"share\s+(your\s+)?(pin|otp|password)",
    r"provide\s+(your\s+)?(pin|otp|password)",
    r"enter\s+(your\s+)?(pin|otp|password)",
    r"verify\s+(your\s+)?(pin|otp|password)",
    r"confirm\s+(your\s+)?(pin|otp|password)",
    r"send\s+(us\s+)?(your\s+)?(pin|otp|password)",
    r"tell\s+(us\s+)?(your\s+)?(pin|otp|password)",
]

_COMPILED_CREDENTIAL = [re.compile(p, re.IGNORECASE) for p in CREDENTIAL_PATTERNS]

SAFE_CREDENTIAL_REPLACEMENT = (
    "For security, please never share your PIN, OTP, or password with anyone, "
    "including our support team."
)


def filter_credential_requests(text: str) -> tuple[str, bool]:
    """Scan text for credential requests and replace if found.

    Returns:
        Tuple of (safe_text, violation_detected).
    """
    for pattern in _COMPILED_CREDENTIAL:
        if pattern.search(text):
            # Replace the entire reply with the safe template
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


# ── 5c. Third-Party Contact Filter ──────────────────────────────────────────

THIRD_PARTY_PATTERNS = [
    r"contact\s+(this|the|that|an?\s+)?(unofficial|external|third[\s-]?party)",
    r"call\s+(this|the|that)\s+(number|hotline|line)",
    r"call\s+\+?\d{10,}",
    r"whatsapp\s+(us|me|at|on)",
    r"telegram\s+(us|me|at|on|group|channel)",
    r"visit\s+(this|the|that)\s+(link|url|website|site)",
    r"(click|open)\s+(this|the|that)\s+(link|url)",
    r"contact\s+(a|any)\s+third\s+party",
    r"unofficial\s+(number|line|channel|link|hotline|support)",
    r"external\s+(link|url|website|number|support|channel)",
    r"reach\s+out\s+to\s+(this|that)\s+(number|link)",
    r"message\s+(us|me)\s+on\s+(whatsapp|telegram|facebook|messenger)",
]

_COMPILED_THIRD_PARTY = [re.compile(p, re.IGNORECASE) for p in THIRD_PARTY_PATTERNS]

SAFE_THIRD_PARTY_REPLACEMENT = (
    "For your security, please contact us only through our official in-app support "
    "channel or call center. Do not use unofficial numbers, links, or third-party "
    "contacts."
)


def filter_third_party_instructions(text: str) -> tuple[str, bool]:
    """Scan text for instructions to contact unofficial third parties.

    Returns:
        Tuple of (safe_text, violation_detected).
    """
    for pattern in _COMPILED_THIRD_PARTY:
        if pattern.search(text):
            return SAFE_THIRD_PARTY_REPLACEMENT, True
    return text, False


# ── 5d. Schema Validator ────────────────────────────────────────────────────

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
    third_party_violation: bool = False
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

    # 5c. Third-party contact filter on customer_reply
    if "customer_reply" in result.data:
        safe_reply, tp_violation = filter_third_party_instructions(
            result.data["customer_reply"]
        )
        if tp_violation:
            result.third_party_violation = True
            result.data["customer_reply"] = safe_reply

    # 5d. Schema validation
    valid_tx_ids = {tx.transaction_id for tx in transactions}
    validation = validate_response_schema(
        result.data, request_ticket_id, valid_tx_ids
    )
    result.schema_errors = validation.errors
    if validation.fixed_data:
        result.data = validation.fixed_data

    return result
