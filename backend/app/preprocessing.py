"""
Layer 3 — Pre-Processing.

Three micro-steps that run on raw input before the LLM call:
  3a. Language Detector (Unicode range check)
  3b. Transaction Pre-Matcher (structured summary + derived signals)
  3c. Prompt Injection Scrubber (regex neutralization)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.schemas import TicketRequest, TransactionEntry


# ── 3a. Language Detector ────────────────────────────────────────────────────

_BANGLA_RANGE = re.compile(r"[\u0980-\u09FF]")


def detect_language(text: str) -> str:
    """Detect language from complaint text via Unicode range analysis.

    Returns one of: 'en', 'bn', 'mixed'.
    """
    if not text:
        return "en"

    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "en"

    bangla_count = len(_BANGLA_RANGE.findall(text))
    ratio = bangla_count / total_alpha

    if ratio > 0.8:
        return "bn"
    elif ratio > 0.2:
        return "mixed"
    return "en"


# ── 3b. Transaction Pre-Matcher ──────────────────────────────────────────────

@dataclass
class MatchSignals:
    """Derived signals from comparing complaint text to transaction history."""
    amount_matches: list[str] = field(default_factory=list)
    counterparty_matches: list[str] = field(default_factory=list)
    best_candidate_id: Optional[str] = None
    formatted_history: str = ""


def _extract_amounts_from_text(text: str) -> list[float]:
    """Extract numeric amounts from complaint text."""
    # Match patterns like "5000", "5,000", "5000.00", "৫০০০"
    # Also handles "taka", "tk", "BDT" suffixes
    patterns = [
        r"(\d[\d,]*\.?\d*)\s*(?:taka|tk|bdt|টাকা)?",
        r"(?:taka|tk|bdt|টাকা)\s*(\d[\d,]*\.?\d*)",
    ]
    amounts: list[float] = []
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            try:
                val = float(m.group(1).replace(",", ""))
                if val > 0:
                    amounts.append(val)
            except (ValueError, IndexError):
                continue
    return amounts


def _amounts_match(a: float, b: float, tolerance: float = 0.05) -> bool:
    """Check if two amounts match within ±tolerance (5% by default)."""
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(a, b) <= tolerance


def _format_transaction(tx: TransactionEntry) -> str:
    """Format a single transaction as a readable line."""
    try:
        ts = datetime.fromisoformat(tx.timestamp.replace("Z", "+00:00"))
        time_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        time_str = tx.timestamp

    return (
        f"  • [{tx.transaction_id}] {tx.type.value.upper()} | "
        f"Amount: {tx.amount:.2f} BDT | "
        f"Status: {tx.status.value} | "
        f"Counterparty: {tx.counterparty} | "
        f"Time: {time_str}"
    )


def prematch_transactions(
    complaint: str,
    transactions: list[TransactionEntry],
) -> MatchSignals:
    """Build a structured transaction summary and compute match signals."""
    signals = MatchSignals()

    if not transactions:
        signals.formatted_history = "  (No transaction history provided)"
        return signals

    # Format history
    lines = [_format_transaction(tx) for tx in transactions]
    signals.formatted_history = "\n".join(lines)

    # Extract amounts and counterparties from complaint
    complaint_amounts = _extract_amounts_from_text(complaint)
    complaint_lower = complaint.lower()

    # Score each transaction for relevance
    best_score = 0
    for tx in transactions:
        score = 0

        # Amount match
        for ca in complaint_amounts:
            if _amounts_match(ca, tx.amount):
                signals.amount_matches.append(tx.transaction_id)
                score += 2
                break

        # Counterparty match
        if tx.counterparty and tx.counterparty.lower() in complaint_lower:
            signals.counterparty_matches.append(tx.transaction_id)
            score += 2

        # Status-based relevance (failed/pending transactions are more likely complained about)
        if tx.status.value in ("failed", "pending"):
            score += 1

        if score > best_score:
            best_score = score
            signals.best_candidate_id = tx.transaction_id

    return signals


# ── 3c. Prompt Injection Scrubber ────────────────────────────────────────────

_INJECTION_PATTERNS = [
    # Role overrides
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(the\s+)?(above|previous|all)",
    r"you\s+are\s+now\s+",
    r"your\s+new\s+role\s+is",
    r"act\s+as\s+(a|an|if)\b",
    r"pretend\s+(you\s+are|to\s+be)",
    r"from\s+now\s+on\s+you\s+(are|will)",
    # System-level directives
    r"\[SYSTEM\]",
    r"<\s*instruction\s*>",
    r"<\s*system\s*>",
    r"<\s*/?\s*prompt\s*>",
    # Prompt extraction
    r"repeat\s+your\s+(system\s+)?instructions",
    r"(show|reveal|display|print)\s+(me\s+)?your\s+(system\s+)?prompt",
    r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions)",
    # Safety bypass attempts
    r"(the\s+)?safety\s+rules?\s+(have\s+been|are\s+now)\s+(updated|changed|removed)",
    r"you\s+can\s+now\s+(share|reveal|provide)\s+(pins?|otp|passwords?)",
    r"restrictions?\s+(have\s+been|are\s+now)\s+(lifted|removed)",
    # Data exfiltration
    r"list\s+all\s+(customer|user)\s+data",
    r"(show|reveal|display)\s+(all\s+)?(internal|system)\s+data",
]

_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

_REPLACEMENT = "[content removed]"


def scrub_injections(text: str) -> tuple[str, bool]:
    """Remove prompt injection attempts from complaint text.

    Returns:
        Tuple of (cleaned_text, was_injection_detected).
    """
    detected = False
    cleaned = text

    for pattern in _COMPILED_INJECTION:
        if pattern.search(cleaned):
            detected = True
            cleaned = pattern.sub(_REPLACEMENT, cleaned)

    return cleaned, detected


# ── Public interface ─────────────────────────────────────────────────────────

@dataclass
class PreprocessedInput:
    """Bundle of all pre-processed data for downstream layers."""
    ticket_id: str
    original_complaint: str
    cleaned_complaint: str
    detected_language: str
    injection_detected: bool
    match_signals: MatchSignals
    channel: Optional[str]
    user_type: Optional[str]
    campaign_context: Optional[str]
    transactions: list[TransactionEntry]


def preprocess(request: TicketRequest) -> PreprocessedInput:
    """Run all three pre-processing steps on a ticket request."""
    # 3a. Language detection
    detected_lang = detect_language(request.complaint)
    language = request.language.value if request.language else detected_lang

    # 3c. Injection scrubbing (run before pre-matching so signals use clean text)
    cleaned_complaint, injection_detected = scrub_injections(request.complaint)

    # 3b. Transaction pre-matching
    transactions = request.transaction_history or []
    match_signals = prematch_transactions(cleaned_complaint, transactions)

    return PreprocessedInput(
        ticket_id=request.ticket_id,
        original_complaint=request.complaint,
        cleaned_complaint=cleaned_complaint,
        detected_language=language,
        injection_detected=injection_detected,
        match_signals=match_signals,
        channel=request.channel.value if request.channel else None,
        user_type=request.user_type.value if request.user_type else None,
        campaign_context=request.campaign_context,
        transactions=transactions,
    )
