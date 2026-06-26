"""
Layer 4R — Deterministic Evidence Routing Engine (Rule-Based).

This module is the AI-independent backbone of the investigator. It reverse-
engineers the matching/routing rules from the 10 public sample cases and
produces a COMPLETE, schema-valid, safety-compliant response WITHOUT any LLM
call. It is authoritative for every automatically-scored structured field:

    relevant_transaction_id, evidence_verdict, case_type,
    severity, department, human_review_required

The LLM (when available and not rate-limited) is used only to polish the three
natural-language fields. When the LLM is down — e.g. the Gemini key is blocked
for 60s after too many calls — this engine still answers correctly.

Decision matrices below are derived directly from SUST_Preli_Sample_Cases.json.
Each rule is annotated with the sample case(s) that motivate it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from app.preprocessing import PreprocessedInput
from app.schemas import TransactionEntry


# ════════════════════════════════════════════════════════════════════════════
# 0. Numeric / temporal primitives
# ════════════════════════════════════════════════════════════════════════════

_BN_DIGITS = {
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
}
_BN_TRANS = {ord(k): v for k, v in _BN_DIGITS.items()}


def normalize_digits(text: str) -> str:
    """Convert Bangla numerals to ASCII so matching works on bn/mixed input."""
    return text.translate(_BN_TRANS)


def _amount_forms(amount: float) -> list[str]:
    """String forms an integer-valued amount may appear as in a complaint."""
    forms: list[str] = []
    if amount == int(amount):
        ai = int(amount)
        forms.append(str(ai))
        # Comma-grouped, e.g. 15000 -> "15,000"
        forms.append(f"{ai:,}")
    else:
        forms.append(f"{amount:g}")
    return list(dict.fromkeys(forms))


def amount_present_in_text(amount: float, normalized_text: str) -> bool:
    """True if `amount` appears as a standalone number token in the text.

    Uses digit boundaries so a 5000 amount does NOT match inside an 11-digit
    phone number, and 1000 does not match inside 11000.
    """
    for form in _amount_forms(amount):
        escaped = re.escape(form)
        # Not preceded/followed by a digit; commas inside `form` are literal.
        pattern = rf"(?<![\d]){escaped}(?![\d])"
        if re.search(pattern, normalized_text):
            return True
    return False


_CURRENCY = r"(?:taka|tk|bdt|৳|টাকা)"
_AMOUNT_NEAR_CURRENCY = re.compile(
    rf"(?:{_CURRENCY}\s*([\d][\d,]*(?:\.\d+)?))"
    rf"|(?:([\d][\d,]*(?:\.\d+)?)\s*{_CURRENCY})",
    re.IGNORECASE,
)


def extract_primary_amount(normalized_text: str) -> Optional[float]:
    """Best-effort extraction of the headline amount (currency-tagged)."""
    for m in _AMOUNT_NEAR_CURRENCY.finditer(normalized_text):
        raw = m.group(1) or m.group(2)
        try:
            val = float(raw.replace(",", ""))
            if val > 0:
                return val
        except ValueError:
            continue
    return None


def _parse_ts(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _seconds_between(a: str, b: str) -> Optional[float]:
    da, db = _parse_ts(a), _parse_ts(b)
    if da is None or db is None:
        return None
    if da.tzinfo is None:
        da = da.replace(tzinfo=timezone.utc)
    if db.tzinfo is None:
        db = db.replace(tzinfo=timezone.utc)
    return abs((da - db).total_seconds())


# ════════════════════════════════════════════════════════════════════════════
# 1. Keyword matrices (English + Bangla + common Banglish)
# ════════════════════════════════════════════════════════════════════════════

# Phishing requires a "credential" token AND a "context" token (so that a normal
# complaint that merely promises "I won't share my OTP" is not mis-flagged),
# OR a strong standalone scam phrase. (SAMPLE-05)
_PHISH_CREDENTIAL = [
    "otp", "pin", "password", "passcode", "verification code", "security code",
    "ওটিপি", "পিন", "পাসওয়ার্ড",
]
_PHISH_CONTEXT = [
    "asked for", "asked me", "share", "give", "told me to", "called me", "call",
    "sms", "message", "link", "click", "blocked", "block my", "suspicious",
    "scam", "fraud", "stranger", "won", "prize", "lottery", "claiming",
    "pretend", "is this real", "from bkash", "from nagad", "from the bank",
    "চাইছে", "চেয়েছে", "ফোন", "ম্যাসেজ", "লিংক", "প্রতারণা", "ব্লক",
]
_PHISH_STRONG = [
    "phishing", "social engineering", "scam call", "scam message",
    "asked for my otp", "asked for my pin", "asked for my password",
    "asked for the otp", "asking for otp", "asking for my otp",
    "asked for otp", "প্রতারণা", "জালিয়াতি",
    # Account compromise / unauthorized access — the fraud_risk bucket. None of
    # the 10 public samples contain these, so they do not affect those cases.
    "hacked", "account was hacked", "my account was hacked",
    "unauthorized", "unauthorised", "unauthorized transaction",
    "didn't authorize", "did not authorize", "i didn't authorize",
    "without my permission", "without my consent",
    "account compromised", "account was compromised", "compromised my account",
    "money was stolen", "stolen from my account", "someone stole",
    "someone accessed my account", "someone logged into my account",
    "fraudulent",
    "হ্যাক", "অ্যাকাউন্ট হ্যাক", "অনুমতি ছাড়া", "চুরি",
]

_DUPLICATE_KW = [
    "twice", "two times", "double", "duplicate", "deducted twice",
    "charged twice", "charged two times", "paid once", "only paid once",
    "double charge", "double charged", "two times", "second time",
    "দুইবার", "দুই বার", "ডাবল", "দুবার", "একবারই", "দুটো",
]

_AGENT_KW = [
    "cash in", "cash-in", "cashin", "agent", "deposit", "deposited",
    "ক্যাশ ইন", "ক্যাশইন", "এজেন্ট", "জমা", "টাকা জমা",
]

_SETTLEMENT_KW = [
    "settle", "settled", "settlement", "payout", "pay out", "disburse",
    "not been settled", "sales", "my sale",
    "সেটেলমেন্ট", "নিষ্পত্তি", "বিক্রি",
]

_FAILED_KW = [
    "failed", "fail", "unsuccessful", "did not go through", "didn't go through",
    "declined", "error", "ব্যর্থ", "হয়নি", "ব্যর্থ হয়েছে", "ফেল",
]
_DEDUCTED_KW = [
    "deducted", "deduct", "cut", "balance was deducted", "balance deducted",
    "money was taken", "taken from", "কাটা", "কেটে নিয়েছে", "কেটে গেছে",
]

_WRONG_KW = [
    "wrong number", "wrong person", "wrong recipient", "wrong account",
    "to the wrong", "sent to wrong", "sent it to the wrong", "typed it wrong",
    "by mistake", "mistakenly",
    "didn't get it", "did not get it", "didn't receive", "did not receive",
    "hasn't received", "has not received", "not received it", "never received",
    "ভুল নম্বর", "ভুল মানুষ", "ভুলে", "পায়নি", "পাইনি", "পাননি",
]

_REFUND_KW = [
    "refund", "money back", "return my money", "return the money",
    "changed my mind", "change my mind", "don't want", "do not want",
    "cancel", "no longer want",
    "ফেরত", "রিফান্ড", "টাকা ফেরত", "বাতিল",
]


def _contains_any(haystack: str, needles: list[str]) -> bool:
    return any(n in haystack for n in needles)


# ════════════════════════════════════════════════════════════════════════════
# 2. Duplicate-pair detection (structural)
# ════════════════════════════════════════════════════════════════════════════

DUPLICATE_WINDOW_SECONDS = 300.0  # SAMPLE-10: 12s apart. Generalize to 5 min.


def find_duplicate_pair(
    txns: list[TransactionEntry],
) -> Optional[tuple[str, str]]:
    """Find two near-identical charges (same amount, counterparty, type) that
    occurred within DUPLICATE_WINDOW_SECONDS of each other.

    Returns (later_txn_id, earlier_txn_id) — the later one is the suspected
    duplicate to flag as relevant. (SAMPLE-10)
    """
    n = len(txns)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = txns[i], txns[j]
            if a.amount != b.amount:
                continue
            if a.counterparty != b.counterparty:
                continue
            if a.type != b.type:
                continue
            if a.status.value == "failed" or b.status.value == "failed":
                continue
            delta = _seconds_between(a.timestamp, b.timestamp)
            if delta is None or delta > DUPLICATE_WINDOW_SECONDS:
                continue
            ta, tb = _parse_ts(a.timestamp), _parse_ts(b.timestamp)
            if ta is not None and tb is not None and tb < ta:
                later, earlier = a, b
            else:
                later, earlier = b, a
            return later.transaction_id, earlier.transaction_id
    return None


# ════════════════════════════════════════════════════════════════════════════
# 3. Case-type classification
# ════════════════════════════════════════════════════════════════════════════

def classify_case_type(
    complaint: str,
    txns: list[TransactionEntry],
    user_type: Optional[str],
    channel: Optional[str],
    has_duplicate_pair: bool,
) -> str:
    """Deterministically pick exactly one case_type.

    Priority order matters: a failed+deducted complaint that also says "refund"
    must be payment_failed, not refund_request, etc. (SAMPLE-03 vs SAMPLE-04).
    """
    text = normalize_digits(complaint).lower()

    # Classification is TEXT-FIRST. Transaction-history types never override a
    # clear complaint signal — otherwise a wrong-transfer complaint whose
    # history merely contains a cash_in entry would be mis-routed (SAMPLE-01),
    # and a vague complaint with a cash_in entry would be mis-routed too
    # (SAMPLE-06). History is used later only to find the relevant transaction.

    # 1. Phishing / social engineering — highest priority (safety-critical).
    has_cred = _contains_any(text, _PHISH_CREDENTIAL)
    has_ctx = _contains_any(text, _PHISH_CONTEXT)
    if _contains_any(text, _PHISH_STRONG) or (has_cred and has_ctx):
        return "phishing_or_social_engineering"

    # 2. Duplicate payment — explicit wording OR a structural duplicate pair
    #    (two identical charges to the same counterparty within ~5 minutes).
    if _contains_any(text, _DUPLICATE_KW) or has_duplicate_pair:
        return "duplicate_payment"

    # 3. Merchant settlement delay — explicit settlement wording. (Bare
    #    "merchant" is intentionally NOT a trigger: SAMPLE-04 is a refund that
    #    merely mentions paying "a merchant".)
    if _contains_any(text, _SETTLEMENT_KW):
        return "merchant_settlement_delay"

    # 4. Agent cash-in issue — agent / cash-in wording in the complaint.
    if _contains_any(text, _AGENT_KW):
        return "agent_cash_in_issue"

    # 5. Payment failed — the complaint itself reports a failure or a
    #    deduction on a failed transaction.
    if _contains_any(text, _FAILED_KW) and (
        _contains_any(text, _FAILED_KW) or _contains_any(text, _DEDUCTED_KW)
    ):
        return "payment_failed"

    # 6. Wrong transfer.
    if _contains_any(text, _WRONG_KW):
        return "wrong_transfer"

    # 7. Refund request (no failure, just wants money back / changed mind).
    if _contains_any(text, _REFUND_KW):
        return "refund_request"

    # 7b. Merchant-context fallback: a merchant/portal complaint with no
    #     stronger signal is most likely a settlement query.
    if user_type == "merchant" or channel == "merchant_portal":
        return "merchant_settlement_delay"

    # 8. Fallback.
    return "other"


# ════════════════════════════════════════════════════════════════════════════
# 4. Transaction matching + evidence verdict
# ════════════════════════════════════════════════════════════════════════════

class Decision:
    """Container for the core evidence decision."""

    def __init__(self) -> None:
        self.relevant_id: Optional[str] = None
        self.verdict: str = "insufficient_data"
        self.matched_amount: Optional[float] = None
        self.signals: list[str] = []


def _amount_candidates(
    txns: list[TransactionEntry], normalized_text: str
) -> list[TransactionEntry]:
    return [t for t in txns if amount_present_in_text(t.amount, normalized_text)]


def _counterparty_count(txns: list[TransactionEntry], counterparty: str) -> int:
    return sum(1 for t in txns if t.counterparty == counterparty)


def decide_evidence(
    case_type: str,
    complaint: str,
    txns: list[TransactionEntry],
    duplicate_pair: Optional[tuple[str, str]],
) -> Decision:
    """Isolate relevant_transaction_id and the evidence_verdict.

    Implements the per-case matching logic reverse-engineered from the samples.
    """
    d = Decision()
    text = normalize_digits(complaint).lower()

    # No history at all → cannot verify from data. (SAMPLE-05)
    if not txns:
        d.verdict = "insufficient_data"
        return d

    # ── Phishing: evidence is about a call/SMS, not a transaction. ──────────
    if case_type == "phishing_or_social_engineering":
        d.relevant_id = None
        d.verdict = "insufficient_data"
        return d

    candidates = _amount_candidates(txns, text)

    # ── Duplicate payment: flag the later of the duplicate pair. (SAMPLE-10) ─
    if case_type == "duplicate_payment":
        if duplicate_pair is not None:
            d.relevant_id = duplicate_pair[0]
            d.verdict = "consistent"
            tx = next((t for t in txns if t.transaction_id == d.relevant_id), None)
            if tx:
                d.matched_amount = tx.amount
            d.signals.append("duplicate_pair_detected")
            return d
        # Claimed duplicate but no second charge exists → contradicted.
        if candidates:
            # Exactly one charge of that amount → no duplicate in data.
            d.relevant_id = candidates[-1].transaction_id
            d.matched_amount = candidates[-1].amount
            d.verdict = "inconsistent"
            d.signals.append("single_charge_only")
            return d
        d.verdict = "insufficient_data"
        return d

    # ── Merchant settlement delay. (SAMPLE-09) ─────────────────────────────
    if case_type == "merchant_settlement_delay":
        settle = [t for t in txns if t.type.value == "settlement"]
        pool = settle or candidates or txns
        tx = _pick_by_amount(pool, candidates)
        if tx is not None:
            d.relevant_id = tx.transaction_id
            d.matched_amount = tx.amount
            d.verdict = (
                "consistent" if tx.status.value in ("pending", "failed")
                else "inconsistent" if tx.status.value == "completed"
                else "consistent"
            )
        else:
            d.verdict = "insufficient_data"
        return d

    # ── Agent cash-in issue. (SAMPLE-07) ───────────────────────────────────
    if case_type == "agent_cash_in_issue":
        cashins = [t for t in txns if t.type.value == "cash_in"]
        pool = cashins or candidates or txns
        tx = _pick_by_amount(pool, candidates)
        if tx is not None:
            d.relevant_id = tx.transaction_id
            d.matched_amount = tx.amount
            # pending/failed supports "money didn't arrive"; completed contradicts.
            if tx.status.value in ("pending", "failed"):
                d.verdict = "consistent"
            elif tx.status.value == "completed":
                d.verdict = "inconsistent"
            else:
                d.verdict = "consistent"
        else:
            d.verdict = "insufficient_data"
        return d

    # ── Payment failed (balance deducted). (SAMPLE-03) ─────────────────────
    if case_type == "payment_failed":
        failed = [t for t in candidates if t.status.value == "failed"]
        pool_failed = failed or [t for t in txns if t.status.value == "failed"]
        if pool_failed:
            tx = _pick_by_amount(pool_failed, candidates)
            d.relevant_id = tx.transaction_id
            d.matched_amount = tx.amount
            d.verdict = "consistent"  # status 'failed' supports the claim
            return d
        if candidates:
            # Claimed failed but the matching transaction completed → contradiction.
            tx = candidates[-1]
            d.relevant_id = tx.transaction_id
            d.matched_amount = tx.amount
            d.verdict = "inconsistent"
            return d
        d.verdict = "insufficient_data"
        return d

    # ── Refund request. (SAMPLE-04) ────────────────────────────────────────
    if case_type == "refund_request":
        completed = [t for t in candidates if t.status.value == "completed"]
        pool = completed or candidates
        if pool:
            tx = _pick_by_amount(pool, candidates)
            d.relevant_id = tx.transaction_id
            d.matched_amount = tx.amount
            d.verdict = "consistent"
            return d
        d.verdict = "insufficient_data"
        return d

    # ── Wrong transfer. (SAMPLE-01 / 02 / 08) ──────────────────────────────
    if case_type == "wrong_transfer":
        if not candidates:
            d.verdict = "insufficient_data"
            return d

        # Distinct counterparties among completed candidates → ambiguous.
        completed = [t for t in candidates if t.status.value == "completed"]
        distinct_cps = {t.counterparty for t in (completed or candidates)}

        if len(candidates) > 1 and len(distinct_cps) > 1:
            # Multiple plausible recipients → do not guess. (SAMPLE-08)
            d.relevant_id = None
            d.verdict = "insufficient_data"
            d.signals.append("ambiguous_match")
            return d

        # Single recipient (one candidate, or several to the same counterparty):
        tx = _pick_by_amount(completed or candidates, candidates)
        d.relevant_id = tx.transaction_id
        d.matched_amount = tx.amount

        # Established-recipient pattern contradicts a "wrong recipient" claim.
        if _counterparty_count(txns, tx.counterparty) >= 2:  # (SAMPLE-02)
            d.verdict = "inconsistent"
            d.signals.append("established_recipient_pattern")
        else:  # (SAMPLE-01)
            d.verdict = "consistent"
        return d

    # ── Other / vague. (SAMPLE-06) ─────────────────────────────────────────
    if candidates and len(candidates) == 1:
        d.relevant_id = candidates[0].transaction_id
        d.matched_amount = candidates[0].amount
        d.verdict = "consistent"
        return d
    d.verdict = "insufficient_data"
    return d


def _pick_by_amount(
    pool: list[TransactionEntry], candidates: list[TransactionEntry]
) -> Optional[TransactionEntry]:
    """Prefer an amount-matched transaction; among ties pick the most recent."""
    if not pool:
        return None
    cand_ids = {t.transaction_id for t in candidates}
    preferred = [t for t in pool if t.transaction_id in cand_ids] or pool

    def _key(t: TransactionEntry):
        ts = _parse_ts(t.timestamp)
        return ts.timestamp() if ts else 0.0

    return max(preferred, key=_key)


# ════════════════════════════════════════════════════════════════════════════
# 5. Severity / department / human-review matrices
# ════════════════════════════════════════════════════════════════════════════

def derive_severity(case_type: str, verdict: str, amount: Optional[float]) -> str:
    """Severity matrix derived from the samples (case_type-driven)."""
    if case_type == "phishing_or_social_engineering":
        return "critical"
    if case_type == "wrong_transfer":
        return "high" if verdict == "consistent" else "medium"
    if case_type == "payment_failed":
        return "high"
    if case_type == "duplicate_payment":
        return "high"
    if case_type == "agent_cash_in_issue":
        return "high" if verdict in ("consistent", "inconsistent") else "medium"
    if case_type == "merchant_settlement_delay":
        return "medium"
    if case_type == "refund_request":
        # Change-of-mind refunds are low; large amounts nudge to medium.
        if amount is not None and amount > 10000:
            return "medium"
        return "low"
    return "low"  # other / vague


def derive_department(case_type: str, verdict: str) -> str:
    mapping = {
        "wrong_transfer": "dispute_resolution",
        "payment_failed": "payments_ops",
        "duplicate_payment": "payments_ops",
        "merchant_settlement_delay": "merchant_operations",
        "agent_cash_in_issue": "agent_operations",
        "phishing_or_social_engineering": "fraud_risk",
        "other": "customer_support",
    }
    if case_type == "refund_request":
        # Contested refund (evidence not clean) → dispute_resolution.
        return "dispute_resolution" if verdict == "inconsistent" else "customer_support"
    return mapping.get(case_type, "customer_support")


def derive_human_review(
    case_type: str, verdict: str, severity: str, relevant_id: Optional[str]
) -> bool:
    """Escalation matrix. Verified against all 10 samples.

    True  : phishing, duplicate_payment, any inconsistent verdict, an
            identified wrong_transfer / agent_cash_in dispute, critical severity.
    False : payment_failed, low refund, settlement delay, vague, and
            *ambiguous* wrong_transfer with no identified transaction.
    """
    if case_type in ("phishing_or_social_engineering", "duplicate_payment"):
        return True
    if verdict == "inconsistent":
        return True
    if case_type in ("wrong_transfer", "agent_cash_in_issue") and relevant_id:
        return True
    if severity == "critical":
        return True
    return False


def derive_confidence(verdict: str, case_type: str, relevant_id: Optional[str]) -> float:
    if case_type == "phishing_or_social_engineering":
        return 0.95
    if verdict == "consistent":
        return 0.9
    if verdict == "inconsistent":
        return 0.75
    # insufficient_data
    if relevant_id is None and case_type in ("wrong_transfer", "other"):
        return 0.6
    return 0.6


def build_reason_codes(
    case_type: str, verdict: str, decision: Decision, injection: bool
) -> list[str]:
    codes: list[str] = [case_type]
    if verdict == "consistent":
        codes.append("transaction_match" if decision.relevant_id else "evidence_consistent")
    elif verdict == "inconsistent":
        codes.append("evidence_inconsistent")
    else:
        codes.append("insufficient_data" if decision.relevant_id is None else "evidence_unclear")
    for s in decision.signals:
        if s not in codes:
            codes.append(s)
    if injection:
        codes.append("injection_attempt")
    # De-dup preserving order, cap 8.
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:8]


# ════════════════════════════════════════════════════════════════════════════
# 6. Templated, safety-compliant natural language
# ════════════════════════════════════════════════════════════════════════════

_SAFE_PIN_EN = "Please do not share your PIN or OTP with anyone."
_SAFE_PIN_BN = "অনুগ্রহ করে কারো সাথে আপনার পিন বা ওটিপি শেয়ার করবেন না।"
_ELIGIBLE_EN = (
    "any eligible amount will be returned through official channels"
)


def _tid(relevant_id: Optional[str]) -> str:
    return relevant_id if relevant_id else "the reported transaction"


def build_texts(
    case_type: str,
    verdict: str,
    decision: Decision,
    language: str,
    user_type: Optional[str],
) -> dict[str, str]:
    """Return agent_summary, recommended_next_action, customer_reply.

    agent_summary and recommended_next_action are always English (agent-facing).
    customer_reply is Bangla when language == 'bn' (matches SAMPLE-07).
    """
    tid = _tid(decision.relevant_id)
    amt = (
        f"{int(decision.matched_amount)} BDT"
        if decision.matched_amount is not None
        and decision.matched_amount == int(decision.matched_amount)
        else (f"{decision.matched_amount} BDT" if decision.matched_amount else "the stated amount")
    )
    bn = language == "bn"

    # ── agent_summary + recommended_next_action (English) ──────────────────
    summary, action = _agent_text(case_type, verdict, tid, amt, decision)

    # ── customer_reply ─────────────────────────────────────────────────────
    reply = _customer_reply(case_type, tid, bn)

    return {
        "agent_summary": summary,
        "recommended_next_action": action,
        "customer_reply": reply,
    }


def _agent_text(
    case_type: str, verdict: str, tid: str, amt: str, decision: Decision
) -> tuple[str, str]:
    if case_type == "wrong_transfer":
        if verdict == "inconsistent":
            return (
                f"Customer claims {tid} ({amt}) was a wrong transfer, but the "
                f"transaction history shows repeated transfers to the same "
                f"counterparty, suggesting an established recipient.",
                "Flag for human review. Verify with the customer whether this "
                "was genuinely a wrong transfer given the established pattern "
                "with this recipient before initiating any dispute.",
            )
        if decision.relevant_id is None:
            return (
                "Customer reports a transfer was not received, but multiple "
                "transactions of the stated amount exist to different "
                "recipients. The specific transaction cannot be identified "
                "without more detail.",
                "Reply to the customer asking for the recipient's number to "
                "identify the correct transaction. Do not initiate a dispute "
                "until the transaction is confirmed.",
            )
        return (
            f"Customer reports sending {amt} via {tid} to a recipient they now "
            f"believe was wrong. Evidence in the history supports the claim.",
            f"Verify {tid} details with the customer and initiate the "
            f"wrong-transfer dispute workflow per policy.",
        )

    if case_type == "payment_failed":
        if verdict == "inconsistent":
            return (
                f"Customer reports {tid} ({amt}) failed with a balance "
                f"deduction, but the transaction status in the history is "
                f"completed.",
                f"Verify {tid} ledger status with payments operations and "
                f"confirm whether the service was delivered before any action.",
            )
        return (
            f"Customer attempted a payment ({tid}, {amt}) which failed but "
            f"reports the balance was deducted. Requires payments operations "
            f"investigation.",
            f"Investigate the {tid} ledger status. If the balance was deducted "
            f"on a failed payment, initiate the automatic reversal flow within "
            f"standard SLA.",
        )

    if case_type == "duplicate_payment":
        return (
            f"Customer reports a duplicate charge. Two identical payments were "
            f"recorded close together; {tid} is the suspected duplicate.",
            f"Verify the duplicate with payments operations. If the biller "
            f"confirms only one payment was received, initiate reversal of {tid}.",
        )

    if case_type == "merchant_settlement_delay":
        return (
            f"Merchant reports settlement {tid} ({amt}) is delayed beyond the "
            f"expected window. Settlement status is pending.",
            "Route to merchant operations to verify the settlement batch "
            "status. If the batch is delayed, communicate a revised ETA to the "
            "merchant.",
        )

    if case_type == "agent_cash_in_issue":
        if verdict == "inconsistent":
            return (
                f"Customer reports an agent cash-in ({tid}, {amt}) not reflected "
                f"in balance, but the transaction status is completed.",
                f"Reconcile {tid} with agent operations and confirm the balance "
                f"credit before responding to the customer.",
            )
        return (
            f"Customer reports {amt} cash-in via agent ({tid}) not reflected in "
            f"balance. Transaction status is pending/unsettled.",
            f"Investigate the {tid} status with agent operations. Confirm the "
            f"settlement state and resolve within the standard cash-in SLA.",
        )

    if case_type == "refund_request":
        return (
            f"Customer requests a refund for {tid} ({amt}). This appears to be a "
            f"change-of-mind request rather than a service failure.",
            "Inform the customer that refund eligibility depends on the "
            "merchant's own policy and guide them on contacting the merchant "
            "directly. Do not promise a refund.",
        )

    if case_type == "phishing_or_social_engineering":
        return (
            "Customer reports an unsolicited contact requesting credentials "
            "(PIN/OTP). Likely a social engineering attempt; credentials may "
            "not yet have been shared.",
            "Escalate to the fraud_risk team immediately. Confirm to the "
            "customer that the company never asks for OTP, and log the reported "
            "contact for fraud pattern analysis.",
        )

    # other / vague
    return (
        "Customer reports a vague concern without specifying a transaction, "
        "amount, or issue. Insufficient detail to identify any relevant "
        "transaction.",
        "Reply to the customer asking for specific details: which transaction, "
        "what amount, what went wrong, and the approximate time.",
    )


def _customer_reply(case_type: str, tid: str, bn: bool) -> str:
    if case_type == "phishing_or_social_engineering":
        if bn:
            return (
                "কোনো তথ্য শেয়ার করার আগে আমাদের জানানোর জন্য ধন্যবাদ। আমরা "
                "কখনোই আপনার পিন, ওটিপি বা পাসওয়ার্ড চাই না। কেউ নিজেকে আমাদের "
                "প্রতিনিধি দাবি করলেও এসব শেয়ার করবেন না। আমাদের ফ্রড টিমকে "
                "বিষয়টি জানানো হয়েছে।"
            )
        return (
            "Thank you for reaching out before sharing any information. We never "
            "ask for your PIN, OTP, or password under any circumstances. Please "
            "do not share these with anyone, even if they claim to be from us. "
            "Our fraud team has been notified of this incident."
        )

    if case_type == "refund_request":
        if bn:
            return (
                "যোগাযোগ করার জন্য ধন্যবাদ। সম্পন্ন মার্চেন্ট পেমেন্টের রিফান্ড "
                "সংশ্লিষ্ট মার্চেন্টের নিজস্ব নীতির উপর নির্ভর করে। আমরা সরাসরি "
                "মার্চেন্টের সাথে যোগাযোগের পরামর্শ দিচ্ছি। প্রয়োজনে আমরা সহায়তা "
                f"করব। {_SAFE_PIN_BN}"
            )
        return (
            "Thank you for reaching out. Refunds for completed merchant payments "
            "depend on the merchant's own policy. We recommend contacting the "
            "merchant directly, and we can guide you if needed. " + _SAFE_PIN_EN
        )

    if case_type == "other":
        if bn:
            return (
                "যোগাযোগ করার জন্য ধন্যবাদ। দ্রুত সহায়তার জন্য অনুগ্রহ করে লেনদেন "
                "আইডি, সংশ্লিষ্ট পরিমাণ এবং কী সমস্যা হয়েছে তা জানান। " + _SAFE_PIN_BN
            )
        return (
            "Thank you for reaching out. To help you faster, please share the "
            "transaction ID, the amount involved, and a short description of what "
            "went wrong. " + _SAFE_PIN_EN
        )

    if case_type in ("payment_failed", "duplicate_payment"):
        if bn:
            return (
                f"আপনার লেনদেন {tid} সম্পর্কে আমরা অবগত হয়েছি। আমাদের পেমেন্টস দল "
                f"বিষয়টি যাচাই করবে এবং যাচাই সাপেক্ষে যেকোনো প্রযোজ্য পরিমাণ "
                f"অফিসিয়াল চ্যানেলে ফেরত দেওয়া হবে। {_SAFE_PIN_BN}"
            )
        return (
            f"We have noted your concern about transaction {tid}. Our payments "
            f"team will review the case and {_ELIGIBLE_EN}. " + _SAFE_PIN_EN
        )

    if case_type == "merchant_settlement_delay":
        if bn:
            return (
                f"আপনার সেটেলমেন্ট {tid} সম্পর্কিত উদ্বেগ আমরা নোট করেছি। আমাদের "
                f"মার্চেন্ট অপারেশন্স দল ব্যাচ স্ট্যাটাস যাচাই করে অফিসিয়াল "
                f"চ্যানেলে আপনাকে আপডেট জানাবে।"
            )
        return (
            f"We have noted your concern about settlement {tid}. Our merchant "
            f"operations team will check the batch status and update you on the "
            f"expected settlement time through official channels."
        )

    if case_type == "agent_cash_in_issue":
        if bn:
            return (
                f"আপনার লেনদেন {tid} এর বিষয়ে আমরা অবগত হয়েছি। আমাদের এজেন্ট "
                f"অপারেশন্স দল এটি দ্রুত যাচাই করবে এবং অফিসিয়াল চ্যানেলে আপনাকে "
                f"জানাবে। {_SAFE_PIN_BN}"
            )
        return (
            f"We have noted your concern about transaction {tid}. Our agent "
            f"operations team will verify it promptly and update you through "
            f"official channels. " + _SAFE_PIN_EN
        )

    # wrong_transfer (and default)
    if bn:
        return (
            f"আপনার লেনদেন {tid} সম্পর্কিত উদ্বেগ আমরা নোট করেছি। আমাদের ডিসপিউট "
            f"দল বিষয়টি যত্ন সহকারে পর্যালোচনা করে অফিসিয়াল চ্যানেলে আপনার সাথে "
            f"যোগাযোগ করবে। {_SAFE_PIN_BN}"
        )
    return (
        f"We have noted your concern about transaction {tid}. Our dispute team "
        f"will review the case carefully and contact you through official "
        f"support channels. " + _SAFE_PIN_EN
    )


# ════════════════════════════════════════════════════════════════════════════
# 7. Public entry point
# ════════════════════════════════════════════════════════════════════════════

def run_rule_engine(ctx: PreprocessedInput) -> dict:
    """Produce a complete, schema-valid response from rules alone (no LLM)."""
    complaint = ctx.cleaned_complaint or ""
    txns = ctx.transactions or []

    duplicate_pair = find_duplicate_pair(txns)

    case_type = classify_case_type(
        complaint=complaint,
        txns=txns,
        user_type=ctx.user_type,
        channel=ctx.channel,
        has_duplicate_pair=duplicate_pair is not None,
    )

    decision = decide_evidence(case_type, complaint, txns, duplicate_pair)

    amount = decision.matched_amount
    if amount is None:
        amount = extract_primary_amount(normalize_digits(complaint))

    severity = derive_severity(case_type, decision.verdict, amount)
    department = derive_department(case_type, decision.verdict)
    human_review = derive_human_review(
        case_type, decision.verdict, severity, decision.relevant_id
    )
    confidence = derive_confidence(decision.verdict, case_type, decision.relevant_id)
    reason_codes = build_reason_codes(
        case_type, decision.verdict, decision, ctx.injection_detected
    )

    texts = build_texts(
        case_type=case_type,
        verdict=decision.verdict,
        decision=decision,
        language=ctx.detected_language,
        user_type=ctx.user_type,
    )

    return {
        "ticket_id": ctx.ticket_id,
        "relevant_transaction_id": decision.relevant_id,
        "evidence_verdict": decision.verdict,
        "case_type": case_type,
        "severity": severity,
        "department": department,
        "agent_summary": texts["agent_summary"],
        "recommended_next_action": texts["recommended_next_action"],
        "customer_reply": texts["customer_reply"],
        "human_review_required": human_review,
        "confidence": confidence,
        "reason_codes": reason_codes,
    }
