"""
Layer 4 — AI Reasoning Core (Gemini API).

Single LLM call with structured JSON output, rate-limit retry,
and rule-based fallback when the API is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from app.preprocessing import PreprocessedInput
from app.prompts import SYSTEM_PROMPT, build_user_message
from app.rule_based import rule_based_analyze

logger = logging.getLogger("queuestorm.llm")

# ── Client setup ─────────────────────────────────────────────────────────────

_client: genai.Client | None = None

MODEL_NAME = "gemini-2.5-flash"
MAX_API_RETRIES = 3
DEFAULT_RATE_LIMIT_DELAY = 26.0

# JSON schema for structured output — matches TicketResponse fields
RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ticket_id": {"type": "string"},
        "relevant_transaction_id": {"type": ["string", "null"]},
        "evidence_verdict": {
            "type": "string",
            "enum": ["consistent", "inconsistent", "insufficient_data"],
        },
        "case_type": {
            "type": "string",
            "enum": [
                "wrong_transfer", "payment_failed", "refund_request",
                "duplicate_payment", "merchant_settlement_delay",
                "agent_cash_in_issue", "phishing_or_social_engineering", "other",
            ],
        },
        "severity": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
        },
        "department": {
            "type": "string",
            "enum": [
                "customer_support", "dispute_resolution", "payments_ops",
                "merchant_operations", "agent_operations", "fraud_risk",
            ],
        },
        "agent_summary": {"type": "string"},
        "recommended_next_action": {"type": "string"},
        "customer_reply": {"type": "string"},
        "human_review_required": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason_codes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "ticket_id", "relevant_transaction_id", "evidence_verdict",
        "case_type", "severity", "department", "agent_summary",
        "recommended_next_action", "customer_reply", "human_review_required",
    ],
}


def _get_client() -> genai.Client:
    """Lazy-init the Gemini client."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        _client = genai.Client(api_key=api_key)
    return _client


# ── JSON extraction ──────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not extract valid JSON from LLM response: {text[:200]}...")


def _parse_retry_delay(error_msg: str) -> float:
    """Extract retry delay from a 429 error message."""
    match = re.search(r"retry in ([\d.]+)s", error_msg, re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 1.0, 30.0)
    return DEFAULT_RATE_LIMIT_DELAY


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a Gemini 429 rate-limit error."""
    if isinstance(exc, ClientError):
        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if code == 429:
            return True
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota exceeded" in msg


# ── Core LLM call ───────────────────────────────────────────────────────────

def _build_match_hints(ctx: PreprocessedInput) -> list[str]:
    """Build human-readable pre-match hints from signals."""
    hints: list[str] = []
    if ctx.match_signals.amount_matches:
        hints.append(
            f"Amount match found in transaction(s): "
            f"{', '.join(ctx.match_signals.amount_matches)}"
        )
    if ctx.match_signals.counterparty_matches:
        hints.append(
            f"Counterparty match found in transaction(s): "
            f"{', '.join(ctx.match_signals.counterparty_matches)}"
        )
    if ctx.match_signals.best_candidate_id:
        hints.append(
            f"Best matching transaction candidate: "
            f"{ctx.match_signals.best_candidate_id}"
        )
    if ctx.injection_detected:
        hints.append(
            "WARNING: Prompt injection patterns were detected and removed "
            "from the complaint text."
        )
    return hints


async def _call_gemini(client: genai.Client, user_message: str) -> str:
    """Make a single Gemini API call with structured JSON output."""
    response = await client.aio.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2000,
            temperature=0.2,
            response_mime_type="application/json",
            response_json_schema=RESPONSE_JSON_SCHEMA,
        ),
    )

    text_parts: list[str] = []
    if response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "thought") and part.thought:
                        continue
                    if part.text:
                        text_parts.append(part.text)

    if text_parts:
        return "\n".join(text_parts)

    if response.text:
        return response.text

    raise ValueError("Gemini returned empty response")


async def _call_gemini_with_retry(client: genai.Client, user_message: str) -> str:
    """Call Gemini with automatic retry on 429 rate-limit errors."""
    last_exc: Exception | None = None

    for attempt in range(MAX_API_RETRIES):
        try:
            return await _call_gemini(client, user_message)
        except Exception as e:
            last_exc = e
            if _is_rate_limit_error(e) and attempt < MAX_API_RETRIES - 1:
                delay = _parse_retry_delay(str(e))
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %.0fs...",
                    attempt + 1,
                    MAX_API_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            raise

    raise last_exc or ValueError("Gemini call failed after retries")


async def analyze_with_llm(ctx: PreprocessedInput) -> dict[str, Any]:
    """Call Gemini to analyze a preprocessed ticket.

    Falls back to rule-based analysis if the API is unavailable or
    returns invalid JSON after one correction retry.

    Args:
        ctx: Pre-processed ticket data from Layer 3.

    Returns:
        Parsed JSON dict matching the TicketResponse schema.
    """
    match_hints = _build_match_hints(ctx)
    user_message = build_user_message(
        ticket_id=ctx.ticket_id,
        cleaned_complaint=ctx.cleaned_complaint,
        formatted_history=ctx.match_signals.formatted_history,
        detected_language=ctx.detected_language,
        channel=ctx.channel,
        user_type=ctx.user_type,
        campaign_context=ctx.campaign_context,
        match_hints=match_hints if match_hints else None,
    )

    # Attempt LLM call (with 429 retry built in)
    try:
        client = _get_client()
    except RuntimeError as e:
        logger.warning("No API key configured: %s. Using rule-based fallback.", e)
        return rule_based_analyze(ctx)

    try:
        response = await _call_gemini_with_retry(client, user_message)
        return _extract_json(response)
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning("LLM attempt 1 failed to produce valid JSON: %s", e)
    except Exception as e:
        if _is_rate_limit_error(e):
            logger.error("Rate limit exhausted after retries: %s", e)
        else:
            logger.error("Gemini API call failed: %s", e)
        logger.info("Falling back to rule-based analysis for ticket %s", ctx.ticket_id)
        return rule_based_analyze(ctx)

    # Attempt 2 — JSON correction retry (single extra call)
    try:
        correction_message = (
            f"{user_message}\n\n"
            "IMPORTANT: Respond with ONLY a valid JSON object matching the schema. "
            "No text before or after."
        )
        response = await _call_gemini_with_retry(client, correction_message)
        return _extract_json(response)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error("LLM attempt 2 also failed: %s", e)
    except Exception as e:
        logger.error("LLM correction attempt failed: %s", e)

    logger.warning("Both LLM attempts failed. Using rule-based fallback.")
    return rule_based_analyze(ctx)
