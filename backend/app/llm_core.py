"""
Layer 4 — AI Reasoning Core (Gemini API).

Single LLM call with structured prompt, JSON parsing,
retry logic, and safe fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from google import genai
from google.genai import types

from app.preprocessing import PreprocessedInput
from app.prompts import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger("queuestorm.llm")

# ── Client setup ─────────────────────────────────────────────────────────────

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazy-init the Gemini client."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        _client = genai.Client(api_key=api_key)
    return _client


MODEL_NAME = "gemini-2.5-flash"


# ── JSON extraction ──────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response text.

    Handles raw JSON, code-fenced JSON, and JSON embedded in text.
    Uses balanced brace matching for robust extraction.
    """
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Use balanced brace matching to find the JSON object
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

    # Fallback: try rfind approach
    brace_end = text.rfind("}")
    if start != -1 and brace_end != -1 and brace_end > start:
        try:
            return json.loads(text[start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response: {text[:200]}...")


# ── Safe fallback ────────────────────────────────────────────────────────────

def _build_fallback_response(ticket_id: str) -> dict[str, Any]:
    """Return a safe default response when the LLM completely fails."""
    return {
        "ticket_id": ticket_id,
        "relevant_transaction_id": None,
        "evidence_verdict": "insufficient_data",
        "case_type": "other",
        "severity": "medium",
        "department": "customer_support",
        "agent_summary": (
            "Automated analysis was unable to process this ticket. "
            "Manual investigation required."
        ),
        "recommended_next_action": "Escalate to supervisor for manual review.",
        "customer_reply": (
            "Thank you for contacting us. Your complaint has been received and "
            "is being reviewed by our team. We will update you through official "
            "channels shortly."
        ),
        "human_review_required": True,
        "confidence": 0.3,
        "reason_codes": ["analysis_failed", "manual_review_needed"],
    }


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


async def analyze_with_llm(ctx: PreprocessedInput) -> dict[str, Any]:
    """Call Gemini to analyze a preprocessed ticket.

    Includes one retry on JSON parse failure, and a safe fallback
    if both attempts fail.

    Args:
        ctx: Pre-processed ticket data from Layer 3.

    Returns:
        Parsed JSON dict from the LLM response.
    """
    client = _get_client()

    # Build user message
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

    # Attempt 1
    try:
        response = await _call_gemini(client, user_message)
        return _extract_json(response)
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning("LLM attempt 1 failed to produce valid JSON: %s", e)

    # Attempt 2 — explicit correction prompt
    try:
        correction_message = (
            f"{user_message}\n\n"
            "IMPORTANT: Your previous response was not valid JSON. "
            "Respond with ONLY a valid JSON object. No text before or after."
        )
        response = await _call_gemini(client, correction_message)
        return _extract_json(response)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error("LLM attempt 2 also failed: %s", e)

    # Total failure — return safe fallback
    logger.error("Both LLM attempts failed. Using fallback response.")
    return _build_fallback_response(ctx.ticket_id)


async def _call_gemini(client: genai.Client, user_message: str) -> str:
    """Make a single Gemini API call and return the text response.

    Filters out thinking/reasoning parts from Gemini 2.5 models,
    returning only the actual output text.
    """
    try:
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=1500,
                temperature=0.2,
            ),
        )

        # Extract only non-thinking text parts from the response
        text_parts: list[str] = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # Skip thinking/reasoning parts
                        if hasattr(part, "thought") and part.thought:
                            continue
                        if part.text:
                            text_parts.append(part.text)

        if text_parts:
            return "\n".join(text_parts)

        # Fallback: try response.text (may include thinking)
        if response.text:
            return response.text

        raise ValueError("Gemini returned empty response")

    except Exception as e:
        logger.error("Gemini API call failed: %s", e)
        raise
