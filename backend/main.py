"""
QueueStorm Investigator — API Gateway (Layer 1).

FastAPI application with:
  - GET /health → {"status": "ok"}
  - POST /analyze-ticket → structured investigation response
  - Global exception handling
"""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.orchestrator import finalize_response, process_ticket
from app.preprocessing import preprocess
from app.rule_based import rule_based_analyze
from app.schemas import TicketRequest, TicketResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("queuestorm.api")

# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="QueueStorm Investigator",
    description=(
        "AI-powered internal support copilot for digital finance platforms. "
        "Investigates customer complaints by cross-referencing transaction history."
    ),
    version="1.0.0",
)

# CORS middleware (for development/testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return clean 500 responses.

    Never exposes stack traces, API keys, or internal details.
    """
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again.",
        },
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint. Returns immediately with no logic."""
    return {"status": "ok"}


@app.post("/analyze-ticket", response_model=TicketResponse)
async def analyze_ticket(request: TicketRequest) -> TicketResponse:
    """Analyze a customer complaint ticket.

    Accepts one ticket with complaint text and optional transaction history.
    Returns a structured investigation result with classification, routing,
    evidence analysis, and safe customer reply.

    Responds within 30 seconds.
    """
    start_time = time.time()
    logger.info("Received ticket: %s", request.ticket_id)

    try:
        response = await process_ticket(request)

        elapsed = time.time() - start_time
        logger.info(
            "Processed ticket %s in %.2fs | case=%s severity=%s verdict=%s",
            request.ticket_id,
            elapsed,
            response.case_type,
            response.severity,
            response.evidence_verdict,
        )

        return response

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            "Failed to process ticket %s after %.2fs: %s",
            request.ticket_id,
            elapsed,
            e,
            exc_info=True,
        )

        # Last-resort: rule-based fallback instead of generic error response
        ctx = preprocess(request)
        fallback_data = rule_based_analyze(ctx)
        response = finalize_response(fallback_data, ctx)
        logger.info(
            "Used rule-based fallback for ticket %s | case=%s",
            request.ticket_id,
            response.case_type,
        )
        return response
