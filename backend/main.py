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

# Ensure the backend directory is in the Python path for Vercel
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.orchestrator import process_ticket
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

        # Return a safe fallback rather than a 500
        return TicketResponse(
            ticket_id=request.ticket_id,
            relevant_transaction_id=None,
            evidence_verdict="insufficient_data",
            case_type="other",
            severity="medium",
            department="customer_support",
            agent_summary="System error during analysis. Manual review required.",
            recommended_next_action="Escalate to supervisor for manual review.",
            customer_reply=(
                "Thank you for contacting us. Your complaint has been received and "
                "is being reviewed by our team. We will update you through official "
                "channels shortly."
            ),
            human_review_required=True,
            confidence=0.3,
            reason_codes=["system_error", "manual_review_needed"],
        )
