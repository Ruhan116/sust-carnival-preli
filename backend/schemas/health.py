"""Schemas for the health endpoint."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Readiness response returned by GET /health."""

    status: Literal["ok"] = "ok"
