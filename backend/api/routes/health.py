"""Health check route."""

from fastapi import APIRouter

from schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness probe.

    The judge harness calls this to confirm the service is up before sending
    test cases. Must return {"status": "ok"} within 60 seconds of start.
    """
    return HealthResponse()
