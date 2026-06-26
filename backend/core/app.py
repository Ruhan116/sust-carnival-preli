"""Application factory.

Builds and configures the FastAPI application. Keeping construction in a
factory keeps import-time side effects out of the entrypoint and makes the
app easy to instantiate in tests.
"""

from fastapi import FastAPI

from api.router import api_router
from core import config


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=config.APP_TITLE,
        version=config.APP_VERSION,
        description=config.APP_DESCRIPTION,
    )
    app.include_router(api_router)
    return app
