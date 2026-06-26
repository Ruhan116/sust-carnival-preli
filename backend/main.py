"""Service entrypoint.

Exposes the ASGI ``app`` object that uvicorn runs (``uvicorn main:app``).
All routing and configuration live under the dedicated packages
(``api``, ``core``, ``schemas``, ``services``).
"""

from core.app import create_app

app = create_app()
