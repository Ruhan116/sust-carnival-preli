"""Service entrypoint.

Exposes the ASGI ``app`` object that uvicorn runs (``uvicorn main:app``).
All routing and configuration live under the dedicated packages
(``api``, ``core``, ``schemas``, ``services``).
"""

import sys
import os

# Add the backend directory to the python path so imports work correctly on Vercel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.app import create_app

app = create_app()
