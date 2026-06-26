"""Application configuration.

Centralizes app-level settings so they live in one place instead of being
scattered across modules. Values can be overridden via environment variables.
"""

import os

APP_TITLE = "QueueStorm Investigator"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = (
    "AI/API SupportOps copilot that classifies, routes, and explains "
    "digital finance support tickets."
)


def env(name: str, default: str = "") -> str:
    """Read an environment variable, falling back to ``default``."""
    return os.getenv(name, default)
