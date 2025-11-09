from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Mapping

try:
    from aws_xray_sdk.core import patch_all, xray_recorder

    XRAY_AVAILABLE = True
except Exception:  # noqa: BLE001
    XRAY_AVAILABLE = False

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def _resolve_service_name(settings: Mapping[str, Any] | None = None) -> str:
    if settings and settings.get("observability_service_name"):
        return str(settings["observability_service_name"])
    return os.getenv("OBSERVABILITY_SERVICE_NAME", "agentcore-faq-agent")


_XRAY_ENABLED = False


@lru_cache(maxsize=1)
def configure_logging() -> None:
    """Configure structured logging once per process."""
    logging.basicConfig(
        level=DEFAULT_LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def configure_observability(settings: Mapping[str, Any] | None = None) -> None:
    """Initialise optional observability tooling such as AWS X-Ray."""
    configure_logging()

    if not XRAY_AVAILABLE:
        _set_xray_enabled(False)
        return

    enable_xray = None
    if settings is not None:
        maybe_enabled = settings.get("enable_xray")
        if maybe_enabled is not None:
            enable_xray = bool(maybe_enabled)

    if enable_xray is None:
        enable_xray = os.getenv("ENABLE_XRAY", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    if not enable_xray:
        _set_xray_enabled(False)
        return

    service_name = _resolve_service_name(settings)
    patch_all()  # Patch supported libraries (requests, boto3, etc.)
    xray_recorder.configure(service=service_name)
    _set_xray_enabled(True)


def _set_xray_enabled(enabled: bool) -> None:
    global _XRAY_ENABLED
    _XRAY_ENABLED = bool(enabled)


def begin_subsegment(name: str):
    if not XRAY_AVAILABLE or not _XRAY_ENABLED:
        return None
    try:
        return xray_recorder.begin_subsegment(name)
    except Exception:  # noqa: BLE001
        return None


def end_subsegment(subsegment) -> None:
    if not XRAY_AVAILABLE or not _XRAY_ENABLED or subsegment is None:
        return
    try:
        xray_recorder.end_subsegment()
    except Exception:  # noqa: BLE001
        pass

