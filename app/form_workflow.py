from __future__ import annotations

import shutil
import sqlite3
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .config import Settings

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class SecInquiryRecordError(Exception):
    """Raised when SEC inquiry workflow cannot be completed."""


def _ensure_directories(settings: Settings) -> None:
    settings.sec_inquiry_upload_dir.mkdir(parents=True, exist_ok=True)
    settings.sec_inquiry_db_path.parent.mkdir(parents=True, exist_ok=True)


def _get_db_connection(settings: Settings) -> sqlite3.Connection:
    _ensure_directories(settings)
    conn = sqlite3.connect(settings.sec_inquiry_db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sec_inquiries (
            id TEXT PRIMARY KEY,
            company TEXT NOT NULL,
            cik TEXT,
            form_type TEXT NOT NULL,
            context TEXT,
            image_path TEXT,
            prefill_url TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def build_prefill_url(
    settings: Settings,
    *,
    company: str,
    cik: Optional[str],
    form_type: str,
    context: Optional[str],
) -> Optional[str]:
    base_url = settings.sec_inquiry_form_base_url
    if not base_url:
        return None

    field_map: Dict[str, Optional[str]] = {
        settings.sec_inquiry_field_company: company,
        settings.sec_inquiry_field_cik: cik or "",
        settings.sec_inquiry_field_form: form_type,
        settings.sec_inquiry_field_context: context or "",
    }

    filtered = {
        key: value
        for key, value in field_map.items()
        if key and value is not None
    }
    if not filtered:
        return base_url

    parsed = urllib.parse.urlsplit(base_url)
    query_params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    for key, value in filtered.items():
        query_params[key] = [value]
    encoded_query = urllib.parse.urlencode(query_params, doseq=True)
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            encoded_query,
            parsed.fragment,
        )
    )


def _validate_and_store_image(
    settings: Settings,
    *,
    inquiry_id: str,
    image_path: Optional[str],
) -> Optional[str]:
    if not image_path:
        return None

    source = Path(image_path).expanduser()
    if not source.exists():
        raise SecInquiryRecordError(f"Attachment not found at {source}")

    suffix = source.suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise SecInquiryRecordError(
            f"Unsupported attachment type '{source.suffix}'. "
            "Allowed extensions: .jpg, .jpeg, .png."
        )

    max_bytes = settings.sec_inquiry_max_image_mb * 1024 * 1024
    if source.stat().st_size > max_bytes:
        raise SecInquiryRecordError(
            f"Attachment exceeds size limit of {settings.sec_inquiry_max_image_mb} MB."
        )

    destination = settings.sec_inquiry_upload_dir / f"{inquiry_id}{suffix}"
    shutil.copy2(source, destination)
    return str(destination)


def record_inquiry(
    settings: Settings,
    *,
    company: str,
    cik: Optional[str],
    form_type: str,
    context: Optional[str],
    image_path: Optional[str],
    prefill_url: Optional[str],
) -> str:
    inquiry_id = str(uuid.uuid4())
    stored_image_path = _validate_and_store_image(
        settings, inquiry_id=inquiry_id, image_path=image_path
    )

    conn = _get_db_connection(settings)
    with conn:
        conn.execute(
            """
            INSERT INTO sec_inquiries (
                id,
                company,
                cik,
                form_type,
                context,
                image_path,
                prefill_url,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inquiry_id,
                company,
                cik,
                form_type,
                context,
                stored_image_path,
                prefill_url,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    conn.close()
    return inquiry_id

