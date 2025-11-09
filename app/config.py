from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    environment: str
    aws_region: str
    bedrock_model_id: str
    memory_id: str | None
    actor_id: str
    faq_path: Path
    slack_webhook_url: str | None
    slack_default_channel: str | None
    ticket_form_url: str | None
    http_timeout: float
    enable_network_tools: bool
    observability_service_name: str
    enable_xray: bool
    sec_inquiry_form_base_url: str | None
    sec_inquiry_field_company: str | None
    sec_inquiry_field_cik: str | None
    sec_inquiry_field_form: str | None
    sec_inquiry_field_context: str | None
    sec_inquiry_upload_dir: Path
    sec_inquiry_db_path: Path
    sec_inquiry_max_image_mb: float


def _resolve_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    environment = os.getenv("APP_ENV", "development").lower()
    aws_region = os.getenv("AWS_REGION", "us-west-2")
    bedrock_model_id = os.getenv(
        "BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    memory_id = os.getenv("BEDROCK_AGENTCORE_MEMORY_ID")
    actor_id = os.getenv("DEFAULT_ACTOR_ID", "poc-user")
    faq_path = Path(
        os.getenv("FAQ_PATH", str(BASE_DIR / "data" / "faq.json"))
    ).expanduser()
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    slack_default_channel = os.getenv("SLACK_DEFAULT_CHANNEL", "#alerts")
    ticket_form_url = os.getenv("TICKET_FORM_URL")
    http_timeout = float(os.getenv("HTTP_TIMEOUT", "8"))
    network_tools_default = environment == "production"
    enable_network_tools = _resolve_bool(
        os.getenv("ENABLE_NETWORK_TOOLS"), network_tools_default
    )
    observability_service_name = os.getenv(
        "OBSERVABILITY_SERVICE_NAME", "agentcore-faq-agent"
    )
    enable_xray = _resolve_bool(os.getenv("ENABLE_XRAY"), False)
    return Settings(
        environment=environment,
        aws_region=aws_region,
        bedrock_model_id=bedrock_model_id,
        memory_id=memory_id,
        actor_id=actor_id,
        faq_path=faq_path,
        slack_webhook_url=slack_webhook_url,
        slack_default_channel=slack_default_channel,
        ticket_form_url=ticket_form_url,
        http_timeout=http_timeout,
        enable_network_tools=enable_network_tools,
        observability_service_name=observability_service_name,
        enable_xray=enable_xray,
        sec_inquiry_form_base_url=os.getenv("SEC_INQUIRY_FORM_BASE_URL"),
        sec_inquiry_field_company=os.getenv("SEC_INQUIRY_FIELD_COMPANY"),
        sec_inquiry_field_cik=os.getenv("SEC_INQUIRY_FIELD_CIK"),
        sec_inquiry_field_form=os.getenv("SEC_INQUIRY_FIELD_FORM"),
        sec_inquiry_field_context=os.getenv("SEC_INQUIRY_FIELD_CONTEXT"),
        sec_inquiry_upload_dir=Path(
            os.getenv(
                "SEC_INQUIRY_UPLOAD_DIR",
                str(BASE_DIR / "data" / "sec_inquiries" / "attachments"),
            )
        ).expanduser(),
        sec_inquiry_db_path=Path(
            os.getenv(
                "SEC_INQUIRY_DB_PATH",
                str(BASE_DIR / "data" / "sec_inquiries" / "inquiries.db"),
            )
        ).expanduser(),
        sec_inquiry_max_image_mb=float(os.getenv("SEC_INQUIRY_MAX_IMAGE_MB", "5")),
    )

