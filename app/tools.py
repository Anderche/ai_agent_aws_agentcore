from __future__ import annotations

import json
from typing import Any

import requests
from langchain_core.tools import tool

from .config import load_settings
from .faq import load_faq, lookup_faq
from .form_workflow import (
    SecInquiryRecordError,
    build_prefill_url,
    record_inquiry,
)
from .sec import (
    SecLookupError,
    format_filings,
    get_cik,
    get_filings,
    parse_query,
)


def _network_disabled_response(tool_name: str) -> str:
    return (
        f"{tool_name} is currently disabled to keep this proof-of-concept in a "
        "no-cost configuration. Enable network tools by setting ENABLE_NETWORK_TOOLS=true."
    )


@tool
def download_reference_document(location: str) -> str:
    """Fetch a reference document from HTTP(S), local file, or S3."""
    settings = load_settings()
    if location.startswith("file://"):
        path = location.replace("file://", "")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()
        except OSError as exc:
            return f"Error reading local file: {exc}"

    if not settings.enable_network_tools:
        return _network_disabled_response("download_reference_document")

    try:
        if location.startswith("s3://"):
            import boto3  # Lazy import

            bucket, key = location[5:].split("/", 1)
            s3 = boto3.client("s3", region_name=settings.aws_region)
            obj = s3.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read().decode("utf-8")

        response = requests.get(location, timeout=settings.http_timeout)
        response.raise_for_status()
        return response.text
    except Exception as exc:  # noqa: BLE001
        return f"Unable to download document: {exc}"


@tool
def submit_ticket(details: str) -> str:
    """Submit a support ticket to a Google Form or HTTP endpoint."""
    settings = load_settings()
    if not settings.enable_network_tools:
        return _network_disabled_response("submit_ticket")

    if not settings.ticket_form_url:
        return (
            "Ticket form URL is not configured. Set TICKET_FORM_URL to enable ticket submissions."
        )

    try:
        payload: dict[str, Any] = json.loads(details)
    except json.JSONDecodeError as exc:
        return f"Ticket details must be valid JSON: {exc}"

    try:
        response = requests.post(
            settings.ticket_form_url,
            data=payload,
            timeout=settings.http_timeout,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"Unable to submit ticket: {exc}"
    return "Ticket submitted successfully."


@tool
def send_slack_notification(message: str, channel: str | None = None) -> str:
    """Send a message to a Slack channel via webhook."""
    settings = load_settings()
    if not settings.enable_network_tools:
        return _network_disabled_response("send_slack_notification")

    webhook_url = settings.slack_webhook_url
    if not webhook_url:
        return (
            "Slack webhook URL is not configured. Set SLACK_WEBHOOK_URL to enable notifications."
        )

    payload = {
        "text": message,
        "channel": channel or settings.slack_default_channel,
    }
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=settings.http_timeout,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"Unable to send Slack notification: {exc}"
    return "Notification sent."


@tool
def query_faq(question: str) -> str:
    """Return an answer from the static FAQ data."""
    settings = load_settings()
    faq_data = load_faq(settings.faq_path)
    answer = lookup_faq(question, faq_data)
    if answer:
        return answer
    return (
        "I couldn't find that in the FAQ. Please provide more detail or escalate via submit_ticket."
    )


@tool
def lookup_sec_filings(query: str) -> str:
    """Retrieve recent SEC EDGAR filings for a given 'Company Form' query."""
    settings = load_settings()
    if not settings.enable_network_tools:
        return _network_disabled_response("lookup_sec_filings")
    try:
        company_search, form_type = parse_query(query)
        cik = get_cik(
            company_search,
            timeout=settings.http_timeout,
        )
        filings = get_filings(
            cik,
            form_type,
            timeout=settings.http_timeout,
        )
        return format_filings(filings, form_type, company_search, cik)
    except (ValueError, SecLookupError) as exc:
        return f"Unable to retrieve SEC filings: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Unexpected error retrieving SEC filings: {exc}"


@tool
def initiate_sec_inquiry(details: str) -> str:
    """Create a SEC inquiry record, return a Google Form prefill link, and store attachments."""
    try:
        payload: dict[str, Any] = json.loads(details)
    except json.JSONDecodeError as exc:
        return f"SEC inquiry details must be valid JSON: {exc}"

    company = str(payload.get("company", "")).strip()
    form_type = str(payload.get("form_type", "")).strip()
    cik = payload.get("cik")
    context = payload.get("context")
    image_path = payload.get("image_path")

    if not company:
        return "SEC inquiry requires a 'company' field."
    if not form_type:
        return "SEC inquiry requires a 'form_type' field."

    settings = load_settings()
    prefill_url = build_prefill_url(
        settings,
        company=company,
        cik=str(cik).strip() if cik else None,
        form_type=form_type,
        context=str(context).strip() if context else None,
    )

    try:
        inquiry_id = record_inquiry(
            settings,
            company=company,
            cik=str(cik).strip() if cik else None,
            form_type=form_type,
            context=str(context).strip() if context else None,
            image_path=str(image_path).strip() if image_path else None,
            prefill_url=prefill_url,
        )
    except SecInquiryRecordError as exc:
        return f"Unable to record SEC inquiry: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Unexpected error while recording SEC inquiry: {exc}"

    response_lines = [
        f"SEC inquiry recorded with ID {inquiry_id}.",
    ]
    if prefill_url:
        response_lines.append(f"Prefilled Google Form: {prefill_url}")
    else:
        response_lines.append(
            "Google Form configuration is incomplete; notify operations to configure SEC_INQUIRY_FORM_BASE_URL."
        )
    if image_path:
        response_lines.append("Attachment archived for reviewer access.")
    response_lines.append(
        "Please complete any remaining questions in the Google Form to finalize the review request."
    )
    return "\n".join(response_lines)


TOOLS = [
    download_reference_document,
    submit_ticket,
    send_slack_notification,
    query_faq,
    lookup_sec_filings,
    initiate_sec_inquiry,
]

