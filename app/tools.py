from __future__ import annotations

import ipaddress
import json
import logging
import socket
from typing import Any
from urllib.parse import urlparse

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


logger = logging.getLogger(__name__)

MAX_REFERENCE_DOCUMENT_BYTES = 2 * 1024 * 1024


def _network_disabled_response(tool_name: str) -> str:
    return (
        f"{tool_name} is currently disabled to keep this proof-of-concept in a "
        "no-cost configuration."
    )


def _not_configured_response(feature: str, env_var: str) -> str:
    logger.warning("%s is unavailable because %s is not set.", feature, env_var)
    return f"{feature} is not configured on this deployment."


def _s3_location_allowed(bucket: str, key: str, allowlist: tuple[str, ...]) -> bool:
    for entry in allowlist:
        allowed_bucket, _, allowed_prefix = entry.partition("/")
        if bucket == allowed_bucket and key.startswith(allowed_prefix):
            return True
    return False


def _is_public_host(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            return False
    return True


@tool
def download_reference_document(location: str) -> str:
    """Fetch a reference document from a public HTTPS URL or an allowlisted S3 location."""
    settings = load_settings()
    if not settings.enable_network_tools:
        return _network_disabled_response("download_reference_document")

    try:
        if location.startswith("s3://"):
            bucket, _, key = location[5:].partition("/")
            if not bucket or not key or not _s3_location_allowed(
                bucket, key, settings.reference_s3_allowlist
            ):
                return "That S3 location is not available to this assistant."

            import boto3  # Lazy import

            s3 = boto3.client("s3", region_name=settings.aws_region)
            obj = s3.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read(MAX_REFERENCE_DOCUMENT_BYTES)
            return body.decode("utf-8", errors="replace")

        parsed = urlparse(location)
        if parsed.scheme != "https" or not parsed.hostname:
            return "Only https:// links or allowlisted s3:// locations are supported."
        if not _is_public_host(parsed.hostname):
            return "That address is not reachable from this assistant."

        response = requests.get(
            location,
            timeout=settings.http_timeout,
            allow_redirects=False,
        )
        if response.is_redirect:
            return "That link redirects elsewhere; please provide the final URL."
        response.raise_for_status()
        return response.text[:MAX_REFERENCE_DOCUMENT_BYTES]
    except Exception:  # noqa: BLE001
        logger.exception("download_reference_document failed")
        return "Unable to download that document right now."


@tool
def submit_ticket(details: str) -> str:
    """Submit a support ticket to a Google Form or HTTP endpoint."""
    settings = load_settings()
    if not settings.enable_network_tools:
        return _network_disabled_response("submit_ticket")

    if not settings.ticket_form_url:
        return _not_configured_response("Ticket submission", "TICKET_FORM_URL")

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
    except Exception:  # noqa: BLE001
        logger.exception("submit_ticket failed")
        return "Unable to submit the ticket right now."
    return "Ticket submitted successfully."


@tool
def send_slack_notification(message: str, channel: str | None = None) -> str:
    """Send a message to a Slack channel via webhook."""
    settings = load_settings()
    if not settings.enable_network_tools:
        return _network_disabled_response("send_slack_notification")

    webhook_url = settings.slack_webhook_url
    if not webhook_url:
        return _not_configured_response("Slack notification", "SLACK_WEBHOOK_URL")

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
        # requests errors embed the request URL, which here is the secret webhook.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        logger.warning(
            "send_slack_notification failed: %s (status=%s)", type(exc).__name__, status
        )
        return "Unable to send the Slack notification right now."
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
    except Exception:  # noqa: BLE001
        logger.exception("lookup_sec_filings failed")
        return "Unexpected error retrieving SEC filings."


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
    except Exception:  # noqa: BLE001
        logger.exception("initiate_sec_inquiry failed")
        return "Unexpected error while recording the SEC inquiry."

    response_lines = [
        f"SEC inquiry recorded with ID {inquiry_id}.",
    ]
    if prefill_url:
        response_lines.append(f"Prefilled Google Form: {prefill_url}")
    else:
        response_lines.append(
            _not_configured_response("The SEC inquiry Google Form", "SEC_INQUIRY_FORM_BASE_URL")
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

