from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Tuple

import requests

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"


class SecLookupError(Exception):
    """Raised when SEC data cannot be retrieved or parsed."""


@dataclass
class SecFiling:
    date: str
    form: str
    url: str


def parse_query(query: str) -> Tuple[str, str]:
    parts = query.strip().split()
    if len(parts) < 2:
        raise ValueError("Query must include company and form, e.g., 'Apple 10-K'.")
    form_type = parts[-1].upper()
    company_search = " ".join(parts[:-1]).strip().lower()
    if not company_search:
        raise ValueError("Company search term cannot be empty.")
    return company_search, form_type


@lru_cache(maxsize=1)
def _sec_request_headers() -> dict[str, str]:
    user_agent = os.getenv("SEC_USER_AGENT") or os.getenv("HTTP_USER_AGENT")
    if not user_agent:
        user_agent = "AgentCorePOC/1.0 (contact: you@example.com)"
    return {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }


def _request_json(url: str, timeout: float) -> dict:
    response = requests.get(url, headers=_sec_request_headers(), timeout=timeout)
    if response.status_code == 403:
        raise SecLookupError(
            "SEC returned 403 Forbidden. Set SEC_USER_AGENT with contact info per SEC guidelines."
        )
    if response.status_code != 200:
        raise SecLookupError(f"Request failed with status {response.status_code} for {url}.")
    try:
        return response.json()
    except ValueError as exc:  # JSONDecodeError derives from ValueError
        raise SecLookupError(f"Received invalid JSON from {url}: {exc}") from exc


@dataclass
class CikMatch:
    ticker: str
    title: str
    cik: str


def list_ciks_for_symbol(symbol: str, *, timeout: float) -> List[CikMatch]:
    normalized = symbol.strip().lower()
    if not normalized:
        raise ValueError("Symbol must be a non-empty string.")

    data = _request_json(SEC_COMPANY_TICKERS_URL, timeout=timeout)
    matches: List[CikMatch] = []

    for entry in data.values():
        ticker = str(entry.get("ticker", "")).lower()
        title = str(entry.get("title", "")).strip()
        cik = entry.get("cik_str")

        if not cik:
            continue

        if normalized == ticker:
            matches.append(
                CikMatch(
                    ticker=ticker.upper(),
                    title=title,
                    cik=str(cik).zfill(10),
                )
            )
            break

        if normalized in title.lower():
            matches.append(
                CikMatch(
                    ticker=ticker.upper(),
                    title=title,
                    cik=str(cik).zfill(10),
                )
            )

    if not matches:
        raise SecLookupError(f"Company or ticker not found for search: {symbol}.")

    return matches


def get_cik(company_search: str, *, timeout: float) -> str:
    matches = list_ciks_for_symbol(company_search, timeout=timeout)
    return matches[0].cik


def get_filings(
    cik: str, form_type: str | None, *, timeout: float, limit: int | None = None
) -> List[SecFiling]:
    cik_padded = cik.zfill(10)
    url = SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik_padded)
    data = _request_json(url, timeout=timeout)
    filings = data.get("filings", {}).get("recent", {})
    forms: List[str] = list(filings.get("form", []))
    accession_numbers: List[str] = list(filings.get("accessionNumber", []))
    primary_documents: List[str] = list(filings.get("primaryDocument", []))
    report_dates: List[str] = list(filings.get("reportDate", []))

    matching: List[SecFiling] = []
    for idx, current_form in enumerate(forms):
        if form_type and current_form != form_type:
            continue
        accession = str(accession_numbers[idx]).replace("-", "")
        primary_doc = str(primary_documents[idx])
        report_date = str(report_dates[idx])
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
        matching.append(
            SecFiling(
                date=report_date or "N/A",
                form=current_form,
                url=filing_url,
            )
        )
        if limit and len(matching) >= limit:
            break
    return matching


def get_recent_filings(
    cik: str, *, timeout: float, limit: int = 5
) -> List[SecFiling]:
    return get_filings(cik, None, timeout=timeout, limit=limit)


def format_filings(
    filings: Iterable[SecFiling],
    form_type: str | None,
    company_search: str,
    cik: str,
) -> str:
    filings = list(filings)
    if not filings:
        if form_type:
            return (
                f"No {form_type} filings found for company search '{company_search}' "
                f"(CIK {cik})."
            )
        return (
            f"No recent filings found for company search '{company_search}' "
            f"(CIK {cik})."
        )
    if form_type:
        header = f"Found {len(filings)} {form_type} filings for CIK {cik} (company search: {company_search})."
    else:
        header = (
            f"Found {len(filings)} recent filings for CIK {cik} "
            f"(company search: {company_search})."
        )
    lines = [header]
    for filing in filings:
        lines.append(f"{filing.date} [{filing.form}]: {filing.url}")
    return "\n".join(lines)


def format_cik_matches(matches: Iterable[CikMatch]) -> str:
    matches = list(matches)
    if not matches:
        return "No matching CIKs found."
    lines = ["Matching CIKs:"]
    for match in matches:
        lines.append(f"- {match.cik} ({match.ticker}) {match.title}")
    lines.append("Reply with the desired 10-digit CIK to continue.")
    return "\n".join(lines)


