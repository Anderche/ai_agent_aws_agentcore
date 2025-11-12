from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Literal, Mapping, Tuple
from urllib.parse import urlparse

import requests

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"


class SecLookupError(Exception):
    """Raised when SEC data cannot be retrieved or parsed."""


class SecDownloadError(Exception):
    """Raised when SEC filings cannot be downloaded."""


class SecRateLimitError(SecLookupError):
    """Raised when SEC returns too many requests."""


@dataclass
class SecFiling:
    date: str
    form: str
    url: str
    description: str | None = None
    size_bytes: int | None = None
    accession_number: str | None = None
    primary_document: str | None = None


@dataclass(frozen=True)
class FilingMenuEntry:
    index: int
    kind: Literal["filing", "type", "year"]
    filing: SecFiling | None = None
    form: str | None = None
    year: str | None = None


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


def build_sec_request_headers() -> dict[str, str]:
    """Return SEC-compliant request headers for use by other modules."""
    return dict(_sec_request_headers())


def _format_file_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(size_bytes)} B"


def _parse_content_length(headers: Mapping[str, str]) -> int | None:
    length = headers.get("Content-Length") or headers.get("content-length")
    if not length:
        return None
    try:
        return int(length)
    except (TypeError, ValueError):
        return None


def _retrieve_remote_file_size(url: str, timeout: float) -> int | None:
    headers = dict(_sec_request_headers())
    headers.setdefault("Accept-Encoding", "identity")

    response = None
    try:
        response = requests.head(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
    except requests.RequestException:
        response = None

    if response is not None:
        try:
            if response.status_code == 200:
                length = _parse_content_length(response.headers)
                if length is not None:
                    return length
        finally:
            response.close()

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            stream=True,
        )
    except requests.RequestException:
        return None

    try:
        if response.status_code != 200:
            return None
        length = _parse_content_length(response.headers)
        return length
    finally:
        response.close()


@lru_cache(maxsize=512)
def _cached_remote_file_size(url: str, timeout: float) -> int | None:
    return _retrieve_remote_file_size(url, timeout)


def _get_remote_file_size(url: str, *, timeout: float) -> int | None:
    return _cached_remote_file_size(url, timeout)


def _request_json(url: str, timeout: float) -> dict:
    response = requests.get(url, headers=_sec_request_headers(), timeout=timeout)
    if response.status_code == 403:
        raise SecLookupError(
            "SEC returned 403 Forbidden. Set SEC_USER_AGENT with contact info per SEC guidelines."
        )
    if response.status_code == 429:
        raise SecRateLimitError(
            "SEC returned 429 Too Many Requests. Wait before trying again."
        )
    if response.status_code != 200:
        raise SecLookupError(f"Request failed with status {response.status_code} for {url}.")
    try:
        return response.json()
    except ValueError as exc:  # JSONDecodeError derives from ValueError
        raise SecLookupError(f"Received invalid JSON from {url}: {exc}") from exc


def download_filing_to_directory(
    filing: SecFiling,
    *,
    timeout: float,
    download_dir: Path | None = None,
    max_size_bytes: int | None = None,
) -> Path:
    """
    Download a SEC filing to the provided directory, ensuring only the new file remains.

    Returns the path to the downloaded file.
    """
    # Use './downloads' as the default directory instead of user's Downloads folder
    target_dir = (download_dir or Path("./downloads")).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in target_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    size_bytes = filing.size_bytes
    if size_bytes is None:
        size_bytes = _get_remote_file_size(filing.url, timeout=timeout)
        filing.size_bytes = size_bytes

    if max_size_bytes is not None and size_bytes is not None and size_bytes > max_size_bytes:
        raise SecDownloadError(
            f"Filing {filing.form or ''} dated {filing.date} is "
            f"{_format_file_size(size_bytes)}, exceeding the download limit of "
            f"{_format_file_size(max_size_bytes)}."
        )

    response = requests.get(
        filing.url,
        headers=_sec_request_headers(),
        timeout=timeout,
        stream=True,
    )
    if response.status_code == 403:
        raise SecDownloadError(
            "SEC returned 403 Forbidden when downloading the filing. "
            "Ensure SEC_USER_AGENT is set with contact information per SEC guidelines."
        )
    if response.status_code == 429:
        raise SecDownloadError(
            "SEC returned 429 Too Many Requests while downloading the filing. Try again later."
        )
    if response.status_code != 200:
        raise SecDownloadError(
            f"Unable to download filing {filing.form or ''} from {filing.url} "
            f"(status code {response.status_code})."
        )

    parsed = urlparse(filing.url)
    filename = Path(parsed.path).name or f"{filing.form or 'filing'}.html"
    destination = target_dir / filename
    total_downloaded = 0
    try:
        with destination.open("wb") as buffer:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total_downloaded += len(chunk)
                if max_size_bytes is not None and total_downloaded > max_size_bytes:
                    response.close()
                    buffer.close()
                    try:
                        destination.unlink()
                    except FileNotFoundError:
                        pass
                    raise SecDownloadError(
                        f"Filing download exceeded the limit of {_format_file_size(max_size_bytes)}."
                    )
                buffer.write(chunk)
    finally:
        response.close()

    filing.size_bytes = total_downloaded or size_bytes

    return destination


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
    descriptions: List[str] = list(filings.get("description", []))

    sequence_length = min(
        len(forms),
        len(accession_numbers),
        len(primary_documents),
        len(report_dates),
    )

    matching: List[SecFiling] = []
    for idx in range(sequence_length):
        current_form = str(forms[idx])
        if form_type and current_form != form_type:
            continue
        accession = str(accession_numbers[idx]).replace("-", "")
        primary_doc = str(primary_documents[idx])
        report_date = str(report_dates[idx])
        description = descriptions[idx] if idx < len(descriptions) else None
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
        size_bytes = _get_remote_file_size(filing_url, timeout=timeout)
        matching.append(
            SecFiling(
                date=report_date or "N/A",
                form=current_form,
                url=filing_url,
                description=str(description) if description else None,
                size_bytes=size_bytes,
                accession_number=str(accession_numbers[idx]),
                primary_document=primary_doc,
            )
        )
        if limit and len(matching) >= limit:
            break
    return matching


def get_recent_filings(
    cik: str, *, timeout: float, limit: int = 20
) -> List[SecFiling]:
    return get_filings(cik, None, timeout=timeout, limit=limit)


def bucket_filings_by_type(filings: Iterable[SecFiling]) -> Dict[str, List[SecFiling]]:
    buckets: Dict[str, List[SecFiling]] = defaultdict(list)
    for filing in filings:
        buckets[filing.form].append(filing)
    return dict(buckets)


def bucket_filings_by_year(filings: Iterable[SecFiling]) -> Dict[str, List[SecFiling]]:
    buckets: Dict[str, List[SecFiling]] = defaultdict(list)
    for filing in filings:
        year = "Unknown"
        if filing.date and len(filing.date) >= 4:
            year = filing.date[:4]
        buckets[year].append(filing)
    return dict(buckets)


def format_filings(
    filings: Iterable[SecFiling],
    form_type: str | None,
    company_search: str,
    cik: str,
    *,
    include_index: bool = False,
    include_summary: bool = True,
    return_menu: bool = False,
) -> str | tuple[str, List[FilingMenuEntry]]:
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
    menu_entries: List[FilingMenuEntry] = []
    if form_type:
        header = f"Found {len(filings)} {form_type} filings for CIK {cik} (company search: {company_search})."
    else:
        header = (
            f"Found {len(filings)} recent filings for CIK {cik} "
            f"(company search: {company_search})."
        )
    lines = [header]
    for index, filing in enumerate(filings, start=1):
        prefix = f"{index}. " if include_index else ""
        descriptor = f" – {filing.description}" if filing.description else ""
        size_label = (
            f" ({_format_file_size(filing.size_bytes)})" if filing.size_bytes is not None else ""
        )
        lines.append(f"{prefix}{filing.date} [{filing.form}]: {filing.url}{size_label}{descriptor}")
        if include_index:
            menu_entries.append(
                FilingMenuEntry(
                    index=index,
                    kind="filing",
                    filing=filing,
                )
            )

    if include_summary:
        summary_index = len(menu_entries) + 1 if include_index else 1
        by_type = bucket_filings_by_type(filings)
        if by_type:
            lines.append("")
            lines.append("By type:")
            for type_index, (current_form, form_filings) in enumerate(
                sorted(
                    by_type.items(),
                    key=lambda item: (item[0], item[1][0].date),
                    reverse=False,
                ),
            ):
                lines.append(
                    f"{summary_index}. {current_form}: {len(form_filings)} filing(s) "
                    f"(most recent {form_filings[0].date})"
                )
                if include_index:
                    menu_entries.append(
                        FilingMenuEntry(
                            index=summary_index,
                            kind="type",
                            form=current_form,
                        )
                    )
                summary_index += 1
        by_year = bucket_filings_by_year(filings)
        if by_year:
            lines.append("")
            lines.append("By year:")
            for year_index, (year, year_filings) in enumerate(
                sorted(by_year.items(), key=lambda item: item[0], reverse=True),
            ):
                lines.append(f"{summary_index}. {year}: {len(year_filings)} filing(s)")
                if include_index:
                    menu_entries.append(
                        FilingMenuEntry(
                            index=summary_index,
                            kind="year",
                            year=year,
                        )
                    )
                summary_index += 1
    output = "\n".join(lines)
    if return_menu:
        return output, menu_entries
    return output


def format_cik_matches(matches: Iterable[CikMatch]) -> str:
    matches = list(matches)
    if not matches:
        return "No matching CIKs found."
    lines = ["Matching CIKs:"]
    for index, match in enumerate(matches, start=1):
        lines.append(f"{index}. {match.cik} ({match.ticker}) {match.title}")
    lines.append(
        "Reply with the numbered option or the desired 10-digit CIK to continue."
    )
    return "\n".join(lines)


