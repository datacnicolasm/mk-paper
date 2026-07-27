"""Descarga y parseo de PDFs de acceso abierto (resiliente a DNS/403)."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx
import pypdfium2 as pdfium

from mk_paper.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_DOI_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_ARXIV_ABS_RE = re.compile(
    r"https?://(?:export\.)?arxiv\.org/abs/([0-9]+\.[0-9]+)(v\d+)?",
    re.IGNORECASE,
)
_ARXIV_PDF_RE = re.compile(
    r"https?://(?:export\.)?arxiv\.org/pdf/([0-9]+\.[0-9]+)(v\d+)?\.?(?:pdf)?",
    re.IGNORECASE,
)
_UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"

# UA de navegador: muchos publishers (OUP, Wiley, MDPI) bloquean UAs de bot.
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class PdfTextResult:
    """Resultado de descarga + extracción de texto de un PDF."""

    doi: str | None
    pdf_url: str
    text: str
    char_count: int
    page_count: int
    local_path: str | None
    status: str  # ok | failed | skipped
    error: str | None = None


def _doi_slug(doi: str | None, fallback: str) -> str:
    if doi:
        return _DOI_SAFE_RE.sub("_", doi.lower())[:120]
    parsed = urlparse(fallback)
    name = Path(parsed.path).name or "paper"
    return _DOI_SAFE_RE.sub("_", name)[:120]


def _pdf_cache_dir(settings: Settings) -> Path:
    path = Path(settings.output_dir) / "literature" / "_pdf_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_retryable_error(exc: BaseException) -> bool:
    """DNS, timeouts y fallos de conexión transitorios."""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.NetworkError,
            httpx.TimeoutException,
        ),
    ):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname",
            "getaddrinfo failed",
            "connection reset",
            "network is unreachable",
            "errno -2",
            "errno -3",
            "errno -5",
        )
    )


def _request_headers(
    settings: Settings,
    *,
    referer: str | None = None,
) -> dict[str, str]:
    """Headers tipo navegador + mailto polite."""
    mailto = settings.unpaywall_email or "research@example.com"
    headers = {
        "User-Agent": f"{_BROWSER_UA} mk-paper/0.1 (mailto:{mailto})",
        "Accept": "application/pdf,application/octet-stream;q=0.9,text/html;q=0.8,*/*;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def candidate_pdf_urls(
    pdf_url: str,
    *,
    doi: str | None = None,
    landing_url: str | None = None,
    extra_urls: list[str] | None = None,
) -> list[str]:
    """Genera URLs alternativas cuando el enlace primario falla (DNS/403)."""
    urls: list[str] = []
    seen: set[str] = set()

    def _add(url: str | None) -> None:
        if not url:
            return
        cleaned = str(url).strip()
        if not cleaned.startswith("http"):
            return
        if cleaned in seen:
            return
        seen.add(cleaned)
        urls.append(cleaned)

    _add(pdf_url)
    for extra in extra_urls or []:
        _add(extra)

    abs_match = _ARXIV_ABS_RE.match(pdf_url or "")
    if abs_match:
        paper_id = abs_match.group(1) + (abs_match.group(2) or "")
        _add(f"https://arxiv.org/pdf/{paper_id}.pdf")
        _add(f"https://export.arxiv.org/pdf/{paper_id}.pdf")

    pdf_match = _ARXIV_PDF_RE.match(pdf_url or "")
    if pdf_match:
        paper_id = pdf_match.group(1) + (pdf_match.group(2) or "")
        _add(f"https://arxiv.org/pdf/{paper_id}.pdf")
        _add(f"https://export.arxiv.org/pdf/{paper_id}.pdf")

    if doi:
        _add(f"https://doi.org/{doi}")

    _add(landing_url)

    # Preferir repositorios OA (arxiv/pmc/zenodo) frente a publishers que bloquean bots.
    preferred_hosts = (
        "arxiv.org",
        "export.arxiv.org",
        "europepmc.org",
        "ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "zenodo.org",
        "ssrn.com",
        "osf.io",
        "hal.science",
        "hal.archives-ouvertes.fr",
    )

    def _rank(url: str) -> tuple[int, str]:
        host = urlparse(url).netloc.lower()
        for idx, preferred in enumerate(preferred_hosts):
            if preferred in host:
                return (idx, url)
        return (len(preferred_hosts) + 1, url)

    return sorted(urls, key=_rank)


async def _unpaywall_oa_pdf_urls(
    client: httpx.AsyncClient,
    doi: str,
    settings: Settings,
) -> list[str]:
    """Lista todas las URLs PDF OA conocidas por Unpaywall para un DOI."""
    email = settings.unpaywall_email
    if not email or not doi:
        return []
    url = _UNPAYWALL_URL.format(doi=quote(doi, safe="/"))
    try:
        response = await client.get(
            url,
            params={"email": email},
            headers=_request_headers(settings),
            follow_redirects=True,
        )
        if response.status_code >= 400:
            return []
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unpaywall OA lookup failed for %s: %s", doi, exc)
        return []

    urls: list[str] = []
    seen: set[str] = set()

    def _push(value: object) -> None:
        if isinstance(value, str) and value.startswith("http") and value not in seen:
            seen.add(value)
            urls.append(value)

    best = data.get("best_oa_location") or {}
    _push(best.get("url_for_pdf"))
    _push(best.get("url"))
    for loc in data.get("oa_locations") or []:
        if not isinstance(loc, dict):
            continue
        _push(loc.get("url_for_pdf"))
        _push(loc.get("url"))
    return urls


def extract_text_from_pdf_bytes(
    data: bytes,
    *,
    max_pages: int = 12,
    max_chars: int = 14000,
) -> tuple[str, int]:
    """Extrae texto de bytes PDF con pypdfium2. Retorna (texto, páginas leídas)."""
    pdf = pdfium.PdfDocument(data)
    try:
        n_pages = len(pdf)
        limit = min(n_pages, max(1, max_pages))
        chunks: list[str] = []
        total = 0
        for index in range(limit):
            page = pdf[index]
            try:
                textpage = page.get_textpage()
                page_text = textpage.get_text_bounded() or ""
                textpage.close()
            finally:
                page.close()
            if not page_text.strip():
                continue
            chunks.append(page_text.strip())
            total += len(page_text)
            if total >= max_chars:
                break
        text = "\n\n".join(chunks)
        if len(text) > max_chars:
            text = text[:max_chars]
        return text, limit
    finally:
        pdf.close()


def _parse_pdf_bytes(
    data: bytes,
    *,
    doi: str | None,
    pdf_url: str,
    cfg: Settings,
) -> PdfTextResult:
    """Valida bytes PDF, cachea y extrae texto."""
    if not data.startswith(b"%PDF"):
        return PdfTextResult(
            doi=doi,
            pdf_url=pdf_url,
            text="",
            char_count=0,
            page_count=0,
            local_path=None,
            status="failed",
            error="Not a PDF (missing %PDF header)",
        )

    cache_dir = _pdf_cache_dir(cfg)
    local_path = cache_dir / f"{_doi_slug(doi, pdf_url)}.pdf"
    local_path.write_bytes(data)

    text, page_count = extract_text_from_pdf_bytes(
        data,
        max_pages=cfg.pdf_max_pages,
        max_chars=cfg.pdf_max_chars,
    )
    if not text.strip():
        return PdfTextResult(
            doi=doi,
            pdf_url=pdf_url,
            text="",
            char_count=0,
            page_count=page_count,
            local_path=str(local_path),
            status="failed",
            error="PDF parsed but no extractable text (maybe scanned)",
        )

    return PdfTextResult(
        doi=doi,
        pdf_url=pdf_url,
        text=text,
        char_count=len(text),
        page_count=page_count,
        local_path=str(local_path),
        status="ok",
    )


async def _warmup_cookies(
    client: httpx.AsyncClient,
    *,
    doi: str | None,
    landing_url: str | None,
    settings: Settings,
) -> str | None:
    """Visita la landing page para obtener cookies de sesión (OUP/Wiley)."""
    targets: list[str] = []
    if landing_url:
        targets.append(landing_url)
    if doi:
        targets.append(f"https://doi.org/{doi}")
    referer: str | None = None
    for url in targets:
        try:
            response = await client.get(
                url,
                headers=_request_headers(settings, referer=referer),
                follow_redirects=True,
            )
            referer = str(response.url)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Cookie warmup failed for %s: %s", url, exc)
    return referer


async def _fetch_bytes_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    max_retries: int,
) -> tuple[bytes | None, str | None]:
    """GET con reintentos ante DNS/timeouts; retorna (bytes|None, error|None)."""
    last_error: str | None = None
    attempts = max(1, max_retries)
    for attempt in range(1, attempts + 1):
        try:
            response = await client.get(url, headers=headers, follow_redirects=True)
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code}"
                if response.status_code in {401, 403, 404, 410}:
                    return None, last_error
                if attempt < attempts:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                return None, last_error

            content_type = (response.headers.get("content-type") or "").lower()
            data = response.content
            if data.startswith(b"%PDF") or "pdf" in content_type:
                if data.startswith(b"%PDF"):
                    return data, None
                last_error = f"Not a PDF (content-type={content_type!r})"
                return None, last_error
            last_error = f"Not a PDF (content-type={content_type!r})"
            return None, last_error
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if _is_retryable_error(exc) and attempt < attempts:
                wait = min(2**attempt, 10)
                logger.warning(
                    "PDF fetch retry %s/%s for %s after %s (sleep %ss)",
                    attempt,
                    attempts,
                    url,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            return None, last_error
    return None, last_error or "unknown download error"


async def download_and_parse_pdf(
    client: httpx.AsyncClient,
    *,
    pdf_url: str,
    doi: str | None = None,
    landing_url: str | None = None,
    settings: Settings | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> PdfTextResult:
    """Descarga un PDF OA (con fallbacks) y extrae texto."""
    cfg = settings or get_settings()
    if not pdf_url and not doi:
        return PdfTextResult(
            doi=doi,
            pdf_url=pdf_url or "",
            text="",
            char_count=0,
            page_count=0,
            local_path=None,
            status="skipped",
            error="missing pdf_url",
        )

    cache_path = _pdf_cache_dir(cfg) / f"{_doi_slug(doi, pdf_url or doi or 'paper')}.pdf"
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        try:
            data = cache_path.read_bytes()
            if data.startswith(b"%PDF"):
                parsed = _parse_pdf_bytes(
                    data, doi=doi, pdf_url=pdf_url or str(cache_path), cfg=cfg
                )
                if parsed.status == "ok":
                    return parsed
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF cache read failed for %s: %s", cache_path, exc)

    async def _run() -> PdfTextResult:
        extra: list[str] = []
        if doi:
            extra = await _unpaywall_oa_pdf_urls(client, doi, cfg)

        urls = candidate_pdf_urls(
            pdf_url,
            doi=doi,
            landing_url=landing_url,
            extra_urls=extra,
        )
        errors: list[str] = []
        referer = await _warmup_cookies(
            client,
            doi=doi,
            landing_url=landing_url,
            settings=cfg,
        )
        if not referer:
            referer = landing_url or (f"https://doi.org/{doi}" if doi else None)

        for url in urls:
            headers = _request_headers(cfg, referer=referer)
            data, error = await _fetch_bytes_with_retries(
                client,
                url,
                headers=headers,
                max_retries=max(1, cfg.http_max_retries),
            )
            if data is None:
                errors.append(f"{url} → {error}")
                continue
            result = _parse_pdf_bytes(data, doi=doi, pdf_url=url, cfg=cfg)
            if result.status == "ok":
                return result
            errors.append(f"{url} → {result.error}")

        return PdfTextResult(
            doi=doi,
            pdf_url=pdf_url or "",
            text="",
            char_count=0,
            page_count=0,
            local_path=None,
            status="failed",
            error="; ".join(errors[:5]) if errors else "download failed",
        )

    if semaphore is None:
        return await _run()
    async with semaphore:
        return await _run()


async def enrich_candidates_with_pdf_text(
    candidates: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Descarga/parsea PDFs OA de candidatos y adjunta `full_text_excerpt`."""
    cfg = settings or get_settings()
    warnings: list[str] = []
    if not cfg.pdf_fulltext_enabled:
        warnings.append("PDF full-text extraction disabled (PDF_FULLTEXT_ENABLED=false).")
        return candidates, warnings

    targets = [
        c
        for c in candidates
        if c.get("pdf_url") and (c.get("is_oa") or c.get("pdf_url"))
    ]
    if not targets:
        warnings.append("No OA PDF URLs available for full-text extraction.")
        return candidates, warnings

    semaphore = asyncio.Semaphore(max(1, cfg.pdf_download_concurrency))
    timeout = httpx.Timeout(
        connect=min(20.0, cfg.http_timeout_seconds),
        read=max(60.0, cfg.http_timeout_seconds * 2),
        write=30.0,
        pool=30.0,
    )
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        limits=limits,
        http2=False,
    ) as client:
        results = await asyncio.gather(
            *[
                download_and_parse_pdf(
                    client,
                    pdf_url=str(c["pdf_url"]),
                    doi=c.get("doi"),
                    landing_url=c.get("landing_url"),
                    settings=cfg,
                    semaphore=semaphore,
                )
                for c in targets
            ]
        )

    by_doi = {r.doi: r for r in results if r.doi}
    by_url = {r.pdf_url: r for r in results}

    enriched: list[dict[str, Any]] = []
    ok_count = 0
    for candidate in candidates:
        item = dict(candidate)
        result = None
        if candidate.get("doi") and candidate["doi"] in by_doi:
            result = by_doi[candidate["doi"]]
        elif candidate.get("pdf_url") in by_url:
            result = by_url[str(candidate["pdf_url"])]

        if result is None:
            item["pdf_parse_status"] = "skipped"
            enriched.append(item)
            continue

        item["pdf_parse_status"] = result.status
        item["pdf_local_path"] = result.local_path
        item["full_text_chars"] = result.char_count
        if result.status == "ok":
            item["full_text_excerpt"] = result.text
            ok_count += 1
        else:
            item["full_text_excerpt"] = None
            if result.error:
                warnings.append(
                    f"PDF parse failed for {candidate.get('doi') or candidate.get('pdf_url')}: "
                    f"{result.error}"
                )
        enriched.append(item)

    warnings.append(
        f"PDF full-text extracted for {ok_count}/{len(targets)} OA candidates."
    )
    return enriched, warnings
