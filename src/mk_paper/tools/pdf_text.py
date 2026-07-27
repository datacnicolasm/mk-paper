"""Descarga y parseo de PDFs de acceso abierto."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pypdfium2 as pdfium

from mk_paper.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_DOI_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")


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


async def download_and_parse_pdf(
    client: httpx.AsyncClient,
    *,
    pdf_url: str,
    doi: str | None = None,
    settings: Settings | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> PdfTextResult:
    """Descarga un PDF OA y extrae texto (con límites de páginas/caracteres)."""
    cfg = settings or get_settings()
    if not pdf_url:
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

    async def _run() -> PdfTextResult:
        try:
            response = await client.get(
                pdf_url,
                headers={
                    "User-Agent": (
                        f"mk-paper/0.1 (mailto:{cfg.unpaywall_email})"
                        if cfg.unpaywall_email
                        else "mk-paper/0.1"
                    ),
                    "Accept": "application/pdf,*/*",
                },
                follow_redirects=True,
            )
            if response.status_code >= 400:
                return PdfTextResult(
                    doi=doi,
                    pdf_url=pdf_url,
                    text="",
                    char_count=0,
                    page_count=0,
                    local_path=None,
                    status="failed",
                    error=f"HTTP {response.status_code}",
                )

            content_type = (response.headers.get("content-type") or "").lower()
            data = response.content
            if "pdf" not in content_type and not data.startswith(b"%PDF"):
                return PdfTextResult(
                    doi=doi,
                    pdf_url=pdf_url,
                    text="",
                    char_count=0,
                    page_count=0,
                    local_path=None,
                    status="failed",
                    error=f"Not a PDF (content-type={content_type!r})",
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF download/parse failed for %s: %s", doi or pdf_url, exc)
            return PdfTextResult(
                doi=doi,
                pdf_url=pdf_url,
                text="",
                char_count=0,
                page_count=0,
                local_path=None,
                status="failed",
                error=str(exc),
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
    timeout = httpx.Timeout(cfg.http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        results = await asyncio.gather(
            *[
                download_and_parse_pdf(
                    client,
                    pdf_url=str(c["pdf_url"]),
                    doi=c.get("doi"),
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
            result = by_url[candidate["pdf_url"]]

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
