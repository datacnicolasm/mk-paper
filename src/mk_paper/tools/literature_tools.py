"""Herramientas de búsqueda bibliográfica científica.

Combina Semantic Scholar, OpenAlex y Unpaywall, uniendo resultados por DOI
y devolviendo un JSON limpio priorizando acceso abierto al texto completo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd
from crewai.tools import tool

from mk_paper.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_WORK_URL = "https://api.openalex.org/works/https://doi.org/{doi}"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"

S2_FIELDS = "title,year,abstract,citationCount,externalIds,url,venue,authors"
_DOI_PREFIX_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)
_CONCURRENCY = 5


def _normalize_doi(doi: str | None) -> str | None:
    """Normaliza un DOI a forma canónica minúscula sin prefijo URL."""
    if not doi or not isinstance(doi, str):
        return None
    cleaned = _DOI_PREFIX_RE.sub("", doi.strip())
    cleaned = cleaned.lower().rstrip(".")
    if not cleaned or "/" not in cleaned:
        return None
    return cleaned


async def _http_get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
) -> dict[str, Any] | list[Any] | None:
    """GET JSON con reintentos ante 429, 5xx y timeouts."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 404:
                return None
            if response.status_code in {429, 500, 502, 503, 504}:
                wait = min(2**attempt, 16)
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, int(retry_after))
                logger.warning(
                    "HTTP %s for %s (attempt %s/%s); retry in %ss",
                    response.status_code,
                    url,
                    attempt + 1,
                    max_retries + 1,
                    wait,
                )
                if attempt < max_retries:
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()

            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_error = exc
            wait = min(2**attempt, 16)
            logger.warning(
                "Request failed for %s: %s (attempt %s/%s)",
                url,
                exc,
                attempt + 1,
                max_retries + 1,
            )
            if attempt < max_retries:
                await asyncio.sleep(wait)
                continue

    if last_error:
        raise last_error
    return None


async def _search_semantic_scholar(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Busca papers en Semantic Scholar Graph API."""
    headers: dict[str, str] = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    # Over-fetch para compensar papers sin DOI tras el filtro.
    fetch_limit = min(max(limit * 3, limit), 100)
    data = await _http_get_json(
        client,
        SEMANTIC_SCHOLAR_URL,
        params={
            "query": query,
            "limit": fetch_limit,
            "fields": S2_FIELDS,
        },
        headers=headers or None,
        max_retries=settings.http_max_retries,
    )
    if not isinstance(data, dict):
        return []
    papers = data.get("data") or []
    return papers if isinstance(papers, list) else []


async def _fetch_openalex_one(
    client: httpx.AsyncClient,
    doi: str,
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, Any] | None]:
    """Lookup de un work en OpenAlex por DOI."""
    headers: dict[str, str] = {}
    if settings.unpaywall_email:
        headers["User-Agent"] = f"mk-paper/0.1 (mailto:{settings.unpaywall_email})"

    url = OPENALEX_WORK_URL.format(doi=quote(doi, safe="/"))
    async with semaphore:
        try:
            data = await _http_get_json(
                client,
                url,
                headers=headers or None,
                max_retries=settings.http_max_retries,
            )
        except Exception as exc:  # noqa: BLE001 — enriquecer sin tumbar el flujo
            logger.warning("OpenAlex failed for DOI %s: %s", doi, exc)
            return doi, None

    if not isinstance(data, dict):
        return doi, None

    oa = data.get("open_access") or {}
    best = data.get("best_oa_location") or {}
    primary = data.get("primary_location") or {}
    source = primary.get("source") if isinstance(primary, dict) else None
    venue = source.get("display_name") if isinstance(source, dict) else None
    return doi, {
        "openalex_id": data.get("id"),
        "is_oa": bool(oa.get("is_oa")),
        "oa_status": oa.get("oa_status"),
        "pdf_url": best.get("pdf_url"),
        "landing_url": best.get("landing_page_url") or data.get("id"),
        "venue": venue,
    }


async def _enrich_openalex(
    client: httpx.AsyncClient,
    dois: list[str],
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    """Enriquece metadatos OA desde OpenAlex de forma concurrente."""
    if not dois:
        return {}
    semaphore = asyncio.Semaphore(_CONCURRENCY)
    results = await asyncio.gather(
        *[_fetch_openalex_one(client, doi, settings, semaphore) for doi in dois]
    )
    return {doi: meta for doi, meta in results if meta}


async def _fetch_unpaywall_one(
    client: httpx.AsyncClient,
    doi: str,
    email: str,
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, Any] | None]:
    """Lookup OA en Unpaywall por DOI."""
    url = UNPAYWALL_URL.format(doi=quote(doi, safe="/"))
    async with semaphore:
        try:
            data = await _http_get_json(
                client,
                url,
                params={"email": email},
                max_retries=settings.http_max_retries,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unpaywall failed for DOI %s: %s", doi, exc)
            return doi, None

    if not isinstance(data, dict):
        return doi, None

    best = data.get("best_oa_location") or {}
    return doi, {
        "is_oa": bool(data.get("is_oa")),
        "oa_status": data.get("oa_status"),
        "pdf_url": best.get("url_for_pdf"),
        "landing_url": best.get("url_for_landing_page") or best.get("url"),
    }


async def _enrich_unpaywall(
    client: httpx.AsyncClient,
    dois: list[str],
    email: str,
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    """Enriquece PDF OA desde Unpaywall de forma concurrente."""
    if not dois:
        return {}
    semaphore = asyncio.Semaphore(_CONCURRENCY)
    results = await asyncio.gather(
        *[
            _fetch_unpaywall_one(client, doi, email, settings, semaphore)
            for doi in dois
        ]
    )
    return {doi: meta for doi, meta in results if meta}


def _s2_rows(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transforma papers de Semantic Scholar a filas normalizadas."""
    rows: list[dict[str, Any]] = []
    for paper in papers:
        external_ids = paper.get("externalIds") or {}
        doi = _normalize_doi(external_ids.get("DOI") if isinstance(external_ids, dict) else None)
        authors_raw = paper.get("authors") or []
        authors = [
            a.get("name")
            for a in authors_raw
            if isinstance(a, dict) and a.get("name")
        ]
        rows.append(
            {
                "doi": doi,
                "title": paper.get("title"),
                "year": paper.get("year"),
                "abstract": paper.get("abstract"),
                "citation_count": paper.get("citationCount") or 0,
                "venue": paper.get("venue"),
                "authors": authors,
                "landing_url": paper.get("url"),
                "s2_paper_id": paper.get("paperId"),
                "has_s2": True,
            }
        )
    return rows


def _merge_to_dataframe(
    s2_rows: list[dict[str, Any]],
    openalex: dict[str, dict[str, Any]],
    unpaywall: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Une Semantic Scholar con OpenAlex/Unpaywall por DOI."""
    df = pd.DataFrame(s2_rows)
    if df.empty:
        return df

    oa_rows = [{"doi": doi, **meta} for doi, meta in openalex.items()]
    up_rows = [{"doi": doi, **meta} for doi, meta in unpaywall.items()]
    oa_df = pd.DataFrame(oa_rows) if oa_rows else pd.DataFrame(columns=["doi"])
    up_df = pd.DataFrame(up_rows) if up_rows else pd.DataFrame(columns=["doi"])

    if not oa_df.empty:
        oa_df = oa_df.rename(
            columns={
                "is_oa": "oa_is_oa",
                "oa_status": "oa_oa_status",
                "pdf_url": "oa_pdf_url",
                "landing_url": "oa_landing_url",
                "venue": "oa_venue",
            }
        )
        df = df.merge(oa_df, on="doi", how="left")
    else:
        for col in ("oa_is_oa", "oa_oa_status", "oa_pdf_url", "oa_landing_url", "oa_venue", "openalex_id"):
            df[col] = None

    if not up_df.empty:
        up_df = up_df.rename(
            columns={
                "is_oa": "up_is_oa",
                "oa_status": "up_oa_status",
                "pdf_url": "up_pdf_url",
                "landing_url": "up_landing_url",
            }
        )
        df = df.merge(up_df, on="doi", how="left")
    else:
        for col in ("up_is_oa", "up_oa_status", "up_pdf_url", "up_landing_url"):
            df[col] = None

    def _coalesce_bool(row: pd.Series) -> bool:
        if pd.notna(row.get("up_is_oa")):
            return bool(row["up_is_oa"])
        if pd.notna(row.get("oa_is_oa")):
            return bool(row["oa_is_oa"])
        return False

    def _coalesce_str(*values: Any) -> str | None:
        for value in values:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if text and text.lower() != "nan":
                return text
        return None

    df["is_oa"] = df.apply(_coalesce_bool, axis=1)
    df["oa_status"] = df.apply(
        lambda r: _coalesce_str(r.get("up_oa_status"), r.get("oa_oa_status")),
        axis=1,
    )
    df["pdf_url"] = df.apply(
        lambda r: _coalesce_str(r.get("up_pdf_url"), r.get("oa_pdf_url")),
        axis=1,
    )
    df["landing_url"] = df.apply(
        lambda r: _coalesce_str(
            r.get("up_landing_url"),
            r.get("oa_landing_url"),
            r.get("landing_url"),
        ),
        axis=1,
    )
    df["venue"] = df.apply(
        lambda r: _coalesce_str(r.get("venue"), r.get("oa_venue")),
        axis=1,
    )
    df["has_openalex"] = df["doi"].map(lambda d: bool(d) and d in openalex)
    df["has_unpaywall"] = df["doi"].map(lambda d: bool(d) and d in unpaywall)
    return df


def _rank_and_filter(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Filtra papers con DOI y prioriza OA con PDF, citas y año."""
    if df.empty:
        return df

    ranked = df[df["doi"].notna() & (df["doi"] != "")].copy()
    if ranked.empty:
        return ranked

    ranked["has_pdf"] = ranked["pdf_url"].notna() & (ranked["pdf_url"] != "")
    ranked["citation_count"] = pd.to_numeric(ranked["citation_count"], errors="coerce").fillna(0)
    ranked["year"] = pd.to_numeric(ranked["year"], errors="coerce").fillna(0)

    ranked = ranked.sort_values(
        by=["has_pdf", "is_oa", "citation_count", "year"],
        ascending=[False, False, False, False],
    )
    return ranked.head(limit).reset_index(drop=True)


def _paper_sources(row: pd.Series) -> list[str]:
    """Lista de fuentes que contribuyeron al registro."""
    sources: list[str] = []
    if row.get("has_s2"):
        sources.append("semantic_scholar")
    if row.get("has_openalex"):
        sources.append("openalex")
    if row.get("has_unpaywall"):
        sources.append("unpaywall")
    return sources


def _to_json_payload(
    query: str,
    df: pd.DataFrame,
    warnings: list[str],
) -> str:
    """Serializa el DataFrame filtrado a JSON limpio."""
    papers: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        authors = row.get("authors")
        if isinstance(authors, list):
            author_list = [str(a) for a in authors]
        else:
            author_list = []

        year_val = row.get("year")
        year = int(year_val) if pd.notna(year_val) and int(year_val) != 0 else None
        citations = row.get("citation_count")
        citation_count = int(citations) if pd.notna(citations) else 0

        papers.append(
            {
                "doi": row.get("doi"),
                "title": row.get("title"),
                "year": year,
                "abstract": row.get("abstract") if pd.notna(row.get("abstract")) else None,
                "citation_count": citation_count,
                "venue": row.get("venue") if pd.notna(row.get("venue")) else None,
                "authors": author_list,
                "is_oa": bool(row.get("is_oa")),
                "oa_status": row.get("oa_status") if pd.notna(row.get("oa_status")) else None,
                "pdf_url": row.get("pdf_url") if pd.notna(row.get("pdf_url")) else None,
                "landing_url": row.get("landing_url")
                if pd.notna(row.get("landing_url"))
                else None,
                "sources": _paper_sources(row),
            }
        )

    payload = {
        "query": query,
        "count": len(papers),
        "papers": papers,
        "warnings": warnings,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def _run_literature_search(query: str, limit: int) -> str:
    """Orquesta búsqueda, enriquecimiento, join y serialización JSON."""
    settings = get_settings()
    warnings: list[str] = []
    limit = max(1, min(int(limit), 50))
    query = (query or "").strip()
    if not query:
        return json.dumps(
            {"query": query, "count": 0, "papers": [], "warnings": ["Empty query."]},
            ensure_ascii=False,
            indent=2,
        )

    timeout = httpx.Timeout(settings.http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            s2_papers = await _search_semantic_scholar(client, query, limit, settings)
        except Exception as exc:  # noqa: BLE001
            logger.error("Semantic Scholar search failed: %s", exc)
            return json.dumps(
                {
                    "query": query,
                    "count": 0,
                    "papers": [],
                    "warnings": [f"Semantic Scholar failed: {exc}"],
                },
                ensure_ascii=False,
                indent=2,
            )

        if not s2_papers:
            warnings.append("Semantic Scholar returned no results.")
            return _to_json_payload(query, pd.DataFrame(), warnings)

        s2_rows = _s2_rows(s2_papers)
        dois = sorted({row["doi"] for row in s2_rows if row.get("doi")})

        openalex_data: dict[str, dict[str, Any]] = {}
        unpaywall_data: dict[str, dict[str, Any]] = {}

        try:
            openalex_data = await _enrich_openalex(client, dois, settings)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"OpenAlex enrichment failed: {exc}")

        if settings.unpaywall_email:
            try:
                unpaywall_data = await _enrich_unpaywall(
                    client, dois, settings.unpaywall_email, settings
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Unpaywall enrichment failed: {exc}")
        else:
            warnings.append(
                "UNPAYWALL_EMAIL is not set; skipping Unpaywall OA/PDF enrichment."
            )

    merged = _merge_to_dataframe(s2_rows, openalex_data, unpaywall_data)
    ranked = _rank_and_filter(merged, limit)
    if ranked.empty and not merged.empty:
        warnings.append("No papers with a valid DOI after filtering.")
    return _to_json_payload(query, ranked, warnings)


@tool("Search Scientific Literature")
async def search_scientific_literature(query: str, limit: int = 10) -> str:
    """Search high-quality scientific literature and return clean JSON.

    Combines Semantic Scholar (title, year, abstract, citations, DOI) with
    OpenAlex and Unpaywall (open-access status and direct PDF links). Results
    are joined by DOI, filtered to valid DOIs, and ranked to prefer open-access
    full text.

    Args:
        query: Academic search terms (topic, keywords, or research question).
        limit: Maximum number of papers to return (1-50, default 10).

    Returns:
        A JSON string with keys query, count, papers, and warnings.
    """
    try:
        return await _run_literature_search(query, limit)
    except Exception as exc:  # noqa: BLE001 — la tool nunca tumba al agente
        logger.exception("search_scientific_literature failed")
        return json.dumps(
            {
                "query": query,
                "count": 0,
                "papers": [],
                "warnings": [f"Unexpected error: {exc}"],
            },
            ensure_ascii=False,
            indent=2,
        )
