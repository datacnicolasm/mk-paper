"""Herramientas de búsqueda bibliográfica científica.

Combina Semantic Scholar (primario si hay API key), OpenAlex (fallback/enrich)
y Unpaywall (PDF OA), uniendo resultados por DOI en un JSON limpio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd
from crewai.tools import tool

from mk_paper.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_DOI_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
OPENALEX_SEARCH_URL = "https://api.openalex.org/works"
OPENALEX_WORK_URL = "https://api.openalex.org/works/https://doi.org/{doi}"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"

S2_FIELDS = (
    "title,year,abstract,citationCount,externalIds,url,venue,authors,fieldsOfStudy"
)
_DOI_PREFIX_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)
_CONCURRENCY = 5
# Semantic Scholar approved key: 1 request per second across all endpoints.
_S2_MIN_INTERVAL_SECONDS = 1.05

_s2_rate_lock = asyncio.Lock()
_s2_last_request_at = 0.0


def _normalize_doi(doi: str | None) -> str | None:
    """Normaliza un DOI a forma canónica minúscula sin prefijo URL."""
    if not doi or not isinstance(doi, str):
        return None
    cleaned = _DOI_PREFIX_RE.sub("", doi.strip())
    cleaned = cleaned.lower().rstrip(".")
    if not cleaned or "/" not in cleaned:
        return None
    return cleaned


def _openalex_headers(settings: Settings) -> dict[str, str]:
    """Headers polite para OpenAlex."""
    headers: dict[str, str] = {"Accept": "application/json"}
    if settings.unpaywall_email:
        headers["User-Agent"] = f"mk-paper/0.1 (mailto:{settings.unpaywall_email})"
    return headers


def _reconstruct_abstract(inverted: dict[str, Any] | None) -> str | None:
    """Reconstruye abstract desde abstract_inverted_index de OpenAlex."""
    if not inverted or not isinstance(inverted, dict):
        return None
    try:
        positions: list[tuple[int, str]] = []
        for word, idxs in inverted.items():
            if not isinstance(idxs, list):
                continue
            for idx in idxs:
                positions.append((int(idx), str(word)))
        if not positions:
            return None
        positions.sort(key=lambda item: item[0])
        return " ".join(word for _, word in positions)
    except (TypeError, ValueError):
        return None


async def _wait_for_s2_slot() -> None:
    """Espacia requests a Semantic Scholar a >= 1.05s (1 req/s)."""
    global _s2_last_request_at
    async with _s2_rate_lock:
        now = time.monotonic()
        wait = _S2_MIN_INTERVAL_SECONDS - (now - _s2_last_request_at)
        if wait > 0:
            await asyncio.sleep(wait)
        _s2_last_request_at = time.monotonic()


async def _http_get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
    min_retry_wait: float = 0.0,
) -> dict[str, Any] | list[Any] | None:
    """GET JSON con reintentos ante 429, 5xx y timeouts."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 404:
                return None
            if response.status_code in {429, 500, 502, 503, 504}:
                wait = max(min(2**attempt, 16), min_retry_wait)
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, float(retry_after))
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
            wait = max(min(2**attempt, 16), min_retry_wait)
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
    """Busca papers en Semantic Scholar Graph API (con x-api-key y 1 req/s)."""
    headers: dict[str, str] = {"Accept": "application/json"}
    api_key = settings.semantic_scholar_api_key
    if not api_key:
        raise ValueError("SEMANTIC_SCHOLAR_API_KEY is required for Semantic Scholar search")
    headers["x-api-key"] = api_key

    fetch_limit = min(max(limit * 3, limit), 100)
    await _wait_for_s2_slot()
    data = await _http_get_json(
        client,
        SEMANTIC_SCHOLAR_URL,
        params={
            "query": query,
            "limit": fetch_limit,
            "fields": S2_FIELDS,
        },
        headers=headers,
        max_retries=settings.http_max_retries,
        min_retry_wait=_S2_MIN_INTERVAL_SECONDS,
    )
    if not isinstance(data, dict):
        return []
    papers = data.get("data") or []
    return papers if isinstance(papers, list) else []


async def _fetch_s2_by_doi(
    client: httpx.AsyncClient,
    doi: str,
    settings: Settings,
) -> dict[str, Any] | None:
    """Lookup de un paper en Semantic Scholar por DOI."""
    headers: dict[str, str] = {"Accept": "application/json"}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    url = SEMANTIC_SCHOLAR_DOI_URL.format(doi=quote(doi, safe="/"))
    await _wait_for_s2_slot()
    try:
        data = await _http_get_json(
            client,
            url,
            params={"fields": S2_FIELDS},
            headers=headers,
            max_retries=settings.http_max_retries,
            min_retry_wait=_S2_MIN_INTERVAL_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic Scholar DOI lookup failed for %s: %s", doi, exc)
        return None
    return data if isinstance(data, dict) else None


async def fetch_rows_by_dois(
    dois: list[str],
    settings: Settings | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resuelve DOIs whitelist a filas normalizadas (S2 y/o OpenAlex).

    Returns:
        (rows, warnings)
    """
    cfg = settings or get_settings()
    warnings: list[str] = []
    normalized = []
    seen: set[str] = set()
    for raw in dois:
        doi = _normalize_doi(raw)
        if doi and doi not in seen:
            seen.add(doi)
            normalized.append(doi)
    if not normalized:
        return [], warnings

    rows_by_doi: dict[str, dict[str, Any]] = {}
    timeout = httpx.Timeout(cfg.http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for doi in normalized:
            row: dict[str, Any] | None = None
            if cfg.semantic_scholar_api_key:
                paper = await _fetch_s2_by_doi(client, doi, cfg)
                if paper:
                    s2_rows = _s2_rows([paper])
                    if s2_rows:
                        row = s2_rows[0]
            if row is None:
                try:
                    _, meta = await _fetch_openalex_one(
                        client, doi, cfg, asyncio.Semaphore(1)
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Seminal DOI {doi}: OpenAlex failed ({exc})")
                    meta = None
                if meta:
                    # Re-fetch full work via OpenAlex DOI URL for title/abstract/cites.
                    url = OPENALEX_WORK_URL.format(doi=quote(doi, safe="/"))
                    try:
                        work = await _http_get_json(
                            client,
                            url,
                            headers=_openalex_headers(cfg) or None,
                            max_retries=cfg.http_max_retries,
                        )
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"Seminal DOI {doi}: full OpenAlex failed ({exc})")
                        work = None
                    if isinstance(work, dict):
                        oa_rows = _openalex_rows([work])
                        if oa_rows:
                            row = oa_rows[0]
            if row is None:
                warnings.append(f"Seminal DOI not resolved: {doi}")
                continue
            row["doi"] = doi
            rows_by_doi[doi] = row

        # Enrich OA/PDF for resolved seminals.
        resolved = list(rows_by_doi.keys())
        openalex_data: dict[str, dict[str, Any]] = {}
        unpaywall_data: dict[str, dict[str, Any]] = {}
        try:
            openalex_data = await _enrich_openalex(client, resolved, cfg)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Seminal OpenAlex enrich failed: {exc}")
        if cfg.unpaywall_email:
            try:
                unpaywall_data = await _enrich_unpaywall(
                    client, resolved, cfg.unpaywall_email, cfg
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Seminal Unpaywall enrich failed: {exc}")

    merged = _merge_to_dataframe(
        list(rows_by_doi.values()), openalex_data, unpaywall_data
    )
    if merged.empty:
        return [], warnings

    out_rows: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        out_rows.append(r.to_dict())
    return out_rows, warnings


async def _search_openalex(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Busca works en OpenAlex como fuente primaria (fallback)."""
    fetch_limit = min(max(limit * 3, limit), 150)
    params: dict[str, Any] = {
        "search": query,
        "per_page": fetch_limit,
        "sort": "cited_by_count:desc",
    }
    if settings.unpaywall_email:
        params["mailto"] = settings.unpaywall_email

    data = await _http_get_json(
        client,
        OPENALEX_SEARCH_URL,
        params=params,
        headers=_openalex_headers(settings),
        max_retries=settings.http_max_retries,
    )
    if not isinstance(data, dict):
        return []
    results = data.get("results") or []
    return results if isinstance(results, list) else []


async def _fetch_openalex_one(
    client: httpx.AsyncClient,
    doi: str,
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, Any] | None]:
    """Lookup de un work en OpenAlex por DOI."""
    url = OPENALEX_WORK_URL.format(doi=quote(doi, safe="/"))
    async with semaphore:
        try:
            data = await _http_get_json(
                client,
                url,
                headers=_openalex_headers(settings) or None,
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
    concepts = data.get("concepts") or []
    keywords: list[str] = []
    if isinstance(concepts, list):
        for concept in concepts:
            if isinstance(concept, dict) and concept.get("display_name"):
                keywords.append(str(concept["display_name"]))
    return doi, {
        "openalex_id": data.get("id"),
        "is_oa": bool(oa.get("is_oa")),
        "oa_status": oa.get("oa_status"),
        "pdf_url": best.get("pdf_url"),
        "landing_url": best.get("landing_page_url") or data.get("id"),
        "venue": venue,
        "keywords": keywords,
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
        doi = _normalize_doi(
            external_ids.get("DOI") if isinstance(external_ids, dict) else None
        )
        authors_raw = paper.get("authors") or []
        authors = [
            a.get("name")
            for a in authors_raw
            if isinstance(a, dict) and a.get("name")
        ]
        fos = paper.get("fieldsOfStudy") or []
        keywords = [str(x) for x in fos if x] if isinstance(fos, list) else []
        rows.append(
            {
                "doi": doi,
                "title": paper.get("title"),
                "year": paper.get("year"),
                "abstract": paper.get("abstract"),
                "citation_count": paper.get("citationCount") or 0,
                "venue": paper.get("venue"),
                "authors": authors,
                "keywords": keywords,
                "landing_url": paper.get("url"),
                "s2_paper_id": paper.get("paperId"),
                "has_s2": True,
                "has_openalex": False,
                "is_oa": False,
                "oa_status": None,
                "pdf_url": None,
            }
        )
    return rows


def _openalex_rows(works: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transforma works de OpenAlex search a filas normalizadas."""
    rows: list[dict[str, Any]] = []
    for work in works:
        doi = _normalize_doi(work.get("doi"))
        authorships = work.get("authorships") or []
        authors = []
        for authorship in authorships:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author") or {}
            name = author.get("display_name") if isinstance(author, dict) else None
            if name:
                authors.append(name)

        oa = work.get("open_access") or {}
        best = work.get("best_oa_location") or {}
        primary = work.get("primary_location") or {}
        source = primary.get("source") if isinstance(primary, dict) else None
        venue = source.get("display_name") if isinstance(source, dict) else None
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        concepts = work.get("concepts") or []
        keywords: list[str] = []
        if isinstance(concepts, list):
            for concept in concepts:
                if not isinstance(concept, dict):
                    continue
                name = concept.get("display_name")
                if name:
                    keywords.append(str(name))

        rows.append(
            {
                "doi": doi,
                "title": work.get("display_name") or work.get("title"),
                "year": work.get("publication_year"),
                "abstract": abstract,
                "citation_count": work.get("cited_by_count") or 0,
                "venue": venue,
                "authors": authors,
                "keywords": keywords,
                "landing_url": best.get("landing_page_url")
                or (primary.get("landing_page_url") if isinstance(primary, dict) else None)
                or work.get("id"),
                "s2_paper_id": None,
                "openalex_id": work.get("id"),
                "has_s2": False,
                "has_openalex": True,
                "is_oa": bool(oa.get("is_oa")),
                "oa_status": oa.get("oa_status"),
                "pdf_url": best.get("pdf_url"),
            }
        )
    return rows


def _merge_to_dataframe(
    primary_rows: list[dict[str, Any]],
    openalex: dict[str, dict[str, Any]],
    unpaywall: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Une filas primarias con OpenAlex/Unpaywall por DOI."""
    df = pd.DataFrame(primary_rows)
    if df.empty:
        return df

    if "has_s2" not in df.columns:
        df["has_s2"] = False
    if "has_openalex" not in df.columns:
        df["has_openalex"] = False

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
                "keywords": "oa_keywords",
            }
        )
        df = df.merge(oa_df, on="doi", how="left")
    else:
        for col in (
            "oa_is_oa",
            "oa_oa_status",
            "oa_pdf_url",
            "oa_landing_url",
            "oa_venue",
            "oa_keywords",
            "openalex_id",
        ):
            if col not in df.columns:
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
        if pd.notna(row.get("is_oa")):
            return bool(row["is_oa"])
        return False

    def _coalesce_str(*values: Any) -> str | None:
        for value in values:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if text and text.lower() != "nan":
                return text
        return None

    primary_is_oa = df["is_oa"].copy() if "is_oa" in df.columns else None
    primary_pdf = df["pdf_url"].copy() if "pdf_url" in df.columns else None
    primary_status = df["oa_status"].copy() if "oa_status" in df.columns else None
    primary_landing = df["landing_url"].copy() if "landing_url" in df.columns else None
    primary_venue = df["venue"].copy() if "venue" in df.columns else None

    df["is_oa"] = df.apply(_coalesce_bool, axis=1)
    df["oa_status"] = df.apply(
        lambda r: _coalesce_str(
            r.get("up_oa_status"),
            r.get("oa_oa_status"),
            primary_status[r.name] if primary_status is not None else None,
        ),
        axis=1,
    )
    df["pdf_url"] = df.apply(
        lambda r: _coalesce_str(
            r.get("up_pdf_url"),
            r.get("oa_pdf_url"),
            primary_pdf[r.name] if primary_pdf is not None else None,
        ),
        axis=1,
    )
    df["landing_url"] = df.apply(
        lambda r: _coalesce_str(
            r.get("up_landing_url"),
            r.get("oa_landing_url"),
            primary_landing[r.name] if primary_landing is not None else None,
        ),
        axis=1,
    )
    df["venue"] = df.apply(
        lambda r: _coalesce_str(
            r.get("venue") if primary_venue is None else primary_venue[r.name],
            r.get("oa_venue"),
        ),
        axis=1,
    )

    def _merge_keywords(row: pd.Series) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for source in (row.get("keywords"), row.get("oa_keywords")):
            if not isinstance(source, list):
                continue
            for item in source:
                text = str(item).strip()
                key = text.lower()
                if text and key not in seen:
                    seen.add(key)
                    merged.append(text)
        return merged

    if "keywords" not in df.columns:
        df["keywords"] = [[] for _ in range(len(df))]
    df["keywords"] = df.apply(_merge_keywords, axis=1)

    # Preservar flags de la fuente primaria y marcar enrich.
    if primary_is_oa is not None:
        df["is_oa"] = df["is_oa"] | primary_is_oa.fillna(False).astype(bool)

    df["has_openalex"] = df.apply(
        lambda r: bool(r.get("has_openalex"))
        or (bool(r.get("doi")) and r.get("doi") in openalex),
        axis=1,
    )
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
    ranked["citation_count"] = pd.to_numeric(
        ranked["citation_count"], errors="coerce"
    ).fillna(0)
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
    *,
    primary_source: str | None = None,
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
                "abstract": row.get("abstract")
                if pd.notna(row.get("abstract"))
                else None,
                "citation_count": citation_count,
                "venue": row.get("venue") if pd.notna(row.get("venue")) else None,
                "authors": author_list,
                "keywords": list(row.get("keywords") or [])
                if isinstance(row.get("keywords"), list)
                else [],
                "is_oa": bool(row.get("is_oa")),
                "oa_status": row.get("oa_status")
                if pd.notna(row.get("oa_status"))
                else None,
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
        "primary_source": primary_source,
        "papers": papers,
        "warnings": warnings,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def _run_literature_search(query: str, limit: int) -> str:
    """Orquesta búsqueda (S2 → OpenAlex fallback), enrich Unpaywall y JSON."""
    settings = get_settings()
    warnings: list[str] = []
    limit = max(1, min(int(limit), 150))
    query = (query or "").strip()
    if not query:
        return json.dumps(
            {
                "query": query,
                "count": 0,
                "primary_source": None,
                "papers": [],
                "warnings": ["Empty query."],
            },
            ensure_ascii=False,
            indent=2,
        )

    primary_rows: list[dict[str, Any]] = []
    primary_source: str | None = None

    timeout = httpx.Timeout(settings.http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        # Plan A: Semantic Scholar solo si hay API key.
        if settings.semantic_scholar_api_key:
            try:
                s2_papers = await _search_semantic_scholar(
                    client, query, limit, settings
                )
                if s2_papers:
                    primary_rows = _s2_rows(s2_papers)
                    primary_source = "semantic_scholar"
                else:
                    warnings.append(
                        "Semantic Scholar returned no results; falling back to OpenAlex."
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("Semantic Scholar search failed: %s", exc)
                warnings.append(
                    f"Semantic Scholar failed ({exc}); falling back to OpenAlex."
                )
        else:
            warnings.append(
                "SEMANTIC_SCHOLAR_API_KEY is not set; using OpenAlex as primary source."
            )

        # Plan B: OpenAlex search primaria.
        if not primary_rows:
            try:
                oa_works = await _search_openalex(client, query, limit, settings)
                if oa_works:
                    primary_rows = _openalex_rows(oa_works)
                    primary_source = "openalex"
                else:
                    warnings.append("OpenAlex search returned no results.")
            except Exception as exc:  # noqa: BLE001
                logger.error("OpenAlex search failed: %s", exc)
                warnings.append(f"OpenAlex search failed: {exc}")

        if not primary_rows:
            return _to_json_payload(
                query,
                pd.DataFrame(),
                warnings,
                primary_source=primary_source,
            )

        dois = sorted({row["doi"] for row in primary_rows if row.get("doi")})

        openalex_data: dict[str, dict[str, Any]] = {}
        unpaywall_data: dict[str, dict[str, Any]] = {}

        # Enrich OpenAlex-by-DOI solo cuando la primaria fue S2.
        if primary_source == "semantic_scholar":
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

    merged = _merge_to_dataframe(primary_rows, openalex_data, unpaywall_data)
    ranked = _rank_and_filter(merged, limit)
    if ranked.empty and not merged.empty:
        warnings.append("No papers with a valid DOI after filtering.")
    return _to_json_payload(
        query,
        ranked,
        warnings,
        primary_source=primary_source,
    )


@tool("Search Scientific Literature")
async def search_scientific_literature(query: str, limit: int = 10) -> str:
    """Search high-quality scientific literature and return clean JSON.

    Uses Semantic Scholar when ``SEMANTIC_SCHOLAR_API_KEY`` is set; otherwise
    (or on S2 failure) falls back to OpenAlex search and enriches PDFs via
    Unpaywall. Results are joined by DOI and ranked to prefer open-access
    full text.

    Args:
        query: Academic search terms (topic, keywords, or research question).
        limit: Maximum number of papers to return (1-50, default 10).

    Returns:
        A JSON string with keys query, count, primary_source, papers, warnings.
    """
    try:
        return await _run_literature_search(query, limit)
    except Exception as exc:  # noqa: BLE001 — la tool nunca tumba al agente
        logger.exception("search_scientific_literature failed")
        return json.dumps(
            {
                "query": query,
                "count": 0,
                "primary_source": None,
                "papers": [],
                "warnings": [f"Unexpected error: {exc}"],
            },
            ensure_ascii=False,
            indent=2,
        )


# Re-export hybrid alignment API for literature tooling consumers.
from mk_paper.tools.alignment import (  # noqa: E402
    assign_quadrant,
    brief_to_profile_text,
    compute_alignment_scores,
    historical_centrality,
    paper_to_document,
    score_and_route_candidates,
)

__all__ = [
    "search_scientific_literature",
    "fetch_rows_by_dois",
    "brief_to_profile_text",
    "paper_to_document",
    "compute_alignment_scores",
    "historical_centrality",
    "assign_quadrant",
    "score_and_route_candidates",
]
