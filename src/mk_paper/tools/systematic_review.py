"""Embudo sistemático de revisión de literatura (ML + Groq).

Flujo: ResearchBrief → matriz (LLM) → dual S2/OpenAlex/Unpaywall →
ranking → TF-IDF/cosine alignment → cuadrantes (core auto / LLM / discard)
→ JSON Writer-ready con ``alignment_score``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
from crewai.tools import tool

from mk_paper.config.llm import get_fast_llm, get_llm
from mk_paper.config.settings import get_settings
from mk_paper.models.research_brief import (
    ClassifiedPaper,
    LiteratureReviewOutput,
    ResearchBrief,
    SearchMatrix,
)
from mk_paper.tools import literature_tools as lit
from mk_paper.tools.alignment import (
    AlignmentThresholds,
    historical_centrality,
    normalize_doi,
    normalize_doi_set,
    score_and_route_candidates,
)
from mk_paper.tools.pdf_text import enrich_candidates_with_pdf_text

logger = logging.getLogger(__name__)

_MAX_QUERIES = 4
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extrae el primer objeto JSON de una respuesta LLM."""
    raw = (text or "").strip()
    fence = _JSON_FENCE_RE.search(raw)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(raw[start : end + 1])


def _llm_call(prompt: str, *, fast: bool = False) -> str:
    """Invoca Groq vía CrewAI LLM y retorna texto."""
    llm = get_fast_llm() if fast else get_llm()
    result = llm.call(prompt)
    return str(result)


def build_search_matrix(brief: ResearchBrief) -> SearchMatrix:
    """Genera matriz de búsqueda avanzada con Groq a partir del brief."""
    prompt = f"""You are an expert systematic-review librarian.
Given the research brief below, produce an advanced academic search matrix.

Return ONLY valid JSON with this schema:
{{
  "queries": ["3 to 5 boolean/phrase queries for Semantic Scholar / OpenAlex"],
  "synonyms": ["key synonyms and variant terms"],
  "method_terms": ["methodological terms aligned to the brief"],
  "rationale": "one short paragraph explaining the matrix"
}}

Rules:
- queries must be specific, not a single vague phrase
- combine domain keywords, synonyms and method terms
- prefer English academic vocabulary
- max 5 queries

RESEARCH BRIEF:
{brief.model_dump_json(indent=2)}
"""
    try:
        data = _extract_json_object(_llm_call(prompt, fast=True))
        matrix = SearchMatrix.model_validate(data)
        if not matrix.queries:
            raise ValueError("empty queries")
        matrix.queries = matrix.queries[:_MAX_QUERIES]
        return matrix
    except Exception as exc:  # noqa: BLE001
        logger.warning("Groq search matrix failed (%s); using heuristic matrix", exc)
        seeds = []
        for values in brief.variables.values():
            seeds.extend(values)
        seeds.extend(brief.research_questions[:2])
        domain = brief.domain or brief.title
        method = brief.methodology.split()[:6]
        method_terms = [" ".join(method)] if method else ["empirical study"]
        queries = [
            f"{domain} {' '.join(seeds[:3])}".strip(),
            f"{domain} {method_terms[0]}".strip(),
            brief.title,
        ]
        return SearchMatrix(
            queries=[q for q in queries if q][:_MAX_QUERIES],
            synonyms=seeds[:10],
            method_terms=method_terms,
            rationale=f"Heuristic fallback matrix after LLM error: {exc}",
        )


async def _search_one_query(
    client: httpx.AsyncClient,
    query: str,
    per_query_limit: int,
    settings: Any,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Ejecuta una query con S2 (si hay key) o OpenAlex; retorna rows + sources."""
    rows: list[dict[str, Any]] = []
    sources_used: list[str] = []

    if settings.semantic_scholar_api_key:
        try:
            papers = await lit._search_semantic_scholar(
                client, query, per_query_limit, settings
            )
            if papers:
                rows = lit._s2_rows(papers)
                sources_used.append("semantic_scholar")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"S2 failed for query {query!r}: {exc}")

    if not rows:
        try:
            works = await lit._search_openalex(
                client, query, per_query_limit, settings
            )
            if works:
                rows = lit._openalex_rows(works)
                sources_used.append("openalex")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"OpenAlex failed for query {query!r}: {exc}")

    return rows, sources_used


async def run_dual_search(
    matrix: SearchMatrix,
    brief: ResearchBrief,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Busca con la matriz, deduplica por DOI y enriquece Unpaywall/OpenAlex."""
    settings = get_settings()
    warnings: list[str] = []
    sources_used: set[str] = set()
    all_rows: list[dict[str, Any]] = []
    queries = matrix.queries[:_MAX_QUERIES] or [brief.title]
    per_query = max(5, min(brief.max_results, 15))

    timeout = httpx.Timeout(settings.http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for query in queries:
            rows, used = await _search_one_query(
                client, query, per_query, settings, warnings
            )
            sources_used.update(used)
            all_rows.extend(rows)

        # Dedupe by DOI (prefer first occurrence with more complete metadata).
        by_doi: dict[str, dict[str, Any]] = {}
        no_doi: list[dict[str, Any]] = []
        for row in all_rows:
            doi = row.get("doi")
            if not doi:
                no_doi.append(row)
                continue
            prev = by_doi.get(doi)
            if prev is None:
                by_doi[doi] = row
            else:
                # Prefer row with abstract / higher citations.
                prev_score = int(bool(prev.get("abstract"))) + int(
                    prev.get("citation_count") or 0
                )
                new_score = int(bool(row.get("abstract"))) + int(
                    row.get("citation_count") or 0
                )
                if new_score > prev_score:
                    by_doi[doi] = row

        primary_rows = list(by_doi.values())
        dois = sorted(by_doi.keys())

        openalex_data: dict[str, dict[str, Any]] = {}
        unpaywall_data: dict[str, dict[str, Any]] = {}

        # Enrich OpenAlex-by-DOI for S2-origin rows missing OA info.
        need_oa = [
            d
            for d, r in by_doi.items()
            if r.get("has_s2") and not r.get("pdf_url")
        ]
        if need_oa:
            try:
                openalex_data = await lit._enrich_openalex(client, need_oa, settings)
                if openalex_data:
                    sources_used.add("openalex")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"OpenAlex enrichment failed: {exc}")

        if settings.unpaywall_email and dois:
            try:
                unpaywall_data = await lit._enrich_unpaywall(
                    client, dois, settings.unpaywall_email, settings
                )
                if unpaywall_data:
                    sources_used.add("unpaywall")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Unpaywall enrichment failed: {exc}")
        elif not settings.unpaywall_email:
            warnings.append("UNPAYWALL_EMAIL not set; skipping PDF enrichment.")

    merged = lit._merge_to_dataframe(primary_rows, openalex_data, unpaywall_data)
    return merged, sorted(sources_used), warnings


def rank_candidates(
    df: pd.DataFrame,
    *,
    years_back: int = 3,
    limit: int = 20,
    seminal_dois: list[str] | None = None,
) -> pd.DataFrame:
    """Ranking ponderado: recencia + citas; preserva whitelist/seminales.

    Papers en ``seminal_dois`` o con alta centralidad histórica (citas/años)
    no se eliminan por antigüedad y se garantizan en el top aunque el
    ``limit`` favorezca literatura reciente.
    """
    if df.empty:
        return df

    ranked = df[df["doi"].notna() & (df["doi"] != "")].copy()
    if ranked.empty:
        return ranked

    settings = get_settings()
    current_year = datetime.utcnow().year
    cutoff = current_year - max(years_back, 0)
    whitelist = normalize_doi_set(seminal_dois)
    min_age = int(settings.seminal_min_age_years)
    cites_per_year_thr = float(settings.seminal_cites_per_year)
    min_cites = int(settings.seminal_min_citations)

    ranked["citation_count"] = pd.to_numeric(
        ranked["citation_count"], errors="coerce"
    ).fillna(0)
    ranked["year"] = pd.to_numeric(ranked["year"], errors="coerce").fillna(0)
    ranked["has_pdf"] = ranked["pdf_url"].notna() & (ranked["pdf_url"] != "")
    ranked["is_oa"] = ranked["is_oa"].fillna(False).astype(bool)
    ranked["doi_norm"] = ranked["doi"].map(lambda d: normalize_doi(str(d)) if d else None)

    max_cites = float(ranked["citation_count"].max() or 1.0)
    ranked["cite_norm"] = ranked["citation_count"] / max_cites
    ranked["recent"] = (ranked["year"] >= cutoff).astype(float)
    ranked["age"] = (current_year - ranked["year"]).clip(lower=0)
    ranked["recency_score"] = (
        ranked["recent"] * 1.0 + (1.0 / (1.0 + ranked["age"] / 5.0)) * 0.35
    )
    ranked["historical_centrality"] = ranked.apply(
        lambda r: historical_centrality(
            r["citation_count"], r["year"], current_year=current_year
        ),
        axis=1,
    )
    ranked["force_keep"] = ranked.apply(
        lambda r: bool(
            (r.get("doi_norm") in whitelist)
            or (
                int(r.get("age") or 0) >= min_age
                and (
                    float(r.get("historical_centrality") or 0) >= cites_per_year_thr
                    or int(r.get("citation_count") or 0) >= min_cites
                )
            )
        ),
        axis=1,
    )
    ranked["impact_score"] = (
        0.50 * ranked["cite_norm"]
        + 0.30 * ranked["recency_score"]
        + 0.10 * ranked["has_pdf"].astype(float)
        + 0.10 * (ranked["historical_centrality"] / max(cites_per_year_thr, 1.0)).clip(
            upper=1.0
        )
        + 0.25 * ranked["force_keep"].astype(float)
    )

    ranked = ranked.sort_values(
        by=["force_keep", "impact_score", "has_pdf", "is_oa", "citation_count", "year"],
        ascending=[False, False, False, False, False, False],
    )

    forced = ranked[ranked["force_keep"]].copy()
    modern = ranked[~ranked["force_keep"]].copy()
    remaining = max(limit - len(forced), 0)
    selected = pd.concat([forced, modern.head(remaining)], ignore_index=True)
    selected = selected.drop_duplicates(subset=["doi_norm"], keep="first")
    return selected.reset_index(drop=True)


def _candidates_for_llm(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Reduce candidatos a payload compacto para clasificación Groq."""
    items: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        abstract = row.get("abstract")
        if abstract is not None and not (isinstance(abstract, float) and pd.isna(abstract)):
            abstract_text = str(abstract)[:1200]
        else:
            abstract_text = None
        items.append(
            {
                "doi": row.get("doi"),
                "title": row.get("title"),
                "year": int(row["year"])
                if pd.notna(row.get("year")) and int(row["year"]) != 0
                else None,
                "citation_count": int(row.get("citation_count") or 0),
                "abstract": abstract_text,
                "is_oa": bool(row.get("is_oa")),
                "oa_status": row.get("oa_status")
                if pd.notna(row.get("oa_status"))
                else None,
                "pdf_url": row.get("pdf_url")
                if pd.notna(row.get("pdf_url"))
                else None,
                "landing_url": row.get("landing_url")
                if pd.notna(row.get("landing_url"))
                else None,
                "venue": row.get("venue") if pd.notna(row.get("venue")) else None,
                "authors": row.get("authors")
                if isinstance(row.get("authors"), list)
                else [],
                "keywords": row.get("keywords")
                if isinstance(row.get("keywords"), list)
                else [],
                "sources": lit._paper_sources(row),
                "impact_score": float(row.get("impact_score") or 0),
                "historical_centrality": float(row.get("historical_centrality") or 0),
            }
        )
    return items


def _prompt_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compacta candidatos para el prompt (recorta full text)."""
    compact: list[dict[str, Any]] = []
    for item in candidates:
        row = {
            "doi": item.get("doi"),
            "title": item.get("title"),
            "year": item.get("year"),
            "citation_count": item.get("citation_count"),
            "abstract": item.get("abstract"),
            "is_oa": item.get("is_oa"),
            "pdf_url": item.get("pdf_url"),
            "venue": item.get("venue"),
            "authors": item.get("authors"),
            "sources": item.get("sources"),
            "impact_score": item.get("impact_score"),
            "alignment_score": item.get("alignment_score"),
            "alignment_quadrant": item.get("alignment_quadrant"),
            "pdf_parse_status": item.get("pdf_parse_status"),
            "full_text_chars": item.get("full_text_chars") or 0,
        }
        excerpt = item.get("full_text_excerpt")
        if excerpt:
            row["full_text_excerpt"] = str(excerpt)[:8000]
        compact.append(row)
    return compact


def _paper_from_base(
    base: dict[str, Any],
    *,
    level: str,
    utility: float,
    relevance_score: float,
    key_findings: list[str],
    citation_context: str,
    suggested_section: str,
) -> ClassifiedPaper:
    """Construye ClassifiedPaper preservando metadatos de PDF/full text."""
    return ClassifiedPaper(
        doi=base.get("doi"),
        title=base.get("title"),
        year=base.get("year"),
        abstract=base.get("abstract"),
        citation_count=int(base.get("citation_count") or 0),
        venue=base.get("venue"),
        authors=list(base.get("authors") or []),
        is_oa=bool(base.get("is_oa")),
        oa_status=base.get("oa_status"),
        pdf_url=base.get("pdf_url"),
        landing_url=base.get("landing_url"),
        sources=list(base.get("sources") or []),
        level=level,  # type: ignore[arg-type]
        utility=utility,
        relevance_score=relevance_score,
        alignment_score=float(base.get("alignment_score") or 0.0),
        alignment_quadrant=base.get("alignment_quadrant"),
        historical_centrality=float(base.get("historical_centrality") or 0.0),
        seminal_reason=base.get("seminal_reason"),
        key_findings=key_findings,
        citation_context=citation_context,
        suggested_section=suggested_section,
        full_text_excerpt=base.get("full_text_excerpt"),
        full_text_chars=int(base.get("full_text_chars") or 0),
        pdf_local_path=base.get("pdf_local_path"),
        pdf_parse_status=base.get("pdf_parse_status"),
    )


def _seminal_paper(brief: ResearchBrief, item: dict[str, Any]) -> ClassifiedPaper:
    """Paper fundacional: marco teórico/histórico (nunca Nivel 1 empírico)."""
    score = float(item.get("alignment_score") or 0.0)
    centrality = float(item.get("historical_centrality") or 0.0)
    reason = str(item.get("seminal_reason") or "historical_centrality")
    findings = [
        f"Seminal Literature via {reason}; "
        f"historical_centrality={centrality:.2f} cites/year; "
        f"alignment_score={score:.4f}.",
        "Use only to define the origin of the model/concept in the theoretical framework.",
    ]
    return _paper_from_base(
        item,
        level="seminal",
        utility=0.8,
        relevance_score=max(score, min(1.0, centrality / 100.0)),
        key_findings=findings,
        citation_context=(
            f"Citar como fundamento teórico/histórico ({reason}) en el marco "
            f"teórico al definir el origen del modelo/concepto para: {brief.title}. "
            "No usar como hallazgo empírico moderno (Nivel 1)."
        ),
        suggested_section="Background / Theoretical framework (Seminal Literature)",
    )


def _core_auto_paper(brief: ResearchBrief, item: dict[str, Any]) -> ClassifiedPaper:
    """Construye paper Core (Nivel 1) desde cuadrante ``core_auto`` (sin LLM)."""
    score = float(item.get("alignment_score") or 0.0)
    has_text = bool(item.get("full_text_excerpt"))
    findings = [
        f"Auto-classified Core by TF-IDF cosine alignment_score={score:.4f}.",
    ]
    if has_text:
        findings.append(
            f"Full text available ({item.get('full_text_chars', 0)} chars); "
            "extract empirical metrics in writer stage."
        )
    else:
        findings.append(
            "OA PDF present but full text not extracted; use abstract/methods carefully."
        )
    return _paper_from_base(
        item,
        level="core",
        utility=1.0,
        relevance_score=score,
        key_findings=findings,
        citation_context=(
            f"Cita core (alineación matemática {score:.3f}) en Resultados/Metodología "
            f"respecto a: {brief.title}"
        ),
        suggested_section="Results / Methodology",
    )


def _conceptual_fallback(
    brief: ResearchBrief,
    item: dict[str, Any],
    *,
    warning_note: str = "",
) -> ClassifiedPaper:
    """Referencia conceptual heurística (Nivel 2) si Groq falla."""
    score = float(item.get("alignment_score") or 0.0)
    note = f" {warning_note}" if warning_note else ""
    return _paper_from_base(
        item,
        level="conceptual",
        utility=0.5,
        relevance_score=score,
        key_findings=[],
        citation_context=(
            f"Referencia conceptual (alignment_score={score:.3f}) para "
            f"Antecedentes/Marco teórico sobre: {brief.domain or brief.title}.{note}"
        ),
        suggested_section="Background / Theoretical framework",
    )


def _llm_enrich_medium(
    brief: ResearchBrief,
    medium: list[dict[str, Any]],
) -> tuple[list[ClassifiedPaper], list[str]]:
    """Análisis conceptual profundo (Nivel 2) vía Groq para scores intermedios."""
    warnings: list[str] = []
    if not medium:
        return [], warnings

    prompt = f"""You are a rigorous peer reviewer enriching mid-alignment papers.

These papers already passed a TF-IDF cosine filter (medium alignment_score).
They are Nivel 2 (conceptual). Do NOT promote them to core.
Return ONLY valid JSON:
{{
  "papers": [
    {{
      "doi": "...",
      "relevance_score": 0.0-1.0,
      "key_findings": ["short conceptual takeaways; empty if none"],
      "citation_context": "exact guidance for the Academic Writer on where/how to cite",
      "suggested_section": "Introduction|Background|Methodology|Results|Discussion"
    }}
  ]
}}

Rules:
- Match every input paper by doi exactly once.
- Prefer conceptual framing (definitions, theory, contrast, gaps).
- When full_text_excerpt is present, ground citation_context in that text.
- Mention the given alignment_score when useful for the Writer.

RESEARCH BRIEF:
{brief.model_dump_json(indent=2)}

CANDIDATES (already scored):
{json.dumps(_prompt_candidates(medium), ensure_ascii=False, indent=2)}
"""
    by_doi = {c.get("doi"): c for c in medium if c.get("doi")}
    conceptual: list[ClassifiedPaper] = []

    try:
        data = _extract_json_object(_llm_call(prompt, fast=False))
        classified_raw = data.get("papers") or []
        seen: set[str] = set()

        for item in classified_raw:
            doi = item.get("doi")
            base = by_doi.get(doi) or {}
            if not base:
                continue
            seen.add(str(doi))
            score = float(base.get("alignment_score") or 0.0)
            conceptual.append(
                _paper_from_base(
                    base,
                    level="conceptual",
                    utility=0.5,
                    relevance_score=float(item.get("relevance_score") or score),
                    key_findings=list(item.get("key_findings") or []),
                    citation_context=str(item.get("citation_context") or ""),
                    suggested_section=str(item.get("suggested_section") or "Background"),
                )
            )

        for doi, base in by_doi.items():
            if doi in seen:
                continue
            conceptual.append(_conceptual_fallback(brief, base))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Groq mid-alignment enrichment failed: %s", exc)
        warnings.append(f"Groq conceptual enrichment error: {exc}")
        conceptual = [_conceptual_fallback(brief, c, warning_note="(heuristic)") for c in medium]

    return conceptual, warnings


def classify_papers(
    brief: ResearchBrief,
    candidates: list[dict[str, Any]],
    *,
    matrix: SearchMatrix | None = None,
    chapter_text: str | None = None,
) -> LiteratureReviewOutput:
    """Filtro híbrido: TF-IDF/cosine + seminal → Core / Conceptual / Seminal."""
    settings = get_settings()
    thresholds = AlignmentThresholds(
        high=float(settings.alignment_high_threshold),
        low=float(settings.alignment_low_threshold),
    )
    threshold_meta = {
        "high": thresholds.high,
        "low": thresholds.low,
        "seminal_min_age_years": float(settings.seminal_min_age_years),
        "seminal_cites_per_year": float(settings.seminal_cites_per_year),
        "seminal_min_citations": float(settings.seminal_min_citations),
        "seminal_alignment_floor": float(settings.seminal_alignment_floor),
    }

    if not candidates:
        return LiteratureReviewOutput(
            brief_title=brief.title,
            candidate_count=0,
            discarded_count=0,
            alignment_thresholds=threshold_meta,
            warnings=["No candidates to classify."],
        )

    scored, discarded, align_warnings = score_and_route_candidates(
        brief,
        candidates,
        matrix=matrix,
        chapter_text=chapter_text,
        settings=settings,
    )

    core: list[ClassifiedPaper] = []
    seminal: list[ClassifiedPaper] = []
    medium: list[dict[str, Any]] = []
    for item in scored:
        if item.quadrant == "seminal":
            seminal.append(_seminal_paper(brief, item.candidate))
        elif item.quadrant == "core_auto":
            core.append(_core_auto_paper(brief, item.candidate))
        else:
            medium.append(item.candidate)

    conceptual, llm_warnings = _llm_enrich_medium(brief, medium)

    return LiteratureReviewOutput(
        brief_title=brief.title,
        core_findings=core,
        conceptual_references=conceptual,
        seminal_literature=seminal,
        candidate_count=len(candidates),
        discarded_count=len(discarded),
        alignment_thresholds=threshold_meta,
        warnings=[*align_warnings, *llm_warnings],
    )


async def run_systematic_review(
    brief: ResearchBrief,
    *,
    limit: int | None = None,
    chapter_text: str | None = None,
) -> LiteratureReviewOutput:
    """Ejecuta el embudo completo y retorna output Writer-ready."""
    if limit is not None:
        brief = brief.model_copy(update={"max_results": max(1, min(int(limit), 50))})

    warnings: list[str] = []
    logger.info("Building search matrix with Groq for brief=%r", brief.title)
    matrix = build_search_matrix(brief)

    logger.info("Running dual search with %s queries", len(matrix.queries))
    df, sources_used, search_warnings = await run_dual_search(matrix, brief)
    warnings.extend(search_warnings)

    # Inyectar DOIs fundacionales ausentes en la búsqueda dual.
    if brief.seminal_dois:
        existing = {
            normalize_doi(str(d))
            for d in (df["doi"].tolist() if not df.empty and "doi" in df.columns else [])
            if normalize_doi(str(d))
        }
        missing = [
            d
            for d in brief.seminal_dois
            if normalize_doi(d) and normalize_doi(d) not in existing
        ]
        if missing:
            logger.info("Resolving %s missing seminal DOI(s)", len(missing))
            seminal_rows, seminal_warnings = await lit.fetch_rows_by_dois(missing)
            warnings.extend(seminal_warnings)
            if seminal_rows:
                df = pd.concat([df, pd.DataFrame(seminal_rows)], ignore_index=True)
                if any(r.get("has_s2") for r in seminal_rows):
                    if "semantic_scholar" not in sources_used:
                        sources_used.append("semantic_scholar")
                if any(r.get("has_openalex") for r in seminal_rows):
                    if "openalex" not in sources_used:
                        sources_used.append("openalex")

    ranked = rank_candidates(
        df,
        years_back=brief.years_back,
        limit=brief.max_results,
        seminal_dois=brief.seminal_dois,
    )
    candidates = _candidates_for_llm(ranked)

    logger.info("Downloading/parsing OA PDFs for %s candidates", len(candidates))
    candidates, pdf_warnings = await enrich_candidates_with_pdf_text(candidates)
    warnings.extend(pdf_warnings)

    logger.info(
        "Hybrid alignment + classification for %s candidates", len(candidates)
    )
    output = classify_papers(
        brief, candidates, matrix=matrix, chapter_text=chapter_text
    )
    output.search_matrix = matrix
    output.primary_sources_used = sources_used
    output.warnings = [*warnings, *output.warnings]
    output.candidate_count = len(candidates)
    return output


def review_to_markdown(output: LiteratureReviewOutput) -> str:
    """Renderiza el review clasificado en Markdown para el Redactor."""
    thr = output.alignment_thresholds or {}
    thr_txt = (
        f"high>={thr.get('high', '?')} / low<{thr.get('low', '?')}"
        if thr
        else "n/a"
    )
    lines = [
        f"# Literature Review: {output.brief_title}",
        "",
        f"Candidates screened: {output.candidate_count}",
        f"Discarded (low alignment): {output.discarded_count}",
        f"Seminal literature: {len(output.seminal_literature)}",
        f"Alignment thresholds (TF-IDF cosine): {thr_txt}",
        f"Sources: {', '.join(output.primary_sources_used) or 'n/a'}",
        "",
        "## Core Findings (Nivel 1 — utility 1.0, high alignment_score)",
        "",
        "_Hallazgos empíricos modernos. No incluir aquí literatura fundacional._",
        "",
    ]
    if not output.core_findings:
        lines.append("_No core papers classified._")
        lines.append("")
    for i, p in enumerate(output.core_findings, 1):
        lines.extend(
            [
                f"### {i}. {p.title or '(untitled)'}",
                f"- DOI: `{p.doi}`",
                (
                    f"- Year: {p.year} | Citations: {p.citation_count} | "
                    f"Utility: {p.utility} | alignment_score: {p.alignment_score:.4f}"
                    f" ({p.alignment_quadrant or 'n/a'})"
                ),
                f"- PDF: {p.pdf_url or '—'}",
                (
                    f"- PDF local: {p.pdf_local_path or '—'} "
                    f"({p.pdf_parse_status or 'n/a'}, {p.full_text_chars} chars)"
                ),
                f"- Suggested section: {p.suggested_section or '—'}",
                f"- Citation context: {p.citation_context}",
                "- Key findings:",
            ]
        )
        for finding in p.key_findings or ["—"]:
            lines.append(f"  - {finding}")
        lines.append("")

    lines.extend(
        [
            "## Conceptual References (Nivel 2 — utility 0.5, mid alignment → LLM)",
            "",
        ]
    )
    if not output.conceptual_references:
        lines.append("_No conceptual references classified._")
        lines.append("")
    for i, p in enumerate(output.conceptual_references, 1):
        lines.extend(
            [
                f"### {i}. {p.title or '(untitled)'}",
                f"- DOI: `{p.doi}`",
                (
                    f"- Year: {p.year} | Citations: {p.citation_count} | "
                    f"Utility: {p.utility} | alignment_score: {p.alignment_score:.4f}"
                    f" ({p.alignment_quadrant or 'n/a'})"
                ),
                f"- Suggested section: {p.suggested_section or '—'}",
                f"- Citation context: {p.citation_context}",
                "",
            ]
        )

    lines.extend(
        [
            "## Fundamentos Teóricos e Históricos (Seminal Literature)",
            "",
            (
                "_Usar estrictamente para definir el origen del modelo o concepto "
                "en el marco teórico. No mezclar con hallazgos empíricos (Nivel 1)._"
            ),
            "",
        ]
    )
    if not output.seminal_literature:
        lines.append("_No seminal / foundational papers classified._")
        lines.append("")
    for i, p in enumerate(output.seminal_literature, 1):
        lines.extend(
            [
                f"### {i}. {p.title or '(untitled)'}",
                f"- DOI: `{p.doi}`",
                (
                    f"- Year: {p.year} | Citations: {p.citation_count} | "
                    f"Utility: {p.utility} | alignment_score: {p.alignment_score:.4f}"
                    f" | historical_centrality: {p.historical_centrality:.2f} cites/year"
                ),
                f"- Seminal reason: {p.seminal_reason or 'n/a'}",
                f"- Suggested section: {p.suggested_section or '—'}",
                f"- Citation context: {p.citation_context}",
                "- Key findings:",
            ]
        )
        for finding in p.key_findings or ["—"]:
            lines.append(f"  - {finding}")
        lines.append("")

    if output.warnings:
        lines.extend(["## Warnings", ""])
        for w in output.warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


@tool("Run Systematic Literature Review")
def run_systematic_literature_review(research_brief_json: str) -> str:
    """Run a systematic literature funnel for the Academic Writer.

    Accepts a ResearchBrief as JSON string (title, objectives, research_questions,
    variables, methodology, domain, years_back, max_results, optional seminal_dois,
    optional chapter_text). Builds a Groq search matrix, queries Semantic
    Scholar/OpenAlex, enriches OA PDFs via Unpaywall, ranks by citations/recency
    while protecting foundational DOIs, then applies hybrid TF-IDF cosine alignment
    plus historical centrality: high → Core, mid → Groq conceptual, low → discard,
    whitelist/high historical impact → Seminal Literature (theoretical framework).

    Args:
        research_brief_json: JSON object matching the ResearchBrief schema.
            Optional keys: ``chapter_text``, ``seminal_dois`` (list of DOIs).

    Returns:
        Writer-ready JSON with core_findings, conceptual_references,
        seminal_literature, discarded_count, alignment_thresholds, search_matrix.
    """
    try:
        raw = json.loads(research_brief_json)
        chapter_text = raw.pop("chapter_text", None)
        brief = ResearchBrief.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "brief_title": "",
                "core_findings": [],
                "conceptual_references": [],
                "seminal_literature": [],
                "discarded_count": 0,
                "warnings": [f"Invalid ResearchBrief JSON: {exc}"],
            },
            ensure_ascii=False,
            indent=2,
        )

    try:
        output = asyncio.run(
            run_systematic_review(
                brief,
                chapter_text=str(chapter_text) if chapter_text else None,
            )
        )
        return json.dumps(output.to_dict(), ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.exception("systematic literature review failed")
        return json.dumps(
            {
                "brief_title": brief.title,
                "core_findings": [],
                "conceptual_references": [],
                "seminal_literature": [],
                "discarded_count": 0,
                "warnings": [f"Unexpected error: {exc}"],
            },
            ensure_ascii=False,
            indent=2,
        )
