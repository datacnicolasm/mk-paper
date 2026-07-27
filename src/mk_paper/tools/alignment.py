"""Alineación temática híbrida: TF-IDF + similitud de coseno + seminalidad.

Vectoriza el perfil de investigación y los metadatos de papers en el mismo
espacio TF-IDF (uni/bi-gramas), calcula ``alignment_score`` ∈ [0, 1] y aplica
excepciones fundacionales (whitelist DOI + centralidad histórica de citas).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mk_paper.config.settings import Settings, get_settings
from mk_paper.models.research_brief import ResearchBrief, SearchMatrix

logger = logging.getLogger(__name__)

Quadrant = Literal["core_auto", "llm_review", "discard", "seminal"]

_WS_RE = re.compile(r"\s+")
_DOI_PREFIX_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)


@dataclass(frozen=True)
class AlignmentThresholds:
    """Umbrales del filtro cuadrante."""

    high: float = 0.35
    low: float = 0.12


@dataclass(frozen=True)
class SeminalThresholds:
    """Umbrales de literatura fundacional / seminal."""

    min_age_years: int = 10
    cites_per_year: float = 50.0
    min_citations: int = 500
    alignment_floor: float = 0.04
    history_boost: float = 0.15


@dataclass
class ScoredCandidate:
    """Candidato con score de alineación matemática."""

    candidate: dict[str, Any]
    alignment_score: float
    quadrant: Quadrant


def normalize_doi(doi: str | None) -> str | None:
    """Normaliza DOI a forma canónica minúscula."""
    if not doi or not isinstance(doi, str):
        return None
    cleaned = _DOI_PREFIX_RE.sub("", doi.strip())
    cleaned = cleaned.lower().rstrip(".")
    if not cleaned or "/" not in cleaned:
        return None
    return cleaned


def normalize_doi_set(dois: list[str] | None) -> set[str]:
    """Convierte lista de DOIs a set normalizado."""
    out: set[str] = set()
    for item in dois or []:
        doi = normalize_doi(str(item))
        if doi:
            out.add(doi)
    return out


def brief_to_profile_text(
    brief: ResearchBrief,
    matrix: SearchMatrix | None = None,
    chapter_text: str | None = None,
) -> str:
    """Construye el texto base del perfil/capítulo a vectorizar."""
    parts: list[str] = []
    if chapter_text and chapter_text.strip():
        parts.append(chapter_text.strip())

    parts.append(brief.title)
    if brief.domain:
        parts.append(brief.domain)
    if brief.methodology:
        parts.append(brief.methodology)
    parts.extend(brief.objectives or [])
    parts.extend(brief.research_questions or [])
    for values in (brief.variables or {}).values():
        parts.extend(values)

    if matrix is not None:
        parts.extend(matrix.queries or [])
        parts.extend(matrix.synonyms or [])
        parts.extend(matrix.method_terms or [])

    # Sesgo lexical suave hacia orígenes teóricos cuando hay whitelist.
    if brief.seminal_dois:
        parts.extend(
            [
                "foundational theory",
                "seminal model",
                "classical econometric model",
                "theoretical origin",
            ]
        )

    text = " ".join(str(p) for p in parts if p)
    return _WS_RE.sub(" ", text).strip().lower()


def paper_to_document(paper: dict[str, Any]) -> str:
    """Concatena título + abstract + keywords/venue para vectorizar el paper."""
    parts: list[str] = []
    for key in ("title", "abstract", "venue"):
        value = paper.get(key)
        if value:
            parts.append(str(value))

    keywords = paper.get("keywords") or paper.get("fieldsOfStudy") or []
    if isinstance(keywords, list):
        parts.extend(str(k) for k in keywords if k)
    elif keywords:
        parts.append(str(keywords))

    text = " ".join(parts)
    return _WS_RE.sub(" ", text).strip().lower() or "empty document"


def _current_year() -> int:
    return datetime.now(timezone.utc).year


def paper_age_years(
    year: int | float | None,
    *,
    current_year: int | None = None,
) -> int | None:
    """Años transcurridos desde la publicación (mínimo 1 si hay año válido)."""
    if year is None:
        return None
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    if y <= 0:
        return None
    now = current_year or _current_year()
    return max(1, now - y)


def historical_centrality(
    citation_count: int | float | None,
    year: int | float | None,
    *,
    current_year: int | None = None,
) -> float:
    """Impacto histórico = citas / años desde publicación.

    Métrica de centralidad acumulada relativa a la edad del paper.
    """
    age = paper_age_years(year, current_year=current_year)
    if age is None:
        return 0.0
    try:
        cites = float(citation_count or 0)
    except (TypeError, ValueError):
        cites = 0.0
    return max(0.0, cites) / float(age)


def _resolve_thresholds(settings: Settings | None = None) -> AlignmentThresholds:
    cfg = settings or get_settings()
    high = float(getattr(cfg, "alignment_high_threshold", 0.35))
    low = float(getattr(cfg, "alignment_low_threshold", 0.12))
    if low >= high:
        low = max(0.0, high - 0.1)
    return AlignmentThresholds(high=high, low=low)


def _resolve_seminal_thresholds(
    settings: Settings | None = None,
) -> SeminalThresholds:
    cfg = settings or get_settings()
    return SeminalThresholds(
        min_age_years=int(getattr(cfg, "seminal_min_age_years", 10)),
        cites_per_year=float(getattr(cfg, "seminal_cites_per_year", 50.0)),
        min_citations=int(getattr(cfg, "seminal_min_citations", 500)),
        alignment_floor=float(getattr(cfg, "seminal_alignment_floor", 0.04)),
        history_boost=float(getattr(cfg, "seminal_history_boost", 0.15)),
    )


def is_seminal_by_centrality(
    paper: dict[str, Any],
    *,
    alignment_score: float,
    thresholds: SeminalThresholds,
    current_year: int | None = None,
) -> bool:
    """True si el paper antiguo supera umbral crítico de impacto histórico."""
    age = paper_age_years(paper.get("year"), current_year=current_year)
    if age is None or age < thresholds.min_age_years:
        return False

    cites = int(paper.get("citation_count") or 0)
    centrality = historical_centrality(
        cites, paper.get("year"), current_year=current_year
    )
    impact_ok = (
        centrality >= thresholds.cites_per_year or cites >= thresholds.min_citations
    )
    if not impact_ok:
        return False

    # Exige afinidad temática mínima (salvo whitelist, que se maneja aparte).
    return float(alignment_score) >= thresholds.alignment_floor


def apply_historical_alignment_boost(
    alignment_score: float,
    paper: dict[str, Any],
    *,
    thresholds: SeminalThresholds,
    current_year: int | None = None,
) -> float:
    """Pondera al alza la alineación de papers antiguos con alta centralidad."""
    age = paper_age_years(paper.get("year"), current_year=current_year)
    if age is None or age < thresholds.min_age_years:
        return float(alignment_score)

    centrality = historical_centrality(
        paper.get("citation_count"), paper.get("year"), current_year=current_year
    )
    # Normaliza citas/año hacia [0, 1] con saturación suave.
    denom = max(thresholds.cites_per_year, 1.0)
    hist_norm = float(np.clip(centrality / (denom * 2.0), 0.0, 1.0))
    boosted = float(alignment_score) + thresholds.history_boost * hist_norm
    return float(np.clip(boosted, 0.0, 1.0))


def assign_quadrant(
    score: float,
    *,
    thresholds: AlignmentThresholds,
    has_oa_pdf: bool,
) -> Quadrant:
    """Asigna cuadrante según score y disponibilidad de PDF OA."""
    if score < thresholds.low:
        return "discard"
    if score >= thresholds.high and has_oa_pdf:
        return "core_auto"
    if score >= thresholds.high and not has_oa_pdf:
        return "llm_review"
    return "llm_review"


def compute_alignment_scores(
    profile_text: str,
    papers: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> list[float]:
    """Calcula similitud de coseno TF-IDF entre perfil y cada paper.

    Args:
        profile_text: Texto del capítulo / ResearchBrief.
        papers: Dicts con title/abstract/keywords.
        settings: Umbrales/config opcionales.

    Returns:
        Lista de ``alignment_score`` en [0, 1] alineada con ``papers``.
    """
    if not papers:
        return []

    cfg = settings or get_settings()
    docs = [paper_to_document(p) for p in papers]
    corpus = [profile_text or "research profile", *docs]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        stop_words="english",
        sublinear_tf=True,
        max_features=int(getattr(cfg, "alignment_max_features", 12000)),
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        logger.warning("TF-IDF fit failed; returning zero alignment scores")
        return [0.0] * len(papers)

    profile_vec = matrix[0:1]
    paper_vecs = matrix[1:]
    sims = cosine_similarity(profile_vec, paper_vecs).ravel()
    scores = np.clip(sims.astype(float), 0.0, 1.0)
    return [float(s) for s in scores]


def score_and_route_candidates(
    brief: ResearchBrief,
    candidates: list[dict[str, Any]],
    *,
    matrix: SearchMatrix | None = None,
    chapter_text: str | None = None,
    settings: Settings | None = None,
) -> tuple[list[ScoredCandidate], list[dict[str, Any]], list[str]]:
    """Puntúa candidatos y los enruta a core_auto / llm_review / seminal / discard.

    Returns:
        (scored_kept, discarded, warnings)
    """
    warnings: list[str] = []
    if not candidates:
        return [], [], ["No candidates for alignment scoring."]

    cfg = settings or get_settings()
    profile = brief_to_profile_text(brief, matrix=matrix, chapter_text=chapter_text)
    if len(profile.split()) < 3:
        warnings.append(
            "Research profile text is very short; alignment scores may be unstable."
        )

    raw_scores = compute_alignment_scores(profile, candidates, settings=cfg)
    thresholds = _resolve_thresholds(cfg)
    seminal_thr = _resolve_seminal_thresholds(cfg)
    whitelist = normalize_doi_set(brief.seminal_dois)
    now = _current_year()

    kept: list[ScoredCandidate] = []
    discarded: list[dict[str, Any]] = []
    seminal_count = 0

    for paper, raw_score in zip(candidates, raw_scores, strict=True):
        doi = normalize_doi(paper.get("doi"))
        centrality = historical_centrality(
            paper.get("citation_count"), paper.get("year"), current_year=now
        )
        boosted = apply_historical_alignment_boost(
            raw_score,
            paper,
            thresholds=seminal_thr,
            current_year=now,
        )
        has_oa_pdf = bool(paper.get("pdf_url")) and bool(paper.get("is_oa"))

        seminal_reason: str | None = None
        if doi and doi in whitelist:
            quadrant: Quadrant = "seminal"
            seminal_reason = "whitelist"
            score = boosted
        elif is_seminal_by_centrality(
            paper,
            alignment_score=boosted,
            thresholds=seminal_thr,
            current_year=now,
        ):
            quadrant = "seminal"
            seminal_reason = "historical_centrality"
            score = boosted
        else:
            score = boosted
            quadrant = assign_quadrant(
                score, thresholds=thresholds, has_oa_pdf=has_oa_pdf
            )

        enriched = dict(paper)
        enriched["alignment_score"] = round(float(score), 4)
        enriched["alignment_raw"] = round(float(raw_score), 4)
        enriched["alignment_quadrant"] = quadrant
        enriched["historical_centrality"] = round(float(centrality), 4)
        enriched["seminal_reason"] = seminal_reason

        if quadrant == "discard":
            discarded.append(enriched)
            continue
        if quadrant == "seminal":
            seminal_count += 1
        kept.append(
            ScoredCandidate(
                candidate=enriched,
                alignment_score=float(score),
                quadrant=quadrant,
            )
        )

    warnings.append(
        "Alignment filter "
        f"(TF-IDF cosine + seminal): high>={thresholds.high:.2f} "
        f"low<{thresholds.low:.2f}; "
        f"kept={len(kept)} discarded={len(discarded)} "
        f"core_auto={sum(1 for s in kept if s.quadrant == 'core_auto')} "
        f"llm_review={sum(1 for s in kept if s.quadrant == 'llm_review')} "
        f"seminal={seminal_count} whitelist={len(whitelist)}."
    )
    return kept, discarded, warnings
