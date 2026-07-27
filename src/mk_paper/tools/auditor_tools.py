"""Tools del Quality Auditor Q1-Q2: rigor, prosa, feedback loop Writer."""

from __future__ import annotations

import json
import logging
import re
import traceback
from typing import Any

from crewai.tools import tool

from mk_paper.config.llm import get_llm
from mk_paper.config.settings import Settings, get_settings
from mk_paper.models.audit_brief import (
    AuditBrief,
    AuditFeedback,
    AuditFinding,
    AuditVerdict,
    DimensionScores,
)
from mk_paper.models.method_brief import AnalysisReport
from mk_paper.models.writing_brief import CitationCatalog, WritingBrief
from mk_paper.tools.analysis_tools import resolve_sandbox_path
from mk_paper.tools.writer_tools import (
    _CITE_CLUSTER_RE,
    _CITE_KEY_RE,
    _REF_SECTION_RE,
    _strip_references_section,
    build_citation_catalog,
    draft_imrad_paper,
    extract_pandoc_cite_keys,
    load_writing_inputs,
    markdown_to_latex,
    revise_imrad_with_feedback,
    validate_draft_citations,
)

logger = logging.getLogger(__name__)

_BULLET_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+\S")
_HYPE_RE = re.compile(
    r"\b("
    r"breakthrough|groundbreaking|revolutionary|game[- ]?changer|unprecedented|"
    r"amazing|awesome|perfect|guaranteed|best[- ]in[- ]class|cutting[- ]edge|"
    r"paradigm[- ]shifting|world[- ]class|unparalleled|miracle|"
    r"revolucionari[oa]|sin\s+precedentes|impresionante|garantizad[oa]|"
    r"mejor\s+del\s+mundo|disruptiv[oa]|incre[ií]ble"
    r")\b",
    re.IGNORECASE,
)
_EMPIRICAL_CLAIM_RE = re.compile(
    r"\b("
    r"demuestra|demostramos|prueba|probamos|confirma|significativ|"
    r"superior|outperform|proves|demonstrates|significantly|"
    r"evidencia\s+que|shows\s+that|we\s+find"
    r")\b",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(r"(?m)^(#{1,3})\s+(.+)$")


def _split_sections(markdown: str) -> dict[str, str]:
    """Parte el markdown por headings ## / ###."""
    body = _strip_references_section(markdown)
    sections: dict[str, str] = {"_preamble": ""}
    current = "_preamble"
    buf: list[str] = []
    for line in body.splitlines():
        m = _SECTION_RE.match(line)
        if m and len(m.group(1)) <= 3:
            sections[current] = "\n".join(buf).strip()
            current = m.group(2).strip().lower()
            buf = []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf).strip()
    return sections


def _in_code_or_table(line: str, in_fence: bool) -> tuple[bool, bool]:
    stripped = line.strip()
    if stripped.startswith("```"):
        return (not in_fence), True
    if in_fence or stripped.startswith("|"):
        return in_fence, True
    return in_fence, False


def detect_structural_bullets(markdown: str) -> list[AuditFinding]:
    """Prohíbe viñetas en prosa estructural (objetivos, método, etc.)."""
    findings: list[AuditFinding] = []
    structural_hints = (
        "introduc",
        "introduction",
        "metodolog",
        "method",
        "discusi",
        "discussion",
        "objetivo",
        "objective",
        "pregunta",
        "research question",
        "justific",
        "vacío",
        "gap",
        "marco",
    )
    sections = _split_sections(markdown)
    for name, text in sections.items():
        if name.startswith("_"):
            continue
        if not any(h in name for h in structural_hints):
            # También escanear preámbulo largo si parece intro
            continue
        in_fence = False
        for i, line in enumerate(text.splitlines(), start=1):
            in_fence, skip = _in_code_or_table(line, in_fence)
            if skip:
                continue
            if _BULLET_RE.match(line):
                findings.append(
                    AuditFinding(
                        category="structure_bullets",
                        severity="critical",
                        section=name,
                        message=(
                            f"Viñeta/lista numerada en sección estructural "
                            f"'{name}' (línea local {i}): prosa continua requerida."
                        ),
                        suggested_fix=(
                            "Reescribe objetivos, preguntas y justificación en "
                            "párrafos académicos fluidos, sin bullets."
                        ),
                    )
                )
    # Barrido global en cuerpo (excepto tablas/código/referencias)
    in_fence = False
    for i, line in enumerate(_strip_references_section(markdown).splitlines(), start=1):
        in_fence, skip = _in_code_or_table(line, in_fence)
        if skip or _SECTION_RE.match(line):
            continue
        if _BULLET_RE.match(line):
            # Evitar duplicar si ya capturado por sección
            if any(f.section and str(i) in f.message for f in findings):
                continue
            findings.append(
                AuditFinding(
                    category="structure_bullets",
                    severity="critical",
                    section="body",
                    message=f"Lista con viñetas en prosa del manuscrito (línea {i}).",
                    suggested_fix="Convertir a prosa académica continua.",
                )
            )
            if len(findings) >= 12:
                break
    return findings


def detect_hype_tone(markdown: str) -> list[AuditFinding]:
    """Rechaza lenguaje comercial / exagerado."""
    findings: list[AuditFinding] = []
    body = _strip_references_section(markdown)
    for m in _HYPE_RE.finditer(body):
        span = body[max(0, m.start() - 40) : m.end() + 40].replace("\n", " ")
        findings.append(
            AuditFinding(
                category="tone",
                severity="major",
                section="tone",
                message=f"Lenguaje no académico / hype detectado: '{m.group(0)}' … {span}",
                suggested_fix=(
                    "Sustituir por formulación mesurada (p.ej. 'los resultados "
                    "sugieren', 'en esta muestra')."
                ),
            )
        )
        if len(findings) >= 8:
            break
    return findings


def audit_citations(
    markdown: str,
    catalog: CitationCatalog,
) -> list[AuditFinding]:
    """Valida protocolo Pandoc y ausencia de keys desconocidas."""
    findings: list[AuditFinding] = []
    body = _strip_references_section(markdown)
    pandoc_keys = extract_pandoc_cite_keys(body)
    validation = validate_draft_citations(body, catalog)
    if validation.citations_unknown:
        findings.append(
            AuditFinding(
                category="citations",
                severity="critical",
                section="citations",
                message=(
                    "cite_keys Pandoc desconocidos (no están en catálogo): "
                    + ", ".join(validation.citations_unknown)
                ),
                suggested_fix=(
                    "Eliminar keys inventadas; usar únicamente [@cite_key] del catálogo."
                ),
            )
        )

    # Tras expansión APA no quedan [@key]; exigir que las formas APA del catálogo
    # aparezcan al menos en Intro/Discusión si el catálogo no está vacío.
    has_pandoc = bool(pandoc_keys) or bool(_CITE_CLUSTER_RE.search(body))
    apa_hits = 0
    for entry in catalog.entries:
        if entry.apa_parenthetical and entry.apa_parenthetical in body:
            apa_hits += 1
        elif entry.authors and entry.year is not None:
            surname = entry.authors[0].split()[-1]
            if f"{surname}, {entry.year}" in body or f"{surname} ({entry.year})" in body:
                apa_hits += 1

    if catalog.entries and not has_pandoc and apa_hits == 0:
        findings.append(
            AuditFinding(
                category="citations",
                severity="critical",
                section="citations",
                message=(
                    "No se detectaron citas Pandoc [@cite_key] ni formas APA del "
                    "catálogo en el cuerpo del manuscrito."
                ),
                suggested_fix=(
                    "Insertar citas [@cite_key] (seminal/conceptual en Intro; "
                    "core en Discusión) antes de la exportación APA."
                ),
            )
        )

    # Claves sueltas mal formadas tipo @key sin corchetes
    loose = re.findall(r"(?<!\[)@([\w-]+)(?!\w)", body)
    loose = [k for k in loose if k not in {"TABLE_MODEL_COMPARISON"}]
    if loose:
        findings.append(
            AuditFinding(
                category="citations",
                severity="major",
                section="citations",
                message=f"Citas Pandoc mal formadas (falta [@...]): {sorted(set(loose))[:8]}",
                suggested_fix="Usar estrictamente el formato [@cite_key].",
            )
        )
    return findings


def detect_orphan_claims(markdown: str) -> list[AuditFinding]:
    """Señala afirmaciones empíricas fuertes sin cita cercana ni número."""
    findings: list[AuditFinding] = []
    body = _strip_references_section(markdown)
    # Oraciones aproximadas
    sentences = re.split(r"(?<=[.!?])\s+", body)
    for sent in sentences:
        if not _EMPIRICAL_CLAIM_RE.search(sent):
            continue
        if len(sent) < 40:
            continue
        has_cite = bool(_CITE_KEY_RE.search(sent) or _CITE_CLUSTER_RE.search(sent))
        has_apa_year = bool(re.search(r"\((?:19|20)\d{2}[a-z]?", sent))
        has_number = bool(re.search(r"\d+\.\d+|\b\d{2,}\b", sent))
        if has_cite or has_apa_year or has_number:
            continue
        findings.append(
            AuditFinding(
                category="orphan_claims",
                severity="major",
                section="claims",
                message=(
                    "Afirmación empírica sin respaldo bibliográfico/estadístico "
                    f"cercano: «{sent[:180]}…»"
                ),
                suggested_fix=(
                    "Añadir [@cite_key], métrica del AnalysisReport, o atenuar el tono."
                ),
            )
        )
        if len(findings) >= 6:
            break
    return findings


def audit_coherence(
    markdown: str,
    report: AnalysisReport,
) -> list[AuditFinding]:
    """Alinea problema / modelo cuantitativo / conclusiones con el reporte."""
    findings: list[AuditFinding] = []
    body = _strip_references_section(markdown).lower()
    best = (report.best_model or "").lower()
    metric = (report.primary_metric or "").lower()

    if best and best not in body and best.replace("_", " ") not in body:
        findings.append(
            AuditFinding(
                category="coherence",
                severity="critical",
                section="results/discussion",
                message=(
                    f"El mejor modelo del AnalysisReport (`{report.best_model}`) "
                    "no aparece en el manuscrito."
                ),
                suggested_fix=(
                    "Mencionar explícitamente el modelo ganador y su métrica de prueba "
                    "en Resultados/Discusión."
                ),
            )
        )
    if metric and metric not in body:
        findings.append(
            AuditFinding(
                category="coherence",
                severity="major",
                section="results",
                message=(
                    f"La métrica primaria `{report.primary_metric}` no se discute "
                    "en el texto."
                ),
                suggested_fix="Anclar la discusión a la métrica primaria reportada.",
            )
        )

    if report.best_score is not None:
        # Exigir que alguna representación del score aparezca (tolerante)
        score_str = f"{report.best_score:.4f}"
        alt = f"{report.best_score:.2f}"
        if score_str not in markdown and alt not in markdown and str(report.best_score) not in markdown:
            findings.append(
                AuditFinding(
                    category="coherence",
                    severity="minor",
                    section="results",
                    message=(
                        "El score del mejor modelo no aparece de forma identificable "
                        "en el manuscrito (posible desalineación resultados–conclusiones)."
                    ),
                    suggested_fix=(
                        "Conservar las tablas literales del skeleton y referir el score "
                        "en prosa sin redondeos inventados."
                    ),
                )
            )

    sections = _split_sections(markdown)
    has_intro = any("introduc" in k or "introduction" in k for k in sections)
    has_method = any("metod" in k or "method" in k for k in sections)
    has_results = any("result" in k for k in sections)
    has_disc = any("discusi" in k or "discussion" in k for k in sections)
    for flag, label in (
        (has_intro, "Introducción"),
        (has_method, "Metodología"),
        (has_results, "Resultados"),
        (has_disc, "Discusión"),
    ):
        if not flag:
            findings.append(
                AuditFinding(
                    category="coherence",
                    severity="critical",
                    section="structure",
                    message=f"Falta la sección IMRaD obligatoria: {label}.",
                    suggested_fix=f"Restaurar la sección «{label}» en prosa continua.",
                )
            )
    return findings


def _score_from_findings(findings: list[AuditFinding]) -> tuple[float, DimensionScores]:
    dims = DimensionScores()
    penalties = {
        "structure_bullets": ("prose_structure", 2.5, 1.2),
        "tone": ("tone", 1.5, 0.8),
        "citations": ("citations", 2.0, 1.0),
        "coherence": ("coherence", 2.0, 1.0),
        "orphan_claims": ("empirical_support", 1.5, 0.7),
        "methodology": ("coherence", 1.5, 0.8),
        "results": ("empirical_support", 1.5, 0.8),
        "other": ("coherence", 1.0, 0.5),
    }
    for f in findings:
        attr, crit, maj = penalties.get(f.category, ("coherence", 1.0, 0.5))
        hit = crit if f.severity == "critical" else maj if f.severity == "major" else 0.4
        current = getattr(dims, attr)
        setattr(dims, attr, max(0.0, current - hit))

    overall = (
        dims.prose_structure
        + dims.tone
        + dims.citations
        + dims.coherence
        + dims.empirical_support
    ) / 5.0
    return round(overall, 2), dims


def _build_revision_instructions(
    findings: list[AuditFinding],
    *,
    language: str,
) -> str:
    if not findings:
        return (
            "Mantener prosa continua, mesura analítica y alineación método–resultados."
            if language == "es"
            else "Maintain continuous prose, analytical restraint, and method–results alignment."
        )
    lines = []
    for i, f in enumerate(findings, start=1):
        lines.append(
            f"{i}. [{f.severity}/{f.category}] {f.section}: {f.message} "
            f"→ Fix: {f.suggested_fix}"
        )
    header = (
        "Aplique las siguientes correcciones de forma integral en la siguiente pasada "
        "del Scientific Writer:\n"
        if language == "es"
        else "Apply the following corrections comprehensively in the next Writer pass:\n"
    )
    return header + "\n".join(lines)


def evaluate_manuscript(
    markdown: str,
    *,
    catalog: CitationCatalog,
    report: AnalysisReport,
    threshold: float = 8.5,
    language: str = "es",
    use_llm: bool = False,
) -> AuditFeedback:
    """Evalúa el manuscrito (checks deterministas + LLM opcional de coherencia)."""
    findings: list[AuditFinding] = []
    findings.extend(detect_structural_bullets(markdown))
    findings.extend(detect_hype_tone(markdown))
    findings.extend(audit_citations(markdown, catalog))
    findings.extend(detect_orphan_claims(markdown))
    findings.extend(audit_coherence(markdown, report))

    if use_llm:
        try:
            findings.extend(
                _llm_coherence_findings(markdown, report, language=language)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM coherence audit skipped: %s", exc)

    overall, dims = _score_from_findings(findings)
    decision: str
    if overall >= threshold and not any(f.severity == "critical" for f in findings):
        decision = "accept"
    elif overall < max(5.0, threshold - 3.0) and sum(
        1 for f in findings if f.severity == "critical"
    ) >= 3:
        decision = "reject"
    else:
        decision = "revise"

    # Critical findings force revise even if score high
    if any(f.severity == "critical" for f in findings) and decision == "accept":
        decision = "revise"
        overall = min(overall, threshold - 0.1)

    summary = (
        f"Puntaje {overall}/10 (umbral {threshold}). Hallazgos: {len(findings)}. "
        f"Decisión provisional: {decision}."
    )
    return AuditFeedback(
        overall_score=overall,
        threshold=threshold,
        decision=decision,  # type: ignore[arg-type]
        dimension_scores=dims,
        findings=findings,
        revision_instructions=_build_revision_instructions(
            findings, language=language
        ),
        summary=summary,
    )


def _llm_coherence_findings(
    markdown: str,
    report: AnalysisReport,
    *,
    language: str,
) -> list[AuditFinding]:
    prompt = f"""You are a Q1 journal editor (finance/accounting/data science).
Audit coherence between problem, quantitative model, and conclusions.

Return ONLY JSON:
{{"findings":[{{"category":"coherence|tone|methodology|results|other",
"severity":"critical|major|minor","section":"...","message":"...","suggested_fix":"..."}}]}}

Rules:
- Do not invent missing numbers; use the analysis facts below.
- Flag commercial/hype tone.
- Flag misalignment problem↔model↔conclusions.
- Max 5 findings. Empty list if none.

ANALYSIS FACTS:
best_model={report.best_model}
primary_metric={report.primary_metric}
best_score={report.best_score}
task_type={report.task_type}

MANUSCRIPT (truncated):
{_strip_references_section(markdown)[:10000]}
"""
    llm = get_llm()
    raw = str(llm.call(prompt)).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.I)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return []
    data = json.loads(raw[start : end + 1])
    out: list[AuditFinding] = []
    for item in data.get("findings") or []:
        try:
            out.append(AuditFinding.model_validate(item))
        except Exception:  # noqa: BLE001
            continue
    return out[:5]


def polish_manuscript(markdown: str) -> str:
    """Pulido tipográfico ligero previo a exportación LaTeX/PDF."""
    text = (markdown or "").replace("\r\n", "\n")
    # Colapsar 3+ newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Asegurar heading de referencias
    if not _REF_SECTION_RE.search(text):
        text = text.rstrip() + "\n"
    return text.strip() + "\n"


def _resolve_writing_brief(
    audit: AuditBrief,
    *,
    settings: Settings,
) -> WritingBrief | None:
    if audit.writing_brief_path:
        path = resolve_sandbox_path(
            audit.writing_brief_path, settings=settings, must_exist=True
        )
        return WritingBrief.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    if audit.literature_review_path and audit.analysis_report_path:
        return WritingBrief(
            title=audit.title,
            literature_review_path=audit.literature_review_path,
            analysis_report_path=audit.analysis_report_path,
            language=audit.language,
            include_latex=audit.include_latex,
        )
    return None


def load_audit_context(
    audit: AuditBrief,
    *,
    settings: Settings | None = None,
) -> tuple[str, CitationCatalog, AnalysisReport, WritingBrief | None, list[str]]:
    """Carga markdown, catálogo y AnalysisReport para auditar."""
    cfg = settings or get_settings()
    warnings: list[str] = []
    writing = _resolve_writing_brief(audit, settings=cfg)

    markdown = ""
    if audit.paper_draft_path:
        md_path = resolve_sandbox_path(
            audit.paper_draft_path, settings=cfg, must_exist=True
        )
        markdown = md_path.read_text(encoding="utf-8")
    elif writing is not None:
        draft, _ = draft_imrad_paper(
            writing, settings=cfg, use_llm=audit.use_llm
        )
        markdown = draft.markdown
        warnings.extend(draft.warnings)
    else:
        raise ValueError(
            "AuditBrief requiere paper_draft_path o writing_brief_path "
            "(o literature+analysis)."
        )

    # Paths de evidencia
    lit_path = audit.literature_review_path
    ana_path = audit.analysis_report_path
    if writing is not None:
        lit_path = lit_path or writing.literature_review_path
        ana_path = ana_path or writing.analysis_report_path
    if not lit_path or not ana_path:
        # Intentar inferir desde paper_draft.json hermano
        raise ValueError(
            "Se requieren literature_review_path y analysis_report_path "
            "(directos o vía writing_brief)."
        )

    # Reusar WritingBrief temporal para loaders
    tmp_brief = WritingBrief(
        title=audit.title,
        literature_review_path=lit_path,
        analysis_report_path=ana_path,
        language=audit.language,
        include_latex=audit.include_latex,
    )
    review, report, load_warns = load_writing_inputs(tmp_brief, settings=cfg)
    warnings.extend(load_warns)
    catalog = build_citation_catalog(review)
    warnings.extend(catalog.warnings)
    return markdown, catalog, report, writing, warnings


def run_quality_audit(
    audit: AuditBrief,
    *,
    settings: Settings | None = None,
) -> AuditVerdict:
    """Auditoría + bucle de feedback Writer cuando score < umbral."""
    cfg = settings or get_settings()
    warnings: list[str] = []
    history: list[AuditFeedback] = []

    markdown, catalog, report, writing, load_warns = load_audit_context(
        audit, settings=cfg
    )
    warnings.extend(load_warns)

    feedback = evaluate_manuscript(
        markdown,
        catalog=catalog,
        report=report,
        threshold=audit.quality_threshold,
        language=audit.language,
        use_llm=audit.use_llm,
    )
    history.append(feedback)
    rounds = 0

    while (
        feedback.decision != "accept"
        and feedback.overall_score < audit.quality_threshold
        and rounds < audit.max_revision_rounds
        and writing is not None
    ):
        rounds += 1
        logger.info(
            "Audit feedback loop round=%s score=%s threshold=%s",
            rounds,
            feedback.overall_score,
            audit.quality_threshold,
        )
        revised, catalog = revise_imrad_with_feedback(
            writing,
            feedback,
            markdown,
            settings=cfg,
            use_llm=audit.use_llm,
        )
        warnings.extend(revised.warnings)
        markdown = revised.markdown
        feedback = evaluate_manuscript(
            markdown,
            catalog=catalog,
            report=report,
            threshold=audit.quality_threshold,
            language=audit.language,
            use_llm=audit.use_llm,
        )
        history.append(feedback)

    polished = polish_manuscript(markdown)
    latex = (
        markdown_to_latex(polished, title=audit.title) if audit.include_latex else None
    )

    # Decisión final
    decision = feedback.decision
    if (
        feedback.overall_score >= audit.quality_threshold
        and not any(f.severity == "critical" for f in feedback.findings)
    ):
        decision = "accept"
        feedback.decision = "accept"
    elif rounds >= audit.max_revision_rounds and decision != "accept":
        # Agotó revisiones sin alcanzar umbral
        if feedback.overall_score >= audit.quality_threshold - 0.5:
            decision = "revise"
        else:
            decision = "reject"
        feedback.decision = decision

    lit = ""
    ana = ""
    if writing is not None:
        lit = writing.literature_review_path
        ana = writing.analysis_report_path
    else:
        lit = audit.literature_review_path or ""
        ana = audit.analysis_report_path or ""

    status: str = "ok" if decision == "accept" else "partial"
    if decision == "reject":
        status = "error"

    return AuditVerdict(
        title=audit.title,
        decision=decision,  # type: ignore[arg-type]
        overall_score=feedback.overall_score,
        threshold=audit.quality_threshold,
        rounds_completed=rounds,
        feedback=feedback,
        feedback_history=history,
        polished_markdown=polished,
        latex=latex,
        paper_draft_path=audit.paper_draft_path or "",
        literature_path=lit,
        analysis_path=ana,
        warnings=warnings,
        status=status,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# CrewAI tools
# ---------------------------------------------------------------------------


@tool("Run Structural Quality Checks")
def run_structural_quality_checks_tool(
    markdown: str,
    writing_brief_json: str = "",
) -> str:
    """Deterministic Q1 checks: no structural bullets, tone, Pandoc citations.

    Args:
        markdown: IMRaD markdown to audit.
        writing_brief_json: Optional WritingBrief to rebuild citation catalog.

    Returns:
        JSON with findings and preliminary dimension scores.
    """
    try:
        catalog = CitationCatalog()
        report = AnalysisReport(
            brief_title="unknown",
            task_type="regression",
            dataset_path="unknown",
            primary_metric="rmse",
        )
        if writing_brief_json.strip():
            brief = WritingBrief.model_validate(json.loads(writing_brief_json))
            review, report, _ = load_writing_inputs(brief)
            catalog = build_citation_catalog(review)
        findings: list[AuditFinding] = []
        findings.extend(detect_structural_bullets(markdown))
        findings.extend(detect_hype_tone(markdown))
        findings.extend(audit_citations(markdown, catalog))
        findings.extend(detect_orphan_claims(markdown))
        if report.dataset_path != "unknown":
            findings.extend(audit_coherence(markdown, report))
        score, dims = _score_from_findings(findings)
        return json.dumps(
            {
                "status": "ok",
                "overall_score": score,
                "dimension_scores": dims.model_dump(),
                "findings": [f.model_dump() for f in findings],
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )


@tool("Evaluate Manuscript Quality")
def evaluate_manuscript_quality_tool(audit_brief_json: str) -> str:
    """Full quality evaluation producing structured Audit Feedback.

    Args:
        audit_brief_json: AuditBrief JSON (paths + quality_threshold).

    Returns:
        JSON AuditFeedback (score, findings, revision_instructions).
    """
    try:
        audit = AuditBrief.model_validate(json.loads(audit_brief_json))
        markdown, catalog, report, _, warnings = load_audit_context(audit)
        feedback = evaluate_manuscript(
            markdown,
            catalog=catalog,
            report=report,
            threshold=audit.quality_threshold,
            language=audit.language,
            use_llm=audit.use_llm,
        )
        payload = feedback.model_dump(mode="json")
        payload["status"] = "ok"
        payload["warnings"] = warnings
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )


@tool("Run Audit With Writer Feedback Loop")
def run_audit_with_feedback_loop_tool(audit_brief_json: str) -> str:
    """Audit manuscript; if score < threshold, send AuditFeedback to Writer and iterate.

    Args:
        audit_brief_json: AuditBrief with draft/writing paths and threshold (default 8.5).

    Returns:
        JSON AuditVerdict including polished_markdown and decision accept|revise|reject.
    """
    try:
        audit = AuditBrief.model_validate(json.loads(audit_brief_json))
        verdict = run_quality_audit(audit)
        return json.dumps(verdict.to_dict(), ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.exception("audit feedback loop failed")
        return json.dumps(
            {
                "status": "error",
                "decision": "reject",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )


@tool("Polish Final Manuscript")
def polish_final_manuscript_tool(markdown: str, title: str = "Manuscript") -> str:
    """Polish markdown and produce basic LaTeX for publication export.

    Args:
        markdown: Accepted IMRaD markdown.
        title: Title for LaTeX.

    Returns:
        JSON with polished markdown and latex.
    """
    try:
        polished = polish_manuscript(markdown)
        latex = markdown_to_latex(polished, title=title)
        return json.dumps(
            {
                "status": "ok",
                "markdown": polished,
                "latex": latex,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )
