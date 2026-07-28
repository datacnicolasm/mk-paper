"""Tools del Quality Auditor Q1-Q2: prosa lit-review, citas, feedback loop Writer."""

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
from mk_paper.models.writing_brief import CitationCatalog, WritingBrief
from mk_paper.runtime import resolve_sandbox_path
from mk_paper.tools.writer_tools import (
    _CITE_CLUSTER_RE,
    _CITE_KEY_RE,
    _REF_SECTION_RE,
    _strip_references_section,
    build_citation_catalog,
    draft_literature_paper,
    extract_pandoc_cite_keys,
    load_writing_inputs,
    markdown_to_latex,
    revise_literature_with_feedback,
    sanitize_manuscript_for_publication,
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
_JSON_FENCE_RE = re.compile(r"```(?:json|python|text)?\s*[\s\S]*?```", re.IGNORECASE)
_METADATA_LEAK_RE = re.compile(
    r"(?i)\b("
    r"brief anal[íi]tico|core=|conceptual=|seminal=|\[core\]|\[seminal\]|"
    r"pipeline|writer guidance|analysisreport|web search|cat[aá]logo interno|"
    r"literature snapshot|auditbrief|writingbrief"
    r")\b"
)
_ENGLISH_LEAK_RE = re.compile(
    r"(?i)\b("
    r"introduction|literature review|theoretical framework|method|results|"
    r"discussion|writer guidance|analyst notes|state of the art|"
    r"research question|references\b"
    r")\b"
)
_RQ_PHRASE_RE = re.compile(
    r"(?i)pregunta\s+de\s+investigaci[oó]n|research\s+question"
)
_INTRO_HEADING_RE = re.compile(r"(?i)introduc")
_LIT_HEADING_RE = re.compile(
    r"(?i)(revisi[oó]n\s+de\s+literatura|marco\s+te[oó]rico|literature\s+review|"
    r"theoretical\s+framework)"
)
_REFS_HEADING_RE = re.compile(r"(?i)^(#{1,3})\s+referencias\b")


def _split_sections(markdown: str) -> dict[str, str]:
    """Parte el markdown por headings ## / ### (sin sección Referencias)."""
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
    """Prohíbe viñetas en prosa estructural (intro, marco teórico, objetivos)."""
    findings: list[AuditFinding] = []
    structural_hints = (
        "introduc",
        "introduction",
        "revisi",
        "literatura",
        "marco",
        "teóric",
        "teoric",
        "objetivo",
        "objective",
        "pregunta",
        "research question",
        "justific",
        "vacío",
        "gap",
    )
    sections = _split_sections(markdown)
    for name, text in sections.items():
        if name.startswith("_"):
            continue
        if not any(h in name for h in structural_hints):
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
                            "Reescribe objetivos, preguntas y marco teórico en "
                            "párrafos académicos fluidos, sin bullets."
                        ),
                    )
                )
    in_fence = False
    for i, line in enumerate(_strip_references_section(markdown).splitlines(), start=1):
        in_fence, skip = _in_code_or_table(line, in_fence)
        if skip or _SECTION_RE.match(line):
            continue
        if _BULLET_RE.match(line):
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


def detect_json_leaks(markdown: str) -> list[AuditFinding]:
    """Detecta fugas de JSON o bloques de código técnico en el manuscrito."""
    findings: list[AuditFinding] = []
    for block in _JSON_FENCE_RE.finditer(markdown or ""):
        snippet = block.group(0)[:180].replace("\n", " ")
        findings.append(
            AuditFinding(
                category="other",
                severity="critical",
                section="manuscript",
                message=f"Se detectó bloque técnico no publicable: {snippet}…",
                suggested_fix=(
                    "Eliminar bloques JSON/código y convertir su contenido en "
                    "prosa académica en español."
                ),
            )
        )
        if len(findings) >= 5:
            break
    return findings


def detect_internal_metadata_leaks(markdown: str) -> list[AuditFinding]:
    """Detecta metadatos internos y jerga de ingeniería expuesta al lector."""
    findings: list[AuditFinding] = []
    for i, line in enumerate((markdown or "").splitlines(), start=1):
        if _METADATA_LEAK_RE.search(line):
            findings.append(
                AuditFinding(
                    category="other",
                    severity="critical",
                    section="manuscript",
                    message=f"Fuga de metadatos internos en línea {i}: {line[:180]}",
                    suggested_fix=(
                        "Eliminar referencias a arquitectura interna y mantener solo "
                        "narrativa científica orientada a la revista."
                    ),
                )
            )
            if len(findings) >= 8:
                break
    return findings


def detect_english_section_leaks(markdown: str) -> list[AuditFinding]:
    """Detecta mezcla de idioma en headings y frases de control."""
    findings: list[AuditFinding] = []
    for i, line in enumerate((markdown or "").splitlines(), start=1):
        if line.strip().startswith("|"):
            continue
        # "References" heading is English; Spanish must be "Referencias"
        if re.match(r"(?i)^#{1,3}\s+references\b", line.strip()):
            findings.append(
                AuditFinding(
                    category="other",
                    severity="critical",
                    section="language",
                    message=f"Heading de referencias en inglés (línea {i}): {line[:160]}",
                    suggested_fix="Usar el heading «Referencias» en español.",
                )
            )
            if len(findings) >= 8:
                break
            continue
        if _ENGLISH_LEAK_RE.search(line):
            findings.append(
                AuditFinding(
                    category="other",
                    severity="critical",
                    section="language",
                    message=f"Se detectó contenido en inglés en línea {i}: {line[:160]}",
                    suggested_fix=(
                        "Reescribir completamente en español académico sin mezclar idiomas."
                    ),
                )
            )
            if len(findings) >= 8:
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
                    "Sustituir por formulación mesurada (p.ej. 'la literatura "
                    "sugiere', 'en este corpus')."
                ),
            )
        )
        if len(findings) >= 8:
            break
    return findings


def detect_required_sections(markdown: str) -> list[AuditFinding]:
    """Exige Introducción, Revisión de Literatura o Marco Teórico, y Referencias."""
    findings: list[AuditFinding] = []
    headings = [
        m.group(2).strip()
        for m in _SECTION_RE.finditer(markdown or "")
        if len(m.group(1)) <= 3
    ]
    has_intro = any(_INTRO_HEADING_RE.search(h) for h in headings)
    has_lit = any(_LIT_HEADING_RE.search(h) for h in headings)
    has_refs = bool(_REF_SECTION_RE.search(markdown or "")) or any(
        _REFS_HEADING_RE.match(f"## {h}") or re.match(r"(?i)^referencias\b", h)
        for h in headings
    )

    if not has_intro:
        findings.append(
            AuditFinding(
                category="sections",
                severity="critical",
                section="structure",
                message="Falta la sección obligatoria: Introducción.",
                suggested_fix="Restaurar la sección «Introducción» en prosa continua.",
            )
        )
    if not has_lit:
        findings.append(
            AuditFinding(
                category="sections",
                severity="critical",
                section="structure",
                message=(
                    "Falta la sección obligatoria: Revisión de Literatura "
                    "o Marco Teórico."
                ),
                suggested_fix=(
                    "Incluir «Revisión de Literatura», «Marco Teórico» o un heading "
                    "combinado equivalente."
                ),
            )
        )
    if not has_refs:
        findings.append(
            AuditFinding(
                category="sections",
                severity="critical",
                section="structure",
                message="Falta la sección obligatoria: Referencias.",
                suggested_fix=(
                    "Añadir el listado bibliográfico APA bajo el heading «Referencias»."
                ),
            )
        )
    return findings


def detect_research_question(markdown: str) -> list[AuditFinding]:
    """Exige pregunta de investigación cerca del cierre de la Introducción."""
    findings: list[AuditFinding] = []
    sections = _split_sections(markdown)
    intro_key = next(
        (k for k in sections if _INTRO_HEADING_RE.search(k) and not k.startswith("_")),
        None,
    )
    if intro_key is None:
        findings.append(
            AuditFinding(
                category="research_question",
                severity="critical",
                section="introducción",
                message=(
                    "No hay sección Introducción donde verificar el cierre con "
                    "pregunta de investigación."
                ),
                suggested_fix=(
                    "Añadir Introducción y cerrarla con la pregunta de investigación."
                ),
            )
        )
        return findings

    intro = sections.get(intro_key, "").strip()
    if not intro:
        findings.append(
            AuditFinding(
                category="research_question",
                severity="critical",
                section=intro_key,
                message="La Introducción está vacía; falta la pregunta de investigación.",
                suggested_fix=(
                    "Redactar la Introducción y cerrarla con la pregunta de investigación."
                ),
            )
        )
        return findings

    # Prefer the final ~40% of the intro (closure zone).
    cutoff = max(0, int(len(intro) * 0.6))
    tail = intro[cutoff:]
    has_rq_phrase = bool(_RQ_PHRASE_RE.search(tail) or _RQ_PHRASE_RE.search(intro))
    has_question_mark = "?" in tail

    if not (has_rq_phrase or has_question_mark):
        findings.append(
            AuditFinding(
                category="research_question",
                severity="critical",
                section=intro_key,
                message=(
                    "La Introducción no cierra con una pregunta de investigación "
                    "(no se detectó «pregunta de investigación» ni «?» cerca del final)."
                ),
                suggested_fix=(
                    "Cerrar la Introducción de forma natural con la pregunta de "
                    "investigación del estudio."
                ),
            )
        )
    elif not has_question_mark and has_rq_phrase and "?" not in intro:
        # Soft: phrase present but no interrogative form anywhere in intro
        findings.append(
            AuditFinding(
                category="research_question",
                severity="minor",
                section=intro_key,
                message=(
                    "Se menciona la pregunta de investigación, pero no aparece "
                    "formulada como interrogación («?»)."
                ),
                suggested_fix=(
                    "Formular explícitamente la pregunta de investigación al cierre "
                    "de la Introducción."
                ),
            )
        )
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
                    "core en Revisión de Literatura) antes de la exportación APA."
                ),
            )
        )

    loose = re.findall(r"(?<!\[)@([\w-]+)(?!\w)", body)
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
    """Señala afirmaciones fuertes sin cita cercana (chequeo soft)."""
    findings: list[AuditFinding] = []
    body = _strip_references_section(markdown)
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
                    "Afirmación fuerte sin respaldo bibliográfico cercano: "
                    f"«{sent[:180]}…»"
                ),
                suggested_fix=(
                    "Añadir [@cite_key] del catálogo o atenuar el tono afirmativo."
                ),
            )
        )
        if len(findings) >= 6:
            break
    return findings


def audit_narrative_coherence(markdown: str) -> list[AuditFinding]:
    """Chequeos ligeros de hilo conductor general → específico (sin métricas)."""
    findings: list[AuditFinding] = []
    sections = _split_sections(markdown)
    lit_keys = [k for k in sections if _LIT_HEADING_RE.search(k) and not k.startswith("_")]
    if lit_keys:
        lit_text = sections[lit_keys[0]]
        # Soft: revisión muy corta sugiere síntesis incompleta
        words = len(re.findall(r"\w+", lit_text))
        if words and words < 120:
            findings.append(
                AuditFinding(
                    category="coherence",
                    severity="major",
                    section=lit_keys[0],
                    message=(
                        "La Revisión de Literatura / Marco Teórico es excesivamente "
                        f"breve ({words} palabras); falta profundidad de lo general "
                        "a lo específico."
                    ),
                    suggested_fix=(
                        "Ampliar la síntesis con fundamentos seminales, constructos "
                        "conceptuales y evidencia core en prosa continua."
                    ),
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
        "orphan_claims": ("coherence", 1.2, 0.6),
        "research_question": ("research_question", 2.5, 1.2),
        "sections": ("prose_structure", 2.5, 1.2),
        "other": ("coherence", 2.5, 1.2),
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
        + dims.research_question
    ) / 5.0
    return round(overall, 2), dims


def _build_revision_instructions(
    findings: list[AuditFinding],
    *,
    language: str,
) -> str:
    if not findings:
        return (
            "Mantener prosa continua, mesura analítica y cierre de la Introducción "
            "con la pregunta de investigación."
            if language == "es"
            else "Maintain continuous prose, analytical restraint, and close the "
            "Introduction with the research question."
        )
    lines = []
    for i, f in enumerate(findings, start=1):
        lines.append(
            f"{i}. [{f.severity}/{f.category}] {f.section}: {f.message} "
            f"→ Fix: {f.suggested_fix}"
        )
    header = (
        "Aplique las siguientes correcciones de forma integral en la siguiente pasada "
        "del Expert Academic Writer:\n"
        if language == "es"
        else "Apply the following corrections comprehensively in the next Writer pass:\n"
    )
    return header + "\n".join(lines)


def evaluate_manuscript(
    markdown: str,
    *,
    catalog: CitationCatalog,
    threshold: float = 8.5,
    language: str = "es",
    use_llm: bool = False,
) -> AuditFeedback:
    """Evalúa el manuscrito lit-only (checks deterministas + LLM opcional)."""
    findings: list[AuditFinding] = []
    findings.extend(detect_structural_bullets(markdown))
    findings.extend(detect_json_leaks(markdown))
    findings.extend(detect_internal_metadata_leaks(markdown))
    findings.extend(detect_english_section_leaks(markdown))
    findings.extend(detect_hype_tone(markdown))
    findings.extend(detect_required_sections(markdown))
    findings.extend(detect_research_question(markdown))
    findings.extend(audit_citations(markdown, catalog))
    findings.extend(detect_orphan_claims(markdown))
    findings.extend(audit_narrative_coherence(markdown))

    if use_llm:
        try:
            findings.extend(_llm_coherence_findings(markdown, language=language))
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
    *,
    language: str,
) -> list[AuditFinding]:
    lang_note = (
        "El manuscrito debe estar en español académico."
        if language == "es"
        else "The manuscript should be in academic English."
    )
    prompt = f"""You are a Q1 journal editor auditing an Introduction + Literature Review.

Return ONLY JSON:
{{"findings":[{{"category":"coherence|tone|research_question|sections|other",
"severity":"critical|major|minor","section":"...","message":"...","suggested_fix":"..."}}]}}

Rules:
- Flag missing research question at the end of the Introduction.
- Flag commercial/hype tone.
- Flag weak general-to-specific narrative in the literature review.
- Do not invent citations or authors.
- Max 5 findings. Empty list if none.
{lang_note}

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
    text = sanitize_manuscript_for_publication((markdown or "").replace("\r\n", "\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
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
    if audit.literature_review_path:
        return WritingBrief(
            title=audit.title,
            literature_review_path=audit.literature_review_path,
            language=audit.language,
            include_latex=audit.include_latex,
        )
    return None


def load_audit_context(
    audit: AuditBrief,
    *,
    settings: Settings | None = None,
) -> tuple[str, CitationCatalog, WritingBrief | None, list[str]]:
    """Carga markdown y catálogo bibliográfico para auditar (lit-only)."""
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
        draft, _ = draft_literature_paper(
            writing, settings=cfg, use_llm=audit.use_llm
        )
        markdown = draft.markdown
        warnings.extend(draft.warnings)
    else:
        raise ValueError(
            "AuditBrief requiere paper_draft_path o writing_brief_path "
            "(o literature_review_path)."
        )

    lit_path = audit.literature_review_path
    if writing is not None:
        lit_path = lit_path or writing.literature_review_path
    if not lit_path:
        raise ValueError(
            "Se requiere literature_review_path (directo o vía writing_brief)."
        )

    tmp_brief = WritingBrief(
        title=audit.title,
        literature_review_path=lit_path,
        language=audit.language,
        include_latex=audit.include_latex,
    )
    review, load_warns = load_writing_inputs(tmp_brief, settings=cfg)
    warnings.extend(load_warns)
    catalog = build_citation_catalog(review)
    warnings.extend(catalog.warnings)
    return markdown, catalog, writing, warnings


def run_quality_audit(
    audit: AuditBrief,
    *,
    settings: Settings | None = None,
) -> AuditVerdict:
    """Auditoría + bucle de feedback Writer cuando score < umbral."""
    cfg = settings or get_settings()
    warnings: list[str] = []
    history: list[AuditFeedback] = []

    markdown, catalog, writing, load_warns = load_audit_context(audit, settings=cfg)
    warnings.extend(load_warns)

    feedback = evaluate_manuscript(
        markdown,
        catalog=catalog,
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
        revised, catalog = revise_literature_with_feedback(
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
            threshold=audit.quality_threshold,
            language=audit.language,
            use_llm=audit.use_llm,
        )
        history.append(feedback)

    polished = polish_manuscript(markdown)
    latex = (
        markdown_to_latex(polished, title=audit.title) if audit.include_latex else None
    )

    decision = feedback.decision
    if (
        feedback.overall_score >= audit.quality_threshold
        and not any(f.severity == "critical" for f in feedback.findings)
    ):
        decision = "accept"
        feedback.decision = "accept"
    elif rounds >= audit.max_revision_rounds and decision != "accept":
        if feedback.overall_score >= audit.quality_threshold - 0.5:
            decision = "revise"
        else:
            decision = "reject"
        feedback.decision = decision

    lit = ""
    if writing is not None:
        lit = writing.literature_review_path
    else:
        lit = audit.literature_review_path or ""

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
    """Deterministic Q1 checks: sections, RQ, bullets, tone, Pandoc citations.

    Args:
        markdown: Literature manuscript markdown to audit.
        writing_brief_json: Optional WritingBrief to rebuild citation catalog.

    Returns:
        JSON with findings and preliminary dimension scores.
    """
    try:
        catalog = CitationCatalog()
        if writing_brief_json.strip():
            brief = WritingBrief.model_validate(json.loads(writing_brief_json))
            review, _ = load_writing_inputs(brief)
            catalog = build_citation_catalog(review)
        findings: list[AuditFinding] = []
        findings.extend(detect_structural_bullets(markdown))
        findings.extend(detect_json_leaks(markdown))
        findings.extend(detect_internal_metadata_leaks(markdown))
        findings.extend(detect_english_section_leaks(markdown))
        findings.extend(detect_hype_tone(markdown))
        findings.extend(detect_required_sections(markdown))
        findings.extend(detect_research_question(markdown))
        findings.extend(audit_citations(markdown, catalog))
        findings.extend(detect_orphan_claims(markdown))
        findings.extend(audit_narrative_coherence(markdown))
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
        markdown, catalog, _, warnings = load_audit_context(audit)
        feedback = evaluate_manuscript(
            markdown,
            catalog=catalog,
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
        markdown: Accepted literature manuscript markdown.
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
