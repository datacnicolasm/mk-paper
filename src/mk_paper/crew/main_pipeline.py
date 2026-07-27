"""Orquestador end-to-end: Literature → Analysis → Writer → Auditor."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any

from mk_paper.config.settings import Settings, get_settings
from mk_paper.crew.analysis_crew import parse_crew_analysis_output, run_analysis_crew
from mk_paper.crew.auditor_crew import parse_crew_audit_output, run_audit_crew
from mk_paper.crew.literature_crew import parse_crew_review_output, run_literature_crew
from mk_paper.crew.writer_crew import parse_crew_paper_output, run_writer_crew
from mk_paper.models.audit_brief import AuditBrief
from mk_paper.models.method_brief import AnalysisReport, MethodBrief
from mk_paper.models.pipeline import PipelineConfig, PipelineResult, PipelineStepResult
from mk_paper.models.research_brief import LiteratureReviewOutput, ResearchBrief
from mk_paper.models.writing_brief import WritingBrief
from mk_paper.persistence.analysis_store import save_analysis_report
from mk_paper.persistence.audit_store import save_audit_verdict
from mk_paper.persistence.literature_store import save_literature_review
from mk_paper.persistence.paper_store import save_paper_draft
from mk_paper.persistence.run_store import (
    PipelineRunContext,
    create_pipeline_run,
    write_manifest,
)
from mk_paper.tools.analysis_tools import resolve_sandbox_path, run_quantitative_analysis
from mk_paper.tools.auditor_tools import run_quality_audit
from mk_paper.tools.systematic_review import run_systematic_review
from mk_paper.tools.writer_tools import draft_imrad_paper

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _step_ok(
    name: str,
    *,
    message: str = "",
    artifacts: dict[str, str] | None = None,
    warnings: list[str] | None = None,
) -> PipelineStepResult:
    return PipelineStepResult(
        step=name,
        status="ok",
        message=message,
        artifacts=artifacts or {},
        warnings=warnings or [],
    )


def _step_err(name: str, exc: BaseException) -> PipelineStepResult:
    return PipelineStepResult(
        step=name,
        status="error",
        message=f"{type(exc).__name__}: {exc}",
        warnings=[traceback.format_exc()[-2000:]],
    )


def _run_literature(
    research: ResearchBrief,
    config: PipelineConfig,
    ctx: PipelineRunContext,
    settings: Settings,
) -> tuple[LiteratureReviewOutput, PipelineStepResult]:
    """Paso 1: revisión sistemática o carga de review local."""
    if config.skip_literature or config.literature_review_path:
        if not config.literature_review_path:
            raise ValueError(
                "skip_literature=True requiere --literature-review path/to/review.json"
            )
        path = resolve_sandbox_path(
            config.literature_review_path, settings=settings, must_exist=True
        )
        raw = _load_json(path)
        for drop in ("persisted_at", "run_id"):
            raw.pop(drop, None)
        review = LiteratureReviewOutput.model_validate(raw)
        dest = ctx.copy_file(path, "literature/review.json")
        ctx.write_text(
            "literature/review.md",
            f"# Literature review (cached)\n\nSource: `{path}`\n",
        )
        ctx.log(f"Literature skipped/cached from {path}")
        return review, _step_ok(
            "literature",
            message=f"Loaded cached review from {path}",
            artifacts={"review_json": str(dest)},
            warnings=list(review.warnings or []),
        )

    ctx.log(f"Literature search starting for brief={research.title!r}")
    if config.via_crew:
        raw = run_literature_crew(research)
        review = parse_crew_review_output(raw)
        if review is None:
            ctx.log("Literature crew unparseable; falling back to direct pipeline")
            review = asyncio.run(run_systematic_review(research))
            review.warnings.append(
                "Crew output was not valid JSON; used direct systematic review."
            )
    else:
        review = asyncio.run(run_systematic_review(research))

    lit_artifacts = save_literature_review(review, output_dir=settings.output_dir)
    dest_json = ctx.copy_file(lit_artifacts.json_path, "literature/review.json")
    ctx.copy_file(lit_artifacts.md_path, "literature/review.md")
    # Mirror canonical name requested by architecture
    ctx.copy_file(lit_artifacts.json_path, "literature/latest_review.json")
    ctx.log(f"Literature saved: {lit_artifacts.json_path}")
    return review, _step_ok(
        "literature",
        message="Systematic literature review completed",
        artifacts={
            "review_json": str(dest_json),
            "global_review_json": str(lit_artifacts.json_path),
            "global_latest": str(lit_artifacts.latest_json),
        },
        warnings=list(review.warnings or []),
    )


def _run_analysis(
    method: MethodBrief,
    *,
    literature_review_path: str,
    config: PipelineConfig,
    ctx: PipelineRunContext,
    settings: Settings,
) -> tuple[AnalysisReport, PipelineStepResult]:
    """Paso 2: análisis cuantitativo sobre dataset local."""
    updates: dict[str, Any] = {
        "literature_review_path": literature_review_path,
        "enrich_discussion_with_llm": bool(config.enrich_analysis_discussion),
    }
    if config.dataset_path:
        updates["dataset_path"] = config.dataset_path
    brief = method.model_copy(update=updates)
    ctx.write_json("briefs/method_brief.json", brief.model_dump(mode="json"))
    ctx.log(
        f"Analysis starting task={brief.task_type} models={brief.models} "
        f"dataset={brief.dataset_path}"
    )

    if config.via_crew:
        raw = run_analysis_crew(brief)
        report = parse_crew_analysis_output(raw)
        if report is None:
            ctx.log("Analysis crew unparseable; falling back to direct engine")
            report = run_quantitative_analysis(brief, settings=settings)
    else:
        report = run_quantitative_analysis(brief, settings=settings)

    ana_artifacts = save_analysis_report(report, output_dir=settings.output_dir)
    dest = ctx.copy_file(ana_artifacts.json_path, "analysis/report.json")
    ctx.copy_file(ana_artifacts.json_path, "analysis/latest_report.json")
    ctx.copy_file(ana_artifacts.md_path, "analysis/report.md")
    # Also refresh global latest under output/analysis (already done by saver)
    ctx.log(f"Analysis saved: {ana_artifacts.json_path} best={report.best_model}")
    return report, _step_ok(
        "analysis",
        message=f"Best model={report.best_model} score={report.best_score}",
        artifacts={
            "report_json": str(dest),
            "global_latest": str(ana_artifacts.latest_json),
        },
        warnings=list(report.warnings or []),
    )


def _run_writer(
    *,
    title: str,
    literature_path: str,
    analysis_path: str,
    config: PipelineConfig,
    ctx: PipelineRunContext,
    settings: Settings,
) -> tuple[Any, Any, PipelineStepResult]:
    """Paso 3: Scientific Writer IMRaD + Pandoc cite_keys."""
    writing = WritingBrief(
        title=title,
        literature_review_path=literature_path,
        analysis_report_path=analysis_path,
        language=config.language,
        include_latex=config.include_latex,
    )
    writing_path = ctx.write_json(
        "briefs/writing_brief.json", writing.model_dump(mode="json")
    )
    ctx.log(f"Writer starting title={title!r}")

    if config.via_crew:
        raw = run_writer_crew(writing)
        draft = parse_crew_paper_output(raw)
        catalog = None
        if draft is None:
            ctx.log("Writer crew unparseable; falling back to direct engine")
            draft, catalog = draft_imrad_paper(
                writing, settings=settings, use_llm=config.use_llm
            )
    else:
        draft, catalog = draft_imrad_paper(
            writing, settings=settings, use_llm=config.use_llm
        )

    paper_artifacts = save_paper_draft(
        draft, catalog, output_dir=settings.output_dir
    )
    ctx.copy_file(paper_artifacts.draft_md, "paper/draft_imrad.md")
    ctx.copy_file(paper_artifacts.draft_json, "paper/paper_draft.json")
    if paper_artifacts.catalog_json.exists():
        ctx.copy_file(paper_artifacts.catalog_json, "paper/citation_catalog.json")
    if paper_artifacts.draft_tex:
        ctx.copy_file(paper_artifacts.draft_tex, "paper/draft_imrad.tex")
    ctx.log(f"Writer draft saved: {paper_artifacts.draft_md}")
    return draft, writing_path, _step_ok(
        "writer",
        message=f"Draft status={draft.status} citations={len(draft.citations_used)}",
        artifacts={
            "draft_md": str(ctx.paper_dir / "draft_imrad.md"),
            "writing_brief": str(writing_path),
            "global_latest_md": str(paper_artifacts.latest_md),
        },
        warnings=list(draft.warnings or []),
    )


def _run_auditor(
    *,
    title: str,
    draft_md_path: str,
    writing_brief_path: str,
    literature_path: str,
    analysis_path: str,
    config: PipelineConfig,
    ctx: PipelineRunContext,
    settings: Settings,
) -> tuple[Any, PipelineStepResult]:
    """Paso 4: Quality Auditor + feedback loop al Writer."""
    audit = AuditBrief(
        title=f"Audit — {title}",
        paper_draft_path=draft_md_path,
        writing_brief_path=writing_brief_path,
        literature_review_path=literature_path,
        analysis_report_path=analysis_path,
        language=config.language,
        quality_threshold=config.quality_threshold,
        max_revision_rounds=config.max_audit_rounds,
        include_latex=config.include_latex,
        use_llm=config.use_llm,
    )
    ctx.write_json("briefs/audit_brief.json", audit.model_dump(mode="json"))
    ctx.log(
        f"Auditor starting threshold={audit.quality_threshold} "
        f"max_rounds={audit.max_revision_rounds}"
    )

    if config.via_crew:
        raw = run_audit_crew(audit)
        verdict = parse_crew_audit_output(raw)
        if verdict is None:
            ctx.log("Audit crew unparseable; falling back to direct engine")
            verdict = run_quality_audit(audit, settings=settings)
    else:
        verdict = run_quality_audit(audit, settings=settings)

    audit_artifacts = save_audit_verdict(verdict, output_dir=settings.output_dir)
    ctx.copy_file(audit_artifacts.verdict_json, "audit/review_verdict.json")
    ctx.copy_file(audit_artifacts.polished_md, "audit/manuscript_final.md")
    ctx.copy_file(audit_artifacts.polished_md, "final/manuscript.md")
    if audit_artifacts.polished_tex:
        ctx.copy_file(audit_artifacts.polished_tex, "audit/manuscript_final.tex")
        ctx.copy_file(audit_artifacts.polished_tex, "final/manuscript.tex")
    # Also store feedback history snapshot
    ctx.write_json(
        "audit/feedback_history.json",
        [fb.model_dump(mode="json") for fb in verdict.feedback_history],
    )
    ctx.log(
        f"Auditor decision={verdict.decision} score={verdict.overall_score} "
        f"rounds={verdict.rounds_completed}"
    )
    return verdict, _step_ok(
        "auditor",
        message=(
            f"decision={verdict.decision} score={verdict.overall_score}/10 "
            f"rounds={verdict.rounds_completed}"
        ),
        artifacts={
            "review_verdict": str(ctx.audit_dir / "review_verdict.json"),
            "final_md": str(ctx.final_dir / "manuscript.md"),
            "global_verdict": str(audit_artifacts.latest_verdict),
        },
        warnings=list(verdict.warnings or []),
    )


def run_pipeline(
    config: PipelineConfig,
    *,
    settings: Settings | None = None,
) -> PipelineResult:
    """Ejecuta el pipeline completo Literature → Analysis → Writer → Auditor.

    Persiste cada paso en ``output/runs/{timestamp}_paper_run/`` y refleja
    artefactos globales en ``output/literature|analysis|paper|audit/``.
    """
    cfg = settings or get_settings()
    steps: list[PipelineStepResult] = []
    warnings: list[str] = []

    research_path = resolve_sandbox_path(
        config.research_brief_path, settings=cfg, must_exist=True
    )
    method_path = resolve_sandbox_path(
        config.method_brief_path, settings=cfg, must_exist=True
    )
    research = ResearchBrief.model_validate(_load_json(research_path))
    if config.literature_max_results is not None:
        research = research.model_copy(
            update={
                "max_results": max(1, min(int(config.literature_max_results), 50))
            }
        )
    method = MethodBrief.model_validate(_load_json(method_path))

    title = config.title or method.title or research.title or "paper_run"
    ctx = create_pipeline_run(output_dir=cfg.output_dir, title=title)
    ctx.write_json("briefs/research_brief.json", research.model_dump(mode="json"))
    ctx.write_json("briefs/pipeline_config.json", config.model_dump(mode="json"))
    write_manifest(
        ctx,
        {
            "status": "running",
            "title": title,
            "config": config.model_dump(mode="json"),
            "steps": [],
        },
    )

    try:
        # --- Literature ---
        review, lit_step = _run_literature(research, config, ctx, cfg)
        steps.append(lit_step)
        warnings.extend(lit_step.warnings)
        lit_run_path = str(ctx.literature_dir / "review.json")

        # --- Analysis ---
        report, ana_step = _run_analysis(
            method,
            literature_review_path=lit_run_path,
            config=config,
            ctx=ctx,
            settings=cfg,
        )
        steps.append(ana_step)
        warnings.extend(ana_step.warnings)
        ana_run_path = str(ctx.analysis_dir / "latest_report.json")

        # --- Writer ---
        draft, writing_path, wr_step = _run_writer(
            title=title,
            literature_path=lit_run_path,
            analysis_path=ana_run_path,
            config=config,
            ctx=ctx,
            settings=cfg,
        )
        steps.append(wr_step)
        warnings.extend(wr_step.warnings)
        draft_md_path = str(ctx.paper_dir / "draft_imrad.md")

        # --- Auditor (+ Writer feedback loop interno) ---
        verdict, au_step = _run_auditor(
            title=title,
            draft_md_path=draft_md_path,
            writing_brief_path=str(writing_path),
            literature_path=lit_run_path,
            analysis_path=ana_run_path,
            config=config,
            ctx=ctx,
            settings=cfg,
        )
        steps.append(au_step)
        warnings.extend(au_step.warnings)

        # If auditor produced a polished manuscript, refresh paper draft mirror
        if verdict.polished_markdown:
            ctx.write_text("final/manuscript.md", verdict.polished_markdown)
            ctx.write_text("paper/draft_imrad.md", verdict.polished_markdown)
        if verdict.latex:
            ctx.write_text("final/manuscript.tex", verdict.latex)

        status: str = "ok" if verdict.decision == "accept" else "partial"
        if verdict.decision == "reject" or any(s.status == "error" for s in steps):
            status = "error"

        result = PipelineResult(
            run_id=ctx.run_id,
            run_dir=str(ctx.run_dir),
            status=status,  # type: ignore[arg-type]
            decision=verdict.decision,  # type: ignore[arg-type]
            overall_score=verdict.overall_score,
            steps=steps,
            final_manuscript_md=str(ctx.final_dir / "manuscript.md"),
            final_manuscript_tex=(
                str(ctx.final_dir / "manuscript.tex")
                if (ctx.final_dir / "manuscript.tex").exists()
                else ""
            ),
            review_verdict_path=str(ctx.audit_dir / "review_verdict.json"),
            manifest_path=str(ctx.manifest_path),
            warnings=warnings,
        )
        write_manifest(ctx, result.to_dict())
        ctx.log(f"Pipeline finished status={result.status} decision={result.decision}")
        return result

    except Exception as exc:  # noqa: BLE001
        ctx.log(f"Pipeline FAILED: {exc}")
        err_step = _step_err("pipeline", exc)
        steps.append(err_step)
        result = PipelineResult(
            run_id=ctx.run_id,
            run_dir=str(ctx.run_dir),
            status="error",
            decision="reject",
            steps=steps,
            manifest_path=str(ctx.manifest_path),
            warnings=[*warnings, str(exc)],
        )
        write_manifest(ctx, result.to_dict())
        return result
