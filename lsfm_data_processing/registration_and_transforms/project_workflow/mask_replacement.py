"""Explicit replacement of derived outputs after a completed mask is revised."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Sequence

from .decisions import rescue_strategy_lookup
from .masking import MaskingSettings, _masked_fixed_path, _output_dir, _subject_dir
from .models import (
    EvaluationWorkbookData,
    RescueRequestRecord,
    RescueStrategy,
    RescueStrategyKind,
    normalize_run_path,
    normalize_subject_id,
)
from .rescue_catalog import expected_rescue_run_paths
from .workbook_reader import (
    EvaluationWorkbookPlan,
    file_sha256,
    read_evaluation_workbook,
)
from .workbook_writer import apply_evaluation_workbook_plan


@dataclass(frozen=True)
class RevisedMaskReplacementPlan:
    """Reset only artifacts derived from a user-edited completed mask."""

    subject_id: str
    source_run: str
    masking_run: str
    subject_dir: Path
    workbook_plan: EvaluationWorkbookPlan
    completed_mask: Path
    preserved_review_files: tuple[Path, ...]
    files_to_remove: tuple[Path, ...]
    directories_to_remove: tuple[Path, ...]

    @property
    def confirmation(self) -> str:
        return f"REPLACE MASK RESCUE {self.subject_id}"


def build_revised_mask_replacement_plan(
    *,
    workbook_path: Path | str,
    subjects_root: Path | str,
    subject_id: str,
    source_run: str,
    original_fixed_filename: str,
    strategies: Sequence[RescueStrategy],
    settings: MaskingSettings = MaskingSettings(),
) -> RevisedMaskReplacementPlan:
    """Plan an explicit rerun after the user edits a completed mask in place."""

    workbook_path = Path(workbook_path)
    subject_id = normalize_subject_id(subject_id)
    source_run = normalize_run_path(source_run)
    workbook = read_evaluation_workbook(workbook_path)
    strategy_tuple = tuple(strategies)
    lookup = rescue_strategy_lookup(strategy_tuple)
    matches = []
    for record in workbook.rescue_requests:
        strategy = lookup.get(record.request.strategy.casefold())
        if (
            record.request.subject_id == subject_id
            and record.request.source_run == source_run
            and strategy is not None
            and strategy.kind is RescueStrategyKind.MASKING
        ):
            matches.append((record, strategy))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one masking request for "
            f"{subject_id} / {source_run}; found {len(matches)}."
        )
    request_record, strategy = matches[0]
    expected_runs = expected_rescue_run_paths(request_record.request, strategy)
    if len(expected_runs) != 1:
        raise ValueError(
            f"Masking request does not resolve to one output run: {expected_runs}"
        )
    masking_run = expected_runs[0]
    evaluations = [
        row
        for row in workbook.run_evaluations
        if row.subject_id == subject_id and row.run_path == masking_run
    ]
    if len(evaluations) != 1:
        raise ValueError(
            "Expected exactly one masking run evaluation for "
            f"{subject_id} / {masking_run}; found {len(evaluations)}."
        )
    if evaluations[0].selected:
        raise ValueError(
            f"Selected masking run cannot be replaced: {subject_id} / {masking_run}"
        )

    subject_dir = _subject_dir(Path(subjects_root), subject_id, source_run)
    original_fixed = subject_dir / original_fixed_filename
    mask_dir = _output_dir(subject_dir, source_run)
    completed_mask = mask_dir / settings.completed_mask_filename
    review_files = (
        mask_dir / settings.draft_mask_filename,
        mask_dir / settings.review_fixed_filename,
        mask_dir / settings.review_warped_filename,
        completed_mask,
    )
    required = (completed_mask, mask_dir / settings.review_fixed_filename)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Revised-mask inputs are missing for {subject_id}: {missing}"
        )
    files_to_remove = tuple(
        path
        for path in (
            mask_dir / settings.native_mask_filename,
            _masked_fixed_path(original_fixed, settings),
            mask_dir / settings.provenance_filename,
        )
        if path.exists()
    )

    updated_runs = tuple(
        row
        for row in workbook.run_evaluations
        if not (row.subject_id == subject_id and row.run_path == masking_run)
    )
    updated_requests = tuple(
        RescueRequestRecord(record.request, "")
        if record is request_record
        else record
        for record in workbook.rescue_requests
    )
    workbook_plan = EvaluationWorkbookPlan(
        path=workbook_path,
        data=EvaluationWorkbookData(updated_runs, updated_requests),
        strategies=strategy_tuple,
        source_sha256=file_sha256(workbook_path),
        existing_run_count=len(workbook.run_evaluations),
        added_run_count=0,
        migrated_rescue_count=0,
    )
    return RevisedMaskReplacementPlan(
        subject_id=subject_id,
        source_run=source_run,
        masking_run=masking_run,
        subject_dir=subject_dir,
        workbook_plan=workbook_plan,
        completed_mask=completed_mask,
        preserved_review_files=tuple(path for path in review_files if path.exists()),
        files_to_remove=files_to_remove,
        directories_to_remove=(subject_dir / Path(masking_run),),
    )


def summarize_revised_mask_replacement_plan(
    plan: RevisedMaskReplacementPlan,
) -> str:
    lines = [
        f"Revised mask replacement: {plan.subject_id}",
        f"Source run: {plan.source_run}",
        f"Masking run to replace: {plan.masking_run}",
        f"Completed mask preserved: {plan.completed_mask}",
    ]
    lines.extend(f"  remove file: {path}" for path in plan.files_to_remove)
    lines.extend(f"  remove directory: {path}" for path in plan.directories_to_remove)
    lines.append("Workbook: remove the masking-run evaluation and reset its request")
    lines.append(f"Confirmation: {plan.confirmation!r}")
    return "\n".join(lines)


def apply_revised_mask_replacement_plan(
    plan: RevisedMaskReplacementPlan,
    *,
    confirmation: str,
) -> None:
    """Apply an explicit revised-mask reset while preserving user mask files."""

    if confirmation != plan.confirmation:
        raise ValueError(
            f"Confirmation must be exactly {plan.confirmation!r}; got {confirmation!r}."
        )
    for path in (*plan.files_to_remove, *plan.directories_to_remove):
        try:
            path.resolve().relative_to(plan.subject_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Refusing to remove path outside subject directory: {path}"
            ) from exc
    if file_sha256(plan.workbook_plan.path) != plan.workbook_plan.source_sha256:
        raise RuntimeError(
            "Evaluation workbook changed after replacement planning; "
            f"no files were removed: {plan.workbook_plan.path}"
        )
    for path in plan.files_to_remove:
        path.unlink(missing_ok=True)
    for path in plan.directories_to_remove:
        if path.exists():
            shutil.rmtree(path)
    apply_evaluation_workbook_plan(plan.workbook_plan)


__all__ = [
    "RevisedMaskReplacementPlan",
    "apply_revised_mask_replacement_plan",
    "build_revised_mask_replacement_plan",
    "summarize_revised_mask_replacement_plan",
]
