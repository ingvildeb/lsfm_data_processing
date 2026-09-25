"""Canonical rescue strategies and deterministic request expansion."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from .models import (
    RescueRequest,
    RescueRequestPlan,
    RescueRequestStatus,
    RescueStrategy,
    RescueStrategyKind,
    SourceRunRequirement,
)


@dataclass(frozen=True)
class RescueVariant:
    """One concrete registration variant produced by a standard strategy."""

    name: str
    overrides: tuple[tuple[str, float | int], ...] = ()
    hpc_mem_gb: int | None = None

    @property
    def override_mapping(self) -> dict[str, float | int]:
        return dict(self.overrides)


@dataclass(frozen=True)
class RescueDefinition:
    strategy: RescueStrategy
    output_group: str
    description: str
    variants: tuple[RescueVariant, ...] = ()


@dataclass(frozen=True)
class ExpandedRescueJob:
    """A single subject/variant rescue job independent of project configuration."""

    subject_id: str
    strategy_id: str
    output_group: str
    variant: str
    overrides: tuple[tuple[str, float | int], ...]
    rationale: str
    source_run: str = ""
    hpc_mem_gb: int | None = None

    @property
    def override_mapping(self) -> dict[str, float | int]:
        return dict(self.overrides)


def _variant(
    name: str,
    *,
    gradient_step: float | None = None,
    working_resolution_um: int | None = None,
    padding_um: int | None = None,
    hpc_mem_gb: int | None = None,
) -> RescueVariant:
    overrides: list[tuple[str, float | int]] = []
    if working_resolution_um is not None:
        overrides.append(("working_resolution_um", working_resolution_um))
    if gradient_step is not None:
        overrides.append(("syn_gradient_step", gradient_step))
    if padding_um is not None:
        overrides.append(("fixed_padding_um", padding_um))
    return RescueVariant(name, tuple(overrides), hpc_mem_gb)


CANONICAL_RESCUE_DEFINITIONS: tuple[RescueDefinition, ...] = (
    RescueDefinition(
        RescueStrategy("padding", "padding"),
        "padding_rescue",
        "Add 500 um fixed-image padding at the baseline resolution.",
        (_variant("pad500um", padding_um=500),),
    ),
    RescueDefinition(
        RescueStrategy("lower_gradient_step", "lower gradient step"),
        "lower_gradient_step_rescue",
        "Sweep conservative SyN gradient steps below the 0.05 baseline.",
        tuple(_variant(f"gs{str(value).replace('.', 'p')}", gradient_step=value) for value in (0.01, 0.02, 0.025)),
    ),
    RescueDefinition(
        RescueStrategy("padding_lower_gradient_step", "padding + lower gradient step"),
        "padded_lower_gradient_step_rescue",
        "Combine 500 um padding with the lower-gradient-step sweep.",
        tuple(_variant(f"gs{str(value).replace('.', 'p')}_pad500um", gradient_step=value, padding_um=500) for value in (0.01, 0.02, 0.025)),
    ),
    RescueDefinition(
        RescueStrategy("higher_gradient_step", "higher gradient step"),
        "higher_gradient_step_rescue",
        "Try SyN gradient steps above the 0.05 baseline.",
        tuple(_variant(f"gs{str(value).replace('.', 'p')}", gradient_step=value) for value in (0.06, 0.07, 0.08)),
    ),
    RescueDefinition(
        RescueStrategy("padding_higher_gradient_step", "higher gradient step + padding"),
        "padded_higher_gradient_step_rescue",
        "Combine 500 um padding with higher SyN gradient steps.",
        tuple(_variant(f"gs{str(value).replace('.', 'p')}_pad500um", gradient_step=value, padding_um=500) for value in (0.06, 0.07, 0.08)),
    ),
    RescueDefinition(
        RescueStrategy("bidirectional_gradient_step", "bidirectional gradient step"),
        "bidirectional_gradient_step_rescue",
        "Compare lower and higher SyN gradient steps around the baseline.",
        tuple(_variant(f"gs{str(value).replace('.', 'p')}", gradient_step=value) for value in (0.01, 0.025, 0.06)),
    ),
    RescueDefinition(
        RescueStrategy("resolution_sweep", "resolution sweep"),
        "resolution_rescue",
        "Try reduced working resolutions with other baseline settings.",
        tuple(_variant(f"wr{value}um", working_resolution_um=value) for value in (30, 40, 50)),
    ),
    RescueDefinition(
        RescueStrategy("padding_resolution_sweep", "padding + resolution sweep"),
        "padded_resolution_rescue",
        "Combine 500 um padding with baseline and reduced resolutions.",
        tuple(_variant(f"wr{value}um_pad500um", working_resolution_um=value, padding_um=500) for value in (20, 30, 40, 50)),
    ),
    RescueDefinition(
        RescueStrategy(
            "high_memory_retry",
            "high memory retry",
            SourceRunRequirement.REQUIRED,
            RescueStrategyKind.EXACT_RETRY,
        ),
        "high_memory_retry",
        "Repeat the selected source run unchanged with 256 GB of memory.",
        (RescueVariant("mem256gb", hpc_mem_gb=256),),
    ),
    RescueDefinition(
        RescueStrategy(
            "masking",
            "masking",
            SourceRunRequirement.REQUIRED,
            RescueStrategyKind.MASKING,
        ),
        "masking_rescue",
        "Prepare a manual mask before an exact source-parameter rerun.",
    ),
    RescueDefinition(
        RescueStrategy("custom", "custom", kind=RescueStrategyKind.CUSTOM),
        "custom_rescue",
        "Run a user-defined parameter combination.",
    ),
)


def canonical_rescue_definitions() -> tuple[RescueDefinition, ...]:
    return CANONICAL_RESCUE_DEFINITIONS


def canonical_rescue_strategies() -> tuple[RescueStrategy, ...]:
    return tuple(item.strategy for item in CANONICAL_RESCUE_DEFINITIONS)


def canonical_rescue_definition_lookup() -> Mapping[str, RescueDefinition]:
    return {item.strategy.strategy_id: item for item in CANONICAL_RESCUE_DEFINITIONS}


def _source_run_token(source_run: str) -> str:
    source = source_run.removeprefix("registration_runs/")
    source_token = re.sub(r"[^a-z0-9]+", "_", source.casefold()).strip("_")
    if not source_token:
        raise ValueError("A source-based rescue requires a usable source-run path")
    return source_token


def _source_variant_name(source_run: str, variant: str) -> str:
    return f"from_{_source_run_token(source_run)}_{variant}"


def expected_rescue_run_paths(
    request: RescueRequest, strategy: RescueStrategy
) -> tuple[str, ...]:
    """Return the runs generated by a canonical rescue request."""

    definition = canonical_rescue_definition_lookup().get(strategy.strategy_id)
    if definition is None:
        return ()
    if strategy.kind is RescueStrategyKind.MASKING:
        if not request.source_run:
            return ()
        token = _source_run_token(request.source_run)
        return (f"registration_runs/{definition.output_group}/from_{token}",)
    if strategy.kind is RescueStrategyKind.EXACT_RETRY and not request.source_run:
        return ()
    if strategy.kind is RescueStrategyKind.CUSTOM:
        variants = (request.effective_variant_name,)
    else:
        variants = tuple(
            _source_variant_name(request.source_run, variant.name)
            if strategy.kind is RescueStrategyKind.EXACT_RETRY
            else variant.name
            for variant in definition.variants
        )
    return tuple(
        f"registration_runs/{definition.output_group}/{variant}"
        for variant in variants
    )


def expand_rescue_request_plan(plan: RescueRequestPlan) -> tuple[ExpandedRescueJob, ...]:
    """Expand pending requests into immutable one-subject/one-variant jobs."""

    definitions = canonical_rescue_definition_lookup()
    jobs: list[ExpandedRescueJob] = []
    for item in plan.active:
        if item.status is not RescueRequestStatus.PENDING:
            continue
        assert item.strategy is not None
        definition = definitions[item.strategy.strategy_id]
        request = item.request
        if item.strategy.kind is RescueStrategyKind.MASKING:
            continue
        if item.strategy.kind is RescueStrategyKind.CUSTOM:
            jobs.append(
                ExpandedRescueJob(
                    subject_id=request.subject_id,
                    strategy_id="custom",
                    output_group=definition.output_group,
                    variant=request.effective_variant_name,
                    overrides=tuple(
                        (parameter, value)
                        for parameter, value in (
                            ("syn_gradient_step", request.gradient_step),
                            ("working_resolution_um", request.working_resolution_um),
                            ("fixed_padding_um", request.padding_um),
                        )
                        if value is not None
                    ),
                    rationale=request.notes,
                    source_run=request.source_run,
                )
            )
            continue
        for variant in definition.variants:
            variant_name = variant.name
            if item.strategy.kind is RescueStrategyKind.EXACT_RETRY:
                variant_name = _source_variant_name(request.source_run, variant.name)
            jobs.append(
                ExpandedRescueJob(
                    subject_id=request.subject_id,
                    strategy_id=item.strategy.strategy_id,
                    output_group=definition.output_group,
                    variant=variant_name,
                    overrides=variant.overrides,
                    rationale=definition.description,
                    source_run=request.source_run,
                    hpc_mem_gb=variant.hpc_mem_gb,
                )
            )
    return tuple(jobs)
