"""Normalized records used by project-level registration workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any


def normalize_text(value: Any) -> str:
    """Return a stripped string while treating missing spreadsheet values as blank."""

    if value is None:
        return ""
    try:
        if value != value:  # NaN-like values compare unequal to themselves.
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_subject_id(value: Any) -> str:
    subject_id = normalize_text(value)
    if not subject_id:
        raise ValueError("Subject ID must not be blank")
    return subject_id


def normalize_run_path(value: Any) -> str:
    """Normalize a workbook run reference without imposing an OS-specific path."""

    return normalize_text(value).replace("\\", "/").strip("/")


def selected_marker(value: Any) -> bool:
    """Interpret the selected markers accepted by existing project workbooks."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return normalize_text(value).casefold() in {"1", "selected", "true", "yes"}


class FinalDecisionKind(StrEnum):
    SELECTED = "selected"
    EXCLUDED = "excluded"
    UNRESOLVED = "unresolved"


class SourceRunRequirement(StrEnum):
    OPTIONAL = "optional"
    REQUIRED = "required"


class RescueStrategyKind(StrEnum):
    AUTOMATIC = "automatic"
    EXACT_RETRY = "exact_retry"
    MASKING = "masking"
    CUSTOM = "custom"


class RescueRequestStatus(StrEnum):
    PENDING = "pending"
    GENERATED = "generated"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_MANUAL_MASK = "awaiting_manual_mask"
    MASK_READY = "mask_ready"
    BLOCKED_PENDING_RESCUES = "blocked_pending_rescues"
    IGNORED_FINALIZED = "ignored_finalized"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class RunEvaluation:
    """One evaluated registration run from the canonical run table."""

    subject_id: str
    run_path: str
    preset_name: str = ""
    manifest_status: str = ""
    result: str = ""
    comments: str = ""
    selected: bool = False
    evaluator: str = ""
    evaluation_date: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", normalize_subject_id(self.subject_id))
        object.__setattr__(self, "run_path", normalize_run_path(self.run_path))
        object.__setattr__(self, "preset_name", normalize_text(self.preset_name))
        object.__setattr__(
            self, "manifest_status", normalize_text(self.manifest_status).casefold()
        )
        object.__setattr__(self, "result", normalize_text(self.result))
        object.__setattr__(self, "comments", normalize_text(self.comments))
        object.__setattr__(self, "selected", selected_marker(self.selected))
        object.__setattr__(self, "evaluator", normalize_text(self.evaluator))


@dataclass(frozen=True)
class FinalDecision:
    subject_id: str
    kind: FinalDecisionKind
    run_path: str | None = None
    evaluation: RunEvaluation | None = None

    @property
    def finalized(self) -> bool:
        return self.kind in {
            FinalDecisionKind.SELECTED,
            FinalDecisionKind.EXCLUDED,
        }


@dataclass(frozen=True)
class RescueStrategy:
    """A configured rescue choice exposed to a project workbook."""

    strategy_id: str
    workbook_label: str
    source_run_requirement: SourceRunRequirement = SourceRunRequirement.OPTIONAL
    kind: RescueStrategyKind = RescueStrategyKind.AUTOMATIC
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        strategy_id = normalize_text(self.strategy_id)
        workbook_label = normalize_text(self.workbook_label)
        if not strategy_id:
            raise ValueError("Rescue strategy ID must not be blank")
        if not workbook_label:
            raise ValueError("Rescue strategy workbook label must not be blank")
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "workbook_label", workbook_label)
        object.__setattr__(
            self,
            "source_run_requirement",
            SourceRunRequirement(self.source_run_requirement),
        )
        object.__setattr__(self, "kind", RescueStrategyKind(self.kind))
        object.__setattr__(
            self,
            "aliases",
            tuple(alias for value in self.aliases if (alias := normalize_text(value))),
        )


@dataclass(frozen=True)
class RescueRequest:
    """One user-entered rescue request row."""

    subject_id: str
    strategy: str
    source_run: str = ""
    variant_name: str = ""
    gradient_step: float | None = None
    working_resolution_um: float | None = None
    padding_um: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", normalize_subject_id(self.subject_id))
        strategy = normalize_text(self.strategy)
        if not strategy:
            raise ValueError("Rescue strategy must not be blank")
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "source_run", normalize_run_path(self.source_run))
        object.__setattr__(self, "variant_name", normalize_text(self.variant_name))
        object.__setattr__(
            self, "gradient_step", _normalize_optional_number(self.gradient_step)
        )
        object.__setattr__(
            self,
            "working_resolution_um",
            _normalize_optional_number(self.working_resolution_um),
        )
        object.__setattr__(self, "padding_um", _normalize_optional_number(self.padding_um))
        object.__setattr__(self, "notes", normalize_text(self.notes))

    @property
    def custom_parameters(self) -> tuple[tuple[str, float], ...]:
        values = (
            ("gradient_step", self.gradient_step),
            ("working_resolution_um", self.working_resolution_um),
            ("padding_um", self.padding_um),
        )
        return tuple((name, value) for name, value in values if value is not None)

    @property
    def effective_variant_name(self) -> str:
        if self.variant_name:
            return _variant_token(self.variant_name)
        tokens: list[str] = []
        if self.working_resolution_um is not None:
            tokens.append(f"wr{_number_token(self.working_resolution_um)}um")
        if self.gradient_step is not None:
            tokens.append(f"gs{_number_token(self.gradient_step)}")
        if self.padding_um is not None:
            tokens.append(f"pad{_number_token(self.padding_um)}um")
        return "_".join(tokens)


@dataclass(frozen=True, order=True)
class RescueRequestKey:
    """Stable identity before a strategy is expanded into parameter variants."""

    subject_id: str
    strategy_id: str
    required_source_run: str = ""
    variant_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", normalize_subject_id(self.subject_id))
        strategy_id = normalize_text(self.strategy_id)
        if not strategy_id:
            raise ValueError("Rescue request strategy ID must not be blank")
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(
            self,
            "required_source_run",
            normalize_run_path(self.required_source_run),
        )
        object.__setattr__(self, "variant_name", normalize_text(self.variant_name))


@dataclass(frozen=True)
class ResolvedRescueRequest:
    request: RescueRequest
    key: RescueRequestKey
    status: RescueRequestStatus
    strategy: RescueStrategy | None = None
    reason: str = ""


@dataclass(frozen=True)
class DuplicateRescueRequest:
    key: RescueRequestKey
    occurrences: int


@dataclass(frozen=True)
class RescueRequestPlan:
    active: tuple[ResolvedRescueRequest, ...]
    ignored: tuple[ResolvedRescueRequest, ...]
    duplicates: tuple[DuplicateRescueRequest, ...]
    warnings: tuple[str, ...] = ()

    @property
    def all(self) -> tuple[ResolvedRescueRequest, ...]:
        return self.active + self.ignored


@dataclass(frozen=True)
class RescueRequestRecord:
    """A rescue request together with its last synchronized derived status."""

    request: RescueRequest
    status: str = ""

    def __post_init__(self) -> None:
        status = normalize_text(self.status).casefold()
        if status:
            try:
                status = RescueRequestStatus(status).value
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported derived rescue request status: {self.status!r}"
                ) from exc
        object.__setattr__(self, "status", status)


@dataclass(frozen=True)
class EvaluationWorkbookData:
    run_evaluations: tuple[RunEvaluation, ...]
    rescue_requests: tuple[RescueRequestRecord, ...]


def _normalize_optional_number(value: Any) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Custom rescue parameter must be numeric, got {value!r}") from exc


def _number_token(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _variant_token(value: str) -> str:
    token = normalize_text(value).casefold().replace(" ", "_")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", token):
        raise ValueError(
            "Custom rescue variant name must contain only letters, numbers, "
            f"spaces, underscores, or hyphens: {value!r}"
        )
    return token
