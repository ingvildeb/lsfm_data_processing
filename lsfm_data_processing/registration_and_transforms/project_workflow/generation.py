"""Typed, project-neutral registration configuration rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _toml_string(value: object) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_list(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


@dataclass(frozen=True)
class RegistrationSegmentationSpec:
    name: str
    path: str | PurePosixPath


@dataclass(frozen=True)
class RegistrationImageSpec:
    image_id: str
    path: str | PurePosixPath
    space_name: str
    orientation: str
    resolution_um: float
    segmentations: tuple[RegistrationSegmentationSpec, ...] = ()


@dataclass(frozen=True)
class RegistrationBatchSpec:
    registration_presets: tuple[str, ...]
    orientation_alignment: str
    write_input_images: bool
    output_subdir: str
    images: tuple[RegistrationImageSpec, ...]
    image_to_template: tuple[tuple[str, str], ...]
    image_default_resolution_um: float
    template_role: str
    moving_segmentations_enabled: bool
    moving_segmentation_interpolation: str
    moving_segmentations_write_intermediates: bool


@dataclass(frozen=True)
class HpcSubmissionEntry:
    config_filename: str
    mem_gb: int | None = None

    def __post_init__(self) -> None:
        if PurePosixPath(self.config_filename).name != self.config_filename:
            raise ValueError("HPC config filename must be a basename")
        if self.mem_gb is not None and self.mem_gb <= 0:
            raise ValueError("HPC memory override must be positive")


@dataclass(frozen=True)
class HpcConfigSpec:
    """Operational settings consumed by the batch-registration submitter."""

    partition: str
    cpus_per_task: int
    mem_gb: int
    time: str
    exclude_nodes: tuple[str, ...]
    conda_env: str
    python_executable: str
    registration_config: str = "configs/baseline.toml"
    skip_if_output_exists: bool = True
    dry_run: bool = True
    job_name_prefix: str = "registration_baseline"
    log_dir: str = "logs/baseline"

    def __post_init__(self) -> None:
        if self.cpus_per_task <= 0:
            raise ValueError("cpus_per_task must be positive")
        if self.mem_gb <= 0:
            raise ValueError("mem_gb must be positive")


def render_hpc_config(spec: HpcConfigSpec) -> str:
    """Render the operational HPC TOML used by the shared submitter."""

    return "\n".join(
        [
            "# Generated operational configuration. Regenerate instead of editing.",
            "",
            "[cluster]",
            f"partition = {_toml_string(spec.partition)}",
            f"cpus_per_task = {spec.cpus_per_task}",
            f"mem_gb = {spec.mem_gb}",
            f"time = {_toml_string(spec.time)}",
            f"exclude_nodes = {_toml_list(spec.exclude_nodes)}",
            f"conda_env = {_toml_string(spec.conda_env)}",
            f"python_executable = {_toml_string(spec.python_executable)}",
            "",
            "[workflow]",
            f"registration_config = {_toml_string(spec.registration_config)}",
            "skip_if_output_exists = "
            f"{_toml_bool(spec.skip_if_output_exists)}",
            f"dry_run = {_toml_bool(spec.dry_run)}",
            f"job_name_prefix = {_toml_string(spec.job_name_prefix)}",
            "",
            "[logging]",
            f"log_dir = {_toml_string(spec.log_dir)}",
            "",
        ]
    )


def render_registration_batch_config(spec: RegistrationBatchSpec) -> str:
    """Render an atlasspace batch-registration TOML from validated inputs."""

    lines = [
        "# Generated configuration. Regenerate instead of editing this file.",
        "",
        "[run]",
        f"registration_presets = {_toml_list(spec.registration_presets)}",
        f"orientation_alignment = {_toml_string(spec.orientation_alignment)}",
        f"write_input_images = {_toml_bool(spec.write_input_images)}",
        f"output_subdir = {_toml_string(spec.output_subdir)}",
        "",
        "[image_defaults]",
        f"resolution_um = {float(spec.image_default_resolution_um)}",
        "",
        "[moving_segmentations]",
        f"enabled = {_toml_bool(spec.moving_segmentations_enabled)}",
        "interpolation = "
        f"{_toml_string(spec.moving_segmentation_interpolation)}",
        "write_intermediates = "
        f"{_toml_bool(spec.moving_segmentations_write_intermediates)}",
        "",
    ]
    image_ids: set[str] = set()
    for image in spec.images:
        if image.image_id in image_ids:
            raise ValueError(f"Duplicate registration image ID: {image.image_id}")
        image_ids.add(image.image_id)
        lines.extend(
            [
                f"[images.{image.image_id}]",
                f"image = {_toml_string(image.path)}",
                f"space_name = {_toml_string(image.space_name)}",
                f"orientation = {_toml_string(image.orientation)}",
                f"resolution_um = {float(image.resolution_um)}",
                "",
            ]
        )
        if image.segmentations:
            lines.append(f"[images.{image.image_id}.segmentations]")
            lines.extend(
                f"{segmentation.name} = {_toml_string(segmentation.path)}"
                for segmentation in image.segmentations
            )
            lines.append("")
    mappings = dict(spec.image_to_template)
    unknown = (set(mappings) | set(mappings.values())).difference(image_ids)
    if unknown:
        raise ValueError(
            "Registration mapping references unknown image IDs: "
            + ", ".join(sorted(unknown))
        )
    lines.extend(
        [
            "[batch]",
            f"template_role = {_toml_string(spec.template_role)}",
            "",
            "[batch.image_to_template]",
        ]
    )
    lines.extend(
        f"{_toml_string(image_id)} = {_toml_string(template_id)}"
        for image_id, template_id in spec.image_to_template
    )
    lines.append("")
    return "\n".join(lines)


def render_hpc_submission_helper(
    *,
    batch_id: str,
    entries: tuple[HpcSubmissionEntry, ...],
    helper_filename: str,
    job_name_prefix: str,
    log_dir: str,
) -> str:
    """Render a dry-run/submit helper with optional per-job memory overrides."""

    if not entries:
        raise ValueError("At least one registration config is required")
    specifications = "\n".join(
        f'  "{entry.config_filename}|{entry.mem_gb or ""}"' for entry in entries
    )
    return f'''#!/usr/bin/env bash
set -euo pipefail

MODE="${{1:-}}"
case "$MODE" in
  dry-run) DRY_RUN_VALUE=true ;;
  submit) DRY_RUN_VALUE=false ;;
  *) echo "Usage: bash {helper_filename} {{dry-run|submit}}" >&2; exit 2 ;;
esac

BATCH_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
HPC_CONFIG="$BATCH_ROOT/configs/hpc.toml"
CONFIG_SPECS=(
{specifications}
)

[[ -f "$HPC_CONFIG" ]] || {{ echo "Missing HPC config: $HPC_CONFIG" >&2; exit 1; }}
for specification in "${{CONFIG_SPECS[@]}}"; do
  IFS='|' read -r config _ <<< "$specification"
  [[ -f "$BATCH_ROOT/configs/$config" ]] || {{ echo "Missing registration config: $config" >&2; exit 1; }}
done

python -m lsfm_data_processing.registration_and_transforms.runtime_contract

BACKUP="$(mktemp)"
cp "$HPC_CONFIG" "$BACKUP"
restore_hpc_config() {{ cp "$BACKUP" "$HPC_CONFIG"; rm -f "$BACKUP"; }}
trap restore_hpc_config EXIT

DEFAULT_MEM_GB="$(sed -n 's/^mem_gb[[:space:]]*=[[:space:]]*//p' "$HPC_CONFIG" | head -n 1 | tr -d '[:space:]')"
[[ -n "$DEFAULT_MEM_GB" ]] || {{ echo "Could not read mem_gb from $HPC_CONFIG" >&2; exit 1; }}
sed -i "s/^dry_run = .*/dry_run = $DRY_RUN_VALUE/" "$HPC_CONFIG"
sed -i 's/^job_name_prefix = .*/job_name_prefix = "{job_name_prefix}"/' "$HPC_CONFIG"
sed -i 's|^log_dir = .*|log_dir = "{log_dir}"|' "$HPC_CONFIG"

for specification in "${{CONFIG_SPECS[@]}}"; do
  IFS='|' read -r config requested_mem_gb <<< "$specification"
  active_mem_gb="${{requested_mem_gb:-$DEFAULT_MEM_GB}}"
  sed -i "s|^registration_config = .*|registration_config = \\"configs/$config\\"|" "$HPC_CONFIG"
  sed -i "s/^mem_gb = .*/mem_gb = $active_mem_gb/" "$HPC_CONFIG"
  echo
  echo "[{batch_id}] $config (${{active_mem_gb}}G)"
  report="$BATCH_ROOT/${{config%.toml}}_${{MODE}}.txt"
  (cd "$BATCH_ROOT" && python -m lsfm_data_processing.registration_and_transforms.batch_registration.hpc.submit_batch_register) | tee "$report"
done

echo
echo "Processed ${{#CONFIG_SPECS[@]}} registration configurations in $MODE mode."
'''
