"""Discover planned and executed runs in a project registration batch."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

from atlasspace.registration import load_registration_result_manifest

from .distribution import TRANSFERRED_SUFFIX
from .models import RunEvaluation, normalize_run_path, normalize_subject_id


def discover_registration_run_inventory(
    *,
    batch_dir: Path | str,
    config_dir_name: str = "configs",
    subjects_dir_name: str = "subjects",
) -> tuple[RunEvaluation, ...]:
    """Return normalized run records without modifying the batch or workbook."""

    batch_path = Path(batch_dir)
    config_runs = _planned_runs_from_configs(batch_path / config_dir_name)
    subject_dirs = _subject_directories(batch_path / subjects_dir_name)
    manifested_runs = _manifest_runs(subject_dirs)
    records: list[RunEvaluation] = []
    for key in sorted(set(config_runs).union(manifested_runs)):
        subject_id, run_path = key
        manifest_record = manifested_runs.get(key)
        if manifest_record is not None:
            records.append(manifest_record)
            continue
        subject_dir = subject_dirs.get(subject_id)
        run_dir = subject_dir / Path(run_path) if subject_dir is not None else None
        records.append(
            RunEvaluation(
                subject_id=subject_id,
                run_path=run_path,
                preset_name=config_runs[key],
                manifest_status=(
                    "missing_manifest"
                    if run_dir is not None and run_dir.exists()
                    else "planned"
                ),
            )
        )
    return tuple(records)


def _planned_runs_from_configs(config_dir: Path) -> dict[tuple[str, str], str]:
    inventory: dict[tuple[str, str], str] = {}
    if not config_dir.is_dir():
        return inventory
    for path in sorted(config_dir.glob("*.toml")):
        try:
            with path.open("rb") as handle:
                content = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Invalid registration TOML: {path}") from exc
        run = content.get("run")
        batch = content.get("batch")
        if run is None and batch is None:
            continue
        if not isinstance(run, dict) or not isinstance(batch, dict):
            raise ValueError(
                f"Registration TOML must contain [run] and [batch] tables: {path}"
            )
        mapping = batch.get("image_to_template")
        presets = run.get("registration_presets")
        output = run.get("output_subdir")
        if (
            not isinstance(mapping, dict)
            or not mapping
            or not isinstance(presets, list)
            or len(presets) != 1
            or not output
        ):
            raise ValueError(
                "Registration TOML inventory requires one preset, a nonempty "
                f"image_to_template mapping, and output_subdir: {path}"
            )
        run_path = _validated_run_path(output)
        preset_name = Path(str(presets[0])).stem
        if not preset_name:
            raise ValueError(f"Registration preset name is empty: {path}")
        for raw_subject_id in mapping:
            subject_id = normalize_subject_id(raw_subject_id)
            key = (subject_id, run_path)
            previous = inventory.get(key)
            if previous is not None and previous != preset_name:
                raise ValueError(
                    f"Conflicting planned presets for {subject_id} {run_path}: "
                    f"{previous!r} vs {preset_name!r}"
                )
            inventory[key] = preset_name
    return inventory


def _subject_directories(subjects_dir: Path) -> dict[str, Path]:
    if not subjects_dir.is_dir():
        return {}
    active: dict[str, Path] = {}
    transferred: dict[str, Path] = {}
    for directory in sorted(subjects_dir.iterdir()):
        if not directory.is_dir():
            continue
        is_transferred = directory.name.endswith(TRANSFERRED_SUFFIX)
        raw_subject_id = (
            directory.name.removesuffix(TRANSFERRED_SUFFIX)
            if is_transferred
            else directory.name
        )
        subject_id = normalize_subject_id(raw_subject_id)
        destination = transferred if is_transferred else active
        if subject_id in destination:
            state = "transferred" if is_transferred else "active"
            raise ValueError(f"Duplicate {state} subject directory: {subject_id}")
        destination[subject_id] = directory
    conflicts = sorted(set(active).intersection(transferred))
    if conflicts:
        raise ValueError(
            "Subjects have both active and transferred directories; explicitly "
            "resolve the conflicting state before refreshing evaluation: "
            + ", ".join(conflicts)
        )
    return {**active, **transferred}


def _manifest_runs(
    subject_dirs: dict[str, Path],
) -> dict[tuple[str, str], RunEvaluation]:
    inventory: dict[tuple[str, str], RunEvaluation] = {}
    for subject_id, subject_dir in sorted(subject_dirs.items()):
        runs_dir = subject_dir / "registration_runs"
        if not runs_dir.is_dir():
            continue
        for manifest_path in sorted(runs_dir.rglob("registration_result.json")):
            _validate_manifest_json_contract(manifest_path)
            try:
                manifest = load_registration_result_manifest(manifest_path.parent)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Invalid registration manifest: {manifest_path}"
                ) from exc
            if not manifest.preset_name.strip():
                raise ValueError(
                    f"Registration manifest has an empty preset name: {manifest_path}"
                )
            run_path = _validated_run_path(
                manifest_path.parent.relative_to(subject_dir).as_posix()
            )
            key = (subject_id, run_path)
            if key in inventory:
                raise ValueError(
                    f"Duplicate manifested registration run: {subject_id} {run_path}"
                )
            inventory[key] = RunEvaluation(
                subject_id=subject_id,
                run_path=run_path,
                preset_name=manifest.preset_name,
                manifest_status="success" if manifest.success else "failed",
            )
    return inventory


def _validate_manifest_json_contract(path: Path) -> None:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid registration manifest JSON: {path}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"Registration manifest must contain a JSON object: {path}")
    if content.get("schema_version") != 1:
        raise ValueError(
            f"Registration manifest has an unsupported schema version: {path}"
        )
    if not isinstance(content.get("success"), bool):
        raise ValueError(
            f"Registration manifest success must be Boolean: {path}"
        )


def _validated_run_path(value: object) -> str:
    run_path = normalize_run_path(value)
    if not run_path.startswith("registration_runs/"):
        raise ValueError(
            f"Run path must start with registration_runs/: {value!r}"
        )
    return run_path


__all__ = ["discover_registration_run_inventory"]
