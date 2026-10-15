from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from lsfm_data_processing.registration_and_transforms._batch_register_core import (
    BatchJobSpec,
    get_batch_job_specs,
    load_batch_register_settings_from_path,
)
from lsfm_data_processing.registration_and_transforms.batch_registration.hpc.submission_summary import (
    format_missing_job,
    format_ready_job,
    missing_registration_inputs,
)
from lsfm_data_processing.registration_and_transforms.batch_registration.hpc.verify_rescue_input import (
    verify_rescue_input_for_config,
)
from lsfm_data_processing.registration_and_transforms.runtime_contract import (
    validate_registration_runtime,
)
from lsfm_data_processing.utils.io_helpers import (
    load_toml_config,
    normalize_user_path,
    require_dir,
    require_file,
)


def _resolve_project_path(project_dir: Path, value: str) -> Path:
    candidate = normalize_user_path(value)
    if candidate.is_absolute():
        return candidate
    return project_dir / candidate


def _sanitize_job_component(value: str, max_length: int = 80) -> str:
    sanitized = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in value
    )
    return sanitized[:max_length]


def _registration_output_exists(output_dir: Path) -> bool:
    return (output_dir / "ANTsPy_Warped.nii.gz").exists()


def _normalize_excluded_nodes(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("cluster.exclude_nodes must be a TOML array of node names.")

    nodes = tuple(str(node).strip() for node in value)
    if any(not node for node in nodes):
        raise ValueError("cluster.exclude_nodes must not contain empty node names.")
    if len(nodes) != len(set(nodes)):
        raise ValueError("cluster.exclude_nodes must not contain duplicate node names.")
    return nodes


def _build_sbatch_command(
    *,
    sbatch_script: Path,
    project_dir: Path,
    registration_config: Path,
    job_name_component: str,
    job_index: int,
    conda_env: str,
    python_executable: str,
    partition: str,
    cpus_per_task: int,
    mem_gb: int,
    time_limit: str,
    excluded_nodes: tuple[str, ...],
    log_dir: Path,
    job_name_prefix: str,
) -> list[str]:
    job_suffix = _sanitize_job_component(job_name_component)
    job_name = f"{job_name_prefix}_{job_suffix}"
    command = [
        "sbatch",
        f"--partition={partition}",
        f"--cpus-per-task={cpus_per_task}",
        f"--mem={mem_gb}G",
        f"--time={time_limit}",
    ]
    if excluded_nodes:
        command.append(f"--exclude={','.join(excluded_nodes)}")
    command.extend([
        f"--job-name={job_name}",
        f"--output={log_dir / (job_name + '_%j.out')}",
        f"--error={log_dir / (job_name + '_%j.err')}",
        str(sbatch_script),
        str(project_dir),
        str(registration_config),
        str(job_index),
        conda_env,
        python_executable,
    ])
    return command


project_dir = require_dir(Path.cwd(), "Project directory")
hpc_config = require_file(project_dir / "configs" / "hpc.toml", "HPC config")
cfg = load_toml_config(hpc_config)

cluster_cfg = cfg["cluster"]
workflow_cfg = cfg["workflow"]
logging_cfg = cfg["logging"]
registration_config = _resolve_project_path(
    project_dir,
    workflow_cfg["registration_config"],
)
registration_config = require_file(registration_config, "Registration config")
registration_settings = load_batch_register_settings_from_path(registration_config)
validate_registration_runtime(
    require_installed_packages=True,
)

log_dir = _resolve_project_path(project_dir, logging_cfg["log_dir"])
log_dir.mkdir(parents=True, exist_ok=True)

partition = cluster_cfg["partition"]
cpus_per_task = int(cluster_cfg["cpus_per_task"])
mem_gb = int(cluster_cfg["mem_gb"])
time_limit = cluster_cfg["time"]
excluded_nodes = _normalize_excluded_nodes(cluster_cfg.get("exclude_nodes"))
conda_env = cluster_cfg["conda_env"]
python_executable = cluster_cfg.get("python_executable", "python")

skip_if_output_exists = workflow_cfg.get("skip_if_output_exists", True)
dry_run = workflow_cfg.get("dry_run", False)
job_name_prefix = workflow_cfg.get("job_name_prefix", "reg")

job_specs = get_batch_job_specs(registration_settings)
if not job_specs:
    raise RuntimeError("No batch jobs were defined by the registration config.")

sbatch_script = require_file(
    Path(__file__).resolve().with_name("run_one_batch_job.sh"),
    "HPC batch sbatch wrapper",
)

grouped_submission = os.environ.get("LSFM_GROUPED_SUBMISSION") == "1"
if not grouped_submission:
    mode_label = "dry run" if dry_run else "submission"
    excluded = ",".join(excluded_nodes) if excluded_nodes else "none"
    print(f"Registration {mode_label}: {project_dir.name}")
    print(
        f"Jobs: {len(job_specs)} | Resources: {cpus_per_task} CPUs, "
        f"{mem_gb} GB, {time_limit} | Excluded nodes: {excluded}"
    )
    print()

try:
    input_verification = verify_rescue_input_for_config(registration_config)
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    print(f"[error] {exc}", file=sys.stderr)
    raise SystemExit(1) from exc
if input_verification is not None:
    verified_subject = input_verification.split(" input:", maxsplit=1)[0]
    configured_subjects = {job_spec.moving_image_id for job_spec in job_specs}
    if verified_subject not in configured_subjects:
        print(
            "[error] Rescue provenance subject does not match the registration "
            f"configuration: {verified_subject}",
            file=sys.stderr,
        )
        raise SystemExit(1)

ready_jobs: list[tuple[BatchJobSpec, list[str]]] = []
skipped_count = 0
blocked_count = 0
for job_spec, job in zip(job_specs, registration_settings.jobs, strict=True):
    if skip_if_output_exists and _registration_output_exists(job_spec.output_dir):
        print(f"[skip] {job_spec.moving_image_id}: output already exists")
        skipped_count += 1
        continue

    missing = missing_registration_inputs(job)
    if missing:
        print(format_missing_job(job, missing), file=sys.stderr)
        blocked_count += 1
        continue

    hash_verified = (
        input_verification is not None
        and verified_subject == job_spec.moving_image_id
    )
    print(
        format_ready_job(
            job,
            input_sha256_verified=hash_verified,
            mem_gb=mem_gb,
        )
    )

    command = _build_sbatch_command(
        sbatch_script=sbatch_script,
        project_dir=project_dir,
        registration_config=registration_config,
        job_name_component=(
            f"{job_spec.moving_image_id}__"
            f"{job_spec.fixed_image_id}__"
            f"{job_spec.preset_name}"
        ),
        job_index=job_spec.job_index,
        conda_env=conda_env,
        python_executable=python_executable,
        partition=partition,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        time_limit=time_limit,
        excluded_nodes=excluded_nodes,
        log_dir=log_dir,
        job_name_prefix=job_name_prefix,
    )
    ready_jobs.append((job_spec, command))

if blocked_count:
    if not grouped_submission:
        print()
        print(
            f"Dry run failed: {len(ready_jobs)} ready, {blocked_count} blocked, "
            f"{skipped_count} skipped. Nothing was submitted."
        )
    raise SystemExit(1)

submitted_count = 0
if not dry_run:
    for job_spec, command in ready_jobs:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        response = completed.stdout.strip()
        print(f"[submitted] {job_spec.moving_image_id}: {response}")
        submitted_count += 1

if not grouped_submission:
    print()
    if dry_run:
        print(
            f"Dry run passed: {len(ready_jobs)} ready, {skipped_count} skipped. "
            "Nothing was submitted."
        )
    else:
        print(
            f"Submission complete: {submitted_count} submitted, "
            f"{skipped_count} skipped."
        )
