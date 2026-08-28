from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

from lsfm_data_processing.registration_and_transforms._batch_register_core import (
    get_batch_job_specs,
    load_batch_register_settings_from_path,
)
from lsfm_data_processing.registration_and_transforms.runtime_contract import (
    format_registration_runtime,
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
registration_runtime = validate_registration_runtime(
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

submitted_jobs: list[str] = []
skipped_jobs: list[str] = []

print(f"Using HPC config: {hpc_config}")
print("Registration runtime preflight:")
print(format_registration_runtime(registration_runtime))
print(f"Using registration config: {registration_config}")
print(
    "Using registration presets: "
    + ", ".join(dict.fromkeys(job_spec.preset_name for job_spec in job_specs))
)
print(f"Project directory: {project_dir}")
print(f"Example output directory: {job_specs[0].output_dir}")
print(f"Log directory: {log_dir}")
print()

for job_spec in job_specs:
    if skip_if_output_exists and _registration_output_exists(job_spec.output_dir):
        print(f"Skipping {job_spec.label}: output already exists.")
        skipped_jobs.append(job_spec.label)
        continue

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

    if dry_run:
        print(f"[dry-run] {shlex.join(command)}")
        continue

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"Submitted {job_spec.label}: {completed.stdout.strip()}")
    submitted_jobs.append(job_spec.label)

print()
print(f"Submission finished. Submitted: {len(submitted_jobs)}")
if submitted_jobs:
    for label in submitted_jobs:
        print(f"  - {label}")

print(f"Skipped: {len(skipped_jobs)}")
if skipped_jobs:
    for label in skipped_jobs:
        print(f"  - {label}")
