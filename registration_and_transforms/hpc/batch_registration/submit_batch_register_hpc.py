from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys

parent_dir = Path(__file__).resolve().parents[3]
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from lsfm_data_processing.utils.io_helpers import (  # noqa: E402
    load_script_config,
    normalize_user_path,
    require_dir,
    require_file,
)
from registration_and_transforms._batch_register_core import (  # noqa: E402
    load_batch_register_settings_from_path,
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


def _registration_output_exists(subject_dir: Path, output_subdir: str) -> bool:
    output_dir = subject_dir / output_subdir
    return (output_dir / "ANTsPy_Warped.nii.gz").exists()


def _build_sbatch_command(
    *,
    sbatch_script: Path,
    project_dir: Path,
    registration_config: Path,
    subject_dir: Path,
    conda_env: str,
    python_executable: str,
    partition: str,
    cpus_per_task: int,
    mem_gb: int,
    time_limit: str,
    log_dir: Path,
    job_name_prefix: str,
    subject_id: str,
) -> list[str]:
    job_suffix = _sanitize_job_component(subject_id)
    job_name = f"{job_name_prefix}_{job_suffix}"
    return [
        "sbatch",
        f"--partition={partition}",
        f"--cpus-per-task={cpus_per_task}",
        f"--mem={mem_gb}G",
        f"--time={time_limit}",
        f"--job-name={job_name}",
        f"--output={log_dir / (job_name + '_%j.out')}",
        f"--error={log_dir / (job_name + '_%j.err')}",
        str(sbatch_script),
        str(project_dir),
        str(registration_config),
        str(subject_dir),
        conda_env,
        python_executable,
    ]


test_mode = False
cfg = load_script_config(
    Path(__file__),
    "hpc_submission_settings",
    test_mode=test_mode,
)

project_cfg = cfg["project"]
cluster_cfg = cfg["cluster"]
workflow_cfg = cfg["workflow"]
logging_cfg = cfg["logging"]

project_dir = require_dir(project_cfg["project_dir"], "HPC project directory")
registration_config = _resolve_project_path(
    project_dir,
    workflow_cfg["registration_config"],
)
registration_config = require_file(registration_config, "Registration config")
registration_settings = load_batch_register_settings_from_path(registration_config)

subjects_dir = registration_settings.subjects_dir
if not subjects_dir.is_absolute():
    subjects_dir = project_dir / subjects_dir
subjects_dir = require_dir(subjects_dir, "Subjects directory")

log_dir = _resolve_project_path(project_dir, logging_cfg["log_dir"])
log_dir.mkdir(parents=True, exist_ok=True)

partition = cluster_cfg["partition"]
cpus_per_task = int(cluster_cfg["cpus_per_task"])
mem_gb = int(cluster_cfg["mem_gb"])
time_limit = cluster_cfg["time"]
conda_env = cluster_cfg["conda_env"]
python_executable = cluster_cfg.get("python_executable", "python")

skip_if_output_exists = workflow_cfg.get("skip_if_output_exists", True)
dry_run = workflow_cfg.get("dry_run", False)
job_name_prefix = workflow_cfg.get("job_name_prefix", "reg")

subject_ids: list[str] = list(registration_settings.subject_to_template_cfg.keys())
if not subject_ids:
    raise RuntimeError("[subject_to_template] in the registration config must not be empty.")

sbatch_script = require_file(
    Path(__file__).resolve().with_name("submit_batch_register_hpc.sh"),
    "HPC sbatch wrapper",
)

submitted_subjects: list[str] = []
skipped_subjects: list[str] = []

print(f"Using registration config: {registration_config}")
print(f"Using registration preset: {registration_settings.registration_preset}")
print(f"Project directory: {project_dir}")
print(f"Subjects directory: {subjects_dir}")
print(f"Log directory: {log_dir}")
print()

for subject_id in subject_ids:
    subject_dir = require_dir(
        subjects_dir / subject_id,
        f"Subject directory for {subject_id}",
    )

    if skip_if_output_exists and _registration_output_exists(
        subject_dir,
        registration_settings.output_subdir,
    ):
        print(f"Skipping {subject_id}: output already exists.")
        skipped_subjects.append(subject_id)
        continue

    command = _build_sbatch_command(
        sbatch_script=sbatch_script,
        project_dir=project_dir,
        registration_config=registration_config,
        subject_dir=subject_dir,
        conda_env=conda_env,
        python_executable=python_executable,
        partition=partition,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        time_limit=time_limit,
        log_dir=log_dir,
        job_name_prefix=job_name_prefix,
        subject_id=subject_id,
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
    print(f"Submitted {subject_id}: {completed.stdout.strip()}")
    submitted_subjects.append(subject_id)

print()
print(f"Submission finished. Submitted: {len(submitted_subjects)}")
if submitted_subjects:
    for subject_id in submitted_subjects:
        print(f"  - {subject_id}")

print(f"Skipped: {len(skipped_subjects)}")
if skipped_subjects:
    for subject_id in skipped_subjects:
        print(f"  - {subject_id}")
