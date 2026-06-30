from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import tomllib

from lsfm_data_processing.registration_and_transforms._sweep_register_core import (
    _pair_output_dir,
    get_sweep_job_specs,
    load_sweep_register_settings_from_path,
)
from lsfm_data_processing.utils.io_helpers import (
    _normalize_backslashes_in_toml_strings,
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


def _preset_job_component(preset_name: str) -> str:
    preset_path = Path(preset_name)
    if preset_path.suffix.lower() in {".yaml", ".yml"}:
        return preset_path.stem
    return preset_name


def _registration_output_exists(output_dir: Path) -> bool:
    return (output_dir / "ANTsPy_Warped.nii.gz").exists()


def _build_sbatch_command(
    *,
    sbatch_script: Path,
    project_dir: Path,
    registration_config: Path,
    image_key: str,
    template_key: str,
    preset_name: str,
    conda_env: str,
    python_executable: str,
    partition: str,
    cpus_per_task: int,
    mem_gb: int,
    time_limit: str,
    log_dir: Path,
    job_name_prefix: str,
) -> list[str]:
    preset_component = _preset_job_component(preset_name)
    job_suffix = _sanitize_job_component(
        f"{image_key}_{template_key}_{preset_component}",
    )
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
        image_key,
        template_key,
        preset_name,
        conda_env,
        python_executable,
    ]


def _load_hpc_cfg(config_path: Path) -> dict:
    config_text = config_path.read_text(encoding="utf-8")
    try:
        return tomllib.loads(config_text)
    except tomllib.TOMLDecodeError as exc:
        if "Unescaped '\\' in a string" not in str(exc):
            raise
        normalized_text = _normalize_backslashes_in_toml_strings(config_text)
        return tomllib.loads(normalized_text)


project_dir = require_dir(Path.cwd(), "Project directory")
hpc_config = require_file(project_dir / "configs" / "hpc.toml", "HPC config")
cfg = _load_hpc_cfg(hpc_config)

cluster_cfg = cfg["cluster"]
workflow_cfg = cfg["workflow"]
logging_cfg = cfg["logging"]
registration_config = _resolve_project_path(
    project_dir,
    workflow_cfg["registration_config"],
)
registration_config = require_file(registration_config, "Sweep registration config")
registration_settings = load_sweep_register_settings_from_path(registration_config)

if not registration_settings.output_root.is_absolute():
    registration_settings.output_root = project_dir / registration_settings.output_root

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
job_name_prefix = workflow_cfg.get("job_name_prefix", "sweep")

job_specs = get_sweep_job_specs(registration_settings)
if not job_specs:
    raise RuntimeError("No sweep jobs were defined by the registration config.")

sbatch_script = require_file(
    Path(__file__).resolve().with_name("run_one_sweep_job.sh"),
    "HPC sweep sbatch wrapper",
)

submitted_jobs: list[str] = []
skipped_jobs: list[str] = []

print(f"Using HPC config: {hpc_config}")
print(f"Using registration config: {registration_config}")
print(f"Using registration presets: {', '.join(registration_settings.preset_names)}")
print(f"Project directory: {project_dir}")
print(f"Output root: {registration_settings.output_root}")
print(f"Log directory: {log_dir}")
print()

for job_spec in job_specs:
    output_dir = _pair_output_dir(
        registration_settings,
        image_key=job_spec.image_key,
        template_key=job_spec.template_key,
        preset_name=job_spec.preset_name,
    )

    if skip_if_output_exists and _registration_output_exists(output_dir):
        label = (
            f"{job_spec.image_key} -> {job_spec.template_key} "
            f"({job_spec.preset_name})"
        )
        print(f"Skipping {label}: output already exists.")
        skipped_jobs.append(label)
        continue

    command = _build_sbatch_command(
        sbatch_script=sbatch_script,
        project_dir=project_dir,
        registration_config=registration_config,
        image_key=job_spec.image_key,
        template_key=job_spec.template_key,
        preset_name=job_spec.preset_name,
        conda_env=conda_env,
        python_executable=python_executable,
        partition=partition,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        time_limit=time_limit,
        log_dir=log_dir,
        job_name_prefix=job_name_prefix,
    )

    label = (
        f"{job_spec.image_key} -> {job_spec.template_key} "
        f"({job_spec.preset_name})"
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
    print(f"Submitted {label}: {completed.stdout.strip()}")
    submitted_jobs.append(label)

print()
print(f"Submission finished. Submitted: {len(submitted_jobs)}")
if submitted_jobs:
    for label in submitted_jobs:
        print(f"  - {label}")

print(f"Skipped: {len(skipped_jobs)}")
if skipped_jobs:
    for label in skipped_jobs:
        print(f"  - {label}")
