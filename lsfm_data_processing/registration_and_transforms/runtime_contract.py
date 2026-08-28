from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import atlasspace
from atlasspace import registration
import lsfm_data_processing


REGISTRATION_RUNTIME_FILENAME = "registration_runtime.json"
REGISTRATION_RUNTIME_SCHEMA_VERSION = 1
REQUIRED_REGISTRATION_RESULT_SCHEMA_VERSION = 1


def inspect_registration_runtime() -> dict[str, Any]:
    """Describe the exact Python and package runtime used for registration."""

    return {
        "schema_version": REGISTRATION_RUNTIME_SCHEMA_VERSION,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "prefix": sys.prefix,
        },
        "packages": {
            "atlasspace": _package_record("atlasspace", atlasspace.__file__),
            "lsfm_data_processing": _package_record(
                "lsfm-data-processing",
                lsfm_data_processing.__file__,
            ),
        },
        "registration_contract": {
            "result_filename": getattr(
                registration,
                "REGISTRATION_RESULT_FILENAME",
                None,
            ),
            "result_schema_version": getattr(
                registration,
                "REGISTRATION_RESULT_SCHEMA_VERSION",
                None,
            ),
            "legacy_migration_available": callable(
                getattr(registration, "migrate_legacy_registration_output", None)
            ),
        },
    }


def registration_runtime_validation_errors(
    runtime: dict[str, Any],
    *,
    require_installed_packages: bool,
) -> list[str]:
    """Return actionable incompatibilities in a runtime description."""

    errors: list[str] = []
    packages = runtime["packages"]

    for package_name in ("atlasspace", "lsfm_data_processing"):
        package = packages[package_name]
        if package["version"] is None:
            errors.append(
                f"{package_name} has no installed distribution metadata."
            )
        if require_installed_packages and not _is_installed_package_path(
            package["module_file"]
        ):
            errors.append(
                f"{package_name} must resolve from site-packages; "
                f"found {package['module_file']!r}."
            )

    contract = runtime["registration_contract"]
    if contract["result_filename"] != "registration_result.json":
        errors.append(
            "atlasspace does not expose the registration_result.json contract."
        )
    if (
        contract["result_schema_version"]
        != REQUIRED_REGISTRATION_RESULT_SCHEMA_VERSION
    ):
        errors.append(
            "atlasspace registration-result schema must be "
            f"{REQUIRED_REGISTRATION_RESULT_SCHEMA_VERSION}; found "
            f"{contract['result_schema_version']!r}."
        )
    if not contract["legacy_migration_available"]:
        errors.append("atlasspace does not expose legacy-output migration support.")
    return errors


def validate_registration_runtime(
    *,
    require_installed_packages: bool = True,
) -> dict[str, Any]:
    """Validate package versions, locations, and the result-manifest contract."""

    runtime = inspect_registration_runtime()
    errors = registration_runtime_validation_errors(
        runtime,
        require_installed_packages=require_installed_packages,
    )
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(
            "Registration runtime preflight failed:\n"
            f"{details}\n"
            "Install the tagged registration dependencies into the active "
            "environment before submitting jobs."
        )
    return runtime


def write_registration_runtime_provenance(output_dir: Path) -> Path:
    """Write reproducible runtime metadata beside one registration result."""

    resolved_output_dir = Path(output_dir).resolve(strict=False)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    runtime = inspect_registration_runtime()
    runtime["validation_errors"] = registration_runtime_validation_errors(
        runtime,
        require_installed_packages=False,
    )

    output_path = resolved_output_dir / REGISTRATION_RUNTIME_FILENAME
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(runtime, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def format_registration_runtime(runtime: dict[str, Any]) -> str:
    packages = runtime["packages"]
    contract = runtime["registration_contract"]
    return "\n".join(
        (
            f"Python: {runtime['python']['executable']}",
            "atlasspace: "
            f"{packages['atlasspace']['version']} "
            f"({packages['atlasspace']['module_file']})",
            "lsfm_data_processing: "
            f"{packages['lsfm_data_processing']['version']} "
            f"({packages['lsfm_data_processing']['module_file']})",
            "Registration result contract: "
            f"{contract['result_filename']} schema "
            f"{contract['result_schema_version']}",
        )
    )


def main() -> None:
    runtime = validate_registration_runtime(require_installed_packages=True)
    print(format_registration_runtime(runtime))
    print("Registration runtime preflight passed.")


def _package_record(distribution_name: str, module_file: str | None) -> dict[str, Any]:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return {
            "distribution": distribution_name,
            "version": None,
            "module_file": module_file,
            "direct_url": None,
        }

    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else None
    return {
        "distribution": distribution_name,
        "version": distribution.version,
        "module_file": module_file,
        "direct_url": direct_url,
    }


def _is_installed_package_path(module_file: str | None) -> bool:
    if not module_file:
        return False
    lowered_parts = {part.lower() for part in Path(module_file).parts}
    return "site-packages" in lowered_parts or "dist-packages" in lowered_parts


if __name__ == "__main__":
    main()
