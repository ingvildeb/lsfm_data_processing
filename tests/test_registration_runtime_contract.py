from __future__ import annotations

import json
import tempfile
from pathlib import Path

from lsfm_data_processing.registration_and_transforms.runtime_contract import (
    REGISTRATION_RUNTIME_FILENAME,
    inspect_registration_runtime,
    registration_runtime_validation_errors,
    write_registration_runtime_provenance,
)


def _valid_runtime() -> dict:
    return {
        "packages": {
            "atlasspace": {
                "version": "0.2.0",
                "module_file": "/env/lib/python3.11/site-packages/atlasspace/__init__.py",
            },
            "lsfm_data_processing": {
                "version": "0.2.0",
                "module_file": (
                    "/env/lib/python3.11/site-packages/"
                    "lsfm_data_processing/__init__.py"
                ),
            },
        },
        "registration_contract": {
            "result_filename": "registration_result.json",
            "result_schema_version": 1,
            "legacy_migration_available": True,
        },
    }


def test_valid_installed_runtime_has_no_errors() -> None:
    assert registration_runtime_validation_errors(
        _valid_runtime(),
        require_installed_packages=True,
    ) == []


def test_runtime_validation_reports_metadata_path_and_contract_errors() -> None:
    runtime = _valid_runtime()
    runtime["packages"]["atlasspace"]["version"] = None
    runtime["packages"]["atlasspace"]["module_file"] = "/shared/atlasspace/src/atlasspace/__init__.py"
    runtime["registration_contract"]["result_schema_version"] = None

    errors = registration_runtime_validation_errors(
        runtime,
        require_installed_packages=True,
    )

    assert any("no installed distribution metadata" in error for error in errors)
    assert any("must resolve from site-packages" in error for error in errors)
    assert any("schema must be 1" in error for error in errors)


def test_runtime_inspection_exposes_manifest_contract() -> None:
    runtime = inspect_registration_runtime()

    assert runtime["registration_contract"] == {
        "result_filename": "registration_result.json",
        "result_schema_version": 1,
        "legacy_migration_available": True,
    }


def test_runtime_provenance_is_written_as_json() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = write_registration_runtime_provenance(Path(tmp_dir))

        assert output_path.name == REGISTRATION_RUNTIME_FILENAME
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["python"]["executable"]
        assert "validation_errors" in payload
