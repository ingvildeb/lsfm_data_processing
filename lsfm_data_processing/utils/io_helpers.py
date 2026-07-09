from dataclasses import dataclass
from importlib import resources as importlib_resources
import re
import tomllib
from pathlib import Path
from typing import Any


# -------------------------
# PATH NORMALIZATION
# -------------------------

def normalize_user_path(p: str | Path) -> Path:
    """
    Normalize a user-provided path string into a Path object.

    Accepts either a string or Path. If a string is provided, backslashes
    are converted to forward slashes to avoid Windows escape issues and
    improve cross-platform robustness.

    Parameters
    ----------
    p : str | Path
        User-provided filesystem path.

    Returns
    -------
    Path
        Normalized Path object.
    """
    if isinstance(p, Path):
        return p
    return Path(p.replace("\\", "/"))


# -------------------------
# VALIDATION HELPERS
# -------------------------

def require_dir(path: str | Path, name: str = "Directory") -> Path:
    """
    Ensure that a directory exists and is a folder.

    Parameters
    ----------
    path : str | Path
        Directory path to validate.
    name : str, optional
        Human-readable label used in error messages.

    Returns
    -------
    Path
        Normalized directory Path.

    Raises
    ------
    RuntimeError
        If the path does not exist or is not a directory.
    """
    p = normalize_user_path(path)

    if not p.exists():
        raise RuntimeError(f"{name} does not exist:\n{p}")

    if not p.is_dir():
        raise RuntimeError(f"{name} is not a directory:\n{p}")

    return p


def require_file(path: str | Path, name: str = "File") -> Path:
    """
    Ensure that a file exists and is a regular file.

    Parameters
    ----------
    path : str | Path
        File path to validate.
    name : str, optional
        Human-readable label used in error messages.

    Returns
    -------
    Path
        Normalized file Path.

    Raises
    ------
    RuntimeError
        If the path does not exist or is not a file.
    """
    p = normalize_user_path(path)

    if not p.exists():
        raise RuntimeError(f"{name} does not exist:\n{p}")

    if not p.is_file():
        raise RuntimeError(f"{name} is not a file:\n{p}")

    return p


def require_subpath(parent: Path, sub: str, name: str) -> Path:
    """
    Ensure that a required subpath exists inside a parent directory.

    Useful when validating expected folder structures produced by upstream
    pipelines (e.g., registration outputs, stitched folders, etc.).

    Parameters
    ----------
    parent : Path
        Parent directory.
    sub : str
        Required child name (file or folder).
    name : str
        Human-readable label for error reporting.

    Returns
    -------
    Path
        The resolved subpath.

    Raises
    ------
    RuntimeError
        If the subpath does not exist.
    """
    p = parent / sub

    if not p.exists():
        raise RuntimeError(
            f"Missing {name} in:\n{parent}\nExpected:\n{p}"
        )

    return p


def list_tiff_files(folder: Path) -> list[Path]:
    """
    Return sorted TIFF files directly inside a folder.

    Parameters
    ----------
    folder : Path
        Folder to scan (non-recursive).

    Returns
    -------
    list[Path]
        Sorted list of .tif/.tiff files.
    """
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
        ]
    )


# -------------------------
# CONFIG LOADER
# -------------------------

_TEMPLATE_METADATA_PATTERN = re.compile(
    r"^\s*#\s*"
    r"(?P<key>lsfm_template_id|lsfm_schema_version|lsfm_template_revision)"
    r"\s*=\s*(?P<value>.+?)\s*$"
)


@dataclass(frozen=True)
class CanonicalConfigTemplate:
    resource_package: str
    resource_parts: tuple[str, ...]
    config_label: str = "Config"


@dataclass(frozen=True)
class TemplateMetadata:
    template_id: str | None = None
    schema_version: int | None = None
    template_revision: str | None = None


def load_toml_config(path: str | Path) -> dict[str, Any]:
    """
    Load a TOML file with recovery for unescaped Windows backslashes.

    Parameters
    ----------
    path : str | Path
        Path to the TOML file.

    Returns
    -------
    dict[str, Any]
        Parsed TOML configuration dictionary.
    """
    config_path = require_file(path, "Config file")
    config_text = config_path.read_text(encoding="utf-8")

    try:
        return tomllib.loads(config_text)
    except tomllib.TOMLDecodeError as exc:
        if "Unescaped '\\' in a string" not in str(exc):
            raise

        normalized_text = _normalize_backslashes_in_toml_strings(config_text)
        return tomllib.loads(normalized_text)


def prepare_script_config_path(
    script_path: Path,
    config_basename: str,
    *,
    test_mode: bool = False,
    canonical_template: CanonicalConfigTemplate | None = None,
    warn_on_stale: bool = False,
    write_stale_sidecar: bool = False,
) -> Path:
    """
    Resolve a script config path and optionally bootstrap `_local.toml` from a
    canonical package template.

    When a canonical template is provided and no local config exists, the local
    config is created and the process exits so the user can edit it before the
    workflow runs.
    """
    config_dir = script_path.parent / "configs"

    test_path = config_dir / f"{config_basename}_test.toml"
    local_path = config_dir / f"{config_basename}_local.toml"
    template_path = config_dir / f"{config_basename}_template.toml"

    if test_mode:
        if not test_path.exists():
            raise FileNotFoundError(
                "Test mode is enabled but no test config was found.\n"
                f"Expected:\n{test_path}"
            )
        return test_path

    if local_path.exists():
        if canonical_template is not None and warn_on_stale:
            for warning_message in _build_template_staleness_warnings(
                local_path,
                canonical_template,
                write_stale_sidecar=write_stale_sidecar,
            ):
                print(warning_message)
        return local_path

    if canonical_template is not None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(
            _read_canonical_template_text(canonical_template),
            encoding="utf-8",
        )
        print(
            f"Created local {canonical_template.config_label.lower()} "
            f"from canonical template:\n{local_path}"
        )
        print("Edit this file and rerun the script.")
        raise SystemExit(0)

    if template_path.exists():
        return template_path

    raise FileNotFoundError(
        "No config file found.\n"
        f"Expected:\n{local_path}\nOR\n{template_path}"
    )


def resolve_script_config_path(
    script_path: Path,
    config_basename: str,
    test_mode: bool = False,
) -> Path:
    """
    Resolve the config path using test/local/template precedence.
    """
    config_dir = script_path.parent / "configs"

    test_path = config_dir / f"{config_basename}_test.toml"
    local_path = config_dir / f"{config_basename}_local.toml"
    template_path = config_dir / f"{config_basename}_template.toml"

    if test_mode:
        if not test_path.exists():
            raise FileNotFoundError(
                "Test mode is enabled but no test config was found.\n"
                f"Expected:\n{test_path}"
            )
        config_path = test_path
    else:
        config_path = local_path if local_path.exists() else template_path

    if not config_path.exists():
        raise FileNotFoundError(
            "No config file found.\n"
            f"Expected:\n{local_path}\nOR\n{template_path}"
        )

    return config_path


def load_script_config(
    script_path: Path,
    config_basename: str,
    test_mode: bool = False,
    *,
    canonical_template: CanonicalConfigTemplate | None = None,
    warn_on_stale: bool = False,
    write_stale_sidecar: bool = False,
) -> dict[str, Any]:
    """
    Load a TOML configuration file using test/local/template precedence.

    The function searches for config files in a `configs/` folder located
    next to the script. It prefers a user-specific local config and falls
    back to a committed template config. When `canonical_template` is
    provided, missing local configs can be bootstrapped automatically from
    that canonical template.

    Search order
    ------------
    configs/<basename>_test.toml (only if test_mode=True; required)
    configs/<basename>_local.toml
    configs/<basename>_template.toml

    This supports reproducible repositories where template configs are
    version-controlled and local configs are gitignored.

    Parameters
    ----------
    script_path : Path
        Path to the running script file (__file__).
    config_basename : str
        Base name of the config (without suffix).
    test_mode : bool, optional
        If True, require and load <basename>_test.toml.
    canonical_template : CanonicalConfigTemplate | None, optional
        Canonical package template used to bootstrap <basename>_local.toml when
        it does not yet exist.
    warn_on_stale : bool, optional
        If True, compare local template metadata against the canonical template
        and print warnings when the local config is stale.
    write_stale_sidecar : bool, optional
        If True and a stale local config is detected, write a fresh canonical
        template next to the local config for comparison.

    Returns
    -------
    dict[str, Any]
        Parsed TOML configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If neither local nor template config file exists.
    tomllib.TOMLDecodeError
        If the TOML file is invalid.
    """
    config_path = prepare_script_config_path(
        script_path,
        config_basename,
        test_mode=test_mode,
        canonical_template=canonical_template,
        warn_on_stale=warn_on_stale,
        write_stale_sidecar=write_stale_sidecar,
    )
    cfg = load_toml_config(config_path)
    print(f"Using config: {config_path.name}")
    return cfg


def _build_template_staleness_warnings(
    local_path: Path,
    canonical_template: CanonicalConfigTemplate,
    *,
    write_stale_sidecar: bool,
) -> tuple[str, ...]:
    local_metadata = _extract_template_metadata(
        local_path.read_text(encoding="utf-8")
    )
    canonical_text = _read_canonical_template_text(canonical_template)
    canonical_metadata = _extract_template_metadata(canonical_text)

    if (
        canonical_metadata.template_id is None
        or canonical_metadata.schema_version is None
        or canonical_metadata.template_revision is None
    ):
        return ()

    if (
        local_metadata.template_id is None
        or local_metadata.schema_version is None
        or local_metadata.template_revision is None
    ):
        return ()

    if local_metadata.template_id != canonical_metadata.template_id:
        return ()

    warning_messages: list[str] = []

    if local_metadata.schema_version != canonical_metadata.schema_version:
        warning_messages.append(
            "Warning: "
            f"{local_path.name} was created from schema version "
            f"{local_metadata.schema_version}, but the current canonical template "
            f"uses schema version {canonical_metadata.schema_version}. "
            "Please review and update this local config manually."
        )
    elif local_metadata.template_revision != canonical_metadata.template_revision:
        warning_messages.append(
            "Warning: "
            f"{local_path.name} was created from template revision "
            f"{local_metadata.template_revision}, but the current canonical template "
            f"revision is {canonical_metadata.template_revision}."
        )

    if warning_messages and write_stale_sidecar:
        sidecar_path = _write_stale_template_sidecar(local_path, canonical_text)
        warning_messages.append(
            f"Wrote current canonical template for comparison:\n{sidecar_path}"
        )

    return tuple(warning_messages)


def _extract_template_metadata(config_text: str) -> TemplateMetadata:
    metadata_values: dict[str, str] = {}

    for line in config_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if not stripped.startswith("#"):
            break

        match = _TEMPLATE_METADATA_PATTERN.match(line)
        if match is None:
            continue

        metadata_values[match.group("key")] = match.group("value").strip()

    schema_version: int | None = None
    schema_version_value = metadata_values.get("lsfm_schema_version")
    if schema_version_value is not None:
        try:
            schema_version = int(schema_version_value)
        except ValueError:
            schema_version = None

    return TemplateMetadata(
        template_id=metadata_values.get("lsfm_template_id"),
        schema_version=schema_version,
        template_revision=metadata_values.get("lsfm_template_revision"),
    )


def _normalize_backslashes_in_toml_strings(config_text: str) -> str:
    """
    Convert backslashes to forward slashes inside TOML basic strings.

    This specifically helps user-edited Windows paths remain readable and
    parseable without requiring manual slash replacement in config files.
    """
    string_pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')

    def replace_match(match: re.Match[str]) -> str:
        string_content = match.group(1).replace("\\", "/")
        return f'"{string_content}"'

    return string_pattern.sub(replace_match, config_text)


def _read_canonical_template_text(
    canonical_template: CanonicalConfigTemplate,
) -> str:
    template_resource = importlib_resources.files(
        canonical_template.resource_package
    )
    for resource_part in canonical_template.resource_parts:
        template_resource = template_resource.joinpath(resource_part)
    return template_resource.read_text(encoding="utf-8")


def _write_stale_template_sidecar(local_path: Path, canonical_text: str) -> Path:
    sidecar_path = local_path.with_name(
        f"{local_path.stem}.new_template{local_path.suffix}"
    )
    sidecar_path.write_text(canonical_text, encoding="utf-8")
    return sidecar_path
