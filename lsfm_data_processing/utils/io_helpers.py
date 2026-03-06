from pathlib import Path
from typing import Any
import tomllib


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


# -------------------------
# CONFIG LOADER
# -------------------------

def load_script_config(script_path: Path, config_basename: str, test_mode: bool = False) -> dict[str, Any]:
    """
    Load a TOML configuration file using test/local/template precedence.

    The function searches for config files in a `configs/` folder located
    next to the script. It prefers a user-specific local config and falls
    back to a committed template config.

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

    with open(config_path, "rb") as f:
        cfg: dict[str, Any] = tomllib.load(f)

    print(f"Using config: {config_path.name}")
    return cfg
