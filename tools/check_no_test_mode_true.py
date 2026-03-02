"""Fail if any Python file sets `test_mode = True` in code.

Usage:
- Run `python tools/check_no_test_mode_true.py` before committing.
- Intended as a guardrail to avoid committing development test-mode toggles.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Optional: skip common noisy dirs
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
            ".pytest_cache", "build", "dist", "tools"}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def has_test_mode_true_assignment(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if (
                isinstance(node.value, ast.Constant)
                and node.value.value is True
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "test_mode":
                        return True

        if isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "test_mode"
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
            ):
                return True

    return False


def main() -> int:
    bad_files: list[str] = []
    for p in Path(".").rglob("*.py"):
        if should_skip(p):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if has_test_mode_true_assignment(text):
            bad_files.append(str(p))

    if bad_files:
        print("ERROR: Found test_mode enabled (True) in the following file(s):")
        print("\n".join(bad_files))
        print("\nDisable test_mode before committing.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
