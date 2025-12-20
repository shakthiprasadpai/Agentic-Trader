#!/usr/bin/env python3
"""Check environment and dependencies for this project.

Usage:
  python scripts/check_env.py       # run checks
  python scripts/check_env.py --fix # attempt to pip-install missing distributions into active Python

What it does:
- Reads `pyproject.toml` to get the declared dependencies
- Checks for installed distributions (via importlib.metadata)
- Performs targeted import checks for known modules (pydantic_core, pydantic, agents.*)
- Prints clear, actionable instructions to fix missing items
"""

from __future__ import annotations

import re
import sys
import subprocess
from pathlib import Path
import tomllib
from importlib import metadata
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_pyproject_dependencies(pyproject_path: Path) -> List[str]:
    if not pyproject_path.exists():
        return []
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    deps = data.get("project", {}).get("dependencies", [])
    # deps is a list of strings like "openai>=1.0.0" or "openai-agents[litellm,sqlalchemy]>=0.1.0"
    return list(deps)


def dep_to_name(dep: str) -> str:
    """Extract distribution name from a dependency spec (strip extras and versions)."""
    m = re.match(r"^\s*([^\[<>=!\s]+)", dep)
    return m.group(1) if m else dep


def check_distributions(deps: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    missing = []
    present = []

    for dep in deps:
        name = dep_to_name(dep)
        try:
            ver = metadata.version(name)
            present.append((name, ver))
        except metadata.PackageNotFoundError:
            missing.append(name)
    return missing, present


def try_import(module: str) -> Tuple[bool, str]:
    try:
        __import__(module)
        return True, ""
    except Exception as e:
        return False, str(e)


def main(argv: List[str]):
    fix = "--fix" in argv

    print(f"Checking environment using Python: {PYTHON}\nProject root: {ROOT}")

    deps = parse_pyproject_dependencies(ROOT / "pyproject.toml")
    print(f"Found {len(deps)} declared dependencies in pyproject.toml")

    missing, present = check_distributions(deps)

    if present:
        print("\nDistributions present:")
        for name, ver in present:
            print(f"  ✓ {name} ({ver})")

    if missing:
        print("\nDistributions missing:")
        for name in missing:
            print(f"  ✗ {name}")
        print("\nSuggested fix:")
        print(f"  {PYTHON} -m pip install {' '.join(missing)}")
        if fix:
            print("\nAttempting to install missing distributions (this will modify the active environment)...")
            try:
                subprocess.check_call([PYTHON, "-m", "pip", "install", *missing])
            except subprocess.CalledProcessError as e:
                print(f"Installation failed: {e}")
    else:
        print("\nAll declared distributions appear installed.")

    # Additional targeted import checks
    print("\nPerforming targeted import checks...")
    checks = [
        ("pydantic_core._pydantic_core", "pydantic-core"),
        ("pydantic", "pydantic"),
        ("agents.extensions.models.litellm_model", "agents.extensions.models.litellm_model (project or installed package)")
    ]

    failures = []
    for mod, label in checks:
        ok, err = try_import(mod)
        if ok:
            print(f"  ✓ import {mod}")
        else:
            print(f"  ✗ import {mod}: {err}")
            failures.append((mod, label, err))

    if failures:
        print("\nActionable suggestions for import failures:")
        for mod, label, err in failures:
            if mod.startswith("pydantic_core") or label == "pydantic":
                print(f"  - {label}: Try installing via: {PYTHON} -m pip install pydantic-core pydantic")
            elif mod.startswith("agents"):
                print(f"  - {label}: This module appears to be a project package or provided by a distribution like `openai-agents`. Try installing any missing provider packages: {PYTHON} -m pip install openai-agents")
            else:
                print(f"  - {label}: Could not import {mod}. Consider installing the appropriate package or adding the project root to PYTHONPATH.")

        print("\nIf these are project-local modules, ensure you run Python from the project root or add the project root to PYTHONPATH or 'python.analysis.extraPaths' in VS Code.")
        print("Non-zero exit code will be returned to indicate failures.")
        sys.exit(1)

    print("\nAll targeted imports succeeded. Environment looks good.")
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
