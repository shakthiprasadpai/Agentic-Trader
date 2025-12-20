from pathlib import Path

from scripts.check_env import parse_pyproject_dependencies, try_import


def test_parse_pyproject_dependencies():
    deps = parse_pyproject_dependencies(Path("pyproject.toml"))
    assert isinstance(deps, list)
    assert len(deps) > 0


def test_try_import_pydantic():
    ok, err = try_import("pydantic")
    assert ok, f"pydantic import failed: {err}"
