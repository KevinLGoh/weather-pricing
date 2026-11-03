import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)$")


def _lines(name):
    path = ROOT / name
    assert path.is_file(), f"{name} is missing"
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _normalise(name):
    return name.lower().replace("_", "-")


def _name(spec):
    return _normalise(re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0])


def _pins(name):
    pins = []
    for line in _lines(name):
        match = PIN.match(line)
        if match:
            pins.append((_normalise(match.group(1)), match.group(2)))
    return pins


def _project_dependency_names():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return [_name(spec) for spec in project["dependencies"]]


def test_requirements_pin_every_project_dependency_and_streamlit():
    lines = _lines("requirements.txt")
    pinned = [name for name, _ in _pins("requirements.txt")]
    for name in _project_dependency_names() + ["streamlit"]:
        assert pinned.count(name) == 1, f"{name} must be pinned exactly once with =="
        mentions = [line for line in lines if _name(line) == name]
        assert len(mentions) == 1, f"{name} appears {len(mentions)} times in requirements.txt"


def test_pinned_versions_match_the_installed_ones():
    pins = _pins("requirements.txt") + _pins("requirements-dev.txt")
    assert pins, "no == pins found"
    missing = []
    mismatches = []
    for name, version in pins:
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if installed != version:
            mismatches.append(f"{name}: pinned {version}, installed {installed}")
    if len(missing) == len(pins):
        pytest.skip(f"none of the pinned packages are installed: {missing}")
    assert not mismatches, mismatches


def test_requirements_install_the_project_itself():
    assert "." in _lines("requirements.txt")


def test_dev_requirements_include_the_app_requirements_and_pytest():
    lines = _lines("requirements-dev.txt")
    assert "-r requirements.txt" in lines
    assert any(re.fullmatch(r"pytest==\S+", line) for line in lines)


def test_gitignore_ignores_env_files_but_not_the_example():
    lines = [line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()]
    assert ".env" in lines
    assert ".env.*" in lines
    assert "!.env.example" in lines
    assert lines.index("!.env.example") > lines.index(".env.*"), "git applies the last matching rule"


def test_gitignore_ignores_build_output():
    lines = [line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()]
    assert "build/" in lines
