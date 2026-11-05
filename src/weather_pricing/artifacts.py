"""App artifacts: fitted models, out-of-sample predictions, results and a manifest.

`app-artifacts` (every city, all models) or `evaluate --save-artifacts` (one combined run)
writes these to app_artifacts/, so the app can load everything without retraining or
calling any API. models.pkl is a pickle: only load
files you built or trust. XGBoost pickles need the same package versions, which the
manifest records.
"""

from __future__ import annotations

import json
import os
import pickle
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, ROOT

ARTIFACTS_DIR = ROOT / "app_artifacts"

RESULT_FILES = [
    "scores.csv",
    "bootstrap.csv",
    "encompassing.csv",
    "robustness.csv",
    "reliability.csv",
    "valuation.json",
    "split.json",
    "summary.md",
    "data_quality.json",
    "encompassing_extended.csv",
]
# Multi-city layout: evidence/<city>/<results file> and finding/<table>.
EVIDENCE_DIR = "evidence"
FINDING_DIR = "finding"

DIRTY_STATUS_ARGS = ("--porcelain", "--untracked-files=no", "--", ".", f":(exclude){ARTIFACTS_DIR.name}")

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "pyarrow"]


def _git(*args: str) -> str:
    """Output of `git -C ROOT <args>`, or "unknown" if git is missing or fails.

    --no-optional-locks keeps `git status` from writing to .git (read-only use)."""
    cmd = ["git", "--no-optional-locks", "-C", str(ROOT), *args]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return done.stdout.strip() if done.returncode == 0 else "unknown"


def git_info() -> dict:
    """Commit the artifacts were built from, and whether tracked files had changes.

    `src_tree` is the hash of the src/ folder's content. It is the same in every copy of
    the repo with identical code, even where commit hashes differ. Untracked files and
    app_artifacts/ itself do not count as changes: a rebuild overwrites those tracked
    files, which says nothing about the code it ran. If ROOT is not the top of a git repo
    (e.g. a copy installed elsewhere), every value is "unknown", so another repo's commit
    is never recorded.
    """
    top = _git("rev-parse", "--show-toplevel")
    if top == "unknown" or Path(top).resolve() != ROOT.resolve():
        return {"git_commit": "unknown", "src_tree": "unknown", "git_dirty": "unknown"}
    status = _git("status", *DIRTY_STATUS_ARGS)
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "src_tree": _git("rev-parse", "HEAD:src"),
        "git_dirty": "unknown" if status == "unknown" else status != "",
    }


def _package_versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def _noaa_accessed() -> str:
    """UTC date of the newest cached NOAA response (http.get_json stores "fetched"), the
    access date NOAA asks for in the GHCN-Daily citation; "unknown" if none can be read.
    The newest one, because the current year is fetched again on every build."""
    newest = None
    for path in sorted((RAW_DIR / "noaa").glob("*.json")):
        try:
            fetched = json.loads(path.read_text())["fetched"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(fetched, (int, float)) and not isinstance(fetched, bool):
            newest = fetched if newest is None else max(newest, fetched)
    if newest is None:
        return "unknown"
    return datetime.fromtimestamp(newest, tz=timezone.utc).date().isoformat()


def _start(out_dir: Path, models: dict, predictions: pd.DataFrame) -> list[str]:
    """Clear what an earlier run left and write the models and predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # A run that fails part-way must not leave an old manifest describing other files.
    (out_dir / "manifest.json").unlink(missing_ok=True)
    # Results of another layout must not be read alongside the new predictions.
    for folder in (EVIDENCE_DIR, FINDING_DIR):
        shutil.rmtree(out_dir / folder, ignore_errors=True)
    with open(out_dir / "models.pkl", "wb") as f:
        pickle.dump(models, f)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    return ["models.pkl", "predictions.parquet"]


def _copy_results(results_dir: Path, out_dir: Path) -> list[str]:
    """Copy the results files that exist and remove copies of those that do not."""
    copied = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in RESULT_FILES:
        src = Path(results_dir) / name
        if src.exists():
            shutil.copyfile(src, out_dir / name)
            copied.append(name)
        else:
            # Do not leave a copy from an earlier run behind.
            (out_dir / name).unlink(missing_ok=True)
    return copied


def save(out_dir: Path, models: dict, predictions: pd.DataFrame, results_dir: Path, info: dict) -> dict:
    """Write models, predictions, the existing results files and manifest.json."""
    out_dir = Path(out_dir)
    files = _start(out_dir, models, predictions)
    files += _copy_results(results_dir, out_dir)
    return _write_manifest(out_dir, sorted(models), predictions, files, info)


def save_cities(
    out_dir: Path,
    models: dict,
    model_names: list[str],
    predictions: pd.DataFrame,
    city_results: dict[str, Path],
    finding: dict[str, Path],
    info: dict,
) -> dict:
    """Multi-city artifacts: one evidence folder per city, plus the tables across cities.

    `models` holds each city's fitted models and `model_names` the predictions' model
    columns. `finding` maps a file name under finding/ to its source. Top-level results
    files from the single-run layout are removed, so the app cannot mistake them for current
    ones.
    """
    out_dir = Path(out_dir)
    files = _start(out_dir, models, predictions)
    for name in RESULT_FILES:
        (out_dir / name).unlink(missing_ok=True)
    for city, results_dir in city_results.items():
        files += [f"{EVIDENCE_DIR}/{city}/{n}" for n in _copy_results(results_dir, out_dir / EVIDENCE_DIR / city)]
    (out_dir / FINDING_DIR).mkdir(parents=True, exist_ok=True)
    for name, src in finding.items():
        shutil.copyfile(src, out_dir / FINDING_DIR / name)
        files.append(f"{FINDING_DIR}/{name}")
    return _write_manifest(out_dir, sorted(model_names), predictions, files, info)


def _write_manifest(out_dir: Path, model_names: list[str], predictions: pd.DataFrame, files: list[str], info: dict) -> dict:
    # `info` first, so the keys this module sets always win. No default=str: values in
    # `info` must be plain JSON types, so the file matches the returned dict.
    manifest = {
        **info,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_info(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "noaa_accessed": _noaa_accessed(),
        "predictions_rows": len(predictions),
        "models": model_names,
        "files": sorted(files),
    }
    # Temp file + rename, so a crash mid-write never leaves a truncated manifest.
    path = out_dir / "manifest.json"
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return manifest


def load_models(path: Path) -> dict:
    """Fitted models saved by `save`. Unpickling runs code: only load trusted files."""
    with open(path, "rb") as f:
        return pickle.load(f)
