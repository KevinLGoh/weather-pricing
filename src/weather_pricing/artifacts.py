"""App artifacts: fitted models, out-of-sample predictions, results and a manifest.

`evaluate --save-artifacts` writes these to app_artifacts/, so the app can load
everything without retraining or calling any API. models.pkl is a pickle: only load
files you built or trust. XGBoost pickles need the same package versions, which the
manifest records.
"""

from __future__ import annotations

import json
import pickle
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from .config import ROOT

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
]

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
    the repo with identical code, even where commit hashes differ. Untracked files (such
    as app_artifacts/ itself) do not count as changes. If ROOT is not the top of a git
    repo (e.g. a copy installed elsewhere), every value is "unknown", so another
    repo's commit is never recorded.
    """
    top = _git("rev-parse", "--show-toplevel")
    if top == "unknown" or Path(top).resolve() != ROOT.resolve():
        return {"git_commit": "unknown", "src_tree": "unknown", "git_dirty": "unknown"}
    status = _git("status", "--porcelain", "--untracked-files=no")
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


def save(out_dir: Path, models: dict, predictions: pd.DataFrame, results_dir: Path, info: dict) -> dict:
    """Write models, predictions, the existing results files and manifest.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # A run that fails part-way must not leave an old manifest describing other files.
    (out_dir / "manifest.json").unlink(missing_ok=True)
    with open(out_dir / "models.pkl", "wb") as f:
        pickle.dump(models, f)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    files = ["models.pkl", "predictions.parquet"]
    for name in RESULT_FILES:
        src = Path(results_dir) / name
        if src.exists():
            shutil.copyfile(src, out_dir / name)
            files.append(name)
        else:
            # Do not leave a copy from an earlier run behind.
            (out_dir / name).unlink(missing_ok=True)
    # `info` first, so the keys this module sets always win. No default=str: values in
    # `info` must be plain JSON types, so the file matches the returned dict.
    manifest = {
        **info,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_info(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "predictions_rows": len(predictions),
        "models": sorted(models),
        "files": sorted(files),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_models(path: Path) -> dict:
    """Fitted models saved by `save`. Unpickling runs code: only load trusted files."""
    with open(path, "rb") as f:
        return pickle.load(f)
