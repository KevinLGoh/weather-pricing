import json
import subprocess
from datetime import datetime, timezone

import pandas as pd
import pytest

from weather_pricing import artifacts

INFO = {
    "data_date_range": ["2025-09-01", "2026-10-31"],
    "cities": ["NY"],
    "rows_in": 100,
    "rows_used": 90,
    "train": {"rows": 60},
    "test": {"rows": 30},
    "settings": {"test_frac": 0.3},
}
MODELS = {"R_gaussian_rule": {"bias": 1.0}}


def _fake_git(commit="abc123", tree="def456", status="", calls=None, toplevel=str(artifacts.ROOT)):
    answers = {
        ("rev-parse", "--show-toplevel"): toplevel,
        ("rev-parse", "HEAD"): commit,
        ("rev-parse", "HEAD:src"): tree,
        ("status", "--porcelain", "--untracked-files=no"): status,
    }

    def fake(*args):
        if calls is not None:
            calls.append(args)
        return answers.get(args, "unknown")

    return fake


def _predictions():
    return pd.DataFrame(
        {
            "city": ["NY", "NY", "NY"],
            "ticker": ["T-A", "T-B", "T-C"],
            "target_date": ["2026-10-01", "2026-10-02", "2026-10-03"],
            "snapshot": ["d-1", "d-1", "d-0"],
            "y": [1, 0, 1],
            "A_logit": [0.5, -0.2, 1.1],
        }
    )


def _results_dir(tmp_path, names=("scores.csv", "split.json")):
    results = tmp_path / "results"
    results.mkdir()
    for name in names:
        (results / name).write_text("x\n1\n" if name.endswith(".csv") else '{"a": 1}')
    return results


def _raw_dir(monkeypatch, path):
    """Point save() at a test raw folder, so it never reads the real data/raw/noaa."""
    monkeypatch.setattr(artifacts, "RAW_DIR", path, raising=False)


def _noaa_cache_file(folder, name, fetched):
    folder.mkdir(parents=True, exist_ok=True)
    body = {"url": "https://example.test/noaa", "params": {}, "fetched": fetched, "body": {}}
    (folder / name).write_text(json.dumps(body))


def _epoch(*args):
    return datetime(*args, tzinfo=timezone.utc).timestamp()


def test_save_writes_models_predictions_results_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
    _raw_dir(monkeypatch, tmp_path / "raw")
    out = tmp_path / "out"
    manifest = artifacts.save(out, MODELS, _predictions(), _results_dir(tmp_path), INFO)
    for name in ["models.pkl", "predictions.parquet", "scores.csv", "split.json", "manifest.json"]:
        assert (out / name).exists()
    assert manifest["files"] == sorted(["models.pkl", "predictions.parquet", "scores.csv", "split.json"])


def test_manifest_records_commit_date_range_and_row_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
    _raw_dir(monkeypatch, tmp_path / "raw")
    out = tmp_path / "out"
    manifest = artifacts.save(out, MODELS, _predictions(), _results_dir(tmp_path), INFO)
    assert (manifest["git_commit"], manifest["src_tree"], manifest["git_dirty"]) == ("abc123", "def456", False)
    assert manifest["data_date_range"] == ["2025-09-01", "2026-10-31"]
    assert (manifest["rows_in"], manifest["rows_used"]) == (100, 90)
    assert (manifest["train"]["rows"], manifest["test"]["rows"]) == (60, 30)
    assert manifest["predictions_rows"] == 3
    assert manifest["models"] == ["R_gaussian_rule"]
    assert "xgboost" in manifest["packages"]
    assert json.loads((out / "manifest.json").read_text()) == json.loads(json.dumps(manifest))


def test_manifest_records_unknown_when_git_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", lambda *args: "unknown")
    _raw_dir(monkeypatch, tmp_path / "raw")
    manifest = artifacts.save(tmp_path / "out", MODELS, _predictions(), _results_dir(tmp_path), INFO)
    assert (manifest["git_commit"], manifest["src_tree"], manifest["git_dirty"]) == ("unknown", "unknown", "unknown")


def test_git_dirty_ignores_untracked_files(monkeypatch):
    calls = []
    monkeypatch.setattr(artifacts, "_git", _fake_git(status="", calls=calls))
    assert artifacts.git_info()["git_dirty"] is False
    assert ("status", "--porcelain", "--untracked-files=no") in calls
    assert all(c[0] != "status" or c == ("status", "--porcelain", "--untracked-files=no") for c in calls)
    monkeypatch.setattr(artifacts, "_git", _fake_git(status=" M src/weather_pricing/models.py"))
    assert artifacts.git_info()["git_dirty"] is True
    monkeypatch.setattr(artifacts, "_git", _fake_git(status="unknown"))
    assert artifacts.git_info()["git_dirty"] == "unknown"


def test_missing_results_file_is_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
    _raw_dir(monkeypatch, tmp_path / "raw")
    out = tmp_path / "out"
    manifest = artifacts.save(out, MODELS, _predictions(), _results_dir(tmp_path, ["scores.csv"]), INFO)
    assert (out / "scores.csv").exists()
    assert not (out / "bootstrap.csv").exists()
    assert "bootstrap.csv" not in manifest["files"]


def test_saved_models_load_back(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
    _raw_dir(monkeypatch, tmp_path / "raw")
    out = tmp_path / "out"
    artifacts.save(out, MODELS, _predictions(), _results_dir(tmp_path), INFO)
    assert artifacts.load_models(out / "models.pkl") == MODELS
    pd.testing.assert_frame_equal(pd.read_parquet(out / "predictions.parquet"), _predictions())


def test_git_returns_unknown_when_command_fails(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(artifacts.subprocess, "run", boom)
    assert artifacts._git("rev-parse", "HEAD") == "unknown"


def test_rebuild_removes_results_files_that_no_longer_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
    _raw_dir(monkeypatch, tmp_path / "raw")
    out = tmp_path / "out"
    results = _results_dir(tmp_path)
    artifacts.save(out, MODELS, _predictions(), results, INFO)
    (results / "split.json").unlink()
    manifest = artifacts.save(out, MODELS, _predictions(), results, INFO)
    assert not (out / "split.json").exists()
    assert "split.json" not in manifest["files"]


def test_failed_save_leaves_no_stale_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
    _raw_dir(monkeypatch, tmp_path / "raw")
    out = tmp_path / "out"
    results = _results_dir(tmp_path)
    artifacts.save(out, MODELS, _predictions(), results, INFO)
    assert (out / "manifest.json").exists()
    with pytest.raises(TypeError):
        artifacts.save(out, MODELS, _predictions(), results, {"x": object()})
    assert not (out / "manifest.json").exists()


def test_failed_manifest_write_leaves_no_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
    _raw_dir(monkeypatch, tmp_path / "raw")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(artifacts.os, "replace", boom)
    out = tmp_path / "out"
    with pytest.raises(OSError):
        artifacts.save(out, MODELS, _predictions(), _results_dir(tmp_path), INFO)
    assert not (out / "manifest.json").exists()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_git_returns_unknown_on_nonzero_exit(monkeypatch):
    def failed(args, *rest, **kwargs):
        return subprocess.CompletedProcess(args, 128, "", "")

    monkeypatch.setattr(artifacts.subprocess, "run", failed)
    assert artifacts._git("rev-parse", "HEAD") == "unknown"


def test_git_info_unknown_when_root_is_not_the_repo_top(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git(toplevel=str(tmp_path)))
    info = artifacts.git_info()
    assert (info["git_commit"], info["src_tree"], info["git_dirty"]) == ("unknown", "unknown", "unknown")


def test_manifest_records_the_newest_noaa_access_date(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
    _raw_dir(monkeypatch, tmp_path / "raw")
    noaa = tmp_path / "raw" / "noaa"
    # The newest file sorts first by name, so "last file read" would give the wrong date.
    _noaa_cache_file(noaa, "a.json", _epoch(2026, 11, 4, 1, 15))
    _noaa_cache_file(noaa, "b.json", _epoch(2026, 11, 2, 23, 30))
    (noaa / "broken.json").write_text("{")
    out = tmp_path / "out"
    manifest = artifacts.save(out, MODELS, _predictions(), _results_dir(tmp_path), INFO)
    assert manifest["noaa_accessed"] == "2026-11-04"
    assert json.loads((out / "manifest.json").read_text())["noaa_accessed"] == "2026-11-04"


def test_manifest_noaa_access_date_unknown_without_cache_files(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
    _raw_dir(monkeypatch, tmp_path / "raw")
    results = _results_dir(tmp_path)
    manifest = artifacts.save(tmp_path / "out", MODELS, _predictions(), results, INFO)
    assert manifest["noaa_accessed"] == "unknown"
    noaa = tmp_path / "raw" / "noaa"
    _noaa_cache_file(noaa, "text.json", "yesterday")
    _noaa_cache_file(noaa, "missing.json", None)
    (noaa / "broken.json").write_text("{")
    manifest = artifacts.save(tmp_path / "out", MODELS, _predictions(), results, INFO)
    assert manifest["noaa_accessed"] == "unknown"
