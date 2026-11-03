import json
import subprocess

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


def test_save_writes_models_predictions_results_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
    out = tmp_path / "out"
    manifest = artifacts.save(out, MODELS, _predictions(), _results_dir(tmp_path), INFO)
    for name in ["models.pkl", "predictions.parquet", "scores.csv", "split.json", "manifest.json"]:
        assert (out / name).exists()
    assert manifest["files"] == sorted(["models.pkl", "predictions.parquet", "scores.csv", "split.json"])


def test_manifest_records_commit_date_range_and_row_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
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
    out = tmp_path / "out"
    manifest = artifacts.save(out, MODELS, _predictions(), _results_dir(tmp_path, ["scores.csv"]), INFO)
    assert (out / "scores.csv").exists()
    assert not (out / "bootstrap.csv").exists()
    assert "bootstrap.csv" not in manifest["files"]


def test_saved_models_load_back(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
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
    out = tmp_path / "out"
    results = _results_dir(tmp_path)
    artifacts.save(out, MODELS, _predictions(), results, INFO)
    (results / "split.json").unlink()
    manifest = artifacts.save(out, MODELS, _predictions(), results, INFO)
    assert not (out / "split.json").exists()
    assert "split.json" not in manifest["files"]


def test_failed_save_leaves_no_stale_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git())
    out = tmp_path / "out"
    results = _results_dir(tmp_path)
    artifacts.save(out, MODELS, _predictions(), results, INFO)
    assert (out / "manifest.json").exists()
    with pytest.raises(TypeError):
        artifacts.save(out, MODELS, _predictions(), results, {"x": object()})
    assert not (out / "manifest.json").exists()


def test_git_returns_unknown_on_nonzero_exit(monkeypatch):
    def failed(args, *rest, **kwargs):
        return subprocess.CompletedProcess(args, 128, "", "")

    monkeypatch.setattr(artifacts.subprocess, "run", failed)
    assert artifacts._git("rev-parse", "HEAD") == "unknown"


def test_git_info_unknown_when_root_is_not_the_repo_top(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_git", _fake_git(toplevel=str(tmp_path)))
    info = artifacts.git_info()
    assert (info["git_commit"], info["src_tree"], info["git_dirty"]) == ("unknown", "unknown", "unknown")
