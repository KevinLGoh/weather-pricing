import json

import pytest

from weather_pricing import http

URL = "https://example.test/api"
PARAMS = {"station": "KNYC", "limit": 5}
BODY = {"temps": [70, 71, 72]}


class FakeResponse:
    status_code = 200
    headers: dict = {}

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


def fake_get(monkeypatch, tmp_path, body=BODY):
    """Point the cache at tmp_path and replace the session GET; return the call list."""
    monkeypatch.setattr(http, "RAW_DIR", tmp_path)
    calls = []

    def fake(url, params=None, timeout=None):
        calls.append((url, params))
        return FakeResponse(body)

    monkeypatch.setattr(http._session, "get", fake)
    return calls


def test_cached_response_is_reused_without_a_request(monkeypatch, tmp_path):
    calls = fake_get(monkeypatch, tmp_path)
    first = http.get_json(URL, PARAMS, namespace="reuse", min_interval=0)
    second = http.get_json(URL, PARAMS, namespace="reuse", min_interval=0, use_cache=True)
    assert first == BODY
    assert second == BODY
    assert len(calls) == 1


def test_failed_cache_write_leaves_no_partial_file(monkeypatch, tmp_path):
    fake_get(monkeypatch, tmp_path)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(http.os, "replace", boom)
    with pytest.raises(OSError):
        http.get_json(URL, PARAMS, namespace="failed-write", min_interval=0)
    assert not http._cache_path("failed-write", URL, PARAMS).exists()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_truncated_cache_file_is_refetched(monkeypatch, tmp_path):
    calls = fake_get(monkeypatch, tmp_path)
    path = http._cache_path("truncated", URL, PARAMS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"url": "x", "bo')
    result = http.get_json(URL, PARAMS, namespace="truncated", min_interval=0, use_cache=True)
    assert result == BODY
    assert len(calls) == 1
    assert json.loads(path.read_text())["body"] == BODY
