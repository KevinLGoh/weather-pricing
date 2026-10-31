"""Read-only HTTP GET with an on-disk cache and polite retries.

Every response is stored under data/raw/<namespace>/ keyed by URL+params, so the dataset
can be rebuilt offline and the exact bytes used in an experiment are kept.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests

from .config import RAW_DIR

_session = requests.Session()
_session.headers["User-Agent"] = "weather-pricing-research/0.1"
_last_call: dict[str, float] = {}
_lock = threading.Lock()


def _cache_path(namespace: str, url: str, params: dict[str, Any]) -> Path:
    key = json.dumps([url, sorted(params.items())], default=str)
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return RAW_DIR / namespace / f"{digest}.json"


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    namespace: str,
    min_interval: float = 0.1,
    use_cache: bool = True,
    retries: int = 8,
) -> Any:
    params = {k: v for k, v in (params or {}).items() if v is not None}
    path = _cache_path(namespace, url, params)
    if use_cache and path.exists():
        return json.loads(path.read_text())["body"]

    for attempt in range(retries):
        # Reserve a send slot so concurrent callers stay under min_interval in aggregate.
        with _lock:
            slot = max(time.monotonic(), _last_call.get(namespace, 0.0) + min_interval)
            _last_call[namespace] = slot
        time.sleep(max(0.0, slot - time.monotonic()))
        try:
            resp = _session.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2**attempt)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After", "")
            time.sleep(float(retry_after) if retry_after.isdigit() else min(60, 2**attempt))
            continue
        resp.raise_for_status()
        body = resp.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"url": url, "params": params, "fetched": time.time(), "body": body}))
        return body
    raise RuntimeError(f"GET failed after {retries} attempts: {url} {params}")
