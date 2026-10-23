import hashlib
import json
import os
import time
from pathlib import Path

import requests

API = "https://api.github.com"


def load_env(path=".env"):
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("GITHUB_TOKEN", "HASH_SALT"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


class GitHub:
    def __init__(self, token, cache_dir="data/cache", min_remaining=50):
        if not token:
            raise SystemExit("GITHUB_TOKEN missing. Add it to .env")
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_remaining = min_remaining
        self.calls = 0

    def _cache_path(self, url, params):
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        return self.cache / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def get(self, path, params=None, use_cache=True):
        url = path if path.startswith("http") else API + path
        cp = self._cache_path(url, params)
        if use_cache and cp.exists():
            return json.loads(cp.read_text())
        for attempt in range(6):
            r = self.s.get(url, params=params, timeout=60)
            self.calls += 1
            if r.status_code in (403, 429) and (
                r.headers.get("Retry-After") or r.headers.get("X-RateLimit-Remaining") == "0"
            ):
                self._wait(r)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (404, 422):
                data = None
                break
            r.raise_for_status()
            data = r.json()
            self._throttle(r)
            break
        else:
            raise RuntimeError(f"Failed after retries: {url}")
        if use_cache:
            cp.write_text(json.dumps(data))
        return data

    def _wait(self, r):
        if r.headers.get("Retry-After"):
            delay = int(r.headers["Retry-After"]) + 1
        else:
            delay = max(int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time(), 0) + 5
        print(f"rate limited, sleeping {int(delay)}s", flush=True)
        time.sleep(delay)

    def _throttle(self, r):
        remaining = int(r.headers.get("X-RateLimit-Remaining", 1000))
        if remaining < self.min_remaining:
            self._wait(r)


def hash_id(value, salt):
    if not value:
        return ""
    return hashlib.sha256((salt + value.lower()).encode()).hexdigest()[:12]
