from __future__ import annotations

import json
import urllib.request


BASE = "http://127.0.0.1:8000"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    print(
        post(
            "/v1/repos/demo/context",
            {
                "commit_hash": "abc123",
                "blocks": [
                    {
                        "path": "README.md",
                        "content": "# Demo\n\nThis repo demonstrates cache-aware prompts.",
                    },
                    {
                        "path": "src/app.py",
                        "content": "def add(a, b):\n    return a + b\n",
                    },
                ],
            },
        )
    )


if __name__ == "__main__":
    main()

