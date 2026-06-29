from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    data.setdefault("_client", {})
    data["_client"]["latency_ms"] = (time.perf_counter() - start) * 1000.0
    return data


def usage_summary(response: dict) -> dict:
    usage = response.get("usage", {})
    hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    miss = usage.get("prompt_cache_miss_tokens", 0) or 0
    prompt = usage.get("prompt_tokens", 0) or 0
    return {
        "latency_ms": response.get("_client", {}).get("latency_ms"),
        "prompt_tokens": prompt,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "cache_hit_rate": hit / (hit + miss) if hit + miss else None,
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("CACHE_PROXY_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--repo-id", default="demo")
    parser.add_argument("--commit-hash", default="abc123")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    context = {
        "commit_hash": args.commit_hash,
        "blocks": [
            {
                "path": "README.md",
                "content": "# Demo\n\nThis repository contains a tiny Python app.\n" * 200,
            },
            {
                "path": "src/app.py",
                "content": "def add(a, b):\n    return a + b\n\n" * 300,
            },
        ],
    }
    print(post_json(f"{args.base_url}/v1/repos/{args.repo_id}/context", context))

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a precise coding assistant."},
            {"role": "user", "content": "Review the repository and identify one likely improvement."},
        ],
        "cache_context": {
            "repo_id": args.repo_id,
            "commit_hash": args.commit_hash,
            "selected_blocks": ["README.md", "src/app.py"],
        },
        "max_tokens": 80,
        "temperature": 0,
    }

    for i in range(args.rounds):
        response = post_json(f"{args.base_url}/v1/chat/completions", payload)
        print(json.dumps({"round": i + 1, **usage_summary(response)}, indent=2))


if __name__ == "__main__":
    main()

