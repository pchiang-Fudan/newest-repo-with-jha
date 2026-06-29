from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


@dataclass(frozen=True)
class Prices:
    cache_hit_input_per_million: float = 0.0028
    cache_miss_input_per_million: float = 0.14
    output_per_million: float = 0.28


def repo_context() -> str:
    readme = "# Demo\n\nThis repository contains a small Python service.\n" * 250
    app = "def add(a, b):\n    return a + b\n\n" * 450
    tests = "def test_add():\n    assert add(1, 2) == 3\n\n" * 300
    return "\n".join(
        [
            "Repository context follows. Treat it as read-only reference.",
            "--- file: README.md",
            "```",
            readme.rstrip(),
            "```",
            "--- file: src/app.py",
            "```python",
            app.rstrip(),
            "```",
            "--- file: tests/test_app.py",
            "```python",
            tests.rstrip(),
            "```",
        ]
    )


TASKS = [
    "Find one likely improvement in this repository.",
    "Suggest one additional test for this repository.",
    "Explain the behavior of the add function in this repository.",
    "Identify whether this repository has enough tests.",
]


def naive_messages(task: str, context: str) -> list[dict]:
    # The volatile task appears before the stable repo context, so different
    # tasks produce different prefixes and should reduce prefix-cache hits.
    return [
        {"role": "system", "content": "You are a precise coding assistant."},
        {
            "role": "user",
            "content": f"Task: {task}\n\n{context}\n\nAnswer briefly.",
        },
    ]


def cache_aware_messages(task: str, context: str) -> list[dict]:
    # Stable repo context appears before the volatile task. Different tasks now
    # share a longer prefix, which should improve DeepSeek cache hits.
    return [
        {"role": "system", "content": "You are a precise coding assistant."},
        {"role": "system", "content": context},
        {"role": "user", "content": f"Task: {task}\n\nAnswer briefly."},
    ]


def call_deepseek(api_key: str, payload: dict) -> dict:
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {body}") from exc
    data["_latency_ms"] = (time.perf_counter() - start) * 1000.0
    return data


def cost_usd(usage: dict, prices: Prices) -> float:
    hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    miss = usage.get("prompt_cache_miss_tokens", 0) or 0
    out = usage.get("completion_tokens", 0) or 0
    return (
        hit / 1_000_000 * prices.cache_hit_input_per_million
        + miss / 1_000_000 * prices.cache_miss_input_per_million
        + out / 1_000_000 * prices.output_per_million
    )


def no_cache_cost_usd(usage: dict, prices: Prices) -> float:
    prompt = usage.get("prompt_tokens", 0) or 0
    out = usage.get("completion_tokens", 0) or 0
    return (
        prompt / 1_000_000 * prices.cache_miss_input_per_million
        + out / 1_000_000 * prices.output_per_million
    )


def summarize(label: str, responses: list[dict], prices: Prices) -> dict:
    usage_rows = [r.get("usage", {}) for r in responses]
    hit = sum(row.get("prompt_cache_hit_tokens", 0) or 0 for row in usage_rows)
    miss = sum(row.get("prompt_cache_miss_tokens", 0) or 0 for row in usage_rows)
    prompt = sum(row.get("prompt_tokens", 0) or 0 for row in usage_rows)
    out = sum(row.get("completion_tokens", 0) or 0 for row in usage_rows)
    total = sum(row.get("total_tokens", 0) or 0 for row in usage_rows)
    cost = sum(cost_usd(row, prices) for row in usage_rows)
    no_cache_cost = sum(no_cache_cost_usd(row, prices) for row in usage_rows)
    return {
        "label": label,
        "requests": len(responses),
        "avg_latency_ms": sum(r["_latency_ms"] for r in responses) / len(responses),
        "prompt_tokens": prompt,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "cache_hit_rate": hit / (hit + miss) if hit + miss else None,
        "completion_tokens": out,
        "total_tokens": total,
        "estimated_no_internal_cache_cost_usd": no_cache_cost,
        "estimated_cost_usd": cost,
        "estimated_internal_cache_savings_usd": no_cache_cost - cost,
        "estimated_internal_cache_savings_pct": (
            (no_cache_cost - cost) / no_cache_cost if no_cache_cost else None
        ),
        "cost_per_1m_total_tokens_usd": cost / total * 1_000_000 if total else None,
    }


def run_arm(
    label: str,
    api_key: str,
    model: str,
    tasks: list[str],
    context: str,
    cache_aware: bool,
    user_id: str,
) -> list[dict]:
    responses: list[dict] = []
    for index, task in enumerate(tasks, start=1):
        messages = (
            cache_aware_messages(task, context)
            if cache_aware
            else naive_messages(task, context)
        )
        payload = {
            "model": model,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 64,
            # DeepSeek docs say user_id isolates KVCache. For a shared repo-agent
            # cache experiment, use a stable repo-level user_id.
            "user_id": user_id,
        }
        response = call_deepseek(api_key, payload)
        usage = response.get("usage", {})
        print(
            json.dumps(
                {
                    "arm": label,
                    "request": index,
                    "latency_ms": response["_latency_ms"],
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
                    "cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "estimated_cost_usd": cost_usd(usage, Prices()),
                },
                indent=2,
            )
        )
        responses.append(response)
    return responses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--user-id", default="repo-demo-cache-test")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY first.")

    tasks = TASKS[: args.rounds]
    context = repo_context()
    prices = Prices()

    print("Running naive arm...")
    naive = run_arm(
        "naive",
        api_key,
        args.model,
        tasks,
        context,
        cache_aware=False,
        user_id=args.user_id,
    )

    print("Running cache-aware arm...")
    aware = run_arm(
        "cache-aware",
        api_key,
        args.model,
        tasks,
        context,
        cache_aware=True,
        user_id=args.user_id,
    )

    naive_summary = summarize("naive", naive, prices)
    aware_summary = summarize("cache-aware", aware, prices)
    no_internal_cache_cost = naive_summary["estimated_no_internal_cache_cost_usd"]
    deepseek_internal_cache_cost = naive_summary["estimated_cost_usd"]
    contextpilot_plus_deepseek_cost = aware_summary["estimated_cost_usd"]
    result = {
        "comparison": {
            "deepseek_api_no_internal_cache_counterfactual_usd": no_internal_cache_cost,
            "deepseek_api_internal_cache_only_observed_usd": deepseek_internal_cache_cost,
            "contextpilot_plus_deepseek_internal_cache_observed_usd": contextpilot_plus_deepseek_cost,
            "deepseek_internal_cache_savings_pct": (
                (no_internal_cache_cost - deepseek_internal_cache_cost)
                / no_internal_cache_cost
                if no_internal_cache_cost
                else None
            ),
            "contextpilot_incremental_savings_vs_deepseek_internal_cache_pct": (
                (deepseek_internal_cache_cost - contextpilot_plus_deepseek_cost)
                / deepseek_internal_cache_cost
                if deepseek_internal_cache_cost
                else None
            ),
            "total_savings_vs_no_internal_cache_pct": (
                (no_internal_cache_cost - contextpilot_plus_deepseek_cost)
                / no_internal_cache_cost
                if no_internal_cache_cost
                else None
            ),
        },
        "naive": naive_summary,
        "cache_aware": aware_summary,
        "estimated_savings_usd": naive_summary["estimated_cost_usd"]
        - aware_summary["estimated_cost_usd"],
        "estimated_savings_pct": (
            (naive_summary["estimated_cost_usd"] - aware_summary["estimated_cost_usd"])
            / naive_summary["estimated_cost_usd"]
            if naive_summary["estimated_cost_usd"]
            else None
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
