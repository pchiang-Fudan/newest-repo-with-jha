from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


DEFAULT_FILES = [
    "cache_proxy/README.md",
    "cache_proxy/server.py",
    "cache_proxy/context_store.py",
    "cache_proxy/prompt_compiler.py",
    "cache_proxy/ab_test_deepseek.py",
    "cache_proxy/ab_test_workspace_repo.py",
    "cache_proxy/prefix_overlap_sanity.py",
    "tests/test_cache_proxy.py",
    "memos/kv_first_inference_architecture.md",
    "experiments/README.md",
    "experiments/bitnet_ppa_sim.py",
]


TASKS = [
    "Review the proxy implementation and identify the most important bug or limitation.",
    "Suggest one concrete test that should be added for the cache-aware prompt compiler.",
    "Explain how the telemetry summary works and one way it could be misleading.",
    "Propose a small refactor that would make provider adapters easier to add.",
]


@dataclass(frozen=True)
class Prices:
    cache_hit_input_per_million: float = 0.0028
    cache_miss_input_per_million: float = 0.14
    output_per_million: float = 0.28


def read_repo_context(
    paths: list[str],
    max_chars_per_file: int,
    *,
    canonical: bool,
) -> str:
    parts: list[str] = [
        "Repository context follows. Treat it as read-only reference.",
        "",
    ]
    if canonical:
        parts.insert(1, f"repo_root: {ROOT}")
        ordered_paths = sorted(paths)
    else:
        ordered_paths = paths

    for rel in ordered_paths:
        path = ROOT / rel
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        truncated = content[:max_chars_per_file]
        parts.extend(
            [
                f"--- file: {rel}",
                "```",
                truncated.rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(parts).strip()


def naive_messages(task: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a precise coding agent."},
        {
            "role": "user",
            "content": f"Task: {task}\n\n{context}\n\nAnswer with one concise paragraph.",
        },
    ]


def cache_aware_messages(task: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a precise coding agent."},
        {"role": "system", "content": context},
        {"role": "user", "content": f"Task: {task}\n\nAnswer with one concise paragraph."},
    ]


def deepseek_native_messages(task: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a precise coding agent."},
        {
            "role": "user",
            "content": f"{context}\n\nTask: {task}\n\nAnswer with one concise paragraph.",
        },
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
    layout: str,
    user_id: str,
) -> list[dict]:
    responses: list[dict] = []
    for index, task in enumerate(tasks, start=1):
        if layout == "volatile-first":
            messages = naive_messages(task, context)
        elif layout == "stable-first":
            messages = deepseek_native_messages(task, context)
        elif layout == "contextpilot":
            messages = cache_aware_messages(task, context)
        else:
            raise ValueError(f"unknown layout: {layout}")
        payload = {
            "model": model,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 96,
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
    parser.add_argument("--user-id", default="workspace-repo-cache-test")
    parser.add_argument("--max-chars-per-file", type=int, default=40_000)
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY first.")

    native_context = read_repo_context(
        args.files,
        args.max_chars_per_file,
        canonical=False,
    )
    contextpilot_context = read_repo_context(
        args.files,
        args.max_chars_per_file,
        canonical=True,
    )
    tasks = TASKS[: args.rounds]
    prices = Prices()
    print(
        json.dumps(
            {
                "native_context_chars": len(native_context),
                "contextpilot_context_chars": len(contextpilot_context),
                "files": [path for path in args.files if (ROOT / path).exists()],
                "rounds": len(tasks),
            },
            indent=2,
        )
    )

    print("Running naive arm...")
    naive = run_arm(
        "naive",
        api_key,
        args.model,
        tasks,
        native_context,
        layout="volatile-first",
        user_id=f"{args.user_id}-volatile-first",
    )

    print("Running DeepSeek-native stable-prefix arm...")
    native = run_arm(
        "deepseek-native-stable-prefix",
        api_key,
        args.model,
        tasks,
        native_context,
        layout="stable-first",
        user_id=f"{args.user_id}-native",
    )

    print("Running cache-aware arm...")
    aware = run_arm(
        "cache-aware",
        api_key,
        args.model,
        tasks,
        contextpilot_context,
        layout="contextpilot",
        user_id=f"{args.user_id}-contextpilot",
    )

    naive_summary = summarize("naive", naive, prices)
    native_summary = summarize("deepseek-native-stable-prefix", native, prices)
    aware_summary = summarize("cache-aware", aware, prices)
    no_internal_cache_cost = naive_summary["estimated_no_internal_cache_cost_usd"]
    deepseek_internal_cache_cost = native_summary["estimated_cost_usd"]
    contextpilot_plus_deepseek_cost = aware_summary["estimated_cost_usd"]
    print(
        json.dumps(
            {
                "comparison": {
                    "deepseek_api_no_internal_cache_counterfactual_usd": no_internal_cache_cost,
                    "deepseek_api_internal_cache_with_reasonable_stable_prefix_usd": deepseek_internal_cache_cost,
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
                "unoptimized_volatile_first": naive_summary,
                "deepseek_native_stable_prefix": native_summary,
                "naive": naive_summary,
                "cache_aware": aware_summary,
                "estimated_savings_usd": native_summary["estimated_cost_usd"]
                - aware_summary["estimated_cost_usd"],
                "estimated_savings_pct": (
                    (
                        native_summary["estimated_cost_usd"]
                        - aware_summary["estimated_cost_usd"]
                    )
                    / native_summary["estimated_cost_usd"]
                    if native_summary["estimated_cost_usd"]
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
