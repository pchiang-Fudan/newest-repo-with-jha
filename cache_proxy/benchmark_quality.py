from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from .ab_test_workspace_repo import Prices, cost_usd, summarize
from .benchmark_messy_agents import (
    PreparedRequest,
    WorkloadTask,
    adaptive_context_router_requests,
    build_repo_blocks,
    contextpilot_messages,
    included_full_context_paths,
    local_usage,
    read_repo_blocks,
    serialized_prompt,
    upstream_budget_shadow_pruned_contextpilot_requests,
    upstream_contextpilot_requests,
)


@dataclass(frozen=True)
class QualityTask:
    task: str
    family: str
    expected_paths: tuple[str, ...]
    expected_terms: tuple[str, ...]
    explicit_paths: tuple[str, ...] = ()


QUALITY_TASKS = [
    QualityTask(
        task="Review chat_completions request handling and identify one cache-risk.",
        family="code_review",
        expected_paths=(
            "cache_proxy/server.py",
        ),
        expected_terms=("chat_completions", "plan_context_blocks", "risk", "cache"),
    ),
    QualityTask(
        task="Diagnose why plan_context_blocks file selection is flaky in pytest.",
        family="test_debug",
        expected_paths=(
            "cache_proxy/context_planner.py",
            "tests/test_cache_proxy.py",
        ),
        expected_terms=("plan_context_blocks", "pytest", "full_blocks", "regression"),
    ),
    QualityTask(
        task="Plan a provider adapter split around compile_planned_messages and request forwarding.",
        family="refactor",
        expected_paths=(
            "cache_proxy/server.py",
            "cache_proxy/prompt_compiler.py",
        ),
        expected_terms=("provider", "adapter", "messages", "compiled"),
    ),
    QualityTask(
        task="Explain the ContextPilot benchmark result and onboarding notes for docs readers.",
        family="docs",
        expected_paths=(
            "cache_proxy/README.md",
        ),
        expected_terms=("benchmark", "adaptive-router", "context", "onboarding"),
    ),
    QualityTask(
        task="Continue the adaptive-router handoff and avoid repeating completed benchmark work.",
        family="handoff",
        expected_paths=(
            "cache_proxy/benchmark_messy_agents.py",
            "cache_proxy/README.md",
        ),
        expected_terms=("handoff", "budget", "adaptive", "required"),
    ),
    QualityTask(
        task="Audit policy-cache hazards in synthetic_repo/security/policies_57.py.",
        family="synthetic_security",
        explicit_paths=("synthetic_repo/security/policies_57.py",),
        expected_paths=(
            "synthetic_repo/security/policies_57.py",
        ),
        expected_terms=("security", "policy", "cache", "hazard"),
    ),
    QualityTask(
        task="Inspect synthetic_repo/ui/panels_59.py and describe panel telemetry.",
        family="synthetic_ui",
        explicit_paths=("synthetic_repo/ui/panels_59.py",),
        expected_paths=(
            "synthetic_repo/ui/panels_59.py",
        ),
        expected_terms=("ui", "panel", "telemetry", "metric"),
    ),
    QualityTask(
        task="Find the likely cache_proxy ContextStore telemetry persistence risk.",
        family="synthetic_billing",
        expected_paths=(
            "cache_proxy/context_store.py",
        ),
        expected_terms=("telemetry", "sqlite", "cache", "risk"),
    ),
]


def quality_tasks(rounds: int) -> list[QualityTask]:
    return [QUALITY_TASKS[index % len(QUALITY_TASKS)] for index in range(rounds)]


def to_workload_tasks(tasks: list[QualityTask]) -> list[WorkloadTask]:
    return [
        WorkloadTask(
            task=task.task,
            family=task.family,
            required_paths=task.explicit_paths,
        )
        for task in tasks
    ]


def quiet_local_responses(requests: list[PreparedRequest]) -> list[dict]:
    cached_prompts: list[str] = []
    responses = []
    for request in requests:
        usage = local_usage(request.messages, cached_prompts)
        cached_prompts.append(serialized_prompt(request.messages))
        responses.append({"_latency_ms": 0.0, "usage": usage})
    return responses


def message_text(request: PreparedRequest) -> str:
    return "\n".join(str(message.get("content", "")) for message in request.messages)


def available_full_paths(
    request: PreparedRequest,
    all_paths: set[str],
) -> set[str]:
    included = included_full_context_paths(request)
    return set(all_paths) if included is None else included


def score_request(
    request: PreparedRequest,
    task: QualityTask,
    all_paths: set[str],
) -> dict:
    full_paths = available_full_paths(request, all_paths)
    expected_paths = set(task.expected_paths)
    found_paths = expected_paths & full_paths
    path_recall = len(found_paths) / len(expected_paths) if expected_paths else 1.0

    text = message_text(request).lower()
    expected_terms = set(term.lower() for term in task.expected_terms)
    found_terms = {term for term in expected_terms if term in text}
    term_recall = len(found_terms) / len(expected_terms) if expected_terms else 1.0

    quality_score = 0.75 * path_recall + 0.25 * term_recall
    severe_failure = path_recall < 0.5 or (
        bool(task.explicit_paths) and not set(task.explicit_paths).issubset(full_paths)
    )
    return {
        "task": task.task,
        "family": task.family,
        "explicit_paths": list(task.explicit_paths),
        "expected_paths": list(task.expected_paths),
        "found_paths": sorted(found_paths),
        "missing_paths": sorted(expected_paths - found_paths),
        "path_recall": path_recall,
        "expected_terms": list(task.expected_terms),
        "found_terms": sorted(found_terms),
        "missing_terms": sorted(expected_terms - found_terms),
        "term_recall": term_recall,
        "quality_score": quality_score,
        "severe_failure": severe_failure,
    }


def summarize_quality(label: str, rows: list[dict]) -> dict:
    hidden_rows = [row for row in rows if not row["explicit_paths"]]
    return {
        "label": label,
        "tasks": len(rows),
        "avg_quality_score": sum(row["quality_score"] for row in rows) / len(rows),
        "avg_path_recall": sum(row["path_recall"] for row in rows) / len(rows),
        "avg_term_recall": sum(row["term_recall"] for row in rows) / len(rows),
        "hidden_file_recall": (
            sum(row["path_recall"] for row in hidden_rows) / len(hidden_rows)
            if hidden_rows
            else None
        ),
        "severe_failures": sum(1 for row in rows if row["severe_failure"]),
        "severe_failure_rate": sum(1 for row in rows if row["severe_failure"]) / len(rows),
        "failures": [
            row
            for row in rows
            if row["severe_failure"] or row["path_recall"] < 1.0
        ],
    }


def summarize_by_bucket(rows: list[dict]) -> dict:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["family"], []).append(row)
    return {
        bucket: summarize_quality(bucket, bucket_rows)
        for bucket, bucket_rows in sorted(buckets.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--synthetic-file-count", type=int, default=60)
    parser.add_argument("--max-chars-per-file", type=int, default=16_000)
    args = parser.parse_args()

    tasks = quality_tasks(args.rounds)
    workload_tasks = to_workload_tasks(tasks)
    repo_blocks = build_repo_blocks(
        "large",
        args.max_chars_per_file,
        args.synthetic_file_count,
    )
    extra_blocks = read_repo_blocks(
        [
            "cache_proxy/README.md",
            "cache_proxy/benchmark_messy_agents.py",
        ],
        args.max_chars_per_file,
    )
    block_by_path = {block.path: block for block in [*repo_blocks, *extra_blocks]}
    repo_blocks = [block_by_path[path] for path in sorted(block_by_path)]
    all_paths = {block.path for block in repo_blocks}

    upstream = upstream_contextpilot_requests(workload_tasks, repo_blocks)
    fixed_general = [
        contextpilot_messages(
            task,
            repo_blocks,
            index,
            repo_id="quality-repo",
            commit_hash="quality123",
            profile="general",
        )
        for index, task in enumerate(workload_tasks)
    ]
    adaptive = [
        contextpilot_messages(
            task,
            repo_blocks,
            index,
            repo_id="quality-repo",
            commit_hash="quality123",
        )
        for index, task in enumerate(workload_tasks)
    ]
    budget_shadow = upstream_budget_shadow_pruned_contextpilot_requests(
        workload_tasks,
        repo_blocks,
        upstream,
    )
    router = adaptive_context_router_requests(
        workload_tasks,
        repo_blocks,
        upstream,
        fixed_general,
        adaptive,
        budget_shadow,
    )

    upstream_responses = quiet_local_responses(upstream)
    router_responses = quiet_local_responses(router)
    upstream_cost = summarize("upstream-contextpilot", upstream_responses, Prices())
    router_cost = summarize("adaptive-router-contextpilot", router_responses, Prices())

    upstream_rows = [
        score_request(request, task, all_paths)
        for request, task in zip(upstream, tasks)
    ]
    router_rows = [
        score_request(request, task, all_paths)
        for request, task in zip(router, tasks)
    ]
    upstream_quality = summarize_quality("upstream-contextpilot", upstream_rows)
    router_quality = summarize_quality("adaptive-router-contextpilot", router_rows)
    quality_loss = (
        (upstream_quality["avg_quality_score"] - router_quality["avg_quality_score"])
        / upstream_quality["avg_quality_score"]
        if upstream_quality["avg_quality_score"]
        else None
    )
    cost_reduction = (
        upstream_cost["estimated_cost_usd"] / router_cost["estimated_cost_usd"]
        if router_cost["estimated_cost_usd"]
        else None
    )
    acceptance = {
        "cost_reduction_gte_2x": cost_reduction >= 2.0 if cost_reduction else False,
        "quality_loss_lt_5pct": quality_loss < 0.05 if quality_loss is not None else False,
        "severe_failures_near_zero": router_quality["severe_failures"] == 0,
        "hidden_file_recall_gte_98pct": (
            router_quality["hidden_file_recall"] >= 0.98
            if router_quality["hidden_file_recall"] is not None
            else False
        ),
    }
    result = {
        "note": (
            "Local quality is a deterministic context-availability proxy, not "
            "a live LLM answer-quality score."
        ),
        "comparison": {
            "upstream_cost_usd": upstream_cost["estimated_cost_usd"],
            "adaptive_router_cost_usd": router_cost["estimated_cost_usd"],
            "cost_reduction_ratio": cost_reduction,
            "upstream_quality_score": upstream_quality["avg_quality_score"],
            "adaptive_router_quality_score": router_quality["avg_quality_score"],
            "quality_loss_pct": quality_loss,
            "adaptive_router_hidden_file_recall": router_quality["hidden_file_recall"],
            "adaptive_router_severe_failure_rate": router_quality["severe_failure_rate"],
        },
        "acceptance": {
            **acceptance,
            "passed": all(acceptance.values()),
        },
        "cost": {
            "upstream_contextpilot": upstream_cost,
            "adaptive_router_contextpilot": router_cost,
        },
        "quality": {
            "upstream_contextpilot": upstream_quality,
            "adaptive_router_contextpilot": router_quality,
        },
        "quality_by_bucket": {
            "upstream_contextpilot": summarize_by_bucket(upstream_rows),
            "adaptive_router_contextpilot": summarize_by_bucket(router_rows),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
