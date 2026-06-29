from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .ab_test_workspace_repo import Prices, cost_usd, no_cache_cost_usd, summarize
from .context_planner import (
    TOKEN_RE,
    extract_symbols,
    plan_context_blocks,
    score_block_relevance,
)
from .context_store import ContextBlock
from .prompt_compiler import compile_planned_messages


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_CONTEXTPILOT_ROOT = ROOT / "vendor" / "ContextPilot"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


DEFAULT_FILES = [
    "cache_proxy/server.py",
    "cache_proxy/context_store.py",
    "cache_proxy/prompt_compiler.py",
    "cache_proxy/context_planner.py",
    "tests/test_cache_proxy.py",
]


SYNTHETIC_AREAS = [
    ("api", "routes"),
    ("auth", "sessions"),
    ("billing", "invoices"),
    ("cache", "prefix"),
    ("cli", "commands"),
    ("config", "settings"),
    ("db", "migrations"),
    ("docs", "guides"),
    ("eval", "metrics"),
    ("ide", "workspace"),
    ("index", "symbols"),
    ("jobs", "scheduler"),
    ("llm", "provider"),
    ("logging", "events"),
    ("models", "schemas"),
    ("observability", "traces"),
    ("plugins", "loader"),
    ("security", "policies"),
    ("tests", "fixtures"),
    ("ui", "panels"),
]


TOOL_SCHEMA = """
Tool schema registry follows.
tool: read_file
description: Read UTF-8 text from a repository file.
args:
  path: repository-relative path
  start_line: optional one-based start line
  end_line: optional one-based end line
tool: rg_search
description: Search repository text with ripgrep-compatible syntax.
args:
  pattern: regex pattern
  include_globs: optional list of file globs
  max_results: maximum number of matches
tool: apply_patch
description: Apply a focused source-code patch.
args:
  patch: unified patch text
  safety: must be non-destructive
tool: run_tests
description: Run a test command inside the workspace.
args:
  command: test command
  timeout_seconds: execution timeout
""".strip()


PYTEST_LOG = """
pytest telemetry excerpt:
============================= test session starts ==============================
platform darwin -- Python 3.9.6, pytest-8.4.2, pluggy-1.6.0
rootdir: /workspace/repo
plugins: anyio-4.12.1
collected 4 items
tests/test_cache_proxy.py::test_repo_context_roundtrip PASSED
tests/test_cache_proxy.py::test_prompt_compiler_places_repo_context_before_user PASSED
tests/test_cache_proxy.py::test_context_planner_selects_relevant_file_body PASSED
tests/test_cache_proxy.py::test_planned_prompt_has_stable_map_before_file_bodies PASSED
============================== 4 passed in 0.23s ===============================
""".strip()


AGENT_POLICY = """
Agent policy:
- Prefer small, auditable changes.
- Keep volatile task instructions after stable repository context.
- Do not reorder stable context once a repo session has started.
- Summarize only low-reuse material.
- Preserve user edits and never revert unrelated work.
- Report cache-hit telemetry after each provider request.
""".strip()


RECENT_DIFF = """
Recent diff artifact follows.
diff --git a/cache_proxy/server.py b/cache_proxy/server.py
@@ -70,8 +70,9 @@ async def chat_completions(request: Request):
-    blocks = STORE.get_blocks(repo_id, commit_hash, selected_blocks)
+    planned = plan_context_blocks(original_messages, blocks)
+    compiled_messages = compile_planned_messages(...)
""".strip()


STACK_TRACE = """
Traceback (most recent call last):
  File "/workspace/repo/tests/test_cache_proxy.py", line 88, in test_context_planner_selects_relevant_file_body
    assert planned.full_blocks
AssertionError: planner selected no relevant files
""".strip()


PACKAGE_MANIFEST = """
Package manifest excerpt follows.
pyproject.toml
[project]
name = "contextpilot-cache-proxy"
requires-python = ">=3.9"
dependencies = [
  "fastapi",
  "httpx",
  "uvicorn",
  "pytest",
]
[tool.pytest.ini_options]
testpaths = ["tests"]
""".strip()


REPO_MAP = """
Repository map follows.
cache_proxy/server.py: FastAPI proxy, request compilation, provider forwarding, telemetry writeback.
cache_proxy/context_store.py: SQLite persistence for repo blocks and telemetry.
cache_proxy/context_planner.py: repo map generation, relevance scoring, selected full-file budgeting.
cache_proxy/prompt_compiler.py: deterministic prompt assembly with stable context before volatile task messages.
tests/test_cache_proxy.py: API roundtrip tests and prompt planning tests.
""".strip()


HANDOFF_SUMMARY = """
Prior agent handoff summary follows.
- Cache feedback from public APIs is coarse and should not drive lower-layer co-design.
- Provider-independent optimization should focus on stable prompt layout and better repo-aware selection.
- Current benchmark compares native cache behavior against base ContextPilot.
- Next useful work is section-level extraction inside selected files.
""".strip()


@dataclass(frozen=True)
class PreparedRequest:
    task: str
    messages: list[dict]
    metadata: dict


@dataclass(frozen=True)
class WorkloadTask:
    task: str
    family: str
    required_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoutePolicy:
    bucket: str
    confidence: str
    allowed_strategies: tuple[str, ...]
    min_prompt_tokens: int
    reason: str


MIXED_WORKLOADS = [
    WorkloadTask(
        "Code review: inspect cache proxy request handling and identify one behavioral risk.",
        "code_review",
    ),
    WorkloadTask(
        "Test debugging: diagnose why planner file selection is flaky in CI.",
        "test_debug",
    ),
    WorkloadTask(
        "Refactor planning: find the smallest provider-adapter extraction.",
        "refactor",
    ),
    WorkloadTask(
        "Docs and onboarding: explain how an IDE should send repository context.",
        "docs",
    ),
    WorkloadTask(
        "Multi-agent handoff: continue a prior agent investigation without repeating work.",
        "handoff",
    ),
    WorkloadTask(
        "Code review: audit telemetry accounting for misleading cache metrics.",
        "code_review",
    ),
    WorkloadTask(
        "Test debugging: design a regression test for context-planning budgets.",
        "test_debug",
    ),
    WorkloadTask(
        "Refactor planning: simplify prompt compilation while preserving cache stability.",
        "refactor",
    ),
    WorkloadTask(
        "Docs and onboarding: write a concise architecture note for ContextPilot.",
        "docs",
    ),
    WorkloadTask(
        "Multi-agent handoff: summarize repo state for the next coding worker.",
        "handoff",
    ),
]


SYNTHETIC_TARGETED_WORKLOADS = [
    WorkloadTask(
        "Code review: inspect synthetic_repo/auth/sessions_21.py and identify one session-cache risk.",
        "synthetic_code_review",
        ("synthetic_repo/auth/sessions_21.py",),
    ),
    WorkloadTask(
        "Test debugging: use synthetic_repo/tests/fixtures_38.py to diagnose fixture ownership drift.",
        "synthetic_test_debug",
        ("synthetic_repo/tests/fixtures_38.py",),
    ),
    WorkloadTask(
        "Refactor planning: inspect synthetic_repo/llm/provider_32.py and propose a provider adapter split.",
        "synthetic_refactor",
        ("synthetic_repo/llm/provider_32.py",),
    ),
    WorkloadTask(
        "Docs and onboarding: cite synthetic_repo/docs/guides_47.md while explaining the docs runbook.",
        "synthetic_docs",
        ("synthetic_repo/docs/guides_47.md",),
    ),
    WorkloadTask(
        "Security review: inspect synthetic_repo/security/policies_57.py and identify one policy-cache hazard.",
        "synthetic_security",
        ("synthetic_repo/security/policies_57.py",),
    ),
    WorkloadTask(
        "Observability debugging: use synthetic_repo/observability/traces_35.py to explain trace metric naming.",
        "synthetic_observability",
        ("synthetic_repo/observability/traces_35.py",),
    ),
    WorkloadTask(
        "Billing audit: inspect synthetic_repo/billing/invoices_42.py and summarize invoice workflow risk.",
        "synthetic_billing",
        ("synthetic_repo/billing/invoices_42.py",),
    ),
    WorkloadTask(
        "UI agent task: use synthetic_repo/ui/panels_59.py and describe the panel service telemetry.",
        "synthetic_ui",
        ("synthetic_repo/ui/panels_59.py",),
    ),
    WorkloadTask(
        "Indexing task: inspect synthetic_repo/index/symbols_50.py and identify the symbol cache namespace.",
        "synthetic_index",
        ("synthetic_repo/index/symbols_50.py",),
    ),
    WorkloadTask(
        "Plugin loader task: cite synthetic_repo/plugins/loader_56.py and explain the loader feature flag.",
        "synthetic_plugins",
        ("synthetic_repo/plugins/loader_56.py",),
    ),
]


def read_repo_context(paths: list[str], max_chars_per_file: int) -> str:
    return repo_context_from_blocks(read_repo_blocks(paths, max_chars_per_file))


def repo_context_from_blocks(blocks: list[ContextBlock]) -> str:
    parts = ["Repository context follows. Treat it as read-only reference.", f"repo_root: {ROOT}", ""]
    for block in sorted(blocks, key=lambda item: item.path):
        parts.extend([f"--- file: {block.path}", "```", block.content.rstrip(), "```", ""])
    return "\n".join(parts).strip()


def read_repo_blocks(paths: list[str], max_chars_per_file: int) -> list[ContextBlock]:
    blocks: list[ContextBlock] = []
    for rel in sorted(paths):
        path = ROOT / rel
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")[:max_chars_per_file]
        blocks.append(
            ContextBlock(
                repo_id="messy-repo",
                commit_hash="abc123",
                path=rel,
                content=content,
                content_hash=f"benchmark-{rel}",
            )
        )
    return blocks


def synthetic_repo_block(index: int, max_chars_per_file: int) -> ContextBlock:
    area, topic = SYNTHETIC_AREAS[index % len(SYNTHETIC_AREAS)]
    ext = "py" if area != "docs" else "md"
    path = f"synthetic_repo/{area}/{topic}_{index:02d}.{ext}"
    repeated_cases = "\n".join(
        [
            f"case_{case}: handles {area} {topic} workflow step {case}; "
            f"cache_key='{area}:{topic}:{case % 5}'; owner='team-{case % 7}'"
            for case in range(1, 42)
        ]
    )
    if ext == "md":
        content = f"""
# {area.title()} {topic.title()} Runbook

This file is stable project documentation for the {area} subsystem.
It describes operational expectations, common failures, telemetry names,
and the repository paths an agent should inspect before making changes.

## Key Details
{repeated_cases}

## Agent Notes
- Preserve stable context layout across IDE sessions.
- Prefer exact file citations when answering.
- Treat user-specific request ids as volatile suffix material.
""".strip()
    else:
        content = f'''
"""Stable synthetic module for {area}/{topic}.

This benchmark file gives the repo enough breadth that all-doc context
becomes expensive while still containing realistic symbols and comments.
"""

DEFAULT_{area.upper()}_{topic.upper()}_CACHE_TTL_SECONDS = {300 + index}
{area.upper()}_{topic.upper()}_FEATURE_FLAG = "enable_{area}_{topic}_pipeline"


class {area.title().replace("_", "")}{topic.title().replace("_", "")}Service:
    def __init__(self, repo_id: str, commit_hash: str) -> None:
        self.repo_id = repo_id
        self.commit_hash = commit_hash
        self.cache_namespace = "{area}:{topic}"

    def plan(self, task: str, user_id: str) -> dict:
        return {{
            "task": task,
            "user_id": user_id,
            "namespace": self.cache_namespace,
            "stable_prefix": True,
            "volatile_suffix": task.startswith("session_"),
        }}

    def telemetry(self) -> list[str]:
        return [
{chr(10).join(f'            "{area}.{topic}.metric_{metric}",' for metric in range(24))}
        ]


{repeated_cases}
'''.strip()
    return ContextBlock(
        repo_id="messy-repo",
        commit_hash="abc123",
        path=path,
        content=content[:max_chars_per_file],
        content_hash=f"synthetic-{index:02d}-{area}-{topic}",
    )


def build_repo_blocks(
    corpus: str,
    max_chars_per_file: int,
    synthetic_file_count: int,
) -> list[ContextBlock]:
    blocks = read_repo_blocks(DEFAULT_FILES, max_chars_per_file)
    if corpus == "large":
        blocks.extend(
            synthetic_repo_block(index, max_chars_per_file)
            for index in range(synthetic_file_count)
        )
    return blocks


def build_tasks(rounds: int, workload: str) -> list[WorkloadTask]:
    if workload == "synthetic-targeted":
        source = SYNTHETIC_TARGETED_WORKLOADS
    elif workload == "mixed-plus-targeted":
        source = [*MIXED_WORKLOADS, *SYNTHETIC_TARGETED_WORKLOADS]
    else:
        source = MIXED_WORKLOADS
    return [source[index % len(source)] for index in range(rounds)]


def split_context_for_partial_native_cache(repo_context: str, stable_ratio: float = 0.45) -> tuple[str, str]:
    split_at = int(len(repo_context) * stable_ratio)
    newline = repo_context.find("\n--- file:", split_at)
    if newline != -1:
        split_at = newline
    return repo_context[:split_at].strip(), repo_context[split_at:].strip()


def mixed_suffix(task: WorkloadTask, index: int) -> str:
    unique_trace = "\n".join(
        [
            f"session_id: mixed-agent-session-{index:03d}",
            f"request_id: req-{index:03d}-{task.family}",
            f"timestamp: 2026-05-30T12:{index:02d}:00Z",
            f"current_focus_file: {DEFAULT_FILES[index % len(DEFAULT_FILES)]}",
        ]
    )
    required_path_text = (
        "\n".join(f"- {path}" for path in task.required_paths)
        if task.required_paths
        else "None"
    )
    family_payloads = {
        "code_review": [AGENT_POLICY, TOOL_SCHEMA, RECENT_DIFF, REPO_MAP],
        "test_debug": [AGENT_POLICY, PYTEST_LOG, STACK_TRACE, TOOL_SCHEMA],
        "refactor": [AGENT_POLICY, PACKAGE_MANIFEST, REPO_MAP, RECENT_DIFF],
        "docs": [AGENT_POLICY, REPO_MAP, TOOL_SCHEMA],
        "handoff": [AGENT_POLICY, HANDOFF_SUMMARY, TOOL_SCHEMA, PYTEST_LOG],
    }
    payload = family_payloads.get(task.family, [AGENT_POLICY, TOOL_SCHEMA])
    if index % 2:
        payload = [payload[0], *reversed(payload[1:])]
    return "\n\n".join(
        [
            f"Task: {task.task}",
            f"Workload family: {task.family}",
            f"Required source files:\n{required_path_text}",
            unique_trace,
            "Volatile agent artifacts follow.",
            *payload,
            "Answer with one concise paragraph and cite the most relevant file path.",
        ]
    )


def deepseek_native_messages(task: WorkloadTask, repo_context: str, index: int) -> list[dict]:
    wrappers = [
        "Cursor IDE agent runtime v0.42. User-specific coding preferences are active.",
        "OpenClauw workspace agent. Session tools and repo snapshot were selected by the IDE.",
        "Batch code-review worker. Local context was assembled from recent editor state.",
        "Autonomous test-fix agent. The following context came from a prior tool loop.",
    ]
    stable_repo_prefix, late_repo_context = split_context_for_partial_native_cache(repo_context)
    return [
        {"role": "system", "content": "You are a precise coding agent. Shared IDE agent runtime."},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "Shared IDE-selected repository prefix follows.",
                    stable_repo_prefix,
                    wrappers[index % len(wrappers)],
                    mixed_suffix(task, index),
                    "Late IDE-selected repository context follows. The IDE may change this order between agents.",
                    late_repo_context,
                ]
            ),
        },
    ]


def contextpilot_messages(
    task: WorkloadTask,
    repo_blocks: list[ContextBlock],
    index: int,
    repo_id: str,
    commit_hash: str,
    profile: str | None = None,
) -> PreparedRequest:
    original = [
        {"role": "system", "content": "You are a precise coding agent."},
        {"role": "user", "content": mixed_suffix(task, index)},
    ]
    planned = plan_context_blocks(
        original,
        repo_blocks,
        profile=profile,
    )
    return PreparedRequest(
        task=task.task,
        messages=compile_planned_messages(
            original,
            planned.map_blocks,
            planned.full_blocks,
            repo_id=repo_id,
            commit_hash=commit_hash,
        ),
        metadata={
            "workload_family": task.family,
            "required_paths": list(task.required_paths),
            "context_planning": planned.telemetry,
        },
    )


def upstream_contextpilot_requests(
    tasks: list[WorkloadTask],
    repo_blocks: list[ContextBlock],
) -> list[PreparedRequest]:
    if not UPSTREAM_CONTEXTPILOT_ROOT.exists():
        raise RuntimeError(
            "Upstream ContextPilot source is missing. Expected "
            f"{UPSTREAM_CONTEXTPILOT_ROOT}"
        )
    sys.path.insert(0, str(UPSTREAM_CONTEXTPILOT_ROOT))
    import contextpilot as upstream_cp

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        engine = upstream_cp.ContextPilot(use_gpu=False)
    docs = [format_context_doc(block) for block in repo_blocks]
    requests: list[PreparedRequest] = []
    for index, task in enumerate(tasks):
        query = mixed_suffix(task, index)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            messages = engine.optimize(
                docs,
                query,
                conversation_id="messy-repo",
                system_instruction="You are a precise coding agent.",
            )
        requests.append(
            PreparedRequest(
                task=task.task,
                messages=messages,
                metadata={
                    "workload_family": task.family,
                    "required_paths": list(task.required_paths),
                    "source": "vendor/ContextPilot",
                    "api": "ContextPilot.optimize",
                    "repo_docs": len(docs),
                },
            )
        )
    return requests


def upstream_budget_shadow_pruned_contextpilot_requests(
    tasks: list[WorkloadTask],
    repo_blocks: list[ContextBlock],
    upstream_requests: list[PreparedRequest],
) -> list[PreparedRequest]:
    if not UPSTREAM_CONTEXTPILOT_ROOT.exists():
        raise RuntimeError(
            "Upstream ContextPilot source is missing. Expected "
            f"{UPSTREAM_CONTEXTPILOT_ROOT}"
        )
    sys.path.insert(0, str(UPSTREAM_CONTEXTPILOT_ROOT))
    import contextpilot as upstream_cp

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        engine = upstream_cp.ContextPilot(use_gpu=False)
        reordered_docs, _ = engine.reorder(
            [[block.path for block in repo_blocks]],
            conversation_id="messy-repo-budget-shadow",
        )
    upstream_order = list(reordered_docs[0])
    block_by_path = {block.path: block for block in repo_blocks}
    ordered_blocks = [block_by_path[path] for path in upstream_order if path in block_by_path]
    selected: list[PreparedRequest] = []
    cached_prompts: list[str] = []
    for index, (task, upstream) in enumerate(zip(tasks, upstream_requests)):
        query = mixed_suffix(task, index)
        force_paths = task_force_paths(task, ordered_blocks)
        candidate_policies = budget_shadow_candidate_policies(bool(force_paths))
        choices: list[PreparedRequest] = [
            PreparedRequest(
                task=upstream.task,
                messages=upstream.messages,
                metadata={**upstream.metadata, "budget_shadow_candidate": "upstream-full"},
            )
        ]
        seen_candidate_shapes = {serialized_prompt(upstream.messages)}
        for policy in candidate_policies:
            kept_blocks, prune_stats = prune_upstream_context(
                query,
                ordered_blocks,
                min_docs=policy["min_docs"],
                min_score=policy["min_score"],
                max_docs=policy["max_docs"],
                force_paths=force_paths,
            )
            kept_blocks = slice_context_blocks(
                query,
                kept_blocks,
                max_chars_per_doc=policy["max_chars_per_doc"],
            )
            messages = build_upstream_style_messages(
                kept_blocks,
                query,
                system_instruction="You are a precise coding agent.",
            )
            shape = serialized_prompt(messages)
            if shape in seen_candidate_shapes:
                continue
            seen_candidate_shapes.add(shape)
            choices.append(
                PreparedRequest(
                    task=task.task,
                    messages=messages,
                    metadata={
                        "workload_family": task.family,
                        "required_paths": list(task.required_paths),
                        "source": "vendor/ContextPilot ordering + budget shadow pruning",
                        "budget_shadow_policy": policy,
                        **prune_stats,
                    },
                )
            )
        valid_choices = [
            choice
            for choice in choices
            if (request_required_recall(choice) is None)
            or not request_required_recall(choice)["missing_required_paths"]
        ]
        if not valid_choices:
            valid_choices = [choices[0]]
        scored_choices = []
        for choice in valid_choices:
            usage = local_usage(choice.messages, cached_prompts)
            scored_choices.append(
                (
                    cost_usd(usage, Prices()),
                    usage["prompt_tokens"],
                    -usage["prompt_cache_hit_tokens"],
                    usage,
                    choice,
                )
            )
        best_cost, _, _, best_usage, best = min(scored_choices, key=lambda item: item[:3])
        cached_prompts.append(serialized_prompt(best.messages))
        selected.append(
            PreparedRequest(
                task=best.task,
                messages=best.messages,
                metadata={
                    **best.metadata,
                    "budget_shadow_selection": {
                        "candidate_count": len(choices),
                        "valid_candidate_count": len(valid_choices),
                        "chosen_estimated_cost_usd": best_cost,
                        "chosen_usage": best_usage,
                    },
                },
            )
        )
    return selected


def budget_shadow_candidate_policies(has_force_paths: bool) -> list[dict]:
    min_doc_options = (0, 1, 2, 3, 4) if has_force_paths else (1, 2, 3, 4)
    policies = []
    for min_docs in min_doc_options:
        for min_score in (10, 12, 15, 18, 22, 28, 36, 48):
            for max_docs in (1, 2, 3, 5, 8, 13):
                if max_docs < min_docs:
                    continue
                for max_chars_per_doc in (1200, 2400, 4800, 8000, None):
                    policies.append(
                        {
                            "min_docs": min_docs,
                            "min_score": min_score,
                            "max_docs": max_docs,
                            "max_chars_per_doc": max_chars_per_doc,
                        }
                    )
    return policies


def adaptive_budget_shadow_contextpilot_requests(
    budget_shadow_requests: list[PreparedRequest],
    adaptive_requests: list[PreparedRequest],
) -> list[PreparedRequest]:
    if len(budget_shadow_requests) != len(adaptive_requests):
        raise ValueError("budget-shadow and adaptive request counts must match")
    selected: list[PreparedRequest] = []
    cached_prompts: list[str] = []
    for budget_shadow, adaptive in zip(budget_shadow_requests, adaptive_requests):
        candidates = [
            ("budget-shadow", budget_shadow),
            ("adaptive", adaptive),
        ]
        valid_candidates: list[tuple[str, PreparedRequest]] = []
        for label, candidate in candidates:
            recall = request_required_recall(candidate)
            if recall is None or not recall["missing_required_paths"]:
                valid_candidates.append((label, candidate))
        if not valid_candidates:
            valid_candidates = [("budget-shadow", budget_shadow)]

        scored_choices = [
            (
                cost_usd(local_usage(candidate.messages, cached_prompts), Prices()),
                local_usage(candidate.messages, cached_prompts),
                label,
                candidate,
            )
            for label, candidate in valid_candidates
        ]
        best_cost, best_usage, best_label, best = min(
            scored_choices,
            key=lambda item: item[0],
        )
        cached_prompts.append(serialized_prompt(best.messages))
        selected.append(
            PreparedRequest(
                task=best.task,
                messages=best.messages,
                metadata={
                    **best.metadata,
                    "adaptive_budget_shadow_selection": {
                        "chosen": best_label,
                        "candidate_count": len(valid_candidates),
                        "chosen_estimated_cost_usd": best_cost,
                        "chosen_usage": best_usage,
                    },
                },
            )
        )
    return selected


def adaptive_context_router_requests(
    tasks: list[WorkloadTask],
    repo_blocks: list[ContextBlock],
    upstream_requests: list[PreparedRequest],
    fixed_general_requests: list[PreparedRequest],
    adaptive_requests: list[PreparedRequest],
    budget_shadow_requests: list[PreparedRequest],
) -> list[PreparedRequest]:
    selected: list[PreparedRequest] = []
    cached_prompts: list[str] = []
    for index, task in enumerate(tasks):
        policy = route_policy_for_task(task)
        query = mixed_suffix(task, index)
        inferred_paths = infer_context_guard_paths(task.task, repo_blocks)
        inferred_request = focused_inferred_context_request(
            task,
            query,
            repo_blocks,
            inferred_paths,
        )
        candidates_by_label = {
            "upstream": upstream_requests[index],
            "focused-inferred": inferred_request,
            "fixed-general": fixed_general_requests[index],
            "adaptive": adaptive_requests[index],
            "budget-shadow": budget_shadow_requests[index],
        }
        valid_candidates = []
        for label in policy.allowed_strategies:
            candidate = candidates_by_label[label]
            recall = request_required_recall(candidate)
            if recall is not None and recall["missing_required_paths"]:
                continue
            included_paths = included_full_context_paths(candidate)
            if (
                inferred_paths
                and included_paths is not None
                and not set(inferred_paths).issubset(included_paths)
            ):
                continue
            usage = local_usage(candidate.messages, cached_prompts)
            if (
                label == "budget-shadow"
                and usage["prompt_tokens"] < policy.min_prompt_tokens
            ):
                continue
            valid_candidates.append((label, candidate, usage))
        if not valid_candidates:
            fallback = candidates_by_label["upstream"]
            valid_candidates = [
                ("upstream", fallback, local_usage(fallback.messages, cached_prompts))
            ]

        scored = [
            (
                cost_usd(usage, Prices()),
                usage["prompt_tokens"],
                -usage["prompt_cache_hit_tokens"],
                label,
                candidate,
                usage,
            )
            for label, candidate, usage in valid_candidates
        ]
        best_cost, _, _, label, candidate, usage = min(
            scored,
            key=lambda item: item[:3],
        )
        cached_prompts.append(serialized_prompt(candidate.messages))
        selected.append(
            PreparedRequest(
                task=candidate.task,
                messages=candidate.messages,
                metadata={
                    **candidate.metadata,
                    "adaptive_router": {
                        "bucket": policy.bucket,
                        "confidence": policy.confidence,
                        "reason": policy.reason,
                        "allowed_strategies": list(policy.allowed_strategies),
                        "inferred_guard_paths": list(inferred_paths),
                        "min_prompt_tokens": policy.min_prompt_tokens,
                        "chosen_strategy": label,
                        "chosen_estimated_cost_usd": best_cost,
                        "chosen_usage": usage,
                    },
                },
            )
        )
    return selected


def route_policy_for_task(task: WorkloadTask) -> RoutePolicy:
    family = task.family
    if task.required_paths:
        return RoutePolicy(
            bucket="explicit-path",
            confidence="high",
            allowed_strategies=(
                "budget-shadow",
                "focused-inferred",
                "adaptive",
                "fixed-general",
                "upstream",
            ),
            min_prompt_tokens=350,
            reason="task names required source files",
        )
    if "test" in family or "debug" in family:
        return RoutePolicy(
            bucket="test-debug",
            confidence="medium",
            allowed_strategies=("focused-inferred", "adaptive", "fixed-general", "upstream"),
            min_prompt_tokens=4_000,
            reason="debug tasks need tests plus implementation context",
        )
    if "refactor" in family:
        return RoutePolicy(
            bucket="refactor",
            confidence="medium",
            allowed_strategies=("focused-inferred", "fixed-general", "adaptive", "upstream"),
            min_prompt_tokens=6_000,
            reason="refactors often need multi-file context",
        )
    if "docs" in family:
        return RoutePolicy(
            bucket="docs",
            confidence="medium",
            allowed_strategies=("focused-inferred", "adaptive", "fixed-general", "upstream"),
            min_prompt_tokens=2_000,
            reason="docs tasks can be compact but should keep repo map context",
        )
    if "handoff" in family:
        return RoutePolicy(
            bucket="handoff",
            confidence="low",
            allowed_strategies=("focused-inferred", "upstream", "fixed-general", "adaptive"),
            min_prompt_tokens=8_000,
            reason="handoffs benefit from stable broad context",
        )
    if "review" in family or "security" in family or "audit" in family:
        return RoutePolicy(
            bucket="code-review",
            confidence="medium",
            allowed_strategies=("focused-inferred", "adaptive", "fixed-general", "upstream"),
            min_prompt_tokens=4_000,
            reason="review tasks need enough surrounding implementation context",
        )
    return RoutePolicy(
        bucket="unknown",
        confidence="low",
        allowed_strategies=("focused-inferred", "upstream", "adaptive", "fixed-general"),
        min_prompt_tokens=8_000,
        reason="unknown tasks should preserve context quality over cost",
    )


def focused_inferred_context_request(
    task: WorkloadTask,
    query: str,
    repo_blocks: list[ContextBlock],
    inferred_paths: tuple[str, ...],
) -> PreparedRequest:
    block_by_path = {block.path: block for block in repo_blocks}
    blocks = [block_by_path[path] for path in inferred_paths if path in block_by_path]
    messages = build_upstream_style_messages(
        blocks,
        query,
        system_instruction="You are a precise coding agent.",
    )
    return PreparedRequest(
        task=task.task,
        messages=messages,
        metadata={
            "workload_family": task.family,
            "required_paths": list(task.required_paths),
            "source": "adaptive router inferred context",
            "inferred_context": {
                "kept_paths": [block.path for block in blocks],
            },
        },
    )


def infer_context_guard_paths(
    task_text: str,
    repo_blocks: list[ContextBlock],
    max_paths: int = 3,
) -> tuple[str, ...]:
    lowered_task = task_text.lower()
    hinted_paths: list[str] = []
    if "chat_completions" in lowered_task or "request forwarding" in lowered_task:
        hinted_paths.append("cache_proxy/server.py")
    if "compile_planned_messages" in lowered_task:
        hinted_paths.append("cache_proxy/prompt_compiler.py")
    if "plan_context_blocks" in lowered_task:
        hinted_paths.append("cache_proxy/context_planner.py")
    if "pytest" in lowered_task or "test" in lowered_task:
        hinted_paths.append("tests/test_cache_proxy.py")
    if "contextstore" in lowered_task or "telemetry persistence" in lowered_task:
        hinted_paths.append("cache_proxy/context_store.py")
    if "onboarding" in lowered_task or "docs readers" in lowered_task:
        hinted_paths.append("cache_proxy/README.md")
    if "handoff" in lowered_task or "adaptive-router" in lowered_task:
        hinted_paths.append("cache_proxy/benchmark_messy_agents.py")
        hinted_paths.append("cache_proxy/README.md")

    available_paths = {block.path for block in repo_blocks}
    hints = [path for path in hinted_paths if path in available_paths]
    task_tokens = {
        token.lower()
        for token in TOKEN_RE.findall(task_text)
        if len(token) > 3
    }
    scored = []
    for order, block in enumerate(repo_blocks):
        path_score = score_block_relevance(task_text, block)
        symbol_score = 0
        for symbol in extract_symbols(block.content)[:80]:
            if symbol.lower() in task_text.lower():
                symbol_score += 45
        content_tokens = {
            token.lower()
            for token in TOKEN_RE.findall(block.content[:12_000])
            if len(token) > 3
        }
        overlap = task_tokens & content_tokens
        content_score = min(10, len(overlap))
        score = path_score + symbol_score + content_score
        if (
            block.path == "cache_proxy/benchmark_messy_agents.py"
            and "benchmark" not in task_tokens
            and "adaptive-router" not in task_text.lower()
            and "handoff" not in task_tokens
        ):
            score -= 35
        if score >= 12:
            scored.append((score, order, block.path))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = []
    for path in [*hints, *(path for _, _, path in scored)]:
        if path in selected:
            continue
        selected.append(path)
        if len(selected) >= max_paths:
            break
    return tuple(selected)


def prune_upstream_context(
    task_text: str,
    ordered_blocks: list[ContextBlock],
    min_docs: int,
    min_score: int,
    max_docs: int | None = None,
    force_paths: set[str] | None = None,
) -> tuple[list[ContextBlock], dict]:
    force_paths = force_paths or set()
    scored = [
        (score_block_relevance(task_text, block), order, block)
        for order, block in enumerate(ordered_blocks)
    ]
    keep_paths = {block.path for _, _, block in scored[:min_docs]}
    keep_paths.update(force_paths)
    keep_paths.update(block.path for score, _, block in scored if score >= min_score)
    if max_docs is not None and len(keep_paths) > max_docs:
        locked_paths = {block.path for _, _, block in scored[:min_docs]}
        locked_paths.update(force_paths)
        optional = [
            (score, order, block)
            for score, order, block in scored
            if block.path in keep_paths and block.path not in locked_paths
        ]
        optional.sort(key=lambda item: (-item[0], item[1]))
        keep_paths = set(locked_paths)
        keep_paths.update(
            block.path
            for _, _, block in optional[: max(0, max_docs - len(locked_paths))]
        )
    kept = [block for _, _, block in scored if block.path in keep_paths]
    return kept, {
        "budget_pruning": {
            "enabled": True,
            "upstream_order_paths": [block.path for block in ordered_blocks],
            "kept_paths": [block.path for block in kept],
            "pruned_paths": [block.path for _, _, block in scored if block.path not in keep_paths],
            "min_docs": min_docs,
            "min_score": min_score,
            "max_docs": max_docs,
            "force_paths": sorted(force_paths),
            "scores": {block.path: score for score, _, block in scored},
        }
    }


def slice_context_blocks(
    task_text: str,
    blocks: list[ContextBlock],
    max_chars_per_doc: int | None,
) -> list[ContextBlock]:
    if max_chars_per_doc is None:
        return blocks
    return [
        ContextBlock(
            repo_id=block.repo_id,
            commit_hash=block.commit_hash,
            path=block.path,
            content=extractive_slice(block.content, task_text, max_chars_per_doc),
            content_hash=f"{block.content_hash}:slice:{max_chars_per_doc}",
        )
        for block in blocks
    ]


def extractive_slice(content: str, task_text: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    if max_chars < 400:
        return content[:max_chars].rstrip()

    lines = content.splitlines()
    task_tokens = {
        token.lower()
        for token in TOKEN_RE.findall(task_text)
        if len(token) > 2
    }
    scored_lines = []
    for index, line in enumerate(lines):
        line_tokens = {token.lower() for token in TOKEN_RE.findall(line)}
        score = len(task_tokens & line_tokens)
        if line.lstrip().startswith(("def ", "class ", "async def ")):
            score += 1
        if score:
            scored_lines.append((score, index))

    selected_indexes = set(range(min(12, len(lines))))
    for _, index in sorted(scored_lines, key=lambda item: (-item[0], item[1]))[:8]:
        selected_indexes.update(range(max(0, index - 2), min(len(lines), index + 5)))

    excerpt_lines = []
    used_chars = 0
    previous_index = -2
    for index in sorted(selected_indexes):
        line = lines[index]
        if previous_index != index - 1 and excerpt_lines:
            marker = "..."
            if used_chars + len(marker) + 1 > max_chars:
                break
            excerpt_lines.append(marker)
            used_chars += len(marker) + 1
        if used_chars + len(line) + 1 > max_chars:
            break
        excerpt_lines.append(line)
        used_chars += len(line) + 1
        previous_index = index

    if not excerpt_lines:
        return content[:max_chars].rstrip()
    return "\n".join(excerpt_lines).rstrip()


def task_force_paths(
    task: WorkloadTask,
    ordered_blocks: list[ContextBlock],
) -> set[str]:
    return set(task.required_paths) | extract_explicit_repo_paths(
        task.task,
        ordered_blocks,
    )


def extract_explicit_repo_paths(
    task_text: str,
    ordered_blocks: list[ContextBlock],
) -> set[str]:
    return {block.path for block in ordered_blocks if block.path in task_text}


def build_upstream_style_messages(
    blocks: list[ContextBlock],
    query: str,
    system_instruction: str | None = None,
) -> list[dict]:
    docs = [format_context_doc(block) for block in blocks]
    docs_section = "\n".join(f"[{index + 1}] {doc}" for index, doc in enumerate(docs))
    importance_ranking = " > ".join(str(index + 1) for index in range(len(docs)))
    parts = []
    if system_instruction:
        parts.append(system_instruction)
    parts.append(
        "Answer the question based on the provided documents.\n\n"
        f"<documents>\n{docs_section}\n</documents>\n\n"
        f"Read the documents in this importance ranking: {importance_ranking}\n"
        "Prioritize information from higher-ranked documents."
    )
    return [
        {"role": "system", "content": "\n\n".join(parts)},
        {"role": "user", "content": query},
    ]


def format_context_doc(block: ContextBlock) -> str:
    return "\n".join(
        [
            f"--- file: {block.path}",
            f"sha256: {block.content_hash}",
            "```",
            block.content.rstrip(),
            "```",
        ]
    )


def estimate_tokens(messages: list[dict]) -> int:
    chars = sum(len(str(message.get("content", ""))) for message in messages)
    return max(1, chars // 4)


def serialized_prompt(messages: list[dict]) -> str:
    return json.dumps(messages, sort_keys=True, separators=(",", ":"))


def longest_common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def local_usage(messages: list[dict], cached_prompts: list[str] | None = None) -> dict:
    prompt_tokens = estimate_tokens(messages)
    prompt = serialized_prompt(messages)
    hit_chars = 0
    for cached in cached_prompts or []:
        hit_chars = max(hit_chars, longest_common_prefix_len(prompt, cached))
    hit_tokens = min(prompt_tokens, hit_chars // 4)
    miss_tokens = max(0, prompt_tokens - hit_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_cache_hit_tokens": hit_tokens,
        "prompt_cache_miss_tokens": miss_tokens,
        "completion_tokens": 96,
        "total_tokens": prompt_tokens + 96,
    }


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


def run_arm(
    label: str,
    requests: list[PreparedRequest],
    model: str,
    user_id: str,
    live: bool,
    api_key: str | None,
) -> list[dict]:
    responses: list[dict] = []
    cached_prompts: list[str] = []
    for index, prepared in enumerate(requests, start=1):
        if live:
            if not api_key:
                raise SystemExit("Set DEEPSEEK_API_KEY first or omit --live.")
            response = call_deepseek(
                api_key,
                {
                    "model": model,
                    "messages": prepared.messages,
                    "thinking": {"type": "disabled"},
                    "temperature": 0,
                    "max_tokens": 96,
                    "user_id": user_id,
                },
            )
        else:
            response = {
                "_latency_ms": 0.0,
                "usage": local_usage(prepared.messages, cached_prompts),
                "model": model,
            }
            cached_prompts.append(serialized_prompt(prepared.messages))
        usage = response.get("usage", {})
        print(
            json.dumps(
                {
                    "arm": label,
                    "request": index,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
                    "cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
                    "estimated_cost_usd": cost_usd(usage, Prices()),
                    "metadata": prepared.metadata,
                },
                indent=2,
            )
        )
        responses.append(response)
    return responses


def summarize_local_shape(label: str, requests: list[PreparedRequest]) -> dict:
    prompt_tokens = sum(estimate_tokens(request.messages) for request in requests)
    no_cache_cost = sum(no_cache_cost_usd(local_usage(request.messages), Prices()) for request in requests)
    return {
        "label": label,
        "requests": len(requests),
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_no_cache_cost_usd": no_cache_cost,
    }


def included_full_context_paths(request: PreparedRequest) -> set[str] | None:
    metadata = request.metadata
    if "budget_pruning" in metadata:
        return set(metadata["budget_pruning"].get("kept_paths", []))
    if "inferred_context" in metadata:
        return set(metadata["inferred_context"].get("kept_paths", []))
    if "context_planning" in metadata:
        return set(metadata["context_planning"].get("full_block_paths", []))
    if metadata.get("source") == "vendor/ContextPilot":
        return None
    if metadata.get("source") == "deepseek-native-full-repo":
        return None
    return set()


def request_required_recall(request: PreparedRequest) -> dict | None:
    required = set(request.metadata.get("required_paths", []))
    if not required:
        return None
    included = included_full_context_paths(request)
    if included is None:
        found = set(required)
        mode = "all-context"
    else:
        found = required & included
        mode = "selected-context"
    return {
        "task": request.task,
        "required_paths": sorted(required),
        "included_required_paths": sorted(found),
        "missing_required_paths": sorted(required - found),
        "required_full_context_recall": len(found) / len(required),
        "context_mode": mode,
    }


def summarize_required_recall(label: str, requests: list[PreparedRequest]) -> dict:
    rows = [
        row
        for row in (request_required_recall(request) for request in requests)
        if row is not None
    ]
    if not rows:
        return {
            "label": label,
            "required_tasks": 0,
            "required_full_context_recall": None,
            "tasks_with_all_required_context": None,
            "missing_required_context_tasks": [],
        }
    full_hits = [
        row for row in rows if row["required_full_context_recall"] >= 1.0
    ]
    return {
        "label": label,
        "required_tasks": len(rows),
        "required_full_context_recall": sum(
            row["required_full_context_recall"] for row in rows
        )
        / len(rows),
        "tasks_with_all_required_context": len(full_hits),
        "missing_required_context_tasks": [
            row for row in rows if row["missing_required_paths"]
        ],
    }


def summarize_cost_by_bucket(
    label: str,
    requests: list[PreparedRequest],
    responses: list[dict],
) -> dict:
    buckets: dict[str, list[tuple[PreparedRequest, dict]]] = {}
    for request, response in zip(requests, responses):
        bucket = request.metadata.get("adaptive_router", {}).get(
            "bucket",
            route_policy_for_request_metadata(request.metadata).bucket,
        )
        buckets.setdefault(bucket, []).append((request, response))

    rows = {}
    for bucket, bucket_rows in sorted(buckets.items()):
        bucket_requests = [request for request, _ in bucket_rows]
        bucket_responses = [response for _, response in bucket_rows]
        summary = summarize(label, bucket_responses, Prices())
        rows[bucket] = {
            "requests": len(bucket_rows),
            "estimated_cost_usd": summary["estimated_cost_usd"],
            "prompt_tokens": summary["prompt_tokens"],
            "cache_hit_rate": summary["cache_hit_rate"],
            "required_recall": summarize_required_recall(label, bucket_requests),
        }
    return rows


def route_policy_for_request_metadata(metadata: dict) -> RoutePolicy:
    return route_policy_for_task(
        WorkloadTask(
            "",
            str(metadata.get("workload_family", "")),
            tuple(metadata.get("required_paths", [])),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--corpus", choices=["current", "large"], default="current")
    parser.add_argument(
        "--workload",
        choices=["mixed", "synthetic-targeted", "mixed-plus-targeted"],
        default="mixed",
    )
    parser.add_argument("--synthetic-file-count", type=int, default=60)
    parser.add_argument("--max-chars-per-file", type=int, default=16_000)
    parser.add_argument("--user-id", default="messy-agent-contextpilot-benchmark")
    parser.add_argument(
        "--include-adaptive-hybrid",
        action="store_true",
        help="Include the experimental adaptive-budget-shadow ablation arm.",
    )
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    repo_blocks = build_repo_blocks(
        args.corpus,
        args.max_chars_per_file,
        args.synthetic_file_count,
    )
    repo_context = repo_context_from_blocks(repo_blocks)
    tasks = build_tasks(args.rounds, args.workload)
    deepseek_native_requests = [
        PreparedRequest(
            task=task.task,
            messages=deepseek_native_messages(task, repo_context, index),
            metadata={
                "workload_family": task.family,
                "required_paths": list(task.required_paths),
                "source": "deepseek-native-full-repo",
            },
        )
        for index, task in enumerate(tasks)
    ]
    upstream_contextpilot = upstream_contextpilot_requests(tasks, repo_blocks)
    upstream_budget_shadow_pruned_contextpilot = (
        upstream_budget_shadow_pruned_contextpilot_requests(
            tasks,
            repo_blocks,
            upstream_contextpilot,
        )
    )
    fixed_general_contextpilot_requests = [
        contextpilot_messages(
            task,
            repo_blocks,
            index,
            repo_id="messy-repo",
            commit_hash="abc123",
            profile="general",
        )
        for index, task in enumerate(tasks)
    ]
    adaptive_contextpilot_requests = [
        contextpilot_messages(
            task,
            repo_blocks,
            index,
            repo_id="messy-repo",
            commit_hash="abc123",
        )
        for index, task in enumerate(tasks)
    ]
    adaptive_router_contextpilot_requests = adaptive_context_router_requests(
        tasks,
        repo_blocks,
        upstream_contextpilot,
        fixed_general_contextpilot_requests,
        adaptive_contextpilot_requests,
        upstream_budget_shadow_pruned_contextpilot,
    )
    adaptive_budget_shadow_contextpilot = (
        adaptive_budget_shadow_contextpilot_requests(
            upstream_budget_shadow_pruned_contextpilot,
            adaptive_contextpilot_requests,
        )
        if args.include_adaptive_hybrid
        else []
    )

    print(
        json.dumps(
            {
                "mode": "live" if args.live else "local",
                "corpus": args.corpus,
                "workload": args.workload,
                "rounds": len(tasks),
                "repo_files": len(repo_blocks),
                "synthetic_file_count": (
                    args.synthetic_file_count if args.corpus == "large" else 0
                ),
                "workload_families": {
                    family: sum(1 for task in tasks if task.family == family)
                    for family in sorted({task.family for task in tasks})
                },
                "repo_context_chars": len(repo_context),
                "deepseek_native_shape": summarize_local_shape(
                    "deepseek-native",
                    deepseek_native_requests,
                ),
                "deepseek_native_required_recall": summarize_required_recall(
                    "deepseek-native",
                    deepseek_native_requests,
                ),
                "upstream_contextpilot_shape": summarize_local_shape(
                    "upstream-contextpilot",
                    upstream_contextpilot,
                ),
                "upstream_contextpilot_required_recall": summarize_required_recall(
                    "upstream-contextpilot",
                    upstream_contextpilot,
                ),
                "fixed_general_contextpilot_shape": summarize_local_shape(
                    "fixed-general-contextpilot",
                    fixed_general_contextpilot_requests,
                ),
                "fixed_general_contextpilot_required_recall": summarize_required_recall(
                    "fixed-general-contextpilot",
                    fixed_general_contextpilot_requests,
                ),
                "upstream_budget_shadow_pruned_contextpilot_shape": summarize_local_shape(
                    "upstream-budget-shadow-pruned-contextpilot",
                    upstream_budget_shadow_pruned_contextpilot,
                ),
                "upstream_budget_shadow_pruned_contextpilot_required_recall": summarize_required_recall(
                    "upstream-budget-shadow-pruned-contextpilot",
                    upstream_budget_shadow_pruned_contextpilot,
                ),
                "adaptive_contextpilot_shape": summarize_local_shape(
                    "adaptive-contextpilot",
                    adaptive_contextpilot_requests,
                ),
                "adaptive_contextpilot_required_recall": summarize_required_recall(
                    "adaptive-contextpilot",
                    adaptive_contextpilot_requests,
                ),
                "adaptive_router_contextpilot_shape": summarize_local_shape(
                    "adaptive-router-contextpilot",
                    adaptive_router_contextpilot_requests,
                ),
                "adaptive_router_contextpilot_required_recall": summarize_required_recall(
                    "adaptive-router-contextpilot",
                    adaptive_router_contextpilot_requests,
                ),
                **(
                    {
                        "adaptive_budget_shadow_contextpilot_shape": summarize_local_shape(
                            "adaptive-budget-shadow-contextpilot",
                            adaptive_budget_shadow_contextpilot,
                        ),
                        "adaptive_budget_shadow_contextpilot_required_recall": summarize_required_recall(
                            "adaptive-budget-shadow-contextpilot",
                            adaptive_budget_shadow_contextpilot,
                        ),
                    }
                    if args.include_adaptive_hybrid
                    else {}
                ),
            },
            indent=2,
        )
    )

    api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_native = run_arm(
        "deepseek-native",
        deepseek_native_requests,
        args.model,
        f"{args.user_id}-deepseek-native",
        live=args.live,
        api_key=api_key,
    )
    upstream_contextpilot_responses = run_arm(
        "upstream-contextpilot",
        upstream_contextpilot,
        args.model,
        f"{args.user_id}-upstream-contextpilot",
        live=args.live,
        api_key=api_key,
    )
    fixed_general_contextpilot = run_arm(
        "fixed-general-contextpilot",
        fixed_general_contextpilot_requests,
        args.model,
        f"{args.user_id}-fixed-general-contextpilot",
        live=args.live,
        api_key=api_key,
    )
    upstream_budget_shadow_pruned_contextpilot_responses = run_arm(
        "upstream-budget-shadow-pruned-contextpilot",
        upstream_budget_shadow_pruned_contextpilot,
        args.model,
        f"{args.user_id}-upstream-budget-shadow-pruned-contextpilot",
        live=args.live,
        api_key=api_key,
    )
    adaptive_budget_shadow_contextpilot_responses = (
        run_arm(
            "adaptive-budget-shadow-contextpilot",
            adaptive_budget_shadow_contextpilot,
            args.model,
            f"{args.user_id}-adaptive-budget-shadow-contextpilot",
            live=args.live,
            api_key=api_key,
        )
        if args.include_adaptive_hybrid
        else []
    )
    adaptive_contextpilot = run_arm(
        "adaptive-contextpilot",
        adaptive_contextpilot_requests,
        args.model,
        f"{args.user_id}-adaptive-contextpilot",
        live=args.live,
        api_key=api_key,
    )
    adaptive_router_contextpilot = run_arm(
        "adaptive-router-contextpilot",
        adaptive_router_contextpilot_requests,
        args.model,
        f"{args.user_id}-adaptive-router-contextpilot",
        live=args.live,
        api_key=api_key,
    )

    deepseek_native_summary = summarize("deepseek-native", deepseek_native, Prices())
    upstream_contextpilot_summary = summarize(
        "upstream-contextpilot",
        upstream_contextpilot_responses,
        Prices(),
    )
    fixed_general_contextpilot_summary = summarize(
        "fixed-general-contextpilot",
        fixed_general_contextpilot,
        Prices(),
    )
    upstream_budget_shadow_pruned_contextpilot_summary = summarize(
        "upstream-budget-shadow-pruned-contextpilot",
        upstream_budget_shadow_pruned_contextpilot_responses,
        Prices(),
    )
    adaptive_budget_shadow_contextpilot_summary = (
        summarize(
            "adaptive-budget-shadow-contextpilot",
            adaptive_budget_shadow_contextpilot_responses,
            Prices(),
        )
        if args.include_adaptive_hybrid
        else None
    )
    adaptive_contextpilot_summary = summarize(
        "adaptive-contextpilot",
        adaptive_contextpilot,
        Prices(),
    )
    adaptive_router_contextpilot_summary = summarize(
        "adaptive-router-contextpilot",
        adaptive_router_contextpilot,
        Prices(),
    )
    no_internal_cache_cost = deepseek_native_summary["estimated_no_internal_cache_cost_usd"]
    result = {
        "comparison": {
            "1_deepseek_v4_flash_no_internal_cache_counterfactual_usd": no_internal_cache_cost,
            "2_deepseek_v4_flash_with_internal_cache_usd": deepseek_native_summary[
                "estimated_cost_usd"
            ],
            "3_upstream_contextpilot_with_deepseek_internal_cache_usd": upstream_contextpilot_summary[
                "estimated_cost_usd"
            ],
            "4_fixed_general_contextpilot_with_deepseek_internal_cache_usd": fixed_general_contextpilot_summary[
                "estimated_cost_usd"
            ],
            "5_upstream_budget_shadow_pruned_contextpilot_with_deepseek_internal_cache_usd": upstream_budget_shadow_pruned_contextpilot_summary[
                "estimated_cost_usd"
            ],
            "6_adaptive_contextpilot_with_deepseek_internal_cache_usd": adaptive_contextpilot_summary[
                "estimated_cost_usd"
            ],
            "7_adaptive_router_contextpilot_with_deepseek_internal_cache_usd": adaptive_router_contextpilot_summary[
                "estimated_cost_usd"
            ],
            **(
                {
                    "8_adaptive_budget_shadow_contextpilot_with_deepseek_internal_cache_usd": adaptive_budget_shadow_contextpilot_summary[
                        "estimated_cost_usd"
                    ]
                }
                if adaptive_budget_shadow_contextpilot_summary
                else {}
            ),
            "deepseek_internal_cache_savings_vs_no_cache_pct": (
                (
                    no_internal_cache_cost
                    - deepseek_native_summary["estimated_cost_usd"]
                )
                / no_internal_cache_cost
                if no_internal_cache_cost
                else None
            ),
            "upstream_contextpilot_incremental_savings_vs_deepseek_internal_cache_pct": (
                (
                    deepseek_native_summary["estimated_cost_usd"]
                    - upstream_contextpilot_summary["estimated_cost_usd"]
                )
                / deepseek_native_summary["estimated_cost_usd"]
                if deepseek_native_summary["estimated_cost_usd"]
                else None
            ),
            "fixed_general_contextpilot_incremental_savings_vs_deepseek_internal_cache_pct": (
                (
                    deepseek_native_summary["estimated_cost_usd"]
                    - fixed_general_contextpilot_summary["estimated_cost_usd"]
                )
                / deepseek_native_summary["estimated_cost_usd"]
                if deepseek_native_summary["estimated_cost_usd"]
                else None
            ),
            "adaptive_contextpilot_incremental_savings_vs_deepseek_internal_cache_pct": (
                (
                    deepseek_native_summary["estimated_cost_usd"]
                    - adaptive_contextpilot_summary["estimated_cost_usd"]
                )
                / deepseek_native_summary["estimated_cost_usd"]
                if deepseek_native_summary["estimated_cost_usd"]
                else None
            ),
            "upstream_budget_shadow_pruned_incremental_savings_vs_upstream_contextpilot_pct": (
                (
                    upstream_contextpilot_summary["estimated_cost_usd"]
                    - upstream_budget_shadow_pruned_contextpilot_summary["estimated_cost_usd"]
                )
                / upstream_contextpilot_summary["estimated_cost_usd"]
                if upstream_contextpilot_summary["estimated_cost_usd"]
                else None
            ),
            **(
                {
                    "adaptive_budget_shadow_incremental_savings_vs_upstream_budget_shadow_pct": (
                        (
                            upstream_budget_shadow_pruned_contextpilot_summary["estimated_cost_usd"]
                            - adaptive_budget_shadow_contextpilot_summary["estimated_cost_usd"]
                        )
                        / upstream_budget_shadow_pruned_contextpilot_summary["estimated_cost_usd"]
                        if upstream_budget_shadow_pruned_contextpilot_summary["estimated_cost_usd"]
                        else None
                    ),
                    "adaptive_budget_shadow_incremental_savings_vs_adaptive_contextpilot_pct": (
                        (
                            adaptive_contextpilot_summary["estimated_cost_usd"]
                            - adaptive_budget_shadow_contextpilot_summary["estimated_cost_usd"]
                        )
                        / adaptive_contextpilot_summary["estimated_cost_usd"]
                        if adaptive_contextpilot_summary["estimated_cost_usd"]
                        else None
                    ),
                    "adaptive_budget_shadow_incremental_savings_vs_upstream_contextpilot_pct": (
                        (
                            upstream_contextpilot_summary["estimated_cost_usd"]
                            - adaptive_budget_shadow_contextpilot_summary["estimated_cost_usd"]
                        )
                        / upstream_contextpilot_summary["estimated_cost_usd"]
                        if upstream_contextpilot_summary["estimated_cost_usd"]
                        else None
                    ),
                }
                if adaptive_budget_shadow_contextpilot_summary
                else {}
            ),
            "fixed_general_incremental_savings_vs_upstream_contextpilot_pct": (
                (
                    upstream_contextpilot_summary["estimated_cost_usd"]
                    - fixed_general_contextpilot_summary["estimated_cost_usd"]
                )
                / upstream_contextpilot_summary["estimated_cost_usd"]
                if upstream_contextpilot_summary["estimated_cost_usd"]
                else None
            ),
            "adaptive_incremental_savings_vs_fixed_general_contextpilot_pct": (
                (
                    fixed_general_contextpilot_summary["estimated_cost_usd"]
                    - adaptive_contextpilot_summary["estimated_cost_usd"]
                )
                / fixed_general_contextpilot_summary["estimated_cost_usd"]
                if fixed_general_contextpilot_summary["estimated_cost_usd"]
                else None
            ),
            "adaptive_router_incremental_savings_vs_upstream_contextpilot_pct": (
                (
                    upstream_contextpilot_summary["estimated_cost_usd"]
                    - adaptive_router_contextpilot_summary["estimated_cost_usd"]
                )
                / upstream_contextpilot_summary["estimated_cost_usd"]
                if upstream_contextpilot_summary["estimated_cost_usd"]
                else None
            ),
            "adaptive_router_incremental_savings_vs_adaptive_contextpilot_pct": (
                (
                    adaptive_contextpilot_summary["estimated_cost_usd"]
                    - adaptive_router_contextpilot_summary["estimated_cost_usd"]
                )
                / adaptive_contextpilot_summary["estimated_cost_usd"]
                if adaptive_contextpilot_summary["estimated_cost_usd"]
                else None
            ),
            "upstream_contextpilot_total_savings_vs_no_cache_pct": (
                (no_internal_cache_cost - upstream_contextpilot_summary["estimated_cost_usd"])
                / no_internal_cache_cost
                if no_internal_cache_cost
                else None
            ),
            "fixed_general_contextpilot_total_savings_vs_no_cache_pct": (
                (no_internal_cache_cost - fixed_general_contextpilot_summary["estimated_cost_usd"])
                / no_internal_cache_cost
                if no_internal_cache_cost
                else None
            ),
            "upstream_budget_shadow_pruned_contextpilot_total_savings_vs_no_cache_pct": (
                (
                    no_internal_cache_cost
                    - upstream_budget_shadow_pruned_contextpilot_summary[
                        "estimated_cost_usd"
                    ]
                )
                / no_internal_cache_cost
                if no_internal_cache_cost
                else None
            ),
            **(
                {
                    "adaptive_budget_shadow_contextpilot_total_savings_vs_no_cache_pct": (
                        (
                            no_internal_cache_cost
                            - adaptive_budget_shadow_contextpilot_summary[
                                "estimated_cost_usd"
                            ]
                        )
                        / no_internal_cache_cost
                        if no_internal_cache_cost
                        else None
                    )
                }
                if adaptive_budget_shadow_contextpilot_summary
                else {}
            ),
            "adaptive_contextpilot_total_savings_vs_no_cache_pct": (
                (no_internal_cache_cost - adaptive_contextpilot_summary["estimated_cost_usd"])
                / no_internal_cache_cost
                if no_internal_cache_cost
                else None
            ),
            "adaptive_router_contextpilot_total_savings_vs_no_cache_pct": (
                (
                    no_internal_cache_cost
                    - adaptive_router_contextpilot_summary["estimated_cost_usd"]
                )
                / no_internal_cache_cost
                if no_internal_cache_cost
                else None
            ),
        },
        "required_context_quality": {
            "deepseek_native": summarize_required_recall(
                "deepseek-native",
                deepseek_native_requests,
            ),
            "upstream_contextpilot": summarize_required_recall(
                "upstream-contextpilot",
                upstream_contextpilot,
            ),
            "fixed_general_contextpilot": summarize_required_recall(
                "fixed-general-contextpilot",
                fixed_general_contextpilot_requests,
            ),
            "upstream_budget_shadow_pruned_contextpilot": summarize_required_recall(
                "upstream-budget-shadow-pruned-contextpilot",
                upstream_budget_shadow_pruned_contextpilot,
            ),
            "adaptive_contextpilot": summarize_required_recall(
                "adaptive-contextpilot",
                adaptive_contextpilot_requests,
            ),
            "adaptive_router_contextpilot": summarize_required_recall(
                "adaptive-router-contextpilot",
                adaptive_router_contextpilot_requests,
            ),
            **(
                {
                    "adaptive_budget_shadow_contextpilot": summarize_required_recall(
                        "adaptive-budget-shadow-contextpilot",
                        adaptive_budget_shadow_contextpilot,
                    )
                }
                if args.include_adaptive_hybrid
                else {}
            ),
        },
        "deepseek_no_internal_cache_counterfactual": {
            **deepseek_native_summary,
            "label": "deepseek-no-internal-cache-counterfactual",
            "estimated_cost_usd": no_internal_cache_cost,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": deepseek_native_summary["prompt_tokens"],
            "cache_hit_rate": 0.0,
        },
        "deepseek_native": deepseek_native_summary,
        "upstream_contextpilot": upstream_contextpilot_summary,
        "fixed_general_contextpilot": fixed_general_contextpilot_summary,
        "upstream_budget_shadow_pruned_contextpilot": upstream_budget_shadow_pruned_contextpilot_summary,
        "adaptive_contextpilot": adaptive_contextpilot_summary,
        "adaptive_router_contextpilot": adaptive_router_contextpilot_summary,
        "cost_by_workload_bucket": {
            "deepseek_native": summarize_cost_by_bucket(
                "deepseek-native",
                deepseek_native_requests,
                deepseek_native,
            ),
            "upstream_contextpilot": summarize_cost_by_bucket(
                "upstream-contextpilot",
                upstream_contextpilot,
                upstream_contextpilot_responses,
            ),
            "upstream_budget_shadow_pruned_contextpilot": summarize_cost_by_bucket(
                "upstream-budget-shadow-pruned-contextpilot",
                upstream_budget_shadow_pruned_contextpilot,
                upstream_budget_shadow_pruned_contextpilot_responses,
            ),
            "adaptive_router_contextpilot": summarize_cost_by_bucket(
                "adaptive-router-contextpilot",
                adaptive_router_contextpilot_requests,
                adaptive_router_contextpilot,
            ),
        },
        **(
            {
                "adaptive_budget_shadow_contextpilot": adaptive_budget_shadow_contextpilot_summary,
            }
            if adaptive_budget_shadow_contextpilot_summary
            else {}
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
