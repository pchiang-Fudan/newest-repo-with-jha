from __future__ import annotations

import re
from dataclasses import dataclass

from .context_store import ContextBlock


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*")
DEF_RE = re.compile(r"^\s*(?:class|def|async def)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


@dataclass(frozen=True)
class PlannedContext:
    map_blocks: list[ContextBlock]
    full_blocks: list[ContextBlock]
    telemetry: dict


@dataclass(frozen=True)
class ContextPlanningPolicy:
    profile: str
    max_repo_chars: int
    max_full_files: int
    include_repo_map: bool
    path_boosts: dict[str, int]
    reasons: tuple[str, ...]


def plan_context_blocks(
    messages: list[dict],
    blocks: list[ContextBlock],
    max_repo_chars: int | None = None,
    max_full_files: int | None = None,
    include_repo_map: bool | None = None,
    profile: str | None = None,
) -> PlannedContext:
    ordered_blocks = sorted(blocks, key=lambda block: block.path)
    if not ordered_blocks:
        return PlannedContext([], [], {"enabled": True, "blocks_considered": 0})

    task_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message.get("content"), str)
    )
    policy = classify_context_policy(task_text, ordered_blocks, profile=profile)
    effective_max_repo_chars = (
        policy.max_repo_chars if max_repo_chars is None else max_repo_chars
    )
    effective_max_full_files = (
        policy.max_full_files if max_full_files is None else max_full_files
    )
    effective_include_repo_map = (
        policy.include_repo_map if include_repo_map is None else include_repo_map
    )
    scored = [
        (
            score_block_relevance(
                task_text,
                block,
                path_boosts=policy.path_boosts,
            ),
            block,
        )
        for block in ordered_blocks
    ]
    ranked = sorted(scored, key=lambda row: (-row[0], row[1].path))
    selected: list[ContextBlock] = []
    used_chars = 0
    for score, block in ranked:
        if score <= 0 and selected:
            continue
        if len(selected) >= effective_max_full_files:
            continue
        block_chars = len(canonicalize_context_text(block.content))
        if used_chars + block_chars > effective_max_repo_chars and selected:
            continue
        selected.append(block)
        used_chars += block_chars

    selected_paths = {block.path for block in selected}
    map_blocks = ordered_blocks if effective_include_repo_map else []
    return PlannedContext(
        map_blocks=map_blocks,
        full_blocks=selected,
        telemetry={
            "enabled": True,
            "blocks_considered": len(blocks),
            "repo_map_blocks": len(map_blocks),
            "full_blocks": len(selected),
            "full_block_paths": [block.path for block in selected],
            "omitted_full_block_paths": [
                block.path for block in ordered_blocks if block.path not in selected_paths
            ],
            "policy_profile": policy.profile,
            "policy_reasons": list(policy.reasons),
            "policy_path_boosts": policy.path_boosts,
            "max_repo_chars": effective_max_repo_chars,
            "max_full_files": effective_max_full_files,
            "full_block_chars": used_chars,
            "scores": {
                block.path: score for score, block in sorted(scored, key=lambda row: row[1].path)
            },
        },
    )


def classify_context_policy(
    task_text: str,
    blocks: list[ContextBlock],
    profile: str | None = None,
) -> ContextPlanningPolicy:
    lowered = task_text.lower()
    primary_text = lowered.split("volatile agent artifacts follow.", 1)[0]
    known_profiles = {
        "focused": ContextPlanningPolicy(
            "focused", 14_000, 2, True, {}, ("explicit focused profile",)
        ),
        "debug": ContextPlanningPolicy(
            "debug", 30_000, 4, True, {"tests/": 18, "test_": 12}, ("explicit debug profile",)
        ),
        "review": ContextPlanningPolicy(
            "review", 24_000, 3, True, {}, ("explicit review profile",)
        ),
        "refactor": ContextPlanningPolicy(
            "refactor", 30_000, 4, True, {}, ("explicit refactor profile",)
        ),
        "docs": ContextPlanningPolicy(
            "docs", 16_000, 2, True, {"readme": 18, "docs/": 18, ".md": 10}, ("explicit docs profile",)
        ),
        "handoff": ContextPlanningPolicy(
            "handoff", 14_000, 2, True, {}, ("explicit handoff profile",)
        ),
        "general": ContextPlanningPolicy(
            "general", 24_000, 3, True, {}, ("explicit general profile",)
        ),
    }
    if profile in known_profiles:
        return known_profiles[profile]

    explicit_paths = [
        block.path
        for block in blocks
        if block.path.lower() in lowered
    ]
    reasons: list[str] = []
    if explicit_paths and not has_any_signal(primary_text, ("handoff", "prior agent", "continue")):
        if len(explicit_paths) <= 2:
            reasons.append("explicit path mention")
            return ContextPlanningPolicy(
                "focused",
                max_repo_chars=16_000,
                max_full_files=min(2, max(1, len(explicit_paths) + 1)),
                include_repo_map=True,
                path_boosts={path: 40 for path in explicit_paths},
                reasons=tuple(reasons),
            )
        reasons.append("repo map path inventory")

    if has_any_signal(primary_text, ("readme", "docs", "onboarding", "architecture note", "explain")):
        reasons.append("documentation or explanation task")
        return ContextPlanningPolicy(
            "docs",
            max_repo_chars=16_000,
            max_full_files=2,
            include_repo_map=True,
            path_boosts={"readme": 18, "docs/": 18, ".md": 10},
            reasons=tuple(reasons),
        )

    if has_any_signal(primary_text, ("refactor", "extraction", "adapter", "simplify", "interface")):
        reasons.append("cross-file refactor task")
        return ContextPlanningPolicy(
            "refactor",
            max_repo_chars=30_000,
            max_full_files=4,
            include_repo_map=True,
            path_boosts={},
            reasons=tuple(reasons),
        )

    if has_any_signal(primary_text, ("handoff", "prior agent", "continue", "summarize repo state")):
        reasons.append("handoff task")
        return ContextPlanningPolicy(
            "handoff",
            max_repo_chars=14_000,
            max_full_files=2,
            include_repo_map=True,
            path_boosts={},
            reasons=tuple(reasons),
        )

    if has_any_signal(
        lowered,
        ("failing test", "pytest", "traceback", "ci", "regression test", "debug"),
    ):
        reasons.append("test or failure artifact")
        return ContextPlanningPolicy(
            "debug",
            max_repo_chars=30_000,
            max_full_files=4,
            include_repo_map=True,
            path_boosts={"tests/": 18, "test_": 12, "pytest": 8},
            reasons=tuple(reasons),
        )

    if has_any_signal(primary_text, ("review", "audit", "risk", "behavioral")):
        reasons.append("code review task")
        return ContextPlanningPolicy(
            "review",
            max_repo_chars=24_000,
            max_full_files=3,
            include_repo_map=True,
            path_boosts={},
            reasons=tuple(reasons),
        )

    return ContextPlanningPolicy(
        "general",
        max_repo_chars=24_000,
        max_full_files=3,
        include_repo_map=True,
        path_boosts={},
        reasons=("fallback general policy",),
    )


def has_any_signal(text: str, signals: tuple[str, ...]) -> bool:
    for signal in signals:
        if " " in signal:
            if signal in text:
                return True
        elif re.search(rf"(?<![a-z0-9_]){re.escape(signal)}(?![a-z0-9_])", text):
            return True
    return False


def score_block_relevance(
    task_text: str,
    block: ContextBlock,
    path_boosts: dict[str, int] | None = None,
) -> int:
    lowered_task = task_text.lower()
    path_lower = block.path.lower()
    score = 0
    for marker, boost in (path_boosts or {}).items():
        if marker.lower() in path_lower:
            score += boost
    if path_lower in lowered_task:
        score += 20
    filename = path_lower.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0]
    if filename and filename in lowered_task:
        score += 12
    if stem and stem in lowered_task:
        score += 8
    for part in path_lower.split("/"):
        if part and part in lowered_task:
            score += 3
    for symbol in extract_symbols(block.content)[:40]:
        if symbol.lower() in lowered_task:
            score += 5
    task_tokens = set(token.lower() for token in TOKEN_RE.findall(task_text))
    path_tokens = set(token.lower() for token in re.split(r"[/_.-]+", block.path) if token)
    score += min(10, len(task_tokens & path_tokens) * 2)
    return score


def canonicalize_context_text(content: str) -> str:
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def extract_symbols(content: str) -> list[str]:
    symbols = DEF_RE.findall(content)
    seen: set[str] = set()
    deduped: list[str] = []
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


def summarize_block(block: ContextBlock, max_symbols: int = 8) -> str:
    content = canonicalize_context_text(block.content)
    symbols = extract_symbols(content)[:max_symbols]
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    summary_parts = [f"path: {block.path}"]
    if symbols:
        summary_parts.append("symbols: " + ", ".join(symbols))
    if first_line:
        summary_parts.append("first_line: " + first_line[:160])
    return "\n".join(summary_parts)
