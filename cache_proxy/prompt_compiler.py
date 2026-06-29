from __future__ import annotations

from typing import Any

from .context_store import ContextBlock
from .context_planner import summarize_block


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def canonicalize_context_text(content: str) -> str:
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def compile_messages(
    messages: list[dict],
    blocks: list[ContextBlock],
    repo_id: str | None,
    commit_hash: str | None,
) -> list[dict]:
    """Build a cache-friendly message list.

    Stable context is placed before volatile user/task content. This is the
    simple version; later we can add block budgets, summaries, and ranking.
    """
    system_messages = [m for m in messages if m.get("role") == "system"]
    other_messages = [m for m in messages if m.get("role") != "system"]

    compiled = list(system_messages)
    if blocks:
        compiled.append(
            {
                "role": "system",
                "content": render_repo_context(blocks, repo_id, commit_hash),
            }
        )
    compiled.extend(other_messages)
    return compiled


def compile_planned_messages(
    messages: list[dict],
    map_blocks: list[ContextBlock],
    full_blocks: list[ContextBlock],
    repo_id: str | None,
    commit_hash: str | None,
) -> list[dict]:
    system_messages = [m for m in messages if m.get("role") == "system"]
    other_messages = [m for m in messages if m.get("role") != "system"]

    compiled = list(system_messages)
    compiled.append(
        {
            "role": "system",
            "content": render_repo_map(map_blocks, repo_id, commit_hash),
        }
    )
    compiled.append(
        {
            "role": "system",
            "content": render_repo_context(full_blocks, repo_id, commit_hash),
        }
    )
    compiled.extend(other_messages)
    return compiled


def render_repo_context(
    blocks: list[ContextBlock],
    repo_id: str | None,
    commit_hash: str | None,
) -> str:
    lines: list[str] = []
    lines.append("Repository context follows. Treat it as read-only reference.")
    if repo_id:
        lines.append(f"repo_id: {repo_id}")
    if commit_hash:
        lines.append(f"commit_hash: {commit_hash}")
    lines.append("")
    for block in sorted(blocks, key=lambda b: b.path):
        lines.append(f"--- file: {block.path}")
        lines.append(f"sha256: {block.content_hash}")
        lines.append("```")
        lines.append(canonicalize_context_text(block.content))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).strip()


def render_repo_map(
    blocks: list[ContextBlock],
    repo_id: str | None,
    commit_hash: str | None,
) -> str:
    lines: list[str] = []
    lines.append("Repository map follows. Use it for orientation; full file bodies may be partial.")
    if repo_id:
        lines.append(f"repo_id: {repo_id}")
    if commit_hash:
        lines.append(f"commit_hash: {commit_hash}")
    lines.append("")
    for block in sorted(blocks, key=lambda b: b.path):
        lines.append(f"--- map: {block.path}")
        lines.append(summarize_block(block))
        lines.append("")
    return "\n".join(lines).strip()


def estimate_payload_change(original: list[dict], compiled: list[dict]) -> dict:
    original_chars = sum(len(_content_text(m.get("content", ""))) for m in original)
    compiled_chars = sum(len(_content_text(m.get("content", ""))) for m in compiled)
    return {
        "original_chars": original_chars,
        "compiled_chars": compiled_chars,
        "added_chars": compiled_chars - original_chars,
    }
