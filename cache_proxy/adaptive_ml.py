from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .context_planner import TOKEN_RE, extract_symbols
from .context_store import ContextBlock, ContextStore


DEFAULT_WEIGHTS = {
    "bias": -1.35,
    "path_exact": 4.5,
    "filename": 2.2,
    "stem": 1.5,
    "path_token_overlap": 0.9,
    "symbol_overlap": 1.3,
    "test_path_debug": 1.4,
    "docs_path_docs": 1.4,
    "code_path_review": 0.8,
    "code_path_refactor": 0.7,
    "short_file": 0.25,
}


@dataclass(frozen=True)
class MlRankedBlock:
    block: ContextBlock
    score: float
    features: dict[str, float]


@dataclass(frozen=True)
class MlSelection:
    selected_blocks: list[ContextBlock]
    ranked_blocks: list[MlRankedBlock]
    telemetry: dict


def select_context_with_ml(
    store: ContextStore,
    repo_id: str,
    task_text: str,
    blocks: list[ContextBlock],
    required_paths: list[str],
    max_blocks: int = 3,
    min_score: float = 0.58,
) -> MlSelection:
    weights = {**DEFAULT_WEIGHTS, **store.get_ml_weights(repo_id)}
    ranked = rank_blocks(task_text, blocks, weights)
    required = {path for path in required_paths if isinstance(path, str)}
    selected: list[ContextBlock] = []
    seen: set[str] = set()

    for row in ranked:
        if row.block.path in required:
            selected.append(row.block)
            seen.add(row.block.path)

    for row in ranked:
        if row.block.path in seen:
            continue
        if row.score < min_score and selected:
            continue
        if len(selected) >= max_blocks:
            break
        selected.append(row.block)
        seen.add(row.block.path)

    feedback_count = store.ml_feedback_count(repo_id)
    confident = bool(selected) and (
        feedback_count > 0
        or any(row.score >= 0.80 for row in ranked[: min(3, len(ranked))])
        or bool(required)
    )
    return MlSelection(
        selected_blocks=selected if confident else [],
        ranked_blocks=ranked,
        telemetry={
            "enabled": True,
            "model": "online-logistic-file-ranker",
            "feedback_examples": feedback_count,
            "confidence": "learned" if feedback_count else "prior",
            "used": confident,
            "min_score": min_score,
            "top_scores": [
                {"path": row.block.path, "score": round(row.score, 4)}
                for row in ranked[:5]
            ],
            "selected_paths": [block.path for block in selected] if confident else [],
        },
    )


def train_from_feedback(
    store: ContextStore,
    repo_id: str,
    commit_hash: str | None,
    task_text: str,
    blocks: list[ContextBlock],
    positive_paths: list[str],
    negative_paths: list[str] | None = None,
    learning_rate: float = 0.18,
) -> dict:
    positive = {path for path in positive_paths if isinstance(path, str)}
    negative = {path for path in (negative_paths or []) if isinstance(path, str)}
    if not positive:
        return {"updated": False, "reason": "no positive paths"}

    block_by_path = {block.path: block for block in blocks}
    training_paths = [path for path in sorted(positive | negative) if path in block_by_path]
    if not training_paths:
        return {"updated": False, "reason": "no feedback paths found in repo context"}

    weights = {**DEFAULT_WEIGHTS, **store.get_ml_weights(repo_id)}
    updates: dict[str, float] = {}
    examples = 0
    for path in training_paths:
        label = 1.0 if path in positive else 0.0
        features = extract_features(task_text, block_by_path[path])
        prediction = predict(features, weights)
        error = label - prediction
        for feature, value in features.items():
            updates[feature] = updates.get(feature, 0.0) + learning_rate * error * value
        examples += 1

    store.update_ml_weights(repo_id, updates)
    store.log_ml_feedback(
        repo_id=repo_id,
        commit_hash=commit_hash,
        task_text=task_text,
        positive_paths=sorted(positive),
        negative_paths=sorted(negative),
    )
    return {
        "updated": True,
        "examples": examples,
        "positive_paths": sorted(positive),
        "negative_paths": sorted(negative),
        "features_updated": len(updates),
    }


def rank_blocks(
    task_text: str,
    blocks: list[ContextBlock],
    weights: dict[str, float],
) -> list[MlRankedBlock]:
    ranked = [
        MlRankedBlock(
            block=block,
            score=predict(extract_features(task_text, block), weights),
            features=extract_features(task_text, block),
        )
        for block in blocks
    ]
    return sorted(ranked, key=lambda row: (-row.score, row.block.path))


def predict(features: dict[str, float], weights: dict[str, float]) -> float:
    z = sum(weights.get(feature, 0.0) * value for feature, value in features.items())
    if z < -40:
        return 0.0
    if z > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def extract_features(task_text: str, block: ContextBlock) -> dict[str, float]:
    lowered = task_text.lower()
    path_lower = block.path.lower()
    filename = path_lower.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0]
    task_tokens = set(token.lower() for token in TOKEN_RE.findall(task_text))
    path_tokens = set(token.lower() for token in re.split(r"[/_.-]+", block.path) if token)
    symbol_tokens = {symbol.lower() for symbol in extract_symbols(block.content)[:60]}

    task_family = _task_family(lowered)
    features = {
        "bias": 1.0,
        "path_exact": 1.0 if path_lower in lowered else 0.0,
        "filename": 1.0 if filename and filename in lowered else 0.0,
        "stem": 1.0 if stem and stem in lowered else 0.0,
        "path_token_overlap": min(1.0, len(task_tokens & path_tokens) / 4.0),
        "symbol_overlap": min(1.0, len(task_tokens & symbol_tokens) / 3.0),
        "test_path_debug": 1.0 if task_family == "debug" and _is_test_path(path_lower) else 0.0,
        "docs_path_docs": 1.0 if task_family == "docs" and _is_docs_path(path_lower) else 0.0,
        "code_path_review": 1.0 if task_family == "review" and _is_code_path(path_lower) else 0.0,
        "code_path_refactor": 1.0 if task_family == "refactor" and _is_code_path(path_lower) else 0.0,
        "short_file": 1.0 if len(block.content) < 12_000 else 0.0,
    }
    return features


def _task_family(lowered: str) -> str:
    if any(term in lowered for term in ("pytest", "traceback", "failing", "debug", "ci")):
        return "debug"
    if any(term in lowered for term in ("docs", "onboarding", "explain", "readme")):
        return "docs"
    if any(term in lowered for term in ("refactor", "adapter", "split", "interface")):
        return "refactor"
    if any(term in lowered for term in ("review", "audit", "risk", "behavioral")):
        return "review"
    return "general"


def _is_test_path(path: str) -> bool:
    return "/tests/" in f"/{path}" or path.startswith("tests/") or "test_" in path


def _is_docs_path(path: str) -> bool:
    return path.startswith("docs/") or path.endswith(".md") or "readme" in path


def _is_code_path(path: str) -> bool:
    return path.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".cpp", ".h"))
