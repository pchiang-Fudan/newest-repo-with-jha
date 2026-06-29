from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .adaptive_ml import select_context_with_ml, train_from_feedback
from .context_store import ContextStore
from .context_planner import plan_context_blocks
from .prompt_compiler import compile_messages, compile_planned_messages, estimate_payload_change
from .benchmark_messy_agents import (
    WorkloadTask,
    infer_context_guard_paths,
    local_usage,
    route_policy_for_task,
    slice_context_blocks,
)


CACHE_PROXY_PROVIDER = os.getenv("CACHE_PROXY_PROVIDER", "deepseek").strip().lower()
CACHE_PROXY_BASE_URL = os.getenv("CACHE_PROXY_BASE_URL")
CACHE_PROXY_API_KEY = os.getenv("CACHE_PROXY_API_KEY")
CACHE_PROXY_API_KEY_ENV = os.getenv("CACHE_PROXY_API_KEY_ENV")
CACHE_PROXY_PUBLIC_TOKEN = os.getenv("CACHE_PROXY_PUBLIC_TOKEN")
DATA_DIR = Path(os.getenv("CACHE_PROXY_DATA_DIR", "cache_proxy_data"))
STORE = ContextStore(DATA_DIR / "cache_proxy.sqlite3")
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Adaptive-Router Context Proxy")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    provider = _provider_config()
    return {
        "ok": True,
        "provider": provider["provider"],
        "base_url": provider["base_url"],
        "api_key_env": provider["api_key_env"],
        "has_api_key": bool(provider["api_key"]),
    }


@app.get("/ui/config")
def ui_config() -> dict:
    provider = _provider_config()
    return {
        "provider": provider["provider"],
        "base_url": provider["base_url"],
        "api_key_env": provider["api_key_env"],
        "has_api_key": bool(provider["api_key"]),
        "requires_public_token": bool(CACHE_PROXY_PUBLIC_TOKEN),
    }


@app.post("/v1/repos/{repo_id}/context")
async def upsert_repo_context(repo_id: str, request: Request) -> dict:
    _require_public_token(request)
    body = await request.json()
    commit_hash = body.get("commit_hash")
    blocks = body.get("blocks", [])
    if not commit_hash:
        raise HTTPException(status_code=400, detail="commit_hash is required")
    if not isinstance(blocks, list):
        raise HTTPException(status_code=400, detail="blocks must be a list")
    saved = STORE.upsert_blocks(repo_id, commit_hash, blocks)
    return {
        "repo_id": repo_id,
        "commit_hash": commit_hash,
        "blocks": len(saved),
    }


@app.get("/v1/telemetry/summary")
def telemetry_summary(request: Request, limit: int = 100) -> dict:
    _require_public_token(request)
    return STORE.telemetry_summary(limit=limit)


@app.post("/v1/ml/feedback")
async def ml_feedback(request: Request) -> dict:
    _require_public_token(request)
    body = await request.json()
    repo_id = body.get("repo_id")
    commit_hash = body.get("commit_hash")
    messages = body.get("messages", [])
    if not repo_id or not commit_hash:
        raise HTTPException(status_code=400, detail="repo_id and commit_hash are required")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    blocks = STORE.get_blocks(repo_id, commit_hash)
    result = train_from_feedback(
        STORE,
        repo_id=repo_id,
        commit_hash=commit_hash,
        task_text=_messages_text(messages),
        blocks=blocks,
        positive_paths=body.get("positive_paths") or [],
        negative_paths=body.get("negative_paths") or [],
        learning_rate=float(body.get("learning_rate", 0.18)),
    )
    return {
        "repo_id": repo_id,
        "commit_hash": commit_hash,
        "adaptive_ml": result,
        "feedback_examples": STORE.ml_feedback_count(repo_id),
    }


@app.post("/v1/adaptive-router/plan")
async def adaptive_router_plan(request: Request) -> dict:
    _require_public_token(request)
    body = await request.json()
    return _adaptive_router_plan_payload(body)


@app.post("/v1/demo/plan")
async def demo_plan(request: Request) -> dict:
    body = await request.json()
    _seed_demo_context()
    body["repo_id"] = "demo"
    body["commit_hash"] = "abc123"
    return _adaptive_router_plan_payload(body)


def _adaptive_router_plan_payload(body: dict) -> dict:
    repo_id = body.get("repo_id")
    commit_hash = body.get("commit_hash")
    messages = body.get("messages", [])
    if not repo_id or not commit_hash:
        raise HTTPException(status_code=400, detail="repo_id and commit_hash are required")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    blocks = STORE.get_blocks(repo_id, commit_hash, body.get("selected_blocks"))
    compiled_messages, telemetry = _compile_adaptive_router_messages(
        messages,
        blocks,
        repo_id=repo_id,
        commit_hash=commit_hash,
        task_family=body.get("task_family"),
        required_paths=body.get("required_paths") or [],
        max_chars_per_doc=body.get("max_chars_per_doc", 8_000),
    )
    return {
        "repo_id": repo_id,
        "commit_hash": commit_hash,
        "messages": compiled_messages,
        "adaptive_router": telemetry,
        "usage_estimate": local_usage(compiled_messages),
        "payload_change": estimate_payload_change(messages, compiled_messages),
    }


def _seed_demo_context() -> None:
    STORE.upsert_blocks(
        "demo",
        "abc123",
        [
            {
                "path": "cache_proxy/server.py",
                "content": (
                    "async def chat_completions(request):\n"
                    "    provider = _provider_config()\n"
                    "    cache_context = body.pop('cache_context', None) or {}\n"
                    "    if repo_id and commit_hash:\n"
                    "        compiled_messages = _compile_adaptive_router_messages(...)\n"
                    "    return await _forward_json(body, provider=provider)\n"
                ),
            },
            {
                "path": "cache_proxy/context_planner.py",
                "content": (
                    "def plan_context_blocks(messages, blocks, profile=None):\n"
                    "    ranked = score_blocks(messages, blocks)\n"
                    "    return PlannedContext(map_blocks=blocks, full_blocks=ranked[:max_full_files])\n"
                ),
            },
            {
                "path": "tests/test_cache_proxy.py",
                "content": (
                    "def test_adaptive_router_plan_endpoint_selects_inferred_context(client):\n"
                    "    response = client.post('/v1/adaptive-router/plan', json={...})\n"
                    "    assert response.status_code == 200\n"
                ),
            },
            {
                "path": "README.md",
                "content": (
                    "# Adaptive-Router Context Proxy\n\n"
                    "A provider-adjustable proxy that compiles repository context before forwarding "
                    "chat completions to an OpenAI-compatible model API.\n"
                ),
            },
        ],
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _require_public_token(request)
    provider = _provider_config()
    if not provider["api_key"]:
        raise HTTPException(
            status_code=500,
            detail=f"{provider['api_key_env']} is not set",
        )

    body = await request.json()
    stream = bool(body.get("stream"))
    cache_context = body.pop("cache_context", None) or {}
    repo_id = cache_context.get("repo_id")
    commit_hash = cache_context.get("commit_hash")
    selected_blocks = cache_context.get("selected_blocks")
    context_budget = cache_context.get("context_budget") or {}

    original_messages = body.get("messages", [])
    if repo_id and commit_hash:
        blocks = STORE.get_blocks(repo_id, commit_hash, selected_blocks)
        enable_context_planning = _as_bool(context_budget.get("enabled", True))
        context_planning_telemetry: dict[str, Any] = {"enabled": enable_context_planning}
        if enable_context_planning:
            if context_budget.get("mode") == "adaptive-router":
                compiled_messages, context_planning_telemetry = (
                    _compile_adaptive_router_messages(
                        original_messages,
                        blocks,
                        repo_id=repo_id,
                        commit_hash=commit_hash,
                        task_family=context_budget.get("task_family"),
                        required_paths=context_budget.get("required_paths") or [],
                        max_chars_per_doc=context_budget.get("max_chars_per_doc", 8_000),
                    )
                )
            else:
                planned = plan_context_blocks(
                    original_messages,
                    blocks,
                    max_repo_chars=(
                        int(context_budget["max_repo_chars"])
                        if "max_repo_chars" in context_budget
                        else None
                    ),
                    max_full_files=(
                        int(context_budget["max_full_files"])
                        if "max_full_files" in context_budget
                        else None
                    ),
                    include_repo_map=(
                        _as_bool(context_budget["include_repo_map"])
                        if "include_repo_map" in context_budget
                        else None
                    ),
                    profile=context_budget.get("profile"),
                )
                compiled_messages = compile_planned_messages(
                    original_messages,
                    planned.map_blocks,
                    planned.full_blocks,
                    repo_id=repo_id,
                    commit_hash=commit_hash,
                )
                context_planning_telemetry = planned.telemetry
        else:
            compiled_messages = compile_messages(
                original_messages,
                blocks,
                repo_id=repo_id,
                commit_hash=commit_hash,
            )
        body["messages"] = compiled_messages
        body.setdefault("metadata", {})
        body["metadata"]["cache_proxy"] = estimate_payload_change(
            original_messages,
            compiled_messages,
        )
        body["metadata"]["cache_proxy"]["context_planning"] = context_planning_telemetry

    if stream:
        return await _forward_stream(body, provider)
    return await _forward_json(
        body,
        repo_id=repo_id,
        commit_hash=commit_hash,
        provider=provider,
    )


async def _forward_json(
    body: dict,
    repo_id: str | None,
    commit_hash: str | None,
    provider: dict,
) -> JSONResponse:
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            _chat_completions_url(provider),
            headers=_provider_headers(provider),
            json=body,
        )
    latency_ms = (time.perf_counter() - start) * 1000.0
    try:
        payload: Any = response.json()
    except ValueError:
        return JSONResponse(
            status_code=response.status_code,
            content={"error": response.text, "latency_ms": latency_ms},
        )

    if response.status_code < 400:
        usage = payload.get("usage", {})
        cache_tokens = _extract_cache_tokens(usage)
        STORE.log_telemetry(
            {
                "model": payload.get("model") or body.get("model"),
                "repo_id": repo_id,
                "commit_hash": commit_hash,
                "latency_ms": latency_ms,
                "prompt_tokens": usage.get("prompt_tokens"),
                "cache_hit_tokens": cache_tokens["hit"],
                "cache_miss_tokens": cache_tokens["miss"],
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )
        payload.setdefault("cache_proxy", {})
        payload["cache_proxy"]["latency_ms"] = latency_ms
        payload["cache_proxy"]["provider"] = provider["provider"]
        payload["cache_proxy"]["cache_hit_tokens"] = cache_tokens["hit"]
        payload["cache_proxy"]["cache_miss_tokens"] = cache_tokens["miss"]
        payload["cache_proxy"]["cache_hit_rate"] = _cache_hit_rate(cache_tokens)
    return JSONResponse(status_code=response.status_code, content=payload)


async def _forward_stream(body: dict, provider: dict) -> StreamingResponse:
    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                _chat_completions_url(provider),
                headers=_provider_headers(provider),
                json=body,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _provider_config() -> dict:
    defaults = {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "openai": {
            "base_url": "https://api.openai.com",
            "api_key_env": "OPENAI_API_KEY",
        },
        "openai-compatible": {
            "base_url": "https://api.openai.com",
            "api_key_env": "CACHE_PROXY_API_KEY",
        },
    }
    provider = CACHE_PROXY_PROVIDER or "deepseek"
    if provider not in defaults:
        provider = "openai-compatible"
    api_key_env = CACHE_PROXY_API_KEY_ENV or defaults[provider]["api_key_env"]
    return {
        "provider": provider,
        "base_url": (CACHE_PROXY_BASE_URL or defaults[provider]["base_url"]).rstrip("/"),
        "api_key_env": api_key_env,
        "api_key": CACHE_PROXY_API_KEY or os.getenv(api_key_env),
    }


def _chat_completions_url(provider: dict) -> str:
    return f"{provider['base_url']}/v1/chat/completions"


def _provider_headers(provider: dict) -> dict:
    return {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }


def _extract_cache_tokens(usage: dict) -> dict:
    prompt_tokens = usage.get("prompt_tokens") or 0
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is None:
        details = usage.get("prompt_tokens_details") or {}
        hit = details.get("cached_tokens")
    if hit is None:
        hit = 0
    if miss is None:
        miss = max(0, prompt_tokens - hit) if prompt_tokens is not None else None
    return {"hit": hit, "miss": miss}


def _require_public_token(request: Request | None) -> None:
    if not CACHE_PROXY_PUBLIC_TOKEN:
        return
    if request is None:
        raise HTTPException(status_code=401, detail="Authorization required")
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {CACHE_PROXY_PUBLIC_TOKEN}"
    if auth != expected:
        raise HTTPException(status_code=401, detail="Authorization required")


def _compile_adaptive_router_messages(
    messages: list[dict],
    blocks: list,
    repo_id: str,
    commit_hash: str,
    task_family: str | None,
    required_paths: list[str],
    max_chars_per_doc: int | None,
) -> tuple[list[dict], dict]:
    task_text = _messages_text(messages)
    available_paths = {block.path for block in blocks}
    clean_required_paths = [
        path for path in required_paths if isinstance(path, str) and path in available_paths
    ]
    if clean_required_paths:
        train_from_feedback(
            STORE,
            repo_id=repo_id,
            commit_hash=commit_hash,
            task_text=task_text,
            blocks=blocks,
            positive_paths=clean_required_paths,
            negative_paths=[],
            learning_rate=0.08,
        )
    policy = route_policy_for_task(
        WorkloadTask(
            task=task_text,
            family=task_family or _infer_task_family(task_text),
            required_paths=tuple(clean_required_paths),
        )
    )
    inferred_paths = list(infer_context_guard_paths(task_text, blocks))
    ml_selection = select_context_with_ml(
        STORE,
        repo_id=repo_id,
        task_text=task_text,
        blocks=blocks,
        required_paths=_dedupe_paths([*clean_required_paths, *inferred_paths]),
        max_blocks=max(1, min(4, len(blocks))),
    )
    selected_paths = _dedupe_paths(
        [block.path for block in ml_selection.selected_blocks]
        or [*clean_required_paths, *inferred_paths]
    )

    if not selected_paths and policy.confidence == "low":
        planned = plan_context_blocks(messages, blocks, profile="general")
        selected_blocks = planned.full_blocks
        strategy = "fallback-general"
        planning_telemetry = planned.telemetry
    else:
        block_by_path = {block.path: block for block in blocks}
        selected_blocks = [
            block_by_path[path]
            for path in selected_paths
            if path in block_by_path
        ]
        strategy = "ml-ranked" if ml_selection.telemetry["used"] else "focused-inferred"
        planning_telemetry = {
            "enabled": True,
            "policy_profile": "adaptive-ml-router",
            "full_block_paths": [block.path for block in selected_blocks],
            "repo_map_blocks": len(blocks),
            "full_blocks": len(selected_blocks),
        }

    selected_blocks = slice_context_blocks(
        task_text,
        selected_blocks,
        max_chars_per_doc=(
            int(max_chars_per_doc)
            if max_chars_per_doc is not None
            else None
        ),
    )
    compiled = compile_planned_messages(
        messages,
        blocks,
        selected_blocks,
        repo_id=repo_id,
        commit_hash=commit_hash,
    )
    telemetry = {
        **planning_telemetry,
        "router": {
            "bucket": policy.bucket,
            "confidence": policy.confidence,
            "reason": policy.reason,
            "strategy": strategy,
            "required_paths": clean_required_paths,
            "inferred_guard_paths": inferred_paths,
            "selected_paths": [block.path for block in selected_blocks],
            "adaptive_ml": ml_selection.telemetry,
            "usage_estimate": local_usage(compiled),
        },
    }
    return compiled, telemetry


def _messages_text(messages: list[dict]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
    )


def _infer_task_family(task_text: str) -> str:
    lowered = task_text.lower()
    if any(term in lowered for term in ("pytest", "traceback", "failing", "debug")):
        return "test_debug"
    if any(term in lowered for term in ("refactor", "adapter", "split", "interface")):
        return "refactor"
    if any(term in lowered for term in ("docs", "onboarding", "explain")):
        return "docs"
    if any(term in lowered for term in ("handoff", "continue", "prior agent")):
        return "handoff"
    if any(term in lowered for term in ("review", "audit", "risk")):
        return "code_review"
    return "unknown"


def _dedupe_paths(paths: list[str]) -> list[str]:
    deduped = []
    for path in paths:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _cache_hit_rate(cache_tokens: dict) -> float | None:
    hit = cache_tokens.get("hit")
    miss = cache_tokens.get("miss")
    if hit is None or miss is None or hit + miss == 0:
        return None
    return hit / (hit + miss)


def main() -> None:
    import uvicorn

    host = os.getenv("CACHE_PROXY_HOST", "127.0.0.1")
    port = int(os.getenv("CACHE_PROXY_PORT", "8000"))
    uvicorn.run("cache_proxy.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
