from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_repo_context_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_PROXY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    import cache_proxy.server as server

    importlib.reload(server)
    client = TestClient(server.app)

    response = client.post(
        "/v1/repos/demo/context",
        json={
            "commit_hash": "abc123",
            "blocks": [
                {"path": "src/b.py", "content": "B = 2\n"},
                {"path": "src/a.py", "content": "A = 1\n"},
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["blocks"] == 2

    blocks = server.STORE.get_blocks("demo", "abc123", ["src/a.py", "src/b.py"])
    assert [block.path for block in blocks] == ["src/a.py", "src/b.py"]


def test_prompt_compiler_places_repo_context_before_user(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_PROXY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    import cache_proxy.server as server

    importlib.reload(server)
    server.STORE.upsert_blocks(
        "demo",
        "abc123",
        [
            {"path": "README.md", "content": "# Demo\n"},
            {"path": "src/app.py", "content": "def main():\n    pass\n"},
        ],
    )
    blocks = server.STORE.get_blocks("demo", "abc123", ["README.md", "src/app.py"])

    from cache_proxy.prompt_compiler import compile_messages

    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Fix the app."},
    ]
    compiled = compile_messages(messages, blocks, "demo", "abc123")

    assert compiled[0]["role"] == "system"
    assert compiled[0]["content"] == "You are a coding assistant."
    assert compiled[1]["role"] == "system"
    assert "README.md" in compiled[1]["content"]
    assert "src/app.py" in compiled[1]["content"]
    assert compiled[2]["role"] == "user"
    assert compiled[2]["content"] == "Fix the app."


def test_context_planner_selects_relevant_file_body(tmp_path):
    from cache_proxy.context_planner import plan_context_blocks
    from cache_proxy.context_store import ContextStore

    store = ContextStore(tmp_path / "cache.sqlite3")
    blocks = store.upsert_blocks(
        "demo",
        "abc123",
        [
            {"path": "cache_proxy/server.py", "content": "def chat_completions():\n    pass\n"},
            {"path": "README.md", "content": "# Demo\n"},
            {"path": "tests/test_cache_proxy.py", "content": "def test_repo_context_roundtrip():\n    pass\n"},
        ],
    )
    messages = [{"role": "user", "content": "Fix chat_completions in cache_proxy/server.py"}]

    planned = plan_context_blocks(messages, blocks, max_full_files=1)

    assert [block.path for block in planned.full_blocks] == ["cache_proxy/server.py"]
    assert planned.telemetry["repo_map_blocks"] == 3
    assert planned.telemetry["full_blocks"] == 1


def test_planned_prompt_has_stable_map_before_file_bodies(tmp_path):
    from cache_proxy.context_planner import plan_context_blocks
    from cache_proxy.context_store import ContextStore
    from cache_proxy.prompt_compiler import compile_planned_messages

    store = ContextStore(tmp_path / "cache.sqlite3")
    blocks = store.upsert_blocks(
        "demo",
        "abc123",
        [
            {"path": "src/app.py", "content": "def main():\n    pass\n"},
            {"path": "README.md", "content": "# Demo\n"},
        ],
    )
    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Inspect src/app.py"},
    ]
    planned = plan_context_blocks(messages, blocks, max_full_files=1)

    compiled = compile_planned_messages(
        messages,
        planned.map_blocks,
        planned.full_blocks,
        "demo",
        "abc123",
    )

    assert compiled[1]["content"].startswith("Repository map follows.")
    assert compiled[2]["content"].startswith("Repository context follows.")
    assert "--- file: src/app.py" in compiled[2]["content"]
    assert compiled[-1]["content"] == "Inspect src/app.py"


def test_context_planner_classifier_uses_debug_policy_for_test_failures(tmp_path):
    from cache_proxy.context_planner import plan_context_blocks
    from cache_proxy.context_store import ContextStore

    store = ContextStore(tmp_path / "cache.sqlite3")
    blocks = store.upsert_blocks(
        "demo",
        "abc123",
        [
            {"path": "src/app.py", "content": "def main():\n    pass\n"},
            {"path": "tests/test_app.py", "content": "def test_main():\n    assert main()\n"},
            {"path": "README.md", "content": "# Demo\n"},
        ],
    )
    messages = [
        {
            "role": "user",
            "content": "Pytest traceback: failing test_main in CI. Diagnose the failure.",
        }
    ]

    planned = plan_context_blocks(messages, blocks)

    assert planned.telemetry["policy_profile"] == "debug"
    assert planned.telemetry["max_full_files"] == 4
    assert planned.full_blocks[0].path == "tests/test_app.py"


def test_context_planner_classifier_uses_docs_policy_for_onboarding(tmp_path):
    from cache_proxy.context_planner import plan_context_blocks
    from cache_proxy.context_store import ContextStore

    store = ContextStore(tmp_path / "cache.sqlite3")
    blocks = store.upsert_blocks(
        "demo",
        "abc123",
        [
            {"path": "src/app.py", "content": "def main():\n    pass\n"},
            {"path": "README.md", "content": "# Demo\n"},
            {"path": "docs/architecture.md", "content": "# Architecture\n"},
        ],
    )
    messages = [{"role": "user", "content": "Write onboarding docs for this repo."}]

    planned = plan_context_blocks(messages, blocks)

    assert planned.telemetry["policy_profile"] == "docs"
    assert planned.telemetry["max_full_files"] == 2
    assert planned.full_blocks[0].path in {"README.md", "docs/architecture.md"}


def test_adaptive_router_plan_endpoint_selects_inferred_context(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_PROXY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    import cache_proxy.server as server

    importlib.reload(server)
    client = TestClient(server.app)
    client.post(
        "/v1/repos/demo/context",
        json={
            "commit_hash": "abc123",
            "blocks": [
                {
                    "path": "cache_proxy/server.py",
                    "content": "async def chat_completions():\n    return plan_context_blocks()\n",
                },
                {
                    "path": "README.md",
                    "content": "# Demo\n",
                },
            ],
        },
    )

    response = client.post(
        "/v1/adaptive-router/plan",
        json={
            "repo_id": "demo",
            "commit_hash": "abc123",
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {
                    "role": "user",
                    "content": "Review chat_completions and identify one cache risk.",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    router = payload["adaptive_router"]["router"]
    assert router["bucket"] == "code-review"
    assert "cache_proxy/server.py" in router["selected_paths"]
    assert payload["usage_estimate"]["prompt_tokens"] > 0


def test_provider_config_supports_openai_cache_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_PROXY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CACHE_PROXY_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    import cache_proxy.server as server

    importlib.reload(server)

    config = server._provider_config()
    assert config["provider"] == "openai"
    assert config["base_url"] == "https://api.openai.com"
    assert config["api_key"] == "test-openai-key"
    assert server._chat_completions_url(config) == "https://api.openai.com/v1/chat/completions"

    cache_tokens = server._extract_cache_tokens(
        {
            "prompt_tokens": 2000,
            "prompt_tokens_details": {"cached_tokens": 1500},
        }
    )
    assert cache_tokens == {"hit": 1500, "miss": 500}
    assert server._cache_hit_rate(cache_tokens) == 0.75


def test_adaptive_ml_feedback_improves_repo_specific_ranking(tmp_path):
    from cache_proxy.adaptive_ml import select_context_with_ml, train_from_feedback
    from cache_proxy.context_store import ContextStore

    store = ContextStore(tmp_path / "cache.sqlite3")
    blocks = store.upsert_blocks(
        "demo",
        "abc123",
        [
            {"path": "src/cache.py", "content": "def cache_plan():\n    pass\n"},
            {"path": "docs/cache.md", "content": "# Cache guide\n"},
        ],
    )
    task = "Explain cache behavior for the service."

    train_from_feedback(
        store,
        repo_id="demo",
        commit_hash="abc123",
        task_text=task,
        blocks=blocks,
        positive_paths=["docs/cache.md"],
        negative_paths=["src/cache.py"],
        learning_rate=0.8,
    )
    after = select_context_with_ml(store, "demo", task, blocks, required_paths=[], max_blocks=1)

    assert store.ml_feedback_count("demo") == 1
    assert after.selected_blocks[0].path == "docs/cache.md"
    assert after.ranked_blocks[0].score > after.ranked_blocks[1].score


def test_adaptive_router_plan_includes_ml_telemetry(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_PROXY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    import cache_proxy.server as server

    importlib.reload(server)
    client = TestClient(server.app)
    client.post(
        "/v1/repos/demo/context",
        json={
            "commit_hash": "abc123",
            "blocks": [
                {"path": "src/cache.py", "content": "def cache_plan():\n    pass\n"},
                {"path": "README.md", "content": "# Demo\n"},
            ],
        },
    )

    response = client.post(
        "/v1/adaptive-router/plan",
        json={
            "repo_id": "demo",
            "commit_hash": "abc123",
            "messages": [{"role": "user", "content": "Review src/cache.py"}],
            "required_paths": ["src/cache.py"],
        },
    )

    assert response.status_code == 200
    router = response.json()["adaptive_router"]["router"]
    assert router["adaptive_ml"]["model"] == "online-logistic-file-ranker"
    assert router["adaptive_ml"]["used"] is True
    assert "src/cache.py" in router["selected_paths"]
