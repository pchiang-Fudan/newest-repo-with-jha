from __future__ import annotations

import ast
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
TOKENIZER = ROOT / "BitNet/build/bin/llama-tokenize"


def repo_context() -> str:
    readme = "# Demo\n\nThis repository contains a small Python service.\n" * 100
    app = "def add(a, b):\n    return a + b\n\n" * 180
    tests = "def test_add():\n    assert add(1, 2) == 3\n\n" * 120
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


def naive_prompt(task: str, context: str) -> str:
    return "\n\n".join(
        [
            "System: You are a precise coding assistant.",
            f"User task: {task}",
            context,
            "Answer briefly.",
        ]
    )


def cache_aware_prompt(task: str, context: str) -> str:
    return "\n\n".join(
        [
            "System: You are a precise coding assistant.",
            context,
            f"User task: {task}",
            "Answer briefly.",
        ]
    )


def tokenize(text: str) -> list[int]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write(text)
        name = f.name
    try:
        result = subprocess.run(
            [
                str(TOKENIZER),
                "-m",
                str(MODEL),
                "--ids",
                "--no-bos",
                "--log-disable",
                "-f",
                name,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        Path(name).unlink(missing_ok=True)
    return list(ast.literal_eval(result.stdout.strip()))


def common_prefix_len(a: list[int], b: list[int]) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def summarize(label: str, tokenized: list[list[int]]) -> dict:
    pair_prefixes: list[int] = []
    for i in range(len(tokenized)):
        for j in range(i + 1, len(tokenized)):
            pair_prefixes.append(common_prefix_len(tokenized[i], tokenized[j]))
    lengths = [len(tokens) for tokens in tokenized]
    return {
        "label": label,
        "requests": len(tokenized),
        "avg_tokens_per_request": sum(lengths) / len(lengths),
        "min_shared_prefix_tokens": min(pair_prefixes),
        "avg_shared_prefix_tokens": sum(pair_prefixes) / len(pair_prefixes),
        "max_shared_prefix_tokens": max(pair_prefixes),
        "avg_shared_prefix_fraction": (
            sum(pair_prefixes) / len(pair_prefixes) / (sum(lengths) / len(lengths))
        ),
    }


def main() -> None:
    context = repo_context()
    naive = [tokenize(naive_prompt(task, context)) for task in TASKS]
    aware = [tokenize(cache_aware_prompt(task, context)) for task in TASKS]
    result = {
        "model": str(MODEL),
        "naive": summarize("naive", naive),
        "cache_aware": summarize("cache_aware", aware),
    }
    naive_avg = result["naive"]["avg_shared_prefix_tokens"]
    aware_avg = result["cache_aware"]["avg_shared_prefix_tokens"]
    result["shared_prefix_lift"] = aware_avg / naive_avg if naive_avg else None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

