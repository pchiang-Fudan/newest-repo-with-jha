from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ContextBlock:
    repo_id: str
    commit_hash: str
    path: str
    content: str
    content_hash: str


class ContextStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS context_blocks (
                  repo_id TEXT NOT NULL,
                  commit_hash TEXT NOT NULL,
                  path TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  content TEXT NOT NULL,
                  updated_at REAL NOT NULL,
                  PRIMARY KEY (repo_id, commit_hash, path)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  model TEXT,
                  repo_id TEXT,
                  commit_hash TEXT,
                  latency_ms REAL NOT NULL,
                  prompt_tokens INTEGER,
                  cache_hit_tokens INTEGER,
                  cache_miss_tokens INTEGER,
                  completion_tokens INTEGER,
                  total_tokens INTEGER
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_feature_weights (
                  repo_id TEXT NOT NULL,
                  feature TEXT NOT NULL,
                  weight REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  PRIMARY KEY (repo_id, feature)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_feedback (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  repo_id TEXT NOT NULL,
                  commit_hash TEXT,
                  task_text TEXT NOT NULL,
                  positive_paths TEXT NOT NULL,
                  negative_paths TEXT NOT NULL
                )
                """
            )

    def upsert_blocks(
        self,
        repo_id: str,
        commit_hash: str,
        blocks: Iterable[dict],
    ) -> list[ContextBlock]:
        saved: list[ContextBlock] = []
        now = time.time()
        with self._connect() as con:
            for block in blocks:
                path = str(block["path"])
                content = str(block["content"])
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                con.execute(
                    """
                    INSERT INTO context_blocks
                      (repo_id, commit_hash, path, content_hash, content, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(repo_id, commit_hash, path) DO UPDATE SET
                      content_hash=excluded.content_hash,
                      content=excluded.content,
                      updated_at=excluded.updated_at
                    """,
                    (repo_id, commit_hash, path, content_hash, content, now),
                )
                saved.append(
                    ContextBlock(
                        repo_id=repo_id,
                        commit_hash=commit_hash,
                        path=path,
                        content=content,
                        content_hash=content_hash,
                    )
                )
        return saved

    def get_blocks(
        self,
        repo_id: str,
        commit_hash: str,
        selected_paths: Iterable[str] | None = None,
    ) -> list[ContextBlock]:
        params: list[str] = [repo_id, commit_hash]
        where = "repo_id = ? AND commit_hash = ?"
        if selected_paths:
            paths = sorted(set(selected_paths))
            placeholders = ",".join("?" for _ in paths)
            where += f" AND path IN ({placeholders})"
            params.extend(paths)
        query = f"""
            SELECT repo_id, commit_hash, path, content_hash, content
            FROM context_blocks
            WHERE {where}
            ORDER BY path ASC
        """
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [
            ContextBlock(
                repo_id=row[0],
                commit_hash=row[1],
                path=row[2],
                content_hash=row[3],
                content=row[4],
            )
            for row in rows
        ]

    def log_telemetry(self, row: dict) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO telemetry (
                  created_at, model, repo_id, commit_hash, latency_ms,
                  prompt_tokens, cache_hit_tokens, cache_miss_tokens,
                  completion_tokens, total_tokens
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    row.get("model"),
                    row.get("repo_id"),
                    row.get("commit_hash"),
                    row.get("latency_ms"),
                    row.get("prompt_tokens"),
                    row.get("cache_hit_tokens"),
                    row.get("cache_miss_tokens"),
                    row.get("completion_tokens"),
                    row.get("total_tokens"),
                ),
            )

    def telemetry_summary(self, limit: int = 100) -> dict:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT latency_ms, prompt_tokens, cache_hit_tokens,
                       cache_miss_tokens, completion_tokens, total_tokens
                FROM telemetry
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        if not rows:
            return {"requests": 0}
        sums = [sum(row[i] or 0 for row in rows) for i in range(6)]
        hit = sums[2]
        miss = sums[3]
        return {
            "requests": len(rows),
            "avg_latency_ms": sums[0] / len(rows),
            "prompt_tokens": sums[1],
            "cache_hit_tokens": hit,
            "cache_miss_tokens": miss,
            "cache_hit_rate": hit / (hit + miss) if hit + miss else None,
            "completion_tokens": sums[4],
            "total_tokens": sums[5],
        }

    def get_ml_weights(self, repo_id: str) -> dict[str, float]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT feature, weight
                FROM ml_feature_weights
                WHERE repo_id IN (?, ?)
                ORDER BY CASE WHEN repo_id = ? THEN 1 ELSE 0 END
                """,
                ("*", repo_id, repo_id),
            ).fetchall()
        weights: dict[str, float] = {}
        for feature, weight in rows:
            weights[str(feature)] = float(weight)
        return weights

    def update_ml_weights(self, repo_id: str, updates: dict[str, float]) -> None:
        now = time.time()
        with self._connect() as con:
            for feature, delta in updates.items():
                con.execute(
                    """
                    INSERT INTO ml_feature_weights (repo_id, feature, weight, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(repo_id, feature) DO UPDATE SET
                      weight=ml_feature_weights.weight + excluded.weight,
                      updated_at=excluded.updated_at
                    """,
                    (repo_id, feature, float(delta), now),
                )

    def log_ml_feedback(
        self,
        repo_id: str,
        commit_hash: str | None,
        task_text: str,
        positive_paths: list[str],
        negative_paths: list[str],
    ) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO ml_feedback (
                  created_at, repo_id, commit_hash, task_text,
                  positive_paths, negative_paths
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    repo_id,
                    commit_hash,
                    task_text,
                    "\n".join(positive_paths),
                    "\n".join(negative_paths),
                ),
            )

    def ml_feedback_count(self, repo_id: str) -> int:
        with self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM ml_feedback WHERE repo_id = ?",
                (repo_id,),
            ).fetchone()
        return int(row[0] if row else 0)
