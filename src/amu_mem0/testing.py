"""Offline helpers for tests and the benchmark. Not for production use.

``offline_memory()`` builds a real ``mem0.Memory`` (local Qdrant, SQLite
history) that never touches the network: embeddings come from a deterministic
hashed bag-of-words embedder, and the LLM is never called because every write
uses ``infer=False``. mem0 telemetry must be disabled via ``MEM0_TELEMETRY``
*before* mem0 is imported; this module sets it if unset.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Literal

os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory  # noqa: E402
from mem0.embeddings.base import EmbeddingBase  # noqa: E402

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedding(EmbeddingBase):
    """Deterministic hashed bag-of-words embedding (L2-normalized)."""

    def __init__(self, dims: int = 128) -> None:
        self.dims = dims

    def embed(self, text: str, memory_action: Literal["add", "search", "update"] | None = None) -> list[float]:
        vec = [0.0] * self.dims
        for tok in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            vec[h % self.dims] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def offline_memory(workdir: str, dims: int = 128, collection: str = "amu_mem0") -> Any:
    """A real mem0 ``Memory`` backed by local Qdrant under ``workdir``; no network."""
    # mem0 constructs an OpenAI client at init even though infer=False never calls it.
    os.environ.setdefault("OPENAI_API_KEY", "sk-offline-unused")
    memory = Memory.from_config(
        {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": os.path.join(workdir, "qdrant"),
                    "collection_name": collection,
                    "embedding_model_dims": dims,
                },
            },
            "embedder": {"provider": "openai", "config": {"embedding_dims": dims}},
            "history_db_path": os.path.join(workdir, "history.db"),
        }
    )
    memory.embedding_model = HashEmbedding(dims)
    return memory
