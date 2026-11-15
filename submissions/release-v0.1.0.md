# amu-mem0 v0.1.0

First release. `GatedMemory` wraps an existing mem0 `Memory` so that reads return a memory only if the requester is entitled to every sensitive label in its lineage, including the sources of the memories it was derived from. The allow/deny check is `amu-governance`'s `S(a) ⊆ P(d)` (Sangaraju & Vissa, IEEE Access 2026, doi:10.1109/ACCESS.2026.3730363).

## What's included
- `add(..., sources=..., derived_from=...)` records lineage and closes it at write time (multi-hop, mixed sources).
- `search`, `get` and `get_all` require `entitlements=` and report removed candidates in `amu_denied`, with no memory text in logs.
- Fail-closed by default on missing, incomplete or malformed lineage. `add(infer=True)` is rejected by default.
- Over-fetch so filtering does not starve `top_k`.
- 23 tests against a real offline mem0 `Memory`. CI on Python 3.10–3.13.
- `make bench`: a deterministic synthetic benchmark.

## Benchmark (synthetic multi-team setup, deterministic embedder, 420 memories)
- Filtering on each memory's own labels leaked 563 of 2,400 returned items, all of them derived memories.
- The gate returned no restricted items with complete lineage (recall@10 0.999 for entitled users).
- With 90% of derivation edges recorded, 11.7% of queries still returned a restricted item.

See `results/BENCH.md` for definitions and caveats. These are not production estimates.

## Limitations
Zero leakage depends on complete lineage. Inference through undeclared derived content is not addressed. `infer=True`, `AsyncMemory` and the hosted `MemoryClient` are unsupported. Tested only with mem0ai 2.2.1 and local Qdrant.
