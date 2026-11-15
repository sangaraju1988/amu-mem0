# amu-mem0

Derivation-gated reads for [mem0](https://github.com/mem0ai/mem0). A memory is
returned only if the requester is entitled to every sensitive label anywhere in
its lineage, including the sources of the memories it was derived from.

mem0 scopes memories by identity (`user_id`, `agent_id`, `run_id`). That
answers *whose* memory it is, not *what it was derived from*. When several
teams share one agent's memory, a summary written from restricted records looks
like any other memory and is returned to anyone who can search that scope.
`amu-mem0` records where each memory came from and checks it on every read.

The check is the rule from Sangaraju & Vissa, *Lineage-Aware Memory Governance*,
IEEE Access 14 (2026), [doi:10.1109/ACCESS.2026.3730363](https://doi.org/10.1109/ACCESS.2026.3730363):
serve a stored result only if `S(a) ⊆ P(d)`, where `S(a)` is the set of sensitive labels in
its derivation and `P(d)` is the set of labels the requester holds. The decision is made by
[`amu-governance`](https://github.com/sangaraju1988/amu-governance), not reimplemented here.

## Install

```bash
pip install amu-mem0
```

Requires Python 3.10+, `mem0ai>=2.2,<3`, `amu-governance>=0.1.2,<0.2`. No other
dependencies.

## Quickstart

```python
from mem0 import Memory
from amu_mem0 import GatedMemory

memory = GatedMemory(Memory())  # wrap your existing instance

payroll = memory.add(
    "Q3 payroll total is 4.2M",
    agent_id="analyst",
    sources={"hris": ["hr.salary"]},  # raw source -> its labels
)["results"][0]["id"]
memory.add("Payroll cost grew 8% QoQ", agent_id="analyst", derived_from=[payroll])

memory.search("payroll", filters={"agent_id": "analyst"}, entitlements={"mkt.campaign"})
# {'results': [], 'amu_denied': [{'id': ..., 'reason': 'unentitled', 'unentitled_labels': ['hr.salary']}, ...]}

memory.search("payroll", filters={"agent_id": "analyst"}, entitlements={"hr.salary"})
# both memories
```

`examples/quickstart.py` is the same code running fully offline. It prints:

```
marketing sees: []
marketing denied: [('unentitled', ['hr.salary']), ('unentitled', ['hr.salary'])]
finance sees: ['Payroll cost grew 8% QoQ', 'Q3 payroll total is 4.2M']
```

## How it works

- **Writes.** `add(..., sources=..., derived_from=...)` stores lineage under the
  reserved metadata key `amu`. Here `sources` maps each raw source to its labels,
  and `derived_from` lists the ids of mem0 memories the new one was written from.
  The transitive closure (`closed`) is computed once, at write time, from the
  parents' stored closures. A summary of a summary of a summary therefore carries
  every label beneath it, and a memory with mixed sources carries the union.
- **Reads.** `search`, `get` and `get_all` require `entitlements=` and keep a
  result only if every sensitive label in its closed lineage is in that set.
  `search` and `get_all` return mem0's normal shape plus `amu_denied`, which gives
  the id, a reason code and the unentitled labels for every removed candidate.
  Denials are also logged on the `amu_mem0` logger, with ids and label names only,
  never memory text.
- **Labels.** By default every label gates a read. Pass
  `GatedMemory(memory, sensitive_labels={...})` to gate only on a registry
  (the paper's semantics). Labels outside the registry are then ignored.
- **Entitlement changes** take effect on the next read. Verdicts are never
  stored; only labels are.
- **Recall.** Filtering after retrieval would starve `top_k`, so the gate
  over-fetches. It doubles the fetch until it has `top_k` allowed results, the
  fetch reaches `max_fetch` (default 1000), or a larger fetch adds no new
  candidates.

### Fail-closed defaults

| Situation | Default | Reason code |
|---|---|---|
| Memory has no `amu` metadata (e.g. written with plain mem0) | denied | `missing_lineage` |
| `add()` with neither `sources` nor `derived_from` | stored, then denied | `incomplete_lineage` |
| A `derived_from` id is missing, or the parent has no or incomplete lineage | stored, then denied | `incomplete_lineage` |
| `amu` metadata is malformed | denied | `malformed_lineage` |
| `add(infer=True)` | raises `InferNotAllowedError` | n/a |

`on_missing_lineage="allow"` or `"allow_with_warning"` relaxes the first four
rows. A label already known to be forbidden still denies. `sources={}`
explicitly declares "no restricted inputs".

`infer=True` is rejected because mem0's extraction step shows the LLM up to 10
existing memories from the same scope. The facts it writes can draw on any of
them, and the caller cannot declare which. With `GatedMemory(allow_infer=True)`
such memories are stored but marked incomplete, so they stay hidden under the
default policy.

## Benchmark

> **Synthetic multi-team setup, deterministic embedder, 420 memories.** These
> results describe this benchmark only. They are not estimates of leak rates in
> production.

`make bench` runs a deterministic benchmark: fixed seed, offline hashed
bag-of-words embedder, local Qdrant, no LLM. Two teams share one mem0 scope
containing 420 memories: 180 raw records and 240 agent-written summaries,
aggregates, and 2- and 3-hop summaries. It runs 240 queries at `top_k=10`.
Full results, definitions, versions and the commit hash are in
[`results/BENCH.md`](results/BENCH.md).

Run at commit `8783e3b`, 2026-11-15 (synthetic multi-team setup, deterministic
embedder, 420 memories):

| Condition | Leaked / returned | Leak rate | Queries with a leak | Recall@10 for entitled users | p50 / p95 search ms |
|---|---|---|---|---|---|
| Plain mem0 | 857 / 2400 | 35.7% | 71.2% | 0.650 | 2.05 / 2.18 |
| Content-only filter (each memory's own labels) | 563 / 2400 | 23.5% | 62.9% | 0.816 | 2.91 / 3.03 |
| amu-mem0 gate | 0 / 2164 | 0.0% | 0.0% | 0.999 | 3.07 / 12.57 |

- **Content-only filtering misses derived memories.** It removes raw restricted
  records, but none of what was derived from them: all 563 of its leaks are
  derived memories.
- **Recall@10, exactly.** For each query, mem0 is searched with `top_k=500` and
  the results are kept, in mem0's order, only if the requester is truly entitled
  to them. The first min(10, n) of those form the ideal set. A returned memory is
  a hit if it is in that entitled list with a score, in the same `top_k=500`
  call, at least that of the 10th ideal item, so ties at rank 10 are not
  penalised. Recall@10 = min(hits, |ideal|) / |ideal|, averaged over all 240
  queries (every query has at least one entitled memory).
- **Plain mem0's lower recall is displacement.** It always returns 10 results,
  and restricted memories take slots entitled ones would otherwise fill. In 170
  of the 171 queries where it misses entitled memories, every miss is matched by
  a restricted memory in its top 10. The remainder reflects mem0's ranking
  depending on `top_k`.
- **The denominators differ: 2,164 vs 2,400.** The gate returns fewer than 10
  results in 34 queries (236 empty slots). In all 34 it returned every entitled
  memory mem0 surfaces for that query at any depth, because fewer than 10
  entitled memories pass mem0's own score threshold. Leak rates are per returned
  item, so each uses its own denominator.
- **Latency is tail overhead from over-fetch, at small scale.** When most top
  candidates are restricted, the gate issues several underlying searches. That
  shows up in p95, not p50. These are in-process timings on 420 memories with
  local Qdrant. Latency was not measured at production scale, with a remote
  vector store, or with real embeddings.
- **Comparison with the paper.** The content-only leak rate falls inside the
  paper's 18.8–25.5% range for naive memory. That is not a replication: the
  workload, store and leak definition all differ.

**Lineage coverage sweep (gate only; synthetic multi-team setup, deterministic
embedder, 420 memories).** Each `derived_from` edge is recorded with
probability c, and dropped edges are silent.

| Derivation edges recorded | Queries with a leak | Leak rate | Recall@10 |
|---|---|---|---|
| 100% | 0.0% | 0.0% | 0.999 |
| 90.2% | 11.7% | 1.4% | 0.995 |
| 75.9% | 22.9% | 4.8% | 0.983 |
| 49.9% | 35.4% | 11.2% | 0.949 |

The paper recommends at least 90% lineage completeness, based on 0% measured
leakage at 75–90% on its workloads. This workload is different: it drops whole
derivation edges in a multi-hop graph, where the paper dropped columns within a
flat result. Here some leakage remains at every coverage level below 100%.

## Threat model

**In scope.** A requester using the same mem0 scope as other teams receives a
memory derived, directly or through other memories, from a source whose label
they do not hold. Callers are trusted to pass the requester's real
entitlements and to declare lineage truthfully.

**Out of scope.**
- Calls made on the wrapped `Memory` directly (`gated.memory.search(...)`,
  `history()`, `update()`). These bypass the gate.
- Anyone who can write mem0 metadata or the vector store directly. They can
  forge or strip `amu` lineage. The gate is an application-level filter, not
  storage-level enforcement. (For storage-level enforcement on Postgres, see
  [amu-pgvector](https://github.com/sangaraju1988/amu-pgvector).)
- Content that encodes restricted information without naming its source
  (inference through undeclared derived features, threat T4 in the paper), and
  aggregation across many permitted reads (T6).
- mem0's SQLite history and message log, entity store and graph memory.

## Limitations

- **Zero leakage depends on lineage coverage.** The guarantee holds only if
  every derivation is declared. An undeclared edge is invisible to the gate.
  In the synthetic benchmark, with 90% of derivation edges recorded, 11.7% of
  queries still returned at least one leaked memory. The paper recommends at
  least 90% lineage completeness, and notes that deeper derivations raise the
  requirement. Multi-hop agent memory is exactly that case. See the coverage
  sweep in `results/BENCH.md`.
- **Derived-feature inference remains open.** A memory whose text paraphrases
  restricted data without a declared source is not caught.
- **`infer=True` is unsupported** for gated data (see above).
- **Only the sync OSS `Memory` is supported.** `AsyncMemory` and the hosted
  `MemoryClient` are not wrapped or tested.
- **`update()` is not gated or re-closed.** A child's closure is computed when
  the child is written. Changing a parent's lineage later does not update its
  children. Deleting a parent leaves children with their recorded closure.
- **Search cost grows with filtering.** When most top candidates are denied,
  the gate issues several underlying searches. This was measured only as
  small-scale tail overhead (p95 in the benchmark), not at production scale.
- Tested against mem0ai 2.2.1 with local Qdrant. Other vector stores are
  expected to work because only metadata round-tripping is required, but they
  are untested.

## Upstream discussion

The behaviour this package addresses is described for mem0's maintainers in
[mem0ai/mem0#7927](https://github.com/mem0ai/mem0/issues/7927). The issue
proposes no change to mem0's core; it asks whether a docs link would be welcome.

## Development

```bash
make install   # uv sync
make test      # pytest, against a real offline mem0 Memory (nothing mocked)
make lint
make bench
```

## Citation

If you use this, please cite the paper:

> V. Sangaraju and S. Vissa, "Lineage-Aware Memory Governance: A Derivation-Gated
> Framework for Privacy-Preserving Column-Level Access Control in Enterprise AI
> Agents," *IEEE Access*, vol. 14, pp. 139683–139693, 2026,
> doi:10.1109/ACCESS.2026.3730363.

MIT licensed. Not affiliated with or endorsed by Mem0.
