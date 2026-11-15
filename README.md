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
pip install amu-mem0        # not yet published; for now: pip install -e .
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

`make bench` runs a synthetic, deterministic benchmark (fixed seed, offline
embedder, local Qdrant, no LLM). Two teams share one mem0 scope containing
420 memories: 180 raw records and 240 agent-written summaries, aggregates and
2- and 3-hop summaries. The benchmark measures 240 queries at `top_k=10`.
Full results, versions and the commit hash are in
[`results/BENCH.md`](results/BENCH.md).

Run at commit `99dd6d5`, 2026-11-15:

| Condition | Leak rate (returned items) | Queries with a leak | Recall@10 for entitled users | p50 / p95 search ms |
|---|---|---|---|---|
| Plain mem0 | 35.7% (857 / 2400) | 71.2% | 0.650 | 2.05 / 2.11 |
| Content-only filter (each memory's own labels) | 23.5% (563 / 2400) | 62.9% | 0.816 | 2.92 / 3.00 |
| amu-mem0 gate | 0.0% (0 / 2164) | 0.0% | 0.999 | 3.06 / 12.63 |

How to read it:
- **Content-only removes raw restricted records but not what was derived
  from them.** All 563 of its leaks are derived memories.
- **The gate returns fewer than 10 results only when fewer than 10 entitled
  memories pass mem0's own score threshold** for that query (2164 of 2400 slots).
  Recall is measured against an oracle: mem0's ranking filtered by the true
  entitlements.
- **The p95 cost comes from over-fetching** when most top candidates are
  restricted, e.g. Marketing searching Finance topics.
- **Comparison with the paper.** The content-only leak rate lands inside the
  paper's 18.8–25.5% range for naive memory. That is not a replication: the
  workload, store and leak definition all differ.

The lineage-coverage sweep differs from the paper's degradation result. Here,
leakage is not eliminated at 90% coverage (1.4% of returned items leak). The
paper measured 0% at 75–90% completeness. The setups differ: the paper drops
columns within each step of a flat cached result, while this benchmark drops
whole `derived_from` edges in a multi-hop graph. A missing edge hides every label
beneath it from all descendants. The paper notes that join depth raises the
completeness requirement, and multi-hop agent memory is that case.

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
  In the benchmark, recording 90% of derivation edges still leaked 1.4% of
  returned items. The paper recommends at least 90% lineage completeness, and
  notes that deeper derivations raise the requirement. Multi-hop agent memory is
  exactly that case. See the coverage sweep in `results/BENCH.md`.
- **Derived-feature inference remains open.** A memory whose text paraphrases
  restricted data without a declared source is not caught.
- **`infer=True` is unsupported** for gated data (see above).
- **Only the sync OSS `Memory` is supported.** `AsyncMemory` and the hosted
  `MemoryClient` are not wrapped or tested.
- **`update()` is not gated or re-closed.** A child's closure is computed when
  the child is written. Changing a parent's lineage later does not update its
  children. Deleting a parent leaves children with their recorded closure.
- **Search cost grows with filtering.** When most top candidates are denied,
  the gate issues several underlying searches (see p95 in the benchmark).
- Tested against mem0ai 2.2.1 with local Qdrant. Other vector stores are
  expected to work because only metadata round-tripping is required, but they
  are untested.

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
