# mem0 feature request (posted as mem0ai/mem0#7927, 2026-11-15T04:54:56Z)

Target: https://github.com/mem0ai/mem0/issues/new?template=feature_request.yml

**Title:** Derived memories don't carry their sources' restrictions when teams share one agent's memory

**Component (dropdown):** Python SDK

**AI Assistance (dropdown):** AI-assisted, but the need is mine

**Description (field body below the line):**

---

### Use Case

When several teams share one agent's memory, mem0's scoping tells you whose memory it is, not what it was derived from. A memory the agent writes *from* a restricted record keeps no trace of that record.

I measured this in a synthetic multi-team setup: deterministic embedder, 420 memories, 240 queries, reproducible with `make bench`. Filtering on each memory's own labels still returned 563 restricted items out of 2,400, and all 563 were derived memories. Tracking lineage closes that gap only as far as lineage is recorded:

| Derivation edges recorded | Queries returning a restricted item |
|---|---|
| 100% | 0.0% |
| 90% | 11.7% |
| 75% | 22.9% |
| 50% | 35.4% |

Plain mem0 returned a restricted item in 71.2% of queries. With complete lineage the lineage-aware filter returned none, which is the correct result when lineage is complete. Recall@10 for entitled users was 0.999. These numbers describe this synthetic setup only.

Minimal repro (mem0ai 2.2.1, local Qdrant):

```python
from mem0 import Memory

m = Memory()
scope = {"agent_id": "analyst"}  # one agent's memory, used by Finance and Marketing
m.add("Q3 payroll total is 4.2M", **scope, infer=False, metadata={"label": "hr.salary"})
m.add("Payroll cost grew 8% QoQ", **scope, infer=False)  # the agent's summary of the record above

for r in m.search("payroll", filters=scope)["results"]:
    print(r["memory"], "|", r.get("metadata"))
```

```
Payroll cost grew 8% QoQ | None
Q3 payroll total is 4.2M | {'label': 'hr.salary'}
```

With `infer=True` this happens implicitly. The extraction prompt includes up to 10 existing memories from the scope ([main.py#L928](https://github.com/mem0ai/mem0/blob/f1791daa/mem0/memory/main.py#L928)), but extracted facts get only the current `add()` call's metadata ([#L1031](https://github.com/mem0ai/mem0/blob/f1791daa/mem0/memory/main.py#L1031)).

### Proposed Solution

No core change. `amu-mem0` is an external wrapper, `GatedMemory(Memory())`. It records `sources`/`derived_from` in metadata, closes lineage at write time, and filters `search`/`get`/`get_all` against the requester's entitlements. It fails closed on missing lineage and rejects `infer=True`. Limitations are in its README.

One question: would you like this linked from the docs (I'd write the page), or is it out of scope for mem0?

### Alternatives Considered

A store-side hook (#7870) or typed provenance (#7782) would let a check like this sit closer to the store. This doesn't depend on either.

Disclosure: I co-authored the paper this rule comes from (Sangaraju & Vissa, IEEE Access 2026, doi:10.1109/ACCESS.2026.3730363) and wrote amu-mem0. AI-assisted (Claude Code); I designed the approach and reviewed all code and the benchmark.

Code and benchmark: https://github.com/sangaraju1988/amu-mem0
