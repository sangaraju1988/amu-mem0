# Draft: mem0 feature request (NOT POSTED)

Target: https://github.com/mem0ai/mem0/issues/new?template=feature_request.yml

**Title:** Derived memories don't carry their sources' restrictions when teams share one agent's memory

**Component (dropdown):** Python SDK

**AI Assistance (dropdown):** AI-assisted, but the need is mine

**Description (field body below the line):**

---

### Use Case

When several teams share one agent's memory (one `agent_id`), mem0's scoping tells you whose memory it is, not what it was derived from. A memory the agent writes *from* a restricted record keeps no trace of that record, so no filter on the source's metadata can hide it.

Repro (mem0ai 2.2.1, local Qdrant):

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

Measured on a synthetic, deterministic benchmark: 2 teams, one shared scope, 420 memories including 2- and 3-hop summaries, 240 queries at top_k=10, reproducible with `make bench`.

| | Restricted items returned | Recall@10 for entitled users |
|---|---|---|
| plain mem0 | 35.7% | 0.650 |
| filter on each memory's own labels | 23.5% | 0.816 |
| lineage-gated wrapper | 0.0% | 0.999 |

The 0% holds only when lineage is fully declared. With 90% of derivation edges recorded, 1.4% still leaked.

### Proposed Solution

No core change. `amu-mem0` is an external wrapper, `GatedMemory(Memory())`. It records `sources`/`derived_from` in metadata, closes lineage at write time, and filters `search`/`get`/`get_all` against the requester's entitlements. It fails closed on missing lineage and rejects `infer=True`. It relies only on metadata round-tripping. Limitations are listed in its README.

One question: would you like this linked from the docs (I'd write the page), or is it out of scope for mem0? Either answer is fine.

### Alternatives Considered

A store-side hook (#7870) or typed provenance (#7782) would let a check like this sit closer to the store. This doesn't depend on either.

Disclosure: I co-authored the paper this rule comes from (Sangaraju & Vissa, IEEE Access 2026, doi:10.1109/ACCESS.2026.3730363) and wrote amu-mem0. AI-assisted (Claude Code); I designed the approach, reviewed all code, and ran the benchmark myself.

Code and benchmark: <REPO_URL>
