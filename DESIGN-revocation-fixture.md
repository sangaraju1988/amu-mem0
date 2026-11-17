# Revocation-after-write fixture

Branch `feat/revocation-fixture`. Tests: [`tests/test_revocation.py`](tests/test_revocation.py)
(14 tests). This is White Paper 2's H1 experiment, built to the spec from mem0
Discussion #7390.

## How to run

```bash
uv sync                                    # installs the pinned versions below from uv.lock
make test                                  # whole suite
uv run pytest tests/test_revocation.py -k test_revocation_after_write_mixed_sources_all_paths
make bench                                 # synthetic leak/recall/latency benchmark -> results/BENCH.md
```

Pinned versions, as read from `uv.lock` on 2026-11-17: **mem0ai 2.2.1, amu-governance 0.1.2,
amu-mem0 0.2.0, qdrant-client 1.20.0**. Every test runs offline. The store is a real mem0
`Memory` on local Qdrant and SQLite, with a deterministic hashed embedder, `infer=False`, and
no network or LLM calls.

**No permission cache.** amu-mem0 re-evaluates the requester's current entitlements on every
read, so the fixture's sequence is allow, then change entitlements, then re-read. There is no
cache to warm.

## Cases

The shared shape is: **A** carries `hr.salary`; **pub** is from an unrestricted source; **B** is
derived from both (`derived_from=[a, pub]`). "All three paths" means `search`, `get` and
`get_all`.

**Revocation after write.**
- `test_revocation_after_write_mixed_sources_all_paths`: B is readable on all three paths while
  the requester holds `hr.salary`. After the label is dropped, B is denied on all three. pub
  stays readable throughout, which shows the denial comes from B's closure, not from the
  requester being locked out.
- `test_grant_then_revoke_mixed_sources_all_paths`: starting unentitled, B goes denied, then
  allowed after a grant, then denied after a revoke, on all three paths.
- `test_revocation_is_per_requester`: revoking one requester's access leaves a second, still
  entitled requester able to read B on all three paths.

**Ancestor deleted.** These are separate from the authority-change test below.
- `test_delete_ancestor_denies_descendant_to_entitled_requester`: after `gm.delete(a)`, B is
  denied even to a requester holding `hr.salary`, on all three paths. `check()` reports
  `incomplete_lineage`.
- `test_delete_ancestor_marks_descendants_incomplete_naming_ancestor`: B and a grandchild C are
  stored with `complete=False` and `incomplete_reason` containing `ancestor <a> deleted`. Their
  known labels are kept.
- `test_delete_without_descendants_changes_nothing_else`: deleting an unrelated leaf leaves
  every other lineage record byte-identical, and B stays readable.
- `test_delete_refuses_over_partial_index_and_keeps_memory`: with `max_index_scan` exceeded,
  `delete()` raises `IndexScanLimitError` and deletes nothing.

**Ancestor authority changed** (A still exists).
- `test_ancestor_authority_change_denies_descendant`: A is unrestricted at write time, then
  `restrict_source("hris", ["hr.salary"])`. B is then denied to a requester without the label
  and allowed to one with it, on all three paths.

**Public-response privacy.**
- `test_public_response_reveals_no_denied_candidates`: `search` and `get_all` return exactly
  `{"results": [...]}`. The denied record's id, content, source and label appear nowhere in
  the serialized response.
- `test_short_page_carries_no_denial_signal`: a page with fewer than `top_k` results says
  nothing about why, or how many records were withheld.
- `test_get_denied_and_missing_are_indistinguishable`: `get` returns `None` for both.
- `test_denial_accounting_is_operator_side_only`: per-item `check()` decisions and the
  `amu_mem0` log record each denial (id, reason, labels). The log contains no memory text.

**Pagination.**
- `test_refill_when_denied_candidates_consume_initial_top_k`: mem0's own top 5 for the query
  is entirely restricted memories (asserted as a precondition). The gated page still holds 5
  allowed memories, in mem0's score order.
- `test_pages_by_deeper_fetch_are_stable_without_duplicates`: page 2 is items 5–10 of a
  `top_k=10` fetch. It extends page 1 without reordering it and shares no ids with it.

## Decisions and findings

**`amu_denied` is internal-only.** The released 0.1.0 returned an `amu_denied` list (ids,
reasons, labels) from `search`/`get_all`, which told the caller that restricted records exist.
The public response now carries only allowed results. Denial detail goes to the `amu_mem0`
logger and is available per item via `check()`. There is no public accessor for denied lists.

**Delete-ancestor mechanism: delete-time marking.** A probe on 2026-11-17 showed that after
`gm.delete(a)`, B kept `complete: True` and was still returned to an entitled requester on
`get` and `search`, because `check()` reads only B's own stored metadata. `delete()` now:
1. builds the reverse index first (bounded; raises before deleting if it can't);
2. deletes the memory;
3. walks its descendants through the kept parent→child edges, with a visited set;
4. rewrites each descendant with `complete=False` and a reason naming the deleted id.

I chose this over a read-time liveness check because it keeps reads at their current cost and
reuses the `restrict_source()` index. The trade-off is that deletes made directly on
`gm.memory`, or by another process, are not seen; that boundary is documented. Repairing
lineage after a delete is future work: descendants stay denied until a repair API exists.

**Ordering and paging, verified empirically.**
- mem0 2.2.1's `search`/`get_all` accept no offset or cursor. Unknown keyword arguments are
  silently ignored, so a test that passed `offset=` would test nothing. Paging is therefore a
  deeper fetch.
- With the deterministic embedder and allowed items that score distinctly, five fresh stores
  all gave: identical repeated pages, page 1 equal to the first 5 items of the deeper fetch,
  non-increasing scores, and no duplicates.
- That stability holds for this setup only. mem0's hybrid scores depend on `top_k`, so other
  embedders or stores may reorder ties.

**Declassification is out of scope.** The note from #7390 is recorded as design input only:
the definition hash identifies the computation, and releasing restricted data needs a separately
versioned policy decision that is re-evaluated when the policy changes. No declassification
registry is built here. That is White Paper 2's H2.

## Known limitations

- **An allowed result can name an ancestor the requester can't read.** This was probed on
  2026-11-17. A parent gains restrictions through `update()` after its child was written, and
  updates don't propagate to children by design, so the child stays readable. Its
  `metadata.amu.derived_from` then contains the parent id, which reveals that the parent exists
  (not its content or labels). `restrict_source()` cascades to children and doesn't have this
  gap. Closing it would mean redacting lineage metadata in public results, which is not done here.
- **Deletes need the gated path.** Calls on `gm.memory` are ungated and skip descendant
  marking.
- **The first `delete()` per `GatedMemory` scans the whole store** (bounded by
  `max_index_scan`).
- **Scope of the evidence.** The fixture checks correctness on a small deterministic store. It
  does not measure revocation at scale or under concurrent writers.
