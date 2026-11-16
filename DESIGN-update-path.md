# Design: gating the update path, and cascading source restrictions

Status: implemented on branch `fix/update-path-gate` (targets 0.2.0).

Prompted by a comment on [mem0ai/mem0#7927](https://github.com/mem0ai/mem0/issues/7927#issuecomment-6911830883):
an update can rewrite a memory with content derived from a more restricted record,
and unless the lineage union is recomputed the access check drifts downward. Separately,
restricting a source has to reach every memory derived from it, which needs a reverse index.

## 0. What mem0 actually does (probed 2026-11-16, mem0ai 2.2.1, local Qdrant)

The probe was a throwaway script run against `amu_mem0.testing.offline_memory()`. It was not committed.

| Probe | Observed |
|---|---|
| `Memory.update` signature | `update(memory_id, text=None, metadata=None, expiration_date=<unset>, data=None)`. `data` is a deprecated alias for `text`. With no `text`, `metadata` or `expiration_date`, it raises `ValueError`. |
| Text-only `memory.update(id, text=...)` on a gated memory | The `amu` metadata is **preserved unchanged**. mem0 deep-copies the existing payload and only replaces `data`, `hash`, `text_lemmatized` and `updated_at`. |
| Re-embedding | **Yes.** The memory is found by a search for the new text and no longer by a search for the old text. |
| `memory.update(id, metadata={...})` | Metadata is **merged by top-level key**. `{"amu": X}` replaces only `amu`; other user keys (e.g. `team`) survive. Identity keys are stripped. |
| The gap | A public memory (`sources={}`) rewritten via `memory.update(id, text=<restricted content>)` is **still returned** by `gm.get(id, entitlements=set())`. This is a leak, not a fail-closed outcome. |
| Metadata-only update | mem0 re-embeds the unchanged text and writes an `UPDATE` history row. It costs one embedding call per memory touched. |

So the fix can re-attach the full lineage record in the **same** call that rewrites the text:
`memory.update(id, text=..., metadata={**user_metadata, "amu": new_record})`. No second write is needed.

API facts used: `merge_closed`, `LineageRecord` and `normalize_sources` are in amu-mem0's own
`lineage.py`. The decision rule (`S(a) ⊆ P(d)`) stays in amu-governance's `GovernancePolicy`
and `Lineage.sensitive_columns`, through `unentitled_labels()`, unchanged.

## 1. `GatedMemory.update()`

```python
update(memory_id, text=None, *, sources=None, derived_from=None, metadata=None, **kwargs)
```

The first parameter is named `text`, not `data` as in the original brief, because mem0 deprecates
`data` in favour of `text`. Positional calls (`gm.update(mid, "new text")`) behave the same either way.
`**kwargs` (e.g. `expiration_date`) pass through to `Memory.update`.

Rules:

1. **Monotonic lineage.** The new record is a union and never shrinks:
   - `sources` = the existing direct sources ∪ the newly declared `sources`;
   - `derived_from` = the existing parents ∪ the newly declared parents;
   - `closed` = the existing closed ∪ the new sources' labels ∪ each new parent's stored closure.
2. **A new text needs a declaration.** If `text` is given and neither `sources` nor `derived_from`
   is declared, the record becomes `complete=False` ("text replaced without declaring its
   sources"). The new content's provenance is unknown, and this matches `add()`, which stores
   undeclared memories as incomplete. Pass `sources={}` to declare "no restricted inputs".
   (This rule goes beyond the original brief; it closes the case where a gated update rewrites
   the text and declares nothing.)
3. **No existing lineage, no declaration:** the record is stored `complete=False`, which fails closed.
4. **No existing lineage, with a declaration:** the update establishes lineage. It is complete if
   every declared parent resolves to a memory with complete lineage. As with `add()`, this trusts
   the caller's declaration (see the README threat model).
5. **Unresolved or incomplete new parents** make the record `complete=False`. Labels already known
   from them are still inherited, as in `add()`.
6. **Existing incomplete or malformed lineage** stays incomplete. Malformed metadata is treated as
   "no lineage" and noted in `incomplete_reason`.
7. `metadata={"amu": ...}` raises `ValueError`, as in `add()`. Other user metadata keys are passed
   through and merged by mem0.
8. `infer` is not involved: `Memory.update` has no LLM step.
9. A missing `memory_id` raises `ValueError`.
10. Updating a memory does **not** rewrite its existing children. They were derived from the
    previous content, which the new sources did not contribute to. To push a restriction to
    descendants, use `restrict_source()`.

## 2. Reverse index

An in-memory, per-`GatedMemory` index with two maps:

- `by_source: source -> set[memory_id]`, for every memory whose **closed** lineage names `source`;
- `by_parent: parent_id -> set[child_id]`, from each memory's `derived_from`.

**Built lazily** on the first `restrict_source()` by scanning the whole collection with
`memory.vector_store.list(filters=None, top_k=max_index_scan + 1)`. The scan cannot use
`Memory.get_all`, because that requires a `user_id`/`agent_id`/`run_id` filter, and a source
restriction must reach every scope. **Bound:** `max_index_scan` (new optional constructor
argument, default 100,000). If the scan returns more rows than that, `restrict_source()` raises
`IndexScanLimitError`. It never cascades over a partial index.

**Kept in sync** once built:
- `add()` indexes each returned id with its record;
- `update()` replaces the id's entries;
- `delete()` removes the id from `by_source` and from its parents' child sets. The deleted id's
  own outgoing `by_parent` edges are kept, because its children still name it in `derived_from`,
  and a rebuild from scan would recreate those edges anyway. The walk passes through a deleted
  id without writing to it.

Before the index is first built, writes don't touch it; the later scan picks them up.

**Limitation:** the index lives in process memory. Writes made directly on `gm.memory`, or by
another process, are not reflected until a new `GatedMemory` rebuilds it. This is the same
boundary as the gate itself.

## 3. `restrict_source(source, labels)`

```python
restrict_source(source: str, labels: Iterable[str]) -> RestrictionResult
```

1. Validate: `source` is a non-empty string and `labels` is a non-empty iterable of strings
   (not a bare string).
2. Build the index if needed.
3. Seeds = `by_source[source]`. Walk breadth-first from the seeds through `by_parent`, with a
   **visited set** so that a `derived_from` cycle terminates. This reaches the full dependent
   closure, including descendants whose stored closure predates the source (for example, a child
   written before its parent gained that source). Restricting those descendants too is a
   deliberate **over-approximation**, in the fail-closed direction.
4. For each visited id: fetch it. If it was deleted, skip it but keep walking. If its lineage is
   missing or malformed, skip it and report it; it is already denied by default. Otherwise set
   `closed = merge_closed(closed, {source: labels})`. If `source` is one of its direct `sources`,
   union the labels there too. Write the record back with `memory.update(id, metadata={"amu": record})`
   and update the index.
5. Return `RestrictionResult(source, labels, updated=[ids], skipped=[ids])`. Log ids and label
   names only, never memory text.

Restriction only ever **adds** labels. Removing a label (declassification) is out of scope.

## 4. Out of scope

- The revocation-after-write experiment: changing a permission after a derived memory is
  written, then repeating `search` and `get_all`. `restrict_source()` is the mechanism that
  experiment will use, but **revocation is not tested here** beyond the unit tests listed in
  `tests/test_update.py`.
- A declassification registry (removing labels).
- A persistent (on-disk) index.
- Any change to the `LineageRecord` schema. `v` stays `1`; the stored metadata format is unchanged.
- Direct calls on `gm.memory` (including mem0's own `update`). These remain ungated, and a test
  pins that boundary explicitly.
- Concurrency: the index is not locked. Concurrent writers in one process can race.
