"""``GatedMemory``: a derivation gate around an existing mem0 ``Memory``."""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from .lineage import (
    METADATA_KEY,
    LineageRecord,
    MalformedLineageError,
    merge_closed,
    normalize_sources,
    unentitled_labels,
)

logger = logging.getLogger("amu_mem0")

MissingLineagePolicy = Literal["deny", "allow", "allow_with_warning"]

# Reason codes reported for every decision.
ALLOWED = "allowed"
MISSING_LINEAGE = "missing_lineage"
INCOMPLETE_LINEAGE = "incomplete_lineage"
MALFORMED_LINEAGE = "malformed_lineage"
UNENTITLED = "unentitled"


class InferNotAllowedError(ValueError):
    """Raised when ``add(infer=True)`` is called on a gate that does not allow it."""


class LineageMissingWarning(UserWarning):
    """Emitted when a memory without complete lineage is returned under ``allow_with_warning``."""


class IndexScanLimitError(RuntimeError):
    """Raised when building the reverse index would need to scan more than ``max_index_scan`` memories.

    The cascade never runs over a partial index.
    """


@dataclass(frozen=True)
class RestrictionResult:
    """Outcome of ``GatedMemory.restrict_source``."""

    source: str
    labels: tuple[str, ...]
    updated: tuple[str, ...]
    skipped: tuple[str, ...]


class _LineageIndex:
    """In-memory reverse index: source -> memory ids, parent id -> child ids."""

    def __init__(self) -> None:
        self.by_source: dict[str, set[str]] = defaultdict(set)
        self.by_parent: dict[str, set[str]] = defaultdict(set)
        self._entries: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}

    def put(self, memory_id: str, record: LineageRecord) -> None:
        self.remove(memory_id)
        sources = tuple(record.closed)
        for src in sources:
            self.by_source[src].add(memory_id)
        for pid in record.derived_from:
            self.by_parent[pid].add(memory_id)
        self._entries[memory_id] = (sources, record.derived_from)

    def remove(self, memory_id: str) -> None:
        """Drop ``memory_id`` as a source holder and as a child. Its own child edges are kept:
        its children still name it in ``derived_from``."""
        sources, parents = self._entries.pop(memory_id, ((), ()))
        for src in sources:
            self.by_source[src].discard(memory_id)
        for pid in parents:
            self.by_parent[pid].discard(memory_id)

    def dependents(self, source: str) -> list[str]:
        """Every memory whose closed lineage names ``source``, plus all their descendants."""
        return self._walk(sorted(self.by_source.get(source, ())))

    def descendants(self, memory_id: str) -> list[str]:
        """Every memory derived, directly or transitively, from ``memory_id`` (excluding itself)."""
        return [m for m in self._walk(sorted(self.by_parent.get(memory_id, ()))) if m != memory_id]

    def _walk(self, start: list[str]) -> list[str]:
        """Breadth-first over child edges with a visited set, so ``derived_from`` cycles terminate."""
        order: list[str] = []
        seen: set[str] = set()
        queue = deque(start)
        while queue:
            mid = queue.popleft()
            if mid in seen:
                continue
            seen.add(mid)
            order.append(mid)
            queue.extend(sorted(self.by_parent.get(mid, ())))
        return order


@dataclass(frozen=True)
class Decision:
    """The gate's verdict for one memory and one requester."""

    memory_id: str
    allowed: bool
    reason: str
    unentitled_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.memory_id, "reason": self.reason, "unentitled_labels": list(self.unentitled_labels)}


class GatedMemory:
    """Wrap a mem0 ``Memory`` so reads are gated on derivation lineage.

    Example::

        from mem0 import Memory
        from amu_mem0 import GatedMemory

        gm = GatedMemory(Memory())
        src = gm.add("Q3 payroll is 4.2M", agent_id="analyst", sources={"hris": ["hr.salary"]})
        gm.add("Payroll grew 8% QoQ", agent_id="analyst", derived_from=[src["results"][0]["id"]])
        gm.search("payroll", filters={"agent_id": "analyst"}, entitlements={"mkt.campaign"})
        # -> neither memory is returned; the summary inherits hr.salary from its source.

    Args:
        memory: an existing ``mem0.Memory`` instance. It is used as-is.
        sensitive_labels: registry of labels that gate reads. ``None`` (default)
            treats every label as sensitive.
        on_missing_lineage: what to do with memories whose lineage is absent,
            incomplete or malformed: ``"deny"`` (default), ``"allow"``, or
            ``"allow_with_warning"``.
        allow_infer: permit ``add(infer=True)``. Such memories are stored with
            ``complete=False`` because mem0's extraction step reads other
            memories in scope that the caller cannot declare as sources.
        fetch_multiplier: initial over-fetch factor for search/get_all, so that
            filtered-out items do not starve ``top_k``.
        max_fetch: upper bound on how many candidates one gated read requests
            from mem0.
        max_index_scan: upper bound on how many memories ``restrict_source`` may
            scan to build its reverse index. Exceeding it raises
            ``IndexScanLimitError`` instead of cascading over a partial index.
    """

    def __init__(
        self,
        memory: Any,
        *,
        sensitive_labels: Iterable[str] | None = None,
        on_missing_lineage: MissingLineagePolicy = "deny",
        allow_infer: bool = False,
        fetch_multiplier: int = 4,
        max_fetch: int = 1000,
        max_index_scan: int = 100_000,
    ) -> None:
        if on_missing_lineage not in ("deny", "allow", "allow_with_warning"):
            raise ValueError(f"invalid on_missing_lineage: {on_missing_lineage!r}")
        if fetch_multiplier < 1 or max_fetch < 1 or max_index_scan < 1:
            raise ValueError("fetch_multiplier, max_fetch and max_index_scan must be >= 1")
        self._memory = memory
        self.sensitive_labels = None if sensitive_labels is None else frozenset(sensitive_labels)
        self.on_missing_lineage = on_missing_lineage
        self.allow_infer = allow_infer
        self.fetch_multiplier = fetch_multiplier
        self.max_fetch = max_fetch
        self.max_index_scan = max_index_scan
        self._index: _LineageIndex | None = None

    @property
    def memory(self) -> Any:
        """The wrapped mem0 ``Memory``. Calls made on it directly are NOT gated."""
        return self._memory

    # ------------------------------------------------------------------ write

    def add(
        self,
        messages: Any,
        *,
        sources: Mapping[str, Iterable[str]] | None = None,
        derived_from: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a memory with declared lineage.

        Args:
            messages: passed to ``Memory.add`` unchanged.
            sources: raw sources this memory was directly derived from, mapped to
                their labels, e.g. ``{"hris": ["hr.salary"], "crm": []}``.
                ``{}`` declares "no restricted inputs".
            derived_from: ids of mem0 memories this memory was derived from.
                Their closed lineage is inherited.
            metadata: extra metadata; the ``"amu"`` key is reserved.
            infer: mem0's LLM extraction. Rejected unless ``allow_infer=True``.
            **kwargs: forwarded to ``Memory.add`` (``user_id``, ``agent_id``, ...).

        If neither ``sources`` nor ``derived_from`` is given, the memory is
        stored with incomplete lineage and is hidden under the default policy.
        """
        if infer and not self.allow_infer:
            raise InferNotAllowedError(
                "add(infer=True) is rejected by default: mem0's extraction step can draw on other "
                "memories in scope, so the result's lineage cannot be declared. Pass infer=False, or "
                "construct GatedMemory(allow_infer=True) to store such memories as incomplete lineage."
            )
        if metadata and METADATA_KEY in metadata:
            raise ValueError(f"metadata key {METADATA_KEY!r} is reserved for lineage")

        record = self._build_record(sources, derived_from, inferred=infer)
        merged = dict(metadata or {})
        merged[METADATA_KEY] = record.to_metadata()
        if not record.complete:
            logger.info("amu_mem0 add stored with incomplete lineage: %s", record.incomplete_reason)
        result = self._memory.add(messages, metadata=merged, infer=infer, **kwargs)
        if self._index is not None:
            for item in result.get("results", []):
                if item.get("id"):
                    self._index.put(str(item["id"]), record)
        return result

    def update(
        self,
        memory_id: str,
        text: str | None = None,
        *,
        sources: Mapping[str, Iterable[str]] | None = None,
        derived_from: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update a memory and recompute its lineage as a union that never shrinks.

        Args:
            memory_id: the memory to update.
            text: new content, passed to ``Memory.update`` (re-embedded by mem0).
            sources: raw sources the new content was derived from, as in ``add()``.
            derived_from: ids of memories the new content was derived from.
            metadata: extra metadata merged by mem0; the ``"amu"`` key is reserved.
            **kwargs: forwarded to ``Memory.update`` (e.g. ``expiration_date``).

        The stored lineage becomes existing ∪ declared: labels are never removed.
        Replacing ``text`` without declaring ``sources`` or ``derived_from`` marks the
        lineage incomplete, as ``add()`` does for undeclared memories; pass
        ``sources={}`` to declare "no restricted inputs". Existing children are not
        rewritten; use ``restrict_source()`` to push a restriction to descendants.
        """
        if metadata and METADATA_KEY in metadata:
            raise ValueError(f"metadata key {METADATA_KEY!r} is reserved for lineage")
        direct = normalize_sources(sources)
        parents = _dedupe_ids(derived_from)
        declared = direct is not None or bool(parents)
        if text is None and not declared and not metadata and not kwargs:
            raise ValueError("update() needs text, sources, derived_from, metadata or a mem0 keyword argument")
        existing = self._memory.get(memory_id)
        if existing is None:
            raise ValueError(f"memory {memory_id} not found")

        problems: list[str] = []
        try:
            old = LineageRecord.from_metadata(existing.get("metadata"))
        except MalformedLineageError:
            old = None
            problems.append("existing lineage was malformed")
        if old is None and not problems and not declared:
            problems.append("no existing lineage and none declared")
        if old is not None and not old.complete:
            problems.append(old.incomplete_reason or "existing lineage incomplete")
        if text is not None and not declared:
            problems.append("text replaced without declaring its sources")
        parent_maps, parent_problems = self._resolve_parents(parents)
        problems += parent_problems

        base = old or LineageRecord()
        record = LineageRecord(
            sources=merge_closed(base.sources, direct or {}),
            derived_from=_dedupe_ids(base.derived_from + parents),
            closed=merge_closed(base.closed, direct or {}, *parent_maps),
            complete=not problems,
            incomplete_reason="; ".join(dict.fromkeys(problems)) or None,
        )
        merged = dict(metadata or {})
        merged[METADATA_KEY] = record.to_metadata()
        result = self._memory.update(memory_id, text=text, metadata=merged, **kwargs)
        if not record.complete:
            logger.info("amu_mem0 update id=%s stored with incomplete lineage", memory_id)
        if self._index is not None:
            self._index.put(str(memory_id), record)
        return result

    # ----------------------------------------------------------- restriction

    def restrict_source(self, source: str, labels: Iterable[str]) -> RestrictionResult:
        """Add ``labels`` to ``source`` and cascade them to every dependent memory.

        Every memory whose closed lineage names ``source`` gets the labels, and so does
        each of their descendants (via ``derived_from``), walked with a visited set so
        cycles terminate. Labels are only ever added. Descendants are restricted even if
        their stored lineage predates ``source``; this over-approximates, failing closed.

        The reverse index is built on first use by scanning the whole collection
        (bounded by ``max_index_scan``) and kept in sync by this instance's
        ``add``/``update``/``delete``. Writes made directly on ``memory`` are not seen.
        """
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        if isinstance(labels, (str, bytes)):
            raise TypeError("labels must be an iterable of strings, not a string")
        new_labels = tuple(sorted({str(label) for label in labels}))
        if not new_labels:
            raise ValueError("labels must not be empty")

        index = self._ensure_index()
        updated: list[str] = []
        skipped: list[str] = []
        for mid in index.dependents(source):
            item = self._memory.get(mid)
            if item is None:
                skipped.append(mid)  # deleted; its children were still walked
                continue
            try:
                rec = LineageRecord.from_metadata(item.get("metadata"))
            except MalformedLineageError:
                rec = None
            if rec is None:
                skipped.append(mid)  # already denied by default
                continue
            add = {source: new_labels}
            new_rec = LineageRecord(
                sources=merge_closed(rec.sources, add) if source in rec.sources else rec.sources,
                derived_from=rec.derived_from,
                closed=merge_closed(rec.closed, add),
                complete=rec.complete,
                incomplete_reason=rec.incomplete_reason,
            )
            self._memory.update(mid, metadata={METADATA_KEY: new_rec.to_metadata()})
            index.put(mid, new_rec)
            updated.append(mid)
        logger.info(
            "amu_mem0 restrict_source source=%s labels=%s updated=%d skipped=%d",
            source,
            ",".join(new_labels),
            len(updated),
            len(skipped),
        )
        return RestrictionResult(source, new_labels, tuple(updated), tuple(skipped))

    def _ensure_index(self) -> _LineageIndex:
        if self._index is not None:
            return self._index
        listed = self._memory.vector_store.list(filters=None, top_k=self.max_index_scan + 1)
        if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], (list, tuple)):
            listed = listed[0]  # some stores return (rows, next_offset)
        rows = list(listed or ())
        if len(rows) > self.max_index_scan:
            raise IndexScanLimitError(
                f"reverse index needs more than max_index_scan={self.max_index_scan} memories; "
                "raise the limit to cascade over this store"
            )
        index = _LineageIndex()
        for row in rows:
            try:
                rec = LineageRecord.from_metadata(getattr(row, "payload", None) or {})
            except MalformedLineageError:
                continue
            if rec is not None:
                index.put(str(row.id), rec)
        self._index = index
        return index

    def _build_record(
        self,
        sources: Mapping[str, Iterable[str]] | None,
        derived_from: Iterable[str] | None,
        *,
        inferred: bool,
    ) -> LineageRecord:
        direct = normalize_sources(sources)
        parents = _dedupe_ids(derived_from)
        problems: list[str] = []
        if direct is None and not parents:
            problems.append("no sources or derived_from declared")
        parent_maps, parent_problems = self._resolve_parents(parents)
        problems += parent_problems
        if inferred:
            problems.append("written with infer=True")
        return LineageRecord(
            sources=direct or {},
            derived_from=parents,
            closed=merge_closed(direct or {}, *parent_maps),
            complete=not problems,
            incomplete_reason="; ".join(problems) or None,
        )

    def _resolve_parents(self, parents: Iterable[str]) -> tuple[list[dict[str, tuple[str, ...]]], list[str]]:
        """Fetch each parent's closed lineage. Returns (closed maps, problems)."""
        maps: list[dict[str, tuple[str, ...]]] = []
        problems: list[str] = []
        for pid in parents:
            parent = self._memory.get(pid)
            if parent is None:
                problems.append(f"parent {pid} not found")
                continue
            try:
                prec = LineageRecord.from_metadata(parent.get("metadata"))
            except MalformedLineageError:
                problems.append(f"parent {pid} has malformed lineage")
                continue
            if prec is None:
                problems.append(f"parent {pid} has no lineage")
                continue
            # Inherit what is known even when the parent is incomplete; the child is
            # then incomplete too, so it can never be "more visible" than its parent.
            maps.append(prec.closed)
            if not prec.complete:
                problems.append(f"parent {pid} has incomplete lineage")
        return maps, problems

    # ------------------------------------------------------------------- read

    def check(self, item: Mapping[str, Any], *, entitlements: Iterable[str]) -> Decision:
        """Decide whether a mem0 result dict may be returned to a requester."""
        memory_id = str(item.get("id"))
        try:
            record = LineageRecord.from_metadata(item.get("metadata"))
        except MalformedLineageError:
            return self._missing(memory_id, MALFORMED_LINEAGE)
        if record is None:
            return self._missing(memory_id, MISSING_LINEAGE)
        bad = unentitled_labels(record, entitlements, self.sensitive_labels)
        if bad:
            # Known-forbidden labels deny regardless of the missing-lineage policy.
            return Decision(memory_id, False, UNENTITLED, tuple(sorted(bad)))
        if not record.complete:
            return self._missing(memory_id, INCOMPLETE_LINEAGE)
        return Decision(memory_id, True, ALLOWED)

    def _missing(self, memory_id: str, reason: str) -> Decision:
        if self.on_missing_lineage == "deny":
            return Decision(memory_id, False, reason)
        if self.on_missing_lineage == "allow_with_warning":
            warnings.warn(f"memory {memory_id} returned with {reason}", LineageMissingWarning, stacklevel=4)
        return Decision(memory_id, True, reason)

    def _log_denial(self, op: str, decision: Decision) -> None:
        # Never log memory content: only id, reason and label names.
        logger.info(
            "amu_mem0 %s denied id=%s reason=%s unentitled=%s",
            op,
            decision.memory_id,
            decision.reason,
            ",".join(decision.unentitled_labels) or "-",
        )

    def _gated_fetch(self, op: str, fetch_fn: Any, top_k: int, entitlements: Iterable[str]) -> dict[str, Any]:
        ents = frozenset(entitlements)
        fetch = min(max(top_k * self.fetch_multiplier, top_k), self.max_fetch)
        prev_len = -1
        while True:
            raw = fetch_fn(fetch)["results"]
            allowed: list[dict[str, Any]] = []
            denied: list[Decision] = []
            for item in raw:
                d = self.check(item, entitlements=ents)
                if d.allowed:
                    allowed.append(item)
                else:
                    denied.append(d)
            # mem0 can return fewer than ``fetch`` results even when more exist: it applies a
            # score threshold and sizes its candidate pool from top_k. So "fewer than asked"
            # does not mean exhausted; stop only when a larger fetch adds no new candidates.
            stalled = len(raw) == prev_len
            if len(allowed) >= top_k or stalled or fetch >= self.max_fetch:
                break
            prev_len = len(raw)
            fetch = min(fetch * 2, self.max_fetch)
        # Denials are internal diagnostics: logged for operators (ids, reasons, label
        # names; never content) but never returned. The public response must not reveal
        # that denied records exist, so it carries no ids, labels or counts for them.
        for d in denied:
            self._log_denial(op, d)
        return {"results": allowed[:top_k]}

    def search(
        self,
        query: str,
        *,
        entitlements: Iterable[str],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """``Memory.search`` with the gate applied.

        Returns ``{"results": [...]}`` holding only allowed memories. Nothing about
        denied candidates (ids, labels, reasons or how many there were) appears in the
        response; a short page is simply short. Denials are logged on the ``amu_mem0``
        logger (ids, reasons and label names, never memory text), and ``check()`` gives
        the decision for an individual item.
        """
        return self._gated_fetch(
            "search",
            lambda k: self._memory.search(query, top_k=k, filters=filters, **kwargs),
            top_k,
            entitlements,
        )

    def get_all(
        self,
        *,
        entitlements: Iterable[str],
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """``Memory.get_all`` with the gate applied. Returns ``{"results": [...]}`` of allowed
        memories only, with no information about denied ones (see ``search``)."""
        return self._gated_fetch(
            "get_all",
            lambda k: self._memory.get_all(filters=filters, top_k=k, **kwargs),
            top_k,
            entitlements,
        )

    def get(self, memory_id: str, *, entitlements: Iterable[str]) -> dict[str, Any] | None:
        """``Memory.get`` with the gate applied. Returns None if missing or denied."""
        item = self._memory.get(memory_id)
        if item is None:
            return None
        d = self.check(item, entitlements=entitlements)
        if not d.allowed:
            self._log_denial("get", d)
            return None
        return item

    def delete(self, memory_id: str) -> Any:
        """Delete a memory and mark everything derived from it as unverified.

        Once an ancestor is gone, the store can no longer prove a descendant's full
        source closure, so every descendant (walked through ``derived_from`` with a
        visited set) is rewritten with ``complete=False`` and an ``incomplete_reason``
        naming the deleted id. Under the default policy they are then denied, even to
        requesters holding every label, until their lineage is repaired.

        The reverse index is built first (bounded by ``max_index_scan``). If it cannot
        be built, ``IndexScanLimitError`` is raised before anything is deleted. Returns
        mem0's ``delete`` result.
        """
        index = self._ensure_index()
        result = self._memory.delete(memory_id)
        mid = str(memory_id)
        index.remove(mid)
        marked = 0
        for desc in index.descendants(mid):
            item = self._memory.get(desc)
            if item is None:
                continue
            try:
                rec = LineageRecord.from_metadata(item.get("metadata"))
            except MalformedLineageError:
                continue  # already denied by default
            if rec is None:
                continue
            reasons = [r for r in (rec.incomplete_reason, f"ancestor {mid} deleted") if r]
            new_rec = replace(rec, complete=False, incomplete_reason="; ".join(dict.fromkeys(reasons)))
            self._memory.update(desc, metadata={METADATA_KEY: new_rec.to_metadata()})
            index.put(desc, new_rec)
            marked += 1
        logger.info("amu_mem0 delete id=%s descendants_marked_incomplete=%d", mid, marked)
        return result


def _dedupe_ids(ids: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(i) for i in (ids or ())))
