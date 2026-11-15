"""``GatedMemory``: a derivation gate around an existing mem0 ``Memory``."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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
    ) -> None:
        if on_missing_lineage not in ("deny", "allow", "allow_with_warning"):
            raise ValueError(f"invalid on_missing_lineage: {on_missing_lineage!r}")
        if fetch_multiplier < 1 or max_fetch < 1:
            raise ValueError("fetch_multiplier and max_fetch must be >= 1")
        self._memory = memory
        self.sensitive_labels = None if sensitive_labels is None else frozenset(sensitive_labels)
        self.on_missing_lineage = on_missing_lineage
        self.allow_infer = allow_infer
        self.fetch_multiplier = fetch_multiplier
        self.max_fetch = max_fetch

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
        return self._memory.add(messages, metadata=merged, infer=infer, **kwargs)

    def _build_record(
        self,
        sources: Mapping[str, Iterable[str]] | None,
        derived_from: Iterable[str] | None,
        *,
        inferred: bool,
    ) -> LineageRecord:
        direct = normalize_sources(sources)
        parents = tuple(dict.fromkeys(str(p) for p in (derived_from or ())))
        problems: list[str] = []
        if direct is None and not parents:
            problems.append("no sources or derived_from declared")
        closed_maps: list[Mapping[str, Iterable[str]]] = [direct or {}]
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
            closed_maps.append(prec.closed)
            if not prec.complete:
                problems.append(f"parent {pid} has incomplete lineage")
        if inferred:
            problems.append("written with infer=True")
        return LineageRecord(
            sources=direct or {},
            derived_from=parents,
            closed=merge_closed(*closed_maps),
            complete=not problems,
            incomplete_reason="; ".join(problems) or None,
        )

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
            exhausted = len(raw) < fetch
            if len(allowed) >= top_k or exhausted or fetch >= self.max_fetch:
                break
            fetch = min(fetch * 2, self.max_fetch)
        for d in denied:
            self._log_denial(op, d)
        return {"results": allowed[:top_k], "amu_denied": [d.to_dict() for d in denied]}

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

        Returns mem0's result shape plus ``"amu_denied"``: one entry (id, reason,
        unentitled labels) per candidate the gate removed. Memory text of denied
        items is never included or logged.
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
        """``Memory.get_all`` with the gate applied (same return shape as ``search``)."""
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
        """Pass-through to ``Memory.delete``. Children keep their closed lineage."""
        return self._memory.delete(memory_id)
