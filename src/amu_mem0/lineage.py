"""Lineage records stored in mem0 metadata, and the read-time check.

A memory's lineage is kept under one reserved metadata key (``"amu"``):

- ``sources``: the raw sources the memory was *directly* derived from, each
  mapped to its sensitivity labels, e.g. ``{"hris": ["hr.salary"]}``.
- ``derived_from``: ids of other mem0 memories it was derived from.
- ``closed``: the transitive closure of ``sources`` over ``derived_from``,
  computed once at write time. A summary of a summary carries every label of
  every source anywhere beneath it.
- ``complete``: False when any part of the lineage could not be established
  (undeclared, unresolved parent, parent with incomplete lineage, or written
  through LLM extraction). Incomplete lineage is denied by default.

The allow/deny decision itself is delegated to ``amu_governance``: each closed
source becomes a ``LineageStep(table=source, columns_used=labels)`` and the
requester's entitlements become a one-department ``GovernancePolicy``. A read is
allowed iff ``S(a) ⊆ P(d)`` (Sangaraju & Vissa, IEEE Access 2026,
doi:10.1109/ACCESS.2026.3730363).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from amu_governance import GovernancePolicy, Lineage, LineageStep

METADATA_KEY = "amu"
SCHEMA_VERSION = 1

_REQUESTER = "__requester__"


class MalformedLineageError(ValueError):
    """Raised when stored ``amu`` metadata cannot be parsed."""


@dataclass(frozen=True)
class LineageRecord:
    """Lineage of one memory, as stored in its metadata."""

    sources: dict[str, tuple[str, ...]] = field(default_factory=dict)
    derived_from: tuple[str, ...] = ()
    closed: dict[str, tuple[str, ...]] = field(default_factory=dict)
    complete: bool = True
    incomplete_reason: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for mem0 metadata."""
        return {
            "v": SCHEMA_VERSION,
            "sources": {s: list(labels) for s, labels in self.sources.items()},
            "derived_from": list(self.derived_from),
            "closed": {s: list(labels) for s, labels in self.closed.items()},
            "complete": self.complete,
            "incomplete_reason": self.incomplete_reason,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None) -> LineageRecord | None:
        """Parse the ``amu`` entry of a mem0 result's ``metadata``.

        Returns None when the memory carries no lineage at all. Raises
        MalformedLineageError when the entry exists but is not well formed.
        """
        if not metadata or METADATA_KEY not in metadata:
            return None
        raw = metadata[METADATA_KEY]
        try:
            if not isinstance(raw, Mapping) or raw.get("v") != SCHEMA_VERSION:
                raise TypeError("unsupported or missing schema version")
            complete = raw["complete"]
            if not isinstance(complete, bool):
                raise TypeError("'complete' must be a bool")
            return cls(
                sources=_label_map(raw["sources"]),
                derived_from=tuple(str(x) for x in raw["derived_from"]),
                closed=_label_map(raw["closed"]),
                complete=complete,
                incomplete_reason=raw.get("incomplete_reason"),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise MalformedLineageError(str(e)) from e

    def all_labels(self) -> set[str]:
        """Every label in the closed lineage."""
        return {label for labels in self.closed.values() for label in labels}

    def as_amu_lineage(self) -> Lineage:
        """The closed lineage as an ``amu_governance.Lineage``."""
        steps = tuple(LineageStep(table=src, columns_used=labels) for src, labels in sorted(self.closed.items()))
        return Lineage(steps=steps, filter_logic="")


def _label_map(raw: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        raise TypeError("expected a mapping of source -> labels")
    out: dict[str, tuple[str, ...]] = {}
    for src, labels in raw.items():
        if isinstance(labels, (str, bytes)):
            raise TypeError(f"labels for source {src!r} must be a list, not a string")
        out[str(src)] = tuple(sorted({str(label) for label in labels}))
    return out


def normalize_sources(sources: Mapping[str, Iterable[str]] | None) -> dict[str, tuple[str, ...]] | None:
    """Validate and normalize a caller-supplied ``sources`` mapping."""
    if sources is None:
        return None
    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a mapping of source name -> iterable of labels")
    return _label_map(sources)


def merge_closed(*maps: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    """Union several source -> labels maps."""
    acc: dict[str, set[str]] = {}
    for m in maps:
        for src, labels in m.items():
            acc.setdefault(src, set()).update(labels)
    return {src: tuple(sorted(labels)) for src, labels in sorted(acc.items())}


def unentitled_labels(
    record: LineageRecord,
    entitlements: Iterable[str],
    sensitive_labels: Iterable[str] | None,
) -> set[str]:
    """Sensitive labels in ``record``'s closed lineage that ``entitlements`` does not cover.

    When ``sensitive_labels`` is None every label is treated as sensitive, so a
    requester must hold every label anywhere in the lineage. Otherwise only
    labels in the registry gate the read (the paper's semantics).
    """
    registry = record.all_labels() if sensitive_labels is None else set(sensitive_labels)
    policy = GovernancePolicy(
        sensitive_columns=registry,
        department_permissions={_REQUESTER: set(entitlements)},
    )
    return record.as_amu_lineage().sensitive_columns(policy) - policy.permitted_columns(_REQUESTER)
