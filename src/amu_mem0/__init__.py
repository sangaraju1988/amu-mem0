"""amu-mem0: a derivation gate for mem0.

Reads return a memory only if the requester is entitled to every sensitive
label in its transitive lineage. Built on ``amu-governance``; see
Sangaraju & Vissa, IEEE Access 14 (2026), doi:10.1109/ACCESS.2026.3730363.
"""

from .gate import (
    ALLOWED,
    INCOMPLETE_LINEAGE,
    MALFORMED_LINEAGE,
    MISSING_LINEAGE,
    UNENTITLED,
    Decision,
    GatedMemory,
    IndexScanLimitError,
    InferNotAllowedError,
    LineageMissingWarning,
    RestrictionResult,
)
from .lineage import METADATA_KEY, LineageRecord, MalformedLineageError

__version__ = "0.2.0"

__all__ = [
    "GatedMemory",
    "Decision",
    "RestrictionResult",
    "IndexScanLimitError",
    "LineageRecord",
    "InferNotAllowedError",
    "LineageMissingWarning",
    "MalformedLineageError",
    "METADATA_KEY",
    "ALLOWED",
    "MISSING_LINEAGE",
    "INCOMPLETE_LINEAGE",
    "MALFORMED_LINEAGE",
    "UNENTITLED",
]
