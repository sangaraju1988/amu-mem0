"""Gated update() and source-restriction cascade, against a real (offline) mem0 Memory."""

import pytest

from amu_mem0 import METADATA_KEY, GatedMemory, IndexScanLimitError

from .conftest import SCOPE, add_id

HR = {"hr.salary"}
MKT = {"mkt.campaign"}


def test_update_with_restricted_source_denies(gm):
    mid = add_id(gm, "office closed on friday", sources={"intranet": []})
    gm.update(mid, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary"]})
    assert gm.get(mid, entitlements=MKT) is None
    assert gm.get(mid, entitlements=HR)["memory"] == "q3 payroll total is 4.2m"


def test_restrict_source_cascades_to_derived(gm):
    s = add_id(gm, "q3 payroll export", sources={"hris": []})
    b = add_id(gm, "payroll grew eight percent", derived_from=[s])
    assert gm.get(b, entitlements=MKT) is not None  # public before the restriction
    gm.restrict_source("hris", ["hr.salary"])
    assert gm.get(b, entitlements=MKT) is None
    assert gm.get(b, entitlements=HR) is not None


def _amu(gm, mid):
    return gm.memory.get(mid)["metadata"][METADATA_KEY]


def test_update_lineage_is_monotonic(gm):
    mid = add_id(gm, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary"]})
    gm.update(mid, "payroll figures", sources={})
    assert gm.get(mid, entitlements=MKT) is None
    assert _amu(gm, mid)["closed"] == {"hris": ["hr.salary"]}


def test_update_inherits_new_parent_lineage(gm):
    p = add_id(gm, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary"]})
    m = add_id(gm, "spend outlook", sources={"wiki": []})
    gm.update(m, derived_from=[p])
    assert gm.get(m, entitlements=MKT) is None
    assert gm.get(m, entitlements=HR) is not None
    assert _amu(gm, m)["derived_from"] == [p]


def test_update_without_lineage_is_incomplete(gm):
    raw = gm.memory.add("legacy note", infer=False, **SCOPE)["results"][0]["id"]
    gm.update(raw, "legacy note, revised")
    assert _amu(gm, raw)["complete"] is False
    assert gm.get(raw, entitlements=HR | MKT) is None


def test_update_text_without_declared_sources_is_incomplete(gm):
    mid = add_id(gm, "office closed on friday", sources={"intranet": []})
    gm.update(mid, "q3 payroll total is 4.2m")  # new content, provenance not declared
    rec = _amu(gm, mid)
    assert rec["complete"] is False and "without declaring" in rec["incomplete_reason"]
    assert gm.get(mid, entitlements=set()) is None


def test_restrict_source_cascades_transitively(gm):
    s = add_id(gm, "q3 payroll export", sources={"hris": []})
    b1 = add_id(gm, "payroll summary", derived_from=[s])
    b2 = add_id(gm, "payroll digest", derived_from=[b1])
    b3 = add_id(gm, "payroll briefing", derived_from=[b2])
    res = gm.restrict_source("hris", ["hr.salary"])
    assert set(res.updated) == {s, b1, b2, b3}
    for mid in (s, b1, b2, b3):
        assert gm.get(mid, entitlements=MKT) is None
        assert gm.get(mid, entitlements=HR) is not None


def test_restrict_source_is_cycle_safe(gm):
    a = add_id(gm, "note a", sources={"hris": []})
    b = add_id(gm, "note b", sources={"hris": []})
    # Build an a <-> b derived_from cycle by writing metadata directly (the gate never creates one).
    for mid, other in ((a, b), (b, a)):
        rec = dict(_amu(gm, mid), derived_from=[other])
        gm.memory.update(mid, metadata={METADATA_KEY: rec})
    res = gm.restrict_source("hris", ["hr.salary"])
    assert sorted(res.updated) == sorted([a, b])
    for mid in (a, b):
        assert _amu(gm, mid)["closed"]["hris"] == ["hr.salary"]


def test_reverse_index_survives_restart(memory):
    gm = GatedMemory(memory)
    s = add_id(gm, "q3 payroll export", sources={"hris": []})
    b = add_id(gm, "payroll summary", derived_from=[s])
    c = add_id(gm, "payroll digest", derived_from=[b])
    fresh = GatedMemory(memory)  # new instance: index rebuilt from a scan of the store
    fresh.restrict_source("hris", ["hr.salary"])
    for mid in (s, b, c):
        assert fresh.get(mid, entitlements=MKT) is None


def test_delete_removes_from_index(gm):
    s = add_id(gm, "q3 payroll export", sources={"hris": []})
    b = add_id(gm, "payroll summary", derived_from=[s])
    gm.restrict_source("unrelated", ["x"])  # builds the index before the delete
    gm.delete(b)
    res = gm.restrict_source("hris", ["hr.salary"])
    assert res.updated == (s,)
    assert b not in res.updated and b not in res.skipped
    remaining = gm.get_all(filters=SCOPE, entitlements=HR)["results"]
    assert b not in [r["id"] for r in remaining]


def test_direct_mem0_update_is_still_ungated(gm):
    # Documented boundary, pinned on purpose: calls made directly on gm.memory are NOT gated.
    # mem0's own update() keeps the old "amu" record, so rewritten content keeps public lineage.
    mid = add_id(gm, "office closed on friday", sources={"intranet": []})
    gm.memory.update(mid, text="q3 payroll total is 4.2m")
    leaked = gm.get(mid, entitlements=set())
    assert leaked is not None and leaked["memory"] == "q3 payroll total is 4.2m"


def test_update_preserves_user_metadata(gm):
    mid = add_id(gm, "tagged note", sources={}, metadata={"team": "finance"})
    gm.update(mid, "tagged note, revised", sources={}, metadata={"priority": "high"})
    md = gm.memory.get(mid)["metadata"]
    assert md["team"] == "finance" and md["priority"] == "high"


def test_update_rejects_reserved_key(gm):
    mid = add_id(gm, "note", sources={})
    with pytest.raises(ValueError, match="reserved"):
        gm.update(mid, "x", sources={}, metadata={METADATA_KEY: {}})


def test_index_scan_limit_raises_instead_of_partial_cascade(memory):
    gm = GatedMemory(memory, max_index_scan=2)
    for i in range(3):
        add_id(gm, f"record {i}", sources={"hris": []})
    with pytest.raises(IndexScanLimitError):
        gm.restrict_source("hris", ["hr.salary"])


def test_restrict_source_reaches_descendants_written_before_the_source(gm):
    # c is derived from p before p gains the "hris" source, so c's stored closure lacks it.
    # The cascade still reaches c through the parent->children walk (deliberate over-approximation).
    p = add_id(gm, "spend outlook", sources={"wiki": []})
    c = add_id(gm, "spend outlook summary", derived_from=[p])
    gm.update(p, derived_from=[add_id(gm, "q3 payroll export", sources={"hris": []})])
    assert "hris" not in _amu(gm, c)["closed"]
    res = gm.restrict_source("hris", ["hr.salary"])
    assert c in res.updated
    assert gm.get(c, entitlements=MKT) is None
