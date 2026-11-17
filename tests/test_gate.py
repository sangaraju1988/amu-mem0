"""Behavioural tests for GatedMemory against a real (offline) mem0 Memory."""

import logging

import pytest

from amu_mem0 import (
    INCOMPLETE_LINEAGE,
    METADATA_KEY,
    MISSING_LINEAGE,
    UNENTITLED,
    Decision,
    GatedMemory,
    InferNotAllowedError,
    LineageMissingWarning,
)

from .conftest import SCOPE, add_id, ids


def assert_reveals_nothing(res, memory_id, *secrets):
    """The public response carries only allowed results: no denial key, and the denied
    record's id, content and labels appear nowhere in the serialized response."""
    assert set(res) == {"results"}
    dumped = repr(res)
    for needle in (memory_id, *secrets):
        assert needle not in dumped


HR = {"hr.salary"}
MKT = {"mkt.campaign"}


def test_source_visible_only_to_entitled(gm):
    mid = add_id(gm, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary"]})
    assert ids(gm.search("payroll total", filters=SCOPE, entitlements=HR)) == [mid]
    res = gm.search("payroll total", filters=SCOPE, entitlements=MKT)
    assert res["results"] == []
    assert_reveals_nothing(res, mid, "q3 payroll total", "hr.salary")
    # Operator-side accounting stays available through check().
    assert gm.check(gm.memory.get(mid), entitlements=MKT) == Decision(mid, False, UNENTITLED, ("hr.salary",))


def test_public_source_visible_to_everyone(gm):
    mid = add_id(gm, "office closed on friday", sources={"intranet": []})
    assert ids(gm.search("office closed", filters=SCOPE, entitlements=set())) == [mid]


def test_derived_memory_inherits_source_restrictions(gm):
    src = add_id(gm, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary"]})
    summary = add_id(gm, "payroll cost grew eight percent", derived_from=[src])
    assert gm.get(summary, entitlements=MKT) is None
    assert gm.get(summary, entitlements=HR)["id"] == summary


def test_multi_hop_summary_of_summary(gm):
    src = add_id(gm, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary"]})
    s1 = add_id(gm, "payroll cost grew eight percent", derived_from=[src])
    s2 = add_id(gm, "headcount spend trend is upward", derived_from=[s1])
    s3 = add_id(gm, "quarterly spend outlook remains upward", derived_from=[s2])
    for mid in (s1, s2, s3):
        assert gm.get(mid, entitlements=MKT) is None
        assert gm.get(mid, entitlements=HR) is not None
    stored = gm.memory.get(s3)["metadata"][METADATA_KEY]
    assert stored["closed"] == {"hris": ["hr.salary"]}
    assert stored["derived_from"] == [s2]


def test_mixed_sources_require_every_label(gm):
    mid = add_id(gm, "campaign roi per salary band", sources={"hris": ["hr.salary"], "ads": ["mkt.campaign"]})
    assert gm.get(mid, entitlements=HR) is None
    assert gm.get(mid, entitlements=MKT) is None
    assert gm.get(mid, entitlements=HR | MKT) is not None


def test_mixed_parents_union_their_lineage(gm):
    a = add_id(gm, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary"]})
    b = add_id(gm, "q3 campaign budget is 1.1m", sources={"ads": ["mkt.campaign"]})
    pub = add_id(gm, "q3 has 13 weeks", sources={"calendar": []})
    agg = add_id(gm, "q3 total spend combined", derived_from=[a, b, pub])
    assert gm.get(agg, entitlements=HR) is None
    assert gm.get(agg, entitlements=MKT) is None
    assert gm.get(agg, entitlements=HR | MKT) is not None


def test_entitlement_changes_after_write_take_effect_immediately(gm):
    src = add_id(gm, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary"]})
    summary = add_id(gm, "payroll cost grew eight percent", derived_from=[src])
    entitlements = set(MKT)
    assert gm.get(summary, entitlements=entitlements) is None
    entitlements.add("hr.salary")  # granted
    assert gm.get(summary, entitlements=entitlements) is not None
    entitlements.discard("hr.salary")  # revoked
    assert gm.get(summary, entitlements=entitlements) is None


def test_sensitive_registry_limits_which_labels_gate(memory):
    gm = GatedMemory(memory, sensitive_labels={"hr.salary"})
    mid = add_id(gm, "office wifi password rotated", sources={"it": ["it.internal"]})
    # it.internal is not in the registry, so it does not gate (paper semantics).
    assert gm.get(mid, entitlements=set()) is not None
    sal = add_id(gm, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary", "it.internal"]})
    assert gm.get(sal, entitlements={"it.internal"}) is None


# ---------------------------------------------------------- missing lineage


def test_memory_written_outside_gate_is_denied_by_default(gm):
    raw = gm.memory.add("legacy note about payroll", infer=False, **SCOPE)["results"][0]["id"]
    res = gm.search("payroll", filters=SCOPE, entitlements=HR | MKT)
    assert raw not in ids(res)
    assert_reveals_nothing(res, raw, "legacy note")
    assert gm.check(gm.memory.get(raw), entitlements=HR | MKT) == Decision(raw, False, MISSING_LINEAGE)


def test_undeclared_lineage_is_incomplete_and_denied(gm):
    mid = add_id(gm, "agent wrote this without saying where from")
    assert gm.memory.get(mid)["metadata"][METADATA_KEY]["complete"] is False
    res = gm.get_all(filters=SCOPE, entitlements=HR | MKT)
    assert res["results"] == []
    assert_reveals_nothing(res, mid, "without saying where from")
    assert gm.check(gm.memory.get(mid), entitlements=HR | MKT).reason == INCOMPLETE_LINEAGE


def test_unresolved_parent_makes_child_incomplete(gm):
    mid = add_id(gm, "summary of a deleted thing", derived_from=["00000000-0000-0000-0000-000000000000"])
    rec = gm.memory.get(mid)["metadata"][METADATA_KEY]
    assert rec["complete"] is False and "not found" in rec["incomplete_reason"]
    assert gm.get(mid, entitlements=HR | MKT) is None


def test_child_of_incomplete_parent_is_incomplete_but_keeps_known_labels(memory):
    gm = GatedMemory(memory, on_missing_lineage="allow")
    parent = add_id(gm, "partial lineage parent", sources={"hris": ["hr.salary"]}, derived_from=["missing-id"])
    child = add_id(gm, "child of partial parent", derived_from=[parent])
    rec = gm.memory.get(child)["metadata"][METADATA_KEY]
    assert rec["complete"] is False
    # Even with on_missing_lineage="allow", a known forbidden label still denies.
    assert gm.get(child, entitlements=MKT) is None
    assert gm.get(child, entitlements=HR) is not None


def test_allow_and_allow_with_warning_modes(memory):
    raw = memory.add("legacy note", infer=False, **SCOPE)["results"][0]["id"]
    assert GatedMemory(memory, on_missing_lineage="allow").get(raw, entitlements=set()) is not None
    with pytest.warns(LineageMissingWarning):
        assert GatedMemory(memory, on_missing_lineage="allow_with_warning").get(raw, entitlements=set()) is not None


def test_malformed_lineage_is_denied(gm):
    raw = gm.memory.add("tampered", infer=False, metadata={"amu": {"v": 1, "closed": "oops"}}, **SCOPE)
    mid = raw["results"][0]["id"]
    assert gm.get(mid, entitlements=HR | MKT) is None


# ------------------------------------------------------------- write guards


def test_infer_true_is_rejected_by_default(gm):
    with pytest.raises(InferNotAllowedError):
        gm.add("anything", agent_id="a", sources={}, infer=True)


def test_reserved_metadata_key_rejected(gm):
    with pytest.raises(ValueError, match="reserved"):
        gm.add("x", agent_id="a", sources={}, metadata={"amu": {}})


def test_entitlements_are_required_on_reads(gm):
    with pytest.raises(TypeError):
        gm.search("x", filters=SCOPE)  # type: ignore[call-arg]


def test_user_metadata_is_preserved(gm):
    mid = add_id(gm, "tagged note", sources={}, metadata={"team": "finance"})
    assert gm.get(mid, entitlements=set())["metadata"]["team"] == "finance"


# ------------------------------------------------------- recall & overhead


def test_over_fetch_keeps_top_k_for_unentitled_requester(gm):
    # 40 restricted memories outrank 5 public ones for this query.
    for i in range(40):
        add_id(gm, f"payroll payroll payroll record {i}", sources={"hris": ["hr.salary"]})
    public = {add_id(gm, f"payroll policy faq {i}", sources={"wiki": []}) for i in range(5)}
    res = gm.search("payroll payroll", filters=SCOPE, entitlements=MKT, top_k=5)
    assert set(ids(res)) == public


def test_underlying_calls_are_bounded(memory):
    calls = []
    real_search = memory.search

    def counting_search(*a, **kw):
        calls.append(kw["top_k"])
        return real_search(*a, **kw)

    memory.search = counting_search
    gm = GatedMemory(memory, fetch_multiplier=2, max_fetch=40)
    for i in range(30):
        add_id(gm, f"payroll record {i}", sources={"hris": ["hr.salary"]})
    gm.search("payroll", filters=SCOPE, entitlements=MKT, top_k=5)
    assert calls == [10, 20, 40]  # doubles until max_fetch, then stops


def test_short_page_from_mem0_does_not_end_over_fetch(memory):
    # mem0 may return fewer than top_k results while more exist at a larger top_k
    # (score threshold + top_k-sized candidate pool). Simulate a short first page.
    real_search = memory.search
    calls = []

    def short_first_page(*a, **kw):
        calls.append(kw["top_k"])
        out = real_search(*a, **kw)
        if len(calls) == 1:
            out["results"] = out["results"][: kw["top_k"] - 1]
        return out

    memory.search = short_first_page
    gm = GatedMemory(memory, fetch_multiplier=1, max_fetch=64)
    for i in range(10):
        add_id(gm, f"payroll payroll record {i}", sources={"hris": ["hr.salary"]})
    public = {add_id(gm, f"payroll faq {i}", sources={"wiki": []}) for i in range(3)}
    res = gm.search("payroll payroll", filters=SCOPE, entitlements=MKT, top_k=3)
    assert set(ids(res)) == public
    assert len(calls) > 1


def test_over_fetch_stops_when_no_new_candidates(memory):
    calls = []
    real_search = memory.search

    def counting_search(*a, **kw):
        calls.append(kw["top_k"])
        return real_search(*a, **kw)

    memory.search = counting_search
    gm = GatedMemory(memory, fetch_multiplier=1, max_fetch=1000)
    for i in range(6):
        add_id(gm, f"payroll record {i}", sources={"hris": ["hr.salary"]})
    gm.search("payroll", filters=SCOPE, entitlements=MKT, top_k=5)
    assert calls == [5, 10, 20]  # 5 -> 6 -> 6 results: stalled, stop well before max_fetch


def test_denial_log_has_no_memory_content(gm, caplog):
    add_id(gm, "the secret number is 4815162342", sources={"hris": ["hr.salary"]})
    with caplog.at_level(logging.INFO, logger="amu_mem0"):
        gm.search("secret number", filters=SCOPE, entitlements=MKT)
    assert "reason=unentitled" in caplog.text
    assert "4815162342" not in caplog.text
