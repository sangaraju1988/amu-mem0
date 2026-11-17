"""Revocation-after-write fixture (White Paper 2, H1).

Spec from mem0 Discussion #7390. amu-mem0 has no permission cache: every read
re-evaluates the requester's current entitlements, so "revocation" here is
allow -> change entitlements -> re-read, on all three read paths.

Run one case:  uv run pytest tests/test_revocation.py -k <test name>
Pins (uv.lock): mem0ai 2.2.1, amu-governance 0.1.2, amu-mem0 0.2.0, qdrant-client 1.20.0.
"""

import logging

import pytest

from amu_mem0 import INCOMPLETE_LINEAGE, METADATA_KEY, UNENTITLED, Decision, GatedMemory, IndexScanLimitError

from .conftest import SCOPE, add_id, ids

HR = {"hr.salary"}
MKT = {"mkt.campaign"}


def visible(gm, mid, entitlements, query):
    """Whether ``mid`` is returned on each read path: (search, get, get_all)."""
    return (
        mid in ids(gm.search(query, filters=SCOPE, entitlements=entitlements)),
        gm.get(mid, entitlements=entitlements) is not None,
        mid in ids(gm.get_all(filters=SCOPE, entitlements=entitlements)),
    )


ALL = (True, True, True)
NONE = (False, False, False)


def mixed_b(gm):
    """A: restricted source (hr.salary). pub: unrestricted source. B derives from both."""
    a = add_id(gm, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary"]})
    pub = add_id(gm, "q3 has thirteen weeks", sources={"calendar": []})
    b = add_id(gm, "weekly payroll digest", derived_from=[a, pub])
    return a, pub, b


# --------------------------------------------------------------- revocation


def test_revocation_after_write_mixed_sources_all_paths(gm):
    _, pub, b = mixed_b(gm)
    requester = {"hr.salary"}
    assert visible(gm, b, requester, "weekly payroll digest") == ALL  # phase 1: entitled

    requester.discard("hr.salary")  # revoke; no cache exists, the next read re-evaluates

    assert visible(gm, b, requester, "weekly payroll digest") == NONE  # phase 2: denied
    # Control: the requester is not locked out; only B's closure is now forbidden.
    assert visible(gm, pub, requester, "thirteen weeks") == ALL


def test_grant_then_revoke_mixed_sources_all_paths(gm):
    _, _, b = mixed_b(gm)
    requester: set[str] = set()
    assert visible(gm, b, requester, "weekly payroll digest") == NONE
    requester.add("hr.salary")  # grant
    assert visible(gm, b, requester, "weekly payroll digest") == ALL
    requester.discard("hr.salary")  # revoke
    assert visible(gm, b, requester, "weekly payroll digest") == NONE


def test_revocation_is_per_requester(gm):
    _, _, b = mixed_b(gm)
    requester_1, requester_2 = {"hr.salary"}, {"hr.salary"}
    requester_1.discard("hr.salary")
    assert visible(gm, b, requester_1, "weekly payroll digest") == NONE
    assert visible(gm, b, requester_2, "weekly payroll digest") == ALL


# ------------------------------------------- ancestor deleted (post-write)


def test_delete_ancestor_denies_descendant_to_entitled_requester(gm):
    a, _, b = mixed_b(gm)
    assert visible(gm, b, HR, "weekly payroll digest") == ALL
    gm.delete(a)
    # The full source closure can no longer be proven, so B is unverified: denied even to HR.
    assert visible(gm, b, HR, "weekly payroll digest") == NONE
    assert gm.check(gm.memory.get(b), entitlements=HR).reason == INCOMPLETE_LINEAGE


def test_delete_ancestor_marks_descendants_incomplete_naming_ancestor(gm):
    a, _, b = mixed_b(gm)
    c = add_id(gm, "payroll digest briefing", derived_from=[b])  # grandchild of A
    gm.delete(a)
    for mid in (b, c):
        rec = gm.memory.get(mid)["metadata"][METADATA_KEY]
        assert rec["complete"] is False
        assert f"ancestor {a} deleted" in rec["incomplete_reason"]
        assert rec["closed"]["hris"] == ["hr.salary"]  # known labels are kept, not dropped


def test_delete_without_descendants_changes_nothing_else(gm):
    a, pub, b = mixed_b(gm)
    leaf = add_id(gm, "unrelated office note", sources={"wiki": []})
    before = {mid: gm.memory.get(mid)["metadata"][METADATA_KEY] for mid in (a, pub, b)}
    gm.delete(leaf)
    after = {mid: gm.memory.get(mid)["metadata"][METADATA_KEY] for mid in (a, pub, b)}
    assert after == before
    assert visible(gm, b, HR, "weekly payroll digest") == ALL


def test_delete_refuses_over_partial_index_and_keeps_memory(memory):
    gm = GatedMemory(memory, max_index_scan=2)
    a, _, b = mixed_b(gm)
    with pytest.raises(IndexScanLimitError):
        gm.delete(a)
    assert gm.memory.get(a) is not None  # nothing was deleted
    assert gm.memory.get(b)["metadata"][METADATA_KEY]["complete"] is True


# ------------------------------- ancestor authority changed (no deletion)


def test_ancestor_authority_change_denies_descendant(gm):
    a = add_id(gm, "q3 payroll export", sources={"hris": []})  # unrestricted at write time
    pub = add_id(gm, "q3 has thirteen weeks", sources={"calendar": []})
    b = add_id(gm, "weekly payroll digest", derived_from=[a, pub])
    assert visible(gm, b, MKT, "weekly payroll digest") == ALL
    gm.restrict_source("hris", ["hr.salary"])  # A's authority changes; A still exists
    assert gm.memory.get(a) is not None
    assert visible(gm, b, MKT, "weekly payroll digest") == NONE
    assert visible(gm, b, HR, "weekly payroll digest") == ALL


# --------------------------------------------- public-response privacy


def test_public_response_reveals_no_denied_candidates(gm):
    secret = add_id(gm, "payroll secret 4815162342", sources={"hris": ["hr.salary"]})
    public = add_id(gm, "payroll policy faq", sources={"wiki": []})
    for res in (
        gm.search("payroll", filters=SCOPE, entitlements=MKT),
        gm.get_all(filters=SCOPE, entitlements=MKT),
    ):
        assert set(res) == {"results"}
        assert ids(res) == [public]
        dumped = repr(res)
        for needle in (secret, "4815162342", "payroll secret", "hr.salary", "hris"):
            assert needle not in dumped


def test_short_page_carries_no_denial_signal(gm):
    for i in range(10):
        add_id(gm, f"payroll record {i}", sources={"hris": ["hr.salary"]})
    public = {add_id(gm, f"payroll faq {i}", sources={"wiki": []}) for i in range(2)}
    res = gm.search("payroll", filters=SCOPE, entitlements=MKT, top_k=5)
    assert set(ids(res)) == public  # fewer than top_k ...
    assert set(res) == {"results"}  # ... and nothing says why or how many were withheld


def test_get_denied_and_missing_are_indistinguishable(gm):
    denied = add_id(gm, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary"]})
    missing = "00000000-0000-0000-0000-000000000000"
    assert gm.get(denied, entitlements=MKT) is None
    assert gm.get(missing, entitlements=MKT) is None


def test_denial_accounting_is_operator_side_only(gm, caplog):
    denied = {add_id(gm, f"payroll secret {i} 4815162342", sources={"hris": ["hr.salary"]}) for i in range(3)}
    with caplog.at_level(logging.INFO, logger="amu_mem0"):
        res = gm.search("payroll secret", filters=SCOPE, entitlements=MKT)
    assert res == {"results": []}
    for mid in denied:  # operator-side: per-item decisions and the log
        assert gm.check(gm.memory.get(mid), entitlements=MKT) == Decision(mid, False, UNENTITLED, ("hr.salary",))
        assert f"id={mid} reason=unentitled unentitled=hr.salary" in caplog.text
    assert "4815162342" not in caplog.text


def assert_no_lineage(item, *hidden_ids):
    """A public result carries no lineage record and names none of ``hidden_ids``."""
    assert METADATA_KEY not in (item.get("metadata") or {})
    for mid in hidden_ids:
        assert mid not in repr(item)


def test_public_results_carry_no_lineage_record(gm):
    a, pub, b = mixed_b(gm)
    tagged = add_id(gm, "tagged weekly note", sources={"calendar": []}, metadata={"team": "finance"})
    found = gm.search("weekly payroll digest", filters=SCOPE, entitlements=HR)["results"]
    listed = gm.get_all(filters=SCOPE, entitlements=HR)["results"]
    b_items = [r for r in found + listed if r["id"] == b] + [gm.get(b, entitlements=HR)]
    assert len(b_items) == 3  # B came back on all three paths ...
    for item in b_items:
        assert_no_lineage(item, a, pub)  # ... without its lineage or its parents' ids
    for item in found + listed:
        assert METADATA_KEY not in (item.get("metadata") or {})
    assert gm.get(tagged, entitlements=HR)["metadata"] == {"team": "finance"}  # user metadata kept
    # The stored record is untouched: lineage is still there for the gate and operators.
    assert gm.memory.get(b)["metadata"][METADATA_KEY]["derived_from"] == [a, pub]


def test_child_of_restricted_parent_reveals_neither_lineage_nor_parent(gm):
    parent = add_id(gm, "q3 payroll total is 4.2m", sources={"hris": ["hr.salary"]})
    child = add_id(gm, "payroll grew eight percent", derived_from=[parent])
    for ents in (MKT, HR):  # denied to MKT; allowed to HR, still without lineage
        responses = [
            gm.search("payroll grew", filters=SCOPE, entitlements=ents),
            gm.get_all(filters=SCOPE, entitlements=ents),
            {"results": [r for r in [gm.get(child, entitlements=ents)] if r]},
        ]
        for res in responses:
            for item in res["results"]:
                assert_no_lineage(item)
            if ents == MKT:
                assert child not in repr(res) and parent not in repr(res)
            else:
                child_items = [r for r in res["results"] if r["id"] == child]
                assert child_items and all(parent not in repr(r) for r in child_items)


# ------------------------------------------------------------ pagination


def ranked_public(gm, n):
    """n public memories with distinct, decreasing scores for the query "payroll payroll"."""
    return [add_id(gm, "payroll " + " ".join(f"faq{j}" for j in range(i + 1)), sources={"wiki": []}) for i in range(n)]


def test_refill_when_denied_candidates_consume_initial_top_k(gm):
    for i in range(40):
        add_id(gm, f"payroll payroll payroll record {i}", sources={"hris": ["hr.salary"]})
    public = ranked_public(gm, 12)
    # Precondition: mem0's own top-5 is entirely restricted memories.
    raw = gm.memory.search("payroll payroll", filters=SCOPE, top_k=5)["results"]
    assert len(raw) == 5 and not set(ids({"results": raw})) & set(public)

    page = gm.search("payroll payroll", filters=SCOPE, entitlements=MKT, top_k=5)["results"]
    # Ordering verified empirically (2026-11-17, 5 fresh stores): with the deterministic
    # embedder, gated results keep mem0's score order and repeated calls are identical.
    assert [r["id"] for r in page] == public[:5]
    scores = [r["score"] for r in page]
    assert scores == sorted(scores, reverse=True)
    assert all(gm.check(gm.memory.get(r["id"]), entitlements=MKT).allowed for r in page)  # stored record


def test_pages_by_deeper_fetch_are_stable_without_duplicates(gm):
    # Paging mechanism, verified 2026-11-17: mem0 2.2.1's search/get_all take no offset
    # or cursor (unknown kwargs are silently ignored), and the gate adds none. The only
    # supported way to page is a deeper fetch: page n = results[(n-1)*k : n*k] of top_k=n*k.
    for i in range(40):
        add_id(gm, f"payroll payroll payroll record {i}", sources={"hris": ["hr.salary"]})
    public = ranked_public(gm, 12)

    def search(k):
        return [r["id"] for r in gm.search("payroll payroll", filters=SCOPE, entitlements=MKT, top_k=k)["results"]]

    page_1 = search(5)
    deeper = search(10)
    page_2 = deeper[5:10]
    assert search(5) == page_1  # repeatable
    assert deeper[:5] == page_1  # a deeper fetch extends page 1, it doesn't reorder it
    assert not set(page_1) & set(page_2)  # no duplicates across pages
    assert page_1 + page_2 == public[:10]


def test_redaction_does_not_mutate_what_mem0_returned(gm):
    _, _, b = mixed_b(gm)
    handed_back = []
    real_get, real_search = gm.memory.get, gm.memory.search

    def spy_get(*a, **kw):
        out = real_get(*a, **kw)
        handed_back.append(out)
        return out

    def spy_search(*a, **kw):
        out = real_search(*a, **kw)
        handed_back.extend(out["results"])
        return out

    gm.memory.get, gm.memory.search = spy_get, spy_search
    gm.get(b, entitlements=HR)
    gm.search("weekly payroll digest", filters=SCOPE, entitlements=HR)
    b_objects = [o for o in handed_back if o and o.get("id") == b]
    assert len(b_objects) >= 2
    for obj in b_objects:  # the exact dicts mem0 returned still hold their lineage
        assert METADATA_KEY in obj["metadata"]
