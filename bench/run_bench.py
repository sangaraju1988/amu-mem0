"""Synthetic, deterministic leak/recall/latency benchmark for amu-mem0.

Run with ``make bench``. Writes ``results/BENCH.md`` and ``results/bench.json``.

Setup: two teams (Finance, Marketing) share ONE mem0 scope
(``agent_id="analytics-agent"``). Raw source records carry sensitivity labels
from their source system. An agent then writes derived memories (summaries,
cross-topic aggregates, 2-hop and 3-hop summaries) that declare only
``derived_from`` -- the ids of the memories they were written from.

Conditions, all over the same mem0 store and the same queries:
  plain         mem0 ``search`` as-is.
  content_only  drop a result if its OWN direct source labels are not covered by
                the requester's entitlements (tag/content gating; no lineage).
  gate          ``GatedMemory.search`` (closed lineage, fail-closed).

Ground truth for every memory is the true closed label set computed by this
script from the generator, independent of what was recorded in mem0.

  leak            a returned item whose TRUE closed labels are not all held by
                  the requester.
  recall@k        |returned ∩ ideal| / |ideal|, where ideal = the top-k of mem0's
                  own ranking restricted to items the requester is truly
                  entitled to (an oracle filter over a deep plain search).
  latency         wall-clock per search call, in-process, local Qdrant.

Coverage sweep: each derived_from edge is recorded with probability c. Dropped
edges are silent (if all are dropped the writer declares ``sources={}``), which
models incomplete lineage the gate cannot detect.
"""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "False")
# String hashing is randomized per process by default, which changes set/dict
# iteration order inside mem0 and qdrant and therefore how score ties are broken.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import logging  # noqa: E402

from amu_mem0 import GatedMemory  # noqa: E402
from amu_mem0.testing import offline_memory  # noqa: E402


class _SeededUUID:
    """Deterministic uuid4 for mem0's memory ids (bench only).

    mem0 breaks score ties in an order that depends on memory ids; seeding them
    makes every count in BENCH.md reproducible run to run. ``uuid.UUID`` is
    unchanged, so mem0 still sees valid UUIDs.
    """

    UUID = uuid.UUID

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def uuid4(self) -> uuid.UUID:
        return uuid.UUID(int=self._rng.getrandbits(128), version=4)


def seed_mem0_ids(seed: int) -> None:
    import mem0.memory.main as mem0_main

    mem0_main.uuid = _SeededUUID(seed)


logging.getLogger("amu_mem0").setLevel(logging.WARNING)
logging.getLogger("mem0").setLevel(logging.ERROR)

SEED = 20261115
TOP_K = 10
SCOPE = {"agent_id": "analytics-agent"}
DIMS = 256
COVERAGES = [1.0, 0.9, 0.75, 0.5]

TEAMS = {
    "Finance": frozenset({"fin.payroll", "fin.revenue"}),
    "Marketing": frozenset({"mkt.campaign", "mkt.leads"}),
}

# topic -> (source system, labels, vocabulary)
TOPICS = {
    "payroll": ("hris", ("fin.payroll",), "payroll salary wages bonus compensation headcount"),
    "revenue": ("erp", ("fin.revenue",), "revenue bookings invoices billing arr receivables"),
    "forecast": ("erp", ("fin.revenue",), "forecast projection outlook pipeline quarter guidance"),
    "expenses": ("hris", ("fin.payroll",), "expenses reimbursement travel per-diem stipend allowance"),
    "campaign": ("ads", ("mkt.campaign",), "campaign ads impressions clicks creative spend"),
    "leads": ("crm", ("mkt.leads",), "leads prospects signups conversion funnel contacts"),
    "brand": ("ads", ("mkt.campaign",), "brand awareness sentiment survey recall reach"),
    "events": ("crm", ("mkt.leads",), "events webinar attendees booth registrations sponsors"),
    "holidays": ("wiki", (), "holidays calendar office closed observance schedule"),
    "onboarding": ("wiki", (), "onboarding laptop badge orientation checklist mentor"),
    "policies": ("wiki", (), "policy handbook guideline security training compliance"),
    "facilities": ("wiki", (), "facilities parking cafeteria desk booking meeting rooms"),
}

RAW_PER_TOPIC = 15
N_SUMMARY = 120
N_AGGREGATE = 60
N_HOP2 = 40
N_HOP3 = 20
QUERIES_PER_TOPIC_PER_TEAM = 10
DEEP_K = 500


@dataclass
class Item:
    key: int
    kind: str
    topics: tuple[str, ...]
    text: str
    sources: dict[str, tuple[str, ...]] | None  # direct sources (raw only)
    parents: tuple[int, ...] = ()
    true_closed: frozenset[str] = field(default_factory=frozenset)


def words(rng: random.Random, topic: str, n: int) -> list[str]:
    return rng.sample(TOPICS[topic][2].split(), n)


def generate(rng: random.Random) -> list[Item]:
    items: list[Item] = []

    def new(kind, topics, text, sources=None, parents=()):
        it = Item(len(items), kind, tuple(topics), text, sources, tuple(parents))
        labels = set(sources and next(iter(sources.values())) or ())
        for p in parents:
            labels |= items[p].true_closed
        it.true_closed = frozenset(labels)
        items.append(it)
        return it

    raw_by_topic: dict[str, list[int]] = {}
    for topic, (system, labels, _) in TOPICS.items():
        for i in range(RAW_PER_TOPIC):
            text = f"{' '.join(words(rng, topic, 3))} record {topic}{i} r{rng.randrange(10**6)}"
            raw_by_topic.setdefault(topic, []).append(new("raw", [topic], text, {system: labels}).key)

    topics = list(TOPICS)
    summaries_by_topic: dict[str, list[int]] = {}
    for i in range(N_SUMMARY):
        t = rng.choice(topics)
        parents = rng.sample(raw_by_topic[t], rng.randint(1, 3))
        text = f"summary {' '.join(words(rng, t, 2))} trend noted s{i} u{rng.randrange(10**6)}"
        summaries_by_topic.setdefault(t, []).append(new("summary", [t], text, parents=parents).key)
    for i in range(N_AGGREGATE):
        t1, t2 = rng.sample(topics, 2)
        parents = rng.sample(raw_by_topic[t1], 2) + rng.sample(raw_by_topic[t2], rng.randint(1, 2))
        w1, w2 = " ".join(words(rng, t1, 2)), " ".join(words(rng, t2, 2))
        text = f"aggregate {w1} with {w2} a{i} u{rng.randrange(10**6)}"
        new("aggregate", [t1, t2], text, parents=parents)
    hop2: list[int] = []
    for i in range(N_HOP2):
        t = rng.choice([t for t in topics if len(summaries_by_topic.get(t, [])) >= 2])
        parents = rng.sample(summaries_by_topic[t], 2)
        text = f"digest {' '.join(words(rng, t, 2))} overview h{i} u{rng.randrange(10**6)}"
        hop2.append(new("hop2", [t], text, parents=parents).key)
    for i in range(N_HOP3):
        p = rng.choice(hop2)
        t = items[p].topics[0]
        text = f"briefing {' '.join(words(rng, t, 2))} highlights b{i} u{rng.randrange(10**6)}"
        new("hop3", [t], text, parents=[p])
    return items


def build_store(items: list[Item], coverage: float, rng: random.Random, workdir: str):
    seed_mem0_ids(SEED + 3)
    memory = offline_memory(workdir, dims=DIMS)
    gm = GatedMemory(memory, max_fetch=DEEP_K)
    mem_id: dict[int, str] = {}
    key_of: dict[str, int] = {}
    edges_total = edges_kept = 0
    for it in items:
        if it.kind == "raw":
            res = gm.add(it.text, sources=it.sources, **SCOPE)
        else:
            kept = [p for p in it.parents if rng.random() < coverage]
            edges_total += len(it.parents)
            edges_kept += len(kept)
            if kept:
                res = gm.add(it.text, derived_from=[mem_id[p] for p in kept], **SCOPE)
            else:
                res = gm.add(it.text, sources={}, **SCOPE)  # silently claims no restricted inputs
        mid = res["results"][0]["id"]
        mem_id[it.key] = mid
        key_of[mid] = it.key
    return memory, gm, key_of, (edges_kept / edges_total if edges_total else 1.0)


def make_queries(rng: random.Random) -> list[tuple[str, str]]:
    qs = []
    for team in TEAMS:
        for topic in TOPICS:
            for _ in range(QUERIES_PER_TOPIC_PER_TEAM):
                qs.append((team, " ".join(words(rng, topic, 2))))
    rng.shuffle(qs)
    return qs


def own_labels(result: dict) -> set[str]:
    rec = (result.get("metadata") or {}).get("amu") or {}
    return {lab for labs in (rec.get("sources") or {}).values() for lab in labs}


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def evaluate(memory, gm, items, key_of, queries, with_latency: bool):
    conds = ["plain", "content_only", "gate"]
    stats = {
        c: {
            "returned": 0,
            "leaked": 0,
            "leaked_derived": 0,
            "queries_with_leak": 0,
            "recall": [],
            "lat": [],
            "queries_with_recall_loss": 0,
            "recall_loss_matched_by_leaks": 0,
            "short_queries": 0,
            "short_slots": 0,
            "short_all_entitled_returned": 0,
        }
        for c in conds
    }
    queries_without_entitled = 0
    for team, q in queries:
        ents = TEAMS[team]

        def truly_ok(mid: str, ents: frozenset[str] = ents) -> bool:
            return items[key_of[mid]].true_closed <= ents

        deep = memory.search(q, top_k=DEEP_K, filters=SCOPE)["results"]
        entitled = [r for r in deep if truly_ok(r["id"])]
        ideal = entitled[:TOP_K]
        # Tie-aware: every entitled item tied (in the oracle call's own scores) with the
        # k-th one is acceptable, since which tied item lands at rank k is arbitrary.
        # Scores are compared only within one call: mem0's hybrid scores depend on top_k.
        acceptable = {r["id"] for r in entitled if ideal and r["score"] >= ideal[-1]["score"] - 1e-9}
        queries_without_entitled += not ideal

        order = conds[:]
        random.Random(f"{team}|{q}").shuffle(order)  # interleave to avoid drift bias
        for c in order:
            t0 = time.perf_counter()
            if c == "plain":
                out = memory.search(q, top_k=TOP_K, filters=SCOPE)["results"]
            elif c == "content_only":
                fetch = TOP_K * 4
                out = []
                while True:
                    raw = memory.search(q, top_k=fetch, filters=SCOPE)["results"]
                    out = [r for r in raw if own_labels(r) <= ents]
                    if len(out) >= TOP_K or len(raw) < fetch or fetch >= DEEP_K:
                        break
                    fetch = min(fetch * 2, DEEP_K)
                out = out[:TOP_K]
            else:
                out = gm.search(q, top_k=TOP_K, filters=SCOPE, entitlements=ents)["results"]
            dt = time.perf_counter() - t0
            s = stats[c]
            if with_latency:
                s["lat"].append(dt * 1000)
            leaks = [r for r in out if not truly_ok(r["id"])]
            s["returned"] += len(out)
            s["leaked"] += len(leaks)
            s["leaked_derived"] += sum(1 for r in leaks if items[key_of[r["id"]]].kind != "raw")
            s["queries_with_leak"] += bool(leaks)
            if ideal:
                hits = min(sum(1 for r in out if r["id"] in acceptable), len(ideal))
                s["recall"].append(hits / len(ideal))
                missed = len(ideal) - hits
                if missed:
                    s["queries_with_recall_loss"] += 1
                    # Displacement: every missed entitled slot is taken by a restricted item.
                    s["recall_loss_matched_by_leaks"] += missed <= len(leaks)
            if len(out) < TOP_K:
                s["short_queries"] += 1
                s["short_slots"] += TOP_K - len(out)
                # Short only because fewer entitled items exist at any depth mem0 returns.
                s["short_all_entitled_returned"] += len(out) == len(entitled)
    summary = {}
    for c, s in stats.items():
        summary[c] = {
            "returned": s["returned"],
            "leaked": s["leaked"],
            "leaked_derived": s["leaked_derived"],
            "leak_rate": s["leaked"] / s["returned"] if s["returned"] else 0.0,
            "queries_with_leak": s["queries_with_leak"] / len(queries),
            "recall_at_k": statistics.mean(s["recall"]) if s["recall"] else float("nan"),
            "recall_queries": len(s["recall"]),
            "queries_without_entitled": queries_without_entitled,
            **{
                k: s[k]
                for k in (
                    "queries_with_recall_loss",
                    "recall_loss_matched_by_leaks",
                    "short_queries",
                    "short_slots",
                    "short_all_entitled_returned",
                )
            },
        }
        if with_latency:
            summary[c].update(p50_ms=statistics.median(s["lat"]), p95_ms=pct(s["lat"], 0.95))
    return summary


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return "unknown"

    return {
        "commit": run("rev-parse", "--short=12", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--", ".", ":!results")),
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    items = generate(random.Random(SEED))
    queries = make_queries(random.Random(SEED + 1))
    kinds = {k: sum(1 for i in items if i.kind == k) for k in ("raw", "summary", "aggregate", "hop2", "hop3")}
    restricted_derived = sum(1 for i in items if i.kind != "raw" and i.true_closed)

    results = {}
    for cov in COVERAGES:
        work = tempfile.mkdtemp(prefix="amu_mem0_bench_")
        try:
            memory, gm, key_of, realized = build_store(items, cov, random.Random(SEED + 2), work)
            # warm-up (not measured)
            for team, q in queries[:10]:
                memory.search(q, top_k=TOP_K, filters=SCOPE)
                gm.search(q, top_k=TOP_K, filters=SCOPE, entitlements=TEAMS[team])
            results[str(cov)] = {
                "realized_edge_coverage": realized,
                "metrics": evaluate(memory, gm, items, key_of, queries, with_latency=(cov == 1.0)),
            }
            memory.close()
        finally:
            shutil.rmtree(work, ignore_errors=True)
        print(f"coverage {cov}: done", file=sys.stderr)

    meta = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **git_info(),
        "seed": SEED,
        "top_k": TOP_K,
        "memories": len(items),
        "by_kind": kinds,
        "restricted_derived": restricted_derived,
        "queries": len(queries),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "versions": {p: version(p) for p in ("mem0ai", "amu-governance", "qdrant-client", "amu-mem0")},
        "spacy_installed": _has("spacy"),
    }
    out = {"meta": meta, "results": results}
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "bench.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (root / "results" / "BENCH.md").write_text(render(out))
    print((root / "results" / "BENCH.md").read_text())


def _has(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


LABEL = "Synthetic multi-team setup, deterministic embedder, 420 memories."


def render(out: dict) -> str:
    m = out["meta"]
    full = out["results"]["1.0"]["metrics"]
    plain, content, gate = full["plain"], full["content_only"], full["gate"]
    names = {"plain": "Plain mem0", "content_only": "Content-only filter (own labels)", "gate": "amu-mem0 gate"}
    lines = [
        "# amu-mem0 benchmark",
        "",
        f"> **{LABEL}** These numbers describe this benchmark only. They are not estimates of leak "
        "rates, recall or latency in any production deployment.",
        "",
        f"Generated by `make bench` at {m['utc']} from commit `{m['commit']}`"
        + (" (working tree dirty)" if m["dirty"] else "")
        + f". Seed {m['seed']}.",
        "",
        f"- Store: {m['memories']} memories in one shared mem0 scope "
        f"({', '.join(f'{v} {k}' for k, v in m['by_kind'].items())}); "
        f"{m['restricted_derived']} derived memories have at least one restricted source.",
        f"- Queries: {m['queries']} (2 teams x 12 topics x 10), top_k={m['top_k']}.",
        f"- Versions: {', '.join(f'{k} {v}' for k, v in m['versions'].items())}; Python {m['python']}, "
        f"{m['platform']}; spaCy installed: {m['spacy_installed']}.",
        "- Embeddings: deterministic hashed bag-of-words (offline). Vector store: local Qdrant. No LLM calls "
        "(`infer=False`). mem0's memory ids are seeded (bench only) so score-tie order is reproducible.",
        "",
        f"## Full lineage coverage ({LABEL[:-1].lower()})",
        "",
        "| Condition | Returned | Leaked | of which derived | Leak rate | Queries with a leak "
        "| Recall@10 (entitled) | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in ("plain", "content_only", "gate"):
        r = full[c]
        lines.append(
            f"| {names[c]} | {r['returned']} | {r['leaked']} | {r['leaked_derived']} | {r['leak_rate']:.1%} "
            f"| {r['queries_with_leak']:.1%} | {r['recall_at_k']:.3f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} |"
        )
    lines += [
        "",
        "### Definitions",
        "",
        "- **Leak:** a returned memory whose true transitive source labels (from the generator, not from what "
        "was recorded in mem0) are not all held by the requester. **Leak rate** = leaked / returned.",
        "- **Recall@10 (entitled):** for each query, run mem0 `search` with `top_k=500` and keep, in mem0's "
        "order, only memories the requester is truly entitled to; the ideal set is the first "
        "min(10, n) of these. A returned memory counts as a hit if it is in that entitled list with a score "
        "(in the same `top_k=500` call) at least that of the 10th ideal item, so ties at rank 10 are not "
        "penalised. Recall@10 = min(hits, |ideal|) / |ideal|, averaged over the "
        f"{gate['recall_queries']} queries with at least one entitled memory "
        f"({gate['queries_without_entitled']} queries had none and are excluded).",
        "",
        "### Reading the table",
        "",
        f"- **Content-only filtering removes raw restricted records but none of what was derived from them:** "
        f"all {content['leaked']} of its leaks are derived memories.",
        f"- **Plain mem0's lower recall is displacement.** It always returns 10 results, and restricted memories "
        f"take slots that entitled ones would otherwise fill. In {plain['recall_loss_matched_by_leaks']} of the "
        f"{plain['queries_with_recall_loss']} queries where plain mem0 misses entitled memories, every miss is "
        "matched by a restricted memory in its top 10. The remainder reflects mem0's ranking depending on "
        "`top_k`.",
        f"- **Different denominators.** Plain mem0 and the content-only filter return {plain['returned']} results "
        f"(10 per query); the gate returns {gate['returned']}. The gate came up short of 10 in "
        f"{gate['short_queries']} queries ({gate['short_slots']} empty slots). In "
        f"{gate['short_all_entitled_returned']} of those it returned every memory the requester is entitled to "
        "among all results mem0 returns for the query at any depth, i.e. fewer than 10 entitled memories pass "
        "mem0's score threshold. Leak rates are per returned item, so they use these different denominators.",
        f"- **Latency is tail overhead from over-fetch, at small scale.** Wall-clock per in-process `search` "
        f"call on {m['memories']} memories with local Qdrant. When most top candidates are restricted, the gate "
        "issues several underlying searches, which shows up in p95 rather than p50. Latency was not measured "
        "at production scale, with a remote vector store, or with real embeddings.",
        "",
        f"## Lineage coverage sweep, gate only ({LABEL[:-1].lower()})",
        "",
        "Each `derived_from` edge is recorded with probability c; dropped edges are silent (if all of a "
        "memory's edges are dropped, it is written with `sources={}`). Ground truth is unchanged.",
        "",
        "| Target c | Realized edge coverage | Queries with a leak | Leak rate | Leaked | Recall@10 |",
        "|---|---|---|---|---|---|",
    ]
    for cov, r in out["results"].items():
        g = r["metrics"]["gate"]
        lines.append(
            f"| {float(cov):.2f} | {r['realized_edge_coverage']:.1%} | {g['queries_with_leak']:.1%} "
            f"| {g['leak_rate']:.1%} | {g['leaked']} | {g['recall_at_k']:.3f} |"
        )
    lines += [
        "",
        "The paper (doi:10.1109/ACCESS.2026.3730363) recommends at least 90% lineage completeness, having measured "
        "0% leakage at 75-90% on its workloads. This is a different workload: whole derivation edges are dropped in "
        "a multi-hop graph rather than columns within a flat result, and here residual leakage remains at every "
        "coverage below 100%.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
