#!/usr/bin/env python
"""
Grow protein-viz-agent/rag_examples.json from real use (suggestion.txt S11).

Two sources of (prompt, tool trace) candidates:

  feedback   protein-viz-agent/feedback/rag_candidates.jsonl — one line per
             👍 / 👎 click in app.py's chat (latest rating per reply wins;
             only 👍 is used). PARORA_FEEDBACK_DIR moves it.
  reports    --from-report evals/reports/<new>/report.json: one-turn cases that
             pass there and checked tools or asking (a say-only pass vouches
             for the reply, not the calls). With --compare <old>/report.json, only the ones that
             failed in <old> — "failed S1 cases become examples once fixed".

Each candidate's grounding is written from its trace — the calls that ran, in
order, with their final (gate-rewritten, S8-recovered) args:
"Call select(name='lig', expression='hetero'), then color(...)". Rejected,
with the reason printed:
  - a clarified request (" — clarification: ") — the example would show the
    model a choice the prompt alone does not make
  - nothing ran, a call that ran failed, or S8 recovery rewrote one
  - a string arg carries a number or PDB id the prompt does not (from the scene
    or an earlier turn; as an example it would leak into other prompts)
  - more than MAX_CALLS calls (a plan, not a pattern)
  - the same prompt, or a near-identical one (TF-IDF >= --dup) naming the same
    tools, is already in the store
  - it shifts retrieval the wrong way (--allow-shift skips this). Probes =
    every turn in evals/prompts.jsonl plus every stored prompt; with the
    candidate stored, a probe that
      gains an example naming a tool the probe never needed — the tools a
      passing run of it used (--baseline reports; default the newest), or
      for a stored prompt the tools its grounding names — or
      loses a hand-written example (no "source" key: those carry reply
      guidance a trace cannot; an exact match drops everything under half
      its score) that is guidance only (names no call) or names a tool the
      probe needs — any added document moves every IDF, so other losses
      happen all the time and are not counted
    rejects it. A probe with no passing run counts any gain. Order is not
    checked. The 2026-10-22 merge test lost 3 cases to exactly these
    (describe → describe_fold; residue examples crowding out the ask
    guidance; an exact match dropping "say what tools cannot show").
A near-identical prompt naming *different* tools is a conflict: shown, never
merged automatically.

Stdlib only, no model, no Ollama:

    python evals/curate_examples.py                    # dry run: what would merge
    python evals/curate_examples.py --merge            # append to rag_examples.json
    python evals/curate_examples.py --review --merge   # y / n / e(dit) each one
    python evals/curate_examples.py --from-report evals/reports/<new>/report.json \\
        --compare evals/reports/<old>/report.json --merge

A rejected-in-review candidate is remembered in feedback/rejected.json and not
offered again. app.py reads the store once per process: restart it to use new
examples. Re-run evals/run.py after merging — an example changes routing (S10
ranks tools by example prompts too).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "protein-viz-agent"
sys.path.insert(0, str(AGENT))
from rag_grounding import DocRanker, _TfidfIndex   # noqa: E402

STORE = AGENT / "rag_examples.json"
CASES = ROOT / "evals" / "prompts.jsonl"
FEEDBACK_DIR = Path(os.environ.get("PARORA_FEEDBACK_DIR", AGENT / "feedback"))
CANDIDATES = FEEDBACK_DIR / "rag_candidates.jsonl"
REJECTED = FEEDBACK_DIR / "rejected.json"

MAX_CALLS = 6
SKIP_TOOLS = {"list_more_tools"}          # retrieval plumbing, not the answer
_NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?(?![A-Za-z0-9])")
_PDB_ID = re.compile(r"(?<![A-Za-z0-9])\d[A-Za-z0-9]{3}(?![A-Za-z0-9])")
_CALLED = re.compile(r"\b([a-z_][a-z0-9_]*)\(")


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


# ── Candidates ───────────────────────────────────────────────────────────────

def from_feedback() -> list:
    if not CANDIDATES.exists():
        return []
    latest = {}
    for line in CANDIDATES.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        latest[row["id"]] = row
    return [{"id": f"fb:{r['id']}", "prompt": r.get("request") or r["typed"],
             "trace": r.get("trace", []), "source": "feedback"}
            for r in latest.values() if r.get("rating") == "up"]


def from_report(path: Path, old: Path = None) -> list:
    results = json.loads(path.read_text())["results"]
    failed_before = None
    if old:
        failed_before = {r["case"] for r in json.loads(old.read_text())["results"]
                         if not r["passed"]}
    out = []
    for r in results:
        # A later turn leans on the turns before it; only one-turn cases stand alone.
        if not r["passed"] or len(r["turns"]) != 1:
            continue
        if failed_before is not None and r["case"] not in failed_before:
            continue
        # A pass vouches for the calls only when the case checked them; a
        # say/not_say-only pass (most recovery cases) is about the reply.
        if not any(ch["check"].startswith("tools") or ch["check"] == "ask"
                   for ch in r["checks"]):
            continue
        t = r["turns"][0]
        out.append({"id": f"eval:{r['case']}", "prompt": t["prompt"],
                    "trace": t["trace"], "source": f"eval:{r['case']}"})
    return out


def eval_prompts() -> list:
    """Every turn's prompt in the eval cases, in file order, once each."""
    out = []
    for line in CASES.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        for t in json.loads(line)["turns"]:
            out.append(t["prompt"] if isinstance(t, dict) else t)
    return list(dict.fromkeys(out))


# ── Retrieval shift ──────────────────────────────────────────────────────────

def _retrieved(store: list, probes: list) -> dict:
    # Same index, k and floors as format_grounding() in app.py.
    index = _TfidfIndex(store)
    return {p: index.query(p, 4) for p in probes}


def good_tools(reports: list, store: list) -> dict:
    """Probe prompt → tools a passing eval run used for it (union over the
    reports), or the tools a stored prompt's own grounding names."""
    good = {}
    for path in reports:
        for r in json.loads(Path(path).read_text())["results"]:
            if r["passed"]:
                for t in r["turns"]:
                    good.setdefault(t["prompt"], set()).update(
                        c["tool"] for c in t["trace"] if c.get("status") == "ran")
    for ex in store:
        if "source" not in ex:
            good.setdefault(ex["prompt"], set()).update(_CALLED.findall(ex["grounding"]))
    return good


def shifted(before: dict, store: list, good: dict) -> list:
    """Probes the candidate (last in `store`) would steer wrong."""
    after = _retrieved(store, list(before))
    out = []
    for p, old in before.items():
        old_p = [ex["prompt"] for ex in old]
        new_p = [ex["prompt"] for ex in after[p]]
        need = good.get(p)
        # Adding a document moves every IDF, so unrelated hand-written examples
        # slip under the floor all the time; a loss matters when the example
        # was pure guidance (no call) or about a tool this prompt needs.
        lost = [ex for ex in old if ex["prompt"] not in new_p and "source" not in ex
                and (need is None or not set(_CALLED.findall(ex["grounding"]))
                     or set(_CALLED.findall(ex["grounding"])) & need)]
        if lost:
            out.append(f"{p[:50]!r} loses {lost[0]['prompt'][:40]!r}")
            continue
        for ex in after[p]:
            if ex["prompt"] in old_p:
                continue
            extra = set(_CALLED.findall(ex["grounding"])) - (need or set())
            if extra or need is None:
                out.append(f"{p[:50]!r} gains {ex['prompt'][:30]!r}"
                           + (f" ({', '.join(sorted(extra))})" if extra else " (no passing run)"))
                break
    return out


# ── Grounding ────────────────────────────────────────────────────────────────

def _call_text(tool: str, args: dict) -> str:
    parts = []
    for k, v in args.items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        if isinstance(v, str) and len(v) > 80:
            v = v[:77] + "..."
        parts.append(f"{k}={v!r}")
    return f"{tool}({', '.join(parts)})"


def ground(cand: dict):
    """(grounding, tools) or (None, reason)."""
    prompt = cand["prompt"]
    if " — clarification: " in prompt:
        return None, "clarified request"
    calls = [c for c in cand["trace"] if c.get("status") == "ran" and c["tool"] not in SKIP_TOOLS]
    if not calls:
        return None, "no tool ran"
    bad = [c["tool"] for c in calls if c.get("ok") is False or c.get("failed")]
    if bad:
        return None, f"failed call: {', '.join(bad)}"
    # S8 fixed the model's args from the loaded file (erlotinib → AQ4): the
    # prompt alone does not lead there.
    fixed = [c["tool"] for c in calls if c.get("recovery")]
    if fixed:
        return None, f"recovered call: {', '.join(fixed)}"
    said = set(_NUMBER.findall(prompt))
    ids_said = {i.upper() for i in _PDB_ID.findall(prompt)}
    for c in calls:
        for k, v in (c.get("args") or {}).items():
            # Strings anywhere inside (ask_user options are a list); a bare
            # number arg (radius=4) is a setting, not a residue.
            text = v if isinstance(v, str) else json.dumps(v) if isinstance(v, (list, dict)) else ""
            extra = (set(_NUMBER.findall(text)) - said) | (
                {i.upper() for i in _PDB_ID.findall(text)} - ids_said)
            if extra:
                return None, f"{c['tool']}.{k} has {', '.join(sorted(extra))}, not in the prompt"
    steps = []
    for c in calls:                       # one line per distinct call
        text = _call_text(c["tool"], c.get("args") or {})
        if not steps or steps[-1] != text:
            steps.append(text)
    if len(steps) > MAX_CALLS:
        return None, f"{len(steps)} calls (plan, not a pattern)"
    grounding = "Call " + ", then ".join(steps) + "."
    if said:
        grounding += " Use the exact numbers from this prompt."
    return grounding, {c["tool"] for c in calls}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--merge", action="store_true", help="write accepted examples to the store")
    ap.add_argument("--review", action="store_true", help="ask y / n / e for each candidate")
    ap.add_argument("--from-report", type=Path, help="evals report.json to harvest")
    ap.add_argument("--compare", type=Path, help="older report.json: keep only cases fixed since")
    ap.add_argument("--no-feedback", action="store_true", help="ignore the 👍 candidates file")
    ap.add_argument("--store", type=Path, default=STORE, help="example store to read and merge into")
    ap.add_argument("--baseline", type=Path, action="append", default=[],
                    help="passing report.json whose traces say which tools each eval prompt "
                         "needs (repeatable; default: the newest report)")
    ap.add_argument("--allow-shift", action="store_true",
                    help="merge even when a candidate changes another prompt's retrieved examples")
    ap.add_argument("--dup", type=float, default=0.8,
                    help="TF-IDF similarity at which a prompt counts as already in the store")
    a = ap.parse_args()

    cands = [] if a.no_feedback else from_feedback()
    if a.from_report:
        cands += from_report(a.from_report, a.compare)
        if not a.compare:
            print("note: no --compare, so every passing case is a candidate, not only fixed ones")
    store = json.loads(a.store.read_text(encoding="utf-8"))
    rejected = set(json.loads(REJECTED.read_text())) if REJECTED.exists() else set()
    print(f"{len(cands)} candidates, {len(store)} examples in {a.store.name}")

    probes = list(dict.fromkeys(eval_prompts() + [ex["prompt"] for ex in store]))
    retrieved = _retrieved(store, probes)
    baselines = a.baseline or [p for p in sorted((ROOT / "evals" / "reports").glob("2*/report.json"))[-1:]]
    good = good_tools(baselines + ([a.from_report] if a.from_report else []), store)
    print("tools per eval prompt from: " + ", ".join(str(Path(b).parent.name) for b in baselines))
    accepted, newly_rejected = [], []
    for cand in cands:
        if cand["id"] in rejected:
            continue
        label = f"  {cand['prompt'][:70]!r}"
        grounding, tools = ground(cand)
        if grounding is None:
            print(f"skip {label} — {tools}")
            continue
        pool = store + accepted
        seen = {_norm(ex["prompt"]) for ex in pool}
        if _norm(cand["prompt"]) in seen:
            print(f"dup  {label} — same prompt already stored")
            continue
        sims = DocRanker([(i, ex["prompt"]) for i, ex in enumerate(pool)]).scores(cand["prompt"])
        near = [(s, pool[i]) for i, s in sims.items() if s >= a.dup]
        if any(set(_CALLED.findall(ex["grounding"])) >= tools for _, ex in near):
            print(f"dup  {label} — near-identical example names the same tools")
            continue
        if near:
            s, ex = max(near, key=lambda p: p[0])
            print(f"CONFLICT {label}\n     → {grounding}\n     stored ({s:.2f}): "
                  f"{ex['prompt']!r} → {ex['grounding']}\n     not merged; fix by hand if the new one is right")
            continue
        moved = [] if a.allow_shift else shifted(
            retrieved, pool + [{"prompt": cand["prompt"], "grounding": grounding, "source": cand["source"]}], good)
        if moved:
            print(f"shift{label} — changes retrieval for {len(moved)} prompt(s), e.g. {moved[0]}")
            continue
        print(f"new  {label}\n     → {grounding}")
        if a.review:
            ans = input("     keep? [y/n/e] ").strip().lower()
            if ans.startswith("e"):
                grounding = input("     grounding: ").strip() or grounding
            elif not ans.startswith("y"):
                newly_rejected.append(cand["id"])
                continue
        accepted.append({"prompt": cand["prompt"], "grounding": grounding,
                         "source": cand["source"], "added": time.strftime("%Y-%m-%d")})
        retrieved = _retrieved(store + accepted, probes)

    print(f"\n{len(accepted)} new example(s)")
    if newly_rejected:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        REJECTED.write_text(json.dumps(sorted(rejected | set(newly_rejected)), indent=1))
    if not a.merge:
        if accepted:
            print("dry run — add --merge to write them")
        return
    if accepted:
        a.store.write_text(json.dumps(store + accepted, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
        print(f"merged into {a.store} — restart app.py, then re-run evals/run.py")


if __name__ == "__main__":
    main()
