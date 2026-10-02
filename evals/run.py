#!/usr/bin/env python
"""
PARORA agent evaluation harness (suggestion.txt S1).

Runs app.py's real run_agent() headless — Streamlit in bare mode, so
st.session_state is a plain in-process store and nothing opens a browser —
against the cases in evals/prompts.jsonl, scores each one from the tool-call
trace run_agent records (st.session_state.agent_trace) plus the final reply,
prints a pass rate per category and per TOOL_GROUPS group, and saves a JSON
report to diff between runs.

Needs the parora conda env and a running Ollama with the model pulled:

    ~/miniconda3/envs/parora/bin/python evals/run.py
    ... evals/run.py --only ambiguity,destructive      # categories or ids
    ... evals/run.py --model qwen2.5:14b               # benchmark another model
    ... evals/run.py --model qwen3:14b --think false   # reasoning model, thinking off
    ... evals/run.py --repeat 3                        # sampling noise
    ... evals/run.py --compare evals/reports/<old>/report.json
    ... evals/run.py --list

Case format (one JSON object per line; '#'-lines and blank lines ignored):

    id        unique slug
    category  lookup | multistep | ambiguity | destructive | recovery |
              guardrail | multiturn
    groups    TOOL_GROUPS names the case exercises (coverage report)
    source    "log" (seeded from logs/parora.log) or "synthetic"
    requires  optional: network, fpocket, foldseek, esm, dssp, pymol, amber;
              "!name" = only when that backend is absent (e.g. "!amber" for
              the "AmberTools not installed" path)
    preload   optional list run before the first turn, without the model:
              "1CRN" (fetch_structure) or {"tool": name, "args": {...}}
    turns     list of prompts; an item may be {"prompt": ..., "expect": {...}}
              to score an intermediate turn
    expect    scored against the last turn:
      tools      [{"name": regex, "args": {key: regex}, "ok": bool}] — each
                 must have run (name full-matched, arg values searched, both
                 case-insensitive; ok → only a call that did not fail
                 counts). {"any": [spec, ...]} accepts any one of several.
      ordered    true → the `tools` entries ran in that order
      forbid     [name regex] — none may run (a gate blocking it is a pass)
      no_tools   true → nothing ran at all
      ask        true → the turn ended asking the user; false → it did not
      say        [regex] — each must match the reply
      not_say    [regex] — none may match the reply
      max_calls  upper bound on calls that ran
      no_failed  true → no call that ran reported failure

Side effects are real: tools that write (save_structure, remove_solvent, ...)
write under protein-viz-agent/, and uncached structures are downloaded.
"""

import argparse
import copy
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent / "protein-viz-agent"
CASES = HERE / "prompts.jsonl"
REPORTS = HERE / "reports"
COURTESY = re.compile(r"anything else|let me know|can i (help|assist) you|"
                      r"further (help|assist|action|request)|"
                      r"would you like (me )?to (do|perform|try) (anything|something|any) "
                      r"(else|further|more)", re.I)
DISCOVERED = ("PACKMOL_MEMGEN", "DSSP_BIN", "FOLDSEEK_BIN", "FOLDSEEK_DB", "FOLDSEEK_AFDB",
              "FPOCKET_BIN", "PYMOL_PYTHON", "ESM_PYTHON")


def discover_tools() -> None:
    """
    Export what run.sh's tool discovery finds, so the eval sees the backends
    the app does. Without it, a plain `python evals/run.py` found none of
    AmberTools, fpocket, Foldseek or DSSP and skipped or mis-scored their cases.
    Variables already set win.
    """
    import subprocess
    script = HERE.parent / "discover_tools.sh"
    if not script.exists():
        return
    try:
        out = subprocess.run(["bash", "-c", f'source "{script}" >/dev/null 2>&1; env -0'],
                             capture_output=True, timeout=120).stdout
    except Exception:
        return
    for item in out.decode(errors="replace").split("\0"):
        key, _, value = item.partition("=")
        if key in DISCOVERED and value and not os.getenv(key):
            os.environ[key] = value


def load_cases(path: Path) -> list:
    cases, seen = [], set()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"{path.name}:{n}: {e}")
        if c["id"] in seen:
            sys.exit(f"{path.name}:{n}: duplicate id {c['id']}")
        seen.add(c["id"])
        c.setdefault("groups", [])
        c.setdefault("requires", [])
        c.setdefault("preload", [])
        c.setdefault("expect", {})
        cases.append(c)
    return cases


# ── Scoring ──────────────────────────────────────────────────────────────────

def _search(pattern: str, value) -> bool:
    if isinstance(value, (dict, list)):
        value = json.dumps(value)
    return re.search(pattern, str(value), re.I | re.S) is not None


def _matches(spec: dict, call: dict) -> bool:
    if "any" in spec:
        return any(_matches(s, call) for s in spec["any"])
    if not re.fullmatch(spec["name"], call["tool"], re.I):
        return False
    if spec.get("ok") and call.get("failed"):
        return False
    return all(k in call["args"] and _search(p, call["args"][k])
               for k, p in spec.get("args", {}).items())


def _describe(spec: dict) -> str:
    if "any" in spec:
        return " | ".join(_describe(s) for s in spec["any"])
    args = ", ".join(f"{k}~/{v}/" for k, v in spec.get("args", {}).items())
    return f"{spec['name']}({args}){' ok' if spec.get('ok') else ''}"


def score(expect: dict, trace: list, reply: str, asked: bool) -> list:
    """Each check → (label, passed, detail)."""
    ran = [c for c in trace if c["status"] == "ran"]
    checks = []

    if expect.get("tools"):
        pos, missing = -1, []
        for spec in expect["tools"]:
            hits = [i for i, c in enumerate(ran) if _matches(spec, c)]
            if expect.get("ordered"):
                hits = [i for i in hits if i > pos]
            if hits:
                pos = hits[0]
            else:
                missing.append(_describe(spec))
        checks.append(("tools" + (" (ordered)" if expect.get("ordered") else ""),
                       not missing, "missing " + "; ".join(missing) if missing else ""))

    if expect.get("forbid"):
        bad = [c["tool"] for c in ran
               if any(re.fullmatch(p, c["tool"], re.I) for p in expect["forbid"])]
        checks.append(("forbid", not bad, "ran " + ", ".join(bad) if bad else ""))

    if expect.get("no_tools"):
        checks.append(("no_tools", not ran,
                       "ran " + ", ".join(c["tool"] for c in ran) if ran else ""))

    if "ask" in expect:
        checks.append(("ask" if expect["ask"] else "no_ask", asked == expect["ask"],
                       "" if asked == expect["ask"] else
                       ("did not ask" if expect["ask"] else "asked instead of acting")))

    for p in expect.get("say", []):
        checks.append((f"say /{p}/", _search(p, reply), ""))
    for p in expect.get("not_say", []):
        m = re.search(p, reply, re.I | re.S)
        checks.append((f"not_say /{p}/", m is None, f"found '{m.group(0)}'" if m else ""))

    if "max_calls" in expect:
        checks.append(("max_calls", len(ran) <= expect["max_calls"],
                       f"{len(ran)} ran" if len(ran) > expect["max_calls"] else ""))

    if expect.get("no_failed"):
        bad = [c["tool"] for c in ran if c.get("failed")]
        checks.append(("no_failed", not bad, "failed " + ", ".join(bad) if bad else ""))

    return checks


# ── Headless app ─────────────────────────────────────────────────────────────

class Harness:
    """app.py imported once in Streamlit bare mode; one fresh session per case."""

    def __init__(self, log_dir: Path):
        os.environ["PARORA_LOG_DIR"] = str(log_dir)
        os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
        os.chdir(APP_DIR)                     # app.py uses ./structures etc.
        sys.path.insert(0, str(APP_DIR))
        import logging
        logging.getLogger("streamlit").setLevel(logging.ERROR)
        import streamlit as st
        import app
        self.st, self.app = st, app
        # The full agent log goes to the report dir's parora.log; keep the
        # console for the scoreboard. (FileHandler subclasses StreamHandler.)
        plog = logging.getLogger("parora")
        for hd in list(plog.handlers):
            if type(hd) is logging.StreamHandler:
                plog.removeHandler(hd)
        # Count model round trips per case without touching app.py.
        self.llm_calls = 0
        chat = app.ollama_client.chat

        def counted(*a, **kw):
            self.llm_calls += 1
            return chat(*a, **kw)

        app.ollama_client.chat = counted

    def reset(self):
        ss = self.st.session_state
        for k in list(ss.keys()):
            del ss[k]
        for k, v in self.app.defaults.items():
            ss[k] = copy.deepcopy(v)

    def preload(self, items: list) -> list:
        errors = []
        for item in items:
            if isinstance(item, str):
                item = {"tool": "fetch_structure", "args": {"pdb_id": item}}
            fn = self.app.TOOL_DISPATCH[item["tool"]]
            res = self.app.as_tool_result(fn(item.get("args", {})))
            if not res.ok:
                errors.append(f"{item['tool']}: {res.summary[:200]}")
        # A preload that asked something must not leak into the first turn.
        self.st.session_state.pop("clarify", None)
        return errors

    def turn(self, prompt: str) -> dict:
        ss = self.st.session_state
        ss.messages.append({"role": "user", "content": prompt})
        n0, t0 = self.llm_calls, time.monotonic()
        try:
            reply = self.app.run_agent(prompt)
        except Exception as e:                # a crash is a scored failure
            import traceback
            reply = f"HARNESS: run_agent raised {type(e).__name__}: {e}"
            traceback.print_exc()
        ss.messages.append({"role": "assistant", "content": reply})
        trace = list(ss.get("agent_trace", []))
        # Asked = a pending clarify card, an ask_user call, or a short reply
        # that puts a question to the user in plain text — not counting a
        # closing courtesy offer ("Anything else?").
        questions = [q for q in re.findall(r"[^.?!\n]*\?", reply)
                     if not COURTESY.search(q)]
        asked = (bool(ss.get("clarify"))
                 or any(c["tool"] == "ask_user" and c["status"] == "ran" for c in trace)
                 or (bool(questions) and len(reply) < 600))
        return {"prompt": prompt, "reply": reply, "asked": asked, "trace": trace,
                "seconds": round(time.monotonic() - t0, 2),
                "llm_calls": self.llm_calls - n0}


def availability() -> dict:
    """Which optional backends this machine has — cases needing others skip."""
    import requests
    out = {}
    try:
        requests.head("https://files.rcsb.org", timeout=5)
        out["network"] = True
    except Exception:
        out["network"] = False
    for name, probe in {
        "fpocket": lambda: __import__("pockets").find_fpocket(),
        "foldseek": lambda: (__import__("structure_search").find_foldseek()
                             and Path(__import__("structure_search").find_database()
                                      + ".dbtype").exists()),
        "esm": lambda: (__import__("esm_tools").find_esm_python() or (None,))[0],
        "dssp": lambda: __import__("topology").find_dssp(),
        "pymol": lambda: __import__("pymol_render").find_pymol_python(),
        "amber": lambda: __import__("simulation").available(),
    }.items():
        try:
            out[name] = bool(probe())
        except Exception:
            out[name] = False
    out.update({f"!{k}": not v for k, v in list(out.items())})
    return out


# ── Reporting ────────────────────────────────────────────────────────────────

def _rate(results: list) -> str:
    n = len(results)
    p = sum(r["passed"] for r in results)
    return f"{p:3d}/{n:<3d} {100 * p / n:5.1f}%" if n else "   -"


def print_summary(results: list, skipped: list, groups: list):
    scored = [r for r in results if not r.get("preload_errors")]
    by_cat, by_group = defaultdict(list), defaultdict(list)
    for r in scored:
        by_cat[r["category"]].append(r)
        for g in r["groups"]:
            by_group[g].append(r)
    print("\n── Pass rate by category " + "─" * 40)
    for cat in sorted(by_cat):
        print(f"  {cat:<14} {_rate(by_cat[cat])}")
    print("\n── Pass rate by tool group " + "─" * 38)
    for g in groups:
        print(f"  {g:<14} {_rate(by_group.get(g, [])) if by_group.get(g) else '   - (no case ran)'}")
    secs = [t["seconds"] for r in scored for t in r["turns"]]
    print("\n── Overall " + "─" * 54)
    print(f"  cases          {_rate(scored)}")
    if scored:
        checks = [c for r in scored for c in r["checks"]]
        print(f"  checks         {sum(c['passed'] for c in checks)}/{len(checks)}")
        print(f"  latency/turn   median {statistics.median(secs):.1f}s, "
              f"max {max(secs):.1f}s")
        print(f"  llm calls      {sum(t['llm_calls'] for r in scored for t in r['turns'])}")
    broken = [r for r in results if r.get("preload_errors")]
    if broken:
        print(f"  preload failed {len(broken)}: " + ", ".join(r["id"] for r in broken))
    if skipped:
        print(f"  skipped        {len(skipped)} (missing: "
              + ", ".join(sorted({m for s in skipped for m in s['missing']})) + ")")
    fails = [r for r in scored if not r["passed"]]
    if fails:
        print("\n── Failed " + "─" * 55)
        for r in fails:
            why = "; ".join(f"{c['check']}: {c['detail']}" if c["detail"] else c["check"]
                            for c in r["checks"] if not c["passed"])
            print(f"  {r['id']:<34} {why[:140]}")


def compare(old_path: Path, results: list):
    old = {r["id"]: r for r in json.loads(old_path.read_text())["results"]}
    new = {r["id"]: r for r in results}
    fixed = [i for i in new if i in old and new[i]["passed"] and not old[i]["passed"]]
    broke = [i for i in new if i in old and not new[i]["passed"] and old[i]["passed"]]
    print(f"\n── Compared with {old_path} " + "─" * 10)
    print(f"  now passing ({len(fixed)}): {', '.join(fixed) or '-'}")
    print(f"  now failing ({len(broke)}): {', '.join(broke) or '-'}")
    common = [i for i in new if i in old]
    if common:
        o = sum(old[i]["passed"] for i in common)
        n = sum(new[i]["passed"] for i in common)
        print(f"  shared cases   {o} → {n} of {len(common)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--only", help="comma-separated categories, groups or case ids")
    ap.add_argument("--model", help="override the app model (PARORA_MODEL_APP)")
    ap.add_argument("--think", help="reasoning-model thinking: true/false/low/medium/high (PARORA_THINK)")
    ap.add_argument("--repeat", type=int, default=1, help="run each case N times")
    ap.add_argument("--compare", type=Path, help="earlier report.json to diff against")
    ap.add_argument("--list", action="store_true", help="list cases and exit")
    ap.add_argument("--no-skip", action="store_true",
                    help="run cases even when a required backend is missing")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # live progress through a pipe

    cases = load_cases(a.cases)
    if a.only:
        want = {w.strip() for w in a.only.split(",")}
        cases = [c for c in cases
                 if c["id"] in want or c["category"] in want or want & set(c["groups"])]
    if a.list:
        for c in cases:
            first = c["turns"][0]
            first = first["prompt"] if isinstance(first, dict) else first
            print(f"{c['id']:<34} {c['category']:<12} {first[:70]}")
        print(f"{len(cases)} cases")
        return

    if a.compare:
        a.compare = a.compare.resolve()   # Harness chdirs into protein-viz-agent/
    if a.model:
        os.environ["PARORA_MODEL_APP"] = a.model
    if a.think:
        os.environ["PARORA_THINK"] = a.think
    discover_tools()                      # before app.py is imported
    stamp =dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = REPORTS / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    h = Harness(out_dir)
    import requests
    try:
        requests.get(f"{h.app.OLLAMA_HOST}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"Ollama not reachable at {h.app.OLLAMA_HOST}: {e}")
    have = availability()
    print(f"model={h.app.MODEL}  think={h.app.THINK}  cases={len(cases)}x{a.repeat}  backends="
          + ", ".join(f"{k}:{'y' if v else 'n'}" for k, v in have.items()))

    results, skipped = [], []
    for c in cases:
        missing = [r for r in c["requires"] if not have.get(r, False)]
        if missing and not a.no_skip:
            skipped.append({"id": c["id"], "missing": missing})
            print(f"  skip {c['id']} (needs {', '.join(missing)})")
            continue
        for rep in range(a.repeat):
            h.reset()
            rid = c["id"] if a.repeat == 1 else f"{c['id']}#{rep + 1}"
            r = {"id": rid, "case": c["id"], "category": c["category"],
                 "groups": c["groups"], "turns": [], "checks": []}
            r["preload_errors"] = h.preload(c["preload"])
            if r["preload_errors"]:
                r["passed"] = False
                results.append(r)
                print(f"  ERR  {rid}: preload failed — {r['preload_errors'][0]}")
                continue
            for i, t in enumerate(c["turns"]):
                prompt, expect = (t["prompt"], t.get("expect")) if isinstance(t, dict) else (t, None)
                if i == len(c["turns"]) - 1:
                    expect = {**(expect or {}), **c["expect"]}
                tr = h.turn(prompt)
                r["turns"].append(tr)
                for label, ok, detail in score(expect or {}, tr["trace"], tr["reply"], tr["asked"]):
                    r["checks"].append({"turn": i + 1, "check": label,
                                        "passed": ok, "detail": detail})
            r["passed"] = all(ch["passed"] for ch in r["checks"])
            results.append(r)
            secs = sum(t["seconds"] for t in r["turns"])
            print(f"  {'PASS' if r['passed'] else 'FAIL'} {rid:<34} {secs:6.1f}s  "
                  + " → ".join(f"{x['tool']}{'' if x['status'] == 'ran' else '[' + x['status'] + ']'}"
                               for t in r["turns"] for x in t["trace"])[:90])

    print_summary(results, skipped, list(h.app.TOOL_GROUPS))
    report = {"stamp": stamp, "model": h.app.MODEL, "think": h.app.THINK, "options": h.app.OLLAMA_OPTIONS,
              "cases_file": str(a.cases), "repeat": a.repeat, "backends": have,
              "skipped": skipped, "results": results}
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nReport: {path}\nLog:    {out_dir / 'parora.log'}")
    if a.compare:
        compare(a.compare, results)


if __name__ == "__main__":
    main()
