#!/usr/bin/env python
"""
Replay real failed tool calls through the recovery ladder (suggestion.txt S8).

No model: each case preloads structures, runs one failed call the way the
eval traces of 2026-10-01/02 recorded it, then hands the failure to app.py's
_recover() and checks which rung caught it (normalize / equivalent / ask /
report / none) and with what arguments. Deterministic, ~10 s, so it runs
before the slower evals/run.py:

    ~/miniconda3/envs/parora/bin/python evals/recovery_replay.py
"""

import json
import re
import sys

from run import Harness, REPORTS

# (id, preload, setup calls, failed call, user prompt, expected step,
#  {arg: regex} the standing call must carry, or None)
CASES = [
    ("structure-prep-glued", ["1CRN", {"tool": "prepare_structure", "args": {"target": "1CRN"}}],
     [], ("describe_structure", {"target": "_PREP1CRN"}),
     "describe the prepared structure", "normalize", {"target": r"^1CRN_PREP$"}),
    ("structure-prep-bare", ["1CRN", {"tool": "prepare_structure", "args": {"target": "1CRN"}}],
     [], ("inspect_preparation", {"target": "_PREP"}),
     "inspect the prepared structure", "normalize", {"target": r"^1CRN_PREP$"}),
    ("structure-no-guess-down", ["1M17"], [], ("describe_structure", {"target": "1M17_PREP"}),
     "describe the prepared structure", None, None),
    ("residue-typo-ly48", ["1UBQ"], [], ("highlight", {"target": "A/LY48"}),
     "highlight lysine 48", "normalize", {"target": r"A/LYS48"}),
    ("residue-wrong-name", ["1UBQ"], [], ("highlight", {"target": "A/HIS48"}),
     "highlight his 48", "report", None),
    ("drug-name-erlotinib", ["1M17"], [], ("find_interactions", {"target": "erlotinib"}),
     "what does erlotinib interact with", "normalize", {"target": r"^AQ4$"}),
    ("drug-name-contacts", ["1M17"], [], ("find_contacts", {"target": "erlotinib"}),
     "which residues touch erlotinib", "normalize", {"target": r"^AQ4$"}),
    ("heme-word", ["4HHB"], [], ("find_contacts", {"target": "heme"}),
     "what surrounds the heme", "normalize", {"target": r"^HEM$"}),
    ("selection-made-up", ["4HHB"],
     [("select_within", {"name": "contacts_HEM_A", "radius": 4,
                         "target_selection": "HEM and chain A"})],
     ("highlight", {"target": "heme_contacts"}),
     "highlight the heme contacts", "normalize", {"target": r"^contacts_HEM_A$"}),
    ("ask-unknown-drug", ["1M17"], [], ("highlight", {"target": "gefitinib"}),
     "highlight gefitinib", "ask", None),
    ("ask-truncated-word", ["1M17"], [], ("highlight", {"target": "gef"}),
     "highlight gefitinib", "ask", None),
    ("no-ask-generic-word", ["4HHB"], [],
     ("select_within", {"name": "pocket_1_lining", "radius": 6, "target_selection": "pocket 1"}),
     "find the pockets, then list the residues lining pocket 1", None, None),
    ("ask-not-user-word", ["4HHB"], [], ("highlight", {"target": "heme_neighbors"}),
     "highlight what is near the heme", "report", None),
    ("residue-900", ["4HHB"], [], ("highlight", {"target": "A/900"}),
     "highlight residue 900 in chain A", "report", None),
    ("chain-z", ["1UBQ"], [], ("highlight", {"target": "Z/10"}),
     "show residue 10 of chain Z", "report", None),
    ("mda-empty-not-backend", ["1CRN"], [],
     ("measure_mda_distance", {"sel1": "resid 500 and name CA", "sel2": "resid 10 and name CA"}),
     "measure resid 500 to 10", None, None),
    # MDAnalysis made unavailable for these: the backend, not the input, fails.
    ("equiv-nearby-no-mda", ["4HHB"], [], ("nearby_residues", {"selection": "A/HIS87"}),
     "what residues are near HIS 87 in chain A", "equivalent", {"target": r"A/HIS87"}),
    ("equiv-contacts-no-mda", ["4HHB"], [],
     ("detect_contacts", {"sel1": "HEM", "sel2": "protein", "cutoff": 4.0}),
     "what touches the heme", "equivalent", {"target": r"^HEM$"}),
    ("equiv-distance-no-mda", ["1CRN"], [],
     ("measure_mda_distance", {"sel1": "resid 3 and name CA", "sel2": "resid 40 and name CA"}),
     "distance between the CA atoms of 3 and 40", "equivalent", {"a": r"resid 3"}),
]
NO_MDA = {"equiv-nearby-no-mda", "equiv-contacts-no-mda", "equiv-distance-no-mda"}


def run_case(h, case) -> dict:
    cid, preload, setup, (tool, args), prompt, want_step, want_args = case
    h.reset()
    errors = h.preload(preload)
    for t, a in setup:
        r = h.app._call_tool(t, a)
        if not r.ok:
            errors.append(f"setup {t}: {r.summary[:120]}")
    ss = h.st.session_state
    ss.current_request = prompt
    real_universe = h.app.get_universe
    if cid in NO_MDA:
        h.app.get_universe = lambda *a, **k: None
    try:
        return _replay(h, cid, tool, args, prompt, want_step, want_args, errors)
    finally:
        h.app.get_universe = real_universe


def _replay(h, cid, tool, args, prompt, want_step, want_args, errors) -> dict:
    ss = h.st.session_state
    first = h.app._call_tool(tool, args)
    out = {"id": cid, "tool": tool, "args": args, "first_ok": first.ok,
           "first": first.summary[:160], "errors": errors}
    if first.ok:
        out.update(step=None, passed=want_step is None and not errors,
                   note="call did not fail")
        return out
    fixed = h.app._recover(tool, args, first, prompt)
    step = fixed[3]["step"] if fixed else None
    out.update(step=step, record=fixed[3] if fixed else None,
               standing=fixed[2].summary[:200] if fixed else first.summary[:200])
    ok = step == want_step and not errors
    if ok and want_args:
        ok = all(re.search(p, str(fixed[1].get(k, ""))) for k, p in want_args.items())
    if ok and step == "ask":
        ok = bool(ss.get("clarify")) and len(ss.clarify["options"]) >= 2
    ss.pop("clarify", None)
    out["passed"] = ok
    return out


def main() -> int:
    h = Harness(REPORTS / "_replay")
    results = [run_case(h, c) for c in CASES]
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{mark}  {r['id']:<24} {r['tool']}({json.dumps(r['args'])[:50]}) "
              f"→ {r['step'] or '-'}")
        if not r["passed"]:
            print(f"      first: {r['first']}\n      standing: {r.get('standing', '')}"
                  f"\n      errors: {r['errors']}\n      record: {r.get('record')}")
    n = sum(r["passed"] for r in results)
    print(f"\n{n}/{len(results)} recovery replays pass")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
