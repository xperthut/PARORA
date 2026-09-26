# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-25
# Summary   : Streamlit-side bridge to ESM-2 zero-shot mutation-effect
#             scoring (todo.txt P10).
#
#             Method: masked-marginal log-likelihood ratio (Meier et al. 2021,
#             "Language models enable zero-shot prediction of the effects of
#             mutations on protein function"). Mask the position, read the
#             model's distribution over the 20 amino acids there, and report
#             log p(mutant) - log p(wild type). Negative = the model finds the
#             substitution less compatible with the sequence than the native
#             residue. A sequence-statistics prediction, never a measurement.
#
#             torch/transformers never enter the Streamlit env or the Docker
#             image: esm_worker.py runs in a separate interpreter (ESM_PYTHON,
#             else a conda env whose name mentions esm or torch), same pattern
#             as pymol_render.py. Nothing here imports torch.
# =============================================================================

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from sequence_utils import AA3_TO_1, parse_structure_residues

RESULT_MARKER = "__ESM_RESULT__"
WORKER = Path(__file__).parent / "esm_worker.py"
DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# Unobserved stretches longer than this are not padded with <mask> tokens —
# a very long run of masks drowns the real context instead of preserving it.
MAX_GAP_FILL = 60

CONDA_ROOTS = [
    "/opt/anaconda3/envs", "/opt/miniconda3/envs",
    "/opt/homebrew/Caskroom/miniconda/base/envs",
    os.path.expanduser("~/anaconda3/envs"), os.path.expanduser("~/miniconda3/envs"),
    os.path.expanduser("~/mambaforge/envs"), os.path.expanduser("~/miniforge3/envs"),
]

AA_NAMES = {
    "alanine": "A", "arginine": "R", "asparagine": "N", "aspartate": "D",
    "aspartic acid": "D", "cysteine": "C", "glutamine": "Q", "glutamate": "E",
    "glutamic acid": "E", "glycine": "G", "histidine": "H", "isoleucine": "I",
    "leucine": "L", "lysine": "K", "methionine": "M", "phenylalanine": "F",
    "proline": "P", "serine": "S", "threonine": "T", "tryptophan": "W",
    "tyrosine": "Y", "valine": "V",
}
ONE_TO_3 = {v: k for k, v in AA3_TO_1.items() if len(k) == 3 and v in AMINO_ACIDS
            and k not in ("MSE", "HSD", "HSE", "HSP", "CSO", "PTR", "SEP", "TPO",
                          "KCX", "MLY")}

# Heuristic reading of the log-likelihood ratio, checked against residues with
# documented effects (see validate() below / todo.txt P10 notes). These are
# rough bands for wording, not calibrated probabilities.
BANDS = [
    (0.0, "fits as well as or better than the native residue (score ≥ 0)"),
    (-3.0, "likely tolerated (-3 to 0)"),
    (-7.0, "possibly deleterious (-7 to -3)"),
    (float("-inf"), "likely deleterious (below -7)"),
]


def model_name() -> str:
    return os.getenv("ESM_MODEL", DEFAULT_MODEL)


def _conda_candidates():
    found = []
    for root in CONDA_ROOTS:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            low = name.lower()
            if "esm" in low or "torch" in low:
                exe = os.path.join(root, name, "bin", "python")
                if os.path.exists(exe):
                    found.append(exe)
    # Envs named for ESM before generic torch envs.
    return sorted(found, key=lambda p: "esm" not in Path(p).parent.parent.name.lower())


_PROBE = (
    "import sys, torch, transformers\n"
    "from huggingface_hub import try_to_load_from_cache\n"
    "ok = all(isinstance(try_to_load_from_cache(sys.argv[1], f), str)\n"
    "         for f in ('config.json', 'model.safetensors'))\n"
    "print('ok' if ok else 'nomodel')\n"
)

_cached = {}


def find_esm_python(force_rescan: bool = False):
    """
    Locate an interpreter that can import torch + transformers.

    Resolution order: ESM_PYTHON, this process's own interpreter, then conda
    envs whose name mentions esm or torch.

    Returns:
        (interpreter or None, model_cached: bool)
    """
    key = model_name()
    if key in _cached and not force_rescan:
        return _cached[key]
    candidates = []
    if os.getenv("ESM_PYTHON"):
        candidates.append(os.getenv("ESM_PYTHON"))
    candidates.append(sys.executable)
    candidates.extend(_conda_candidates())

    fallback = (None, False)
    for exe in candidates:
        if not exe or not Path(exe).exists():
            continue
        try:
            r = subprocess.run([exe, "-c", _PROBE, key], capture_output=True,
                               text=True, timeout=60)
        except Exception:
            continue
        out = r.stdout.strip()
        if r.returncode == 0 and out == "ok":
            _cached[key] = (exe, True)
            return _cached[key]
        if r.returncode == 0 and out == "nomodel" and fallback[0] is None:
            fallback = (exe, False)
    _cached[key] = fallback
    return fallback


def esm_available() -> bool:
    exe, cached = find_esm_python()
    return exe is not None and cached


def unavailable_reason() -> str:
    exe, cached = find_esm_python()
    if exe is None:
        return ("ESM-2 unavailable — no Python environment with torch + "
                "transformers found. Set up with: conda create -n esm -c "
                "conda-forge pytorch transformers (or set ESM_PYTHON), then "
                "run 'bash setup_tools.sh' to download the model.")
    return (f"ESM-2 unavailable — the {model_name()} checkpoint is not "
            f"downloaded yet (interpreter: {exe}). Run 'bash setup_tools.sh' "
            "to download it once (~2.6 GB); it then runs fully offline.")


def parse_amino_acid(text: str) -> str:
    """'A', 'Ala', 'ALA', 'alanine' -> 'A'; '' if not a standard amino acid."""
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) == 1 and t.upper() in AMINO_ACIDS:
        return t.upper()
    three = AA3_TO_1.get(t.upper())
    if three and three in AMINO_ACIDS:
        return three
    return AA_NAMES.get(t.lower(), "")


def chain_sequence(pdb_path, chain):
    """
    The chain's protein sequence as the model sees it, indexed by residue.

    Built from observed ATOM records (numbering matches what the user sees in
    the viewer). Unobserved numbering gaps up to MAX_GAP_FILL are padded with
    None (<mask> in the worker) so neighbours keep their true spacing; longer
    gaps are closed up and counted.

    Returns:
        (tokens, index_of, notes): tokens is a list of one-letter codes or
        None; index_of maps (resseq, icode) -> index into tokens.
    """
    residues = [r for r in parse_structure_residues(pdb_path).get(chain, [])
                if r["kind"] == "protein"]
    tokens, index_of = [], {}
    masked, closed, prev = 0, 0, None
    for r in residues:
        if prev is not None and not r["icode"]:
            gap = r["resseq"] - prev - 1
            if 0 < gap <= MAX_GAP_FILL:
                tokens.extend([None] * gap)
                masked += gap
            elif gap > MAX_GAP_FILL:
                closed += 1
        index_of[(r["resseq"], r["icode"])] = len(tokens)
        tokens.append(r["one"] if r["one"] in AMINO_ACIDS else "X")
        prev = r["resseq"]
    notes = []
    if masked:
        notes.append(f"{masked} unobserved residue(s) inside the chain were "
                     "masked (unknown identity) to keep spacing")
    if closed:
        notes.append(f"{closed} long unobserved gap(s) were closed up, so "
                     "context across them is approximate")
    return tokens, index_of, notes


def _run_worker(exe, job, timeout=600):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false")
    try:
        proc = subprocess.run([exe, str(WORKER)], input=json.dumps(job),
                              capture_output=True, text=True, timeout=timeout,
                              env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"ESM-2 worker timed out after {timeout}s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                return json.loads(line[len(RESULT_MARKER):])
            except ValueError:
                break
    tail = " | ".join((proc.stderr or proc.stdout or "").strip().splitlines()[-4:])
    return {"ok": False, "error": f"ESM-2 worker returned no result (exit "
                                  f"{proc.returncode}). {tail}"}


def position_log_probs(tokens, index):
    """Masked-marginal log-probabilities over the 20 amino acids at index."""
    exe, cached = find_esm_python()
    if exe is None or not cached:
        return {"ok": False, "error": unavailable_reason()}
    return _run_worker(exe, {"model": model_name(), "sequence": tokens,
                             "index": index})


def band(llr: float) -> str:
    for cutoff, text in BANDS:
        if llr >= cutoff:
            return text
    return BANDS[-1][1]
