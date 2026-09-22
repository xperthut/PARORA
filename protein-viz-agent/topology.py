# =============================================================================
# Summary   : Real secondary-structure and fold/topology reporting.
#
#             Two independent sources, never blended into a guess:
#             - DSSP (`mkdssp`, external binary, optional — same
#               discovery-at-runtime / report-unavailable pattern as
#               AmberTools and PyMOL) gives a real per-residue secondary-
#               structure assignment, computed from this structure's own
#               coordinates. From it, a coarse topology string
#               ("beta1-alpha1-beta2") is derived — real, computed data,
#               not a model guess.
#             - PDBe's SIFTS REST API gives the existing CATH/SCOP
#               classification for a PDB entry, when curators have already
#               classified it. A database lookup beats any computed
#               approximation when one exists, so callers should prefer it
#               and fall back to the DSSP-derived string only when SIFTS has
#               nothing for this entry (AlphaFold models, very recent
#               depositions, obsolete entries).
#
#             Nothing here talks to Streamlit or app.py's session state —
#             same split as interactions.py/sequence_utils.py/
#             structure_report.py, so this is testable standalone and app.py
#             owns the @st.cache_data wrapping.
# =============================================================================

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests

SCHEMA_VERSION = 1

SIFTS_BASE = "https://www.ebi.ac.uk/pdbe/api/mappings"

# DSSP's own 8-letter secondary-structure code, collapsed for topology-string
# purposes. H/G/I are all helices (alpha, 3-10, pi); E/B are both strand
# (extended and isolated beta bridge); everything else is a gap between
# elements. This is deliberately coarser than the code DSSP itself reports —
# callers that want the raw letter still get it in ss_raw.
_HELIX_CODES = {"H", "G", "I"}
_STRAND_CODES = {"E", "B"}

_GREEK = {"H": "α", "E": "β"}   # alpha, beta — used in topology strings


# ── DSSP discovery ──────────────────────────────────────────────────────────

_cached_dssp_bin = None


def find_dssp(force_rescan: bool = False):
    """
    Locate a working `mkdssp` binary.

    Resolution order: the DSSP_BIN env var, then PATH (shutil.which). A
    single static binary, so this mirrors prepare.py's `reduce` discovery
    rather than pymol_render.py's conda-env scan (there's no interpreter to
    pick between here).

    Returns:
        Path to a working mkdssp, or None if none was found.
    """
    global _cached_dssp_bin
    if _cached_dssp_bin and not force_rescan:
        return _cached_dssp_bin

    candidates = []
    env_bin = os.getenv("DSSP_BIN")
    if env_bin:
        candidates.append(env_bin)
    found = shutil.which("mkdssp")
    if found:
        candidates.append(found)
    found = shutil.which("dssp")   # some conda-forge builds still name it `dssp`
    if found:
        candidates.append(found)

    for exe in candidates:
        if not exe or not Path(exe).exists():
            continue
        try:
            r = subprocess.run([exe, "--version"], capture_output=True,
                              text=True, timeout=30)
            if r.returncode == 0:
                _cached_dssp_bin = exe
                return exe
        except Exception:
            continue
    return None


def dssp_available() -> bool:
    """True if a working `mkdssp` was found on this machine."""
    return find_dssp() is not None


# ── Running DSSP ─────────────────────────────────────────────────────────────

def run_dssp(pdb_path: str, timeout: int = 120) -> tuple:
    """
    Run DSSP on a structure and parse its per-residue secondary structure.

    Tries the modern mkdssp (PDB-REDO/libcifpp rewrite, DSSP 4.x) invocation
    first, falling back to the classic 2.x argument order — this hasn't been
    exercised against a live binary in this environment (no mkdssp installed
    here to test against), so both are attempted defensively rather than
    assuming one syntax.

    Args:
        pdb_path: Local .pdb file to run DSSP on.
        timeout : Seconds before the subprocess is killed.

    Returns:
        (ok, message, per_residue). per_residue is an ordered list of
        {chain, resnum, icode, aa, ss_raw} dicts — DSSP's own H/B/E/G/I/T/S/P
        codes (space for coil), one entry per observed residue, in file
        order. Empty list when ok is False.
    """
    exe = find_dssp()
    if exe is None:
        return (False,
                "DSSP not installed — computed topology unavailable. Set "
                "DSSP_BIN to an mkdssp binary, or install one: "
                "conda create -n dssp -c conda-forge dssp",
                [])

    if not Path(pdb_path).exists():
        return (False, f"Structure file not found: {pdb_path}", [])

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "out.dssp"
        attempts = [
            [exe, "--output-format", "dssp", pdb_path, str(out_path)],
            [exe, pdb_path, str(out_path)],
        ]
        last_err = ""
        for cmd in attempts:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
            except subprocess.TimeoutExpired:
                return (False, f"DSSP timed out after {timeout}s", [])
            except Exception as e:
                last_err = str(e)
                continue
            if r.returncode == 0 and out_path.exists():
                text = out_path.read_text(errors="replace")
                return (True, "", _parse_dssp_text(text))
            last_err = (r.stderr or r.stdout or "").strip()[-500:]
        return (False, f"DSSP failed: {last_err}" if last_err else "DSSP failed", [])


def _parse_dssp_text(text: str) -> list:
    """
    Parse classic DSSP-format text into per-residue records.

    Fixed-column format (same one Biopython's DSSP parser reads): residue
    number in columns 6-10, chain in column 12, amino acid in column 14,
    secondary structure in column 17 (1-based; see the slices below for the
    0-based equivalents). A chain-break line has "!" in the amino-acid
    column and carries no residue data, so it is skipped.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("  #  RESIDUE"):
            start = i + 1
            break
    if start is None:
        return []

    residues = []
    for line in lines[start:]:
        if len(line) < 17:
            continue
        aa = line[13]
        if aa == "!":
            continue
        resnum = line[5:10].strip()
        if not resnum:
            continue
        residues.append({
            "chain": line[11].strip() or "?",
            "resnum": int(resnum),
            "icode": line[10].strip(),
            "aa": aa,
            "ss_raw": line[16] if line[16] != " " else "C",
        })
    return residues


def coarse_ss(ss_raw: str) -> str:
    """Collapse a DSSP code to 'H' (helix), 'E' (strand) or 'C' (everything else)."""
    if ss_raw in _HELIX_CODES:
        return "H"
    if ss_raw in _STRAND_CODES:
        return "E"
    return "C"


# ── Topology string ──────────────────────────────────────────────────────────

def topology_string(per_residue: list, chain: str) -> str:
    """
    Build a coarse topology string for one chain from DSSP output.

    Consecutive residues of the same coarse type (helix/strand) collapse
    into one element; elements are numbered in sequence order, separately
    per type (first strand is beta1 regardless of how many helices came
    before it). Coil/turn/bend residues are gaps between elements and are
    not themselves numbered.

    Args:
        per_residue: Output of run_dssp (or a chain-filtered subset).
        chain      : Chain id to restrict to.

    Returns:
        e.g. "β1-α1-β2-α2-β3", or "" if the chain has no
        helix/strand content at all (e.g. an unstructured loop or DSSP found
        nothing for it).
    """
    counts = {"H": 0, "E": 0}
    elements = []
    prev = None
    for res in per_residue:
        if res["chain"] != chain:
            continue
        cs = coarse_ss(res["ss_raw"])
        if cs == prev:
            continue
        prev = cs
        if cs in counts:
            counts[cs] += 1
            elements.append(f"{_GREEK[cs]}{counts[cs]}")
    return "-".join(elements)


# ── CATH / SCOP classification (PDBe SIFTS) ─────────────────────────────────

def lookup_classification(pdb_id: str, timeout: int = 10) -> dict:
    """
    Look up an existing CATH/SCOP fold classification for a PDB entry.

    Queries PDBe's SIFTS REST mapping API (no key needed):
    https://www.ebi.ac.uk/pdbe/api/mappings/cath/{id} and .../scop/{id}.
    Confirmed live: 1crn returns a CATH hit, 4hhb returns a SCOP hit, both
    keyed by chain with human-readable classification names already
    attached — reported verbatim, never re-derived or paraphrased.

    A 404, timeout, network error or a response with no mapping for this
    entry all mean "no classification on file", not an exception — callers
    should treat an empty dict as "fall back to computed topology", not as
    a failure to be surfaced as an error.

    Args:
        pdb_id : 4-character PDB accession.
        timeout: Seconds before giving up on each of the two requests.

    Returns:
        {chain_id: {"source": "CATH", "class": ..., "architecture": ...,
                     "topology": ..., "homology": ..., "name": ...}}
        or the SCOP-shaped equivalent
        ({"source": "SCOP", "class": ..., "fold": ..., "superfamily": ...,
          "name": ...}) when only SCOP has a hit for that chain. CATH is
        preferred over SCOP per chain when both exist — it is the more
        actively maintained of the two. Empty dict when neither has
        anything for this entry.
    """
    pid = (pdb_id or "").strip().lower()
    if not pid:
        return {}

    by_chain = {}

    try:
        r = requests.get(f"{SIFTS_BASE}/cath/{pid}", timeout=timeout)
        if r.status_code == 200:
            entry = (r.json().get(pid) or {}).get("CATH") or {}
            for cath_id, dom in entry.items():
                for m in dom.get("mappings", []):
                    ch = m.get("chain_id")
                    if ch and ch not in by_chain:
                        by_chain[ch] = {
                            "source": "CATH",
                            "cath_id": cath_id,
                            "class": dom.get("class", ""),
                            "architecture": dom.get("architecture", ""),
                            "topology": dom.get("topology", ""),
                            "homology": dom.get("homology", ""),
                            "name": dom.get("identifier", ""),
                        }
    except requests.RequestException:
        pass

    try:
        r = requests.get(f"{SIFTS_BASE}/scop/{pid}", timeout=timeout)
        if r.status_code == 200:
            entry = (r.json().get(pid) or {}).get("SCOP") or {}
            for sun_id, dom in entry.items():
                for m in dom.get("mappings", []):
                    ch = m.get("chain_id")
                    if ch and ch not in by_chain:
                        by_chain[ch] = {
                            "source": "SCOP",
                            "sunid": sun_id,
                            "class": (dom.get("class") or {}).get("description", ""),
                            "fold": (dom.get("fold") or {}).get("description", ""),
                            "superfamily": (dom.get("superfamily") or {}).get("description", ""),
                            "name": dom.get("description", ""),
                        }
    except requests.RequestException:
        pass

    return by_chain
