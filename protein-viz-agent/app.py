# =============================================================================
# Developer : Abdullah Al Mamun, Methun Kamruzzaman
# Date      : 2026-03-11
# Summary   : Full-featured agentic protein structure visualizer with
#             MCP-style multi-turn tool-calling via Ollama. Integrates
#             MDAnalysis for server-side structural analysis (B-factor
#             filtering, proximity selections, solvent removal), maintains
#             named selections and layered NGL.js representations, and
#             enforces a gate-based agent loop to prevent redundant or
#             destructive tool calls.
#
#             Any number of structures can share the viewer so they can be
#             compared. Superposition (superpose.py) is applied as a per-
#             structure 4x4 transform rather than by rewriting coordinates, so
#             a fit is reversible and the files on disk always agree with what
#             MDAnalysis measured. Camera orientation and the drag mode are
#             persisted across reruns via localStorage.
#
#             measure.py answers geometric questions, distance between two
#             residues, angle at a residue, what lies within a cutoff of a
#             ligand, from residue specifications
#
#             structure_report.py reads the deposited PDB records directly to
#             report a structure's composition, chains, residue counts,
#             modified residues, ligands, cofactors, ions and crystallization
#             additives, with no MDAnalysis, Biopython or network call.
# =============================================================================

import streamlit as st
import base64
import os
import json
import logging
import math
import re
import shutil
import time
import uuid
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ollama import Client
from analysis_tools import (
    summarize_chains_from_universe,
    list_residues_from_universe,
    bfactor_summary_from_universe,
    measure_distance_from_universe,
    nearby_residues_from_universe,
    measure_angle_from_universe,
    measure_dihedral_from_universe,
    contact_detection_from_universe,
    salt_bridge_detection_from_universe,
    hydrogen_bond_detection_from_universe,
    nearby_residues_from_universe,
    measure_angle_from_universe,
    measure_dihedral_from_universe,
)

# MDAnalysis is optional — structural analysis features degrade gracefully
mda = None
try:
    import MDAnalysis as mda  # type: ignore[no-redef]
    MDA_AVAILABLE = True
except ImportError:
    MDA_AVAILABLE = False

# PyMOL ray tracing is optional. pymol_render never imports pymol itself — it
# shells out to a PyMOL-capable interpreter — so this import is safe even when
# PyMOL is absent from the Streamlit environment.
try:
    import pymol_render
    PYMOL_AVAILABLE = pymol_render.pymol_available()
except Exception:
    pymol_render = None
    PYMOL_AVAILABLE = False

# Sequence browsing parses PDB records directly, with no MDAnalysis dependency,
# so the residue picker keeps working when MDAnalysis fails to import.
import sequence_utils as squ

# Composition report (chains, residues, ligands, cofactors). Parses PDB records
# directly, so it needs neither MDAnalysis nor a network call.
import structure_report as srep

# Non-covalent interaction detection — salt bridges, hydrogen bonds,
# disulfides, stacking, cation-pi, metal coordination.
import interactions as ixn

# Secondary structure (DSSP, optional external binary) and fold/topology
# classification (CATH/SCOP via PDBe's SIFTS REST API).
import topology as topo

# Structure-based fold/homology search (Foldseek, optional external binary +
# reference database) — the fallback when CATH/SCOP has nothing to key on.
import structure_search as fsk

# Zero-shot mutation-effect scoring with ESM-2 (optional: torch lives in a
# separate interpreter, driven as a subprocess — same pattern as PyMOL).
import esm_tools as esm

# Binding-pocket / druggability detection (fpocket, optional external binary)
# — candidate sites from the fold itself, bound ligand or not.
import pockets as pkt

# Geometric measurement — distances, angles, contact shells. Reads coordinates
# from the PDB records directly, so it works without MDAnalysis.
import measure as mz

# Rigid-body superposition. Imports MDAnalysis/Biopython defensively and
# reports its own availability, so the module is always safe to import.
import superpose as sup

# Structure preparation — one NMR state instead of twenty, one conformation
# instead of two, the cryoprotectant gone, hydrogens present or deliberately
# absent, and a list of what is still wrong. Pure PDB-record editing, so it
# works wherever the app does; `reduce` is used for hydrogens.
import prepare as prp

# Quantum chemistry: cut a QM region out of a structure and write the input
# for Gaussian, ORCA or Psi4; and ONIOM QM/MM on top of an Amber topology.
import oniom
import quantum as qm

# Simulation setup — Amber topology (it runs tleap and antechamber for real),
# GROMACS run files and Rosetta job files.
import simulation as sim

# Membrane embedding — orient a membrane protein across the bilayer and pack
# lipids around it with PACKMOL-Memgen. Shells out to AmberTools rather than
# importing it, and reports its own availability, so the import is always safe.
import membrane as mem

# Protein-level lookup: a name ("insulin receptor", "EGFR", "P06213") resolved
# through UniProt to every PDB structure of that protein, with what each one
# covers, at what resolution, with which ligands. The layer above the viewer:
# it answers "which file do I want" before anything is loaded.
import Protein_accession as pacc

from parora_logging import setup_logging
from parora_config import get_config
from rag_grounding import format_grounding

log = setup_logging("app")

# ── Asset resolution: works in both Docker (/app/logo/) and local dev (../logo/)
_HERE = Path(__file__).parent
_LOGO_NOTEXT = _HERE / "logo" / "logo_notext.png"
if not _LOGO_NOTEXT.exists():
    _LOGO_NOTEXT = _HERE.parent / "logo" / "logo_notext.png"
_LOGO = _HERE / "logo" / "logo.png"
if not _LOGO.exists():
    _LOGO = _HERE.parent / "logo" / "logo.png"

st.set_page_config(
    page_title="Molecular Agent",
    page_icon=str(_LOGO_NOTEXT) if _LOGO_NOTEXT.exists() else "🧬",
    layout="wide",
)

st.markdown(
    "<style>"
    "header[data-testid='stHeader'] { height: 0; visibility: hidden; }"
    "[data-testid='stToolbar'] { display: none; }"
    ".block-container { padding-top: 1rem !important; "
    "padding-left: 2rem !important; padding-right: 2rem !important; "
    "max-width: 100% !important; }"
    "</style>",
    unsafe_allow_html=True,
)

# ── Compact header: logo + product name + one-line description ────────────────
# Built as one inline-flex block, not st.columns — columns reserve a fixed
# fraction of the page width per column regardless of the image's actual
# size, which left a wide dead gap between the 48px logo and the title.
_logo_b64 = base64.b64encode(_LOGO.read_bytes()).decode() if _LOGO.exists() else None
_logo_img = (
    f"<img src='data:image/png;base64,{_logo_b64}' "
    "style='width:48px;height:48px;object-fit:contain;flex-shrink:0;'>"
    if _logo_b64 else ""
)
st.markdown(
    f"<div style='display:flex;align-items:center;gap:12px;'>"
    f"{_logo_img}"
    "<div style='display:flex;flex-direction:column;justify-content:center;'>"
    "<span style='font-size:1.6rem;font-weight:700;line-height:1.2;'>Molecular Agent</span>"
    "<span style='font-size:0.85rem;color:#888;'>Natural language → Tool-calling agent → Analyze and visualize molecular structures</span>"
    "</div></div>",
    unsafe_allow_html=True,
)

st.divider()

# ── Model + Ollama host: driven by config.yaml, env vars still win ────────────
_CFG = get_config("app")
OLLAMA_HOST = _CFG["ollama_host"]
MODEL = _CFG["model"]

# Ollama allocates a KV cache for the whole context it is given, and when
# num_ctx is left unset it uses the model's full trained window (32,768 for
# qwen2.5:7b). config.yaml's default of 16k is still far more than a turn of
# this app ever uses and keeps the KV cache modest on top of the model's
# quantized weights.
OLLAMA_OPTIONS = {"temperature": _CFG["temperature"], "num_ctx": _CFG["num_ctx"]}

# Hold the model in memory between messages, so a pause in the conversation
# does not cost a reload from disk on the next one.
KEEP_ALIVE = _CFG["keep_alive"]

# Reasoning models (qwen3) think before every tool call unless told not to;
# None leaves the flag unsent for models that have no thinking mode.
THINK = _CFG["think"]
ollama_client = Client(host=OLLAMA_HOST)


@st.cache_resource(show_spinner=False)
def _log_startup_once() -> bool:
    """
    Log one summary of what's actually available in this environment.

    Runs exactly once per process -- st.cache_resource, not a plain module
    global, because Streamlit re-executes this whole script on every chat
    message, and a plain global would be reset (and this would re-log) on
    every single turn.
    """
    log.info("PARORA app.py starting -- model=%s ollama_host=%s", MODEL, OLLAMA_HOST)
    log.info("MDAnalysis available: %s", MDA_AVAILABLE)
    log.info("PyMOL available: %s (PYMOL_PYTHON=%s)",
             PYMOL_AVAILABLE, os.getenv("PYMOL_PYTHON", "<not set>"))
    amber = mem.find_backend()
    if amber["ready"]:
        log.info("AmberTools backend ready: amberhome=%s", amber["amberhome"])
    else:
        log.warning("AmberTools backend not ready: %s", amber["detail"] or "not found")
    return True


_log_startup_once()

# ── Storage ───────────────────────────────────────────────────────────────────
# PDB files downloaded during the session are cached here to avoid re-fetching
STRUCTURES_DIR = Path("./structures")
STRUCTURES_DIR.mkdir(exist_ok=True)

# Membrane builds get their own directory per job: PACKMOL-Memgen writes a
# dozen intermediate files into its working directory and reuses them by name,
# so two builds sharing a directory would read each other's leftovers.
MEMBRANES_DIR = Path("./membranes")
MEMBRANES_DIR.mkdir(exist_ok=True)

# Cleaned-up structures, and the leap scripts generated alongside them.
PREPARED_DIR = Path("./prepared")
PREPARED_DIR.mkdir(exist_ok=True)

# One directory per simulation setup: topology, coordinates, run inputs and
# the ligand parameters they depend on all have to travel together.
SIMULATIONS_DIR = Path("./simulations")
SIMULATIONS_DIR.mkdir(exist_ok=True)

# ── Session state defaults ────────────────────────────────────────────────────
# Initialise every key on first run; subsequent reruns leave existing values.
defaults = {
    "messages":      [],        # chat history shown in the left panel
    "debug_logs":    [],        # internal tool-call trace for the debug expander
    "structures":    [],        # every loaded structure — see register_structure()
    "active_sid":    None,      # sid of the structure analysis tools operate on
    "pdb_id":        None,      # active PDB accession
    "pdb_path":      None,      # local path of the downloaded .pdb file
    "universe":      None,      # MDAnalysis Universe object (lazy-loaded)
    "selections":    {},        # name → NGL selection string (named selections)
    "representations": [],      # list of {type, selection, color, transparency}
    "background":    "black",   # NGL viewer background colour
    "camera_target": None,      # NGL selection to zoom/focus on after load
    "render_path":   None,      # path of the most recent PyMOL ray-traced PNG
    "render_msg":    None,      # status line from the most recent render
    "superpose_msg": None,      # result line from the most recent superposition
    "measurements":  [],        # distances drawn in the viewer — see _remember_measurement
    "interactions":  [],        # detected interactions drawn in the viewer
    "interaction_msg": None,    # report text from the most recent scan
    "protein_query":   "",      # last protein name looked up
    "protein_hits":    [],      # UniProt entries matching that name
    "protein_profile": None,    # full profile of the chosen UniProt entry
    "protein_msg":     None,    # status line from the most recent lookup
    "prepared":        {},      # pdb_id → {path, report, finding} for cleaned copies
    "prepare_msg":     None,    # status line from the most recent preparation
    "qm_region":       None,    # the most recent QM region and its inputs
    "oniom_model":     None,    # the most recent ONIOM layering
    "ligand_params":   {},      # (structure, code) → {mol2, frcmod, charge, note}
    "amber_build":     None,    # the most recent tleap result
    "membrane_job":    None,    # the running or finished membrane build
    "membrane_msg":    None,    # status line from the most recent membrane step
    "oriented":        {},      # pdb_id → {path, info, source} for oriented copies
    "focus":           None,    # the protein the conversation is about — see set_focus
    "annotations":     [],      # text labels pinned to regions — see tool_add_label
    "label_msg":       None,    # status line from the most recent labelling action
    "active_only":     False,   # draw only the active structure — see _viewer_payload
    "viewer_seq":      0,       # last viewer event applied — see handle_viewer_event
    "toolbar_open":    None,    # (top, sub) of the open toolbar dialog — see _toolbar_dialog
    "facts":           [],      # every successful agent tool result — see _remember (S5)
    "request_no":      0,       # chat requests run_agent has handled
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# Structure registry
# ═══════════════════════════════════════════════════════════════════════════════
# Several structures can be open at once so they can be superimposed and compared.
# st.session_state.structures is the ordered list of them; each entry is
#
#   sid        stable short id — the handle used by representations, widget
#              keys and the viewer, so renaming or reloading never mixes layers
#   pdb_id     display label (accession, or the file stem for a local file)
#   path       local .pdb path; every structure is on disk, including uploads
#   source     "rcsb" (fetched, so the viewer can stream it from RCSB) or
#              "local" (the file's text must be embedded in the viewer HTML)
#   visible    per-structure show/hide, independent of representation layers
#   color      hex colour identifying this structure in the "By structure" scheme
#   matrix     16 floats, column-major, from a superposition — or None
#   fit        the superposition result line, shown next to the structure
#
# st.session_state.pdb_id / pdb_path / universe mirror the *active* structure.
# Every analysis tool reads those, so the whole MDAnalysis side of the app keeps
# working on one structure at a time and only the viewer is multi-structure.

# Distinct hues for telling superposed structures apart at a glance. Ordered so
# the first two — the common case of one structure fitted onto another — are the
# most clearly separated.
STRUCTURE_COLORS = [
    "#4C9BE8", "#F2A93B", "#59C36A", "#E4595B",
    "#B07AD9", "#3FBFB4", "#F06BA8", "#C2B280",
]

# Sentinel colour meaning "whatever colour this structure was assigned".
# Resolved per structure when the viewer HTML is built.
BY_STRUCTURE = "__structure__"


def structures() -> list:
    """The loaded structures, in the order they were added."""
    return st.session_state.structures


def find_structure(key: str):
    """
    Look a structure up by sid, PDB id or label, case-insensitively.

    The agent refers to structures the way the user does, while the
    UI and representations refer to them by sid, so both have to resolve.
    """
    if not key:
        return None
    k = str(key).strip().lower()
    for s in structures():
        if k in (s["sid"].lower(), s["pdb_id"].lower()):
            return s
    return None


def active_structure():
    """The structure that analysis tools and the sequence browser operate on."""
    return find_structure(st.session_state.active_sid) or (
        structures()[0] if structures() else None)


@st.cache_data(show_spinner=False)
def _cached_chain_molecules(path: str, mtime: float):
    """srep.chain_molecules, cached on the file's contents."""
    return srep.chain_molecules(path)


def chain_molecules(entry) -> list:
    """Which molecule each chain of a loaded structure is, or [] if unreadable."""
    if not entry or not Path(entry["path"]).exists():
        return []
    p = Path(entry["path"])
    return _cached_chain_molecules(str(p), p.stat().st_mtime)


_WHOLE_SELECTIONS = {"", "*", "all", "protein", "polymer", "backbone", "sidechain",
                     "chains", "not water", "not hetero"}


def chains_drawn(entry) -> set:
    """
    Chains of a structure that some visible layer draws, as far as can be told.

    A layer on 'protein' draws every chain; ':A' or 'chain A' draws one.
    Anything else (a ligand, a residue range) is not counted as drawing a
    chain. Without this the model read "cartoon on 'protein'" and told the
    user the non-HER2 chains of 7MN5 were hidden, when they were on screen.
    """
    if not entry or not entry.get("visible", True):
        return set()
    every = {c["chain"] for c in chain_molecules(entry)}
    drawn = set()
    for rep in st.session_state.representations:
        if not rep.get("visible", True) or not rep_applies_to(rep, entry["sid"]):
            continue
        sel = str(rep.get("selection", "")).strip()
        if sel.lower() in _WHOLE_SELECTIONS:
            return every
        drawn |= {c for c in re.findall(r"(?::|\bchain\s+)([A-Za-z0-9])\b", sel)} & every
    return drawn


def chain_map_line(entry, accession: str = "", show_drawn: bool = False) -> str:
    """
    '7MN5 chains: A = erbb-3 (P21860, res 28-630); B = erbb-2 ... ← focus protein'.

    Every chain in the file, not only the one belonging to the protein that was
    searched for: the agent used to answer "which chain is loaded" with "chain
    A" for 7MN5 — which is HER3, not the HER2 the user asked for.
    """
    rows = chain_molecules(entry)
    if not rows:
        return ""
    base = accession.split("-")[0] if accession else ""
    drawn = chains_drawn(entry) if show_drawn else set()
    parts = []
    for r in rows:
        mol = r["molecule"] or "unnamed molecule"
        acc = ", ".join(r["uniprot"])
        bit = (f"{r['chain']} = {mol}" + (f" ({acc})" if acc else "")
               + f", residues {r['first']}-{r['last']} observed")
        if show_drawn:
            bit += ", drawn in the viewer" if r["chain"] in drawn else ", not drawn"
        if base and any(a.split("-")[0] == base for a in r["uniprot"]):
            bit += " ← the working-context protein"
        parts.append(bit)
    return f"{entry['pdb_id']} chains: " + "; ".join(parts)


def _sync_active() -> None:
    """
    Mirror the active structure into the legacy single-structure session keys.

    pdb_id / pdb_path / universe predate multi-structure support and are read
    all over the analysis tools. Keeping them in step here means those tools
    did not have to change.
    """
    s = active_structure()
    st.session_state.active_sid = s["sid"] if s else None
    st.session_state.pdb_id = s["pdb_id"] if s else None
    st.session_state.pdb_path = s["path"] if s else None
    st.session_state.universe = None      # force an MDAnalysis reload


def register_structure(pdb_id: str, path: str, source: str = "rcsb",
                       make_active: bool = True) -> dict:
    """
    Add a structure to the registry, or return the existing entry for it.

    Args:
        pdb_id     : Accession or label to show in the UI.
        path       : Local .pdb path.
        source     : "rcsb" if the viewer may stream it from RCSB by accession,
                     "local" if its text must be embedded in the viewer HTML.
        make_active: Point the analysis tools at this structure.

    Returns:
        The registry entry.
    """
    existing = find_structure(pdb_id)
    if existing:
        if make_active:
            st.session_state.active_sid = existing["sid"]
            _sync_active()
        return existing

    used = {s["color"] for s in structures()}
    color = next((c for c in STRUCTURE_COLORS if c not in used),
                 STRUCTURE_COLORS[len(structures()) % len(STRUCTURE_COLORS)])
    entry = {
        "sid": "s" + uuid.uuid4().hex[:6],
        "pdb_id": pdb_id.upper(),
        "path": str(path),
        "source": source,
        "visible": True,
        "color": color,
        "matrix": None,
        "fit": None,
    }
    st.session_state.structures.append(entry)
    note_focus_structure(entry["pdb_id"])
    if make_active:
        st.session_state.active_sid = entry["sid"]
    _sync_active()
    return entry


def drop_structure(key: str) -> bool:
    """Remove a structure and every representation layer scoped to it."""
    s = find_structure(key)
    if not s:
        return False
    sid = s["sid"]
    st.session_state.structures = [x for x in structures() if x["sid"] != sid]
    st.session_state.representations = [
        r for r in st.session_state.representations if r.get("sid") != sid
    ]
    if st.session_state.active_sid == sid:
        st.session_state.active_sid = None
    _sync_active()
    return True


def reset_structures() -> None:
    """Clear the registry — used when a fetch replaces the whole scene."""
    st.session_state.structures = []
    st.session_state.active_sid = None


# ═══════════════════════════════════════════════════════════════════════════════
# Working context
# ═══════════════════════════════════════════════════════════════════════════════
# What the conversation is *about*, as distinct from what happens to be loaded.
#
# run_agent() rebuilds its message list from scratch every turn, and carries
# only a few trimmed chat turns (_history_block). Without a focus, an unqualified follow-up — "find
# residues 1-500", "what ligands are there" — carries no trace of the protein
# looked up three messages ago, and the model is free to start the search over
# and land on a different entry. The focus pins that subject: it is set the
# moment a UniProt entry is resolved, carried into the system prompt, shown
# above the chat input, and replaced only when the user names another protein
# or clears it.
#
#   accession / gene / protein_name / organism / length  the UniProt entry
#   query    what the user typed to get here
#   entries  pdb_id → the span of the canonical sequence that entry covers, for
#            every structure of this protein now in the scene. Residue numbers
#            in a deposited file mean nothing without it.


def set_focus(prof: dict) -> None:
    """Make a resolved UniProt entry the subject of the conversation."""
    cur = st.session_state.focus
    keep = cur["entries"] if cur and cur["accession"] == prof["accession"] else {}
    st.session_state.focus = {
        "accession":    prof["accession"],
        "gene":         prof.get("gene") or "",
        "protein_name": prof.get("protein_name") or prof["accession"],
        "organism":     prof.get("organism") or "",
        "length":       prof.get("length") or 0,
        "query":        st.session_state.protein_query,
        "entries":      keep,
    }
    # Adopt anything already in the scene that turns out to be this protein —
    # looking a protein up after loading one of its structures is common.
    for s in structures():
        note_focus_structure(s["pdb_id"])


def clear_focus() -> None:
    """Forget the subject, so the next protein request starts a fresh lookup."""
    st.session_state.focus = None
    st.session_state.protein_query = ""
    st.session_state.protein_hits = []
    st.session_state.protein_profile = None
    st.session_state.protein_msg = None


def note_focus_structure(pdb_id: str) -> None:
    """
    Record a newly loaded entry against the focus, with the residues it covers.

    Called from register_structure, so every route into the scene lands here —
    the Proteins panel, load_protein, or a bare accession typed into chat.
    Entries that do not belong to the focus protein are ignored, so loading
    something unrelated never rewrites the working context.
    """
    focus = st.session_state.focus
    prof = st.session_state.protein_profile
    if not focus or not prof or prof["accession"] != focus["accession"]:
        return
    key = pdb_id.upper()
    row = next((r for r in prof["structures"] if r["pdb_id"].upper() == key), None)
    if not row:
        return
    # UniProt's start/end is the span the deposited construct maps to, which
    # can be far wider than what was resolved: 7MN5's HER2 construct maps to
    # 1-1029, but the file only has residues 24-629. Record what is actually
    # in the file for the focus protein's own chains.
    observed = [c for c in chain_molecules(find_structure(key))
                if any(a.split("-")[0] == focus["accession"] for a in c["uniprot"])]
    focus["entries"][key] = {
        "start":    row.get("start"),
        "end":      row.get("end"),
        "coverage": row.get("coverage_pct"),
        "method":   row.get("method") or "",
        "chains":   [c["chain"] for c in observed] or list(row.get("chains") or []),
        "observed": [(c["chain"], c["first"], c["last"]) for c in observed],
    }


def forget_focus_structures() -> None:
    """Drop the loaded-entry list but keep the protein — the scene was replaced."""
    if st.session_state.focus:
        st.session_state.focus["entries"] = {}


def focus_line() -> str:
    """The working context as one line for the system prompt, or '' if there is none."""
    f = st.session_state.focus
    if not f:
        return ""
    who = f["protein_name"] + (f" ({f['gene']})" if f["gene"] else "")
    out = [f"Working context — the protein this conversation is about: {who}, "
           f"UniProt {f['accession']}"
           + (f", {f['organism']}" if f["organism"] else "")
           + (f", canonical sequence {f['length']} aa" if f["length"] else "") + "."]
    loaded = [k for k in f["entries"] if find_structure(k)]
    if loaded:
        parts = []
        for k in loaded:
            e = f["entries"][k]
            span = (f"construct maps to UniProt residues {e['start']}-{e['end']}"
                    if e["start"] and e["end"] else "span unknown")
            obs = e.get("observed") or []
            if obs:
                span += "; " + ", ".join(
                    f"chain {c} has residues {a}-{b} in the file" for c, a, b in obs)
            parts.append(f"{k} ({e['method'] or 'method unknown'}, {span})")
        out.append("Structures of it already in the scene: " + "; ".join(parts) + ".")
    else:
        out.append("No structure of it is loaded yet.")
    return " ".join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Representation vocabulary
# ═══════════════════════════════════════════════════════════════════════════════
# Every name below was verified against the NGL v2 bundle actually loaded by
# build_ngl_html(). NGL silently ignores an unknown type or colour scheme and
# falls back to a default, so an invalid name fails invisibly — hence the
# curated lists rather than free text.

NGL_REP_TYPES = [
    "cartoon", "surface", "ball+stick", "licorice", "spacefill", "line",
    "backbone", "ribbon", "rope", "tube", "trace", "hyperball", "rocket",
    "base", "point", "helixorient", "axes", "unitcell",
]

# Display label → NGL colour scheme. NGL has no "spectrum" scheme (the name
# appears nowhere in the bundle); the rainbow-by-residue equivalent that
# PyMOL calls spectrum is "residueindex".
NGL_COLOR_SCHEMES = {
    "Rainbow (by residue)": "residueindex",
    "By structure":         BY_STRUCTURE,
    "By element":           "element",
    "By chain (name)":      "chainname",
    "By chain (index)":     "chainindex",
    "By secondary structure": "sstruc",
    "By B-factor":          "bfactor",
    "By residue name":      "resname",
    "By hydrophobicity":    "hydrophobicity",
    "By molecule type":     "moleculetype",
    "By occupancy":         "occupancy",
    "By atom index":        "atomindex",
    "Random":               "random",
    "Red": "red", "Blue": "blue", "Green": "green", "Yellow": "yellow",
    "Orange": "orange", "Magenta": "magenta", "Cyan": "cyan",
    "White": "white", "Grey": "grey", "Salmon": "salmon", "Sky blue": "skyblue",
}

# Display label → NGL selection expression, for the selection dropdown.
SELECTION_PRESETS = {
    "Whole structure":        "all",
    "Protein":                "protein",
    "Ligand / small molecule": "ligand",
    "Nucleic acid":           "nucleic",
    "Water":                  "water",
    "Hetero (non-polymer)":   "hetero",
    "Backbone":               "backbone",
    "Side chains":            "sidechain",
    "Helices":                "helix",
    "Sheets":                 "sheet",
    "Loops / turns":          "turn",
    "Polymer (all chains)":   "polymer",
}

# One-click scenes. Each entry fully replaces the representation stack.
REP_PRESETS = {
    "Cartoon": [
        {"type": "cartoon", "selection": "protein", "color": "residueindex", "transparency": 0.0},
    ],
    "Cartoon + ligand": [
        {"type": "cartoon", "selection": "protein", "color": "residueindex", "transparency": 0.0},
        {"type": "ball+stick", "selection": "ligand", "color": "element", "transparency": 0.0},
    ],
    "Surface": [
        {"type": "surface", "selection": "protein", "color": "hydrophobicity", "transparency": 0.0},
    ],
    "Surface + cartoon": [
        {"type": "cartoon", "selection": "protein", "color": "residueindex", "transparency": 0.0},
        {"type": "surface", "selection": "protein", "color": "grey", "transparency": 0.6},
        {"type": "ball+stick", "selection": "ligand", "color": "element", "transparency": 0.0},
    ],
    "Ball & stick": [
        {"type": "ball+stick", "selection": "all", "color": "element", "transparency": 0.0},
    ],
    "Secondary structure": [
        {"type": "cartoon", "selection": "protein", "color": "sstruc", "transparency": 0.0},
    ],
    "B-factor": [
        {"type": "cartoon", "selection": "protein", "color": "bfactor", "transparency": 0.0},
        {"type": "licorice", "selection": "ligand", "color": "element", "transparency": 0.0},
    ],
    "Chains": [
        {"type": "cartoon", "selection": "polymer", "color": "chainname", "transparency": 0.0},
    ],
}


def _ensure_rep_ids() -> None:
    """
    Give every representation a stable id.

    Streamlit widget keys must not be positional: deleting layer 1 would
    otherwise shift layer 2 into its keys and carry over its stale widget
    values. Ids are assigned lazily so representations created by the agent
    tools (which do not set one) still work.
    """
    for rep in st.session_state.representations:
        if "id" not in rep:
            rep["id"] = uuid.uuid4().hex[:8]


def add_representation(rep_type: str, selection: str, color: str,
                       transparency: float = 0.0, sid: str | None = None) -> None:
    """
    Append a representation layer, replacing any existing layer with the same
    type + selection + structure scope.

    Args:
        rep_type    : NGL representation type.
        selection   : NGL selection string.
        color       : NGL colour scheme, colour name, or BY_STRUCTURE.
        transparency: 0.0 (opaque) to 1.0 (invisible).
        sid         : Structure this layer belongs to, or None for every loaded
                      structure. None is the right default: a layer added
                      before a second structure is loaded should keep applying
                      to both, which is what makes a superposed pair render
                      with matching styling and no extra clicks.
    """
    st.session_state.representations = [
        r for r in st.session_state.representations
        if not (r["type"] == rep_type and r["selection"] == selection
                and r.get("sid") == sid)
    ]
    st.session_state.representations.append({
        "id": uuid.uuid4().hex[:8],
        "type": rep_type, "selection": selection,
        "color": color, "transparency": float(transparency),
        "visible": True, "sid": sid,
    })


def rep_applies_to(rep: dict, sid: str) -> bool:
    """True when a representation layer should be drawn on the given structure."""
    scope = rep.get("sid")
    return scope in (None, "", "*", sid)


def _scope_for(ngl_sel: str):
    """
    Structure scope for a layer built from a resolved selection.

    A selection that went through MDAnalysis can come back as an explicit atom
    serial list ("@12,13,14"). Those serials only mean anything in the
    structure they were computed from, so a layer built from one has to be
    pinned to the active structure — drawn on a second structure it would
    highlight whatever atoms happen to hold those numbers there. Vocabulary
    selections ("protein", ":A", "[ATP]") are portable and stay global, so
    "show cartoon" still styles everything in the scene.
    """
    return st.session_state.active_sid if "@" in (ngl_sel or "") else None


# ═══════════════════════════════════════════════════════════════════════════════
# MDAnalysis helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_universe():
    """
    Return the MDAnalysis Universe for the currently loaded structure.

    Lazily constructs the Universe on first call and caches it in session
    state. Returns None if MDAnalysis is unavailable or no file is loaded.
    """
    if not MDA_AVAILABLE or mda is None:
        return None
    path = st.session_state.pdb_path
    if not path or not Path(path).exists():
        return None
    if st.session_state.universe is None:
        try:
            st.session_state.universe = mda.Universe(path)  # type: ignore[union-attr]
        except Exception:
            return None
    return st.session_state.universe


def mda_to_ngl_serial(ag) -> str:
    """Convert MDAnalysis AtomGroup → NGL @serial selection string."""
    serials = ag.atoms.ids if hasattr(ag.atoms, "ids") else ag.atoms.indices + 1
    if len(serials) == 0:
        return "none"
    return "@" + ",".join(map(str, serials))


def resolve_selection(sel_name_or_expr: str) -> str:
    """
    Resolve a selection identifier to its NGL selection string.

    If the argument matches a key in the named selections dict, returns the
    stored NGL expression. Otherwise passes the value through unchanged,
    assuming it is already a valid NGL expression.

    Args:
        sel_name_or_expr: Named selection key or raw NGL expression.

    Returns:
        NGL selection string ready for use in addRepresentation / setSelection.
    """
    sels = st.session_state.selections
    if sel_name_or_expr in sels:
        return sels[sel_name_or_expr]
    # Translate plain-English phrases. NGL has no "chain" keyword: a raw
    # "chain A" matched nothing, so "protein and not chain A" matched every
    # chain and coloring the rest grey painted chains A and B grey too.
    return _expression_to_ngl(sel_name_or_expr, get_universe())[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Tool implementations
# ═══════════════════════════════════════════════════════════════════════════════

class ToolResult(str):
    """
    What every tool_* returns (suggestion.txt S4): {ok, summary, data, files,
    next_hints}.

    A str whose text is `summary` — what the model reads — so the panels that
    show a tool's reply (st.toast, st.markdown, *_msg session keys) work
    unchanged. run_agent's guards read the fields instead of re-parsing that
    text with regexes:
      ok          False: the call failed and changed nothing.
      data        ids, counts, measured values and flags the result rests on;
                  added to the fact check's evidence (_unsupported_facts).
      files       paths the call wrote.
      next_hints  follow-ups the model may offer the user, never run unasked.
    Concatenating or formatting one returns a plain str without the fields —
    build a new result instead.
    """

    def __new__(cls, summary="", ok: bool = True, data: dict = None,
                files=None, next_hints=None):
        obj = super().__new__(cls, str(summary))
        obj.ok = bool(ok)
        obj.data = dict(data or {})
        obj.files = [str(f) for f in (files or []) if f]
        obj.next_hints = [str(h) for h in (next_hints or []) if h]
        return obj

    @property
    def summary(self) -> str:
        return str.__str__(self)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "summary": self.summary, "data": self.data,
                "files": self.files, "next_hints": self.next_hints}


def _ok(summary, data: dict = None, files=None, hints=None) -> ToolResult:
    """A successful tool result. A ToolResult passed in keeps its own status."""
    if isinstance(summary, ToolResult):
        return summary
    return ToolResult(summary, True, data, files, hints)


def _fail(summary, data: dict = None, hints=None) -> ToolResult:
    """
    A failed tool result: nothing happened. A ToolResult passed in keeps its
    own status — a helper's `err` may be a question for the user (see
    ask_clarification), which is not a failure.
    """
    if isinstance(summary, ToolResult):
        return summary
    return ToolResult(summary, False, data, None, hints)


NO_STRUCTURE = "No structure is loaded — fetch one first."


# Migration fallback only: a tool that still returns a bare string is read
# the way every result was before S4. run_agent logs each one it meets.
_LEGACY_FAILED = re.compile(
    r"^(error|tool error|blocked|warning|unknown tool|no structure|nothing)|"
    r"could not|couldn't|cannot|not found|no such|failed|must be different|"
    r"matched 0 atoms|nothing was", re.IGNORECASE)


def as_tool_result(raw) -> ToolResult:
    """Any tool return value as a ToolResult (adapter for old string returns)."""
    if isinstance(raw, ToolResult):
        return raw
    text = "" if raw is None else str(raw)
    return ToolResult(text, not _LEGACY_FAILED.search(text[:240]))


def _blocked(text: str) -> ToolResult:
    """A call a run_agent gate stopped before it ran."""
    return ToolResult(text, False, {"status": "blocked"})


def _skipped(text: str, ok: bool = True) -> ToolResult:
    """A duplicate call run_agent did not repeat; ok=False if the first one failed."""
    return ToolResult(text, ok, {"status": "skipped"})


def _for_model(result: ToolResult) -> str:
    """A result as the model reads it: the summary, then any follow-up hints."""
    if not result.next_hints:
        return result.summary
    return (f"{result.summary}\n(Possible follow-ups — offer them, do not run them unasked: "
            + "; ".join(result.next_hints) + ")")


# ── Protein lookup: name → UniProt → every PDB structure of that protein ─────
# The PDB's own text search answers "which titles contain these words", which
# is a different question from "which structures are of this protein" and
# quietly mixes in other species and homologues. UniProt knows the protein,
# and cross-references every deposition of it with the residues it covers.

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_protein_search(name: str, organism: str, reviewed_only: bool, schema: int):
    """Cached UniProt name search. `schema` busts the cache when pacc changes."""
    return pacc.search_proteins(name, organism, reviewed_only)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_protein_profile(accession: str, schema: int):
    """Cached full profile for one accession — a dozen HTTP requests at worst."""
    return pacc.profile(accession)


# ── Clarification ────────────────────────────────────────────────────────────
# When a request has more than one reasonable reading, the agent asks instead
# of guessing. A question is stored here with its choices and the request that
# raised it; run_agent returns it to the user at once, the chat shows each
# choice as a button, and the reply ("2", "the second one", the choice's own
# text) re-runs the original request with that meaning spelled out.
#
#   question  one sentence
#   options   [{"label": shown to the user, "meaning": what the choice means,
#              written as an instruction the agent can act on}]
#   request   the user message that raised the question

def ask_clarification(question: str, options: list, request: str = "") -> ToolResult:
    """
    Record a question for the user; returns the tool result for it. Asking is
    not a failure (ok=True); data["needs_user_choice"] marks it.
    """
    opts = []
    for o in options:
        if isinstance(o, dict):
            label = str(o.get("label") or o.get("meaning") or "").strip()
            meaning = str(o.get("meaning") or label).strip()
        else:
            label = meaning = str(o).strip()
        if label:
            opts.append({"label": label, "meaning": meaning})
    if len(opts) == 1:
        return _fail("ask_user needs at least two distinct options, or none for an open question.")
    st.session_state.clarify = {
        "question": question.strip(),
        "options": opts[:5],
        "request": request or st.session_state.get("current_request", ""),
    }
    return _ok("NEEDS USER CHOICE — asked the user: " + clarification_text(),
               {"needs_user_choice": True,
                "options": [o["label"] for o in opts[:5]]})


def clarification_text() -> str:
    """The pending question, as the chat reply shows it."""
    c = st.session_state.get("clarify")
    if not c:
        return ""
    lines = [c["question"]]
    lines += [f"{i}. {o['label']}" for i, o in enumerate(c["options"], 1)]
    return "\n".join(lines)


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "former": 1, "latter": 2, "last": -1}


def resolve_clarification(pending: dict, reply: str):
    """
    Which option a reply to a pending question picks, or None.

    Accepts "2", "option 2", "#2", "the second one", "the latter", the
    option's own text, or a reply containing exactly one option's text.
    """
    opts = pending["options"]
    if not opts:                      # an open question: the reply is the answer
        return None
    r = reply.strip().lower().rstrip(".!")
    m = re.fullmatch(r"(?:option\s*|choice\s*|#|no\.?\s*)?(\d)\)?", r)
    if m and 1 <= int(m.group(1)) <= len(opts):
        return opts[int(m.group(1)) - 1]
    m = re.fullmatch(r"(?:the\s+)?(?:option\s+)?(\w+)(?:\s+(?:one|option|choice))?", r)
    if m and m.group(1) in _ORDINALS:
        i = _ORDINALS[m.group(1)]
        if i == -1 or 1 <= i <= len(opts):
            return opts[i - 1] if i != -1 else opts[-1]
    for o in opts:
        if r == o["label"].lower():
            return o
    hits = [o for o in opts if o["label"].lower() in r]
    return hits[0] if len(hits) == 1 else None


def _hit_exactness(h: dict, q: str) -> int:
    """0 gene/accession, 1 name/alt name, 2 gene synonym, 3 no exact match."""
    q = q.strip().lower()
    if h["gene"].lower() == q or h["accession"].lower() == q:
        return 0
    if h["protein_name"].lower() == q or q in {a.lower() for a in h["alt_names"]}:
        return 1
    if q in {g.lower() for g in h["gene_synonyms"]}:
        return 2
    return 3


def resolve_protein(name: str, organism: str = "human"):
    """
    Look a protein name up and remember the result for the Proteins panel.

    Returns:
        (profile, error). The profile and the full hit list are also written to
        session state so the panel shows whatever the agent just found. When
        the name matches no entry exactly, the error is a NEEDS USER CHOICE
        question instead of a silent pick of the first hit.
    """
    hits, err = _cached_protein_search(name, organism, True, pacc.SCHEMA_VERSION)
    if err:
        return None, err
    if not hits:
        return None, f"No UniProt entry matches '{name}'" + (f" in {organism}." if organism else ".")
    # UniProt's first hit is only trustworthy when the name matches it exactly
    # (gene, accession, name or synonym). Otherwise it is a text-relevance
    # guess: "spike" → moesin, "actin" → gelsolin, "ras" → RIN2, "ubiquitin" →
    # USP46. Ask rather than load the wrong protein.
    focus = st.session_state.get("focus") or {}
    already = name.strip().lower() in {str(focus.get(k, "")).lower()
                                       for k in ("accession", "gene", "query")} - {""}
    if not already and _hit_exactness(hits[0], name) == 3:
        opts = [{"label": f"{h['gene'] or h['accession']} — {h['protein_name']} "
                          f"({h['organism'] or 'unknown organism'}, {h['accession']})",
                 "meaning": f"by '{name}' I mean UniProt {h['accession']} "
                            f"({h['protein_name']}); use the accession {h['accession']} "
                            f"as the protein name"}
                for h in hits[:4]]
        opts.append({"label": f"None of these — search '{name}' in every organism",
                     "meaning": f"search '{name}' with organism='' (any species), then "
                                f"ask me which entry if it is still not an exact match"})
        return None, ask_clarification(
            f"'{name}' does not exactly match a UniProt protein"
            + (f" in {organism}" if organism else "") + ". Which one do you mean?", opts)
    prof, err = _cached_protein_profile(hits[0]["accession"], pacc.SCHEMA_VERSION)
    if err:
        return None, err
    st.session_state.protein_query = name
    st.session_state.protein_hits = hits
    st.session_state.protein_profile = prof
    set_focus(prof)
    return prof, None


def tool_find_protein(name: str, organism: str = "human") -> ToolResult:
    """
    Identify a protein by name and summarise the structures available for it.

    Answers "what structures exist for X" without loading anything: what the
    protein is, how many depositions there are and by which technique, which
    parts of the sequence they cover, and which single file is the best
    starting point.

    Args:
        name    : Protein name, gene symbol or UniProt accession.
        organism: Species name or taxon id. "human" by default; "" for any.

    Returns:
        A plain-text summary, or an error message.
    """
    prof, err = resolve_protein(name, organism)
    if err:
        return _fail(err)
    lines = [pacc.as_brief(prof)]
    if prof["regions"]:
        lines.append("Regions of the sequence with structures:")
        for r in prof["regions"][:6]:
            lines.append(f"  residues {r['start']}-{r['end']} ({r['label']}): "
                         f"{r['count']} structures, best {r['best']}")
    if prof["totals"]["cif_only"]:
        lines.append(f"{prof['totals']['cif_only']} of them are too large for the PDB "
                     f"file format and cannot be loaded here.")
    others = st.session_state.protein_hits[1:5]
    if others:
        lines.append("Other UniProt matches: " + ", ".join(
            f"{h['accession']} ({h['gene'] or h['protein_name']})" for h in others))
    return _ok("\n".join(x for x in lines if x),
               {"accession": prof["accession"], "gene": prof.get("gene"),
                "structures": prof["totals"].get("structures"),
                "regions": [[r["start"], r["end"], r["count"], r["best"]]
                            for r in prof["regions"][:6]],
                "other_matches": [h["accession"] for h in others]},
               hints=["load_protein to load the best structure",
                      "protein_structures to list them"])


def tool_protein_structures(name: str = "", method: str = "",
                            max_resolution: float = 0.0,
                            ligands_only: bool = False, limit: int = 10) -> ToolResult:
    """
    List the PDB structures of a protein as a table, best first.

    Args:
        name          : Protein name or accession. Empty reuses the last lookup.
        method        : "X-ray", "NMR" or "Cryo-EM" to restrict the technique.
        max_resolution: Keep structures at or better than this, in Å. 0 = any.
        ligands_only  : Only structures with a bound ligand or cofactor.
        limit         : How many rows to return.

    Returns:
        A plain-text table, or an error message.
    """
    prof = st.session_state.protein_profile
    if name or prof is None:
        prof, err = resolve_protein(name or st.session_state.protein_query)
        if err:
            return _fail(err)
    rows = pacc.filter_structures(
        prof["structures"], method=method,
        max_resolution=max_resolution if max_resolution and max_resolution > 0 else None,
        ligands_only=ligands_only, loadable_only=True)
    if not rows:
        return _ok(f"No structure of {prof['protein_name']} matches those criteria "
                   f"({prof['totals']['structures']} exist in total).",
                   {"accession": prof["accession"], "matching": 0,
                    "total": prof["totals"]["structures"]})
    who = prof.get("gene") or "this protein"

    def row_line(r):
        res = f"{r['resolution']:.2f} A" if r["resolution"] is not None else "no resolution"
        lig = ", ".join(l["code"] for l in pacc.notable_ligands(r)[:3]) or "none"
        partners = ", ".join((r.get("partners") or [])[:3])
        return (f"  {r['pdb_id']}: {r['method']}, {res}, construct maps to residues "
                f"{r['start']}-{r['end']} ({r['coverage_pct']:.0f}% of the protein), "
                f"{who} is chain {'/'.join(r['chains']) or '?'} in this entry, ligands: {lig}"
                + (f", bound to: {partners}" if partners else ", no other protein")
                + (f" — {r['title'][:60]}" if r["title"] else ""))

    out = [f"{prof['protein_name']} ({prof['accession']}, {prof.get('length') or '?'} aa) — "
           f"{len(rows)} matching structures. Chain letters are labels inside each "
           f"entry file, not parts of the protein: {who} can be chain A in one entry "
           f"and chain B in another. Group answers by region/entry, never by chain letter."]
    # Unfiltered, the question is usually "what is there for this protein" —
    # which parts of it have structures. A flat top-10 by coverage showed only
    # the extracellular-domain complexes for HER2 and hid its kinase domain.
    filtered = bool(method or (max_resolution and max_resolution > 0) or ligands_only)
    regions = prof.get("regions") or []
    shown = set()
    if regions and not filtered:
        keep = {r["pdb_id"] for r in rows}
        for g in regions[:6]:
            ids = [i for i in g["pdb_ids"] if i in keep]
            if not ids:
                continue
            members = pacc.rank_structures([r for r in rows if r["pdb_id"] in ids])[:3]
            out.append(f"Region {g['start']}-{g['end']} ({g['label']}): "
                       f"{len(ids)} structures, best:")
            out.extend(row_line(r) for r in members)
            shown |= {r["pdb_id"] for r in members}
    else:
        for r in pacc.rank_structures(rows)[:max(1, int(limit or 10))]:
            out.append(row_line(r))
            shown.add(r["pdb_id"])
    rest = len(rows) - len(shown)
    if rest > 0:
        out.append(f"{rest} more not listed; filter by method, resolution or ligands to narrow.")
    return _ok("\n".join(out), {"accession": prof["accession"], "matching": len(rows),
                                "listed": sorted(shown)})


def tool_protein_function(name: str = "") -> ToolResult:
    """
    Report a protein's biological role from UniProt's curated annotation.

    Answers "what does X do", "what is X's function", "what does X interact
    with" — biological role, not structure composition (that's
    describe_structure) and not structure availability (that's
    find_protein). Uses the FUNCTION and SUBUNIT comments UniProt curators
    wrote for this entry; never invents a role from the protein's name.

    Args:
        name: Protein name, gene symbol or UniProt accession. Empty reuses
              the last lookup, so this can follow a find_protein call
              without re-naming the protein.

    Returns:
        The FUNCTION and SUBUNIT text UniProt has on file, or a plain
        statement that UniProt has no function annotation for this entry —
        never a guess assembled from the name alone.
    """
    prof = st.session_state.protein_profile
    if name or prof is None:
        prof, err = resolve_protein(name or st.session_state.protein_query)
        if err:
            return _fail(err)
    lines = [f"{prof['protein_name']} ({prof['accession']})"]
    if prof["function"]:
        if prof.get("function_automatic"):
            lines.append("Function (AUTOMATIC annotation — written by UniProt's UniRule/ARBA "
                         "rules from a sequence-family match, not reviewed by a curator): "
                         + prof["function"])
        else:
            lines.append("Function: " + prof["function"])
    if prof["subunit"]:
        lines.append("Subunit structure: " + prof["subunit"])
    data = {"accession": prof["accession"], "curated_function": bool(prof["function"]),
            "automatic": bool(prof["function"] and prof.get("function_automatic")),
            "inferred": False, "structural_neighbours": False}
    if not prof["function"]:
        lines.append("UniProt has no FUNCTION annotation for this entry — this protein "
                     "may be uncharacterized or under-studied.")
        fallback = _function_fallback(prof["accession"])
        neighbours = _structure_function_lines(prof["accession"])
        lines.extend(fallback)
        lines.extend(neighbours)
        # run_agent's caveat footer keys on these, not on the heading text.
        data["inferred"] = any(l.startswith("INFERRED FUNCTION") for l in fallback + neighbours)
        data["structural_neighbours"] = any(
            l.startswith("INFERRED FUNCTION FROM STRUCTURAL NEIGHBOURS") for l in neighbours)
    return _ok("\n".join(lines), data)


# P13: only reached when UniProt has no FUNCTION text; never shown next to one.
_GO_ASPECT = {"F": "Molecular function", "P": "Biological process"}


def _function_fallback(accession: str) -> list:
    """
    Report what UniProt still holds for an entry with no FUNCTION comment.

    GO terms with experimental or curator evidence codes are observations and
    are reported as such. Family membership, InterPro domains, electronically
    transferred GO terms and EC numbers are computational inferences, so they
    go under one explicit LOW CONFIDENCE heading that run_agent keys its
    caveat footer on.
    """
    ev, err = pacc.function_evidence(accession)
    if err:
        return [f"Could not fetch family/GO data to infer a role: {err}"]

    def go_lines(terms):
        out = []
        for aspect, label in _GO_ASPECT.items():
            picked = [f"{t} [{c}]" for a, t, c in terms if a == aspect]
            if picked:
                more = f" (+{len(picked) - 8} more)" if len(picked) > 8 else ""
                out.append(f"  {label}: " + "; ".join(picked[:8]) + more)
        return out

    lines = []
    recorded = go_lines(ev["recorded_go"])
    if recorded:
        lines.append("Recorded GO annotations (experimental or curator evidence code in "
                     "brackets — observations, not predictions):")
        lines.extend(recorded)

    inferred = []
    if ev["family"]:
        inferred.append("  Sequence family: " + "; ".join(ev["family"]))
    if ev["interpro"]:
        inferred.append("  InterPro families/domains: " + "; ".join(
            f"{n} ({i})" for i, n in ev["interpro"][:6]))
    inferred.extend(go_lines(ev["inferred_go"]))
    if ev["ec"]:
        inferred.append("  EC number (rule-assigned): " + ", ".join(ev["ec"]))
    if ev["keywords"]:
        inferred.append("  UniProt keywords: " + ", ".join(n for _, n in ev["keywords"]))
    if inferred:
        lines.append("INFERRED FUNCTION (LOW CONFIDENCE — computational, from sequence-"
                     "family and domain matches, not a curated or measured function):")
        lines.extend(inferred)
        lines.append("Caveat: belonging to a family or carrying a domain suggests a role but "
                     "does not establish it — family members can differ in substrate or "
                     "partner, or have lost activity. Report these as a computational guess, "
                     "not as this protein's known function, and keep this heading's "
                     "LOW CONFIDENCE label in the reply.")
    if not recorded and not inferred:
        lines.append("UniProt has no family, domain or GO information for it either — there "
                     "is nothing to base even a low-confidence guess on. Say so; do not "
                     "guess a role from the name.")
    return lines


def tool_load_protein(name: str, organism: str = "human", prefer: str = "balanced",
                      method: str = "") -> ToolResult:
    """
    Load the best PDB structure of a protein named in words.

    The bridge between the protein layer and the viewer: "load the human
    insulin receptor" picks a file the way the Proteins panel would and adds
    it to the scene, rather than making the user find an accession first.

    Falls back to the AlphaFold DB predicted model when the protein has no
    experimental structure at all (not merely none matching `method`) — most
    UniProt entries have no PDB deposition, and AlphaFold covers nearly all
    of them.

    Args:
        name    : Protein name, gene symbol or UniProt accession.
        organism: Species. "human" by default.
        prefer  : "balanced" (default), "coverage" or "resolution".
        method  : Restrict to a technique, e.g. "X-ray".

    Returns:
        The load confirmation with a line on why that file was chosen.
    """
    prof, err = resolve_protein(name, organism)
    if err:
        return _fail(err)
    rows = pacc.filter_structures(prof["structures"], method=method, loadable_only=True)
    if not rows:
        # A method filter narrowing existing structures to zero is a different
        # situation from the protein having none at all — only the latter
        # should fall back to a predicted model, or the filter would be
        # silently ignored.
        any_rows = (pacc.filter_structures(prof["structures"], loadable_only=True)
                    if method else rows)
        if any_rows:
            return _fail(f"{prof['protein_name']} ({prof['accession']}) has no structure "
                         f"that can be loaded here by {method}.",
                         {"accession": prof["accession"], "loadable": len(any_rows)})
        dest, af_err = download_alphafold(prof["accession"])
        if af_err:
            return _fail(f"{prof['protein_name']} ({prof['accession']}) has no experimental "
                         f"structure in the PDB, and {af_err[0].lower()}{af_err[1:]}",
                         {"accession": prof["accession"]})
        label = f"AF-{prof['accession']}"
        first = not structures()
        note = "" if first else _recolor_for_comparison()
        register_structure(label, dest, source="alphafold")
        st.session_state.camera_target = None
        if first and not st.session_state.representations:
            st.session_state.representations = [dict(r, id=uuid.uuid4().hex[:8])
                                                for r in DEFAULT_REPS]
        return _ok(f"{prof['protein_name']} ({prof['accession']}) has no experimental "
                   f"structure in the PDB — loaded the AlphaFold predicted model "
                   f"({label}) instead. Confidence (pLDDT) varies by residue; treat "
                   f"low-confidence regions as illustrative, not a fitted structure."
                   f"{note}",
                   {"accession": prof["accession"], "loaded": label, "alphafold": True},
                   files=[dest])
    best = pacc.rank_structures(rows, prefer)[0]
    msg = tool_add_structure(best["pdb_id"])
    if not msg.ok:
        return msg
    res = f"{best['resolution']:.2f} Å" if best["resolution"] is not None else "no resolution"
    # Say which chain is the protein asked for, and what else is in the file —
    # the top-ranked entry is often a complex, and its chain A is not
    # necessarily the protein that was searched for.
    entry = find_structure(best["pdb_id"])
    chains = chain_molecules(entry)
    ours = [c for c in chains
            if any(a.split("-")[0] == prof["accession"] for a in c["uniprot"])]
    if ours:
        where = "; ".join(f"chain {c['chain']}, residues {c['first']}-{c['last']} resolved"
                          for c in ours)
        others = [c for c in chains if c not in ours]
        where = (f" {prof.get('gene') or prof['protein_name']} is {where}"
                 + (" (the construct maps to UniProt "
                    f"{best['start']}-{best['end']}, but only the resolved residues are in the file)"
                    if best["start"] and best["end"] else ""))
        if others:
            where += (". The file also contains " + "; ".join(
                f"chain {c['chain']} = {c['molecule'] or 'unnamed'}" for c in others))
        drawn = chains_drawn(entry)
        if drawn and len(drawn) == len(chains):
            where += f". All {len(chains)} chains are drawn in the viewer"
        where += "."
    else:
        where = (f" Residues {best['start']}–{best['end']} of the sequence "
                 f"({best['coverage_pct']:.0f}%).")
    return _ok(f"{msg} {best['pdb_id']} is the {prefer} choice for "
               f"{prof['protein_name']} ({prof['accession']}): {best['method']}, {res}"
               + (f", {best['title']}" if best["title"] else "") + "."
               + where
               + f" {len(rows)} structures of this protein are loadable in total "
                 "(protein_structures lists them).",
               {"accession": prof["accession"], "loaded": best["pdb_id"],
                "method": best["method"], "resolution": best["resolution"],
                "protein_chains": [c["chain"] for c in ours],
                "chains": {c["chain"]: c["molecule"] for c in chains},
                "loadable": len(rows)},
               files=msg.files)


# ── Structure preparation ────────────────────────────────────────────────────
# The step between downloading a structure and doing anything quantitative
# with it. A deposited file is a record of an experiment, not a model of a
# molecule: twenty NMR states, side chains modelled in two places at once, and
# the cryoprotectant still in the box.

@st.cache_data(show_spinner=False)
def _cached_inspection(path: str, mtime: float, schema: int):
    """Findings for a file, cached on its contents and the module's schema."""
    return prp.inspect(path)


def inspection_for(target: str = ""):
    """The inspect() findings for a loaded structure, or None."""
    entry = find_structure(target) if target else active_structure()
    if not entry or not Path(entry["path"]).exists():
        return None, None
    return entry, _cached_inspection(entry["path"],
                                     Path(entry["path"]).stat().st_mtime,
                                     prp.SCHEMA_VERSION)


def tool_inspect_preparation(target: str = "") -> ToolResult:
    """
    Say what is wrong with a structure before anyone simulates it.

    Reports the things that make a deposited file unusable as a model —
    multiple NMR states, alternate conformations, crystallisation additives,
    missing residues, ligands with no parameters — without changing anything.

    Args:
        target: PDB id to check. Empty for the active structure.

    Returns:
        A plain-text checklist.
    """
    if not structures():
        return _fail(NO_STRUCTURE)
    entry, finding = inspection_for(target)
    if not finding:
        return _fail(f"Could not read '{target or 'the active structure'}'.")
    lines = [f"{entry['pdb_id']}: {finding['atoms']} atoms, {finding['residues']} "
             f"residues, {finding['models']} model(s), {finding['waters']} waters, "
             f"{finding['hydrogens']} hydrogens."]
    if not finding["issues"]:
        lines.append("Nothing needs fixing before simulation.")
    for issue in finding["issues"]:
        lines.append(f"[{issue['level']}] {issue['title']} — {issue['detail']}")
    return _ok("\n".join(lines),
               {"pdb_id": entry["pdb_id"],
                **{k: finding[k] for k in ("atoms", "residues", "models", "waters", "hydrogens")},
                "issues": [i["title"] for i in finding["issues"]]},
               hints=["prepare_structure to fix them"] if finding["issues"] else None)


def prepare_structure(target: str = "", profile: str = "amber", model=None,
                      chains=None, add_h: bool = False, **overrides):
    """
    Clean a loaded structure and register the result.

    Returns:
        (record, error). The record holds the prepared file's path, the report
        of what changed, and the findings for the cleaned structure.
    """
    entry, finding = inspection_for(target)
    if not entry:
        return None, "No structure is loaded to prepare."
    if not finding:
        return None, f"Could not read {entry['pdb_id']}."

    settings = dict(prp.PROFILES.get(profile, prp.PROFILES["amber"]))
    settings.pop("label", None)
    settings.pop("note", None)
    settings.update({k: v for k, v in overrides.items() if v is not None})
    if model is None:
        model = finding.get("representative_model") or finding["model_ids"][0]

    dest = PREPARED_DIR / f"{entry['pdb_id']}_{profile}.pdb"
    path, report = prp.prepare(entry["path"], dest, model=model, chains=chains,
                               finding=finding,
                               provenance=f"{profile} profile, from {entry['pdb_id']}",
                               **settings)
    if path is None:
        return None, report.get("error", "Preparation failed.")

    if add_h:
        with_h = dest.with_name(dest.stem + "_H.pdb")
        hpath, added, err = prp.add_hydrogens(path, with_h)
        if err:
            report["notes"].append(f"Hydrogens were not added: {err}")
        else:
            path, report["hydrogens_added"] = hpath, added
            # The report's atom count was taken before protonation; left alone
            # it would describe a file that no longer exists.
            report["atoms_after"] = sum(
                1 for line in prp.read_lines(path) if prp._is_coord(line))

    record = {
        "path": path, "report": report, "profile": profile,
        "finding": prp.inspect(path), "source": entry["pdb_id"],
    }
    st.session_state.prepared[entry["pdb_id"]] = record
    return record, None


def tool_prepare_structure(target: str = "", profile: str = "amber",
                           model: int = 0, add_hydrogens: bool = False,
                           keep_waters=None, keep_ligands=None) -> ToolResult:
    """
    Clean a structure up and load the cleaned copy.

    Keeps one model of an NMR ensemble, resolves alternate conformations,
    drops crystallisation additives, and renames disulfide cysteines for
    Amber. What exactly it does depends on the profile.

    Args:
        target       : PDB id to prepare. Empty for the active structure.
        profile      : "amber", "rosetta", "md_explicit" or "clean".
        model        : Which NMR state to keep. 0 means the representative one.
        add_hydrogens: Also run `reduce` to protonate it.
        keep_waters  : Override the profile's choice about waters.
        keep_ligands : Override the profile's choice about ligands.

    Returns:
        A description of what changed, or an error message.
    """
    if not structures():
        return _fail(NO_STRUCTURE)
    overrides = {}
    if keep_waters is not None:
        overrides["keep_waters"] = bool(keep_waters)
    if keep_ligands is not None:
        overrides["keep_ligands"] = bool(keep_ligands)
    record, err = prepare_structure(target, profile, model=int(model) or None,
                                    add_h=bool(add_hydrogens), **overrides)
    if err:
        return _fail(err)

    label = f"{record['source']}_prep"
    reg = register_structure(label, record["path"], source="local")
    source_entry = find_structure(record["source"])
    if source_entry and source_entry["sid"] != reg["sid"]:
        source_entry["visible"] = False
    st.session_state.prepare_msg = None

    outstanding = [c["text"] for c in prp.readiness(record["finding"],
                                                    "rosetta" if profile == "rosetta"
                                                    else "amber") if not c["ok"]]
    out = [f"Prepared {record['source']} with the {profile} profile and loaded it "
           f"as {label}: {prp.report_text(record['report'], oneline=True)}"]
    if outstanding:
        out.append("Still outstanding: " + "; ".join(outstanding))
    rep_ = record["report"]
    return _ok(" ".join(out),
               {"source": record["source"], "loaded": label, "profile": profile,
                "atoms_after": rep_.get("atoms_after"),
                "hydrogens_added": rep_.get("hydrogens_added"),
                "disulfide_cysteines": rep_.get("cys_renamed"),
                "outstanding": outstanding},
               files=[record["path"]])


# ── Membrane embedding ───────────────────────────────────────────────────────
# Two steps with very different costs: orienting the protein across the
# bilayer takes seconds and is done inline, packing the lipids takes minutes
# to hours and runs as a background job. See membrane.py for why.

# NGL selections for the parts of a packed system. Everything that is not
# protein, water, ion or a membrane-plane marker is lipid.
LIPID_SELECTION = "not protein and not water and not ion and not DUM"
PLANE_SELECTION = "DUM"


def _membrane_layers(sid: str, planes: bool = True) -> None:
    """
    Style a structure that has a membrane: lipids as sticks, planes as points.

    Without this a packed system renders as the default protein cartoon
    floating in an empty box -- every lipid loaded, none of them drawn.
    """
    add_representation("licorice", LIPID_SELECTION, "element", 0.0, sid)
    if planes:
        # The DUM markers are a dense grid on both bilayer planes, so drawn as
        # points they read as the membrane slab itself -- the cheapest
        # possible way to see whether the protein sits in it properly.
        add_representation("point", PLANE_SELECTION, "lightgrey", 0.5, sid)


def _load_oriented(entry: dict, record: dict) -> str:
    """
    Put an oriented copy in the scene and hide the structure it came from.

    Hiding rather than replacing: the oriented copy is the same protein in a
    different pose, so leaving both visible draws the molecule twice at two
    angles, which looks like a superposition that went wrong. The original
    stays loaded and one click away in the Structures tab.
    """
    label = f"{entry['pdb_id']}_mem"
    reg = register_structure(label, record["path"], source="local")
    _membrane_layers(reg["sid"])
    if reg["sid"] != entry["sid"]:
        entry["visible"] = False
    return label


def _oriented_entry(target: str = ""):
    """The oriented copy recorded for a structure, or None."""
    entry = find_structure(target) if target else active_structure()
    if not entry:
        return None, None
    return entry, st.session_state.oriented.get(entry["pdb_id"])


def orient_structure(target: str = "", source: str = "auto", n_ter: str = "in",
                     barrel: bool = False):
    """
    Orient a loaded structure across the membrane and remember the result.

    Args:
        target: PDB id or label. Empty means the active structure.
        source: "opm" (published orientation), "memembed" (computed here) or
                "auto" — OPM when the entry is in it, MEMEMBED otherwise.
        n_ter : "in" or "out" — which side of the membrane residue 1 starts on.
        barrel: Beta-barrel mode, for porins and outer-membrane proteins.

    Returns:
        (record, error). The record holds the oriented file's path, where the
        orientation came from, and the bilayer thickness it implies.
    """
    entry = find_structure(target) if target else active_structure()
    if not entry:
        return None, "No structure is loaded to orient."
    pdb_id = entry["pdb_id"]
    job_dir = MEMBRANES_DIR / f"{pdb_id}_oriented"
    job_dir.mkdir(parents=True, exist_ok=True)

    if mem.is_oriented(entry["path"]):
        record = {"path": entry["path"], "source": "the file itself",
                  "info": {"planes": mem.membrane_planes(entry["path"])}}
        st.session_state.oriented[pdb_id] = record
        return record, None

    if source in ("auto", "opm"):
        path, info, err = mem.fetch_opm(pdb_id, job_dir)
        if path and not err:
            record = {"path": path, "source": "OPM", "info": info}
            st.session_state.oriented[pdb_id] = record
            return record, None
        if source == "opm":
            return None, err or f"{pdb_id} is not in OPM."

    path, info, err = mem.orient(entry["path"], job_dir, n_ter=n_ter, barrel=barrel)
    if err:
        return None, err
    record = {"path": path, "source": "MEMEMBED", "info": info}
    st.session_state.oriented[pdb_id] = record
    return record, None


def tool_orient_membrane(target: str = "", source: str = "auto",
                         n_ter: str = "in", barrel: bool = False) -> ToolResult:
    """
    Place a membrane protein in the bilayer and show it there.

    Uses OPM's published orientation when the entry is in that database and
    computes one with MEMEMBED otherwise. The oriented copy is loaded into the
    viewer with the two membrane planes drawn, so the orientation can be
    checked before anything expensive is done with it.

    Args:
        target: PDB id of the structure to orient. Empty for the active one.
        source: "auto" (default), "opm" or "memembed".
        n_ter : "in" or "out" — which side residue 1 starts on.
        barrel: True for beta-barrel outer-membrane proteins.

    Returns:
        A description of the orientation, or an error message.
    """
    if not structures():
        return _fail(NO_STRUCTURE)
    record, err = orient_structure(target, source, n_ter, bool(barrel))
    if err:
        return _fail(err)
    entry = find_structure(target) if target else active_structure()
    label = _load_oriented(entry, record)
    planes = record["info"].get("planes")
    thickness = (round(planes[1] - planes[0], 1) if planes else None)
    bits = [f"Oriented {entry['pdb_id']} in the membrane using {record['source']}",
            f"loaded as {label}"]
    if thickness:
        bits.append(f"hydrophobic thickness {thickness} Å")
    if record["source"] == "OPM" and record["info"].get("name"):
        bits.append(f"OPM calls it {record['info']['name']}")
    if record["source"] == "MEMEMBED":
        bits.append("this is a computed orientation — check it before building on it")
    return _ok(". ".join(bits) + ".",
               {"pdb_id": entry["pdb_id"], "loaded": label, "source": record["source"],
                "thickness": thickness},
               files=[record["path"]], hints=["build_membrane to pack lipids around it"])


def tool_build_membrane(target: str = "", composition: str = "popc",
                        salt_concentration: float = 0.15,
                        lipids: str = "", ratios: str = "") -> ToolResult:
    """
    Start packing a lipid bilayer around a membrane protein.

    The build runs in the background because it takes minutes to hours. This
    returns as soon as it has started; call membrane_status to find out how it
    is getting on.

    Args:
        target            : PDB id to embed. Empty for the active structure.
        composition       : A preset key (popc, popc_chol, plasma_sym,
                            plasma_asym, raft, ecoli, er, mito, dppc, dmpc).
        salt_concentration: Molar. 0 for no salt.
        lipids            : Explicit PACKMOL-Memgen lipid string, overriding the
                            preset, e.g. "POPC:CHL1" or "POPE:POPG//POPC".
        ratios            : Matching ratio string, e.g. "3:1".

    Returns:
        Confirmation that the build started, or an error message.
    """
    if not mem.available():
        return _fail(mem.backend_report(), {"unavailable": "packmol-memgen"})
    if not structures():
        return _fail(NO_STRUCTURE)
    job = st.session_state.membrane_job
    if job and job.get("status") == "running":
        return _fail(f"A membrane build for {job['label']} is already running "
                     f"({job.get('stage', 'running')}). Wait for it or cancel it first.",
                     hints=["membrane_status"])

    entry = find_structure(target) if target else active_structure()
    if not entry:
        return _fail(f"No structure called '{target}' is loaded. Loaded: "
                     + (", ".join(x["pdb_id"] for x in structures()) or "none") + ".")

    record, err = orient_structure(entry["pdb_id"])
    if err:
        return _fail(f"Could not orient {entry['pdb_id']}: {err}")

    if lipids and ratios:
        lip, rat, described = lipids, ratios, f"{lipids} in {ratios}"
    else:
        preset = mem.PRESET_BY_KEY.get((composition or "popc").lower())
        if not preset:
            return _fail("Unknown composition. Available presets: "
                         + ", ".join(p["key"] for p in mem.PRESETS))
        lip, rat = mem.composition_args(preset["lower"], preset["upper"])
        described = preset["label"]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    job = mem.start_build(
        record["path"], MEMBRANES_DIR / f"{entry['pdb_id']}_{stamp}", lip, rat,
        label=entry["pdb_id"], preoriented=True,
        salt=bool(salt_concentration), salt_concentration=salt_concentration or 0.15)
    st.session_state.membrane_job = job
    if job["status"] == "failed":
        return _fail(job["error"])
    return _ok(f"Started packing {described} around {entry['pdb_id']} "
               f"(oriented by {record['source']}). This runs in the background and "
               f"takes minutes to hours depending on the protein — ask for the "
               f"membrane status to check on it, or watch the Membrane tab.",
               {"pdb_id": entry["pdb_id"], "lipids": lip, "ratios": rat,
                "status": "running"},
               hints=["membrane_status"])


def tool_membrane_status() -> ToolResult:
    """
    Report how the membrane build is getting on, and load it when it is done.

    Returns:
        A status line, or a description of the finished system.
    """
    job = st.session_state.membrane_job
    if not job:
        return _ok("No membrane build has been started in this session.", {"status": None})
    mem.poll(job)
    minutes = job.get("elapsed", 0) / 60
    data = {"pdb_id": job["label"], "status": job["status"], "minutes": round(minutes, 1)}
    if job["status"] == "running":
        return _ok(f"Building a membrane for {job['label']}: {job['stage']}, "
                   f"{minutes:.1f} minutes in.", data)
    if job["status"] == "cancelled":
        return _ok(f"The membrane build for {job['label']} was cancelled.", data)
    if job["status"] == "failed":
        # The status report itself worked; the build it reports on did not.
        return _ok(f"The membrane build for {job['label']} failed: {job['error']}", data)
    summary = job.get("summary") or mem.system_summary(job["output"])
    return _ok(f"The membrane system for {job['label']} finished in "
               f"{minutes:.1f} minutes. {mem.summary_text(summary)} "
               f"It is at {job['output']}; load it from the Membrane tab "
               f"(the viewer copy leaves the water out).", data, files=[job["output"]])


# ── Simulation setup ─────────────────────────────────────────────────────────
# One code path for the Simulate panels and the agent (suggestion.txt S3):
# the *_ui panels collect widget values and call these; the tool_* wrappers
# below collect tool arguments and call the same ones.

PREPARED_SUFFIXES = ("_PREP", "_MEM", "_MEMBRANE")


def default_simulation_name() -> str:
    """The newest prepared copy in the scene, else the active structure."""
    names = [x["pdb_id"] for x in structures()]
    if not names:
        return ""
    prepared = [n for n in names if n.upper().endswith(PREPARED_SUFFIXES)]
    if prepared:
        return prepared[-1]
    active = active_structure()
    return active["pdb_id"] if active else names[0]


def simulation_target(target: str = ""):
    """(name, entry) to set a calculation up from; (None, None) if not loaded."""
    entry = find_structure(target or default_simulation_name())
    return (entry["pdb_id"], entry) if entry else (None, None)


def write_simulation_files(files: dict, dest: Path) -> Path:
    """
    Write a setup's files into `dest`, and the same set as `dest`.zip.

    Same value convention as sim.bundle: a Path is copied, str/bytes is content.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        if content is None:
            continue
        out = dest / fname
        if isinstance(content, Path):
            if content.exists() and content.resolve() != out.resolve():
                shutil.copyfile(content, out)
        elif isinstance(content, (bytes, bytearray)):
            out.write_bytes(content)
        else:
            out.write_text(str(content))
    archive = dest.with_suffix(".zip")
    archive.write_bytes(sim.bundle(files))
    return archive


def parameterize_ligand(name: str, entry: dict, code: str, charge: int,
                        method: str = "bcc") -> dict:
    """Run antechamber + parmchk2 on one component; stores and returns the record."""
    work = SIMULATIONS_DIR / f"{name}_params" / code
    ligand_pdb, _, _ = sim.extract_residue(entry["path"], code, work / f"{code}_raw.pdb")
    mol2, frcmod, log, err = sim.run_antechamber(
        ligand_pdb, work, code, net_charge=charge, charge_method=method)
    stored = {
        "mol2": mol2, "frcmod": frcmod, "charge": charge,
        "error": err, "log": log,
        "guessed": sim.missing_parameters(frcmod) if frcmod else [],
    }
    st.session_state.ligand_params[(name, code)] = stored
    return stored


def stored_ligand_params(name: str, entry: dict) -> tuple:
    """([(code, mol2, frcmod)] parameterised, [codes still without parameters])."""
    params, missing = [], []
    for lig in sim.ligand_candidates(entry["path"]):
        stored = st.session_state.ligand_params.get((name, lig["code"]))
        # A prepared copy (3PTB_PREP) keeps its source's ligand coordinates,
        # so the source's parameters apply: "parameterize BEN", then a
        # prepare_structure the model ran itself, made setup_amber refuse.
        base = name
        while not (stored and not stored.get("error")):
            suffix = next((s for s in PREPARED_SUFFIXES if base.upper().endswith(s)), None)
            if not suffix:
                break
            base = base[:-len(suffix)]
            stored = st.session_state.ligand_params.get((base, lig["code"]))
        if stored and not stored.get("error"):
            params.append((lig["code"], stored["mol2"], stored["frcmod"]))
        else:
            missing.append(lig["code"])
    return params, missing


def build_amber_system(name: str, entry: dict, params: list, ff: str, water: str,
                       shape: str, buffer_a: float, salt: float, pair: str) -> dict:
    """Write the leap script and run tleap; stores and returns the result."""
    cation, anion, _ = sim.ION_PAIRS[pair]
    source_pdb = entry["path"]
    work = SIMULATIONS_DIR / f"{name}_amber"
    work.mkdir(parents=True, exist_ok=True)
    structure_path = source_pdb
    combine = []
    ligand_codes = [code for code, _, _ in params]
    if ligand_codes:
        # The ligand comes in from its mol2, so it has to come out of the
        # PDB: antechamber renames atoms when it reads real bond orders,
        # and leap matches residues by atom name.
        stripped = work / (Path(source_pdb).stem + "_noligand.pdb")
        structure_path, _ = sim.strip_components(source_pdb, ligand_codes, stripped)
        combine = [(code, mol2) for code, mol2, _ in params]
    # tleap reads CONECT records as covalent bonds (unprepared 3PTB: "no bond
    # parameter for Ca2+ - OW"); the script's disulfide bond commands need
    # CYX, which only prepare_structure renamed (SH - SH otherwise), and
    # sequential numbering, without which tleap_script comments them out
    # (3PTB's chymotrypsin numbering: 12 CYX, zero S-S bonds). Fix those three
    # and change nothing else, so an unprepared file builds too.
    leap_in, _ = prp.prepare(structure_path, work / (Path(structure_path).stem + "_leapin.pdb"),
                             altloc="keep", keep_waters=True, keep_ions=True,
                             keep_ligands=True, keep_additives=True, hydrogens="keep",
                             cys_to_cyx=True, renumber=True)
    structure_path = leap_in or structure_path
    finding = prp.inspect(structure_path)
    script = prp.tleap_script(
        structure_path, finding, unit="prot", ff=ff, water=water,
        solvate=True, box=buffer_a, box_shape=shape,
        neutralise=True, cation=cation, anion=anion,
        ion_counts=sim.ion_counts(
            sim.estimate_waters(structure_path, buffer_a, shape), salt,
            cation, anion, sim.estimate_charge(structure_path)),
        ligand_params=params, combine_units=combine,
        lipid=bool(mem.membrane_planes(structure_path)),
        outputs=("system.prmtop", "system.inpcrd"))
    extra = [structure_path] + [p for _, m, f in params for p in (m, f)]
    result = sim.run_tleap(script, work, extra_files=extra)
    result["script"] = script
    result["name"] = name
    st.session_state.amber_build = result
    return result


def amber_run_files(entry: dict, build, params: list, temperature: float,
                    pressure: float, ns: float, engine: str) -> dict:
    """The five mdin stages, run.sh, and the topology when `build` succeeded."""
    is_membrane = bool(mem.membrane_planes(entry["path"]))
    files = {filename: sim.amber_mdin(stage, membrane_system=is_membrane,
                                      temperature=temperature, pressure=pressure,
                                      nanoseconds=ns)
             for filename, stage in sim.AMBER_STAGES}
    if build and build.get("ok"):
        files["system.prmtop"] = Path(build["prmtop"])
        files["system.inpcrd"] = Path(build["inpcrd"])
        files["build.leap"] = build.get("script", "")
        files["leap.log"] = build["log"]
        for code, mol2, frcmod in params:
            files[f"{code}.mol2"] = Path(mol2)
            files[f"{code}.frcmod"] = Path(frcmod)
    files["run.sh"] = sim.amber_run_script("system.prmtop", "system.inpcrd", engine)
    return files


def gromacs_run_files(entry: dict, temperature: float, pressure: float,
                      ns: float) -> dict:
    """em/nvt/npt/md .mdp files plus the topology-conversion README."""
    is_membrane = bool(mem.membrane_planes(entry["path"]))
    files = {filename: sim.gromacs_mdp(stage, temperature, pressure, ns,
                                       membrane_system=is_membrane)
             for filename, stage in sim.GROMACS_STAGES}
    files["README.txt"] = sim.GROMACS_CONVERSION_NOTE
    return files


def rosetta_job_files(name: str, entry: dict, code: str, chain: str,
                      nstruct: int) -> dict:
    """Ligand-docking XML, options, params command, README, the structure, mol2 if any."""
    files = {
        "dock.xml": sim.rosetta_ligand_docking_xml(code, chain),
        "dock.options": sim.rosetta_options(Path(entry["path"]).name, code, nstruct),
        "params_command.txt": sim.rosetta_params_command(code),
        "README.txt": sim.ROSETTA_README,
        Path(entry["path"]).name: Path(entry["path"]),
    }
    stored = st.session_state.ligand_params.get((name, code))
    if stored and not stored.get("error"):
        files[f"{code}.mol2"] = Path(stored["mol2"])
    return files


def build_qm_region(name: str, entry: dict, codes: list, residue_keys: list,
                    side_chains_only: bool, center_keys=None) -> dict:
    """Cut the QM cluster out; stores and returns the region."""
    region = qm.build_region(entry["path"], center_codes=codes, center_keys=center_keys,
                             residue_keys_wanted=residue_keys,
                             side_chains_only=side_chains_only)
    region["name"] = name
    st.session_state.qm_region = region
    return region


def qm_input_files(region: dict, charge: int, multiplicity: int, method: str,
                   basis: str, job: str, solvent: str, processors: int,
                   memory_gb: int, freeze: bool) -> dict:
    """Gaussian, ORCA, Psi4 and xyz inputs for one region, plus its notes."""
    return {
        "region.gjf": qm.gaussian_input(region, charge, multiplicity, method, basis,
                                         job, solvent, processors, memory_gb, freeze),
        "region.inp": qm.orca_input(region, charge, multiplicity, method, basis,
                                     job, solvent, processors, memory_gb * 1000 // 8,
                                     freeze),
        "region.psi4": qm.psi4_input(region, charge, multiplicity, method, basis,
                                      job, memory_gb),
        "region.xyz": qm.xyz_file(region),
        "region_notes.txt": qm.region_report(region),
    }


def oniom_pairing(build: dict):
    """(leap_pdb, paired, error) for a successful Amber build."""
    # The file this build's own `savepdb` wrote: a glob picked a stale
    # *_leap.pdb from an earlier build of another input name (38,414 vs
    # 38,454 atoms on 3PTB).
    saved = re.search(r"^savepdb\s+\S+\s+(\S+)", build.get("script", ""), re.M)
    leap_pdbs = ([Path(build["dir"]) / saved.group(1)] if saved else
                 sorted(Path(build["dir"]).glob("*_leap.pdb"),
                        key=lambda p: p.stat().st_mtime, reverse=True))
    leap_pdbs = [p for p in leap_pdbs if p.exists()]
    if not leap_pdbs:
        return None, None, ("The build directory has no leap-written PDB. Rebuild the "
                            "topology — `savepdb` is what makes the atom order match.")
    paired = oniom.pair_with_structure(leap_pdbs[0], oniom.read_topology(build["prmtop"]))
    if not paired["ok"]:
        return leap_pdbs[0], None, paired["error"]
    return leap_pdbs[0], paired, None


def build_oniom_model(build: dict, paired: dict, codes: list, residue_keys: list,
                      side_chains_only: bool, sphere: float) -> dict:
    """Assign QM/MM layers and the boundary; stores and returns the model."""
    layered = oniom.assign_layers(paired, residue_keys, high_codes=codes,
                                  side_chains_only=side_chains_only,
                                  sphere_radius=sphere)
    model = {
        "layered": layered,
        "boundary": oniom.find_boundary(layered),
        "bonds": oniom.topology_bonds(build["prmtop"]),
        "name": build.get("name"),
    }
    st.session_state.oniom_model = model
    return model


def oniom_input_files(model: dict, method: str, embedding: str, job: str,
                      charge: int, multiplicity: int, processors: int,
                      freeze_mm: bool) -> dict:
    """The Gaussian ONIOM input and its notes."""
    layered, boundary = model["layered"], model["boundary"]
    text = oniom.gaussian_oniom_input(
        layered, boundary, method, embedding, charge, multiplicity,
        job=job, freeze_mm=freeze_mm, processors=processors, bonds=model["bonds"])
    return {"oniom.gjf": text, "oniom_notes.txt": oniom.report(layered, boundary)}


# ── Simulation setup: agent tools ────────────────────────────────────────────

def _pick_option(value, options, default: str = ""):
    """Case/punctuation-insensitive match of `value` to a key of `options`."""
    if not _given(value):
        return default
    norm = lambda s: re.sub(r"[^a-z0-9+]", "", str(s).lower())
    want = norm(value)
    return next((k for k in options if norm(k) == want), None)


def _bad_option(what: str, value, options) -> ToolResult:
    return _fail(f"Error: unknown {what} '{value}'. Options: " + ", ".join(options) + ".",
                 {"bad_option": what, "options": list(options)})


# Common words for components whose dictionary name does not contain them
# (HEM is "PROTOPORPHYRIN IX CONTAINING FE").
_LIGAND_WORDS = {
    "heme": {"HEM", "HEC", "HEA", "HEB"}, "haem": {"HEM", "HEC", "HEA", "HEB"},
    "hemes": {"HEM", "HEC", "HEA", "HEB"},
}


def _pick_ligand(entry: dict, code: str, purpose: str):
    """
    (code, None) for the component to use, or (None, message).

    Empty `code` means "the ligand": the only component, or the only one that
    is not a crystallisation additive; several real candidates → ask_user.
    """
    candidates = sim.ligand_candidates(entry["path"])
    if code:
        hit = next((c for c in candidates if c["code"].upper() == str(code).strip().upper()), None)
        if hit:
            return hit["code"], None
        # A word for the ligand instead of its code: qwen2.5:14b sent
        # setup_qm(center='HEME') for "QM around the heme" on 4HHB (HEM).
        word = str(code).strip().upper()
        codes = _LIGAND_WORDS.get(word.lower(), set())
        named = [c for c in candidates
                 if c["code"] in codes
                 or (len(c["code"]) >= 3 and word.startswith(c["code"]))
                 or (len(word) >= 4 and word in str(c.get("name", "")).upper())]
        if len({c["code"] for c in named}) == 1:
            return named[0]["code"], None
        have = ", ".join(c["code"] for c in candidates) or "none"
        return None, (f"Error: {entry['pdb_id']} has no component '{code}'. "
                      f"Components that could be used: {have}.")
    if not candidates:
        return None, (f"{entry['pdb_id']} has no ligand or cofactor to {purpose} — "
                      f"nothing was set up.")
    real = [c for c in candidates if c["kind"] != "additive"] or candidates
    if len(real) == 1:
        return real[0]["code"], None
    return None, ask_clarification(
        f"{entry['pdb_id']} has several components. Which one should I {purpose}?",
        [{"label": f"{c['code']} — {srep.pretty_chemical(c['name']) or c['kind']}",
          "meaning": f"use {c['code']}"} for c in real])


def _unprepared_note(name: str) -> str:
    if name.upper().endswith(PREPARED_SUFFIXES):
        return ""
    return (f"Note: {name} is not a prepared copy — alternate conformations, extra "
            f"NMR states and additives become clashes here, not errors. Run "
            f"prepare_structure first for a production setup. ")


def _dictionary_charge(code: str):
    """Formal charge from the PDB chemical dictionary, or None if unknown."""
    try:
        info = sim.chem_component(code) or {}
    except Exception:
        return None
    q = info.get("charge")
    return int(q) if q is not None else None


def _dictionary_note(code: str) -> str:
    q = _dictionary_charge(code)
    if q is None:
        return f"The PDB chemical dictionary records no charge for {code}."
    return (f"The PDB chemical dictionary records {q:+d} for the deposited form, which "
            f"is not necessarily its state at pH 7.")


def _files_line(files: dict, dest: Path, archive: Path) -> str:
    return (f"Files written to {dest}/ ({', '.join(sorted(files))}); "
            f"the same set zipped as {archive}.")


def _written(files: dict, dest: Path, archive: Path) -> list:
    """Paths a simulation-setup tool wrote, for ToolResult.files."""
    return [str(dest / k) for k, v in sorted(files.items()) if v is not None] + [str(archive)]


def tool_parameterize_ligand(ligand: str = "", net_charge=None,
                             charge_method: str = "bcc", target: str = "") -> ToolResult:
    """
    GAFF2 parameters and AM1-BCC charges for one ligand (antechamber + parmchk2).

    The net charge is never guessed: the user's number, else ask. The PDB
    chemical dictionary's value is quoted in the question, not used — it is
    the deposited form (ATP is recorded as 0, not -4 at pH 7).
    """
    if not structures():
        return _fail(NO_STRUCTURE)
    tools = sim.tools_available()
    if not (tools["antechamber"] and tools["parmchk2"]):
        return _fail("Cannot parameterise: AmberTools (antechamber, parmchk2) is not "
                     "installed here. Install it with setup_tools.sh, then ask again.",
                     {"unavailable": "ambertools"})
    name, entry = simulation_target(target)
    if not entry:
        return _fail(f"No structure called '{target}' is loaded. Loaded: "
                     + (", ".join(x["pdb_id"] for x in structures()) or "none") + ".")
    code, msg = _pick_ligand(entry, ligand, "parameterise")
    if not code:
        return _fail(msg)
    method = _pick_option(charge_method, {"bcc": 0, "gas": 0}, "bcc")
    if not method:
        return _bad_option("charge method", charge_method, ["bcc", "gas"])

    if not _given(net_charge):
        return ask_clarification(
            f"What net charge should {code} have? antechamber builds whatever charge "
            f"it is given. {_dictionary_note(code)}", [])
    charge = int(_num(net_charge, 0))

    stored = parameterize_ligand(name, entry, code, charge, method)
    if stored.get("error"):
        return _fail(f"antechamber failed on {code} in {name}: {stored['error']}",
                     {"ligand": code})
    out = [f"Parameterised {code} in {name} with GAFF2 and "
           f"{'AM1-BCC' if method == 'bcc' else 'Gasteiger'} charges, net charge "
           f"{charge:+d} (given): {Path(stored['mol2']).name} and "
           f"{Path(stored['frcmod']).name} in {Path(stored['mol2']).parent}/."]
    note = (stored.get("log") or "").splitlines()
    if note and "hydrogens were added" in note[0]:
        shift = [n for n in note[1:4] if n.startswith("To reach the stated net charge")]
        out.append(" ".join([note[0]] + shift)
                   + " Check that this is the protonation state intended.")
    if stored["guessed"]:
        out.append(f"parmchk2 guessed {len(stored['guessed'])} parameter(s) it had no "
                   f"data for — check the geometry after minimisation.")
    if method == "gas":
        out.append("Gasteiger charges are crude: fine to test a setup, not for production.")
    return _ok(" ".join(out),
               {"ligand": code, "structure": name, "net_charge": charge,
                "charge_method": method, "guessed_parameters": len(stored["guessed"])},
               files=[stored["mol2"], stored["frcmod"]],
               hints=["setup_amber to build the system with it"])


def tool_setup_amber(target: str = "", force_field: str = "ff19SB",
                     water_model: str = "", box_shape: str = "octahedron",
                     buffer: float = 12.0, salt_concentration: float = 0.15,
                     ions: str = "Na+/Cl-", temperature: float = 300.0,
                     pressure: float = 1.0, nanoseconds: float = 100.0,
                     engine: str = "pmemd.cuda") -> ToolResult:
    """
    Build an Amber system with tleap and write the run inputs.

    Topology + coordinates (solvated, neutralised, salted), the five mdin
    stages (two minimisations, heating, NPT equilibration, production) and a
    run.sh. Without tleap the run inputs are still written, and the result
    says the topology was not built.
    """
    if not structures():
        return _fail(NO_STRUCTURE)
    name, entry = simulation_target(target)
    if not entry:
        return _fail(f"No structure called '{target}' is loaded. Loaded: "
                     + (", ".join(x["pdb_id"] for x in structures()) or "none") + ".")
    ff = _pick_option(force_field, sim.PROTEIN_FFS, "ff19SB")
    if not ff:
        return _bad_option("force field", force_field, sim.PROTEIN_FFS)
    water = _pick_option(water_model, sim.WATER_MODELS, sim.PROTEIN_FFS[ff]["water"])
    if not water:
        return _bad_option("water model", water_model, sim.WATER_MODELS)
    shape = _pick_option(box_shape, sim.BOX_SHAPES, "octahedron")
    if not shape:
        return _bad_option("box shape", box_shape, sim.BOX_SHAPES)
    pair = _pick_option(ions, sim.ION_PAIRS, "Na+/Cl-") or {
        "nacl": "Na+/Cl-", "na": "Na+/Cl-", "sodium": "Na+/Cl-",
        "kcl": "K+/Cl-", "k": "K+/Cl-", "potassium": "K+/Cl-",
    }.get(re.sub(r"[^a-z]", "", str(ions).lower()))
    if not pair:
        return _bad_option("ion pair", ions, sim.ION_PAIRS)
    engines = ["pmemd.cuda", "pmemd.MPI", "sander"]
    eng = _pick_option(engine, engines, "pmemd.cuda")
    if not eng:
        return _bad_option("engine", engine, engines)
    buffer_a = min(max(_num(buffer, 12.0), 8.0), 25.0)
    salt = min(max(_num(salt_concentration, 0.15), 0.0), 1.0)
    temp = _num(temperature, 300.0)
    press = _num(pressure, 1.0)
    ns = _num(nanoseconds, 100.0)

    params, missing = stored_ligand_params(name, entry)
    tleap = sim.tools_available()["tleap"]
    if tleap and missing:
        return _fail(f"Cannot build the Amber topology for {name} yet: {', '.join(missing)} "
                     f"has no parameters. Call parameterize_ligand for each (it needs the "
                     f"net charge), or prepare the structure without ligands.",
                     {"missing_parameters": list(missing)},
                     hints=[f"parameterize_ligand {c}" for c in missing])

    out = [_unprepared_note(name).strip()] if _unprepared_note(name) else []
    build = None
    if tleap:
        build = build_amber_system(name, entry, params, ff, water, shape,
                                   buffer_a, salt, pair)
        if build["ok"]:
            out.append(
                f"Built the Amber system for {name} ({ff}, {water}, {shape} box, "
                f"{buffer_a:g} Å buffer, {salt:g} M {pair}): {build['atoms']:,} atoms, "
                f"{build['residues']:,} residues, {build['waters']:,} waters, net charge "
                f"{build['charge']:+.3f}, {build['disulfides']} disulfide(s).")
            if abs(build["charge"]) > 0.01:
                out.append(f"WARNING: the system is not neutral ({build['charge']:+.3f}) — "
                           f"check the ion counts and the ligand charges.")
        else:
            errs = "; ".join(build["errors"][:3]) or "see leap.log"
            out.append(f"tleap failed to build a topology for {name}: {errs}")
    files = amber_run_files(entry, build, params, temp, press, ns, eng)
    dest = SIMULATIONS_DIR / f"{name}_amber"
    archive = write_simulation_files(files, dest)
    out.append(f"Run inputs: two minimisations, heating to {temp:g} K, NPT "
               f"equilibration at {press:g} bar, {ns:g} ns production, run.sh for "
               f"{eng}. " + _files_line(files, dest, archive))
    if not tleap:
        out.append("NO topology: AmberTools (tleap) is not installed here, so "
                   "system.prmtop/system.inpcrd were NOT built — install AmberTools "
                   "(setup_tools.sh) and ask again.")
        if missing:
            out.append(f"{', '.join(missing)} will also need parameters (parameterize_ligand).")
    data = {"structure": name, "force_field": ff, "water_model": water, "box": shape,
            "nanoseconds": ns, "temperature": temp, "topology": bool(build and build["ok"])}
    if build and build["ok"]:
        data.update({k: build[k] for k in ("atoms", "residues", "waters", "charge", "disulfides")})
        # run_agent attaches this line verbatim: qwen2.5:14b's paraphrase kept
        # the atom count and dropped the force field, net charge and the
        # disulfide count — the one number that shows S–S bonds were built.
        data["summary"] = next(o for o in out if o.startswith("Built the Amber system"))
    # Run inputs without tleap are a partial success; a tleap that ran and
    # failed is a failure, whatever else was written.
    return ToolResult(" ".join(out), not (build and not build["ok"]), data,
                      _written(files, dest, archive),
                      ["setup_oniom on this system"] if data["topology"] else None)


def tool_setup_gromacs(target: str = "", temperature: float = 300.0,
                       pressure: float = 1.0, nanoseconds: float = 100.0) -> ToolResult:
    """GROMACS .mdp files (em, nvt, npt, md). No topology — the README says how."""
    if not structures():
        return _fail(NO_STRUCTURE)
    name, entry = simulation_target(target)
    if not entry:
        return _fail(f"No structure called '{target}' is loaded. Loaded: "
                     + (", ".join(x["pdb_id"] for x in structures()) or "none") + ".")
    temp = _num(temperature, 300.0)
    press = _num(pressure, 1.0)
    ns = _num(nanoseconds, 100.0)
    files = gromacs_run_files(entry, temp, press, ns)
    dest = SIMULATIONS_DIR / f"{name}_gromacs"
    archive = write_simulation_files(files, dest)
    membrane = mem.membrane_planes(entry["path"])
    return _ok(_unprepared_note(name)
               + f"Wrote GROMACS run parameters for {name}: energy minimisation, NVT "
               f"and NPT equilibration, and {ns:g} ns production at {temp:g} K, "
               f"{press:g} bar ({'semi-isotropic, membrane' if membrane else 'isotropic'} "
               f"pressure coupling). " + _files_line(files, dest, archive)
               + " No GROMACS topology was made: PARORA does not build one. README.txt "
               "has the commands — pdb2gmx on the prepared PDB, or convert an Amber "
               "topology (setup_amber) with ParmEd/acpype.",
               {"structure": name, "nanoseconds": ns, "temperature": temp,
                "pressure": press, "membrane": bool(membrane), "topology": False},
               files=_written(files, dest, archive))


def tool_setup_rosetta_docking(target: str = "", ligand: str = "",
                               ligand_chain: str = "X", nstruct: int = 100) -> ToolResult:
    """RosettaLigand docking job files. Written, not run: Rosetta is not installed."""
    if not structures():
        return _fail(NO_STRUCTURE)
    name, entry = simulation_target(target)
    if not entry:
        return _fail(f"No structure called '{target}' is loaded. Loaded: "
                     + (", ".join(x["pdb_id"] for x in structures()) or "none") + ".")
    code, msg = _pick_ligand(entry, ligand, "dock")
    if not code:
        return _fail(msg)
    chain = (str(ligand_chain or "X").strip() or "X")[:1].upper()
    n = min(max(int(_num(nstruct, 100)), 1), 100000)
    files = rosetta_job_files(name, entry, code, chain, n)
    dest = SIMULATIONS_DIR / f"{name}_rosetta"
    archive = write_simulation_files(files, dest)
    out = [f"Wrote RosettaLigand docking files for {code} in {name} (ligand chain "
           f"{chain}, {n} output structures). " + _files_line(files, dest, archive),
           "Rosetta is not installed here, so the job is written, not run — "
           "README.txt has the commands."]
    if f"{code}.mol2" not in files:
        out.append(f"No {code}.mol2 yet: params_command.txt needs one "
                   f"(parameterize_ligand makes it).")
    prepared = st.session_state.prepared.get(re.sub(r"_PREP$", "", name, flags=re.I))
    if not prepared or prepared.get("profile") != "rosetta":
        out.append("Prepare it with the rosetta profile first: mixed deposited and "
                   "rebuilt hydrogens cause duplicate-atom errors.")
    return _ok(" ".join(out),
               {"structure": name, "ligand": code, "ligand_chain": chain, "nstruct": n,
                "has_mol2": f"{code}.mol2" in files, "run": False},
               files=_written(files, dest, archive))


def _region_keys(path, code: str, radius: float, center_keys=None) -> list:
    if radius <= 0:
        return []
    near = (qm.residues_near(path, center_keys=center_keys, radius=radius) if center_keys
            else qm.residues_near(path, center_codes=[code], radius=radius))
    return [key for key, _, _ in near]


def _copy_keys(path, code: str) -> list:
    """Residue keys of every copy of component `code`, in file order."""
    atoms = mz.read_atoms(path)
    keys = []
    for key, res in zip(qm.residue_keys(atoms), atoms["resname"]):
        if res.upper() == code and key not in keys:
            keys.append(key)
    return keys


def _residue_names(region: dict, limit: int = 25) -> str:
    names = [f"{res}{key[1]}{key[2]}({key[0]})" for key, res in region["residues"]]
    more = f" and {len(names) - limit} more" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more


def tool_setup_qm(target: str = "", center: str = "", chain: str = "",
                  radius: float = 4.0,
                  side_chains_only: bool = True, charge=None, multiplicity: int = 1,
                  method: str = "B3LYP-D3", basis: str = "6-31G(d)",
                  job: str = "opt freq", solvent: str = "none", cores: int = 8,
                  memory_gb: int = 16) -> ToolResult:
    """
    Cut a QM cluster model around a ligand and write Gaussian/ORCA/Psi4 inputs.

    Charge: the user's number, else ask — quoting the residues' formal charge
    sum and the dictionary's value for the ligand, never adding them up
    silently. Several copies
    of the ligand (4HHB's four hemes) and no chain given → ask which one.
    """
    if not structures():
        return _fail(NO_STRUCTURE)
    name, entry = simulation_target(target)
    if not entry:
        return _fail(f"No structure called '{target}' is loaded. Loaded: "
                     + (", ".join(x["pdb_id"] for x in structures()) or "none") + ".")
    code, msg = _pick_ligand(entry, center, "centre the QM region on")
    if not code:
        return _fail(msg)
    meth = _pick_option(method, qm.METHODS, "B3LYP-D3")
    if not meth:
        return _bad_option("method", method, qm.METHODS)
    bas = _pick_option(basis, qm.BASIS_SETS, "6-31G(d)")
    if not bas:
        return _bad_option("basis set", basis, qm.BASIS_SETS)
    jb = _pick_option(job, qm.JOB_TYPES, "opt freq")
    if not jb:
        return _bad_option("job type", job, qm.JOB_TYPES)
    solv = _pick_option(solvent, qm.SOLVENT_MODELS, "none")
    if not solv:
        return _bad_option("solvent model", solvent, qm.SOLVENT_MODELS)
    r = min(max(_num(radius, 4.0), 0.0), 12.0)

    copies = _copy_keys(entry["path"], code)
    ch = str(chain or "").strip().upper()
    if ch == "ALL" or len(copies) <= 1:
        keys = None
    elif ch:
        keys = [k for k in copies if str(k[0]).upper() == ch]
        if not keys:
            return _fail(f"Error: {code} in {name} is in chain(s) "
                         f"{', '.join(sorted({str(k[0]) for k in copies}))}, not {ch}.")
    else:
        opts = [{"label": f"{code} {k[1]}{k[2]} in chain {k[0]}",
                 "meaning": f"use the {code} copy in chain {k[0]}"} for k in copies[:4]]
        opts.append({"label": f"All {len(copies)} copies together",
                     "meaning": f"use all copies of {code} (chain='all')"})
        return ask_clarification(
            f"{name} has {len(copies)} copies of {code}. Which one should the QM "
            f"region be built around?", opts)

    region = build_qm_region(name, entry, [code] if keys is None else [],
                             _region_keys(entry["path"], code, r, keys),
                             bool(side_chains_only), center_keys=keys)
    if not _given(charge):
        return ask_clarification(
            f"What total charge should the QM region around {code} have? Its "
            f"{len(region['residues'])} residues carry {region['charge']:+d} at neutral "
            f"pH; {code}'s own charge is not in that sum. {_dictionary_note(code)}", [])
    total = int(_num(charge, 0))
    charge_note = f"charge {total:+d} (given)"
    mult = max(int(_num(multiplicity, 1)), 1)

    files = qm_input_files(region, total, mult, meth, bas, jb, solv,
                           max(int(_num(cores, 8)), 1), max(int(_num(memory_gb, 16)), 1),
                           True)
    dest = SIMULATIONS_DIR / f"{name}_qm"
    dest.mkdir(parents=True, exist_ok=True)
    files["region.pdb"] = Path(qm.region_pdb(region, dest / "region.pdb"))
    archive = write_simulation_files(files, dest)
    where = f" in chain {ch}" if keys else (f" (all {len(copies)} copies)" if len(copies) > 1 else "")
    out = [f"Built a QM cluster model of {name} around {code}{where} "
           f"({'residues within ' + format(r, 'g') + ' Å' if r else 'the ligand alone'}, "
           f"{'side chains only' if side_chains_only else 'whole residues'}): "
           f"{len(region['atoms'])} atoms incl. {len(region['links'])} link hydrogens, "
           f"formula {region['formula']}, {charge_note}, multiplicity {mult}."]
    if region["residues"]:
        out.append(f"Residues: {_residue_names(region)}.")
    out.append(f"Level: {meth}/{bas}, job '{jb}', solvent {solv}, link atoms frozen. "
               + _files_line(files, dest, archive))
    out += [f"Caution: {w}" for w in region["warnings"]]
    return _ok(" ".join(out),
               {"structure": name, "center": code, "chain": ch or None, "radius": r,
                "atoms": len(region["atoms"]), "link_atoms": len(region["links"]),
                "formula": region["formula"], "charge": total, "multiplicity": mult,
                "method": meth, "basis": bas, "job": jb,
                "residues": [f"{res}{key[1]}{key[2]}({key[0]})"
                             for key, res in region["residues"]]},
               files=_written(files, dest, archive))


def tool_setup_oniom(center: str = "", radius: float = 4.0,
                     side_chains_only: bool = True, mm_radius: float = 15.0,
                     method: str = "B3LYP/6-31G(d):Amber",
                     embedding: str = "electronic", job: str = "opt",
                     charge=None, multiplicity: int = 1, cores: int = 8) -> ToolResult:
    """
    A Gaussian ONIOM (QM/MM) input on the Amber system built by setup_amber.

    MM atom types and charges come from that topology, so it must exist.
    """
    build = st.session_state.amber_build
    if not (build and build.get("ok")):
        return _fail("Cannot set up ONIOM yet: it needs an Amber topology for the MM "
                     "layer's atom types and charges. Build one first with setup_amber "
                     "(needs AmberTools and, for a ligand, parameterize_ligand).",
                     hints=["setup_amber"])
    meth = _pick_option(method, oniom.ONIOM_METHODS, "B3LYP/6-31G(d):Amber")
    if not meth:
        return _bad_option("ONIOM method", method, oniom.ONIOM_METHODS)
    emb = _pick_option(embedding, oniom.EMBEDDING, "electronic")
    if not emb:
        return _bad_option("embedding", embedding, oniom.EMBEDDING)
    jb = _pick_option(job, ["opt", "sp", "opt freq"], "opt")
    if not jb:
        return _bad_option("job type", job, ["opt", "sp", "opt freq"])
    leap_pdb, paired, err = oniom_pairing(build)
    if err:
        return _fail(f"Cannot set up ONIOM: {err}")
    code, msg = _pick_ligand({"path": str(leap_pdb), "pdb_id": build.get("name", "")},
                             center, "put in the QM layer")
    if not code:
        return _fail(msg)
    r = min(max(_num(radius, 4.0), 0.0), 12.0)
    sphere = min(max(_num(mm_radius, 15.0), 0.0), 30.0)
    model = build_oniom_model(build, paired, [code], _region_keys(leap_pdb, code, r),
                              bool(side_chains_only), sphere)
    layered = model["layered"]
    q = int(_num(charge, 0)) if _given(charge) else int(round(layered["charge_high"]))
    mult = max(int(_num(multiplicity, 1)), 1)
    files = oniom_input_files(model, meth, emb, jb, q, mult,
                              max(int(_num(cores, 8)), 1), True)
    dest = SIMULATIONS_DIR / f"{build.get('name', 'system')}_oniom"
    archive = write_simulation_files(files, dest)
    n_copies = len(_copy_keys(leap_pdb, code))
    return _ok(f"Built an ONIOM model on {build.get('name')}'s Amber system around {code}"
            f"{f' (all {n_copies} copies are in the QM layer)' if n_copies > 1 else ''}: "
            f"{layered['high']} QM atoms, {layered['low']} MM atoms "
            f"({'MM layer within ' + format(sphere, 'g') + ' Å' if sphere else 'whole system'}), "
            f"{layered['dropped']} left out, {len(model['boundary'])} link atoms. "
            f"{meth}, {emb} embedding, job '{jb}', QM charge {q:+d} "
            f"({'given' if _given(charge) else 'from the topology partial charges'}), "
            f"multiplicity {mult}, MM layer frozen. Residue numbers are leap's (from 1). "
            + _files_line(files, dest, archive),
               {"structure": build.get("name"), "center": code, "copies": n_copies,
                "qm_atoms": layered["high"], "mm_atoms": layered["low"],
                "left_out": layered["dropped"], "link_atoms": len(model["boundary"]),
                "method": meth, "embedding": emb, "job": jb, "charge": q,
                "charge_given": _given(charge), "multiplicity": mult},
               files=_written(files, dest, archive))


DEFAULT_REPS = [
    {"type": "cartoon", "selection": "protein", "color": "residueindex",
     "transparency": 0.0, "sid": None},
]


def download_pdb(pdb_id: str):
    """
    Fetch a PDB entry into STRUCTURES_DIR, reusing the cached copy if present.

    Returns:
        (Path, None) on success, or (None, error message).
    """
    pdb_id = pdb_id.upper().strip()
    dest = STRUCTURES_DIR / f"{pdb_id}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return dest, None
    try:
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest, None
    except Exception as e:
        return None, f"Error downloading {pdb_id}: {e}"


def download_alphafold(accession: str):
    """
    Fetch the AlphaFold DB predicted model for a UniProt accession into
    STRUCTURES_DIR, reusing the cached copy if present.

    Fallback path for `tool_load_protein` when a protein has no experimental
    PDB deposition at all — most of UniProt does not. AlphaFold DB covers
    essentially every UniProt accession with a per-residue confidence
    (pLDDT) model, so this is the difference between "no structure" and a
    usable one for the majority of proteins someone might name.

    Returns:
        (Path, None) on success, or (None, error message).
    """
    accession = accession.upper().strip()
    dest = STRUCTURES_DIR / f"AF_{accession}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return dest, None
    try:
        r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}",
                         timeout=30)
        r.raise_for_status()
        hits = r.json()
        if not hits:
            return None, f"AlphaFold DB has no predicted model for {accession}."
        pdb_url = hits[0].get("pdbUrl")
        if not pdb_url:
            return None, f"AlphaFold DB entry for {accession} has no PDB file."
        r2 = requests.get(pdb_url, timeout=30)
        r2.raise_for_status()
        dest.write_bytes(r2.content)
        return dest, None
    except Exception as e:
        return None, f"Error downloading the AlphaFold model for {accession}: {e}"


def _recolor_for_comparison() -> str:
    """
    Switch a default rainbow cartoon to per-structure colouring.

    Called when a second structure joins the scene. Two structures both drawn
    rainbow-by-residue are impossible to tell apart, which is the opposite of
    what someone loading a second structure wants; colouring by structure makes
    the comparison readable immediately. Only the untouched default layer is
    rewritten, so a deliberately chosen colour scheme is never overridden.
    """
    changed = False
    for rep in st.session_state.representations:
        if (rep.get("color") == "residueindex" and rep.get("sid") in (None, "", "*")
                and rep.get("selection") in ("protein", "polymer", "all")):
            rep["color"] = BY_STRUCTURE
            changed = True
    return " Cartoons are now coloured by structure so the two can be told apart." \
        if changed else ""


def clear_scene() -> None:
    """
    Unload everything: structures, selections, layers, measurements, fits.

    Only ever reached from an explicit user action — the "Replace" button, the
    "Clear scene" button, or asking for it in words. Loading a structure never
    calls this.
    """
    reset_structures()
    forget_focus_structures()
    st.session_state.selections = {}
    st.session_state.representations = [dict(r, id=uuid.uuid4().hex[:8])
                                        for r in DEFAULT_REPS]
    st.session_state.camera_target = None
    st.session_state.superpose_msg = None
    st.session_state.measurements = []
    st.session_state.interactions = []
    st.session_state.annotations = []


def tool_fetch_structure(pdb_id: str) -> ToolResult:
    """
    Load a PDB structure from RCSB, keeping whatever is already in the scene.

    Loading is additive. Throwing away work the user has already done —
    selections, styling, superpositions, measurements — because they named a
    second structure is never what they meant, so nothing here clears the
    scene; that takes an explicit "Replace" or "Clear scene".

    Args:
        pdb_id: 4-character PDB accession code (case-insensitive).

    Returns:
        Confirmation string, or an error message.
    """
    return tool_add_structure(pdb_id)


def tool_replace_scene(pdb_id: str) -> ToolResult:
    """
    Unload everything and load one structure in its place.

    Args:
        pdb_id: 4-character PDB accession code.

    Returns:
        Confirmation string, or an error message.
    """
    pdb_id = pdb_id.upper().strip()
    dest, err = download_pdb(pdb_id)
    if err:
        return _fail(err, {"pdb_id": pdb_id})
    clear_scene()
    register_structure(pdb_id, dest, source="rcsb")
    return _ok(f"Cleared the scene and loaded {pdb_id}.", {"loaded": pdb_id}, files=[dest])


def tool_clear_scene() -> ToolResult:
    """Unload every structure and reset the scene."""
    had = [s["pdb_id"] for s in structures()]
    clear_scene()
    return _ok("Cleared the scene" + (f" (was: {', '.join(had)})" if had else "")
               + ". Nothing is loaded now.", {"removed": had})


def tool_add_structure(pdb_id: str) -> ToolResult:
    """
    Load a PDB structure *alongside* whatever is already in the scene.

    This is the entry point for comparing structures: both stay loaded with
    their deposited coordinates, and superpose_structures then fits one onto
    the other. Existing selections and representation layers are left alone.

    Args:
        pdb_id: 4-character PDB accession code (case-insensitive).

    Returns:
        Confirmation string, or an error message.
    """
    pdb_id = pdb_id.upper().strip()
    if find_structure(pdb_id):
        return _ok(f"{pdb_id} is already in the scene.", {"loaded": pdb_id, "already": True})
    dest, err = download_pdb(pdb_id)
    if err:
        return _fail(err, {"pdb_id": pdb_id})

    first = not structures()
    note = "" if first else _recolor_for_comparison()
    entry = register_structure(pdb_id, dest, source="rcsb")
    st.session_state.camera_target = None
    if first:
        if not st.session_state.representations:
            st.session_state.representations = [dict(r, id=uuid.uuid4().hex[:8])
                                                for r in DEFAULT_REPS]
        return _ok(f"Loaded {pdb_id} → {dest}", {"loaded": pdb_id}, files=[dest])
    others = [s["pdb_id"] for s in structures() if s["sid"] != entry["sid"]]
    return _ok(f"Added {pdb_id} alongside {', '.join(others)} — the earlier "
               f"structures are still loaded. It is drawn at its deposited "
               f"coordinates; superpose it to compare them.{note}",
               {"loaded": pdb_id, "also_loaded": others}, files=[dest],
               hints=[f"superpose_structures {pdb_id} onto {others[0]}"] if others else None)


def tool_load_local(filepath: str, replace: bool = True) -> ToolResult:
    """
    Load a PDB structure from a local file path into the viewer.

    Uses the file stem (without extension) as the label.

    Args:
        filepath: Absolute or relative path to a .pdb file.
        replace : True to make it the only structure, False to add it alongside
                  the structures already loaded.

    Returns:
        Confirmation string, or an error if the file is not found.
    """
    p = Path(filepath)
    if not p.exists():
        return _fail(f"File not found: {filepath}")
    if replace or not structures():
        reset_structures()
        st.session_state.selections = {}
        st.session_state.representations = [dict(r, id=uuid.uuid4().hex[:8])
                                            for r in DEFAULT_REPS]
        st.session_state.camera_target = None
        st.session_state.superpose_msg = None
        register_structure(p.stem, p, source="local")
        return _ok(f"Loaded local file: {p.name}", {"loaded": p.stem, "replaced": True})
    note = _recolor_for_comparison()
    register_structure(p.stem, p, source="local")
    return _ok(f"Added local file {p.name} to the scene.{note}", {"loaded": p.stem})


def tool_remove_structure(target: str) -> ToolResult:
    """
    Remove one structure from the scene, along with its representation layers.

    Args:
        target: PDB id, label or sid of the structure to remove.

    Returns:
        Confirmation string, or an error if no such structure is loaded.
    """
    s = find_structure(target)
    if not s:
        return _fail(f"No structure called '{target}' is loaded. Loaded: "
                     + (", ".join(x["pdb_id"] for x in structures()) or "none") + ".")
    label = s["pdb_id"]
    drop_structure(s["sid"])
    left = [x["pdb_id"] for x in structures()]
    return _ok(f"Removed {label}. Still loaded: {', '.join(left) or 'none'}.",
               {"removed": label, "loaded": left})


@st.cache_data(show_spinner=False)
def _cached_summary(path: str, mtime: float, schema: int):
    """
    Parse and cache a structure's composition.

    mtime busts the cache when the file changes; schema busts it when the shape
    of the report changes. The second one is not optional: st.cache_data keys on
    this function's own code, not on srep.summarize's, so without it an edit to
    the report would leave the running app feeding stale dicts to the new
    formatters.
    """
    return srep.summarize(path)


def structure_summary(target: str = ""):
    """
    Composition report for a loaded structure, or None if there is no such one.

    Args:
        target: PDB id, label or sid. Empty means the active structure.
    """
    entry = find_structure(target) if target else active_structure()
    if not entry or not Path(entry["path"]).exists():
        return None
    return _cached_summary(entry["path"], Path(entry["path"]).stat().st_mtime,
                           srep.SCHEMA_VERSION)


def tool_describe_structure(target: str = "", detail: str = "brief") -> ToolResult:
    """
    Describe what a structure contains, in plain language.

    "brief" is four lines: what the structure is, how large, and which residues
    are non-standard. "full" adds the per-chain breakdown, missing residues,
    chain breaks and every component's full chemical name.

    Standard means one of the 20 amino acids; everything else that occupies a
    residue slot — modified amino acids, nucleotides, ligands, cofactors, ions,
    buffer components — is reported as non-standard, with water counted
    separately.

    Args:
        target: PDB id of the structure to describe. Empty for the active one.
        detail: "brief" (default) or "full".

    Returns:
        A plain-text report, or an error message.
    """
    if not structures():
        return _fail(NO_STRUCTURE)
    summary = structure_summary(target)
    if summary is None:
        return _fail(f"No structure called '{target}' is loaded. Loaded: "
                     + ", ".join(s["pdb_id"] for s in structures()))
    text = (srep.as_text(summary)
            if (detail or "brief").lower() in ("full", "detailed", "long", "all")
            else srep.as_brief(summary))
    entry = find_structure(target) if target else active_structure()
    return _ok(text + _pocket_summary(entry),
               {"pdb_id": entry["pdb_id"] if entry else target,
                "totals": summary["totals"], "models": summary["models"],
                "waters": summary["waters"],
                "components": {c["code"]: c["count"] for c in summary["components"]}})


def _pocket_summary(entry) -> str:
    """
    One line on pockets already found for this file by find_pockets, so a
    later composition question does not read as "no ligand = no binding
    site". Only reuses a finished run — never starts fpocket itself.
    """
    found = st.session_state.get("pockets")
    if not entry or not found or found["path"] != entry["path"] or not found["by_rank"]:
        return ""
    ps = list(found["by_rank"].values())
    drug = [p for p in ps if p.get("drug_score", 0) >= pkt.DRUGGABLE]
    tail = (f"; predicted druggable: " + ", ".join(
        f"pocket {p['rank']} ({p['drug_score']:.2f}"
        + (", holds " + p["occupants"][0]["label"] if p["occupants"] else ", empty") + ")"
        for p in drug) if drug else "; none predicted druggable")
    return f"\nPockets (fpocket, found earlier): {len(ps)} candidate pocket(s){tail}."


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_classification(pdb_id: str, schema: int):
    """
    Look up and cache a PDB entry's CATH/SCOP classification (PDBe SIFTS).

    ttl caps how long a transient SIFTS outage would otherwise be remembered
    as "no classification" — same reasoning as _cached_protein_profile's ttl.
    """
    return topo.lookup_classification(pdb_id)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_observed_segments(pdb_id: str, schema: int):
    """Cache a PDB entry's observed-residue segments (PDBe polymer_coverage)."""
    return topo.observed_segments(pdb_id)


@st.cache_data(show_spinner=False)
def _cached_dssp(path: str, mtime: float, schema: int):
    """Run and cache DSSP on a structure; mtime and schema bust the cache."""
    return topo.run_dssp(path)


@st.cache_data(show_spinner=False)
def _cached_foldseek(path: str, mtime: float, chains: tuple, max_hits: int,
                     exclude: str, db: str, schema: int):
    """
    Run and cache a Foldseek search; mtime, db and schema bust the cache.
    db is a local database prefix, or "online" for search.foldseek.com.
    """
    if db == "online":
        return fsk.search_online(path, list(chains), max_hits=max_hits,
                                 exclude_pdb_id=exclude)
    return fsk.search(path, list(chains), max_hits=max_hits, exclude_pdb_id=exclude,
                      db=db)


def _foldseek_choice(entry: dict, info: dict = None) -> str:
    """
    The question to put to the user when there is no local Foldseek
    database and they have not agreed to an online search: which of the two
    they want. Deliberately a question — uploading their structure and
    downloading gigabytes are both the user's call, never the model's.
    """
    state, detail = fsk.download_status()
    if state == "running":
        return ("Structural similarity search: the local Foldseek database is still "
                "downloading — try again in a few minutes. Or, if the user prefers, "
                "search online now (uploads this structure's coordinates to "
                "search.foldseek.com).")
    download = (f"download the Foldseek PDB database locally — one-time, "
                f"~{fsk.DOWNLOAD_GB} GB download, ~{fsk.DISK_GB} GB on disk, runs in "
                "the background; searches then stay on this machine")
    if not fsk.find_foldseek():
        download += (" (needs the foldseek binary first: conda create -n foldseek "
                     "-c conda-forge -c bioconda foldseek)")
    failed = (f" A previous download failed: {detail.strip()[-150:]}"
              if state == "failed" else "")
    st.session_state.foldseek_pending = entry["pdb_id"]
    if info is not None:
        info["needs_user_choice"] = True
    # The model tends to shorten this question to "online or download?" and
    # drop the upload warning and the size, so run_agent appends this exact
    # wording to its reply instead of trusting the paraphrase.
    st.session_state.foldseek_question = (
        f"**Structural similarity search needs your choice** — no local Foldseek "
        f"database is installed.\n"
        f"1. **Search online** — uploads {entry['pdb_id']}'s coordinates to "
        f"search.foldseek.com, a public third-party server. Fine for published "
        f"structures, not for confidential ones.\n"
        f"2. **{download[0].upper()}{download[1:]}.**{failed}\n\n"
        f"Reply *search online* or *download the database*.")
    return ("NEEDS USER CHOICE — structural similarity search has no local Foldseek "
            "database. Ask the user which they want, and do not pick for them: "
            f"(1) search online — uploads {entry['pdb_id']}'s coordinates to "
            "search.foldseek.com, a public third-party server; fine for published "
            "structures, not for confidential ones; or "
            f"(2) {download}.{failed}")


def tool_download_foldseek_database() -> ToolResult:
    """
    Start downloading the local Foldseek PDB database, in the background.

    Only reachable when the user asked for it in their own words (a hard
    gate in run_agent, not just the system prompt) — this is gigabytes of
    someone else's disk and bandwidth.
    """
    state, _ = fsk.download_status()
    if state == "ready":
        return _ok("The local Foldseek database is already installed — searches run locally.",
                   {"status": "ready"})
    ok, msg = fsk.start_download()
    if ok:
        st.session_state.pop("foldseek_pending", None)
    return ToolResult(msg, ok, {"status": "downloading" if ok else "failed"})


def _fold_key(c: dict):
    """(label, name) of a CATH topology / SCOP fold — what 'same fold' means."""
    if c["source"] == "CATH":
        return (f"CATH topology {'.'.join(c['cath_id'].split('.')[:3])}", c["topology"])
    return ("SCOP fold", c["fold"])


# Hits kept per chain for the fold consensus, however many are displayed
# (Foldseek itself stops near 1000): the top hits of a well-studied protein
# are mostly its own other depositions (6VXX spike: 552 of 586 hits are
# >= 80% identical spike entries), which would otherwise crowd out
# classified homologs further down.
FOLD_POOL = 1000
# Most hits of each kind (distant / near-identical) whose classification is
# looked up for the vote — bounds the SIFTS round-trips per search.
FOLD_VOTERS = 20
# At or above this sequence identity a hit is (almost certainly) the same
# protein — one more deposition, not independent evidence of the fold.
NEAR_IDENTICAL = 0.95


def _domain_text(c: dict) -> str:
    """One CATH/SCOP domain, as reported to the model."""
    if c["source"] == "CATH":
        return (f"CATH {c['cath_id']} {c['name']} ({c['class']} / "
                f"{c['architecture']} / {c['topology']} / {c['homology']})")
    return f"SCOP {c['name']} (fold: {c['fold']})"


def _range_text(c: dict) -> str:
    """A domain's author-numbered residue ranges, e.g. '82-145' or '5-40, 90-120'."""
    return ", ".join(f"{s}-{e}" for s, e in c.get("ranges", []) if s is not None)


def _hit_domains(hits: list) -> dict:
    """
    For every PDB hit, the classified domains its alignment actually covers.

    SIFTS lookups for all distinct hit entries run in parallel (a pool of 20
    hits sequentially is ~20 s of HTTP round-trips). Observed-residue
    segments are fetched only for multi-domain chains — the only case where
    the exact numbering decides the answer.

    Returns:
        {id(hit): (all_domains_of_chain, covered_domains)}.
    """
    pdb_hits = [h for h in hits if h["kind"] == "pdb"]
    ids = sorted({h["pdb_id"] for h in pdb_hits})
    if not ids:
        return {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        cls = dict(zip(ids, pool.map(
            lambda i: _cached_classification(i, topo.SCHEMA_VERSION), ids)))
        multi = sorted({h["pdb_id"] for h in pdb_hits
                        if len(cls[h["pdb_id"]].get(h["chain"], [])) > 1})
        segs = dict(zip(multi, pool.map(
            lambda i: _cached_observed_segments(i, topo.SCHEMA_VERSION), multi)))
    out = {}
    for h in pdb_hits:
        doms = cls[h["pdb_id"]].get(h["chain"], [])
        seg = segs.get(h["pdb_id"], {}).get(h["chain"])
        out[id(h)] = (doms, topo.pick_domains(doms, h["tstart"], h["tend"], seg))
    return out


def _spread(hits: list, n: int) -> list:
    """
    n hits spread evenly over the identity range, lowest identity first.

    Taking the first n by E-value picks n variants of the query's own
    protein at 90-94% identity; spreading picks distinct homolog families,
    which are the independent evidence a fold vote needs.
    """
    ranked = sorted(hits, key=lambda h: h["fident"])
    if len(ranked) <= n:
        return ranked
    step = len(ranked) / n
    return [ranked[int(i * step)] for i in range(n)]


def _fold_consensus(hits: list, doms_of: dict) -> str:
    """
    Which known fold(s) the classified hits in the pool point at.

    Every hit votes once for each fold its aligned domain(s) belong to.
    Near-identical hits (>= NEAR_IDENTICAL identity: the same protein again)
    are left out of the vote whenever more distant classified hits exist, so
    twenty copies of one protein cannot outvote real homologs — and are
    used, but labelled, when they are all there is.
    """
    classified = [h for h in hits if doms_of.get(id(h), ((), ()))[1]]
    if not classified:
        return (f"    None of the hits checked ({len(hits)} found, classification "
                f"looked up for up to {FOLD_VOTERS} spread across the identity "
                "range) has a CATH/SCOP classification covering the aligned "
                "region, so no known fold can be named from them.")
    distant = [h for h in classified if h["fident"] < NEAR_IDENTICAL]
    voters = distant or classified
    votes = {}
    for h in voters:
        for key in {_fold_key(d) for d in doms_of[id(h)][1]}:
            votes.setdefault(key, set()).add(h["pdb_id"].upper())
    ranked = sorted(votes.items(), key=lambda kv: -len(kv[1]))
    (kind, name), top = ranked[0]
    note = (f"{len(hits)} hits found; classification looked up for up to "
            f"{FOLD_VOTERS} spread across the identity range")
    skipped = len(classified) - len(distant)
    if distant and skipped:
        note += (f"; {skipped} near-identical hit(s) (≥{NEAR_IDENTICAL:.0%} "
                 "sequence identity, same protein) left out of the vote")
    elif not distant:
        note += (f"; every classified hit is ≥{NEAR_IDENTICAL:.0%} identical — "
                 "other depositions of the same protein, not independent homologs")
    line = (f"    Fold consensus: {len(top)} of {len(voters)} classified hits "
            f"share {kind} ({name}) [{note}].")
    others = [f"{k} ({n}) — {len(v)} hit(s)" for (k, n), v in ranked[1:3] if len(v) >= 2]
    if others:
        line += " Also aligned to: " + "; ".join(others) + "."
    return line


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_uniprot_by_chain(pdb_id: str, schema: int):
    """Cache a PDB entry's chain -> UniProt accession map (PDBe SIFTS)."""
    return topo.uniprot_by_chain(pdb_id)


# Neighbours whose UniProt entry is looked up for a function to transfer.
FUNCTION_TRANSFER_CANDIDATES = 12


def _structure_function_lines(accession: str) -> list:
    """
    P13 structure-based transfer: the curated functions of the proteins a
    loaded structure of `accession` resembles (Foldseek), for an entry with
    no FUNCTION of its own.

    Local databases only — this runs without being asked for, so it must not
    upload the user's structure; online search stays behind
    find_structural_neighbors' consent question. Every transferred function
    sits under an INFERRED ... LOW CONFIDENCE heading with the evidence
    (TM-score, identity, coverage) next to it.
    """
    acc = (accession or "").strip().upper()
    entry, chains = None, []
    for s in structures():
        if _foldseek_exclude(s).upper() == acc:
            entry, chains = s, list(_atoms_of(s)["chains_present"])
            break
        mine = [m["chain"] for m in srep.chain_molecules(s["path"])
                if acc in [u.upper() for u in m.get("uniprot") or []]]
        if mine:
            entry, chains = s, mine
            break
    if not entry:
        return ["Structure-based check: not run — no structure of this protein is loaded "
                "(load it to have its structural neighbours' known functions compared)."]
    dbs = [d for d in (fsk.find_database() if fsk.availability()[0] else None,
                       fsk.find_afdb_database() if fsk.find_foldseek() else None) if d]
    if not dbs:
        return ["Structure-based check: not run — no local Foldseek database. "
                "find_structural_neighbors can search online if the user agrees to upload "
                "the structure."]

    hits = []
    for db in dbs:
        ok, msg, result = _cached_foldseek(
            entry["path"], Path(entry["path"]).stat().st_mtime, tuple(chains),
            FOLD_POOL, _foldseek_exclude(entry), db, fsk.SCHEMA_VERSION)
        if ok:
            hits += [h for g in result.values() for h in g["hits"]]
    hits.sort(key=lambda h: h["evalue"])
    pdb_ids = sorted({h["pdb_id"] for h in hits if h["kind"] == "pdb"}
                     )[:FUNCTION_TRANSFER_CANDIDATES * 3]
    with ThreadPoolExecutor(max_workers=8) as pool:
        maps = dict(zip(pdb_ids, pool.map(
            lambda i: _cached_uniprot_by_chain(i, topo.SCHEMA_VERSION), pdb_ids)))
    best = {}          # neighbour accession -> its best (lowest E) hit
    for h in hits:
        n_acc = (h.get("accession") if h["kind"] == "afdb"
                 else maps.get(h["pdb_id"], {}).get(h["chain"]))
        n_acc = (n_acc or "").split("-")[0].upper()
        if n_acc and n_acc != acc and n_acc not in best:
            best[n_acc] = h
        if len(best) >= FUNCTION_TRANSFER_CANDIDATES:
            break
    if not best:
        return ["Structure-based check: Foldseek found no confident structural neighbour "
                "(E-value ≤ 1e-3) that maps to a UniProt entry — nothing to transfer."]
    with ThreadPoolExecutor(max_workers=8) as pool:
        known = list(pool.map(lambda a: pacc.known_function(a)[0], list(best)))
    rows = [(k, best[k["accession"]]) for k in known
            if k and k["function"] and not k["automatic"]][:3]
    if not rows:
        return [f"Structure-based check: {len(best)} structural neighbours checked; none has "
                "a curated UniProt function to transfer."]
    lines = ["INFERRED FUNCTION FROM STRUCTURAL NEIGHBOURS (LOW CONFIDENCE — Foldseek "
             "similarity of this structure to proteins with curated functions; similar "
             "shape does not mean same function):"]
    for k, h in rows:
        tm = f"TM-score {min(h['tmscore'], 1.0):.2f}, " if h["tmscore"] is not None else ""
        cov = (h["qend"] - h["qstart"] + 1) / max(h["qlen"], 1)
        src = (f"PDB {h['pdb_id'].upper()} chain {h['chain']}" if h["kind"] == "pdb"
               else "AlphaFold model")
        text = k["function"] if len(k["function"]) <= 300 else k["function"][:297] + "…"
        lines.append(f"  {k['name']} ({k['accession']}, {k['organism']}; {src}) — {tm}"
                     f"E={h['evalue']:.1e}, {h['fident']:.0%} sequence identity, aligned "
                     f"over {cov:.0%} of the query: {text}")
    lines.append("Caveat: a shared fold often goes with a shared broad role but can differ "
                 "in substrate, partner or activity — and sequence identity below ~30% makes "
                 "a specific transfer unreliable. Report these as candidates, not as this "
                 "protein's function, and keep the LOW CONFIDENCE label.")
    return lines


def _foldseek_exclude(entry: dict) -> str:
    """The id of the query's own entry, so it is not reported as its own neighbour."""
    # An AlphaFold model, however it was loaded: "AF-O95905" (load_protein),
    # "AF_O95905" (its cached file name, re-fetched or loaded locally).
    m = re.match(r"AF[-_]([A-Za-z0-9]+)", entry["pdb_id"])
    if m:
        return m.group(1)
    if entry.get("source") == "rcsb":
        return entry["pdb_id"]
    return ""


def _afdb_neighbor_lines(entry: dict, chains: list, max_hits: int,
                         info: dict = None) -> list:
    """
    Foldseek hits against the local AlphaFold DB (Swiss-Prot) database.

    Reaches reviewed proteins with no experimental structure — the case
    where the PDB search finds nothing (AF-O95905: no PDB match, but its
    mouse, plant and fly orthologs here). Hits are predicted models with no
    CATH/SCOP classification, so they name proteins, never a fold.
    Local only: nothing is uploaded, so no consent question is needed.
    Returns [] when the database is not installed. `info`, when given, gets
    "afdb_neighbours" (the hits' accessions) for run_agent's provenance note.
    """
    info = {} if info is None else info
    db = fsk.find_afdb_database()
    if not db or not fsk.find_foldseek():
        return []
    ok, msg, result = _cached_foldseek(
        entry["path"], Path(entry["path"]).stat().st_mtime, tuple(chains),
        max_hits, _foldseek_exclude(entry), db, fsk.SCHEMA_VERSION)
    if not ok:
        return [f"  AlphaFold DB search failed: {msg}"]
    info["afdb_neighbours"] = [h.get("accession") or h.get("target")
                               for g in result.values() for h in g["hits"]]
    lines = [f"  AlphaFold DB neighbours (Foldseek vs the local "
             f"'{Path(db).name}' database — AlphaFold predicted models of reviewed "
             "UniProt entries, E-value ≤ 1e-3; no CATH/SCOP classification exists "
             "for these, so they name similar proteins, not a fold):"]
    for group in result.values():
        label = ", ".join(group["chains"])
        if not group["hits"]:
            lines.append(f"    chain {label}: no confident match here either.")
            continue
        lines.append(f"    chain {label}:")
        for i, h in enumerate(group["hits"], 1):
            tm = (f"TM-score {min(h['tmscore'], 1.0):.2f}, "
                  if h["tmscore"] is not None else "")
            who = (f"UniProt {h['accession']}" if h["kind"] == "afdb" else h["target"])
            lines.append(
                f"      {i}. {who} — {h['description']} — {tm}homology probability "
                f"{h['prob']:.2f}, E={h['evalue']:.1e}, {h['fident']:.0%} sequence "
                f"identity, query positions {h['qstart']}-{h['qend']} of {h['qlen']}")
    return lines


def _structural_neighbor_lines(entry: dict, chains: list, max_hits: int,
                               where: str = "auto", database: str = "pdb",
                               info: dict = None) -> list:
    """
    Foldseek hits for some chains of a structure, each annotated with the
    hit's own CATH/SCOP classification, plus a per-chain fold consensus.

    Shared by tool_find_structural_neighbors and tool_describe_fold's
    no-classification fallback. Every fold named here is a curated
    classification of a real aligned PDB chain — never inferred.

    where: "local", "online" or "auto". Auto uses the local database when
    installed, the online server only if the user already agreed to it this
    session, and otherwise returns the question of which one they want.

    database: "pdb" (default) — chains with no confident PDB match then also
    get the local AlphaFold DB search, when installed; "alphafold" — only
    the local AlphaFold DB search.

    info: optional dict filled with what the lines say, for ToolResult.data —
    "needs_user_choice", "error", "hits" (PDB id + chain), "no_pdb_match"
    (chains), "afdb_neighbours".
    """
    info = {} if info is None else info
    if database == "alphafold":
        lines = _afdb_neighbor_lines(entry, chains, max_hits, info)
        if not lines:
            info["error"] = "AlphaFold DB Foldseek database not installed"
        return lines or [
            "The AlphaFold DB (Swiss-Prot) Foldseek database is not installed "
            "(~1.6 GB download, ~2.4 GB on disk). Install: foldseek databases "
            f"Alphafold/Swiss-Prot {fsk.DEFAULT_AFDB} /tmp/fs — or set FOLDSEEK_AFDB."]

    local_ok, local_msg = fsk.availability()
    online_ok = entry["path"] in st.session_state.get("foldseek_online_ok", set())
    if where == "local" and not local_ok:
        if fsk.find_database():
            info["error"] = local_msg
            return [local_msg]
        return [_foldseek_choice(entry, info)]
    if where == "online" and not online_ok:
        return [_foldseek_choice(entry, info)]
    if where == "auto":
        if local_ok:
            where = "local"
        elif online_ok:
            where = "online"
        else:
            return [_foldseek_choice(entry, info)]

    exclude = _foldseek_exclude(entry)
    db = "online" if where == "online" else fsk.find_database()
    ok, msg, result = _cached_foldseek(
        entry["path"], Path(entry["path"]).stat().st_mtime, tuple(chains),
        max(max_hits, FOLD_POOL), exclude, db, fsk.SCHEMA_VERSION)
    if not ok:
        info["error"] = msg
        return [msg]
    st.session_state.pop("foldseek_pending", None)
    info["where"] = where

    against = (f"the {fsk.ONLINE_DB} database on search.foldseek.com (online)"
               if where == "online" else f"the local '{Path(db).name}' database")
    lines = [f"Structural neighbours (Foldseek, this structure's own coordinates vs "
             f"{against}, E-value ≤ 1e-3):"]
    no_pdb_match = []
    for rep, group in result.items():
        label = ", ".join(group["chains"])
        pool_hits = group["hits"]
        if not pool_hits:
            lines.append(f"  chain {label}: no confident structural match to any "
                         "entry in the database — no known fold can be named from "
                         "structural similarity.")
            no_pdb_match.append(rep)
            continue
        lines.append(f"  chain {label}:")
        info.setdefault("hits", []).extend(
            f"{h['pdb_id'].upper()}:{h['chain']}" if h["kind"] == "pdb"
            else (h.get("accession") or h.get("name") or "") for h in pool_hits[:max_hits])
        # Classify what is shown plus the distant hits (the independent
        # evidence); near-identical ones only when no distant hit is classified.
        shown = pool_hits[:max_hits]
        distant = _spread([h for h in pool_hits if h["fident"] < NEAR_IDENTICAL],
                          FOLD_VOTERS)
        doms_of = _hit_domains(shown + [h for h in distant if h not in shown])
        if not any(doms_of.get(id(h), ((), ()))[1] for h in distant):
            near = [h for h in pool_hits if h["fident"] >= NEAR_IDENTICAL
                    and id(h) not in doms_of][:FOLD_VOTERS]
            doms_of.update(_hit_domains(near))
        for i, h in enumerate(shown, 1):
            tm = (f"TM-score {min(h['tmscore'], 1.0):.2f}, "
                  if h["tmscore"] is not None else "")
            stats = (f"{tm}homology probability "
                     f"{h['prob']:.2f}, E={h['evalue']:.1e}, {h['fident']:.0%} sequence "
                     f"identity, query positions {h['qstart']}-{h['qend']} of {h['qlen']}")
            if h["kind"] == "pdb":
                all_doms, covered = doms_of.get(id(h), ([], []))
                if not all_doms:
                    cls = "no CATH/SCOP classification on file"
                elif len(all_doms) == 1:
                    cls = _domain_text(all_doms[0])
                elif covered:
                    cls = "; ".join(
                        f"aligned to its domain {all_doms.index(d) + 1} of "
                        f"{len(all_doms)} (residues {_range_text(d)}): {_domain_text(d)}"
                        for d in covered)
                else:
                    cls = (f"chain has {len(all_doms)} classified domains, none "
                           "covering the aligned region")
                lines.append(f"    {i}. PDB {h['pdb_id'].upper()} chain {h['chain']} — "
                             f"{h['description']} — {stats} — {cls}")
            elif h["kind"] == "afdb":
                lines.append(f"    {i}. AlphaFold DB model of UniProt {h['accession']} — "
                             f"{h['description']} — {stats} — predicted model, no "
                             "CATH/SCOP lookup")
            else:
                lines.append(f"    {i}. {h['name']} — {h['description']} — {stats}")
        lines.append(_fold_consensus(pool_hits, doms_of))
    if no_pdb_match:
        info["no_pdb_match"] = list(no_pdb_match)
        lines.extend(_afdb_neighbor_lines(entry, no_pdb_match, max_hits, info))
    return lines


def tool_describe_fold(chain: str = "") -> ToolResult:
    """
    Report a structure's fold/topology — real, sourced data, never a guess.

    Two independent sources, reported separately: an existing CATH/SCOP
    classification (a database lookup via PDBe SIFTS, used when the entry
    has one — more accurate than anything computed) and a DSSP-computed
    secondary-structure topology string from this structure's own
    coordinates (attempted always, and the fallback when no classification
    exists — e.g. an AlphaFold model, which has no PDB accession for SIFTS
    to key on). This is fold/topology, not composition — use
    describe_structure for ligands, chains and residues — and never a
    model-guessed fold name assembled from the structure's name.

    Args:
        chain: Restrict to one chain, e.g. "A". Empty covers every chain,
               grouping identical folds together (e.g. hemoglobin's A/C and
               B/D) instead of repeating them.

    Returns:
        A plain-text report, or an honest statement that neither source has
        anything for this structure — never an invented fold name.
    """
    entry = active_structure()
    if not entry:
        return _fail(NO_STRUCTURE)

    atoms = _atoms_of(entry)
    chains = atoms["chains_present"]
    wanted_chain = ""
    if (chain or "").strip():
        wanted_chain = chain.strip().upper().replace("CHAIN", "").strip()
        if wanted_chain not in chains:
            return _fail(f"Chain {wanted_chain} is not in {entry['pdb_id']}. "
                         f"Chains present: {', '.join(chains)}", {"chains": list(chains)})
        chains = [wanted_chain]

    lines = [f"Fold/topology for {entry['pdb_id']}"
            + (f", chain {wanted_chain}" if wanted_chain else "")]

    # 1. Existing classification, only meaningful for a real PDB accession.
    classification = {}
    if entry.get("source") == "rcsb":
        classification = _cached_classification(entry["pdb_id"], topo.SCHEMA_VERSION)
        groups = {}
        for ch in chains:
            doms = classification.get(ch)
            if not doms:
                continue
            key = tuple((d["source"], d.get("cath_id") or d.get("sunid"),
                         tuple(d["ranges"])) for d in doms)
            groups.setdefault(key, {"doms": doms, "chains": []})["chains"].append(ch)

        def _dom_line(c):
            if c["source"] == "CATH":
                return (f"CATH {c['cath_id']} — {c['name']} ({c['class']} / "
                        f"{c['architecture']} / {c['topology']} / {c['homology']})")
            return (f"SCOP — {c['name']} ({c['class']} / fold: {c['fold']} / "
                    f"superfamily: {c['superfamily']})")

        if groups:
            lines.append("Existing classification (CATH/SCOP):")
            for g in groups.values():
                doms, chain_label = g["doms"], ", ".join(g["chains"])
                if len(doms) == 1:
                    lines.append(f"  chain {chain_label}: {_dom_line(doms[0])}")
                    continue
                lines.append(f"  chain {chain_label}: {len(doms)} domains, "
                             "N- to C-terminal:")
                for n, d in enumerate(doms, 1):
                    lines.append(f"    domain {n}, residues {_range_text(d)}: {_dom_line(d)}")
        else:
            lines.append("No CATH/SCOP classification on file for this entry.")
    else:
        lines.append(
            f"No CATH/SCOP lookup attempted — {entry['pdb_id']} is "
            + ("an AlphaFold predicted model" if entry.get("source") == "alphafold"
               else "a local file")
            + ", not a PDB accession SIFTS can classify.")

    # 2. DSSP-computed topology, attempted regardless — corroborates a
    #    database hit, or stands in when there isn't one.
    if topo.dssp_available():
        ok, msg, per_residue = _cached_dssp(
            entry["path"], Path(entry["path"]).stat().st_mtime, topo.SCHEMA_VERSION)
        if ok:
            seen = {}
            for ch in chains:
                s = topo.topology_string(per_residue, ch)
                if s:
                    seen.setdefault(s, []).append(ch)
            if seen:
                lines.append("Computed topology (DSSP, this structure's own coordinates):")
                for s, chs in seen.items():
                    lines.append(f"  chain {', '.join(chs)}: {s}")
            else:
                lines.append("DSSP found no helix or strand content for the requested chain(s).")
        else:
            lines.append(msg)
    else:
        lines.append(
            "DSSP not installed — computed topology unavailable. Set DSSP_BIN "
            "or install: conda create -n dssp -c conda-forge dssp")

    # 3. Chains with no classification of their own: which classified
    #    structures do they resemble? (Foldseek, optional.)
    unclassified = [ch for ch in chains if ch not in classification]
    info = {}
    if unclassified:
        lines.extend(_structural_neighbor_lines(entry, unclassified, max_hits=3, info=info))

    # One verdict line up front. With the classification, DSSP and the
    # "no classification" wording all in one report, qwen2.5:7b wrote "does
    # not have an existing fold classification... categorized as Globin-like"
    # for 4HHB chain A, which has one.
    classified = [ch for ch in chains if ch in classification]
    names = []
    for ch in classified:
        for d in classification[ch]:
            n = d.get("name") or d.get("fold") or ""
            if n and n not in names:
                names.append(n)
    if classified and not unclassified:
        verdict = (f"every requested chain ({', '.join(classified)}) HAS a CATH/SCOP "
                   f"classification: {'; '.join(names)}.")
    elif classified:
        verdict = (f"chain(s) {', '.join(classified)} HAVE a CATH/SCOP classification "
                   f"({'; '.join(names)}); chain(s) {', '.join(unclassified)} have "
                   "none of their own — see the Foldseek section for those.")
    else:
        verdict = ("no requested chain has a CATH/SCOP classification — any fold named "
                   "below comes from DSSP topology or structural neighbours, not a database "
                   "entry for this structure.")
    lines.insert(1, "SUMMARY: " + verdict)

    return _ok("\n".join(lines),
               {"pdb_id": entry["pdb_id"], "classified": classified,
                "unclassified": unclassified, "folds": names,
                "classification": {ch: [d.get("cath_id") or d.get("sunid")
                                        for d in classification[ch]] for ch in classified},
                **{k: v for k, v in info.items() if k != "error"}})


def tool_find_structural_neighbors(chain: str = "", max_hits: int = 5,
                                   where: str = "auto", database: str = "pdb") -> ToolResult:
    """
    Find known structures that the loaded structure resembles in 3D.

    Runs Foldseek on this structure's own coordinates — against a local
    reference database, or search.foldseek.com if the user agreed to an
    online search — and reports each hit's own CATH/SCOP classification
    plus a per-chain fold consensus: evidence for "which known fold is
    this like", even for an AlphaFold model or a local file with no
    classification of its own. Identical chains are searched once. With no
    local database and no online consent yet, returns the question of which
    the user wants instead of choosing.

    Args:
        chain   : Restrict to one chain, e.g. "A". Empty searches every
                  distinct protein chain.
        max_hits: Neighbours reported per chain (1-20).
        where   : "auto" (default), "local" or "online".
        database: "pdb" (default; chains with no PDB match fall back to the
                  local AlphaFold DB when installed) or "alphafold" (local
                  AlphaFold DB Swiss-Prot only).

    Returns:
        A plain-text report, or why the search could not run.
    """
    entry = active_structure()
    if not entry:
        return _fail(NO_STRUCTURE)
    chains = []
    if (chain or "").strip():
        wanted = chain.strip().upper().replace("CHAIN", "").strip()
        present = _atoms_of(entry)["chains_present"]
        if wanted not in present:
            return _fail(f"Chain {wanted} is not in {entry['pdb_id']}. "
                         f"Chains present: {', '.join(present)}", {"chains": list(present)})
        chains = [wanted]
    try:
        max_hits = max(1, min(int(max_hits), 20))
    except (TypeError, ValueError):
        max_hits = 5
    lines = [f"Structural similarity search for {entry['pdb_id']}"
             + (f", chain {chains[0]}" if chains else "")]
    where = (where or "auto").strip().lower()
    if where not in ("auto", "local", "online"):
        where = "auto"
    database = (database or "pdb").strip().lower()
    database = "alphafold" if database in ("alphafold", "afdb", "af", "swissprot",
                                          "swiss-prot") else "pdb"
    info = {}
    lines.extend(_structural_neighbor_lines(entry, chains, max_hits, where, database, info))
    return ToolResult("\n".join(lines), "error" not in info,
                      {"pdb_id": entry["pdb_id"], "database": database, **info})


@st.cache_data(show_spinner=False)
def _cached_pockets(path: str, mtime: float, schema: int):
    """Run and cache fpocket on a structure; mtime and schema bust the cache."""
    return pkt.find_pockets(path)


def _pocket_residues(p: dict) -> str:
    """'A: F98 H103 · B: Y35' — a pocket's lining residues, grouped by chain."""
    by_chain = {}
    for chain, resseq, icode, resname in p["residues"]:
        one = squ.AA3_TO_1.get(resname, resname)
        by_chain.setdefault(chain, []).append(f"{one}{resseq}{icode}")
    return " · ".join(f"{c}: {' '.join(r)}" for c, r in by_chain.items())


def _pocket_chains(p: dict) -> str:
    """'chain B' / 'chains A, D' — who lines a pocket, most residues first."""
    counts = {}
    for r in p["residues"]:
        counts[r[0]] = counts.get(r[0], 0) + 1
    chains = sorted(counts, key=lambda c: -counts[c])
    return ("chain " if len(chains) == 1 else "chains ") + ", ".join(chains)


def _pocket_line(p: dict) -> str:
    """One pocket's headline: rank, scores, size."""
    drug = p.get("drug_score", 0.0)
    verdict = "predicted druggable" if drug >= pkt.DRUGGABLE else "not predicted druggable"
    return (f"Pocket {p['rank']} — fpocket score {p.get('score', 0):.2f}, druggability "
            f"{drug:.2f} ({verdict}), volume ~{p.get('volume', 0):.0f} Å³, "
            f"{len(p['residues'])} lining residues on {_pocket_chains(p)}")


def _pocket_occupancy(p: dict) -> str:
    """What the deposited file has in this pocket, or that it is empty."""
    if not p["occupants"]:
        return "empty in this file (nothing is bound here in the deposited structure)"
    return "occupied in this file by " + "; ".join(
        f"{o['label']} ({srep.CATEGORY_LABELS.get(o['kind'], o['kind']).lower()}, "
        f"{o['n_in']} of {o['n_atoms']} atoms inside)" for o in p["occupants"])


def tool_find_pockets(chain: str = "", max_pockets: int = 5) -> ToolResult:
    """
    Find candidate ligand-binding pockets from the structure's own geometry.

    Runs fpocket on a protein-only copy of the loaded structure (ligands,
    ions and water removed), so occupied and empty sites are found on the
    same footing, then says for each pocket what, if anything, the
    deposited file has bound there. Works with no ligand in the file at
    all — which is the point: "where could something bind" is a question
    about the fold, not about what happened to be co-crystallised.

    Shows the top pockets by fpocket's own score plus any further pocket
    fpocket predicts druggable (druggability >= 0.5), since the two
    rankings disagree: in 4HHB the heme pockets rank 5th and below by
    pocket score but are the most druggable sites in the file.

    Args:
        chain      : Only pockets lined by this chain, e.g. "A". Empty for all.
        max_pockets: How many top-ranked pockets to show (1-15).

    Returns:
        A plain-text report, or why pocket detection could not run.
    """
    entry = active_structure()
    if not entry:
        return _fail(NO_STRUCTURE)
    if not pkt.fpocket_available():
        return _fail(pkt.UNAVAILABLE, {"unavailable": "fpocket"})
    wanted = ""
    if (chain or "").strip():
        wanted = chain.strip().upper().replace("CHAIN", "").strip()
        present = _atoms_of(entry)["chains_present"]
        if wanted not in present:
            return _fail(f"Chain {wanted} is not in {entry['pdb_id']}. "
                         f"Chains present: {', '.join(present)}", {"chains": list(present)})
    try:
        max_pockets = max(1, min(int(max_pockets), 15))
    except (TypeError, ValueError):
        max_pockets = 5

    ok, msg, result = _cached_pockets(entry["path"], Path(entry["path"]).stat().st_mtime,
                                      pkt.SCHEMA_VERSION)
    if not ok:
        return _fail(msg)
    pockets = [p for p in result["pockets"]
               if not wanted or any(r[0] == wanted for r in p["residues"])]
    scope = f", chain {wanted}" if wanted else ""
    st.session_state.pockets = {"path": entry["path"], "pdb_id": entry["pdb_id"],
                                "by_rank": {p["rank"]: p for p in pockets}}

    stripped = (f"{len(result['hetero'])} hetero group(s) removed before the search "
                "(ligands, ions; water too)" if result["hetero"]
                else "no ligand is bound in this file; water removed")
    lines = [f"Candidate binding pockets in {entry['pdb_id']}{scope} — fpocket, from this "
             f"structure's own geometry ({stripped}, so occupied and empty sites are found "
             "alike)."]
    # run_agent hands a pocket-only request this report verbatim, and attaches
    # data["summary"] to a mixed one (P12).
    data = {"pdb_id": entry["pdb_id"], "chain": wanted or None, "count": len(pockets),
            "relaxed": bool(result["relaxed"]), "summary": "", "pockets": []}
    if not pockets:
        lines.append("fpocket found no pocket"
                     + (f" lined by chain {wanted}" if wanted else "")
                     + ", even with a relaxed minimum size — there is no cavity here that "
                     "could hold a ligand.")
        data["summary"] = lines[-1]
        return _ok("\n".join(lines), data)

    if result["relaxed"]:
        best = max(p.get("drug_score", 0) for p in pockets)
        lines.append(
            f"No pocket met fpocket's default minimum size ({pkt.DEFAULT_MIN_SPHERES} alpha "
            "spheres). A relaxed "
            f"pass (≥{pkt.RELAXED_MIN_SPHERES} spheres) found {len(pockets)} small, shallow "
            f"surface grooves; the best druggability is {best:.2f}, so none is a "
            "conventional ligand-binding pocket. The protein may simply be too small to "
            "enclose one.")

    shown = pockets[:max_pockets]
    extra = [p for p in pockets[max_pockets:] if p.get("drug_score", 0) >= pkt.DRUGGABLE]
    # Most druggable first: qwen2.5:7b read "pocket 12 (0.87)" as the best
    # when 0.88 came later in rank order.
    druggable = sorted((p for p in pockets if p.get("drug_score", 0) >= pkt.DRUGGABLE),
                       key=lambda p: -p["drug_score"])
    empty_drug = [p for p in druggable if not p["occupants"]]
    lines.append(f"{len(pockets)} pocket(s) detected; top {len(shown)} by fpocket score shown"
                 + (f", plus {len(extra)} lower-ranked pocket(s) predicted druggable" if extra
                    else "") + ".")
    if druggable:
        lines.append(
            f"SUMMARY: {len(druggable)} of {len(pockets)} pocket(s) predicted druggable, most "
            "druggable first: " + "; ".join(
                f"pocket {p['rank']} ({p['drug_score']:.2f}, {_pocket_chains(p)}, "
                + (f"holds {p['occupants'][0]['label']}" if p["occupants"] else "empty") + ")"
                for p in druggable) + ". "
            + (f"Empty and predicted druggable: "
               + ", ".join(f"pocket {p['rank']}" for p in empty_drug) + "."
               if empty_drug else
               "None of them is empty — every predicted-druggable pocket already holds a "
               "bound group in this file.")
            + f" The other {len(pockets) - len(druggable)} pocket(s) are below the "
            f"druggability threshold ({pkt.DRUGGABLE}).")
    else:
        lines.append(f"SUMMARY: none of the {len(pockets)} pocket(s) reaches fpocket's "
                     f"druggability threshold ({pkt.DRUGGABLE}).")
    data["summary"] = lines[-1][len("SUMMARY: "):]
    data["druggable"] = [p["rank"] for p in druggable]
    data["empty_druggable"] = [p["rank"] for p in empty_drug]
    data["pockets"] = [{"rank": p["rank"], "score": round(p.get("score", 0), 2),
                        "druggability": round(p.get("drug_score", 0), 2),
                        "volume": round(p.get("volume", 0)),
                        "chains": sorted({r[0] for r in p["residues"]}),
                        "occupants": [o["label"] for o in p["occupants"]]}
                       for p in shown + extra]
    for p in shown + extra:
        lines.append("")
        lines.append(_pocket_line(p))
        lines.append(f"   lining: {_pocket_residues(p)}")
        lines.append(f"   {_pocket_occupancy(p)}")
    lines.append("")
    lines.append(
        "How to read this: pocket score is fpocket's ranking of how pocket-like a cavity "
        "is; druggability (Schmidtke & Barril 2010, 0-1, ≥0.5 predicted druggable) "
        "estimates whether a drug-like molecule could bind there tightly. Both are "
        "computational predictions from one static conformation, not binding "
        "measurements — a cryptic pocket that only opens when the protein moves or a "
        "ligand binds is not visible here.")
    lines.append("To show one in the viewer, say \"highlight pocket N\".")
    top = (druggable or shown)[0]["rank"]
    return _ok("\n".join(lines), data,
               hints=[f"highlight pocket {top}", f"find_contacts pocket {top}"])


_GENERIC_SITE = re.compile(r"\b(ligands?|pockets?|binding[ _-]?sites?|active[ _-]?sites?|"
                           r"cavit\w*|drugs?|inhibitors?)\b", re.I)


def _pocket_hint(target: str) -> str:
    """
    Appended when find_contacts/find_interactions could not resolve a
    generic 'ligand' / 'pocket' target: the model reached for a contact
    search to answer a pocket question (the pre-P12 answer to "find the
    hidden binding pocket", which then asked the user for residue ranges).
    """
    if not _GENERIC_SITE.search(target or ""):
        return ""
    entry = active_structure()
    het = sorted(r for r in (_atoms_of(entry)["resnames_present"] if entry else [])
                 if r not in srep.STANDARD_AA and r not in squ.AA3_TO_1
                 and r not in ("HOH", "WAT", "DOD"))
    named = (f" For contacts, name one of this file's hetero groups: {', '.join(het)}."
             if het else "")
    return (named + " To find candidate binding pockets from the structure's geometry "
            "— with or without a bound ligand — call find_pockets.")


_GENERIC_LIGAND = re.compile(r"(?:the\s+|a\s+|any\s+|bound\s+|all\s+)*(?:bound\s+)?"
                             r"(?:ligands?|small[ _-]?molecules?|inhibitors?|drugs?|"
                             r"compounds?|het(?:ero)?(?:[ _-]?groups?|atoms?)?)", re.I)


def _generic_ligand(target: str, action: str):
    """
    Resolve a target that says only "the ligand" to this file's real hetero
    groups, before _pick() fails on it.

    4HHB "what does the ligand bind to" used to fail find_contacts('ligand')
    and the model then asked "which ligand?" without naming HEM or PO4.
    One non-water component -> its code. Several -> ask_user with each one
    as an option (ions and additives last). None -> (None, message).

    Args:
        target: The tool's target argument.
        action: What the chosen option should do, e.g. "list the residues
                within 4 Å of" — becomes each option's meaning.

    Returns:
        (target, None) with the code to use (unchanged when target is not a
        generic ligand word), or (None, tool-result text).
    """
    if not _GENERIC_LIGAND.fullmatch((target or "").strip()):
        return target, None
    summary = structure_summary()
    comps = (summary or {}).get("components") or []
    if not comps:
        return None, ("This structure has no bound ligand, cofactor or ion (waters "
                      "aside). To find candidate binding pockets from the geometry, "
                      "call find_pockets.")
    if len(comps) == 1:
        return comps[0]["code"], None
    rank = {"ligand": 0, "cofactor": 1, "ion": 2, "additive": 3}
    comps = sorted(comps, key=lambda c: rank.get(c.get("category"), 4))
    options = []
    for c in comps[:5]:
        name = srep.pretty_chemical(c.get("name") or "") or c["code"]
        where = ", ".join(c.get("chains") or [])
        label = (f"{c['code']} ({name}, {c.get('count', 1)}× "
                 f"{'chain' if len(c.get('chains') or []) == 1 else 'chains'} {where})")
        options.append({"label": label, "meaning": f"{action} {c['code']}"})
    return None, ask_clarification(
        "This structure has more than one hetero group — which one do you mean?", options)


def _pocket_target(spec: str):
    """
    Residue keys of 'pocket N' from the last find_pockets run on the active
    structure: (keys, label) or (None, error). (None, None) when spec does
    not name a pocket at all.
    """
    m = re.fullmatch(r"(?:fpocket\s+)?pocket\s*#?\s*(\d+)", spec.strip(), re.I)
    if not m:
        return None, None
    entry, found = active_structure(), st.session_state.get("pockets")
    if not found or not entry or found["path"] != entry["path"]:
        return None, "No pockets have been found for this structure yet — run find_pockets first."
    p = found["by_rank"].get(int(m.group(1)))
    if not p:
        return None, (f"There is no pocket {m.group(1)} in {found['pdb_id']}. Pockets: "
                      + ", ".join(str(r) for r in list(found["by_rank"])[:20]))
    keys = [(c.strip(), r, i) for c, r, i, _ in p["residues"]]
    return keys, (f"pocket {p['rank']} (druggability {p.get('drug_score', 0):.2f}; "
                  f"{len(keys)} lining residues — {_pocket_residues(p)})")


@st.cache_data(show_spinner=False)
def _cached_esm_position(path: str, mtime: float, chain: str, index: int, model: str):
    """ESM-2 masked-marginal log-probs at one position; mtime/model bust the cache."""
    tokens, _, _ = esm.chain_sequence(path, chain)
    return esm.position_log_probs(tokens, index)


_MUTATION_TOKEN = re.compile(r"^\s*([A-Za-z])(\d+)([A-Za-z])\s*$")
_AA_WORDS = "|".join(sorted(set(esm.AA_NAMES) | {k.lower() for k in esm.ONE_TO_3.values()},
                            key=len, reverse=True))


def _mutation_from_prompt(prompt: str, residue: str) -> str:
    """
    The substituted amino acid the user named for this residue, or "".

    Reads "H92A" / "his92ala" style tokens for the same residue number, then
    "to alanine" / "to Ala" / "to A" (a bare letter only when capitalised, so
    the article "to a" never counts).
    """
    num = re.search(r"\d+", residue or "")
    for wt, n, mt in re.findall(r"\b([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})\b", prompt):
        if (not num or n == num.group()) and esm.parse_amino_acid(wt) and esm.parse_amino_acid(mt):
            return esm.parse_amino_acid(mt)
    m = (re.search(rf"\b(?:to|into|with|by|for)\s+(?:an?\s+)?({_AA_WORDS})\b", prompt, re.I)
         or re.search(r"\b(?:to|into|with|by|for)\s+([A-Z])\b", prompt))
    return esm.parse_amino_acid(m.group(1)) if m else ""


def tool_predict_mutation_effect(residue: str, mutant: str = "", wildtype: str = "",
                                 chain: str = "") -> ToolResult:
    """
    Predict how tolerated a single amino-acid substitution is — a MODEL
    PREDICTION from ESM-2 sequence statistics, never a measurement.

    Method: zero-shot masked-marginal log-likelihood ratio (Meier et al.
    2021). The loaded chain's observed sequence is fed to ESM-2 with the
    position masked; the score is log p(mutant) - log p(wild type) there.
    With no mutant, all 19 substitutions are ranked — the "which residues
    are tolerated here / how constrained is this position" question.

    Args:
        residue : The position, e.g. "45", "A/45", "HIS92", "B:92", or a whole
                  mutation "H92A" (then mutant/wildtype come from it).
        mutant  : Substituted amino acid ("A", "Ala", "alanine"); empty ranks all 19.
        wildtype: The residue the user says is there; checked against the
                  structure, never silently overridden.
        chain   : Chain id; needed only when several chains carry the position.

    Returns:
        A plain-text, caveated report, a NEEDS USER CHOICE question when the
        chain is ambiguous, or why the prediction could not run.
    """
    entry = active_structure()
    if not entry:
        return _fail(NO_STRUCTURE)
    if not esm.esm_available():
        return _fail(esm.unavailable_reason(), {"unavailable": "esm"})

    m = _MUTATION_TOKEN.match(residue or "")
    if m:
        wildtype = wildtype or m.group(1)
        mutant = mutant or m.group(3)
        residue = m.group(2)
    spec = mz.parse_spec(residue or "")
    if spec["resseq"] is None:
        return _fail(f"Could not read a residue number from '{residue}'. Give the position, "
                     "e.g. '45' or 'A/45'.")
    resseq, icode = spec["resseq"], spec["icode"] or ""

    mut = esm.parse_amino_acid(mutant) if (mutant or "").strip() else ""
    if (mutant or "").strip() and not mut:
        return _fail(f"'{mutant}' is not one of the 20 standard amino acids.")
    wt_claim = esm.parse_amino_acid(wildtype) if (wildtype or "").strip() else ""
    if not wt_claim and spec["resname"]:
        wt_claim = esm.parse_amino_acid(spec["resname"])

    path = entry["path"]
    mtime = Path(path).stat().st_mtime
    residues = squ.parse_structure_residues(path)
    wanted = (chain or spec["chain"] or "").strip().upper().replace("CHAIN", "").strip()
    holders = {ch: r for ch, rs in residues.items() for r in rs
               if r["kind"] == "protein" and r["resseq"] == resseq and r["icode"] == icode}
    if wanted:
        if wanted not in residues:
            return _fail(f"Chain {wanted} is not in {entry['pdb_id']}. "
                         f"Chains present: {', '.join(residues)}", {"chains": list(residues)})
        if wanted not in holders:
            return _fail(f"Residue {resseq}{icode} is not an observed amino acid in "
                         f"{entry['pdb_id']} chain {wanted}.")
        use = wanted
    elif not holders:
        return _fail(f"No chain of {entry['pdb_id']} has an observed amino acid numbered "
                     f"{resseq}{icode}.")
    else:
        # "H92A" in hemoglobin: only chain B has a His at 92, so no question.
        if wt_claim and any(r["one"] == wt_claim for r in holders.values()):
            holders = {ch: r for ch, r in holders.items() if r["one"] == wt_claim}
        # Identical chains (hemoglobin A/C) score the same; distinct ones may not.
        seqs = {}
        for ch in holders:
            key = "".join(r["one"] for r in residues[ch] if r["kind"] == "protein")
            seqs.setdefault(key, []).append(ch)
        if len(seqs) > 1:
            names = {c["chain"]: c["molecule"] for c in chain_molecules(entry)}
            opts = [{"label": f"Chain {chs[0]}"
                              + (f" ({names[chs[0]]})" if names.get(chs[0]) else "")
                              + f" — residue {holders[chs[0]]['resname']} {resseq}",
                     "meaning": f"predict the mutation effect at residue {resseq} of chain {chs[0]}"}
                    for chs in seqs.values()]
            return ask_clarification(
                f"Residue {resseq} exists in several different chains of "
                f"{entry['pdb_id']}. Which chain do you mean?", opts)
        use = sorted(holders)[0]
    target = holders[use]
    wt = target["one"]
    if wt not in esm.AMINO_ACIDS:
        return _fail(f"Residue {target['resname']} {resseq} in chain {use} is non-standard; "
                     "ESM-2 scores only the 20 standard amino acids.")
    if wt_claim and wt_claim != wt:
        return _fail(f"Residue {resseq} of {entry['pdb_id']} chain {use} is "
                     f"{target['resname']} ({wt}), not {wt_claim}. Check the position or "
                     "numbering — nothing was predicted.",
                     {"chain": use, "residue": resseq, "wildtype": wt})
    if mut == wt:
        return _fail(f"{wt}{resseq}{mut} is not a substitution — the residue is already {wt}.")

    tokens, index_of, notes = esm.chain_sequence(path, use)
    idx = index_of[(resseq, icode)]
    with st.spinner("Scoring with ESM-2…"):
        res = _cached_esm_position(path, mtime, use, idx, esm.model_name())
    if not res.get("ok"):
        return _fail(f"ESM-2 prediction failed: {res.get('error', 'unknown error')}")
    lp = res["log_probs"]
    ranked = sorted(lp, key=lp.get, reverse=True)
    p_wt, wt_rank = math.exp(lp[wt]), ranked.index(wt) + 1

    names = {c["chain"]: c["molecule"] for c in chain_molecules(entry)}
    who = f" ({names[use]})" if names.get(use) else ""
    model = esm.model_name().split("/")[-1]
    lines = [
        "MODEL PREDICTION — ESM-2 zero-shot masked-marginal log-likelihood ratio "
        f"({model}), from sequence statistics only. Not a measurement, not a "
        "ΔΔG, and not computed from this structure's 3D coordinates.",
        f"Position: {entry['pdb_id']} chain {use}{who}, {target['resname']} {resseq}{icode}",
    ]
    if mut:
        llr = lp[mut] - lp[wt]
        lines.append(f"{wt}{resseq}{mut}: log-likelihood ratio {llr:+.2f} → {esm.band(llr)}")
    # Stated in words: qwen2.5:7b called a native probability of 0.20 "high".
    expect = ("strongly expected — a tightly constrained position" if p_wt >= 0.5 else
              "moderately expected" if p_wt >= 0.2 else
              "only weakly expected — the position is permissive or the model is unsure")
    lines.append(f"Native {wt}: model probability {p_wt:.2f}, rank {wt_rank} of 20 at this "
                 f"masked position — {expect}.")
    top = ", ".join(f"{a} {math.exp(lp[a]):.2f}" for a in ranked[:5])
    lines.append(f"Most compatible residues here: {top}")
    if not mut:
        # Grouped by band, each group self-contained: a flat list of 19 scores
        # followed by a separate band legend was misread by qwen2.5:7b.
        groups = {}
        for a in ranked:
            if a != wt:
                groups.setdefault(esm.band(lp[a] - lp[wt]), []).append(
                    f"{wt}{resseq}{a} ({esm.ONE_TO_3[a].title()}, {lp[a] - lp[wt]:+.1f})")
        lines.append("All 19 substitutions, by predicted effect "
                     "(log-likelihood ratio vs native in brackets):")
        for _, text in esm.BANDS:
            lines.append(f"  {text}: " + (", ".join(groups.get(text, [])) or "none"))
        lines.append("This is the model's sequence constraint at the position, not an "
                     "alignment-based conservation score.")

    caveats = ["the score bands are heuristic cut-offs, not calibrated probabilities",
               "ESM-2 scores how well a residue fits the evolutionary sequence pattern; it "
               "cannot see binding partners, ligands, or gain-of-function effects, so a "
               "mild score does not mean clinically or functionally harmless"]
    if wt_rank > 3 or p_wt < 0.10:
        caveats.insert(0, "LOW CONFIDENCE: the model does not strongly expect the native "
                          "residue here, so this position's scores are weak evidence")
    observed = [i for i, t in enumerate(tokens) if t is not None]
    if idx - observed[0] < 3 or observed[-1] - idx < 3:
        caveats.append("residue is within 3 positions of the observed chain end, where "
                       "ESM predictions are less reliable")
    if len(observed) < 60:
        caveats.append(f"short chain ({len(observed)} observed residues) — little sequence "
                       "context for the model")
    start, end = res["window"]
    if end - start < len(tokens):
        caveats.append(f"chain longer than ESM-2's 1022-residue limit; scored on a "
                       f"{end - start}-residue window around the position")
    caveats.append("sequence taken from the residues observed in this file, not the full "
                   "UniProt sequence")
    caveats.extend(notes)
    lines.append("Caveats: " + "; ".join(caveats) + ".")
    # data["model_prediction"] is what run_agent's ESM caveat footer keys on.
    data = {"model_prediction": True, "model": model, "pdb_id": entry["pdb_id"],
            "chain": use, "residue": f"{resseq}{icode}", "wildtype": wt,
            "native_probability": round(p_wt, 2), "native_rank": wt_rank,
            "top": {a: round(math.exp(lp[a]), 2) for a in ranked[:5]},
            "low_confidence": wt_rank > 3 or p_wt < 0.10}
    if mut:
        data.update({"mutant": mut, "llr": round(lp[mut] - lp[wt], 2),
                     "band": esm.band(lp[mut] - lp[wt])})
    else:
        data["llr"] = {a: round(lp[a] - lp[wt], 1) for a in ranked if a != wt}
    return _ok("\n".join(lines), data)


# P11: a stability question that names a mutation is answerable (as a proxy);
# one that names nothing to compute against is not. These read the mutation
# out of the user's own words so the stability gate in run_agent can tell
# the two apart without asking the model.
_MUTATE_WORDS = re.compile(r"\b(mutat\w*|substitut\w*|replac\w*|swap\w*|variant|mutant)\b", re.I)


def _stability_mutations(prompt: str) -> list:
    """
    (residue, mutant, wildtype, chain) for each mutation a prompt names, or [].

    "H92A" / "Leu99Ala" tokens first (up to 4); otherwise "residue 45" /
    "Leu 99" / "A/45" plus a substitution ("to alanine") or a mutation word
    ("mutating residue 45" → all 19 ranked). A residue with neither — "is
    residue 45 important for stability" — names nothing to compute.
    """
    chain_m = (re.search(r"\bchain\s+([A-Za-z0-9])\b", prompt, re.I)
               or re.search(r"\b([A-Za-z])\s*[/:]\s*\d+\b", prompt))
    chain = chain_m.group(1).upper() if chain_m else ""
    found, seen = [], set()
    for wt, n, mt in re.findall(r"\b([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})\b", prompt):
        if n not in seen and esm.parse_amino_acid(wt) and esm.parse_amino_acid(mt):
            seen.add(n)
            found.append((n, esm.parse_amino_acid(mt), esm.parse_amino_acid(wt), chain))
    if found:
        return found[:4]
    m = (re.search(r"\b(?:residue|res|position|pos|site)\s*(?:#|no\.?\s*)?"
                   r"(?:[A-Za-z]\s*[/:]\s*)?(\d+)\b", prompt, re.I)
         or re.search(rf"\b({_AA_WORDS})\s*-?\s*(\d+)\b", prompt, re.I)
         or re.search(r"\b[A-Za-z]\s*[/:]\s*(\d+)\b", prompt))
    if not m:
        return []
    num = m.group(m.lastindex)
    wt = esm.parse_amino_acid(m.group(1)) if m.lastindex == 2 else ""
    mut = _mutation_from_prompt(prompt, num)
    if not mut and not _MUTATE_WORDS.search(prompt):
        return []
    return [(num, mut, wt, chain)]


# Theoretical max solvent-accessible surface per residue, Å² (Tien et al. 2013).
_MAX_ASA = {"A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "Q": 225, "E": 223,
            "G": 104, "H": 224, "I": 197, "L": 201, "K": 236, "M": 224, "F": 240,
            "P": 159, "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174}
_VDW = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}


def _residue_rsa(path: str, chain: str, resseq: int, icode: str = "", cover: str = "all"):
    """
    Relative solvent accessibility (0-1) of one residue, or None.

    Shrake-Rupley with a 1.4 Å probe over the first model's heavy atoms.
    cover picks what may bury the residue: "all" (every chain and bound
    ligand), "protein" (every chain, no ligands) or "chain" (its own chain
    only) — comparing them tells core burial from an interface or a
    ligand site. Water never counts. Stdlib only, and only atoms near the
    residue are tested, so it is fast.
    """
    atoms, mine, one = [], [], ""
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in " A":
                continue
            if line[17:20].strip() in ("HOH", "WAT", "DOD"):
                continue
            if cover != "all" and line.startswith("HETATM"):
                continue
            if cover == "chain" and (line[21].strip() or "_") != chain:
                continue
            el = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
            if el in ("H", "D"):
                continue
            try:
                xyz = tuple(float(line[c:c + 8]) for c in (30, 38, 46))
                key = (line[21].strip() or "_", int(line[22:26]), line[26].strip())
            except ValueError:
                continue
            atom = (xyz, _VDW.get(el, 1.8) + 1.4)
            atoms.append(atom)
            if key == (chain, resseq, icode) and line.startswith("ATOM  "):
                mine.append(atom)
                one = squ.one_letter(line[17:20].strip())
    if not mine or one not in _MAX_ASA:
        return None
    n = 100
    dots = [(math.cos(2.39996 * i) * math.sqrt(1 - z * z),
             math.sin(2.39996 * i) * math.sqrt(1 - z * z), z)
            for i, z in ((i, 1 - (2 * i + 1) / n) for i in range(n))]
    centre = tuple(sum(a[0][k] for a in mine) / len(mine) for k in range(3))
    near = [a for a in atoms if math.dist(a[0], centre) < 16.0]
    area = 0.0
    for (c, r) in mine:
        others = [a for a in near if a[0] != c and math.dist(a[0], c) < r + a[1]]
        free = sum(1 for d in dots
                   if not any(math.dist((c[0] + r * d[0], c[1] + r * d[1], c[2] + r * d[2]),
                                        o[0]) < o[1] for o in others))
        area += 4 * math.pi * r * r * free / n
    return min(area / _MAX_ASA[one], 1.0)


def _stability_estimate(mutations: list) -> str:
    """
    The P11 answer to a well-posed "would this mutation destabilize it":
    P10's ESM-2 score per mutation, the position's burial, and a reading of
    the two together — framed throughout as a proxy, never a ΔΔG.
    """
    entry = active_structure()
    if not entry:
        return "No structure is loaded — load one first, then ask about the mutation again."
    if not esm.esm_available():
        return ("A mutation-stability estimate needs the ESM-2 model, which is not "
                "available here: " + esm.unavailable_reason())
    parts = ["STABILITY ESTIMATE — a computational proxy, not a measured or computed ΔΔG. "
             "Method: ESM-2 zero-shot mutation score (sequence statistics) read "
             "together with how buried the position is in this structure."]
    for resseq, mut, wt, chain in mutations:
        report = tool_predict_mutation_effect(resseq, mut, wt, chain)
        if not (report.ok and report.data.get("model_prediction")):
            # A chain question or an error (wrong wild type, no such residue):
            # nothing was scored, so there is nothing to frame.
            return report
        use = report.data.get("chain") or chain
        # A float only when one substitution was scored; a dict for all 19.
        llr = report.data.get("llr")
        llr = llr if isinstance(llr, float) else None
        rsa = _residue_rsa(entry["path"], use, int(resseq))
        buried, exposed = rsa is not None and rsa <= 0.10, rsa is not None and rsa >= 0.30
        where = ("of unknown burial" if rsa is None else
                 f"buried ({rsa:.0%} relative solvent accessibility)" if buried else
                 f"exposed (surface; {rsa:.0%} relative solvent accessibility)" if exposed else
                 f"partly buried ({rsa:.0%} relative solvent accessibility)")
        # Buried by what: its own fold, a partner chain, or a bound ligand.
        # Proximal His92 of hemoglobin is 0% accessible only because the
        # heme covers it — a cofactor site, not core packing.
        site = ""
        if rsa is not None and rsa < 0.30:
            own = _residue_rsa(entry["path"], use, int(resseq), cover="chain")
            prot = _residue_rsa(entry["path"], use, int(resseq), cover="protein")
            if prot - rsa >= 0.10:
                site = "a bound ligand/cofactor"
            elif own - prot >= 0.10:
                site = "another chain (an interface)"
            if site:
                where += f"; mostly covered by {site} — {own:.0%} accessible in its own chain alone"
                buried = False
        if llr is None:
            reading = ("No single substitution named — the ranking above shows which "
                       "replacements the model expects this position to accept least. "
                       + ("At a buried position those are the likelier destabilizing ones."
                          if buried else
                          "At a surface position the constraint may be functional rather "
                          "than folding stability." if exposed else
                          "The score alone cannot separate a stability cost from a "
                          "functional one."))
        else:
            if llr >= -3:
                reading = ("No sequence-level signal of destabilization. That is weak "
                           "evidence the mutation is near-neutral for stability, not proof.")
            elif site:
                reading = (f"Disfavoured substitution at a position covered by {site} — the "
                           "constraint is likely about that contact (binding or assembly); "
                           "the mutation may weaken it, which can in turn destabilize the "
                           "complex, but the score cannot say how much of the effect is "
                           "folding stability.")
            elif buried:
                reading = ("Disfavoured substitution at a buried position — consistent with "
                           "a destabilizing mutation (core packing is where substitutions "
                           "most often cost folding stability).")
            elif exposed:
                reading = ("Disfavoured substitution, but at a surface position — the "
                           "constraint may be functional (binding, catalysis, interfaces) "
                           "rather than folding stability, so a destabilizing effect is "
                           "less certain.")
            else:
                reading = ("Disfavoured substitution at a partly buried position — possibly "
                           "destabilizing; the score alone cannot separate a stability cost "
                           "from a functional one.")
        parts.append(f"{report}\nStructural context: position is {where}, computed from this "
                     "structure (Shrake-Rupley; other chains and ligands count as cover; "
                     "cut-offs ≤10% buried, ≥30% exposed are heuristic).\n"
                     f"Reading for stability: {reading}")
    if len(mutations) > 1:
        parts.append("Each mutation was scored on its own; effects of combined mutations "
                     "are not additive in general and were not estimated.")
    parts.append("Why this is only a proxy: ESM-2 scores evolutionary fitness, which mixes "
                 "folding stability with function, and its agreement with measured ΔΔG is "
                 "moderate in published benchmarks. A quantitative answer needs free-energy "
                 "calculations (e.g. FEP) or a thermal/chemical denaturation experiment.")
    return "\n\n".join(parts)


def tool_ask_user(question: str, options) -> ToolResult:
    """
    Put a clarifying question to the user instead of guessing.

    Args:
        question: One short question.
        options : 2-4 concrete readings of the request. A single string of
                  comma/semicolon/newline-separated choices is split, since a
                  small model sometimes sends the list that way.

    Returns:
        NEEDS USER CHOICE text, or why the question could not be asked.
    """
    if isinstance(options, str):
        options = [o.strip(" -*0123456789.)") for o in re.split(r"[;\n]|,\s(?=[A-Z])", options)]
    options = [o for o in options if str(o).strip()]
    if not question.strip():
        return _fail("ask_user needs a question.")
    return ask_clarification(question, options)


def tool_list_structures() -> ToolResult:
    """Describe every structure currently in the scene and how it is placed."""
    if not structures():
        return _ok("No structures are loaded.", {"loaded": []})
    lines = []
    for s in structures():
        bits = [s["pdb_id"]]
        if s["sid"] == st.session_state.active_sid:
            bits.append("(active)")
        if not s["visible"]:
            bits.append("(hidden)")
        if s["fit"]:
            bits.append(f"— superposed: {s['fit']}")
        chains = chain_map_line(s, (st.session_state.focus or {}).get("accession", ""),
                                show_drawn=True)
        if chains:
            bits.append(f"— {chains}")
        lines.append(" ".join(bits))
    return _ok("Loaded structures: " + "\n".join(lines),
               {"loaded": [s["pdb_id"] for s in structures()],
                "active": (active_structure() or {}).get("pdb_id"),
                "chains": {s["pdb_id"]: {c["chain"]: c["molecule"] for c in chain_molecules(s)}
                           for s in structures()}})


def _ngl_resname(resname: str) -> str:
    """
    Render a residue name as an unambiguous NGL selection token.

    NGL classifies a bare token as a residue name only when
    isNaN(parseInt(token)) holds. Modern PDB chemical component ids often start
    with a digit ("03Q", "1N1"), for which parseInt succeeds, so NGL would read
    them as a residue *number* and silently select the wrong atoms. The
    bracketed form is tested earlier in NGL's parser and always means resname.
    """
    rn = resname.strip().upper()
    return f"[{rn}]" if rn and rn[0].isdigit() else rn


def _chain_ids_present(u) -> list:
    """Chain ids in the active structure, in file order ([] when unknown)."""
    if u is None:
        return []
    try:
        ids = getattr(u.atoms, "chainIDs", None)
        if ids is None or not len(ids):
            ids = u.atoms.segids
        return list(dict.fromkeys(str(c) for c in ids if str(c).strip()))
    except Exception:
        return []


# "chain A", "chains C-K", "chains C:K", "chains C to K", "chains A, B and D".
# Each id must be a lone character, so "chain A and not chain B" stops at "and".
_CHAIN_LIST_RE = re.compile(
    r"\bchains?\s+([A-Za-z0-9](?:\s*(?:,|&|/|-|:|\band\b|\bor\b|\bto\b|\bthrough\b|\bthru\b)"
    r"\s*[A-Za-z0-9](?![A-Za-z0-9]))*)(?![A-Za-z0-9])",
    re.IGNORECASE)


def _chain_list_to_ngl(expression: str, u=None) -> str:
    """
    Rewrite every "chain(s) <ids>" phrase in an expression as NGL chain syntax.

    NGL has no "chain" keyword; passed through, "chains C:K" parsed as junk
    and colouring "the other chains" grey painted every chain grey. Ranges
    ("C-K", "C:K", "C to K") expand over the chains actually in the structure
    when it is known, else over the alphabet.
    """
    present = _chain_ids_present(u)

    def expand(spec: str) -> list:
        ids = []
        for part in re.split(r"\s*(?:,|&|/|\band\b|\bor\b)\s*", spec, flags=re.I):
            m = re.fullmatch(r"([A-Za-z0-9])\s*(?:-|:|\bto\b|\bthrough\b|\bthru\b)\s*([A-Za-z0-9])",
                             part.strip(), re.I)
            if m:
                lo, hi = m.group(1), m.group(2)
                if lo in present and hi in present:
                    i, j = present.index(lo), present.index(hi)
                    ids += present[min(i, j):max(i, j) + 1]
                else:
                    ids += [chr(c) for c in range(ord(lo), ord(hi) + 1)]
            elif part.strip():
                ids.append(part.strip())
        return list(dict.fromkeys(ids))

    def sub(m):
        ids = expand(m.group(1))
        if len(ids) == 1:
            return f":{ids[0]}"
        return "(" + " or ".join(f":{c}" for c in ids) + ")"

    return _CHAIN_LIST_RE.sub(sub, expression)


# Words NGL's selection language understands. Anything else lowercase and
# alphabetic in a selection is English the translation missed — NGL would
# read it as a residue name and quietly select the wrong atoms.
_NGL_WORDS = {
    "and", "or", "not", "all", "none", "protein", "nucleic", "rna", "dna",
    "polymer", "hetero", "ligand", "ion", "saccharide", "sugar", "water",
    "organic", "backbone", "sidechain", "sidechainattached", "helix", "sheet",
    "turn", "hydrophobic", "hydrophilic", "aromatic", "polar", "charged",
    "acidic", "basic", "small", "nucleophilic", "amid", "aliphatic", "cyclic",
    "bonded", "ring", "metal", "cg", "hydrogen", "helix", "within",
}


def _ngl_unknown_words(ngl_sel: str) -> list:
    """Lowercase words in an NGL selection that NGL does not know."""
    stripped = re.sub(r"\[[^\]]*\]|[:_@.#%/^][A-Za-z0-9,*-]*", " ", ngl_sel or "")
    return [w for w in re.findall(r"\b[a-z][a-z]+\b", stripped)
            if w not in _NGL_WORDS]


def _expression_to_ngl(expression: str, u) -> tuple[str, str]:
    """
    Translate a PyMOL-style / plain-English expression to an NGL selection string.

    Handles keyword aliases, chain identifiers, residue names, element symbols,
    secondary structure codes, and non-standard residue detection (via MDAnalysis
    when available). Returns (ngl_string, human_label).
    """
    # Hallucinated residue names the LLM sometimes invents instead of "protein"
    _FAKE_RESNAMES = {"STANDARD", "CANONICAL", "NORMAL", "RESIDUE", "AMINO", "AMINOACID"}

    # Direct keyword map: plain-English / PyMOL → NGL selection strings
    ngl_map = {
        # protein / standard residues
        "protein": "protein",
        "polymer.protein": "protein",
        "standard": "protein",
        "standard residues": "protein",
        "std_res": "protein",
        "canonical residues": "protein",
        "amino acids": "protein",
        "amino acid": "protein",
        "std": "protein",
        "ATOM": "protein",
        # ligand / small molecules
        "ligand": "ligand",
        "small molecule": "ligand",
        "small molecules": "ligand",
        "organic": "organic",
        "HETATM": "ligand",
        # solvent / hetero
        "solvent": "water",
        "water": "water",
        "hetero": "hetero",
        # structural
        "backbone": "backbone",
        "sidechain": "sidechain",
        "nucleic": "nucleic",
        "dna": "nucleic",
        "rna": "nucleic",
        # chains (bare word = all polymer chains)
        "chain": "polymer",
        "chains": "polymer",
        "all chains": "polymer",
    }

    expr_lower = expression.strip().lower()

    if expr_lower in ngl_map:
        return ngl_map[expr_lower], expr_lower

    # "chain A" → NGL ":A"
    # Residue numbers, optionally with a chain: "resi 58", "resid 20-30 and
    # chain A", "residue 58 of chain A". NGL has no 'resi' keyword — passed
    # through, "resi 58" silently matched residue 58 of every chain.
    m = re.fullmatch(r"(?:resi|resid|resnum|residues?)\s+(\d+)(?:\s*(?:-|to|through)\s*(\d+))?"
                     r"(?:\s+(?:and\s+|in\s+|of\s+|on\s+)?(?:chain|segid)\s+([a-z0-9]))?",
                     expr_lower)
    if m:
        lo, hi, chain = m.group(1), m.group(2), m.group(3)
        ngl = (f"{lo}-{hi}" if hi else lo) + (f":{chain.upper()}" if chain else "")
        label = (f"residues {lo}-{hi}" if hi else f"residue {lo}") + (
            f" of chain {chain.upper()}" if chain else " (every chain)")
        return ngl, label

    if _CHAIN_LIST_RE.fullmatch(expression.strip()):
        return _chain_list_to_ngl(expression.strip(), u), expression.strip()

    # "resn HEM and chain A" → NGL "HEM and :A". The single-name branch below
    # took the last word as the residue name, so this selected residue "A".
    if (re.match(r"(?:resn|resname)\s", expr_lower)
            and not re.fullmatch(r"(?:resn|resname)\s+\S+", expr_lower)):
        ngl = re.sub(r"\b(?:resn|resname)\s+([A-Za-z0-9]+)",
                     lambda mm: _ngl_resname(mm.group(1)), expression.strip(), flags=re.I)
        return _chain_list_to_ngl(ngl, u), expression.strip()

    # "resn ATP" / "resname ATP" → NGL "[ATP]"
    if expr_lower.startswith("resn ") or expr_lower.startswith("resname "):
        resname = expression.split()[-1].upper()
        # Guard against hallucinated residue names that mean "protein"
        if resname in _FAKE_RESNAMES:
            return "protein", "standard residues (protein)"
        return _ngl_resname(resname), f"resname {resname}"

    # "symbol C" → NGL "_C" (element selection)
    if expr_lower.startswith("symbol "):
        elem = expression.split()[-1].upper()
        return f"_{elem}", f"element {elem}"

    # "ss H/S/L" → NGL secondary structure keyword
    if expr_lower.startswith("ss "):
        ss_char = expression.split()[-1].upper()
        ss_map = {"H": "helix", "S": "sheet", "L": "turn"}
        ngl = ss_map.get(ss_char, "helix")
        return ngl, ngl

    # Non-standard residues — use MDAnalysis to enumerate them when available
    if "non-standard" in expr_lower or "nonstandard" in expr_lower:
        if u:
            std = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                   "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
                   "DA","DC","DG","DT","A","C","G","U","HOH","WAT","TIP","TIP3"}
            try:
                all_res = set(u.select_atoms("not water").resnames)
                nonstd = sorted(all_res - std)
                if nonstd:
                    # Use NGL residue-name selection — robust across PDB sources
                    ngl = " or ".join(_ngl_resname(r) for r in nonstd)
                    label = f"non-standard ({', '.join(nonstd)})"
                else:
                    ngl = "hetero"
                    label = "non-standard residues (hetero fallback)"
            except Exception:
                ngl = "hetero"
                label = "non-standard residues (hetero fallback)"
        else:
            ngl = "hetero"
            label = "non-standard residues (hetero fallback)"
        return ngl, label

    # "b > 50" — B-factor filter; cap serial list at 800 to avoid JS overflow
    if expr_lower.startswith("b "):
        if u:
            try:
                ag = u.select_atoms(f"tempfactor {expression[2:]}")
                serials = list(ag.atoms.ids[:800])
                if serials:
                    return "@" + ",".join(map(str, serials)), f"B-factor {expression[2:]}"
            except Exception:
                pass
        return "all", "B-factor filter (MDAnalysis unavailable)"

    # "chain A" inside a compound expression ("protein and not chain A") →
    # ":A". The whole-string case is handled above; this catches the rest.
    ngl = _chain_list_to_ngl(expression, u)
    ngl = re.sub(r"\bsegid\s+([A-Za-z0-9])\b", r":\1", ngl, flags=re.IGNORECASE)
    return ngl, expression


def _add_highlight(ngl: str, name: str) -> None:
    """Add a ball+stick highlight layer for a new selection (like PyMOL's pink highlight)."""
    highlight_colors = [
        "yellow", "orange", "hotpink", "cyan", "lime",
        "magenta", "gold", "tomato", "deepskyblue", "greenyellow",
    ]
    # Remove any previous highlight for the same selection name before adding a fresh one
    st.session_state.representations = [
        r for r in st.session_state.representations
        if not r.get("_sel_name") == name
    ]
    color_idx = len(st.session_state.selections) % len(highlight_colors)
    st.session_state.representations.append({
        "type": "ball+stick",
        "selection": ngl,
        "color": highlight_colors[color_idx],
        "transparency": 0.0,
        "sid": _scope_for(ngl),
        "_sel_name": name,   # internal tag used for later removal by tool_hide
    })


def tool_select(name: str, expression: str) -> ToolResult:
    """
    Create a named selection and immediately highlight it in the viewer.

    Translates the expression to an NGL selection string, optionally validates
    it against MDAnalysis, stores it under the given name, and adds a
    colour-coded ball+stick highlight layer.

    Expression examples:
      protein, ligand, organic, chain A, resn ATP, ss H, ss S,
      symbol C, b > 50, non-standard residues

    Args:
        name      : Short label for the selection (e.g. "atp_res", "chain_b").
        expression: Plain-English or PyMOL-style selection expression.

    Returns:
        Status string describing the resulting NGL selection, or a warning
        if the selection matches zero atoms in the current structure.
    """
    u = get_universe()
    ngl, label = _expression_to_ngl(expression, u)

    # Warn early if MDAnalysis confirms the selection is empty
    if u and not ngl.startswith("@"):
        try:
            mda_expr = _ngl_to_mda_approx(ngl)
            count = len(u.select_atoms(mda_expr))
            if count == 0:
                return _fail(
                    f"Warning: '{expression}' matched 0 atoms in {st.session_state.pdb_id}. "
                    f"This structure may not contain that residue/selection. "
                    f"Nothing was highlighted.", {"atoms": 0}
                )
        except Exception:
            pass

    st.session_state.selections[name] = ngl
    _add_highlight(ngl, name)
    short = ngl[:60] + "..." if len(ngl) > 60 else ngl
    return _ok(f"Selection '{name}' ({label}) highlighted as ball+stick. NGL: {short}",
               {"name": name, "ngl": ngl})


def tool_select_within(name: str, radius: float, target_selection: str) -> ToolResult:
    """Select all atoms within `radius` Å of `target_selection`, expanded to whole residues."""
    u = get_universe()
    if not u:
        # Approximate fallback when MDAnalysis is unavailable
        ngl = f"({resolve_selection(target_selection)}) or polymer"
        st.session_state.selections[name] = ngl
        return _ok("MDAnalysis unavailable; approximate selection stored",
                   {"name": name, "approximate": True})

    try:
        target_ngl = resolve_selection(target_selection)
        mda_target_expr = _ngl_to_mda_approx(target_ngl)
        target_ag = u.select_atoms(mda_target_expr)
        nearby = u.select_atoms(f"byres (around {radius} group target)", target=target_ag)
        ngl = mda_to_ngl_serial(nearby)
        st.session_state.selections[name] = ngl
        return _ok(f"Selection '{name}': {len(nearby.residues)} residues within {radius}Å "
                   f"of '{target_selection}'",
                   {"name": name, "residues": len(nearby.residues), "radius": radius,
                    "residue_ids": [f"{r.resname}{r.resid}:{r.segid}"
                                    for r in nearby.residues[:200]]})
    except Exception as e:
        return _fail(f"Error: {e}")


def tool_select_by_bfactor(name: str, operator: str, threshold: float) -> ToolResult:
    """
    Select atoms where B-factor satisfies the given comparison.

    Args:
        name      : Label for the resulting named selection.
        operator  : Comparison operator, either ">" or "<".
        threshold : B-factor cutoff value.

    Returns:
        Status string with the atom count, or an error message.
    """
    u = get_universe()
    if not u:
        return _fail("MDAnalysis unavailable", {"unavailable": "mdanalysis"})
    try:
        # MDAnalysis compares attributes only through `prop`; a bare
        # "tempfactor > 50" failed to parse on every call.
        ag = u.select_atoms(f"prop tempfactor {operator} {float(threshold)}")
        ngl = mda_to_ngl_serial(ag)
        st.session_state.selections[name] = ngl
        return _ok(f"Selection '{name}': {len(ag)} atoms with B-factor {operator} {threshold}",
                   {"name": name, "atoms": len(ag), "residues": len(ag.residues)})
    except Exception as e:
        return _fail(f"Error: {e}")


def _normalise_color(color: str) -> str:
    """
    Map a user/LLM colour word onto a scheme NGL actually understands.

    NGL has no "spectrum" scheme -- the rainbow-by-residue equivalent is
    "residueindex" -- and an unrecognised name is silently ignored by NGL
    rather than raising, so aliases are resolved here instead.
    """
    c = (color or "element").lower().strip()
    aliases = {
        "spectrum": "residueindex", "rainbow": "residueindex",
        "chain": "chainname", "chains": "chainname",
        "ss": "sstruc", "secondary structure": "sstruc",
        "b": "bfactor", "b-factor": "bfactor", "temperature": "bfactor",
        "atom": "element", "cpk": "element",
        "gray": "grey", "hydrophobic": "hydrophobicity",
    }
    c = aliases.get(c, c)
    if c in NGL_COLOR_SCHEMES.values():
        return c
    return c   # a plain CSS/NGL colour name such as "red" is passed through


def tool_show(rep_type: str, selection: str, color: str = "element",
              exclusive: bool = False) -> ToolResult:
    """
    Add or replace a visual representation layer for a given selection.

    Normalises PyMOL/VMD representation names to their NGL equivalents, resolves
    named selections to NGL strings, and deduplicates by removing any existing
    layer of the same type+selection before appending the new one.

    Args:
        rep_type : Representation style (e.g. "cartoon", "surface", "licorice").
        selection: Named selection key or raw NGL expression.
        color    : NGL color scheme or named color (default: "element").
        exclusive: When True ("show only X"), every other representation is
            dropped first — otherwise the structure's default full-coverage
            layer (e.g. "protein", added on load) keeps drawing everything
            else right alongside the new one.

    Returns:
        Confirmation string.
    """
    # Map PyMOL/VMD names → NGL.js equivalents. licorice is NOT collapsed into
    # ball+stick: NGL has a distinct licorice representation of its own.
    rep_aliases = {
        "sticks":    "licorice",
        "stick":     "licorice",
        "spheres":   "spacefill",
        "sphere":    "spacefill",
        "vdw":       "spacefill",
        "lines":     "line",
        "dots":      "point",
        "mesh":      "surface",
        "putty":     "tube",
        "wire":      "line",
        "ball and stick": "ball+stick",
        "ball-and-stick": "ball+stick",
    }
    rep_type = rep_aliases.get(rep_type.lower().strip(), rep_type.lower().strip())

    # NGL silently ignores an unknown representation type, which looks to the
    # user like the command was simply dropped. Fail loudly instead.
    if rep_type not in NGL_REP_TYPES:
        return _fail(f"Unknown representation '{rep_type}'. Valid styles: "
                     + ", ".join(NGL_REP_TYPES), {"options": list(NGL_REP_TYPES)})

    color = _normalise_color(color)

    # Named selections take priority; otherwise translate the expression to NGL
    sels = st.session_state.selections
    if selection in sels:
        ngl_sel = sels[selection]
    else:
        u = get_universe()
        ngl_sel, _ = _expression_to_ngl(selection, u)

    if exclusive:
        st.session_state.representations = []
    else:
        # Replace any existing representation of the same type+selection instead of stacking
        st.session_state.representations = [
            r for r in st.session_state.representations
            if not (r["type"] == rep_type and r["selection"] == ngl_sel)
        ]
    st.session_state.representations.append({
        "type": rep_type, "selection": ngl_sel,
        "color": color, "transparency": 0.0,
        "sid": _scope_for(ngl_sel),
    })
    suffix = " — everything else hidden" if exclusive else ""
    return _ok(f"Showing {rep_type} for '{selection}' ({color}){suffix}",
               {"rep_type": rep_type, "ngl": ngl_sel, "color": color,
                "exclusive": bool(exclusive)})


def tool_hide(selection: str) -> ToolResult:
    """
    Hide a selection: drop any layer that is exactly it, and narrow every
    broader surviving layer so it no longer draws those atoms either.

    A structure loads with one "protein"/"all"-scoped cartoon layer covering
    every chain. Asking to hide one chain out of several used to only ever
    remove a layer whose stored selection matched the hidden one exactly —
    which that catch-all layer never does, so it kept right on drawing the
    "hidden" chain and the request silently did nothing. Narrowing every
    other surviving layer with "and not (<hidden>)" is what actually removes
    those atoms from the picture regardless of which layer was drawing them.

    Args:
        selection: Named selection key or raw NGL expression to hide.

    Returns:
        Status string describing what changed.
    """
    ngl_sel = resolve_selection(selection)
    before = len(st.session_state.representations)
    exclude_clause = f"and not ({ngl_sel})"
    survivors = []
    narrowed = 0
    for r in st.session_state.representations:
        # Match by internal selection-name tag OR by NGL string — not both simultaneously
        if r.get("_sel_name") == selection or r["selection"] == ngl_sel:
            continue
        if exclude_clause not in r["selection"]:
            r["selection"] = f"{r['selection']} {exclude_clause}"
            narrowed += 1
        survivors.append(r)
    st.session_state.representations = survivors
    removed = before - len(survivors)
    detail = ", ".join(
        p for p in (
            f"removed {removed} layer(s)" if removed else "",
            f"narrowed {narrowed} layer(s) to exclude it" if narrowed else "",
        ) if p
    ) or "no matching layers found"
    return _ok(f"Hid '{selection}' ({detail})",
               {"ngl": ngl_sel, "removed": removed, "narrowed": narrowed})


def tool_hide_all() -> ToolResult:
    """Remove every representation layer from the viewer, leaving a blank canvas."""
    n = len(st.session_state.representations)
    st.session_state.representations = []
    return _ok("All representations hidden", {"removed": n})


def tool_show_all(rep_type: str = "cartoon") -> ToolResult:
    """
    Replace all current representations with a single full-structure view.

    Args:
        rep_type: NGL representation type to apply to all atoms (default: "cartoon").

    Returns:
        Confirmation string.
    """
    st.session_state.representations = [
        {"type": rep_type, "selection": "all", "color": "residueindex", "transparency": 0.0}
    ]
    return _ok(f"Showing {rep_type} for all atoms", {"rep_type": rep_type})


def tool_color(color: str, selection: str) -> ToolResult:  # noqa: D401
    """
    Apply a color to all existing representation layers for a given selection.

    If no layer currently exists for the selection, a new cartoon layer is
    created with the requested color so the change is still visible.

    Args:
        color    : Named color (e.g. "red") or NGL color scheme (e.g. "bfactor").
        selection: Named selection key or raw NGL expression.

    Returns:
        Confirmation string.
    """
    color = _normalise_color(color)
    bulk = ("cartoon", "ribbon", "tube", "rope", "backbone", "trace", "surface")
    reps = st.session_state.representations

    # "the rest" / "other chains": every layer colour() carved a piece out of
    # still draws exactly what nobody has coloured yet, so recolour those.
    if _REST_RE.fullmatch((selection or "").strip()):
        rest = [r for r in reps if r["type"] in bulk and not r.get("_colored")
                and not r.get("_sel_name")]
        if not rest:
            selection = "polymer"
        else:
            for r in rest:
                r["color"] = color
            return _ok(f"Colored the rest (everything not coloured explicitly) as {color}",
                       {"color": color, "layers": len(rest)})

    ngl_sel = resolve_selection(selection)
    unknown = _ngl_unknown_words(ngl_sel)
    if unknown:
        return _fail(f"Error: could not read the selection '{selection}' (unknown word(s): "
                     f"{', '.join(unknown)}). Nothing was coloured. Use 'chain A', "
                     f"'chains C-K', 'chains C, D and E', or 'rest' for everything "
                     f"not yet coloured.", {"unknown_words": list(unknown)})

    updated = 0
    for r in reps:
        if r["selection"] == ngl_sel:
            r["color"] = color
            updated += 1
    if updated:
        return _ok(f"Colored '{selection}' as {color}",
                   {"color": color, "ngl": ngl_sel, "layers": updated})

    # No layer is exactly this selection, so the atoms are drawn by a broader
    # one (the load-time "protein" cartoon). Just stacking a coloured layer on
    # top left two cartoons over the same atoms and the old colour still
    # showed. Carve the selection out of every broader bulk layer and redraw
    # it, in the same style, with the new colour.
    exclude = f" and not ({ngl_sel})"
    added = {}
    for r in list(reps):
        if r["type"] not in bulk or r.get("_sel_name") or not r.get("visible", True):
            continue
        key = (r["type"], r.get("sid"))
        if key not in added:
            added[key] = {"type": r["type"], "selection": ngl_sel, "color": color,
                          "transparency": r.get("transparency", 0.0),
                          "sid": r.get("sid") or _scope_for(ngl_sel), "_colored": True}
        if exclude not in r["selection"]:
            r["selection"] = f"({r['selection']}){exclude}"
    if not added:
        added[("cartoon", None)] = {"type": "cartoon", "selection": ngl_sel,
                                    "color": color, "transparency": 0.0,
                                    "sid": _scope_for(ngl_sel), "_colored": True}
    reps.extend(added.values())
    return _ok(f"Colored '{selection}' as {color}",
               {"color": color, "ngl": ngl_sel, "layers": len(added)})


_REST_RE = re.compile(
    r"(?:the\s+)?(?:rest|others?|remaining|remainder|everything\s+else|"
    r"(?:all\s+)?(?:the\s+)?(?:other|remaining)\s+(?:chains?|parts?|residues?)|"
    r"rest\s+of\s+(?:the\s+)?(?:protein|chains?|structure|molecule))",
    re.IGNORECASE)


def tool_set_transparency(value: float, selection: str) -> ToolResult:
    """
    Set the transparency of all representation layers for a given selection.

    Args:
        value    : Transparency level — 0.0 is fully opaque, 1.0 fully transparent.
        selection: Named selection key or raw NGL expression.

    Returns:
        Status string with the count of updated layers.
    """
    ngl_sel = resolve_selection(selection)
    updated = 0
    for r in st.session_state.representations:
        if r["selection"] == ngl_sel:
            r["transparency"] = max(0.0, min(1.0, value))  # Clamp to [0, 1]
            updated += 1
    return _ok(f"Set transparency {value} on {updated} representation(s) for '{selection}'",
               {"value": value, "ngl": ngl_sel, "layers": updated})


# ── Geometric measurement ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_atoms(path: str, mtime: float, schema: int):
    """Read and cache a structure's coordinates; mtime and schema bust the cache."""
    return mz.read_atoms(path)


def _atoms_of(entry: dict, in_common_frame: bool = False):
    """
    Coordinates for one loaded structure.

    Args:
        entry          : Registry entry.
        in_common_frame: Apply the structure's superposition transform. Only
                         wanted when measuring against a *different* structure:
                         within one structure a rigid transform cancels out, so
                         applying it would just cost time.
    """
    atoms = _cached_atoms(entry["path"], Path(entry["path"]).stat().st_mtime,
                          mz.SCHEMA_VERSION)
    if in_common_frame and entry.get("matrix"):
        return mz.apply_matrix(atoms, entry["matrix"])
    return atoms


def _pick(spec_text: str, default_entry: dict, cross: bool):
    """
    Resolve a residue specification to (entry, atoms, label, indices, note).

    A specification may name its own structure ("1UBQ/A/12"), which is what
    makes measuring between two superposed structures work.

    Returns:
        (result dict, None) or (None, error message).
    """
    names = [s["pdb_id"] for s in structures()]
    probe = mz.parse_spec(spec_text, names, _atoms_of(default_entry)["resnames_present"])

    entry = find_structure(probe["structure"]) if probe["structure"] else default_entry
    if entry is None:
        return None, f"No structure called '{probe['structure']}' is loaded."

    atoms = _atoms_of(entry, in_common_frame=cross)
    spec = mz.parse_spec(spec_text, names, atoms["resnames_present"])
    groups, err = mz.resolve(spec, atoms)
    if err:
        return None, f"'{spec_text}' — {err} in {entry['pdb_id']}."

    label, indices = groups[0]
    note = ""
    if len(groups) > 1:
        others = ", ".join(l for l, _ in groups[1:4])
        # Worded so it cannot read as a second result: qwen2.5:7b reported
        # "a match for PO4 147 in chain D ... at 3.95 Å" from the old
        # "also matches X — measured against the first".
        note = (f"only {label} was used. Other copies of '{spec_text}', NOT searched and "
                f"with no results above: {others}"
                + ("…" if len(groups) > 4 else "")
                + " — add a chain to use one of them")
    return {"entry": entry, "atoms": atoms, "label": label,
            "indices": indices, "note": note, "groups": groups}, None


def _clean_mda_selection(selection) -> str:
    """Remove accidental list, bracket, or quote wrappers from a selection."""
    if isinstance(selection, (list, tuple)):
        if not selection:
            return ""
        selection = selection[0]

    selection = str(selection or "").strip()

    changed = True
    while changed and len(selection) >= 2:
        changed = False

        if selection[0] == "[" and selection[-1] == "]":
            selection = selection[1:-1].strip()
            changed = True
        elif selection[0] == "(" and selection[-1] == ")":
            selection = selection[1:-1].strip()
            changed = True
        elif selection[0] == selection[-1] and selection[0] in {"'", '"'}:
            selection = selection[1:-1].strip()
            changed = True

    return selection


def _selection_missing(selection) -> bool:
    cleaned = _clean_mda_selection(selection)
    return cleaned.lower() in {
        "",
        "none",
        "null",
        "unknown",
        "unspecified",
    }


def tool_measure_mda_distance(sel1: str, sel2: str) -> ToolResult:
    """Measure a deterministic distance using raw MDAnalysis selection syntax."""
    u = get_universe()
    if not u:
        return _fail("Load a protein structure before measuring a distance.")

    sel1 = _clean_mda_selection(sel1)
    sel2 = _clean_mda_selection(sel2)

    if _selection_missing(sel1) or _selection_missing(sel2):
        return _fail(
            "To calculate a distance, specify two atoms or atom selections. "
            "For example: measure the distance between the CA atoms of "
            "residues 50 and 100 in chain A."
        )

    try:
        n1, n2 = len(u.select_atoms(sel1)), len(u.select_atoms(sel2))
        text = measure_distance_from_universe(u, sel1, sel2)
        if not (n1 and n2):
            return _fail(text, {"atoms": [n1, n2]})
        d = math.dist(u.select_atoms(sel1).center_of_geometry(),
                      u.select_atoms(sel2).center_of_geometry())
        return _ok(text, {"sel1": sel1, "sel2": sel2, "atoms": [n1, n2],
                          "distance": round(d, 2)})
    except Exception as e:
        return _fail(f"Error measuring MDAnalysis distance: {e}")


MAX_MEASUREMENTS = 12


def _ngl_atom_selection(atoms, index) -> str:
    """
    NGL selection naming exactly one atom: "57:A.NE2".

    NGL's distance representation resolves each end of a pair to the first atom
    matching the string, so the string has to pin down residue, chain and atom
    or the line is drawn between the wrong places.
    """
    icode = atoms["icode"][index]
    chain = atoms["chain"][index]
    return (f"{atoms['resseq'][index]}{icode}"
            + (f":{chain}" if chain and chain != "_" else "")
            + f".{atoms['name'][index]}")


def _remember_measurement(first, second, d) -> None:
    """
    Record a measured distance so the viewer can draw it.

    Only distances within one structure are drawn: NGL's distance
    representation belongs to a single component, so a line between two
    separate structures has nowhere to live.
    """
    if first["entry"]["sid"] != second["entry"]["sid"]:
        return
    # Draw between the atoms whose separation is being reported. Drawing CA to
    # CA while labelling the line with the closest approach puts a number on a
    # line that does not measure it.
    ia, ib = d["closest_index"]
    entry = {
        "id": uuid.uuid4().hex[:8],
        "sid": first["entry"]["sid"],
        "a": _ngl_atom_selection(first["atoms"], ia),
        "b": _ngl_atom_selection(second["atoms"], ib),
        "a_index": int(ia),
        "b_index": int(ib),
        "a_label": first["label"],
        "b_label": second["label"],
        "value": d["ca"] if d["ca"] is not None else d["closest"],
        "closest": d["closest"],
    }
    existing = st.session_state.measurements
    if any(m["sid"] == entry["sid"] and {m["a"], m["b"]} == {entry["a"], entry["b"]}
           for m in existing):
        return
    st.session_state.measurements = (existing + [entry])[-MAX_MEASUREMENTS:]


def tool_measure_distance(a: str = "", b: str = "", **legacy) -> ToolResult:
    """
    Measure the distance between two residues, atoms or ligands.

    Reports three numbers, because they answer different questions: the CA–CA
    separation (how far apart the residues sit in the fold), the closest
    approach between any two atoms (whether they are touching), and the
    centre-to-centre distance. Two residues 9 Å apart at the CA can still be
    hydrogen bonded through their side chains.

    Either argument may name its own structure ("1UBQ/A/12"), which measures
    across a superposed pair in the shared coordinate frame.

    Args:
        a: First residue — "12", "A/12", "ARG12", "chain B residue 45",
           "12.CA", a ligand code like "BEN", or "1UBQ/A/12".
        b: Second residue, same syntax.

    Returns:
        A plain-text measurement, or an error message.
    """
    a = a or legacy.get("atom1_sel", "")
    b = b or legacy.get("atom2_sel", "")
    entry = active_structure()
    if not entry:
        return _fail(NO_STRUCTURE)

    names = [s["pdb_id"] for s in structures()]
    resn = _atoms_of(entry)["resnames_present"]
    cross = (mz.parse_spec(a, names, resn)["structure"]
             != mz.parse_spec(b, names, resn)["structure"])

    first, err = _pick(a, entry, cross)
    if err:
        return _fail(err)
    second, err = _pick(b, entry, cross)
    if err:
        return _fail(err)

    d = mz.distance(first["atoms"], first["indices"],
                    second["atoms"], second["indices"])

    def where(side):
        return (f"{side['label']} in {side['entry']['pdb_id']}"
                if len(structures()) > 1 else side["label"])

    lines = [f"{where(first)}  ↔  {where(second)}"]
    if d["ca"] is not None:
        lines.append(f"  CA–CA            {d['ca']:.2f} Å")
    lines.append(f"  closest atoms    {d['closest']:.2f} Å   "
                 f"({d['closest_atoms'][0]} → {d['closest_atoms'][1]})")
    lines.append(f"  centre–centre    {d['centre']:.2f} Å")
    if d["closest"] < 4.0:
        lines.append("  → they are in contact")
    for side in (first, second):
        if side["note"]:
            lines.append(f"  note: {side['note']}")
    if first["entry"]["sid"] != second["entry"]["sid"]:
        fitted = [s for s in (first["entry"], second["entry"]) if s.get("matrix")]
        lines.append("  measured across structures "
                     + ("with the superposition applied" if fitted
                        else "at their deposited coordinates — superpose them first "
                             "if you meant the fitted positions"))

    _remember_measurement(first, second, d)
    return _ok("\n".join(lines),
               {"a": first["label"], "b": second["label"],
                "ca": None if d["ca"] is None else round(d["ca"], 2),
                "closest": round(d["closest"], 2), "centre": round(d["centre"], 2),
                "closest_atoms": list(d["closest_atoms"]), "contact": d["closest"] < 4.0})


def _key_to_ngl(key) -> str:
    """NGL selection for one residue key: ('A', 57, '') → '57:A'."""
    chain, resseq, icode = key
    return f"{resseq}{icode}" + (f":{chain}" if chain and chain != "_" else "")


def keys_to_ngl(keys) -> str:
    """NGL selection matching any of a list of residues."""
    return " or ".join(_key_to_ngl(k) for k in keys) or "none"


MAX_INTERACTION_LINES = 300


def _store_interactions(entry, found) -> None:
    """Keep detected interactions so the viewer can draw them, colour-coded."""
    atoms = _atoms_of(entry)
    drawn = []
    for f in found[:MAX_INTERACTION_LINES]:
        drawn.append({
            "type": f["type"],
            "sid": entry["sid"],
            "a": _ngl_atom_selection(atoms, f["a_index"]),
            "b": _ngl_atom_selection(atoms, f["b_index"]),
            "a_index": f["a_index"],
            "b_index": f["b_index"],
            "distance": f["distance"],
        })
    st.session_state.interactions = drawn


def tool_find_interactions(target: str = "", types: str = "",
                           radius: float = 0.0, include_water: bool = False,
                           limit: int = 25) -> ToolResult:
    """
    Find non-covalent interactions: salt bridges, hydrogen bonds and the rest.

    Args:
        target       : Restrict to interactions involving this residue, ligand
                       or chain — "BEN", "A/57", "chain B". Empty scans the
                       whole structure.
        types        : Comma-separated subset of salt_bridge, hbond, disulfide,
                       pi_stacking, cation_pi, metal, hydrophobic. Empty means
                       everything except hydrophobic contacts, of which any
                       protein has thousands.
        radius       : Override the distance cutoff for the distance-based
                       types, in Ångstroms. 0 keeps the standard criteria.
        include_water: Keep interactions where one partner is a water.
        limit        : How many interactions to list per type.

    Returns:
        A plain-text report with the counts, the criteria used and the closest
        interactions of each type.
    """
    entry = active_structure()
    if not entry:
        return _fail(NO_STRUCTURE)
    atoms = _atoms_of(entry)

    wanted = [t.strip().lower().replace(" ", "_").replace("-", "_")
              for t in (types or "").split(",") if t.strip()]
    aliases = {"salt": "salt_bridge", "saltbridge": "salt_bridge",
               "hydrogen_bond": "hbond", "hydrogen": "hbond", "hbonds": "hbond",
               "ss": "disulfide", "disulphide": "disulfide",
               "stacking": "pi_stacking", "pi_pi": "pi_stacking",
               "pi": "pi_stacking", "cation": "cation_pi",
               "metals": "metal", "all": None}
    resolved = []
    for t in wanted:
        t = aliases.get(t, t)
        if t is None:
            resolved = list(ixn.TYPE_LABELS)
            break
        if t in ixn.TYPE_LABELS:
            resolved.append(t)
    kinds = resolved or None

    cutoffs = {}
    if radius and float(radius) > 0:
        r = max(1.5, min(float(radius), 12.0))
        cutoffs = {"salt_bridge": r, "hbond": r, "metal": r, "hydrophobic": r}

    restrict, scope_label = None, "the whole structure"
    target = (target or "").strip()
    # Whole-structure words are no scope at all: 'protein' failed in _pick
    # and the model reported the failure as "no salt bridges in 4HHB".
    if re.fullmatch(r"(the\s+)?(whole\s+|entire\s+)?(protein|structure|complex|all|"
                    r"everything|molecule|\*)", target, re.I):
        target = ""
    # A chain is a scope, not a residue — and it must be checked before
    # _pick, which read "chain A" as VAL 1 of chain A.
    chain_m = re.fullmatch(r"(?:chain\s*)?([A-Za-z0-9])", target, re.I)
    if chain_m and chain_m.group(1).upper() in atoms["chains_present"] and (
            target.lower().startswith("chain") or len(target) == 1):
        chain = chain_m.group(1).upper()
        restrict = {(c, r, i) for c, r, i in zip(
            atoms["chain"], atoms["resseq"], atoms["icode"]) if c == chain}
        scope_label = f"chain {chain}"
        target = ""
    keys, plabel = _pocket_target(target) if target else (None, None)
    if plabel and not keys:
        return _fail(plabel)
    if keys:
        restrict, scope_label, target = set(keys), plabel, ""
    if target:
        target, msg = _generic_ligand(target, "find the interactions of")
        if msg:
            return _fail(msg)
        side, err = _pick(target, entry, cross=False)
        if err:
            hint = _pocket_hint(target)
            return _fail(err + hint, hints=["find_pockets"] if hint else None)
        else:
            restrict = {(atoms["chain"][i], atoms["resseq"][i], atoms["icode"][i])
                        for i in side["indices"]}
            scope_label = side["label"]

    result = ixn.find(atoms, types=kinds, cutoffs=cutoffs,
                      restrict_to=restrict, include_water=bool(include_water))
    found = result["interactions"]
    _store_interactions(entry, found)

    try:
        limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit = 25

    # Each row is one atom pair; one salt bridge is often 2-4 of them. Said
    # plainly, or "63 salt bridge" here and detect_salt_bridges' "34 pairs"
    # read as two different answers.
    def _pairs(kind):
        return len({frozenset((f["a_label"], f["b_label"])) for f in found
                    if f["type"] == kind})

    def _count(kind, n):
        p = _pairs(kind)
        return f"{n} atom contacts, {p} residue pairs" if p != n else f"{n}"

    lines = [f"Interactions in {entry['pdb_id']} — {scope_label}"]
    if not found:
        lines.append("  none found with these criteria")
    else:
        summary = ", ".join(f"{ixn.TYPE_LABELS[k].lower()}: "
                            f"{_count(k, result['counts'][k])}"
                            for k, _, _ in ixn.INTERACTION_TYPES
                            if result["counts"].get(k))
        lines.append(f"  {summary}")

    for kind, label, _ in ixn.INTERACTION_TYPES:
        hits = [f for f in found if f["type"] == kind]
        if not hits:
            continue
        lines.append("")
        lines.append(f"{label.upper()} ({_count(kind, len(hits))})")
        for f in hits[:limit]:
            row = (f"  {f['a_label']:<14} {f['a_atom']:<4} — "
                   f"{f['b_label']:<14} {f['b_atom']:<4} {f['distance']:.2f} Å")
            if f["note"]:
                row += f"   [{f['note']}]"
            lines.append(row)
        if len(hits) > limit:
            lines.append(f"  …and {len(hits) - limit} more")

    lines.append("")
    lines.append("Criteria used:")
    for c in result["criteria"]:
        lines.append(f"  {c}")
    if found:
        lines.append("")
        lines.append(f"Drawn in the viewer as coloured dashed lines "
                     f"({min(len(found), MAX_INTERACTION_LINES)} of {len(found)}).")

    # Pairs by type: what the disulfide nudge and the fact check read.
    pairs = {}
    for f in found:
        pairs.setdefault(f["type"], set()).add(
            " — ".join(sorted((f["a_label"], f["b_label"]))))
    st.session_state.interaction_msg = _ok(
        "\n".join(lines),
        {"pdb_id": entry["pdb_id"], "scope": scope_label,
         "types": kinds or "default", "counts": dict(result["counts"]),
         "residue_pairs": {k: len(v) for k, v in pairs.items()},
         "pairs": {k: sorted(v)[:100] for k, v in pairs.items()},
         "disulfides": len(pairs.get("disulfide", ()))},
        hints=["highlight interactions"] if found else None)
    return st.session_state.interaction_msg


def tool_highlight(target: str, style: str = "ball+stick",
                   color: str = "yellow", zoom: bool = True) -> ToolResult:
    """
    Highlight one or more residues, ligands or regions in the viewer.

    Args:
        target: What to highlight — a residue ("A/57"), a ligand ("BEN"), a
                comma-separated list ("57, 102, 195"), a range ("57-102"), or
                "interactions" to highlight everything from the last scan.
        style : NGL representation type; ball+stick reads best for a few
                residues, surface or cartoon for a region.
        color : Colour name or scheme.
        zoom  : Move the camera to the highlighted region.

    Returns:
        Confirmation naming what was highlighted, or an error message.
    """
    entry = active_structure()
    if not entry:
        return _fail(NO_STRUCTURE)
    atoms = _atoms_of(entry)

    keys, labels = [], []
    spec = (target or "").strip()

    if spec.lower() in ("interactions", "interaction", "contacts", "last"):
        if not st.session_state.interactions:
            return _fail("No interactions have been found yet — run find_interactions first.",
                         hints=["find_interactions"])
        sels = []
        for m in st.session_state.interactions:
            for end in (m["a"], m["b"]):
                residue = end.split(".")[0]
                if residue not in sels:
                    sels.append(residue)
        ngl = " or ".join(sels)
        labels = [f"{len(sels)} residues from the last interaction scan"]
    elif _pocket_target(spec) != (None, None):
        pocket_keys, label = _pocket_target(spec)
        if pocket_keys is None:
            return _fail(label)
        ngl, labels = keys_to_ngl(pocket_keys), [label]
    else:
        for piece in [p for p in spec.split(",") if p.strip()]:
            piece = piece.strip()
            # Spelled-out ranges the model writes: "A/20 to A/30",
            # "20 to 30 in chain A", "residues 20-30 of chain A".
            piece = re.sub(r"^(?:residues?\s+)", "", piece, flags=re.I)
            cm = re.search(r"\s+(?:in|of|on)?\s*chain\s+([A-Za-z0-9])$", piece, re.I)
            if cm:
                piece = f"{cm.group(1)}/{piece[:cm.start()].strip()}"
            piece = re.sub(r"(\d+)\s*(?:to|through|–|-)\s*(?:[A-Za-z0-9][/:])?(\d+)$",
                           r"\1-\2", piece)
            # A range like "57-102" is a region, not two residues.
            m = re.fullmatch(r"(?:([A-Za-z0-9])[/:])?(\d+)\s*[-–]\s*(\d+)", piece)
            if m:
                chain, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
                for i in range(len(atoms["resseq"])):
                    if lo <= atoms["resseq"][i] <= hi and (
                            not chain or atoms["chain"][i].upper() == chain.upper()):
                        key = (atoms["chain"][i], atoms["resseq"][i], atoms["icode"][i])
                        if key not in keys:
                            keys.append(key)
                labels.append(f"{piece} ({len([k for k in keys])} residues)")
                continue
            side, err = _pick(piece, entry, cross=False)
            if err:
                return _fail(err)
            for i in side["indices"]:
                key = (atoms["chain"][i], atoms["resseq"][i], atoms["icode"][i])
                if key not in keys:
                    keys.append(key)
            labels.append(side["label"])
        if not keys:
            return _fail("Nothing to highlight — name a residue, ligand or range.")
        ngl = keys_to_ngl(keys)

    if style not in NGL_REP_TYPES:
        style = "ball+stick"
    color = _normalise_color(color)

    name = "highlight"
    st.session_state.selections[name] = ngl
    st.session_state.representations = [
        r for r in st.session_state.representations if r.get("_sel_name") != name
    ]
    st.session_state.representations.append({
        "id": uuid.uuid4().hex[:8], "type": style, "selection": ngl,
        "color": color, "transparency": 0.0, "visible": True,
        "sid": entry["sid"], "_sel_name": name,
    })
    if zoom:
        st.session_state.camera_target = ngl
    return _ok(f"Highlighted {'; '.join(labels)} in {entry['pdb_id']} as {style} "
               f"({color}). Saved as the selection 'highlight'.",
               {"pdb_id": entry["pdb_id"], "targets": labels, "ngl": ngl,
                "style": style, "color": color, "zoomed": bool(zoom)})


def tool_measure_angle(a: str, b: str, c: str) -> ToolResult:
    """
    Measure the angle at residue `b` between residues `a` and `c`.

    Uses each residue's representative atom (CA for protein, C4' or P for
    nucleic acids), or the named atom when the specification gives one.

    Args:
        a, b, c: Residue specifications; the angle is measured at `b`.

    Returns:
        The angle in degrees, or an error message.
    """
    entry = active_structure()
    if not entry:
        return _fail(NO_STRUCTURE)

    points, labels = [], []
    for spec_text in (a, b, c):
        side, err = _pick(spec_text, entry, cross=True)
        if err:
            return _fail(err)
        idx = mz._representative(side["atoms"], side["indices"]) or side["indices"][0]
        points.append(side["atoms"]["xyz"][idx])
        labels.append(side["label"])

    value = mz.angle(*points)
    if value is None:
        return _fail("Two of those points coincide, so there is no angle to measure.")
    return _ok(f"Angle at {labels[1]} between {labels[0]} and {labels[2]}: "
               f"{value:.1f}°", {"points": labels, "angle": round(value, 1)})


def tool_measure_dihedral(a: str, b: str, c: str, d: str) -> ToolResult:
    """
    Measure the torsion angle about the b–c axis, in degrees.

    Uses each residue's representative atom unless the specification names one,
    so a backbone torsion is written out atom by atom — phi for residue 57 is
    `56.C`, `57.N`, `57.CA`, `57.C`.

    Args:
        a, b, c, d: Four residue or atom specifications, in order along the
                    torsion.

    Returns:
        The signed torsion in degrees, or an error message.
    """
    entry = active_structure()
    if not entry:
        return _fail(NO_STRUCTURE)

    points, labels = [], []
    for spec_text in (a, b, c, d):
        side, err = _pick(spec_text, entry, cross=True)
        if err:
            return _fail(err)
        idx = mz._representative(side["atoms"], side["indices"]) or side["indices"][0]
        points.append(side["atoms"]["xyz"][idx])
        labels.append(f"{side['label']}·{side['atoms']['name'][idx]}")

    value = mz.dihedral(*points)
    if value is None:
        return _fail("Those four points are collinear, so there is no torsion to measure.")
    return _ok(f"Dihedral {labels[0]} → {labels[1]} → {labels[2]} → {labels[3]}: "
               f"{value:.1f}°", {"points": labels, "dihedral": round(value, 1)})


def tool_find_contacts(target: str, radius: float = 4.0,
                       include_water: bool = False) -> ToolResult:
    """
    List every residue within a cutoff of a residue or ligand.

    This is the tool for "what does the ligand bind to" and "what surrounds
    residue 57". Distances are the closest approach between any two atoms, not
    centre-to-centre, since that is what decides whether two residues interact.

    Args:
        target       : Residue or ligand — "BEN", "A/57", "HIS57", "03Q".
        radius       : Cutoff in Ångstroms; 4.0 catches direct contacts,
                       5–6 the wider shell.
        include_water: Include water molecules in the shell.

    Returns:
        A ranked plain-text list, or an error message.
    """
    entry = active_structure()
    if not entry:
        return _fail(NO_STRUCTURE)
    # An empty target here is "the ligand" with the noun dropped: qwen2.5:7b
    # sent target='' for "what does the ligand bind to" after a find_pockets
    # turn, got "no residue was named", and told the user 4HHB has no ligand.
    target = (target or "").strip()
    # "pocket N" (after find_pockets): contacts of whatever the file has bound
    # there, or the lining itself for an empty pocket. Failing on it sent the
    # model back to find_pockets, whose report it then garbled.
    keys, plabel = _pocket_target(target) if target else (None, None)
    if plabel and not keys:
        return _fail(plabel)
    if keys:
        m = re.search(r"(\d+)", target)
        p = st.session_state.pockets["by_rank"][int(m.group(1))]
        if not p.get("occupants"):
            # An answer, not a failure: the lining is what surrounds an empty pocket.
            return _ok(f"Nothing is bound in {plabel.split(' (')[0]} of {entry['pdb_id']} — "
                       f"it is empty in this file. Its lining residues: {_pocket_residues(p)}.",
                       {"pocket": p["rank"], "occupants": [],
                        "lining": [f"{r[3]}{r[1]}{r[2]}:{r[0]}" for r in p["residues"]]})
        occ = p["occupants"][0]
        target = occ["label"].split(" ", 1)[1]           # "HEM B/148" -> "B/148"
    target, msg = _generic_ligand(target or "ligand", "list the residues in contact with")
    if msg:
        return _fail(msg)
    side, err = _pick(target, entry, cross=False)
    if err:
        hint = _pocket_hint(target)
        return _fail(err + hint, hints=["find_pockets"] if hint else None)

    try:
        radius = float(radius)
    except (TypeError, ValueError):
        radius = 4.0
    radius = max(1.0, min(radius, 15.0))

    # A ligand code with several copies (4HHB: HEM in all four chains) gets
    # every copy searched: reporting only the first led the model to invent
    # results for the others.
    groups = side["groups"]
    code = groups[0][0].split()[0].upper()
    data = {"pdb_id": entry["pdb_id"], "target": side["label"], "radius": radius}
    if (len(groups) > 1 and target.strip().upper() == code
            and code not in srep.STANDARD_AA):
        per = 15
        data["copies"] = {}
        lines = [f"'{target}' has {len(groups)} copies in {entry['pdb_id']}; "
                 f"residues within {radius:g} Å of each:"]
        for label, indices in groups[:6]:
            found = mz.contacts(side["atoms"], indices, radius,
                                include_water=bool(include_water))
            data["copies"][label] = {c["label"]: round(c["distance"], 2) for c in found}
            lines.append(f"{label} ({len(found)} found)" + (":" if found else
                         f" — nothing within {radius:g} Å."))
            for c in found[:per]:
                lines.append(f"  {c['label']:<24} {c['distance']:.2f} Å   "
                             f"({c['atoms'][0]} → {c['atoms'][1]})")
            if len(found) > per:
                lines.append(f"  …and {len(found) - per} more")
        if len(groups) > 6:
            lines.append(f"({len(groups) - 6} more copies not listed — add a chain to pick one.)")
        return _ok("\n".join(lines), data)

    found = mz.contacts(side["atoms"], side["indices"], radius,
                        include_water=bool(include_water))
    data["contacts"] = {c["label"]: round(c["distance"], 2) for c in found}
    if not found:
        # A real empty shell, not a failed search.
        return _ok(f"Nothing is within {radius:g} Å of {side['label']} in "
                   f"{entry['pdb_id']}.", data)

    lines = [f"Residues within {radius:g} Å of {side['label']} in {entry['pdb_id']} "
             f"({len(found)} found):"]
    for c in found[:40]:
        lines.append(f"  {c['label']:<24} {c['distance']:.2f} Å   "
                     f"({c['atoms'][0]} → {c['atoms'][1]})")
    if len(found) > 40:
        lines.append(f"  …and {len(found) - 40} more")
    if side["note"]:
        lines.append(f"  note: {side['note']}")
    return _ok("\n".join(lines), data)


def _ensure_loaded(target: str):
    """
    Resolve a structure name for superposition, fetching it if it is not loaded.

    Superposing is the one operation people reach for with a structure they
    have not opened yet ("superpose 1UBI onto 1UBQ"), so an accession that is
    not in the scene is downloaded and added rather than rejected.

    Returns:
        (entry, None) or (None, error message).
    """
    s = find_structure(target)
    if s:
        return s, None
    name = (target or "").strip()
    if len(name) != 4:
        return None, (f"No structure called '{target}' is loaded, and it is not "
                      f"a 4-character PDB accession that could be fetched.")
    dest, err = download_pdb(name)
    if err:
        return None, err
    _recolor_for_comparison()
    return register_structure(name, dest, source="rcsb", make_active=False), None


def tool_superpose(mobile: str, reference: str, method: str = "auto",
                   mobile_selection: str = "protein",
                   reference_selection: str = "protein") -> ToolResult:
    """
    Superimpose one loaded structure onto another so they share a coordinate frame.

    The mobile structure's file is not modified. The fit is stored as a 4x4
    transform on its registry entry and applied by the viewer, so it can be
    undone, recomputed against a different reference, or restricted to a chain
    without ever touching the deposited coordinates.

    Args:
        mobile             : Structure to move (PDB id, label or sid).
        reference          : Structure to keep fixed.
        method             : "auto" pairs residues by number and falls back to
                             sequence alignment when the numbering disagrees;
                             "resnum" and "sequence" force one or the other.
        mobile_selection   : MDAnalysis selection limiting the mobile fit atoms,
                             e.g. "protein and segid A".
        reference_selection: The same for the reference structure.

    Returns:
        A line giving the RMSD, how many residues were matched and how they
        were paired, or an error message.
    """
    if not sup.available():
        return _fail("MDAnalysis is unavailable, so structures cannot be superposed.",
                     {"unavailable": "mdanalysis"})

    mob, err = _ensure_loaded(mobile)
    if err:
        return _fail(err)
    ref, err = _ensure_loaded(reference)
    if err:
        return _fail(err)
    if mob["sid"] == ref["sid"]:
        return _fail("The mobile and reference structures must be different.")

    result = sup.superpose(
        mob["path"], ref["path"],
        mobile_sel=mobile_selection or "protein",
        reference_sel=reference_selection or "protein",
        method=method or "auto",
    )
    if not result["ok"]:
        st.session_state.superpose_msg = result["message"]
        return _fail(result["message"])

    mob["matrix"] = result["matrix"]
    mob["fit"] = result["message"]
    mob["fit_reference"] = ref["pdb_id"]
    mob["rmsd"] = result["rmsd"]
    st.session_state.camera_target = None      # refit the camera on both
    msg = _ok(f"Superposed {mob['pdb_id']} onto {ref['pdb_id']} — {result['message']}.",
              {"mobile": mob["pdb_id"], "reference": ref["pdb_id"],
               "rmsd": None if result["rmsd"] is None else round(result["rmsd"], 2),
               "matched": result.get("n_atoms"), "identity": result.get("identity"),
               "method": result.get("method")})
    st.session_state.superpose_msg = msg
    return msg


def tool_clear_superposition(target: str = "all") -> ToolResult:
    """
    Put superposed structures back at their deposited coordinates.

    Args:
        target: A structure name, or "all" to reset every one of them.

    Returns:
        Confirmation string.
    """
    if (target or "all").strip().lower() == "all":
        hits = [s for s in structures() if s["matrix"]]
    else:
        s = find_structure(target)
        if not s:
            return _fail(f"No structure called '{target}' is loaded. Loaded: "
                     + (", ".join(x["pdb_id"] for x in structures()) or "none") + ".")
        hits = [s] if s["matrix"] else []

    if not hits:
        return _ok("No structure is currently superposed.", {"reset": []})
    for s in hits:
        s["matrix"] = None
        s["fit"] = None
        s.pop("fit_reference", None)
        s.pop("rmsd", None)
    st.session_state.superpose_msg = None
    return _ok("Reset " + ", ".join(s["pdb_id"] for s in hits) +
               " to the deposited coordinates.", {"reset": [s["pdb_id"] for s in hits]})


def tool_zoom(selection: str) -> ToolResult:
    """
    Focus the NGL camera on a selection after the next page render.

    Stores the target NGL selection in session state; the HTML renderer picks
    it up and calls comp.setSelection() before autoView().

    Args:
        selection: Named selection key or raw NGL expression to zoom into.

    Returns:
        Confirmation string.
    """
    ngl_sel = resolve_selection(selection)
    st.session_state.camera_target = ngl_sel
    return _ok(f"Camera focused on '{selection}'", {"ngl": ngl_sel})


def tool_set_background(color: str) -> ToolResult:
    """
    Set the NGL viewer background color.

    Accepted values: "black", "white", "grey"/"gray". Unknown values fall
    back to "black".

    Args:
        color: Desired background color string.

    Returns:
        Confirmation string with the applied color.
    """
    valid = {"black", "white", "grey", "gray"}
    c = color.lower()
    if c not in valid:
        c = "black"
    st.session_state.background = c
    return _ok(f"Background set to {c}", {"background": c})


def tool_save_structure(filename: str) -> ToolResult:
    """
    Copy the currently loaded PDB file to a new filename in STRUCTURES_DIR.

    Args:
        filename: Target filename (e.g. "my_protein.pdb").

    Returns:
        Confirmation string with the output path, or an error message.
    """
    path = st.session_state.pdb_path
    if not path or not Path(path).exists():
        return _fail("No structure loaded")
    out = STRUCTURES_DIR / filename
    shutil.copy2(path, out)
    return _ok(f"Saved to {out}", files=[out])


def tool_remove_solvent() -> ToolResult:
    """
    Strip all water molecules from the current structure and save the result.

    Uses MDAnalysis to select non-water atoms, writes them to
    <pdb_id>_no_solvent.pdb, and repoints the active structure at the new file.

    The registry entry is switched to source="local" as well. A structure
    fetched from RCSB is streamed into the viewer by accession, so without that
    switch the viewer would keep drawing the deposited file, waters and all,
    while every server-side analysis saw the de-solvated one.

    Returns:
        Confirmation with the output filename, or an error/unavailability message.
    """
    u = get_universe()
    if not u:
        return _fail("MDAnalysis unavailable", {"unavailable": "mdanalysis"})
    entry = active_structure()
    if not entry:
        return _fail("No structure loaded")
    try:
        waters = len(u.select_atoms("water").residues)
        no_water = u.select_atoms("not water")
        out_path = STRUCTURES_DIR / f"{st.session_state.pdb_id}_no_solvent.pdb"
        no_water.write(str(out_path))
        entry["path"] = str(out_path)
        entry["source"] = "local"
        st.session_state.pdb_path = str(out_path)
        st.session_state.universe = None   # Force reload from the new de-solvated file
        return _ok(f"Removed solvent. Saved to {out_path.name}",
                   {"pdb_id": entry["pdb_id"], "waters_removed": waters}, files=[out_path])
    except Exception as e:
        return _fail(f"Error: {e}")


def tool_add_hydrogens() -> ToolResult:
    """
    Placeholder for hydrogen addition (requires OpenBabel or RDKit).

    Returns an informative message instead of silently failing.
    """
    return _fail("Hydrogen addition requires OpenBabel or RDKit (not installed). "
                 "Install openbabel-python to enable.", {"unavailable": "openbabel"},
                 hints=["prepare_structure with add_hydrogens"])

def _format_chain_summary(df) -> str:
    """Convert chain-summary output into readable scientific prose."""
    if df is None or df.empty:
        return "No chain or segment information was found."

    if "error" in df.columns:
        return str(df.iloc[0]["error"])

    rows = df.to_dict(orient="records")

    # Protein residues usually contain multiple atoms per residue. Groups with
    # approximately one atom per residue are commonly waters or ions.
    protein_rows = []
    auxiliary_rows = []

    for row in rows:
        atoms = int(row.get("n_atoms", 0))
        residues = int(row.get("n_residues", 0))
        atoms_per_residue = atoms / residues if residues else 0

        if residues >= 10 and atoms_per_residue >= 2:
            protein_rows.append(row)
        else:
            auxiliary_rows.append(row)

    if not protein_rows:
        protein_rows = rows
        auxiliary_rows = []

    descriptions = []
    for row in protein_rows:
        chain = row.get("chain_or_segment") or "unlabeled"
        descriptions.append(
            f"chain {chain} contains "
            f"{int(row.get('n_residues', 0)):,} residues and "
            f"{int(row.get('n_atoms', 0)):,} atoms"
        )

    count = len(protein_rows)
    label = "primary protein chain" if count == 1 else "primary protein chains"

    response = (
        f"The loaded structure contains {count} {label}: "
        + "; ".join(descriptions)
        + "."
    )

    if auxiliary_rows:
        response += (
            f" It also contains {len(auxiliary_rows)} smaller segment"
            f"{'s' if len(auxiliary_rows) != 1 else ''}, which may represent "
            "ligands, ions, solvent, or other non-protein components."
        )

    return response


def _salt_bridge_pairs(df):
    """(within one chain, between chains) residue pairs of a salt-bridge table."""
    def _chain(label):
        return str(label).rsplit(":", 1)[-1] if ":" in str(label) else ""

    unique_pairs = (df[["acidic_residue", "basic_residue"]]
                    .drop_duplicates().reset_index(drop=True))
    pairs = [(r.acidic_residue, r.basic_residue)
             for r in unique_pairs.itertuples(index=False)]
    return ([p for p in pairs if _chain(p[0]) == _chain(p[1])],
            [p for p in pairs if _chain(p[0]) != _chain(p[1])])


def _df_failed(df) -> bool:
    """An analysis_tools table that reports an error instead of rows."""
    return df is not None and "error" in df.columns


def _df_rows(df) -> int:
    """Data rows in an analysis_tools table (its 'result' row means none)."""
    if df is None or df.empty or "error" in df.columns or "result" in df.columns:
        return 0
    return len(df)


def _format_salt_bridge_summary(df, cutoff: float, max_rows: int = 100) -> str:
    """Summarize geometric acidic-basic contacts."""
    if df is None or df.empty:
        return (
            f"No candidate acidic–basic contacts were found within "
            f"the {cutoff:.1f} Å cutoff."
        )

    if "error" in df.columns:
        return str(df.iloc[0]["error"])

    if "result" in df.columns:
        return str(df.iloc[0]["result"])

    required = {"acidic_residue", "basic_residue"}
    if not required.issubset(df.columns):
        return df.to_string(index=False)

    # Every pair is listed, split by whether it joins two chains: with only
    # three "representative" pairs, qwen2.5:7b said "four" while listing
    # three, and called same-chain pairs inter-chain.
    intra, inter = _salt_bridge_pairs(df)
    pairs = intra + inter

    def _list(ps):
        shown = ", ".join(f"{a}–{b}" for a, b in ps[:40])
        return shown + (f", …and {len(ps) - 40} more" if len(ps) > 40 else "")

    response = (
        f"SUMMARY: {len(pairs)} candidate salt-bridge residue pairs "
        f"({len(df)} acidic–basic atom contacts within {cutoff:.1f} Å): "
        f"{len(intra)} within one chain, {len(inter)} between two chains."
    )
    if intra:
        response += f"\nWithin one chain ({len(intra)}): {_list(intra)}."
    if inter:
        response += f"\nBetween chains ({len(inter)}): {_list(inter)}."
    response += (
        "\nThese are geometry-based candidates rather than confirmed stable "
        "salt bridges. The atom-contact table is shown below"
        + (f" (first {max_rows} of {len(df)} rows)." if len(df) > max_rows else ".")
    )

    return response + "\n\n" + df.head(max_rows).to_string(index=False)


def _format_nearby_residue_summary(df, selection: str, cutoff: float) -> str:
    """Summarize residues located near a target selection."""
    if df is None or df.empty:
        return f"No residues were found within {cutoff:g} Å of {selection}."

    if "error" in df.columns:
        return str(df.iloc[0]["error"])

    residues = []
    for row in df.to_dict(orient="records"):
        label = (
            f"{row.get('resname', '')}{row.get('resid', '')}:"
            f"{row.get('chain_or_segment', '')}"
        ).rstrip(":")
        if label and label not in residues:
            residues.append(label)

    preview = ", ".join(residues[:8])

    response = (
        f"PARORA found {len(residues)} residue"
        f"{'s' if len(residues) != 1 else ''} within "
        f"{cutoff:g} Å of {selection}."
    )

    if preview:
        response += f" Nearby residues include {preview}"
        if len(residues) > 8:
            response += f", and {len(residues) - 8} additional residues"
        response += "."

    response += " These residues have been highlighted in the viewer."

    return response + "\n\n" + df.to_string(index=False)


def tool_summarize_chains() -> ToolResult:
    """Summarize chains/segments in the currently loaded structure."""
    # Chain identities come from the file's own COMPND/DBREF records, so this
    # names each chain's molecule even when MDAnalysis is unavailable.
    names = chain_map_line(active_structure(),
                           (st.session_state.focus or {}).get("accession", ""))
    entry = active_structure()
    data = {"pdb_id": entry["pdb_id"] if entry else None,
            "chains": {c["chain"]: c["molecule"] for c in chain_molecules(entry)} if entry else {}}
    u = get_universe()
    if not u:
        return (_ok(names, data) if names else
                _fail("Load a protein structure before requesting a chain summary."))

    try:
        df = summarize_chains_from_universe(u)
        if _df_rows(df):
            data["segments"] = {str(r["chain_or_segment"]): int(r["n_residues"])
                                for r in df.to_dict(orient="records")}
        return _ok((names + "\n\n" if names else "") + _format_chain_summary(df), data)
    except Exception as e:
        return _ok(names, data) if names else _fail(f"Error summarizing chains: {e}")


def tool_list_residues(
    chain: str = "",
    max_rows: int = 200
) -> ToolResult:
    """List residues, optionally restricted to one chain or segment."""
    u = get_universe()
    if not u:
        return _fail("Load a protein structure before listing residues.")

    chain = str(chain or "").strip()
    max_rows = _safe_int(max_rows, 200)

    try:
        df = list_residues_from_universe(
            u,
            chain=chain or None,
            max_rows=max_rows,
        )

        if "error" in df.columns:
            return _fail(str(df.iloc[0]["error"]))

        prefix = f"Chain {chain} residues:\n" if chain else ""
        return _ok(prefix + df.to_string(index=False),
                   {"chain": chain or None, "listed": len(df),
                    "residues": [f"{r['resname']}{r['resid']}:{r['chain_or_segment']}"
                                 for r in df.to_dict(orient="records")]})

    except Exception as e:
        return _fail(f"Error listing residues: {e}")


def tool_bfactor_summary() -> ToolResult:
    """Summarize B-factor/tempfactor values for the loaded structure."""
    u = get_universe()
    if not u:
        return _fail("MDAnalysis unavailable — cannot summarize B-factors",
                     {"unavailable": "mdanalysis"})

    try:
        df = bfactor_summary_from_universe(u)
        if _df_failed(df):
            return _fail(str(df.iloc[0]["error"]))
        row = df.iloc[0]
        return _ok(df.to_string(index=False),
                   {k: round(float(row[k]), 2) for k in ("mean_bfactor", "min_bfactor",
                                                         "max_bfactor") if k in df.columns})
    except Exception as e:
        return _fail(f"Error summarizing B-factors: {e}")


def tool_detect_contacts(
    sel1: str = "protein",
    sel2: str = "protein",
    cutoff: float = 4.0,
    max_rows: int = 100
) -> ToolResult:
    """Detect residue-level contacts between two selections."""
    u = get_universe()
    if not u:
        return _fail("MDAnalysis unavailable — cannot detect contacts",
                     {"unavailable": "mdanalysis"})

    try:
        df = contact_detection_from_universe(
            u,
            sel1=sel1,
            sel2=sel2,
            cutoff=cutoff,
            max_rows=max_rows
        )
        if _df_failed(df):
            return _fail(str(df.iloc[0]["error"]))
        return _ok(df.to_string(index=False),
                   {"sel1": sel1, "sel2": sel2, "cutoff": cutoff, "contacts": _df_rows(df),
                    "pairs": ([f"{r['residue_1']} — {r['residue_2']} {r['min_distance_A']}"
                               for r in df.to_dict(orient="records")] if _df_rows(df) else [])})
    except Exception as e:
        return _fail(f"Error detecting contacts: {e}")


def tool_detect_salt_bridges(
    chain: str = "",
    cutoff: float = 4.0,
    max_rows: int = 100
) -> ToolResult:
    """
    Detect and summarize candidate salt bridges.

    When a chain is provided, both residues in every returned pair must
    belong to that chain.
    """
    u = get_universe()
    if not u:
        return _fail("Load a protein structure before identifying salt bridges.")

    chain = str(chain).strip()
    cutoff = _safe_float(cutoff, 4.0)
    max_rows = _safe_int(max_rows, 100)

    try:
        df = salt_bridge_detection_from_universe(
            u,
            chain=chain or None,
            cutoff=cutoff,
            # Count every contact; max_rows only trims the printed table. The
            # detector stops at its row limit, which made pair counts wrong.
            max_rows=10**6,
        )

        summary = _format_salt_bridge_summary(df, cutoff, max_rows)
        if _df_failed(df):
            return _fail(summary)
        if chain:
            summary = f"Chain {chain} analysis: {summary}"
        # run_agent attaches this count and the short pair lists to the reply
        # (the model miscounted pairs and mislabelled same-chain ones).
        data = {"chain": chain or None, "cutoff": cutoff, "atom_contacts": _df_rows(df),
                "pairs": 0, "within": [], "between": []}
        if _df_rows(df) and {"acidic_residue", "basic_residue"} <= set(df.columns):
            intra, inter = _salt_bridge_pairs(df)
            data.update(pairs=len(intra) + len(inter),
                        within=[f"{a}–{b}" for a, b in intra],
                        between=[f"{a}–{b}" for a, b in inter],
                        summary=f"{len(intra) + len(inter)} candidate salt-bridge residue "
                                f"pairs ({_df_rows(df)} acidic–basic atom contacts within "
                                f"{cutoff:.1f} Å): {len(intra)} within one chain, "
                                f"{len(inter)} between two chains.")
        return _ok(summary, data)
    except Exception as e:
        return _fail(f"Error detecting salt bridges: {e}")

def tool_detect_hydrogen_bonds(
    cutoff: float = 3.5,
    max_rows: int = 100
) -> ToolResult:
    """Detect candidate hydrogen bonds using a distance-only donor/acceptor screen."""
    u = get_universe()
    if not u:
        return _fail("MDAnalysis unavailable — cannot detect hydrogen bonds",
                     {"unavailable": "mdanalysis"})

    try:
        df = hydrogen_bond_detection_from_universe(
            u,
            cutoff=cutoff,
            max_rows=max_rows
        )
        if _df_failed(df):
            return _fail(str(df.iloc[0]["error"]))
        return _ok(df.to_string(index=False),
                   {"cutoff": cutoff, "hbonds": _df_rows(df),
                    "truncated": _df_rows(df) >= max_rows})
    except Exception as e:
        return _fail(f"Error detecting hydrogen bonds: {e}")

def tool_nearby_residues(
    selection: str,
    cutoff: float = 5.0,
    max_rows: int = 100
) -> ToolResult:
    """Find and summarize residues near an MDAnalysis selection."""
    u = get_universe()
    if not u:
        return _fail("Load a protein structure before finding nearby residues.")

    selection = _clean_mda_selection(selection)
    cutoff = _safe_float(cutoff, 5.0)

    if _selection_missing(selection):
        return _fail(
            "To find nearby residues, specify a ligand, residue, atom, or "
            "MDAnalysis selection. For example: find residues within 5 Å "
            "of resname DCK."
        )

    try:
        df = nearby_residues_from_universe(
            u,
            target_selection=selection,
            radius=cutoff,
        )

        if max_rows and len(df) > _safe_int(max_rows, 100):
            df = df.head(_safe_int(max_rows, 100))

        text = _format_nearby_residue_summary(df, selection, cutoff)
        if _df_failed(df):
            return _fail(text)
        residues = sorted({f"{r.get('resname', '')}{r.get('resid', '')}:"
                           f"{r.get('chain_or_segment', '')}"
                           for r in df.to_dict(orient="records")}) if _df_rows(df) else []
        return _ok(text, {"selection": selection, "cutoff": cutoff, "residues": residues})
    except Exception as e:
        # The model passes residue specs here ("A/HIS87", "A/87") that are not
        # MDAnalysis syntax; find_contacts' parser reads those, so answer with it
        # rather than fail twice and leave the model to make residues up.
        if active_structure() and re.search(r"\d", selection) and not re.search(
                r"\b(resid|resname|segid|name|and|or|around)\b", selection):
            alt = tool_find_contacts(selection, radius=cutoff)
            if alt.ok:
                return alt
        return _fail(f"Error finding nearby residues: {e}")


def _mda_measurement(text: str, kind: str, selections: list) -> ToolResult:
    """
    Wrap analysis_tools' angle/dihedral line ("... = 109.47° using atom
    counts ..." or "<Kind> failed: ...") as a result with the value in data.
    """
    text = str(text)
    m = re.search(r"=\s*(-?\d+(?:\.\d+)?)°", text)
    if text.lower().startswith(f"{kind} failed") or not m:
        return _fail(text)
    return _ok(text, {"selections": selections, kind: float(m.group(1))})


def tool_measure_mda_angle(sel1: str, sel2: str, sel3: str) -> ToolResult:
    """Measure an angle between three MDAnalysis atom selections."""
    u = get_universe()
    if not u:
        return _fail("Load a protein structure before measuring an angle.")

    selections = [
        _clean_mda_selection(sel1),
        _clean_mda_selection(sel2),
        _clean_mda_selection(sel3),
    ]

    if any(_selection_missing(sel) for sel in selections):
        return _fail(
            "To calculate an angle, specify three atoms or atom selections. "
            "For example: calculate the angle between the CA atoms of "
            "residues 50, 51, and 52 in chain A."
        )

    generic = {"protein", "ligand", "nonstandard", "non-standard", "chain", "chains"}
    if any(sel.lower() in generic for sel in selections):
        return _fail(
            "An angle requires three specific atom selections rather than a "
            "whole protein, chain, or ligand."
        )

    try:
        result = measure_angle_from_universe(
            u,
            sel1=selections[0],
            sel2=selections[1],
            sel3=selections[2],
        )
        return _mda_measurement(result, "angle", selections)
    except Exception as e:
        return _fail(f"Error measuring angle: {e}")


def tool_measure_mda_dihedral(sel1: str, sel2: str, sel3: str, sel4: str) -> ToolResult:
    """Measure a dihedral angle between four MDAnalysis atom selections."""
    u = get_universe()
    if not u:
        return _fail("Load a protein structure before measuring a dihedral.")

    selections = [
        _clean_mda_selection(sel1),
        _clean_mda_selection(sel2),
        _clean_mda_selection(sel3),
        _clean_mda_selection(sel4),
    ]

    if any(_selection_missing(sel) for sel in selections):
        return _fail(
            "To calculate a dihedral, specify four atoms or atom selections. "
            "For example: use the N, CA, C, and N atoms across two adjacent "
            "residues in chain A."
        )

    generic = {"protein", "ligand", "nonstandard", "non-standard", "chain", "chains"}
    if any(sel.lower() in generic for sel in selections):
        return _fail(
            "A dihedral requires four specific atom selections rather than a "
            "whole protein, chain, or ligand."
        )

    try:
        result = measure_dihedral_from_universe(
            u,
            sel1=selections[0],
            sel2=selections[1],
            sel3=selections[2],
            sel4=selections[3],
        )
        return _mda_measurement(result, "dihedral", selections)
    except Exception as e:
        return _fail(f"Error measuring dihedral: {e}")


# ── Annotations: text labels pinned to a region ──────────────────────────────
# A figure of a membrane protein is much easier to read with "TM" written
# across the bundle and "extracellular" above it than with a colour key alone.
# NGL can already draw text — it is how a measured distance gets its number —
# so an annotation is just a string, a structure, and a rule for working out
# where in space to put it.
#
# The anchor is recomputed from the target every time the viewer is rebuilt
# rather than frozen at creation, so a label follows its region through a
# superposition instead of being left behind at the old coordinates.

LABEL_COLORS = {
    "yellow": "#FFD24A", "white": "#FFFFFF", "cyan": "#5BE5E5",
    "orange": "#FF9F43", "green": "#59C36A", "magenta": "#F06BA8",
    "red": "#E4595B", "blue": "#4C9BE8", "black": "#101010",
}
DEFAULT_LABEL_COLOR = "#FFD24A"

# Targets that mean "a slab of the structure relative to the bilayer". They
# only resolve for a structure carrying membrane planes — an oriented or built
# system — because without them there is no sense in which a residue is inside.
MEMBRANE_TARGETS = {
    "transmembrane": "tm", "tm": "tm", "membrane": "tm", "bilayer": "tm",
    "tm domain": "tm", "transmembrane domain": "tm", "tm region": "tm",
    "extracellular": "out", "outside": "out", "periplasmic": "out",
    "lumenal": "out", "luminal": "out", "exoplasmic": "out",
    "intracellular": "in", "inside": "in", "cytoplasmic": "in",
    "cytosolic": "in",
}


def _structure_planes(entry: dict):
    """The bilayer plane z values for a loaded structure, or None."""
    try:
        return mem.membrane_planes(entry["path"])
    except Exception:
        return None


def _resolve_label_target(entry: dict, target: str):
    """
    Work out which atoms a label is about.

    Understands, in order: the membrane slabs above, a residue range
    ("100-250", "A/100-250"), a whole chain ("chain A"), and finally anything
    measure.parse_spec understands — "ARG12", "A/12", a ligand code. That last
    fallback is what keeps the vocabulary the same as the measurement tools',
    so a residue is named the same way here as anywhere else in the app.

    Returns:
        (indices, description, error).
    """
    atoms = _atoms_of(entry)
    n = len(atoms["resseq"])
    t = (target or "").strip()
    tl = t.lower()

    # ── Membrane slabs ──────────────────────────────────────────────────────
    slab = MEMBRANE_TARGETS.get(tl)
    if slab:
        planes = _structure_planes(entry)
        if not planes:
            return [], "", (
                f"{entry['pdb_id']} carries no bilayer planes, so '{t}' has no "
                "meaning for it. Orient it first (Membrane → orient), or label a "
                "residue range instead.")
        lo, hi = planes
        z = atoms["xyz"][:, 2]
        if slab == "tm":
            keep = [i for i in range(n) if lo <= z[i] <= hi
                    and atoms["resname"][i] != mem.DUMMY_RESNAME]
            desc = f"the transmembrane slab, z {lo:.1f} to {hi:.1f} Å"
        elif slab == "out":
            keep = [i for i in range(n) if z[i] > hi
                    and atoms["resname"][i] != mem.DUMMY_RESNAME]
            desc = f"everything above the bilayer, z > {hi:.1f} Å"
        else:
            keep = [i for i in range(n) if z[i] < lo
                    and atoms["resname"][i] != mem.DUMMY_RESNAME]
            desc = f"everything below the bilayer, z < {lo:.1f} Å"
        if not keep:
            return [], "", f"No atom of {entry['pdb_id']} lies in {desc}."
        return keep, desc, ""

    # ── Residue range, optionally chain-qualified ───────────────────────────
    m = re.fullmatch(r"(?:residues?\s+)?(?:([A-Za-z0-9])[/:])?(\d+)\s*[-–]\s*(\d+)",
                     t, re.IGNORECASE)
    if m:
        chain, first, last = m.group(1), int(m.group(2)), int(m.group(3))
        if first > last:
            first, last = last, first
        keep = [i for i in range(n)
                if first <= atoms["resseq"][i] <= last
                and (chain is None or atoms["chain"][i].upper() == chain.upper())]
        where = f" of chain {chain.upper()}" if chain else ""
        if not keep:
            return [], "", (f"{entry['pdb_id']} has no residue between {first} and "
                            f"{last}{where}. Its numbering may not start at 1 — "
                            "check the sequence browser.")
        return keep, f"residues {first}-{last}{where}", ""

    # ── Whole chain ─────────────────────────────────────────────────────────
    m = re.fullmatch(r"chain\s+([A-Za-z0-9])", t, re.IGNORECASE)
    if m:
        chain = m.group(1).upper()
        keep = [i for i in range(n) if atoms["chain"][i].upper() == chain]
        if not keep:
            real = sorted({c for c, rn in zip(atoms["chain"], atoms["resname"])
                           if rn != mem.DUMMY_RESNAME})
            return [], "", (f"{entry['pdb_id']} has no chain {chain}. It has: "
                            + ", ".join(real) + ".")
        return keep, f"chain {chain}", ""

    # ── Anything the measurement tools understand ───────────────────────────
    spec = mz.parse_spec(t, resnames_present=atoms["resnames_present"])
    groups, err = mz.resolve(spec, atoms)
    if err or not groups:
        return [], "", (
            f"Could not work out what '{t}' refers to in {entry['pdb_id']}"
            + (f": {err}" if err else ".")
            + " Try a residue range like '100-250', 'chain A', a residue like "
              "'ARG12', or 'transmembrane' for an oriented membrane protein.")
    keep = [i for _, idx in groups for i in idx]
    return keep, ", ".join(lbl for lbl, _ in groups[:4]), ""


def _label_anchor(atoms, indices, offset: float):
    """
    Where to draw the text: the centroid of the region, pushed outwards.

    A label at the bare centroid of a domain sits *inside* it and is hidden by
    the cartoon in front. Offsetting along the horizontal direction away from
    the structure's own centre moves it clear of the bundle, which for a
    membrane protein puts "TM" beside the helices where a figure wants it.

    The bilayer dummy atoms are left out of that centre. They form a wide,
    symmetric box around the protein, and including them drags the reference
    point towards the box centre rather than the molecule's, which sends the
    label off in a direction that has nothing to do with the structure.
    """
    import numpy as np
    pts = atoms["xyz"][indices]
    centre = pts.mean(axis=0)
    if not offset:
        return [round(float(v), 3) for v in centre]
    real = [i for i, rn in enumerate(atoms["resname"]) if rn != mem.DUMMY_RESNAME]
    whole = atoms["xyz"][real].mean(axis=0) if real else atoms["xyz"].mean(axis=0)
    out = np.array([centre[0] - whole[0], centre[1] - whole[1], 0.0])
    norm = float(np.linalg.norm(out))
    if norm < 1e-6:                       # centred on the axis — push along +x
        out, norm = np.array([1.0, 0.0, 0.0]), 1.0
    anchor = centre + out / norm * float(offset)
    return [round(float(v), 3) for v in anchor]


def tool_add_label(text: str, target: str, structure: str = "",
                   color: str = "yellow", offset: float = 0.0) -> ToolResult:
    """
    Pin a text label to a region of a structure so a figure can be read.

    Args:
        text     : What to write, e.g. "TM", "nucleotide-binding domain".
        target   : What to write it on — "transmembrane", "extracellular",
                   "intracellular", a residue range like "100-250" or
                   "A/100-250", "chain A", or a residue such as "ARG12".
        structure: Which loaded structure. Defaults to the active one.
        color    : A colour name from LABEL_COLORS, or a #RRGGBB value.
        offset   : Ångström to push the label outwards from the region's
                   centre, so it clears the structure in front of it.

    Returns:
        Confirmation, or an explanation of what did not resolve.
    """
    entry = find_structure(structure) if structure else active_structure()
    if not entry:
        return _fail("No structure is loaded" + (f" called '{structure}'." if structure
                     else ", so there is nothing to label."))
    if not (text or "").strip():
        return _fail("A label needs some text — say what it should read.")

    indices, desc, err = _resolve_label_target(entry, target)
    if err:
        return _fail(err)

    hexcolor = LABEL_COLORS.get((color or "").strip().lower(), "")
    if not hexcolor:
        c = (color or "").strip()
        hexcolor = c if re.fullmatch(r"#[0-9A-Fa-f]{6}", c) else DEFAULT_LABEL_COLOR

    st.session_state.annotations.append({
        "id":     uuid.uuid4().hex[:8],
        "sid":    entry["sid"],
        "text":   text.strip(),
        "target": (target or "").strip(),
        "desc":   desc,
        "color":  hexcolor,
        "size":   4.0,
        "offset": float(offset or 0.0),
    })
    return _ok(f"Labelled {desc} of {entry['pdb_id']} as '{text.strip()}' "
               f"({len(indices)} atoms).",
               {"pdb_id": entry["pdb_id"], "text": text.strip(), "region": desc,
                "atoms": len(indices)})


def tool_clear_labels(text: str = "") -> ToolResult:
    """
    Remove pinned labels.

    Args:
        text: Remove only labels reading this. Empty removes every label.
    """
    before = len(st.session_state.annotations)
    if (text or "").strip():
        want = text.strip().lower()
        st.session_state.annotations = [
            a for a in st.session_state.annotations if a["text"].lower() != want]
    else:
        st.session_state.annotations = []
    gone = before - len(st.session_state.annotations)
    left = len(st.session_state.annotations)
    if not gone:
        if text:
            return _fail(f"No label reading '{text}' is on the scene.", {"labels": left},
                         hints=["list_labels"])
        return _ok("There are no labels to remove.", {"removed": 0, "labels": 0})
    return _ok(f"Removed {gone} label(s).", {"removed": gone, "labels": left})


def tool_list_labels() -> ToolResult:
    """Every label currently pinned to the scene."""
    rows = st.session_state.annotations
    if not rows:
        return _ok("No labels are pinned to the scene.", {"labels": []})
    out = [f"{len(rows)} label(s):"]
    for a in rows:
        entry = find_structure(a["sid"])
        out.append(f"  '{a['text']}' on {entry['pdb_id'] if entry else '?'} — "
                   f"{a['desc'] or a['target']}")
    return _ok("\n".join(out), {"labels": [a["text"] for a in rows]})


# ── NGL → MDAnalysis expression approximation ────────────────────────────────

def tool_render_image(quality: str = "draft", width: int = 1200,
                      height: int = 900) -> ToolResult:
    """
    Ray trace the current scene with PyMOL and store the resulting PNG path.

    The active NGL representation stack is handed to PyMOL verbatim; the worker
    translates NGL selections and colour schemes into PyMOL equivalents. This
    is a rendering operation only — it does not alter the interactive viewer.

    Args:
        quality: "draft" (seconds) or "publication" (minutes, higher fidelity).
        width  : Output width in pixels.
        height : Output height in pixels.

    Returns:
        Status string describing the render, or an error message.
    """
    if not PYMOL_AVAILABLE or pymol_render is None:
        return _fail("PyMOL is not available. Set PYMOL_PYTHON to an interpreter "
                     "that can `import pymol2`.", {"unavailable": "pymol"})
    if not st.session_state.pdb_id:
        return _fail("No structure loaded — fetch a structure before rendering.")

    # Publication renders are far slower, so they get a correspondingly
    # larger subprocess budget than the interactive default.
    quality = (quality or "draft").lower()
    timeout = 900 if quality in ("publication", "high", "final") else 300

    # The worker renders one structure. When the active one carries a
    # superposition, hand it a copy with the transform baked in so the figure
    # matches the orientation on screen rather than the deposited coordinates.
    active = active_structure()
    pdb_path = st.session_state.pdb_path
    if active and active.get("matrix"):
        baked = STRUCTURES_DIR / f"{active['pdb_id']}_superposed.pdb"
        ok_write, _detail = sup.write_transformed(active["path"], active["matrix"], baked)
        if ok_write:
            pdb_path = str(baked)

    reps = [r for r in st.session_state.representations
            if rep_applies_to(r, active["sid"] if active else "")]
    reps = [dict(r, color=active["color"]) if r.get("color") == BY_STRUCTURE else r
            for r in reps]

    ok, msg, png = pymol_render.render_scene(
        pdb_path=pdb_path,
        pdb_id=st.session_state.pdb_id,
        representations=reps,
        background=st.session_state.background,
        camera_target=st.session_state.camera_target,
        width=int(width), height=int(height),
        quality=quality,
        out_dir="renders",
        timeout=timeout,
    )
    st.session_state.render_path = png if ok else None
    st.session_state.render_msg = msg
    if not ok:
        return _fail("Render failed: " + msg)
    return _ok(msg, {"quality": quality, "width": int(width), "height": int(height)},
               files=[png])


def _ngl_to_mda_approx(ngl_sel: str) -> str:
    """
    Best-effort conversion of an NGL selection string to an MDAnalysis selection.

    Handles common keywords, chain syntax (:A), element syntax (_C), and
    serial-list syntax (@1,2,3). Unknown expressions are passed through.

    Args:
        ngl_sel: NGL selection string produced by _expression_to_ngl or the viewer.

    Returns:
        MDAnalysis-compatible selection string.
    """
    m = {
        "protein":  "protein",
        "ligand":   "not protein and not water",
        "organic":  "not protein and not water",
        "water":    "water",
        "hetero":   "not protein and not water",
        "helix":    "secondary_structure H",
        "sheet":    "secondary_structure E",
        "backbone": "backbone",
        "sidechain":"not backbone",
        "all":      "all",
        "none":     "name XXXX",   # empty selection sentinel
    }
    if ngl_sel in m:
        return m[ngl_sel]
    if ngl_sel.startswith(":"):          # chain identifier → MDAnalysis segid
        return f"segid {ngl_sel[1:]}"
    if ngl_sel.startswith("_"):          # element symbol → MDAnalysis element
        return f"element {ngl_sel[1:]}"
    if ngl_sel.startswith("@"):          # serial list → MDAnalysis index (0-based)
        serials = ngl_sel[1:].split(",")
        return "index " + " ".join(str(int(s) - 1) for s in serials[:100])
    m = re.fullmatch(r"(\d+)(?:-(\d+))?(?::([A-Za-z0-9]))?", ngl_sel)
    if m:                                # residue number/range, optional chain
        res = f"resid {m.group(1)}" + (f":{m.group(2)}" if m.group(2) else "")
        return res + (f" and (segid {m.group(3)} or chainID {m.group(3)})" if m.group(3) else "")
    if ngl_sel.startswith("[") and ngl_sel.endswith("]") and "]" not in ngl_sel[1:-1]:
        return f"resname {ngl_sel[1:-1]}"
    if re.fullmatch(r"[A-Z0-9]{2,4}", ngl_sel) and not ngl_sel.isdigit():
        return f"resname {ngl_sel}"      # bare NGL residue name ("HEM")
    # Compound: "HEM and :A", "[ATP] or (10-20:B)" — convert each term and keep
    # the logic. Passed through whole, select_within('HEM and chain A') failed
    # with "Unknown selection token: 'HEM'" (S6 eval).
    tokens = re.findall(r"\(|\)|[^\s()]+", ngl_sel)
    if len(tokens) > 1:
        out = []
        for t in tokens:
            if t in ("(", ")") or t.lower() in ("and", "or", "not"):
                out.append(t.lower())
            else:
                conv = _ngl_to_mda_approx(t)
                out.append(f"({conv})" if " " in conv else conv)
        return " ".join(out)
    return ngl_sel                       # pass-through for unknown expressions


# ═══════════════════════════════════════════════════════════════════════════════
# Tool registry for Ollama
# There are 38 tools in total, each with a name, description, and JSON schema for 
# parameters. Some of the tools have no agent such as QM, ONIOM and simulation panels, 
# but they are still included in the registry for completeness.
# tools description can be broaded to include more details about the tool's functionality, 
# usage, and any specific requirements or limitations.
# ═══════════════════════════════════════════════════════════════════════════════

# TOOLS is the formal schema sent to Ollama; TOOL_DISPATCH maps names to callables.
TOOLS = [
    {
        "type": "function", "function": {
            "name": "ask_user",
            "description": (
                "Ask the user a clarifying question with 2-4 concrete choices INSTEAD of "
                "guessing, when the request can reasonably mean different things that "
                "need different tools or give different answers (e.g. 'other chains' = "
                "other chains in the loaded file, or other PDB structures of the "
                "protein?), or names something you cannot identify. Do not use it when "
                "the working context or scene already settles the meaning, or when a "
                "tool can simply look the answer up. Nothing else runs after it: the "
                "user answers first."
            ),
            "parameters": {"type": "object", "properties": {
                "question": {"type": "string", "description": "One short question"},
                "options": {"type": "array", "items": {"type": "string"},
                            "description": "2-4 distinct, concrete readings of the request, each a short phrase"}
            }, "required": ["question", "options"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "recall",
            "description": (
                "Return, in full, results that tools produced for EARLIER requests in "
                "this session — pockets, distances, contacts, interactions, mutation "
                "scores, files written, etc. — with the call that produced each. Use it "
                "when the user refers back ('the pocket you found earlier', 'that "
                "distance', 'the file you saved', 'compare with before') and the "
                "'Remembered results' lines in the Scene block do not give enough "
                "detail. Changes nothing."
            ),
            "parameters": {"type": "object", "properties": {
                "topic": {"type": "string", "description": "What to look up: a fact id such as 'F3', a topic ('pocket', 'distance', 'salt bridge', 'file'), a tool name, a PDB id or a residue. Empty = everything remembered."},
                "limit": {"type": "integer", "description": "How many results to return, newest and best matches first (default 5)"}
            }, "required": ["topic"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "find_protein",
            "description": (
                "Identify a protein by name, gene symbol or UniProt accession and "
                "summarise every PDB structure of it — how many, by which technique, "
                "which parts of the sequence they cover, and the best file to start "
                "from. Use this for 'what structures are there for X', 'how many "
                "structures of X exist', 'what is X', 'is there a cryo-EM structure "
                "of X'. Loads nothing."
            ),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "Protein name, gene symbol or UniProt accession"},
                "organism": {"type": "string", "description": "Species, e.g. human (default), mouse, or empty for any"}
            }, "required": ["name"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "protein_structures",
            "description": (
                "List the PDB structures of a protein as a table, best first, with "
                "optional filters on technique, resolution and bound ligands. Each row "
                "gives the residues covered, which chain(s) the protein is in that entry, "
                "and the other molecules it is bound to. Use this for 'list the X-ray "
                "structures of X', 'which structures of X have a ligand bound', 'show me "
                "high resolution structures of X', and for questions about the PROTEIN "
                "beyond the loaded file: 'what other structures / chains / entries / "
                "domains / regions exist for this protein', 'other structures of it'. "
                "Loads nothing."
            ),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "Protein name or accession; empty reuses the last lookup"},
                "method": {"type": "string", "description": "X-ray, NMR or Cryo-EM"},
                "max_resolution": {"type": "number", "description": "Best-or-equal resolution in angstrom; 0 for any"},
                "ligands_only": {"type": "boolean"},
                "limit": {"type": "number"}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "protein_function",
            "description": (
                "Report a protein's biological role from UniProt's curated FUNCTION "
                "and SUBUNIT annotation — 'what does X do', 'what is the function of X', "
                "'what does X interact with', 'what is its subunit structure'. This is "
                "biological role, not structure composition (use describe_structure for "
                "ligands/chains/residues) and not structure availability (use "
                "find_protein for what depositions exist). Report exactly what UniProt "
                "says; never invent a function from the protein's name alone. For an "
                "uncharacterized entry it adds family/domain/GO evidence, and — when a "
                "structure of it is loaded — the curated functions of its structural "
                "neighbours, under INFERRED ... (LOW CONFIDENCE) headings; report those "
                "as a guess."
            ),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "Protein name or accession; empty reuses the last lookup"}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "load_protein",
            "description": (
                "Load the best PDB structure of a protein named in words, when the user "
                "gives a protein NAME rather than a 4-character PDB id — 'load the human "
                "insulin receptor', 'show me EGFR', 'open p53'. Picks the file the way "
                "the Proteins panel would and adds it to the scene. Falls back to the "
                "AlphaFold predicted model when no experimental structure exists. For "
                "an actual PDB accession use fetch_structure instead. Like "
                "fetch_structure it draws the structure itself — no show/color/hide after "
                "a plain load."
            ),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"},
                "organism": {"type": "string", "description": "Species; human by default"},
                "prefer": {"type": "string", "description": "balanced (default), coverage or resolution"},
                "method": {"type": "string", "description": "Restrict to X-ray, NMR or Cryo-EM"}
            }, "required": ["name"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "inspect_preparation",
            "description": (
                "Say what is wrong with a structure before it is simulated: how many "
                "NMR models it holds, alternate conformations, crystallisation "
                "additives, missing residues, ligands with no parameters. Use for "
                "'is this ready for simulation', 'how many states does this have', "
                "'what do I need to fix'. Changes nothing."
            ),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "description": "PDB id; empty for the active structure"}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "prepare_structure",
            "description": (
                "Clean a structure up for simulation and load the cleaned copy: keep "
                "one NMR state and drop the rest, resolve alternate conformations, "
                "remove waters and crystallisation additives, rename disulfide "
                "cysteines, optionally add hydrogens. Use for 'keep only one state', "
                "'remove the other models', 'clean this up', 'prepare it for Amber', "
                "'get it ready for Rosetta', 'add hydrogens'."
            ),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "description": "PDB id; empty for the active structure"},
                "profile": {"type": "string", "enum": ["amber", "rosetta", "md_explicit", "clean"],
                            "description": "amber and rosetta strip waters and additives; clean keeps everything but the extra states"},
                "model": {"type": "number", "description": "which NMR state to keep; 0 for the representative one"},
                "add_hydrogens": {"type": "boolean", "description": "also run reduce to protonate it"},
                "keep_waters": {"type": "boolean"},
                "keep_ligands": {"type": "boolean"}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "orient_membrane",
            "description": (
                "Place a membrane protein in the lipid bilayer and show it there, "
                "with the two membrane planes drawn. Use for 'put this in a membrane', "
                "'orient this in the bilayer', 'where does the membrane sit', 'is this "
                "a transmembrane protein'. Takes seconds. Does NOT pack lipids — for "
                "that use build_membrane."
            ),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "description": "PDB id; empty for the active structure"},
                "source": {"type": "string", "enum": ["auto", "opm", "memembed"],
                           "description": "auto uses OPM's published orientation when the entry is in it"},
                "n_ter": {"type": "string", "enum": ["in", "out"],
                          "description": "which side of the membrane residue 1 starts on"},
                "barrel": {"type": "boolean", "description": "true for beta-barrel outer-membrane proteins"}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "build_membrane",
            "description": (
                "Pack a full lipid bilayer, water and ions around a membrane protein "
                "with PACKMOL-Memgen. Use for 'build a membrane system', 'embed this in "
                "a POPC bilayer', 'put it in a plasma membrane', 'add cholesterol to the "
                "membrane'. The job runs in the background for minutes to hours — this "
                "returns as soon as it has started."
            ),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "description": "PDB id; empty for the active structure"},
                "composition": {"type": "string",
                                "enum": ["popc", "popc_chol", "plasma_sym", "plasma_asym",
                                         "raft", "ecoli", "er", "mito", "dppc", "dmpc"],
                                "description": "lipid composition preset"},
                "salt_concentration": {"type": "number", "description": "molar, e.g. 0.15; 0 for none"},
                "lipids": {"type": "string", "description": "explicit lipid string, e.g. POPC:CHL1 or POPE:POPG//POPC (lower//upper)"},
                "ratios": {"type": "string", "description": "matching ratios, e.g. 3:1"}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "parameterize_ligand",
            "description": (
                "Make GAFF2 force-field parameters and AM1-BCC charges for a ligand "
                "with antechamber (mol2 + frcmod), needed before setup_amber can build "
                "a system that contains it. Use for 'parameterize the ligand', "
                "'make GAFF parameters for HEM', 'the ligand has charge -1'. Pass "
                "net_charge ONLY if the user stated it; otherwise leave it out."
            ),
            "parameters": {"type": "object", "properties": {
                "ligand": {"type": "string", "description": "3-letter component code, e.g. BEN; empty for the structure's only ligand"},
                "net_charge": {"type": "integer", "description": "net formal charge, only if the user said it"},
                "charge_method": {"type": "string", "enum": ["bcc", "gas"],
                                  "description": "bcc = AM1-BCC (default); gas = Gasteiger, quick test only"},
                "target": {"type": "string", "description": "structure; empty for the prepared copy or the active one"}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "setup_amber",
            "description": (
                "Set up an Amber molecular dynamics run: build the solvated, "
                "neutralised topology with tleap and write the minimisation, heating, "
                "equilibration and production inputs plus run.sh. Use for 'set up an "
                "Amber MD run', 'build an Amber topology', 'run 50 ns of MD in Amber', "
                "'solvate it in a TIP3P box'. Writes files; answers about which force "
                "field or settings to use need no tool."
            ),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "description": "structure; empty for the prepared copy or the active one"},
                "force_field": {"type": "string", "enum": ["ff19SB", "ff14SB", "ff15ipq"]},
                "water_model": {"type": "string", "enum": ["opc", "tip3p", "tip4pew", "spce"],
                                "description": "empty for the force field's own water model"},
                "box_shape": {"type": "string", "enum": ["octahedron", "cubic"]},
                "buffer": {"type": "number", "description": "solute-to-box-edge clearance in Å, default 12"},
                "salt_concentration": {"type": "number", "description": "molar, default 0.15"},
                "ions": {"type": "string", "enum": ["Na+/Cl-", "K+/Cl-"]},
                "temperature": {"type": "number", "description": "kelvin, default 300"},
                "pressure": {"type": "number", "description": "bar, default 1"},
                "nanoseconds": {"type": "number", "description": "production length in ns, default 100"},
                "engine": {"type": "string", "enum": ["pmemd.cuda", "pmemd.MPI", "sander"]}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "setup_gromacs",
            "description": (
                "Write GROMACS run-parameter files (em.mdp, nvt.mdp, npt.mdp, md.mdp) "
                "for an MD run of the structure. Use for 'set up a 100 ns GROMACS run', "
                "'give me GROMACS mdp files at 310 K'. Does not build a GROMACS "
                "topology; the README it writes says how."
            ),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "description": "structure; empty for the prepared copy or the active one"},
                "temperature": {"type": "number", "description": "kelvin, default 300"},
                "pressure": {"type": "number", "description": "bar, default 1"},
                "nanoseconds": {"type": "number", "description": "production length in ns, default 100"}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "setup_rosetta_docking",
            "description": (
                "Write RosettaLigand docking job files (dock.xml, options, params "
                "command, README) for a ligand in the structure. Use for 'set up "
                "Rosetta docking', 'redock the ligand with Rosetta', 'generate 500 "
                "docking poses'. Rosetta is not installed, so the job is written, "
                "not run."
            ),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "description": "structure; empty for the prepared copy or the active one"},
                "ligand": {"type": "string", "description": "3-letter component code; empty for the only ligand"},
                "ligand_chain": {"type": "string", "description": "chain letter for the ligand, default X"},
                "nstruct": {"type": "integer", "description": "number of output poses, default 100"}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "setup_qm",
            "description": (
                "Cut a quantum-mechanics cluster model around a ligand (plus residues "
                "within a radius, link hydrogens capping cut bonds) and write "
                "Gaussian, ORCA, Psi4 and xyz inputs. Use for 'set up a DFT "
                "calculation on the active site', 'QM region within 5 Å of the "
                "ligand', 'write a Gaussian input for the ligand'. Pass charge only "
                "if the user stated it."
            ),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "description": "structure; empty for the prepared copy or the active one"},
                "center": {"type": "string", "description": "3-letter ligand code to centre on; empty for the only ligand"},
                "chain": {"type": "string", "description": "chain of the ligand copy, when the ligand occurs more than once; 'all' for every copy"},
                "radius": {"type": "number", "description": "include residues within this many Å, default 4; 0 = ligand alone"},
                "side_chains_only": {"type": "boolean", "description": "cut residues at CA-CB (default true)"},
                "charge": {"type": "integer", "description": "total charge, only if the user said it"},
                "multiplicity": {"type": "integer", "description": "spin multiplicity, default 1"},
                "method": {"type": "string", "enum": ["B3LYP", "B3LYP-D3", "M06-2X", "wB97XD", "PBE0", "HF", "MP2"]},
                "basis": {"type": "string", "enum": ["6-31G(d)", "6-311+G(d,p)", "def2-SVP", "def2-TZVP", "cc-pVDZ", "cc-pVTZ"]},
                "job": {"type": "string", "enum": ["sp", "opt", "opt freq", "freq"]},
                "solvent": {"type": "string", "enum": ["none", "water", "protein-like"]},
                "cores": {"type": "integer"},
                "memory_gb": {"type": "integer"}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "setup_oniom",
            "description": (
                "Write a Gaussian ONIOM (QM/MM) input: the ligand and nearby residues "
                "as the QM layer, the rest of the Amber system as MM. Needs an Amber "
                "topology from setup_amber first. Use for 'set up QM/MM', 'ONIOM "
                "calculation on the binding site'."
            ),
            "parameters": {"type": "object", "properties": {
                "center": {"type": "string", "description": "3-letter ligand code for the QM layer; empty for the only ligand"},
                "radius": {"type": "number", "description": "QM layer: residues within this many Å, default 4"},
                "side_chains_only": {"type": "boolean"},
                "mm_radius": {"type": "number", "description": "MM layer radius in Å, default 15; 0 keeps everything"},
                "method": {"type": "string", "enum": ["B3LYP/6-31G(d):Amber", "wB97XD/6-31G(d):Amber",
                                                      "M06-2X/6-31G(d):Amber", "B3LYP/6-311+G(d,p):Amber",
                                                      "PM6:Amber"]},
                "embedding": {"type": "string", "enum": ["electronic", "mechanical"]},
                "job": {"type": "string", "enum": ["opt", "sp", "opt freq"]},
                "charge": {"type": "integer", "description": "QM-layer charge, only if the user said it"},
                "multiplicity": {"type": "integer"},
                "cores": {"type": "integer"}
            }, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "membrane_status",
            "description": (
                "Report how the running membrane build is getting on, or describe the "
                "finished system. Use for 'is the membrane done', 'how is the build "
                "going', 'what is in the membrane system'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function", "function": {
            "name": "fetch_structure",
            "description": ("Download a PDB structure from RCSB by ID and save it locally. Always call this "
                            "to load a structure before any other operation. The first structure loaded "
                            "is drawn right away as a rainbow cartoon: a plain 'load X' needs no "
                            "show, color or hide call after it."),
            "parameters": {"type": "object", "properties": {
                "pdb_id": {"type": "string"}}, "required": ["pdb_id"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "load_local",
            "description": "Load a PDB structure from a local file path",
            "parameters": {"type": "object", "properties": {
                "filepath": {"type": "string"}}, "required": ["filepath"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "select",
            "description": (
                "Create a named selection. Supported expressions: "
                "protein, ligand, organic, solvent, hetero, backbone, sidechain, "
                "chain A, resn ATP, symbol C, ss H (helix), ss S (sheet), ss L (loop), "
                "non-standard residues, b > 50 (B-factor filter)"
            ),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "Short name for this selection, e.g. atp_res"},
                "expression": {"type": "string"}
            }, "required": ["name", "expression"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "select_within",
            "description": "Select all residues within a given radius (Å) of a named selection",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"},
                "radius": {"type": "number"},
                "target_selection": {"type": "string", "description": "A previously created selection name or NGL expression"}
            }, "required": ["name", "radius", "target_selection"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "select_by_bfactor",
            "description": "Select atoms by B-factor value",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"},
                "operator": {"type": "string", "enum": [">", "<"]},
                "threshold": {"type": "number"}
            }, "required": ["name", "operator", "threshold"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "show",
            "description": "Add a visual representation for a selection",
            "parameters": {"type": "object", "properties": {
                "rep_type": {"type": "string", "enum": NGL_REP_TYPES},
                "selection": {"type": "string", "description": "A selection name or NGL expression"},
                "color": {"type": "string", "default": "element",
                          "description": ("Colour scheme or plain colour name. Schemes: "
                                          + ", ".join(sorted(set(NGL_COLOR_SCHEMES.values()))))},
                "exclusive": {"type": "boolean", "default": False,
                              "description": ("True when the user said 'only', 'just' or "
                                              "'nothing else' — hides every other "
                                              "representation first, so the viewer shows "
                                              "this selection and nothing else.")}
            }, "required": ["rep_type", "selection"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "hide",
            "description": "Remove all representations for a given selection",
            "parameters": {"type": "object", "properties": {
                "selection": {"type": "string"}
            }, "required": ["selection"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "hide_all",
            "description": "Clear all visual representations from the viewer",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function", "function": {
            "name": "show_all",
            "description": "Show all atoms with a given representation type",
            "parameters": {"type": "object", "properties": {
                "rep_type": {"type": "string", "enum": ["cartoon", "ball+stick", "surface", "ribbon", "spacefill"], "default": "cartoon"}
            }}
        }
    },
    {
        "type": "function", "function": {
            "name": "color",
            "description": "Apply a color to a selection. Color can be a name (red, green, blue, white) or scheme (element, spectrum, chainname, residueindex, bfactor). For 'chain A red, chain B blue, rest grey' make one call per part, in that order, the last with selection 'rest'.",
            "parameters": {"type": "object", "properties": {
                "color": {"type": "string"},
                "selection": {"type": "string",
                              "description": "e.g. 'chain A', 'chains C-K', 'chains C, D and E', 'ligand', or 'rest' = everything not coloured by an earlier call"}
            }, "required": ["color", "selection"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "set_transparency",
            "description": "Set transparency (0.0 = opaque, 1.0 = fully transparent) on all representations of a selection",
            "parameters": {"type": "object", "properties": {
                "value": {"type": "number"},
                "selection": {"type": "string"}
            }, "required": ["value", "selection"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "measure_distance",
            "description": ("Measure the distance between two residues, atoms or "
                            "ligands. Reports the CA-CA separation, the closest "
                            "approach between any two atoms, and centre-to-centre"),
            "parameters": {"type": "object", "properties": {
                "a": {"type": "string",
                      "description": ("First residue, atom or ligand. Write it as "
                                      "'12', 'A/12', 'ARG12', '12.CA', a ligand code "
                                      "like 'BEN', or '1UBQ/A/12' to name a structure")},
                "b": {"type": "string", "description": "Second residue, same syntax"}
            }, "required": ["a", "b"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "add_structure",
            "description": ("Load ANOTHER PDB structure into the same 3D scene, keeping the "
                            "ones already loaded. Use this whenever the user wants to compare, "
                            "superimpose or align two structures"),
            "parameters": {"type": "object", "properties": {
                "pdb_id": {"type": "string", "description": "PDB accession to add, e.g. 1UBQ"}
            }, "required": ["pdb_id"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "remove_structure",
            "description": "Remove one structure from the scene",
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "description": "PDB ID of the structure to remove"}
            }, "required": ["target"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "find_interactions",
            "description": ("Find non-covalent interactions — salt bridges, hydrogen "
                            "bonds, disulfide bonds, pi-stacking, cation-pi and metal "
                            "coordination — and draw them in the viewer. Use for 'what "
                            "are the salt bridges', 'show the hydrogen bonds', 'what "
                            "interactions does the ligand make', 'any disulfides'"),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string",
                           "description": ("Restrict to interactions involving this "
                                           "residue, ligand or chain, e.g. 'BEN', "
                                           "'A/57', 'chain B'. Omit to scan everything")},
                "types": {"type": "string",
                          "description": ("Comma-separated: salt_bridge, hbond, "
                                          "disulfide, pi_stacking, cation_pi, metal, "
                                          "hydrophobic. Omit for all but hydrophobic")},
                "radius": {"type": "number",
                           "description": "Override the distance cutoff in Angstroms"},
                "include_water": {"type": "boolean", "description": "Include waters"}
            }}
        }
    },
    {
        "type": "function", "function": {
            "name": "highlight",
            "description": ("Highlight residues, a ligand, or a region in the viewer "
                            "and zoom to it. Use for 'highlight residue 57', 'show me "
                            "the binding site', 'highlight 100-120', 'highlight the "
                            "interacting residues'"),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string",
                           "description": ("What to highlight: '57', 'A/57', 'BEN', a "
                                           "list '57, 102, 195', a range '57-102', "
                                           "'interactions' for the last scan's residues, "
                                           "or 'pocket 3' for a pocket find_pockets found")},
                "style": {"type": "string", "enum": NGL_REP_TYPES,
                          "description": "ball+stick for residues, surface for a region"},
                "color": {"type": "string", "description": "Colour name or scheme"}
            }, "required": ["target"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "clear_scene",
            "description": ("Unload every structure and start over. ONLY when the user "
                            "explicitly asks to clear, reset or start fresh"),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function", "function": {
            "name": "measure_angle",
            "description": ("Measure the angle at one residue between two others, "
                            "in degrees"),
            "parameters": {"type": "object", "properties": {
                "a": {"type": "string", "description": "First residue"},
                "b": {"type": "string", "description": "Residue at the vertex"},
                "c": {"type": "string", "description": "Third residue"}
            }, "required": ["a", "b", "c"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "measure_dihedral",
            "description": ("Measure a torsion (dihedral) angle through four residues "
                            "or atoms, in degrees"),
            "parameters": {"type": "object", "properties": {
                "a": {"type": "string", "description": "First point"},
                "b": {"type": "string", "description": "Second point"},
                "c": {"type": "string", "description": "Third point"},
                "d": {"type": "string", "description": "Fourth point"}
            }, "required": ["a", "b", "c", "d"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "find_contacts",
            "description": ("List every residue within a cutoff of a residue or ligand. "
                            "Use for 'what does the ligand bind to', 'what surrounds "
                            "residue 57', 'what is in the binding site'"),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string",
                           "description": "Residue or ligand, e.g. 'BEN', 'A/57', 'HIS57'"},
                "radius": {"type": "number",
                           "description": "Cutoff in Angstroms; 4.0 for direct contacts"},
                "include_water": {"type": "boolean",
                                  "description": "Include water molecules"}
            }, "required": ["target"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "describe_structure",
            "description": ("Report what a structure contains: how many standard amino "
                            "acids, which residues are non-standard (ligands, cofactors, "
                            "ions, modified residues, nucleotides), chains and waters. Use "
                            "this for any question about composition, contents, ligands, "
                            "cofactors or residue counts"),
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string",
                           "description": "PDB ID to describe; omit for the active structure"},
                "detail": {"type": "string", "enum": ["brief", "full"],
                           "description": ("'brief' (default) for a short answer; 'full' only "
                                           "when the user asks for full or detailed output")}
            }}
        }
    },
    {
        "type": "function", "function": {
            "name": "describe_fold",
            "description": (
                "Report a structure's fold/topology — 'what fold is this', 'describe "
                "the topology of chain A', 'is this a beta barrel', 'what is the "
                "secondary structure'. Reports an existing CATH/SCOP classification "
                "when the entry has one (a database lookup, more accurate than any "
                "guess) and a DSSP-computed secondary-structure topology string from "
                "this structure's own coordinates. This is fold/topology, not "
                "composition — use describe_structure for ligands/chains/residues — "
                "and never invent a fold name; report exactly what this tool returns, "
                "including when it says no classification is available."
            ),
            "parameters": {"type": "object", "properties": {
                "chain": {"type": "string",
                          "description": "Restrict to one chain, e.g. 'A'; omit for all chains"}
            }}
        }
    },
    {
        "type": "function", "function": {
            "name": "predict_mutation_effect",
            "description": (
                "Predict the likely effect of a single amino-acid substitution — 'what "
                "would mutating residue 45 to alanine do', 'is H92A tolerated', 'is this "
                "substitution damaging', 'which residues are tolerated at position 30', "
                "'how constrained / how conserved is residue 12'. ESM-2 protein language "
                "model, zero-shot "
                "masked-marginal log-likelihood ratio on the loaded chain's sequence. "
                "Needs a residue number the user gave. Omit mutant to rank all 19 "
                "substitutions. The result is a MODEL PREDICTION: report its score, "
                "band and caveats as given, and never present it as measured or as a "
                "stability (ΔΔG) value."
            ),
            "parameters": {"type": "object", "properties": {
                "residue":  {"type": "string",
                             "description": "Position as the user gave it, e.g. '45', 'A/45', "
                                            "'HIS92', or a mutation like 'H92A'"},
                "mutant":   {"type": "string",
                             "description": "Substituted amino acid, e.g. 'A' or 'alanine'; "
                                            "omit to rank all substitutions"},
                "wildtype": {"type": "string",
                             "description": "Native residue if the user named it, e.g. 'H'"},
                "chain":    {"type": "string",
                             "description": "Chain id if the user named one, e.g. 'B'"},
            }, "required": ["residue"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "find_structural_neighbors",
            "description": (
                "Find known structures this one resembles in 3D — 'what is this similar "
                "to', 'find structural homologs', 'which known folds look like this', "
                "'search Foldseek'. Runs a Foldseek structural search of the loaded "
                "structure's own coordinates and reports each hit's PDB id, TM-score, "
                "and that hit's own CATH/SCOP classification, plus a fold consensus. "
                "Works for AlphaFold models and local files. For 'what fold is this' "
                "use describe_fold first (it falls back to this search on its own). "
                "Report exactly what this tool returns, including 'no confident match'. "
                "Chains with no PDB match also get AlphaFold DB neighbours (similar "
                "proteins, predicted models, no fold name) — report those as similar "
                "proteins, never as a fold. "
                "If it returns NEEDS USER CHOICE, ask the user that question and stop."
            ),
            "parameters": {"type": "object", "properties": {
                "chain": {"type": "string",
                          "description": "Restrict to one chain, e.g. 'A'; omit for all chains"},
                "max_hits": {"type": "integer",
                             "description": "Neighbours per chain, default 5"},
                "where": {"type": "string", "enum": ["auto", "local", "online"],
                          "description": ("'online' ONLY when the user said to search "
                                          "online / upload; otherwise omit")},
                "database": {"type": "string", "enum": ["pdb", "alphafold"],
                             "description": ("'alphafold' ONLY when the user asks to "
                                             "search AlphaFold DB / predicted models / "
                                             "Swiss-Prot; otherwise omit")}
            }}
        }
    },
    {
        "type": "function", "function": {
            "name": "find_pockets",
            "description": (
                "Find candidate ligand-binding pockets from the structure's own geometry "
                "(fpocket) — 'find the binding pocket', 'find the hidden pocket', 'where "
                "could a drug bind', 'is there a druggable site', 'find cavities'. Works "
                "with NO ligand bound: use this, not find_contacts, when the user asks "
                "where a pocket or binding site is without naming a ligand that is in "
                "the file. Reports each pocket's lining residues, fpocket score, "
                "druggability score and whether the file has something bound there. "
                "Report exactly what it returns; the scores are predictions, not "
                "binding measurements."
            ),
            "parameters": {"type": "object", "properties": {
                "chain": {"type": "string",
                          "description": "Only pockets lined by this chain, e.g. 'A'; omit for all"},
                "max_pockets": {"type": "integer",
                                "description": "Top pockets to show, default 5"}
            }}
        }
    },
    {
        "type": "function", "function": {
            "name": "download_foldseek_database",
            "description": (
                "Start downloading the local Foldseek PDB database (~2.2 GB) in the "
                "background. ONLY when the user explicitly asked to download / install "
                "the database — never on your own initiative."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function", "function": {
            "name": "list_structures",
            "description": ("List the structures currently in the scene, how they are placed, "
                            "and every chain of each file with the molecule it is and the "
                            "residues it has. Only the loaded files — for other structures "
                            "of a protein use protein_structures."),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function", "function": {
            "name": "superpose_structures",
            "description": ("Superimpose one loaded structure onto another so they overlay in "
                            "the same coordinate frame, and report the RMSD. This is what "
                            "'superimpose', 'align', 'overlay' and 'compare structurally' mean"),
            "parameters": {"type": "object", "properties": {
                "mobile": {"type": "string", "description": "PDB ID of the structure to move"},
                "reference": {"type": "string", "description": "PDB ID of the structure that stays fixed"},
                "method": {"type": "string", "enum": ["auto", "resnum", "sequence"],
                           "description": "How to pair residues; 'auto' is almost always right"},
                "mobile_selection": {"type": "string",
                                     "description": "MDAnalysis selection of the mobile atoms to fit, e.g. 'protein and segid A'"},
                "reference_selection": {"type": "string",
                                        "description": "MDAnalysis selection of the reference atoms to fit"}
            }, "required": ["mobile", "reference"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "clear_superposition",
            "description": "Undo a superposition and put structures back at their deposited coordinates",
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "description": "PDB ID, or 'all'"}
            }}
        }
    },
    {
        "type": "function", "function": {
            "name": "zoom",
            "description": "Focus the camera on a selection",
            "parameters": {"type": "object", "properties": {
                "selection": {"type": "string"}
            }, "required": ["selection"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "set_background",
            "description": "Set viewer background color: black, white, grey",
            "parameters": {"type": "object", "properties": {
                "color": {"type": "string"}
            }, "required": ["color"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "save_structure",
            "description": "Save the current structure to a file in the structures/ directory",
            "parameters": {"type": "object", "properties": {
                "filename": {"type": "string", "description": "e.g. my_protein.pdb"}
            }, "required": ["filename"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "remove_solvent",
            "description": "Remove all water molecules from the loaded structure and save",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "measure_mda_distance",
            "description": "Measure the distance between two MDAnalysis atom selections. Use exact MDAnalysis syntax such as 'segid A and resid 50 and name CA'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sel1": {
                        "type": "string",
                        "description": "First MDAnalysis selection"
                    },
                    "sel2": {
                        "type": "string",
                        "description": "Second MDAnalysis selection"
                    }
                },
                "required": ["sel1", "sel2"]
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "summarize_chains",
            "description": ("Summarize the chains of the active loaded structure: which "
                            "molecule each chain is, its UniProt accession, and residue counts. "
                            "Answers 'what chains are in this structure', 'which chain is X'."),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "list_residues",
            "description": (
                "List residues in the currently loaded protein structure, "
                "optionally restricted to a chain or segment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chain": {
                        "type": "string",
                        "description": "Optional chain or segment identifier, such as A or B.",
                        "default": ""
                    },
                    "max_rows": {
                        "type": "integer",
                        "default": 200
                    }
                }
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "bfactor_summary",
            "description": "Summarize B-factor values for the currently loaded structure.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "detect_contacts",
            "description": "Detect residue-level contacts between two selections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sel1": {"type": "string", "default": "protein"},
                    "sel2": {"type": "string", "default": "protein"},
                    "cutoff": {"type": "number", "default": 4.0},
                    "max_rows": {"type": "integer", "default": 100}
                }
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "detect_salt_bridges",
            "description": ("Detect candidate salt bridges between acidic and basic residues. "
                            "Not for disulfide bonds — use find_interactions."),
            "parameters": {
                "type": "object",
                "properties": {
                        "chain": {
                            "type": "string",
                            "description": (
                                "Optional chain or segment identifier, such as A. "
                                "When provided, both interacting residues must "
                                "belong to that chain."
                            )
                        },
                    "cutoff": {"type": "number", "default": 4.0},
                    "max_rows": {"type": "integer", "default": 100}
                }
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "detect_hydrogen_bonds",
            "description": ("Detect candidate hydrogen bonds using a distance-based donor/acceptor "
                            "screen. Not for disulfide bonds — use find_interactions."),
            "parameters": {
                "type": "object",
                "properties": {
                    "cutoff": {"type": "number", "default": 3.5},
                    "max_rows": {"type": "integer", "default": 100}
                }
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "nearby_residues",
            "description": "Find residues near a selected atom, residue, ligand, or protein region using MDAnalysis selections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selection": {"type": "string"},
                    "cutoff": {"type": "number", "default": 5.0},
                    "max_rows": {"type": "integer", "default": 100}
                },
                "required": ["selection"]
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "measure_mda_angle",
            "description": "Measure the angle formed by three atom selections using MDAnalysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sel1": {"type": "string"},
                    "sel2": {"type": "string"},
                    "sel3": {"type": "string"}
                },
                "required": ["sel1", "sel2", "sel3"]
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "measure_mda_dihedral",
            "description": "Measure the dihedral angle formed by four atom selections using MDAnalysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sel1": {"type": "string"},
                    "sel2": {"type": "string"},
                    "sel3": {"type": "string"},
                    "sel4": {"type": "string"}
                },
                "required": ["sel1", "sel2", "sel3", "sel4"]
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "align_structures",
            "description": "Structurally align two downloaded PDB structures by backbone RMSD",
            "parameters": {"type": "object", "properties": {
                "mobile_id": {"type": "string", "description": "PDB ID of structure to move"},
                "reference_id": {"type": "string", "description": "PDB ID of reference structure"}
            }, "required": ["mobile_id", "reference_id"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "render_image",
            "description": ("Ray trace the current view with PyMOL into a "
                            "publication-quality PNG image. Use when the user asks to "
                            "render, ray trace, or make a figure/image/picture. This "
                            "does not change the interactive 3D viewer."),
            "parameters": {"type": "object", "properties": {
                "quality": {"type": "string",
                            "description": "'draft' (fast) or 'publication' (slow, best quality)"},
                "width":   {"type": "integer", "description": "pixels, e.g. 1200"},
                "height":  {"type": "integer", "description": "pixels, e.g. 900"}
            }, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_label",
            "description": ("Pin a text label onto a region of the structure in the 3D "
                            "viewer, so a figure can be read without a colour key. Use for "
                            "'label the transmembrane domain as TM', 'mark residues 1-500 "
                            "as the N-terminal domain', 'annotate chain A'."),
            "parameters": {"type": "object", "properties": {
                "text":      {"type": "string",
                              "description": "What the label should read, e.g. 'TM'"},
                "target":    {"type": "string",
                              "description": ("What to label: 'transmembrane', "
                                              "'extracellular', 'intracellular', a residue "
                                              "range like '100-250' or 'A/100-250', "
                                              "'chain A', or a residue like 'ARG12'")},
                "structure": {"type": "string",
                              "description": "Which loaded structure; empty means the active one"},
                "color":     {"type": "string",
                              "description": "yellow, white, cyan, orange, green, magenta, red, blue, or #RRGGBB"},
                "offset":    {"type": "number",
                              "description": ("Angstrom to push the label out from the "
                                              "region's centre so it clears the structure. "
                                              "Use 25-35 for a domain, 0 for a single residue.")}
            }, "required": ["text", "target"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_labels",
            "description": "Remove pinned text labels from the viewer.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string",
                         "description": "Remove only labels reading this; empty removes all"}
            }, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_labels",
            "description": "List the text labels currently pinned to the scene.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
]
# Lambda dispatch table maps tool names → callables with argument extraction

def _safe_int(value, default):
    if value is None:
        return default

    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("", "none", "null"):
            return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default):
    if value is None:
        return default
    return float(value)


def _num(value, default: float) -> float:
    """A number from a tool argument: 100, "100", "100 ns"; default otherwise."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    m = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(m.group(0)) if m else default


def _given(value) -> bool:
    """False for an argument the model left empty ("", None, "null", "none")."""
    return value is not None and str(value).strip().lower() not in ("", "none", "null")


def _known_args(fn, args: dict) -> dict:
    """The tool arguments `fn` accepts — a stray key must not crash the call."""
    import inspect
    accepted = inspect.signature(fn).parameters
    return {k: v for k, v in (args or {}).items() if k in accepted}

TOOL_DISPATCH = {
    "ask_user": lambda a: tool_ask_user(a.get("question", ""), a.get("options") or []),
    "recall":   lambda a: tool_recall(str(a.get("topic", "")), a.get("limit", 5)),
    # ---------- Analysis tools from analysis_tools.py ----------
    "measure_mda_distance": lambda a: tool_measure_mda_distance(
        a.get("sel1", ""),
        a.get("sel2", ""),
    ),

    "summarize_chains": lambda _: tool_summarize_chains(),

    "list_residues": lambda a: tool_list_residues(
        chain=a.get("chain", ""),
        max_rows=_safe_int(a.get("max_rows", 200), 200),
    ),

    "bfactor_summary": lambda _: tool_bfactor_summary(),

    "detect_contacts": lambda a: tool_detect_contacts(
        a.get("sel1", "protein"),
        a.get("sel2", "protein"),
        _safe_float(a.get("cutoff", 4.0), 4.0),
        _safe_int(a.get("max_rows", 100), 100),
    ),

    "detect_salt_bridges": lambda a: tool_detect_salt_bridges(
        chain=a.get("chain", ""),
        cutoff=_safe_float(a.get("cutoff", 4.0), 4.0),
        max_rows=_safe_int(a.get("max_rows", 100), 100),
    ),

    "detect_hydrogen_bonds": lambda a: tool_detect_hydrogen_bonds(
        _safe_float(a.get("cutoff", 3.5), 3.5),
        _safe_int(a.get("max_rows", 100), 100),
    ),

    "nearby_residues": lambda a: tool_nearby_residues(
        a.get("selection", "protein"),
        _safe_float(a.get("cutoff", 5.0), 5.0),
        _safe_int(a.get("max_rows", 100), 100),
    ),

    "measure_mda_angle": lambda a: tool_measure_mda_angle(
        a.get("sel1", ""),
        a.get("sel2", ""),
        a.get("sel3", ""),
    ),

    "measure_mda_dihedral": lambda a: tool_measure_mda_dihedral(
        a.get("sel1", ""),
        a.get("sel2", ""),
        a.get("sel3", ""),
        a.get("sel4", ""),
    ),

    # ---------- Structure (legacy single-structure MDAnalysis path) ----------
    # Old schema name; tool_align_structures never existed, so every call
    # raised NameError. Same fit as superpose_structures (S4).
    "align_structures": lambda a: tool_superpose(
        a.get("mobile_id", ""),
        a.get("reference_id", ""),
    ),

    # ---------- Everything else: multi-structure agent loop ----------
    "add_label":         lambda a: tool_add_label(
                             a.get("text", ""), a.get("target", ""),
                             a.get("structure", ""), a.get("color", "yellow"),
                             a.get("offset", 0.0)),
    "clear_labels":      lambda a: tool_clear_labels(a.get("text", "")),
    "list_labels":       lambda a: tool_list_labels(),
    "inspect_preparation": lambda a: tool_inspect_preparation(a.get("target", "")),
    "prepare_structure": lambda a: tool_prepare_structure(
        a.get("target", ""), a.get("profile", "amber"), a.get("model", 0),
        bool(a.get("add_hydrogens", False)), a.get("keep_waters"),
        a.get("keep_ligands")),
    "orient_membrane":   lambda a: tool_orient_membrane(a.get("target", ""),
                                                        a.get("source", "auto"),
                                                        a.get("n_ter", "in"),
                                                        bool(a.get("barrel", False))),
    "build_membrane":    lambda a: tool_build_membrane(a.get("target", ""),
                                                       a.get("composition", "popc"),
                                                       a.get("salt_concentration", 0.15),
                                                       a.get("lipids", ""),
                                                       a.get("ratios", "")),
    "membrane_status":   lambda _: tool_membrane_status(),
    "parameterize_ligand":   lambda a: tool_parameterize_ligand(**_known_args(tool_parameterize_ligand, a)),
    "setup_amber":           lambda a: tool_setup_amber(**_known_args(tool_setup_amber, a)),
    "setup_gromacs":         lambda a: tool_setup_gromacs(**_known_args(tool_setup_gromacs, a)),
    "setup_rosetta_docking": lambda a: tool_setup_rosetta_docking(
                                 **_known_args(tool_setup_rosetta_docking, a)),
    "setup_qm":              lambda a: tool_setup_qm(**_known_args(tool_setup_qm, a)),
    "setup_oniom":           lambda a: tool_setup_oniom(**_known_args(tool_setup_oniom, a)),
    "find_protein":     lambda a: tool_find_protein(a.get("name", "") or a.get("protein", ""),
                                                     a.get("organism", "human")),
    "protein_structures": lambda a: tool_protein_structures(
        a.get("name", ""), a.get("method", ""), a.get("max_resolution", 0.0),
        bool(a.get("ligands_only", False)), a.get("limit", 10)),
    "protein_function":  lambda a: tool_protein_function(a.get("name", "") or a.get("protein", "")),
    "load_protein":      lambda a: tool_load_protein(a.get("name", "") or a.get("protein", ""),
                                                     a.get("organism", "human"),
                                                     a.get("prefer", "balanced"),
                                                     a.get("method", "")),
    "fetch_structure":   lambda a: tool_fetch_structure(a.get("pdb_id", "")),
    "load_local":        lambda a: tool_load_local(a.get("filepath", "")),
    "select":            lambda a: tool_select(a.get("name", "sel"), a.get("expression", "all")),
    "select_within":     lambda a: tool_select_within(a.get("name", "pocket"), a.get("radius", 5.0), a.get("target_selection", "ligand")),
    "select_by_bfactor": lambda a: tool_select_by_bfactor(a.get("name", "flex"), a.get("operator", ">"), a.get("threshold", 50.0)),
    "show":              lambda a: tool_show(a.get("rep_type", "cartoon"), a.get("selection", "protein"), a.get("color", "element"), bool(a.get("exclusive", False))),
    "hide":              lambda a: tool_hide(a.get("selection", "all")),
    "hide_all":          lambda _: tool_hide_all(),
    "show_all":          lambda a: tool_show_all(a.get("rep_type", "cartoon")),
    "color":             lambda a: tool_color(a.get("color", "red"), a.get("selection", "all")),
    "set_transparency":  lambda a: tool_set_transparency(a.get("value", 0.5), a.get("selection", "all")),
    "measure_distance":  lambda a: tool_measure_distance(
        a.get("a", "") or a.get("atom1_sel", "") or a.get("residue1", ""),
        a.get("b", "") or a.get("atom2_sel", "") or a.get("residue2", "")),
    "measure_angle":     lambda a: tool_measure_angle(a.get("a", ""), a.get("b", ""),
                                                      a.get("c", "")),
    "measure_dihedral":  lambda a: tool_measure_dihedral(a.get("a", ""), a.get("b", ""),
                                                         a.get("c", ""), a.get("d", "")),
    "find_interactions": lambda a: tool_find_interactions(
        a.get("target", ""), a.get("types", ""), a.get("radius", 0.0),
        a.get("include_water", False)),
    "highlight":         lambda a: tool_highlight(a.get("target", ""),
                                                  a.get("style", "ball+stick"),
                                                  a.get("color", "yellow")),
    "clear_scene":       lambda _: tool_clear_scene(),
    "find_contacts":     lambda a: tool_find_contacts(a.get("target", ""),
                                                      a.get("radius", 4.0),
                                                      a.get("include_water", False)),
    "add_structure":     lambda a: tool_add_structure(a.get("pdb_id", "")),
    "remove_structure":  lambda a: tool_remove_structure(a.get("target", "")),
    "list_structures":   lambda _: tool_list_structures(),
    "describe_structure": lambda a: tool_describe_structure(a.get("target", ""),
                                                            a.get("detail", "brief")),
    "describe_fold":     lambda a: tool_describe_fold(a.get("chain", "")),
    "predict_mutation_effect": lambda a: tool_predict_mutation_effect(
        str(a.get("residue", "")), a.get("mutant", ""), a.get("wildtype", ""),
        a.get("chain", "")),
    "find_structural_neighbors": lambda a: tool_find_structural_neighbors(
        a.get("chain", ""), a.get("max_hits", 5), a.get("where", "auto"),
        a.get("database", "pdb")),
    "download_foldseek_database": lambda _: tool_download_foldseek_database(),
    "find_pockets":      lambda a: tool_find_pockets(a.get("chain", ""),
                                                     a.get("max_pockets", 5)),
    "superpose_structures": lambda a: tool_superpose(
        a.get("mobile", ""), a.get("reference", ""),
        a.get("method", "auto"),
        a.get("mobile_selection", "protein"),
        a.get("reference_selection", "protein")),
    "clear_superposition": lambda a: tool_clear_superposition(a.get("target", "all")),
    "zoom":              lambda a: tool_zoom(a.get("selection", "all")),
    "set_background":    lambda a: tool_set_background(a.get("color", "black")),
    "save_structure":    lambda a: tool_save_structure(a.get("filename", "output.pdb")),
    "remove_solvent":    lambda _: tool_remove_solvent(),
    "render_image":      lambda a: tool_render_image(a.get("quality", "draft"), a.get("width", 1200), a.get("height", 900)),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-turn agent loop
# This one we need to check again for every user turn, because the system prompt 
# includes the current state of the scene.
# ═══════════════════════════════════════════════════════════════════════════════

def _system_prompt() -> str:
    """
    Build the system prompt injected at every agent turn.

    Includes the current structure, named selections, and active representation
    count so the LLM can reason about what already exists before deciding which
    tools to call.
    """
    return (
        "You are a protein structure analysis agent. "
        "The scene you are working on is described in the user message. "
        "Rules — follow exactly: "
        "A. ASK, DON'T GUESS. If the newest request can reasonably mean two or more "
        "   different things that need different tools or give different answers, and "
        "   the working context, the Scene block and the earlier conversation do not "
        "   settle it, call `ask_user` with one short question and 2-4 concrete options "
        "   — and nothing else. Also ask when it names a structure, chain, residue or "
        "   protein you cannot find in the scene or the working context. Do NOT ask "
        "   when one reading is clearly meant, when a tool can simply look the answer "
        "   up, or about a point the user has just clarified; plain commands ('load "
        "   4HHB', 'color chain A red') are never ambiguous. "
        "B. NO ANSWERS FROM MEMORY. Every fact you state — PDB ids, accessions, chain "
        "   letters, residue numbers, resolutions, distances, counts, names — must come "
        "   from a tool result, the Scene block, or the working context in THIS request. "
        "   If none of them has it, call the tool that does; if no tool can, say plainly "
        "   that you cannot tell, and offer what you can do instead. Never fill a gap "
        "   with general knowledge or a plausible-sounding number. "
        "0. Loading is ADDITIVE. `fetch_structure` and `add_structure` both keep the "
        "   structures already in the scene, so loading a new PDB never discards earlier "
        "   work. Only `clear_scene` unloads anything, and you call it ONLY when the user "
        "   explicitly says clear / reset / start over. "
        "   - 'superimpose / align / overlay / compare A and B' → load any of them not yet "
        "     in the scene, then ONE `superpose_structures` call. "
        "   - `superpose_structures` fetches a missing accession itself, so it is fine to "
        "     call it directly when both names are 4-character PDB ids. "
        "   - Report the RMSD it returns; never invent one. "
        "0a. Protein NAMES are not PDB ids. When the user names a protein rather than "
        "   giving a 4-character accession — 'the insulin receptor', 'EGFR', 'p53', "
        "   'human hemoglobin': "
        "   - 'load / show / open <protein name>' → ONE `load_protein` call. Do NOT "
        "     call fetch_structure for a protein name; it takes only a 4-character "
        "     accession and never guesses from a name. "
        "   - 'what structures are there for <protein>', 'how many structures of X', "
        "     'is there a cryo-EM structure of X' → ONE `find_protein` call. "
        "   - 'list the X-ray structures of X', 'which have a ligand bound', "
        "     'high resolution ones' → ONE `protein_structures` call with the filters. "
        "   Report the accessions, resolutions and coverage these tools return; never "
        "   invent a PDB id for a protein name. "
        "0a1. STAY ON THE WORKING CONTEXT. When one is given above, that protein is what "
        "   this conversation is about: every request that does not name a different "
        "   protein applies to it and to the structures already in the scene. Do NOT call "
        "   `find_protein`, `protein_structures` or `load_protein` again for that same "
        "   protein merely because the user wrote 'find', 'get' or 'show me' — those words "
        "   introduce an analysis ('find residues 1-500', 'get the contacts', 'show me the "
        "   binding site') far more often than a new search. Look a protein up again ONLY "
        "   when the user names a different one, or explicitly asks to search again. "
        "0a2. RESIDUE NUMBERS. The working context gives the span of the canonical "
        "   sequence each loaded entry covers. A deposited file contains none of the "
        "   residues outside that span, and does not always number the ones it has the "
        "   same way UniProt does. If the user asks about residues the loaded entry does "
        "   not cover, say what it does cover and stop — never silently shift the range "
        "   or select a different stretch instead. "
        "0a3. Biological role — 'what does X do', 'what is the function of X', 'what "
        "   does X interact with', 'what is its subunit structure' → ONE `protein_function` "
        "   call. This is UniProt's curated FUNCTION/SUBUNIT text, not structure data — "
        "   do NOT answer these from describe_structure (that reports composition: "
        "   ligands, chains, residues, not biological role) and do NOT invent a function "
        "   from the protein's name or general knowledge. If `protein_function` reports no "
        "   FUNCTION annotation, say so first; then, only if it lists an INFERRED "
        "   FUNCTION ... (LOW CONFIDENCE) section, give that as a low-confidence "
        "   computational guess from family/domain matches or structural neighbours "
        "   (name the neighbour and its similarity) — never as the known "
        "   function — and add nothing it does not list. GO terms it lists as "
        "   recorded (experimental codes) are observations; say which is which. "
        "0a4. Fold/topology — 'what fold is this', 'describe the topology of chain A', "
        "   'is this a beta barrel', 'what is the secondary structure' → ONE "
        "   `describe_fold` call. This is fold/topology classification — do NOT answer "
        "   these from describe_structure (that reports composition, not fold) and do "
        "   NOT invent a fold name from the protein's name or general knowledge. Report "
        "   exactly what `describe_fold` returns, including when it says no "
        "   classification is available. When it lists several domains for a chain, "
        "   name each with its residue range — never one fold for the whole chain. "
        "0a5. Structural similarity — 'what is this similar to', 'find structural "
        "   homologs / neighbours', 'what known structures look like this', 'run "
        "   Foldseek' → ONE `find_structural_neighbors` call. Name a fold only when a "
        "   hit's CATH/SCOP classification in its output says so; if it reports no "
        "   confident match, say that — do not guess a fold. If `describe_fold` or "
        "   `find_structural_neighbors` returns NEEDS USER CHOICE, put both options "
        "   to the user and stop — do not choose. When the user then says to search "
        "   online → `find_structural_neighbors` with where='online'; when they say to "
        "   download the database → `download_foldseek_database`. "
        "0a5b. Single mutations — 'what would mutating residue 45 to alanine do', 'is "
        "   H92A tolerated', 'which substitutions are tolerated at 30', 'how conserved "
        "   is residue 63' → ONE "
        "   `predict_mutation_effect` call with the residue the user gave. Never answer "
        "   these from general knowledge. Report its score, band and caveats, and call "
        "   it an ESM-2 model prediction, not a measurement. If no residue number was "
        "   given, ask for one. "
        "0a5c. Pockets — 'find the (hidden) binding pocket', 'where could a drug bind', "
        "   'is there a druggable site', 'find cavities' → ONE `find_pockets` call. It "
        "   finds pockets from the geometry, so it works when no ligand is bound; do not "
        "   use find_contacts or find_interactions for these unless the user names a "
        "   ligand that is in the file. Report its pockets, lining residues and "
        "   druggability as given, as predictions, and never invent a pocket. To show one: "
        "   `highlight` target 'pocket N'. "
        "0a6. SCENE vs PROTEIN. Two different questions: "
        "   - About the LOADED FILE — 'what chain is loaded', 'which chain is HER2', "
        "     'what chains are in this structure' → answer from the 'chains' list in the "
        "     Scene block (every chain of a loaded file is in the scene, with the "
        "     molecule it is). Name the chain that IS the working-context protein; never "
        "     assume it is chain A. Mention the other molecules in the file too. "
        "   - About the PROTEIN beyond the loaded file — 'other chains / structures / "
        "     entries / regions available for this protein', 'I mean the protein, not the "
        "     scene' → ONE `protein_structures` call (no name needed; it reuses the "
        "     working context). Report entries, residue ranges, which chain the protein "
        "     is in each, and the partner molecules. This is NOT a `protein_function` "
        "     question. "
        "   Residue ranges: report what the file actually has ('residues 24-629 "
        "   resolved'), not only the UniProt span the construct maps to. "
        "0a7. EARLIER CONVERSATION. The user message may start with the last few chat "
        "   turns. Use them only to resolve what 'it', 'this protein', 'the other ones' "
        "   or a correction like 'I meant X, not Y' refers to — a correction means your "
        "   previous answer missed the question, so answer the corrected question, not "
        "   the old one and not a different one. Act only on the newest request. "
        "   The earlier turns are TRIMMED and are not a source of facts: never copy "
        "   accessions, residue ranges or chain letters out of them — call the tool that "
        "   answers the question (e.g. `protein_structures`) and report its result. "
        "0a8. EARLIER RESULTS. The 'Remembered results' lines in the Scene block are "
        "   tool output from earlier requests, each with the call that produced it. "
        "   'the pocket you found earlier', 'that distance', 'the file you saved', "
        "   'compare with the salt bridges from before' → use those lines; if they lack "
        "   the detail needed, call `recall` with the topic first. They ARE a source of "
        "   facts (unlike the trimmed chat turns) — quote them, never re-estimate. Say "
        "   which structure a remembered result was for. Run the tool again only when "
        "   the user asks to redo it or nothing remembered matches. "
        "0d2. Preparing a structure — 'keep only one state', 'remove the other "
        "   models', 'this NMR structure has 20 states', 'clean it up', 'add "
        "   hydrogens', 'get it ready for Amber / Rosetta / simulation' → ONE "
        "   `prepare_structure` call with the right profile. 'What needs fixing', "
        "   'is this ready to simulate', 'how many states' → `inspect_preparation`. "
        "   Preparation comes BEFORE membrane embedding: if the user wants both, "
        "   prepare first, then orient or build the membrane on the prepared copy. "
        "0e. Membranes — 'put this in a membrane', 'embed it in a bilayer', "
        "   'build a POPC system', 'add cholesterol to the membrane': "
        "   - Just seeing the protein in the membrane → `orient_membrane` (seconds). "
        "   - Actually packing lipids around it → `build_membrane` (minutes to "
        "     hours, runs in the background). Say that it is running; do NOT claim "
        "     it is finished. "
        "   - 'is it done' → `membrane_status`. Report what it returns. "
        "0e2. Simulation setup — 'set up an MD run / Amber / GROMACS', 'build a "
        "   topology', 'Rosetta docking', 'QM / DFT region', 'QM/MM / ONIOM': "
        "   - Amber → `setup_amber`; GROMACS → `setup_gromacs`; Rosetta ligand "
        "     docking → `setup_rosetta_docking`; QM cluster → `setup_qm`; QM/MM → "
        "     `setup_oniom` (needs `setup_amber` first). Pass the user's numbers "
        "     (ns, K, bar, Å, poses) as arguments; leave the rest at their defaults. "
        "   - If the structure is not a prepared copy (name ends _PREP) and the "
        "     user asked to prepare it too, call `prepare_structure` FIRST, then the "
        "     setup tool on the prepared copy. "
        "   - A ligand needs parameters before `setup_amber` can build: if "
        "     `setup_amber` says so, call `parameterize_ligand` for each, then "
        "     `setup_amber` again. Never invent a net charge or QM charge — pass "
        "     one only if the user said it. "
        "   - Report what the tool returned: the directory written, and anything "
        "     it says was NOT built or not installed. Never claim a topology, "
        "     a finished run or results exist unless the tool said so. "
        "   - A question about settings ('which force field should I use') is "
        "     answered in words, without calling a setup tool. "
        "0b. Composition questions — 'what ligands are in this', 'how many residues', "
        "   'what are the cofactors', 'any non-standard residues', 'what is in this "
        "   structure', 'describe it' — are answered by ONE `describe_structure` call. "
        "   Report exactly what it returns; never guess at residue counts or ligand names, "
        "   and never answer them from the structure name alone. "
        "   Use detail='brief' (the default). Use detail='full' ONLY when the user asks "
        "   for a full, detailed or complete breakdown. If they ask for a SHORT or BRIEF "
        "   description, answer in one or two sentences drawn from the brief report — do "
        "   not paste the whole thing back. "
        "0c. Measurements — 'distance between residue 1 and 5', 'how far is X from Y', "
        "   'what contacts the ligand', 'angle at residue 133' — are answered by "
        "   `measure_distance`, `find_contacts`, `measure_angle` or `measure_dihedral`. Pass the residues "
        "   through exactly as the user wrote them ('12', 'A/12', 'ARG12', 'BEN'); the "
        "   tool understands those spellings. Never compute or estimate a distance "
        "   yourself, and never round or reinterpret what the tool returns. "
        "0d. Interactions and highlighting — 'what are the salt bridges', 'show hydrogen "
        "   bonds', 'what interactions does the ligand make', 'any disulfides' → ONE "
        "   `find_interactions` call (pass `target` when they name a residue, ligand or "
        "   chain). 'highlight X', 'show me the binding site' → `highlight`. Report the "
        "   distances the tool gives; never invent an interaction. "
        "1. Do not call fetch_structure or load_protein for a structure already in the "
        "   scene. Loading a different one is fine and keeps the others. "
        "1a. A chain is NOT a separately loadable structure — it is already part of "
        "   whichever entry it belongs to. 'load chain A too' / 'also load chain H' "
        "   when that chain's structure is already in the scene means show/select that "
        "   chain, not a new load. Never reply that a chain was 'loaded' — say what you "
        "   actually did (e.g. 'chain A is part of 7MN5, already in the scene — showing "
        "   it now') or call `show`/`select` on it if that is what they clearly want. "
        "2. Call the MINIMUM tools needed. Never repeat a tool with the same arguments. "
        "3. Use a short descriptive selection name (e.g. 'nonstandard', 'atp_res', 'chain_a') — never 'sel'. "
        "4. `show` vs `select` are MUTUALLY EXCLUSIVE for the same command: "
        "   - User says 'show <type> for X' → call ONLY `show`. NEVER also call `select`. "
        "   - User says 'select X' or 'highlight X' → call ONLY `select`. Only also call `show` if the user explicitly wants a different rep type (e.g. surface, cartoon) in addition. "
        "4a. 'show ONLY X', 'JUST show X', 'show X and hide/nothing else' → ONE `show` "
        "   call with exclusive=true. Do NOT call `hide` for the other chains/parts one "
        "   by one — exclusive=true already removes every other representation, "
        "   including the full-structure layer added when the structure was loaded, "
        "   which a per-chain `hide` never reaches. Plain 'show X' with no 'only'/'just' "
        "   → exclusive=false (default), adding X alongside what is already shown. "
        "5. Expression rules for `select` and `show`: "
        "   - 'standard residues' or 'protein' → expression='protein' "
        "   - 'ligand' or 'small molecule' → expression='ligand' "
        "   - 'non-standard residues' → expression='non-standard residues' "
        "   - 'chain A' → expression='chain A' "
        "   - 'chains' or 'all chains' → expression='chains' "
        "   - NEVER invent residue names like STANDARD, CANONICAL, NORMAL — use 'protein' instead. "
        "5a. Labelling — 'label the transmembrane domain as TM', 'mark residues "
        "   1-500 as the N-terminal domain', 'annotate chain A', 'put a caption on "
        "   the binding site' → ONE `add_label` call. `target` takes "
        "   'transmembrane', 'extracellular' or 'intracellular' for a membrane "
        "   protein that has been oriented, a residue range such as '100-250' or "
        "   'A/100-250', 'chain A', or a single residue such as 'ARG12'. Pass "
        "   offset=30 when labelling a domain so the text clears the structure, and "
        "   offset=0 for a single residue. 'remove the labels' → `clear_labels`. "
        "   A label is text in the viewer; it does NOT select, colour or hide "
        "   anything, so never pair it with `select` or `show`. "
        "6. Never call `hide` unless the user explicitly asked to hide something. "
        "7. Call `render_image` ONLY when the user asks to render / ray trace / "
        "   make a figure, image, picture or PNG. It produces a still image and does "
        "   not change the interactive viewer, so never call it to 'show' something. "
        "   Use quality='publication' only if the user asks for high/publication quality. "
        "8. After your tools have run, reply with a plain-text summary. Stop calling tools. "
        "   Write the reply for the user only — never add remarks about tools, such as "
        "   'no further tools are needed'. "
        "8a. Questions about current state — 'are you showing all the chains?', 'why did "
        "   you label X and not Y?', 'is chain A hidden?', 'what just happened?' — are "
        "   answered ONLY from the 'Representations currently drawn' and 'Text labels "
        "   currently pinned' lines in the Scene block above, or from a real tool call "
        "   (e.g. `summarize_chains`). NEVER call `add_label`, `show`, `select` or any "
        "   other mutating tool in response to a question — a question is not an "
        "   instruction, even one phrased as 'why did you X' or ending in 'right?'. "
        "   NEVER state a residue range, chain list or visibility fact that is not "
        "   literally present in that Scene block; if it does not say, say you cannot "
        "   tell from the current state instead of guessing. "

        "9. MDAnalysis analysis tools (a deterministic, MDAnalysis-backed fallback path "
        "   alongside the tools above) — use these when the request is phrased in raw "
        "   selection syntax rather than a residue spec like 'A/12': "
        "   - 'summarize the chains', 'what chains are there' → `summarize_chains`. "
        "   - 'list the residues' (optionally 'in chain X') → `list_residues`. "
        "   - 'summarize B-factors' → `bfactor_summary`. "
        "   - 'salt bridges' → `detect_salt_bridges`, passing the requested chain restriction. "
        "   - 'hydrogen bonds' → `detect_hydrogen_bonds`. "
        "   - 'contacts between X and Y' → `detect_contacts`. "
        "   - 'residues near <selection>' → `nearby_residues` or `select_within`; for a "
        "     ligand, use the explicit residue expression 'resname XXX', never the bare "
        "     ligand name or the word 'ligand' as the selection. "
        "   - `measure_mda_distance`, `measure_mda_angle`, `measure_mda_dihedral` measure "
        "     against raw MDAnalysis selections instead of a residue spec — use them only "
        "     when the user gives selections in that syntax. Selections are plain strings "
        "     joined with 'and', e.g. 'segid A and resid 50 and name CA'; never pass "
        "     brackets, comma lists, named selections, or a visualization selection name. "
        "     `measure_mda_angle` needs three selections and `measure_mda_dihedral` needs "
        "     four — ask for whatever is missing rather than guessing."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool routing
# ═══════════════════════════════════════════════════════════════════════════════
# All 38 schemas are ~6,300 tokens, and they were sent on every call of every
# turn. Almost none are relevant to any one message: "color chain A red" does
# not need the membrane builder or the QM tools described to it. Sending the
# core set plus whichever groups the message actually touches cuts the prefix
# by roughly two thirds, and a smaller menu also makes a 3B model pick better.
#
# Routing is deliberately generous — it is far worse to withhold the one tool
# the user needed than to send a few extra. When nothing matches, when the
# routed set would be most of the list anyway, or when the model comes back
# empty-handed on the first turn, the full list is used instead.

TOOL_GROUPS = {
    "lookup":   (r"protein|uniprot|gene|structures for|structures of|entries|"
                 r"depositions|search|look up|homolog|species|isoform",
                 {"find_protein", "protein_structures", "load_protein"}),
    "load":     (r"\bload|fetch|download|open|add\b|remove structure|unload|clear|reset|"
                 r"start over|upload|local file",
                 {"fetch_structure", "load_local", "add_structure", "remove_structure",
                  "clear_scene", "load_protein"}),
    "style":    (r"show|display|hide|colou?r|transparen|opacit|cartoon|surface|"
                 r"ribbon|spacefill|licorice|ball|stick|backbone|highlight|select|"
                 r"zoom|background|render|image|figure|picture|png|ray ?trace|style",
                 {"select", "select_within", "select_by_bfactor", "show", "hide",
                  "hide_all", "show_all", "color", "set_transparency", "highlight",
                  "zoom", "set_background", "render_image"}),
    "measure":  (r"distance|how far|angle|dihedral|torsion|contact|near|within|"
                 r"\bclose to\b|measure|segid|resid",
                 {"measure_distance", "measure_angle", "measure_dihedral",
                  "find_contacts", "measure_mda_distance", "measure_mda_angle",
                  "measure_mda_dihedral", "select_within", "nearby_residues"}),
    "interact": (r"interaction|hydrogen bond|h-?bond|salt bridge|disulfide|"
                 r"pi-?stack|hydrophobic|binding site|active site|pocket",
                 {"find_interactions", "highlight", "find_contacts",
                  "detect_salt_bridges", "detect_hydrogen_bonds", "detect_contacts"}),
    "pocket":   (r"pocket|cavit|cleft|groove|druggab|binding site|where .*\bbind|"
                 r"fpocket",
                 {"find_pockets", "highlight"}),
    "describe": (r"describe|what is in|what.s in|composition|ligand|cofactor|"
                 r"residue|chain|how many|non-?standard|sequence|resolution|summar|"
                 r"b-?factor",
                 {"describe_structure", "list_structures", "summarize_chains",
                  "list_residues", "bfactor_summary"}),
    "prepare":  (r"prepare|clean|fix|protonat|hydrogen|altloc|missing|state|model|"
                 r"nmr|ready|amber|rosetta|simulat|solvent|water|save|write|export",
                 {"inspect_preparation", "prepare_structure", "save_structure",
                  "remove_solvent"}),
    "simulate": (r"simulat|\bmd\b|molecular dynamics|amber|tleap|prmtop|gromacs|\bmdp\b|"
                 r"rosetta|docking|\bdock\b|\bqm\b|quantum|\bdft\b|gaussian|\borca\b|psi4|"
                 r"oniom|qm/mm|force ?field|gaff|antechamber|parameteri[sz]|"
                 r"production run|equilibrat|\d+\s*ns\b|nanosecond",
                 {"parameterize_ligand", "setup_amber", "setup_gromacs",
                  "setup_rosetta_docking", "setup_qm", "setup_oniom",
                  "prepare_structure", "inspect_preparation"}),
    "membrane": (r"membrane|bilayer|lipid|popc|pope|cholesterol|embed|orient|opm",
                 {"orient_membrane", "build_membrane", "membrane_status",
                  "add_label", "clear_labels", "list_labels"}),
    "label":    (r"label|annotat|mark\b|caption|text|name it|write\b|legend|"
                 r"\bdomain\b|\btm\b|transmembrane|extracellular|intracellular",
                 {"add_label", "clear_labels", "list_labels", "describe_structure"}),
    "fold":     (r"\bfold|topolog|secondary structure|barrel|\bcath\b|\bscop\b|"
                 r"foldseek|structural(ly)? (neighbo|homolog|similar|relative)|"
                 r"similar to|resembl|look(s)? like",
                 {"describe_fold", "find_structural_neighbors",
                  "download_foldseek_database"}),
    "mutation": (r"mutat|substitut|variant|\bmutant|tolerat|deleterious|damaging|"
                 r"pathogenic|conserv|constrain|\b[a-z]\d+[a-z]\b|"
                 r"\bto (ala|gly|val|leu|ile|pro|phe|trp|met|ser|thr|cys|tyr|asn|gln|"
                 r"asp|glu|lys|arg|his)",
                 {"predict_mutation_effect"}),
    "superpose": (r"superpose|superimpose|align|overlay|compare|rmsd|fit\b",
                  {"superpose_structures", "clear_superposition", "list_structures",
                   "add_structure", "fetch_structure"}),
    "memory":   (r"earlier|\bbefore\b|previous|\bago\b|at the start|first time|you (found|"
                 r"measured|saved|computed|got|said)|remember|recall|\bthat (pocket|"
                 r"distance|score|file|result)",
                 {"recall"}),
}

# Always offered: orientation, and the escape hatches for a misrouted message.
CORE_TOOLS = {"describe_structure", "list_structures", "select", "show", "color", "recall"}

def _route_tools(prompt_lower: str):
    """
    The tool schemas worth sending for this message.

    Returns (tools, is_subset). is_subset is False when the full list is being
    sent, which tells run_agent there is no fallback left to try.
    """
    wanted = set(CORE_TOOLS)
    hit = False
    for pattern, names in TOOL_GROUPS.values():
        if re.search(pattern, prompt_lower):
            wanted |= names
            hit = True
    # No group matched, or routing barely narrowed anything — send everything.
    if not hit or len(wanted) > 0.6 * len(TOOLS):
        return TOOLS, False
    routed = [t for t in TOOLS if t["function"]["name"] in wanted]
    return routed, len(routed) < len(TOOLS)


def _state_block() -> str:
    """
    The live scene, as a block for the newest user message.

    This used to be the opening sentence of the system prompt, rebuilt every
    turn. That made the first bytes of the prompt change whenever a tool
    loaded a structure or added a selection, and llama.cpp only reuses the KV
    cache for the longest *common prefix* — so each turn re-processed the whole
    ~8,000-token preamble of tool schemas and rules from scratch. Keeping the
    system message byte-identical and putting the volatile part last means the
    preamble is prefilled once and reused for every later turn, and across
    chat messages too.
    """
    pdb = st.session_state.pdb_id or "none"
    sels = ", ".join(st.session_state.selections.keys()) or "none"
    loaded = ", ".join(s["pdb_id"] for s in structures()) or "none"
    fits = "; ".join(f"{s['pdb_id']} superposed on {s.get('fit_reference', '?')}"
                     for s in structures() if s["matrix"]) or "none"
    focus = focus_line()

    # Spelled out layer-by-layer, not just a count — a count told the model
    # nothing to answer "are you showing all the chains?" or "why only chain
    # B?" with, so it improvised an answer instead. This is the actual,
    # complete list of what the viewer draws; nothing outside it is visible.
    rep_desc = "; ".join(
        f"{r['type']} on '{r['selection']}' ({r.get('color', 'element')})"
        for r in st.session_state.representations
    ) or "none"

    label_desc = "; ".join(
        f"'{a['text']}' on {a.get('desc') or a.get('target') or '?'}"
        for a in st.session_state.annotations
    ) or "none"

    # run_agent sends only trimmed history, so a bare "download it" / "search
    # online" reply would otherwise arrive with nothing to answer.
    pending = st.session_state.get("foldseek_pending")
    pending_line = (
        f" Pending question to the user: structural similarity search for {pending} "
        "needs a choice — search online (uploads the structure to search.foldseek.com) "
        "or download the local Foldseek database. If this message answers it: online → "
        "find_structural_neighbors(where='online'); download → download_foldseek_database."
        if pending else "")
    # Chain identities of every loaded file. Every chain of a loaded file is in
    # the scene — "which chain is loaded" was otherwise answered "chain A" by
    # guesswork, for a file whose chain A was a different protein.
    acc = (st.session_state.focus or {}).get("accession", "")
    chain_desc = " ".join(x for x in (chain_map_line(s, acc, show_drawn=True)
                                      for s in structures()) if x)
    memory = _memory_block()
    return (
        f"Scene — current structure: {pdb}. Structures in the scene: [{loaded}]. "
        + (f"Every chain of each loaded file is in the scene: {chain_desc}. " if chain_desc else "")
        + f"Named selections: [{sels}]. "
        f"Representations currently drawn (this is everything the viewer shows, nothing "
        f"else is visible): [{rep_desc}]. "
        f"Text labels currently pinned: [{label_desc}]. "
        f"Superpositions: [{fits}]." + (f" {focus}" if focus else "") + pending_line
        + (f"\n{memory}" if memory else ""))


HISTORY_TURNS = 6          # chat messages (user + assistant) carried into a request
HISTORY_CHARS = 500        # per message; long tool dumps are trimmed


def _history_block() -> str:
    """
    The last few chat turns, as context for the newest request.

    run_agent used to send no history at all, so a follow-up such as "I am
    talking about the protein, not the scene" arrived with nothing to correct
    — the model guessed a new question (protein_function) instead of
    answering the one it had just missed. The focus line pins the protein;
    this carries what was actually asked and answered. Kept short and in the
    volatile user message so the cached system prefix is untouched.
    """
    msgs = st.session_state.get("messages") or []
    # The chat handler appends the current prompt before calling run_agent.
    if msgs and msgs[-1].get("role") == "user":
        msgs = msgs[:-1]
    recent = msgs[-HISTORY_TURNS:]
    if not recent:
        return ""
    lines = []
    for m in recent:
        text = " ".join(str(m.get("content", "")).split())
        if len(text) > HISTORY_CHARS:
            text = text[:HISTORY_CHARS] + " … [trimmed]"
        who = "User" if m.get("role") == "user" else "You"
        lines.append(f"{who}: {text}")
    return "Earlier conversation (context only — act on the newest request):\n" + "\n".join(lines)


# ── Working memory (suggestion.txt S5) ───────────────────────────────────────
# _history_block keeps 6 trimmed messages, so "compare that to the pocket you
# found earlier" had nothing to work from once that answer scrolled out. Every
# tool call that succeeds is kept in st.session_state.facts with the call that
# produced it (provenance): the newest few ride in _state_block as one line
# each, and `recall` returns any of them in full. The history stays context
# only; facts come from here, and everything here is tool-produced.

# Topic words `recall` matches on, besides the tool name and the result text.
MEMORY_TOPICS = {
    "structure load loaded entry": {"fetch_structure", "add_structure", "load_protein",
                                    "load_local", "remove_structure", "clear_scene"},
    "chain chains residue residues composition ligand": {
        "describe_structure", "summarize_chains", "list_residues", "list_structures"},
    "selection residues near within": {"select_within", "select_by_bfactor", "nearby_residues"},
    "measurement distance angle dihedral": {
        "measure_distance", "measure_angle", "measure_dihedral", "measure_mda_distance",
        "measure_mda_angle", "measure_mda_dihedral"},
    "contact contacts near": {"find_contacts", "detect_contacts", "nearby_residues"},
    "interaction interactions bond bonds salt bridge hydrogen disulfide": {
        "find_interactions", "detect_salt_bridges", "detect_hydrogen_bonds"},
    "pocket pockets cavity binding site druggable": {"find_pockets"},
    "mutation mutant variant score esm": {"predict_mutation_effect"},
    "protein function uniprot structures": {"find_protein", "protein_structures",
                                            "protein_function"},
    "fold topology neighbour neighbor foldseek similar": {"describe_fold",
                                                          "find_structural_neighbors"},
    "superposition rmsd alignment compare": {"superpose_structures", "align_structures"},
    "bfactor b-factor flexibility": {"bfactor_summary", "select_by_bfactor"},
    "preparation prepared simulation amber gromacs rosetta qm oniom ligand parameters": {
        "inspect_preparation", "prepare_structure", "parameterize_ligand", "setup_amber",
        "setup_gromacs", "setup_rosetta_docking", "setup_qm", "setup_oniom"},
    "membrane": {"orient_membrane", "build_membrane", "membrane_status"},
    "image render figure": {"render_image"},
    "file files saved written": {"save_structure", "remove_solvent"},
}

# Calls whose outcome is the live scene, which _state_block already spells out,
# or that only ask/list: remembering them would bury the results among clicks.
MEMORY_SKIP = {"ask_user", "recall", "select", "highlight", "show", "show_all", "hide",
               "hide_all", "color", "set_transparency", "zoom", "set_background",
               "add_label", "clear_labels", "list_labels", "clear_superposition",
               "download_foldseek_database"}

MEMORY_LINES = 12          # remembered results shown in the Scene block
MEMORY_LINE_CHARS = 220    # per line; `recall` has the rest
_RECALL_STOP = {"the", "a", "an", "of", "you", "i", "we", "that", "this", "it", "earlier",
                "before", "found", "find", "result", "results", "what", "was", "were",
                "my", "your", "last", "previous", "first", "about", "from", "for", "on",
                "in", "and", "to", "did", "do"}


def _topics_of(tool: str) -> list:
    return [w for words, tools in MEMORY_TOPICS.items() if tool in tools
            for w in words.split()]


def _remember(name: str, args: dict, result: ToolResult) -> None:
    """Keep one successful tool call and its result as a fact (S5)."""
    if not result.ok or name in MEMORY_SKIP or result.data.get("needs_user_choice"):
        return
    facts = st.session_state.setdefault("facts", [])
    facts.append({
        "id": f"F{len(facts) + 1}",
        "request": st.session_state.get("request_no", 0),
        "prompt": st.session_state.get("current_request", ""),
        "tool": name,
        "args": json.loads(json.dumps(args, default=str)),
        "structure": st.session_state.pdb_id or "",
        "summary": result.summary[:4000],
        "data": json.loads(json.dumps(result.data, default=str)),
        "files": list(result.files),
    })


def _fact_call(f: dict) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in f["args"].items())
    return f"{f['tool']}({args})" + (f" on {f['structure']}" if f["structure"] else "")


def _fact_gist(f: dict) -> str:
    """One line of what a remembered call found."""
    text = f["data"].get("summary") if isinstance(f["data"].get("summary"), str) else ""
    if not text:
        text = " ".join(ln.strip() for ln in f["summary"].splitlines()[:3] if ln.strip())
    text = " ".join(text.split())
    if f["files"]:
        text += " Files: " + ", ".join(Path(p).name for p in f["files"][:3])
    return text if len(text) <= MEMORY_LINE_CHARS else text[:MEMORY_LINE_CHARS] + " …"


def _memory_block() -> str:
    """
    Results of earlier requests, one line each, for the Scene block. This
    request's own calls are left out — their full results are in the turn.
    """
    now = st.session_state.get("request_no", 0)
    old = [f for f in st.session_state.get("facts", []) if f["request"] < now]
    if not old:
        return ""
    loaded = {s["pdb_id"] for s in structures()}
    shown = old[-MEMORY_LINES:]
    lines = [f"{f['id']} (request {f['request']}) {_fact_call(f)}"
             + (" [structure no longer loaded]" if f["structure"] and f["structure"] not in loaded
                else "")
             + f" → {_fact_gist(f)}" for f in shown]
    more = (f" {len(old) - len(shown)} older result(s) not listed — `recall` finds them."
            if len(old) > len(shown) else "")
    return ("Remembered results of earlier requests (tool output, with the call that "
            "produced it; `recall` gives the full result):" + more + "\n" + "\n".join(lines))


def tool_recall(topic: str = "", limit: int = 5) -> ToolResult:
    """
    Earlier tool results matching `topic` — a fact id ('F3'), a tool name, a
    topic word ('pocket', 'distance', 'salt bridge', 'file'), a PDB id or a
    residue — newest first, in full.
    """
    facts = st.session_state.get("facts", [])
    if not facts:
        return _ok("Nothing is remembered yet — no tool has produced a result in this "
                   "session.", {"facts": []})
    words = [w for w in re.findall(r"[a-z0-9]+", (topic or "").lower()) if w not in _RECALL_STOP]
    ranked = []
    for i, f in enumerate(facts):
        tags = set(_topics_of(f["tool"])) | set(f["tool"].split("_"))
        hay = " ".join([f["id"], f["tool"], f["structure"], json.dumps(f["args"]),
                        f["summary"][:3000], " ".join(f["files"])]).lower()
        if words and f["id"].lower() in words:
            score = 100
        else:
            score = sum(3 if (w in tags or w.rstrip("s") in tags) else
                        1 if (w in hay or w.rstrip("s") in hay) else 0 for w in words)
        if score or not words:
            ranked.append((score, i, f))
    if not ranked:
        known = sorted({f["tool"] for f in facts})
        return _ok(f"No remembered result matches '{topic}'. Remembered tool results: "
                   + ", ".join(known) + ". Run the tool again if it is needed.",
                   {"facts": [], "topic": topic})
    try:
        limit = max(1, min(int(limit), 10))
    except (TypeError, ValueError):
        limit = 5
    picked = [f for _, _, f in sorted(ranked, key=lambda x: (-x[0], -x[1]))[:limit]]
    out = []
    for f in picked:
        body = f["summary"] if len(f["summary"]) <= 1500 else f["summary"][:1500] + " …"
        asked = f" (\"{f['prompt'][:120]}\")" if f["prompt"] else ""
        out.append(f"{f['id']} — request {f['request']}{asked}: {_fact_call(f)}\n{body}"
                   + (f"\nFiles: {', '.join(f['files'])}" if f["files"] else ""))
    return _ok(f"{len(picked)} remembered result(s) for '{topic or 'everything'}':\n\n"
               + "\n\n".join(out),
               {"facts": [{k: f[k] for k in ("id", "request", "tool", "args", "structure",
                                             "data", "files")} for f in picked],
                "topic": topic})


# Lines a local model sometimes appends that describe its own tool use rather
# than answer the user, e.g. "No further tools are needed based on the current
# request."
_META_LINE = re.compile(
    r"^\s*(\(?\s*)?(no (further|additional|more|other) (tool|function)s?( calls?)? (are |is )?"
    r"(needed|required|necessary)|i (will|do) not (need to )?call (any )?(more |further )?tools?|"
    r"tool execution is complete)\b.*$",
    re.IGNORECASE)


# "Other chains/structures available for this protein", "I mean the protein, not
# the scene" — questions about the protein across the PDB, not the loaded file.
PROTEIN_LEVEL_RE = re.compile(
    r"\b(other|more|all|available|different|additional)\s+(\w+\s+)?"
    r"(chains?|structures?|entries|pdbs?|depositions?|models?)\b"
    r"|\bthe protein,?\s+not\b|\bnot\s+(just\s+)?(what|which|the one)?.{0,20}\b(current\s+)?scene\b"
    r"|\bbeyond\s+(the\s+)?(current\s+|loaded\s+)?(scene|file|structure)\b")


# Identifiers and measurements a reply can state. A 4-character token starting
# with a digit is a PDB id only if it has a letter and is not an ordinal ("20th").
_PDB_TOKEN = re.compile(r"\b[1-9][A-Za-z0-9]{3}\b")
_UNIPROT_TOKEN = re.compile(
    r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b")
_DECIMAL = re.compile(r"(?<![\d.])\d+\.\d+(?![\d.])")


def _unsupported_facts(reply: str, evidence: str) -> list:
    """
    PDB ids, UniProt accessions and decimal numbers in a reply that the
    evidence never mentions. A number counts as supported when some number in
    the evidence rounds to it at the reply's precision (2.9 for 2.93).
    """
    ev_upper = evidence.upper()
    bad = []
    for tok in sorted(set(_PDB_TOKEN.findall(reply))):
        if not re.search(r"[A-Za-z]", tok) or re.fullmatch(r"\d+(st|nd|rd|th|aa|kd|mer)", tok, re.I):
            continue
        if tok.upper() not in ev_upper:
            bad.append(tok)
    for acc in sorted(set(_UNIPROT_TOKEN.findall(reply))):
        if acc not in ev_upper:
            bad.append(acc)
    ev_nums = {float(x) for x in _DECIMAL.findall(evidence)}
    for num in sorted(set(_DECIMAL.findall(reply))):
        places, value = len(num.split(".")[1]), float(num)
        if not any(abs(round(e, places) - value) < 1e-9 for e in ev_nums):
            bad.append(num)
    # Named residues ("GLN86", "Asp 88"): after two failed nearby_residues
    # calls qwen2.5:7b answered "near HIS 87: GLN86, ASP88" — neither is in
    # 4HHB at those positions. Name and number must appear together somewhere.
    ev_res = {(r.upper(), n) for r, n in _RESIDUE_TOKEN.findall(evidence)}
    for r, n in sorted(set(_RESIDUE_TOKEN.findall(reply))):
        if (r.upper(), n) not in ev_res and f"{r.upper()}{n}" not in bad:
            bad.append(f"{r.upper()}{n}")
    return bad


_RESIDUE_TOKEN = re.compile(
    r"\b(ALA|ARG|ASN|ASP|CYS|GLN|GLU|GLY|HIS|ILE|LEU|LYS|MET|PHE|PRO|SER|THR|TRP|"
    r"TYR|VAL)\s?-?(\d{1,5})\b", re.IGNORECASE)


# Tools whose string arguments are identifiers or free text, not residue
# selections — PDB ids, accessions, file names, the question itself.
NUMBER_GATE_EXEMPT = {
    "ask_user", "recall", "fetch_structure", "add_structure", "replace_scene", "load_protein",
    "find_protein", "protein_structures", "protein_function", "load_local",
    "remove_structure", "render_image", "save_structure", "set_background",
    "download_foldseek_database", "find_structural_neighbors", "superpose_structures",
    "clear_superposition", "describe_structure", "list_structures", "clear_scene",
    "prepare_structure", "inspect_preparation", "orient_membrane", "build_membrane",
    "membrane_status", "clear_labels", "list_labels", "describe_fold",
    # Simulation setup: component codes (08Q, 1PE) and physical settings, no
    # residue selections.
    "parameterize_ligand", "setup_amber", "setup_gromacs", "setup_rosetta_docking",
    "setup_qm", "setup_oniom",
}

# Arguments that are free text or styling, never residue numbers.
NUMBER_GATE_FREE_ARGS = {"color", "text", "name", "style", "rep_type", "filename",
                         "quality", "operator", "profile", "types", "where",
                         "mutant", "wildtype"}


_VIEWER_VERBS = re.compile(r"\b(colou?r|show|hide|select|highlight|label|zoom|remove|delete|"
                           r"superpose|align|measure|display|render|make|turn|set)\b")


def _ambiguity_question(prompt: str) -> str:
    """
    Ask up front about requests known to split between two readings.

    Deliberately narrow: only patterns the model has been seen to guess on.
    Everything else is left to the model, which has `ask_user` and a rule for
    it. Returns the question text for the chat, or "" to carry on.
    """
    p = prompt.lower()
    entry, focus = active_structure(), st.session_state.get("focus")

    # "Other chains / chains available" with a protein in focus and a file
    # loaded: the chains inside that file, or the protein's other PDB entries?
    # Both readings are common, and they need different tools.
    if (entry and focus and re.search(r"\bchains?\b", p)
            and re.search(r"\b(other|more|available|else|rest|remaining|additional)\b", p)
            and not _VIEWER_VERBS.search(p)
            and not re.search(r"\b(scene|loaded|viewer|this file|this entry|this structure|"
                              r"screen|displayed|visible|shown)\b", p)
            and not re.search(r"\b(pdb|entries|entry|structures|depositions?|database)\b", p)
            and not any(s["pdb_id"].lower() in p for s in structures())):
        who = focus.get("gene") or focus["protein_name"]
        base = focus["accession"]
        chains = chain_molecules(entry)
        others = [c for c in chains if not any(a.split("-")[0] == base for a in c["uniprot"])]
        inside = (", ".join(f"{c['chain']} = {c['molecule'] or 'unnamed'}" for c in others)
                  if others else f"it holds only {who}")
        opts = [
            {"label": f"The other chains inside the loaded file {entry['pdb_id']} ({inside})",
             "meaning": f"list every chain inside the loaded structure {entry['pdb_id']} and "
                        f"which molecule each one is (answer from the scene chain list)"},
            {"label": f"Other PDB structures of {who} — different entries and regions of the protein",
             "meaning": f"list the other PDB structures of {who} (UniProt {base}) beyond "
                        f"{entry['pdb_id']}, grouped by sequence region — call protein_structures"},
            {"label": f"Molecules {who} is bound to across its PDB structures",
             "meaning": f"list the partner molecules (other chains) {who} is bound to across "
                        f"its PDB structures — call protein_structures and report the "
                        f"'bound to' partners"},
        ]
        ask_clarification(
            f"By “other chains”, do you mean inside the loaded {entry['pdb_id']}, "
            f"or across other structures of {who}?", opts, prompt)
        return clarification_text()

    # "Compare it" / "what is the RMSD" with nothing to compare against: the
    # model superposed a structure onto itself and reported an RMSD of 0.0.
    names_target = (any(t for t in _PDB_TOKEN.findall(prompt) if re.search(r"[A-Za-z]", t))
                    or re.search(r"\b(with|to|against|and|onto|vs\.?|versus)\s+\w", p))
    if (entry and len(structures()) < 2 and not names_target
            and re.search(r"\b(compare|comparison|superpos|superimpos|align|overlay|rmsd)", p)):
        opts = []
        prof = st.session_state.get("protein_profile")
        if focus and prof and prof.get("accession") == focus.get("accession"):
            loaded = {s["pdb_id"] for s in structures()}
            cands = [r for r in pacc.rank_structures(pacc.filter_structures(
                prof["structures"], loadable_only=True)) if r["pdb_id"] not in loaded][:3]
            who = focus.get("gene") or focus["protein_name"]
            for r in cands:
                res = f", {r['resolution']:.2f} Å" if r["resolution"] is not None else ""
                opts.append({
                    "label": f"{r['pdb_id']} — another {who} structure ({r['method']}{res}, "
                             f"residues {r['start']}-{r['end']})",
                    "meaning": f"load {r['pdb_id']} and superpose it onto {entry['pdb_id']}, "
                               f"then report the RMSD"})
        ask_clarification(
            f"Only {entry['pdb_id']} is loaded — compare it with which structure? "
            + ("Pick one, or type a PDB ID." if opts else "Type a PDB ID or a protein name."),
            opts, prompt)
        return clarification_text()
    return ""


def _clean_reply(text: str) -> str:
    """Drop tool-use meta remarks from a final reply."""
    kept = [ln for ln in text.splitlines() if not _META_LINE.match(ln)]
    return "\n".join(kept).strip()


def _tc_args(tc: dict) -> dict:
    """
    Safely extract tool-call arguments as a dict regardless of whether they're a str or dict.

    Null values are dropped: qwen2.5:7b sends "chain": null for "which salt
    bridges stabilize this protein", and a.get("chain", "") then passes None
    through, so detect_salt_bridges searched a chain literally named "None"
    and reported hemoglobin as having no salt bridges.
    """
    raw = tc.get("function", {}).get("arguments", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    if not isinstance(raw, dict):
        return {}
    # In place, so the tool_calls echoed back into the history match what ran.
    for k in [k for k, v in raw.items() if v is None]:
        del raw[k]
    return raw


# ============================================================
# Deterministic workflow pre-router
# ============================================================
#
# Returns:
#   - str: workflow was recognized and completed/failed deterministically
#   - None: prompt was not recognized; continue through the existing agent
#
# This function is intentionally narrow so unrelated PARORA functions
# continue through the original LLM/tool-calling pathway unchanged.
def handle_deterministic_workflow(user_prompt):
    import re

    prompt = str(user_prompt or "").strip()
    prompt_lower = prompt.lower()

    if not prompt:
        return None

    # Workflow 1 (load structure -> summarize chains -> visualize one
    # chain) used to live here as hard-coded regex + a tool_show
    # compatibility shim (P4 in todo.txt). Retired: with the workflow
    # disabled, qwen2.5:7b + the full tool schema list correctly sequences
    # fetch_structure -> summarize_chains -> show on its own, and even
    # handles a nonexistent chain better than the old code did (says so,
    # instead of calling tool_show blindly with no such check) — verified
    # against real Ollama before removal.

    # --------------------------------------------------------
    # Workflow 2:
    # Validate CA atoms -> highlight -> measure angle
    # --------------------------------------------------------
    #
    # Example:
    # Highlight the CA atoms of residues 50, 51, and 52 in
    # chain A and measure the angle they form.
    asks_for_angle = (
        "angle" in prompt_lower
        and "dihedral" not in prompt_lower
    )

    asks_for_ca_atoms = bool(
        re.search(
            r"\bca\b|\bc[-\s]?alpha\b|\balpha carbon",
            prompt_lower,
        )
    )

    asks_for_highlight = any(
        word in prompt_lower
        for word in (
            "highlight",
            "show",
            "display",
            "select",
            "visualize",
        )
    )

    if asks_for_angle and asks_for_ca_atoms and asks_for_highlight:
        angle_chain_match = re.search(
            r"\bchain\s+([A-Za-z0-9]+)\b",
            prompt,
            flags=re.IGNORECASE,
        )

        residue_section = re.search(
            r"\bresidues?\s+(.+?)(?:\s+in\s+chain\b|\s+of\s+chain\b|$)",
            prompt,
            flags=re.IGNORECASE,
        )

        if not angle_chain_match:
            return (
                "Please specify the chain for the angle workflow, "
                "for example: chain A."
            )

        # Restrict number extraction to the residue portion when possible,
        # preventing a PDB identifier or distance value from being mistaken
        # for a residue number.
        number_source = (
            residue_section.group(1)
            if residue_section
            else prompt
        )

        residue_numbers = re.findall(
            r"\b\d+\b",
            number_source,
        )

        if len(residue_numbers) != 3:
            return (
                "Please provide exactly three residue numbers for the "
                "CA-angle workflow."
            )

        chain = angle_chain_match.group(1).upper()
        r1, r2, r3 = residue_numbers

        try:
            universe = get_universe()
        except Exception as exc:
            return (
                "PARORA could not access the currently loaded structure: "
                f"{type(exc).__name__}: {exc}"
            )

        if universe is None:
            return (
                "Load a protein structure before highlighting atoms "
                "or measuring an angle."
            )

        # PDB files may expose chain labels through either segid
        # or chainID in MDAnalysis. Resolve the valid form from the
        # currently loaded structure before building atom selections.
        chain_selector = None

        for candidate in (
            f"segid {chain}",
            f"chainID {chain}",
        ):
            try:
                candidate_atoms = universe.select_atoms(candidate)
            except Exception:
                continue

            if len(candidate_atoms) > 0:
                chain_selector = candidate
                break

        if chain_selector is None:
            available_segids = sorted(
                {
                    str(value).strip()
                    for value in getattr(universe.atoms, "segids", [])
                    if str(value).strip()
                }
            )

            available_chainids = sorted(
                {
                    str(value).strip()
                    for value in getattr(universe.atoms, "chainIDs", [])
                    if str(value).strip()
                }
            )

            return (
                f"Chain {chain} was not found in the currently loaded structure. "
                f"Available segids: {available_segids or 'none'}; "
                f"available chainIDs: {available_chainids or 'none'}."
            )

        atom_selections = [
            f"{chain_selector} and resid {r1} and name CA",
            f"{chain_selector} and resid {r2} and name CA",
            f"{chain_selector} and resid {r3} and name CA",
        ]

        validation_errors = []

        for resid, selection in zip(
            (r1, r2, r3),
            atom_selections,
        ):
            try:
                atoms = universe.select_atoms(selection)
            except Exception as exc:
                validation_errors.append(
                    f"residue {resid}: selection error "
                    f"({type(exc).__name__}: {exc})"
                )
                continue

            if len(atoms) == 0:
                validation_errors.append(
                    f"residue {resid}: no CA atom found in chain {chain}"
                )
            elif len(atoms) > 1:
                validation_errors.append(
                    f"residue {resid}: matched {len(atoms)} CA atoms "
                    f"in chain {chain}"
                )

        if validation_errors:
            return (
                "The requested angle could not be measured because "
                "the atom selections were not valid:\n- "
                + "\n- ".join(validation_errors)
            )

        highlight_expression = (
            f"{chain_selector} and "
            f"(resid {r1} or resid {r2} or resid {r3}) "
            f"and name CA"
        )

        selection_name = (
            f"angle_ca_{chain}_{r1}_{r2}_{r3}"
        )

        # Change the viewer only after all three atoms pass validation.
        try:
            highlight_result = tool_select(
                name=selection_name,
                expression=highlight_expression,
            )
        except Exception as exc:
            return (
                "The atoms were validated, but the viewer highlight failed: "
                f"{type(exc).__name__}: {exc}"
            )

        try:
            angle_result = tool_measure_angle(
                atom_selections[0],
                atom_selections[1],
                atom_selections[2],
            )
        except Exception as exc:
            return (
                f"{highlight_result}\n\n"
                f"The atoms were highlighted, but angle measurement failed: "
                f"{type(exc).__name__}: {exc}"
            )

        return (
            f"{highlight_result}\n\n"
            f"{angle_result}"
        )

    # Prompt did not match a deterministic workflow.
    # Preserve the existing PARORA agent behavior.
    return None


def _log(msg: str, level: int = logging.INFO) -> None:
    """
    Record one agent-loop event both ways: into the in-session debug panel
    (unchanged UI behavior, session-local, lost on restart) and into the
    persistent log file via the module logger (survives restarts/crashes,
    the same across every session, is what `tail -f logs/parora.log` or
    `docker logs` actually shows).
    """
    st.session_state.debug_logs.append(msg)
    log.log(level, msg)


def _trace(event: dict) -> None:
    """
    Record one tool-call decision of the current run_agent() call.

    st.session_state.agent_trace is reset at the start of every request and
    holds, in order, each call the model made: ran (with its final, gate-
    rewritten args and result), blocked by a gate, or skipped as a duplicate.
    Nothing in the UI reads it; evals/run.py scores the agent from it.
    """
    st.session_state.setdefault("agent_trace", []).append(event)


def _progress(status, msg: str) -> None:
    """
    Mirror one high-level agent-loop step into the live "working" dialog.

    `status` is an `st.status(...)` container passed down from the chat_input
    handler, or None when run_agent() is called without one (kept optional so
    nothing else calling run_agent has to change). Writing into it here is
    what turns an opaque multi-second freeze into a readable trace — the same
    events already going to `_log`, just the plain-language subset of them,
    surfaced while the screen is still locked instead of only afterward in
    the debug expander.
    """
    if status is not None:
        status.update(label=msg)
        status.write(msg)


# ── Plan-then-act (S6) ───────────────────────────────────────────────────────
# A long request ("load X, strip water, find the pocket, list H-bonds, render a
# figure, set up Amber") used to lose steps: the loop picks one tool per turn
# and every follow-up says "if the request is satisfied, stop", so the model
# stopped after two or three. Now a request with several actions gets a plan
# first — one JSON call — and the loop works through it: code ticks steps off
# from the calls that ran, the follow-up names the next open step instead of
# inviting a stop, and the reply ends with every step's outcome. Requests with
# one or two actions skip all of this and cost nothing extra.

PLAN_MIN_ACTIONS = 3
PLAN_MAX_STEPS = 10
PLAN_TURN_CAP = 40
_PLAN_SPLIT = re.compile(r"[,;]|\band then\b|\bthen\b|\band also\b|\balso\b|\band\b|\bafter that\b")
_PLAN_VERB = re.compile(
    r"\b(load|fetch|open|download|strip|remove|delete|find|list|detect|identify|show|"
    r"display|colou?r|paint|highlight|select|label|annotate|mark|zoom|focus|render|"
    r"ray ?trace|measure|compute|calculate|save|export|write|superpose|superimpose|"
    r"align|overlay|compare|prepare|clean|protonate|set up|setup|build|parameteri[sz]e|"
    r"predict|tell me|describe|summari[sz]e|hide|embed|orient|dock|search|look up|"
    r"make (a|an) (figure|image|picture))\b")
# Tools that do the same job for a plan step: the planner may say `select`
# where the model then uses `highlight`, or `find_interactions` where it runs
# `detect_salt_bridges`. A step is ticked by its own tool first, else by one
# of these. parameterize_ligand is deliberately not with setup_amber — it is
# a prerequisite, not the build.
_PLAN_EQUIV = [
    {"fetch_structure", "add_structure", "replace_scene", "load_protein", "load_local"},
    {"show", "show_all", "highlight", "select", "color"},
    {"find_interactions", "detect_salt_bridges", "detect_hydrogen_bonds"},
    {"find_contacts", "nearby_residues", "select_within", "detect_contacts"},
    {"measure_distance", "measure_mda_distance"},
    {"measure_angle", "measure_mda_angle"},
    {"measure_dihedral", "measure_mda_dihedral"},
    {"describe_structure", "summarize_chains", "list_structures", "list_residues"},
    {"superpose_structures", "align_structures"},
]
_PLAN_MARK = {"pending": "☐", "done": "✓", "failed": "✗", "skipped": "–"}


def _plan_wanted(prompt: str) -> bool:
    """True when the request reads as PLAN_MIN_ACTIONS or more separate actions."""
    clauses = _PLAN_SPLIT.split(prompt.lower())
    return sum(1 for c in clauses if _PLAN_VERB.search(c)) >= PLAN_MIN_ACTIONS


def _tool_catalog() -> str:
    """One line per tool — name and the first sentence of its description."""
    lines = []
    for t in TOOLS:
        f = t["function"]
        first = re.split(r"(?<=\.)\s", f.get("description", "").strip(), maxsplit=1)[0]
        lines.append(f"- {f['name']}: {first[:140]}")
    return "\n".join(lines)


_PLAN_SCHEMA = {
    "type": "object",
    "properties": {"steps": {"type": "array", "items": {
        "type": "object",
        "properties": {"goal": {"type": "string"}, "tool": {"type": "string"},
                       "depends_on": {"type": "array", "items": {"type": "integer"}}},
        "required": ["goal", "tool"]}}},
    "required": ["steps"],
}


def _plan_request(user_prompt: str) -> dict | None:
    """
    Ask the model for an ordered plan of the request's steps (one JSON call).

    Returns {"request", "steps": [{n, goal, tool, deps, status, note}]} or None
    when planning failed or found fewer than two steps — the loop then runs
    exactly as it does without a plan.
    """
    system = (
        "You plan the steps of a protein-structure request before any tool runs. "
        "Return JSON: {\"steps\": [{\"goal\", \"tool\", \"depends_on\"}]}. Rules: "
        "one step per action the user asked for, in the user's order unless one "
        "step needs another's result first; never add a step the user did not ask "
        "for (no extra describe, hide, save or render); `goal` restates that action "
        "in the user's words with their numbers, names and chains, nothing invented; "
        "`tool` is the one tool from the list that does it, or \"\" if it is answered "
        "in words; `depends_on` lists only the step numbers (from 1) whose OUTPUT this "
        "step uses — a structure it loads or prepares, a selection, file or list it "
        "creates. Steps that merely come later do not depend on each other; showing, "
        "colouring, highlighting, labelling, zooming and rendering never depend on "
        "one another. A step that needs a value "
        "the user did not give (a ligand charge, a residue) is still listed; its tool "
        "will ask. Structures already in the Scene need no load step.")
    user = (f"{_state_block()}\n\nTools:\n{_tool_catalog()}\n\n"
            f"Request: {user_prompt}")
    try:
        response = ollama_client.chat(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            format=_PLAN_SCHEMA,
            options={**OLLAMA_OPTIONS, "temperature": 0},
            keep_alive=KEEP_ALIVE,
            think=THINK,
        )
        raw = json.loads(response.get("message", {}).get("content", "") or "{}")
    except Exception as e:
        _log(f"⚠️ Planning failed ({e}) — running without a plan", logging.WARNING)
        return None
    # Plan text rides in the user messages, which the number gate counts as
    # evidence — so a residue number the planner made up would pass as the
    # user's. Numbers in a goal must come from the request or the scene.
    known = set(re.findall(r"\d+", f"{user_prompt}\n{_state_block()}"))
    steps = []
    for s in (raw.get("steps") or [])[:PLAN_MAX_STEPS]:
        if not isinstance(s, dict) or not str(s.get("goal", "")).strip():
            continue
        tool = str(s.get("tool") or "").strip()
        n = len(steps) + 1
        goal = re.sub(r"\d+", lambda m: m.group() if m.group() in known else "?",
                      str(s["goal"]).strip())
        steps.append({
            "n": n,
            "goal": goal[:160],
            "tool": tool if tool in TOOL_DISPATCH else "",
            "deps": sorted({int(d) for d in (s.get("depends_on") or [])
                            if isinstance(d, (int, float)) and 0 < int(d) < n}),
            "status": "pending",
            "note": "",
        })
    if len(steps) < 2:
        return None
    return {"request": user_prompt, "steps": steps}


def _render_plan(plan: dict) -> str:
    """The plan as a markdown checklist: ✓ done, ✗ failed, – skipped, ☐ open."""
    lines = []
    for s in plan["steps"]:
        line = f"{_PLAN_MARK[s['status']]} {s['n']}. {s['goal']}"
        if s["note"]:
            line += f" — {s['note']}"
        lines.append(line)
    return "\n".join(lines)


def _plan_step_for(plan: dict, name: str) -> dict | None:
    """
    The plan step a call of `name` serves: the earliest open (or failed, so a
    retry counts) step for that exact tool, else for an equivalent one. None
    means a helper call that serves no step.
    """
    open_steps = [s for s in plan["steps"] if s["status"] in ("pending", "failed")]
    step = next((s for s in open_steps if s["tool"] == name), None)
    if step is None:
        family = set().union(*(g for g in _PLAN_EQUIV if name in g)) or {name}
        step = next((s for s in open_steps if s["tool"] in family), None)
    return step


def _advance_plan(plan: dict, name: str, result: ToolResult) -> None:
    """
    Tick the step the call that ran serves (see _plan_step_for). A failed
    step skips the steps that depend on it; when a retry succeeds they open
    again.
    """
    step = _plan_step_for(plan, name)
    if step is None:
        # A helper that worked may have fixed what a failed step lacked
        # (parameterize_ligand before setup_amber): one more retry each.
        if result.ok:
            for s in plan["steps"]:
                if s["status"] == "failed":
                    s["tries"] = min(s.get("tries", 0), 1)
        return
    step["tries"] = step.get("tries", 0) + 1
    blocked_note = f"needs step {step['n']}, which failed"
    if result.ok:
        step["status"], step["note"] = "done", result.summary.strip().splitlines()[0][:120] \
            if result.summary.strip() else ""
        for s in plan["steps"]:
            if s["status"] == "skipped" and s["note"] == blocked_note:
                s["status"], s["note"] = "pending", ""
    else:
        step["status"], step["note"] = "failed", result.summary.splitlines()[0][:140]
        # A tool the error itself names as the fix ("Call parameterize_ligand
        # for each") — offered once even when the retries are spent.
        step["fix"] = next((t for t in re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", result.summary)
                            if t in TOOL_DISPATCH and t != step["tool"]), "")
        for s in plan["steps"]:
            if s["status"] == "pending" and step["n"] in s["deps"]:
                s["status"], s["note"] = "skipped", blocked_note


def _plan_follow_up(plan: dict) -> str:
    """What the model is told after a tool batch while a plan is running."""
    open_steps = [s for s in plan["steps"] if s["status"] == "pending"]
    head = f"Plan progress:\n{_render_plan(plan)}\n\n"
    fixable = [s for s in plan["steps"] if s["status"] == "failed" and s.get("fix")
               and not s.get("fix_offered")]
    if fixable:
        f = fixable[0]
        f["fix_offered"], f["tries"] = True, min(f.get("tries", 0), 1)
        return head + (
            f"Step {f['n']} failed: {f['note']}. The error names `{f['fix']}` as the fix. "
            f"If the user's request (including any clarification) gives what `{f['fix']}` "
            f"needs, call it now, then retry step {f['n']}; if it needs a value the user "
            "has not given, do not invent one — say what is needed in the final reply.")
    retry = [s for s in plan["steps"] if s["status"] == "failed" and s.get("tries", 0) < 2]
    if retry:
        r = retry[0]
        return head + (
            f"Step {r['n']} failed: {r['note']}. Retry it once now with arguments corrected "
            "from that error and from the results above (use names and selections a tool "
            "actually returned — never a guessed name). If the error shows it cannot work, "
            "do not retry; go on with the other steps and say why in the final reply.")
    if not open_steps:
        return head + (
            "Every plan step has run, failed or been skipped. Do not call more tools. "
            "Reply with a plain-text summary that covers EVERY step in order: what it "
            "found or produced (with the tool's numbers), or why it failed or was skipped.")
    nxt = open_steps[0]
    return head + (
        f"Next: step {nxt['n']} — {nxt['goal']}"
        + (f" (expected tool: `{nxt['tool']}`)" if nxt["tool"] else "")
        + ". Call the tool for it now; later steps that do not need its result may "
        "run in the same turn. If a result changes what a later step needs, adapt "
        "that step. If a step cannot be done, say so and why in the final reply — "
        "never claim it was done. Reply in plain text only when every step is done, "
        "failed or skipped, covering each one.")


def run_agent(user_prompt: str, status=None) -> str:
    """
    Execute a gated, deduplicated multi-turn tool-calling loop for one user command.

    The loop runs up to MAX_TURNS iterations. On each turn it:
      1. Sends the message history to Ollama, with the tool schemas routed to
         this message (_route_tools) and the system message held byte-stable so
         the prefix stays cached; the live scene rides in the newest user
         message instead (_state_block).
      2. Inspects any tool calls returned and applies the following gates:
         - select gate  : blocked when a representation show is active/pending
         - destructive  : hide/hide_all blocked unless the user asked to hide
         - load/search  : blocked when a structure is already loaded
         - write        : save/align/remove_solvent blocked unless explicitly requested
      3. Deduplicates exact repeat calls and redundant ball+stick highlights.
      4. After a non-ball+stick show fires, cuts the loop short and asks for
         a one-sentence confirmation instead of more tool calls.

    Args:
        user_prompt: Natural-language command from the chat input.
        status: optional `st.status(...)` container to narrate progress into
            while this runs — see `_progress`. None runs silently, same as
            before this parameter existed.

    Returns:
        Final agent text summary or a concatenation of tool result strings.
    """
    _log(f"📨 User: {user_prompt}")
    _progress(status, "Reading your request…")
    st.session_state.agent_trace = []
    st.session_state.agent_plan = None
    st.session_state.request_no = st.session_state.get("request_no", 0) + 1

    # A reply to the question asked last turn: turn "2" back into the original
    # request with the chosen meaning spelled out, so every gate and the model
    # see what the user actually wants.
    pending = st.session_state.pop("clarify", None)
    paused_plan = st.session_state.pop("plan_paused", None)
    clarify_note, clarified = "", False
    if pending:
        choice = resolve_clarification(pending, user_prompt)
        if not choice and not pending["options"] and len(user_prompt.split()) <= 8:
            # A short answer to an open question ("2W72") is that answer.
            choice = {"meaning": user_prompt.strip()}
        # Same mid-plan (S6), even with options: "net charge 0" answers "What
        # is the net charge of AQ4? 1. Provide the net charge 2. Skip".
        if (not choice and paused_plan and paused_plan["request"] == pending["request"]
                and len(user_prompt.split()) <= 8
                and not _PLAN_VERB.search(user_prompt.lower())):
            choice = {"meaning": user_prompt.strip()}
        if choice:
            clarified = True
            user_prompt = f"{pending['request']} — clarification: {choice['meaning']}"
            _log(f"🧭 Clarified request: {user_prompt}")
        else:
            clarify_note = (
                f"Last turn you asked the user: \"{pending['question']}\" with the options "
                + "; ".join(f"({i}) {o['meaning']}" for i, o in enumerate(pending["options"], 1))
                + f", about their request \"{pending['request']}\". If the newest message "
                "answers that, carry out the original request with the meaning they chose; "
                "if it is a new request, handle that instead.\n\n")
    # S6: the model asked in plain words ("what net charge for AQ4?") with a
    # plan unfinished — a short reply answers it, same as an open question.
    if (not pending and paused_plan and paused_plan.get("asked_in_text")
            and len(user_prompt.split()) <= 8 and not _PLAN_VERB.search(user_prompt.lower())):
        pending = {"request": paused_plan["request"]}
        user_prompt = f"{paused_plan['request']} — clarification: {user_prompt.strip()}"
        clarified = True
        _log(f"🧭 Answer to the plan's question: {user_prompt}")
    st.session_state.current_request = user_prompt

    prompt_lower = user_prompt.lower()

    if not clarified:
        question = _ambiguity_question(user_prompt)
        if question:
            _log(f"❔ Ambiguous request — asking: {question}")
            _progress(status, "Your request can mean more than one thing — asking.")
            return question

    # Narrow deterministic workflow pre-router.
    # Unmatched prompts continue through the original PARORA agent.
    deterministic_result = handle_deterministic_workflow(user_prompt)
    if deterministic_result is not None:
        _progress(status, "Handled by a fast-path workflow, no model call needed.")
        return deterministic_result


    # ------------------------------------------------------------
    # Stage 1A lightweight intent validation
    # ------------------------------------------------------------
    # The angle atom-count check, the sel1=/sel2= MDAnalysis-distance
    # shortcut, and the salt-bridge shortcut used to live here as hard-coded
    # regex (P4 in todo.txt). They're retired in favor of retrieved few-shot
    # grounding (rag_grounding.py) showing the model the same cases —
    # verified against the real qwen2.5:7b + full tool schemas before removal:
    # given an ambiguous 2-residue "angle between X and Y" prompt, the model
    # consistently asks for the missing vertex instead of inventing one.
    #
    # The dihedral atom-count check stays, unlike its angle counterpart:
    # the same test showed qwen2.5:7b reliably invents the two missing
    # residues for an underspecified "dihedral between X and Y" prompt
    # (e.g. treating "between 5 and 8" as the range 5,6,7,8) and reports a
    # confident-sounding but fabricated angle, even with an exact-match
    # grounding example present. Revisit only with evidence this stops
    # happening — a hallucinated number is worse than a blocked call.
    if "dihedral" in prompt_lower:
        residue_numbers = re.findall(r"\b\d+\b", user_prompt)

        if len(residue_numbers) < 4:
            return (
                "I need four atoms or residues to measure a dihedral angle. "
                "Please specify the four atoms or residues."
            )

    # Kept, with evidence (P4 in todo.txt): tested removing this in favor of
    # the system prompt + RAG grounding (rag_grounding.py has two stability
    # examples) alone. Most phrasings degraded gracefully, but "what's the
    # folding stability of this protein" made qwen2.5:7b tally salt
    # bridges/H-bonds/disulfides via find_interactions and conclude "1CRN
    # appears structurally robust" — exactly the overconfident static-
    # structure stability verdict this message exists to prevent, its own
    # hedge about needing MD simulations notwithstanding. A single bad
    # phrasing producing a false claim of assessed stability is worse than
    # every phrasing correctly triggering a hard-coded refusal, so this
    # stays a keyword gate rather than a prompt-only rule.
    # P8 fix (todo.txt): the bare word "folding" wrongly caught classification
    # questions like "describe the folding topology of this domain" (confirmed
    # by direct test) — a describe_fold question, not a dynamics question.
    # Narrowed to require a dynamics-flavored word nearby, so it no longer
    # collides with the refusal this gate exists for; "stable"/"stability"/
    # "thermostable" stay unconditional triggers exactly as P4 proved they
    # need to be — only "folding" was ever the false-positive-prone one.
    folding_is_dynamics = "folding" in prompt_lower and any(
        w in prompt_lower for w in
        ("correctly", "properly", "will", "process", "pathway")
    )
    # "fold" as a verb is the same dynamics question ("will this protein fold
    # correctly" skipped the gate and was only declined thanks to a RAG
    # example); "fold" as a noun ("what fold is this", "does this have a
    # known fold") must not match.
    folding_is_dynamics = folding_is_dynamics or bool(re.search(
        r"\bmisfold|\bfolds?\s+(correctly|properly|incorrectly|at all|in (water|"
        r"solution|vivo|vitro|the cell))\b|\b(will|does|do|would|can|could|should)\s+"
        r"(?:\w+\s+){1,2}fold\b(?!\s*(class|type|famil|topolog))", prompt_lower))
    # P11 split (todo.txt): a stability question that names a mutation
    # ("would mutating residue 45 to alanine destabilize this", "is H92A
    # destabilizing") has something to compute against, so it gets P10's
    # ESM-2 score framed as a stability proxy — answered here, without the
    # model, since qwen2.5:7b already turned that score into "a significant
    # decrease in stability" once. Nothing named → the refusal, unchanged.
    # "(de)stabilize" joins only the mutation route: it contains no "stable",
    # and refusing on it would catch "which salt bridges stabilize the dimer".
    stability_asked = folding_is_dynamics or any(x in prompt_lower for x in (
        "stable",
        "stability",
        "thermostable",
    ))
    if stability_asked or re.search(r"stabili[sz]", prompt_lower):
        mutations = _stability_mutations(user_prompt)
        if mutations:
            _log(f"🧪 Stability question names mutation(s) {mutations} — ESM-2 proxy estimate")
            _progress(status, "Estimating the mutation's effect with ESM-2…")
            return _stability_estimate(mutations)
    if stability_asked:
        return (
            "Protein stability cannot be determined from a single static "
            "PDB structure alone. Determining stability generally requires "
            "molecular dynamics simulations, free-energy calculations, "
            "or experimental measurements. "
            "I can instead analyze structural contacts, salt bridges, "
            "hydrogen bonds, B-factors, or prepare the structure for "
            "simulation. For a specific mutation (e.g. \"would mutating "
            "residue 45 to alanine destabilize it\") I can give a "
            "sequence-based ESM-2 estimate — a proxy, not a ΔΔG."
        )

    # The system message is built once and never touched again — see
    # _state_block() for why that matters to the prefill cost. Retrieved
    # few-shot grounding (rag_grounding.py) rides in the volatile user
    # message alongside the scene state, same reasoning as _state_block().
    # S6: a request with several actions gets a plan before any tool runs. A
    # plan paused by a question last turn resumes where it stopped: its done
    # steps stay done, so the load or analysis before the question is not redone.
    plan = None
    if clarified and paused_plan and pending and paused_plan["request"] == pending["request"]:
        plan = {k: v for k, v in paused_plan.items() if k != "asked_in_text"}
        plan["request"] = user_prompt
        for s in plan["steps"]:
            if s["status"] != "done":
                s["status"], s["note"], s["tries"] = "pending", "", 0
        _log(f"🗺️ Resuming plan after the answer:\n{_render_plan(plan)}")
    elif _plan_wanted(user_prompt):
        _progress(status, "Planning the steps…")
        plan = _plan_request(user_prompt)
        if plan:
            _log(f"🗺️ Plan:\n{_render_plan(plan)}")
    st.session_state.agent_plan = plan
    plan_slot = status.empty() if (plan and status is not None) else None

    def _show_plan() -> None:
        if plan_slot is not None:
            plan_slot.markdown("**Plan**\n\n" + _render_plan(plan).replace("\n", "  \n"))

    _show_plan()
    plan_block = ""
    if plan:
        done = [s for s in plan["steps"] if s["status"] == "done"]
        plan_block = (
            "Plan for this request — work through every step in order; steps that do "
            "not need an earlier result may run in the same turn:\n"
            + "\n".join(f"{s['n']}. {s['goal']}"
                        + (f" [`{s['tool']}`]" if s["tool"] else "")
                        + (" — ALREADY DONE, do not repeat" if s["status"] == "done" else "")
                        for s in plan["steps"])
            + ("\n\n" if not done else
               "\nSteps marked done ran before the user's answer; their results are in "
               "the Scene and Remembered results.\n\n"))

    grounding = format_grounding(user_prompt)
    grounding_block = f"{grounding}\n\n" if grounding else ""
    history = _history_block()
    history_block = f"{history}\n\n" if history else ""
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": (f"{_state_block()}\n\n{grounding_block}{history_block}"
                                     f"{clarify_note}{plan_block}Newest request: {user_prompt}")}
    ]

    # Every tool call that ran this request, in order: (name, ToolResult).
    # The guards below read these results' ok/data, not the message text (S4).
    ran: list[tuple[str, ToolResult]] = []

    def _ran(name: str) -> list:
        return [r for n, r in ran if n == name]

    def _request_evidence() -> str:
        # What this request actually saw: the user's words, scene, history and
        # tool results — minus the retrieved few-shot examples. Their numbers
        # belong to other prompts: "what happens if I mutate it to alanine"
        # ran predict_mutation_effect(residue='45') off the "mutating residue
        # 45" example, and the number gate counted 45 as user-supplied (S2a).
        # Plus each result's data: values a summary trimmed ("…and 12 more")
        # are still evidence the tool produced.
        text = "\n".join(str(m.get("content", "")) for m in messages
                         if m.get("role") == "user")
        text = text.replace(grounding, "") if grounding else text
        return "\n".join([text] + [json.dumps(r.data, default=str) for _, r in ran if r.data])

    def _plan_footer() -> str:
        # Every plan step's outcome, from code, not the model: a step the reply
        # skipped over still shows as done, failed or not run (S6).
        if not plan:
            return ""
        for s in plan["steps"]:
            if s["status"] == "pending":
                s["status"] = "done" if not s["tool"] else "skipped"
                s["note"] = "answered above" if not s["tool"] else "not run"
        _show_plan()
        return f"\n\n**Plan**\n{_render_plan(plan)}"

    active_tools, tools_are_subset = TOOLS, False
    _log(
        f"🧰 {len(active_tools)}/{len(TOOLS)} tool schemas sent")

    # Gate: destructive tools only when user explicitly asked for hiding
    DESTRUCTIVE_TOOLS = {"hide_all", "hide"}
    hide_requested = any(w in prompt_lower for w in {"hide", "clear", "remove", "delete", "clean", "erase"})
    # "reset the view/scene" may mean start over; "reset the colors to
    # default" does not — qwen2.5:14b answered it with hide_all + show
    # cartoon, wiping every rep the user had drawn (S2a).
    if "reset" in prompt_lower and not re.search(
            r"colou?r|transparen|opacity|zoom|camera|orientation|label", prompt_lower):
        hide_requested = True

    # Gate: write/side-effect tools only when explicitly requested by the user
    WRITE_TOOLS = {"save_structure", "remove_solvent", "align_structures"}
    # "remove solvent"/"remove water" as exact phrases missed the equally
    # natural "remove the solvent" / "remove all water" — found while
    # testing this gate for P5 (todo.txt): match the action verb and the
    # object as separate words instead of one rigid phrase.
    write_requested = (
        any(w in prompt_lower for w in {"save", "write", "align", "no water"})
        or (
            any(v in prompt_lower for v in {"remove", "strip"})
            and any(o in prompt_lower for o in {"solvent", "water"})
        )
    )

    # Gate: Foldseek's two costly options are the user's call, never the
    # model's — uploading their coordinates to a third-party server, and a
    # multi-GB download. Both need the user's own words in this message.
    foldseek_online_requested = bool(re.search(
        r"\bonline\b|\bweb\b|upload|foldseek\.com|\bremote\b|\bserver\b|"
        r"option (1|one)|first option|\(1\)", prompt_lower))
    foldseek_download_requested = bool(re.search(
        r"download|install|local (db|database)|option (2|two)|second option|\(2\)",
        prompt_lower))

    # Each turn is a full round trip to the model: one to pick tools, one more
    # to read their results and either summarize or call more. A single-tool
    # request now costs two turns and a load-then-style request four.
    # A plan gets a budget per step instead (S6): 3 turns each, with a hard cap.
    MAX_TURNS = min(PLAN_TURN_CAP, max(16, 3 * len(plan["steps"]) + 4)) if plan else 16
    plan_nudges = 0                        # Retries for plan steps left unrun at the reply
    summary_parts = []
    called_sigs: set[str] = set()          # Tracks (name, args) pairs to avoid exact repeats
    failed_sigs: dict[str, str] = {}       # sig -> result, for calls that failed
    selected_ngl_strs: set[str] = set()    # Tracks NGL strings that already have a highlight
    show_rep_fired = False                 # True once any non-ball+stick show has executed
    protein_nudged = False                 # One retry for a tool-less protein-level answer
    fact_checked = False                   # One correction pass for unsupported facts
    disulfide_nudged = False               # One retry for a disulfide answer with no finder run
    zoom_nudged = False                    # One retry for a zoom request with no zoom call
    user_chains = {c.upper() for c in re.findall(r"\bchain\s+([A-Za-z0-9])\b", user_prompt, re.I)}

    for turn in range(MAX_TURNS):
        _progress(
            status,
            "Thinking…" if turn == 0 else f"Thinking about next step (turn {turn + 1})…",
        )
        try:
            response = ollama_client.chat(
                model=MODEL,
                messages=messages,
                tools=active_tools,
                options=OLLAMA_OPTIONS,
                keep_alive=KEEP_ALIVE,
                think=THINK,
            )
        except ConnectionError:
            err = f"Cannot reach Ollama at {OLLAMA_HOST}. Start Ollama with `ollama serve`."
            _log(f"❌ {err}", logging.ERROR)
            return err
        except Exception as e:
            err = f"Ollama error: {e}"
            _log(f"❌ {err}", logging.ERROR)
            log.exception("Unhandled error calling Ollama")
            return err

        msg = response.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        # No tool calls → agent is done; return its text reply. Unless this was
        # the first turn on a routed subset and nothing has run yet, in which
        # case the tool it wanted may simply not have been offered — retry once
        # with the full list rather than answering from a half-set menu.
        if not tool_calls:
            final_text = msg.get("content", "").strip()
            if tools_are_subset and not summary_parts:
                _log(
                    "↻ No tool chosen from the routed subset — retrying with all "
                    f"{len(TOOLS)} schemas")
                active_tools, tools_are_subset = TOOLS, False
                continue
            # A question about the protein beyond the loaded file, answered
            # with no lookup at all: qwen2.5:7b does this about half the time
            # for "I mean the protein, not the scene", reciting the loaded
            # file's chains as if they were the protein's other structures.
            # One nudge towards the tool that actually answers it.
            if (not summary_parts and not protein_nudged and st.session_state.focus
                    and PROTEIN_LEVEL_RE.search(prompt_lower)
                    and not re.search(r"\b(colou?r|show|hide|select|highlight|label|zoom|"
                                      r"superpose|align|measure|remove)\b", prompt_lower)):
                protein_nudged = True
                _log("↻ Protein-level question answered with no tool — nudging to protein_structures")
                messages.append({"role": "assistant", "content": msg.get("content", "")})
                messages.append({"role": "user", "content": (
                    "That answer used no tool. The question is about the protein's structures "
                    "in the PDB beyond the loaded file. Call `protein_structures` (no filters) "
                    "and answer from its result, grouped by region.")})
                continue
            # Disulfides come from find_interactions only. For "find the
            # disulfide bonds" on 1CRN, qwen2.5:14b ran the salt-bridge and
            # H-bond screens, read the SG–SG line of the H-bond table and
            # reported 2 of the 3 bonds; the fact check passed it, since
            # CYS3/CYS40 were in that table (S2a).
            # A reply that only repeats a tool's own disulfide count (setup_amber,
            # prepare_structure) is not a disulfide answer: nudging it replaced
            # "built ... 6 disulfide(s)" with a wrong "no disulfides" (S3 eval).
            disulfide_re = r"disulf|disulph|\bs-s\b|\bss[ -]bond|cystine"
            tool_said_disulfide = any(k.startswith("disulfide") for _, r in ran for k in r.data)
            if (not disulfide_nudged and st.session_state.pdb_id
                    and (re.search(disulfide_re, prompt_lower)
                         or (re.search(disulfide_re, final_text.lower())
                             and not tool_said_disulfide))
                    and not _ran("find_interactions")):
                disulfide_nudged = True
                _log("↻ Disulfide answer with no find_interactions run — nudging")
                messages.append({"role": "assistant", "content": final_text})
                messages.append({"role": "user", "content": (
                    "Disulfide bonds are identified only by `find_interactions` "
                    "(types='disulfide'); the hydrogen-bond and salt-bridge screens do not "
                    "detect them. Call find_interactions(types='disulfide') and answer from "
                    "its result only. Do not mention this note.")})
                continue
            # Zoom happens only through the zoom tool. For "fetch 1UBQ, show
            # lysine 48 as sticks and zoom to it" qwen2.5:14b ran fetch and
            # select, then replied "highlighted ... and zoomed to" (S3 eval).
            zoom_re = r"\bzoom(?:ed|ing|s)?\b"
            zoomed = (any(r.ok for r in _ran("zoom"))
                      or any(r.ok and r.data.get("zoomed") for r in _ran("highlight")))
            if (not zoom_nudged and st.session_state.pdb_id
                    and (re.search(zoom_re, prompt_lower)
                         or re.search(zoom_re, final_text.lower()))
                    and not zoomed):
                zoom_nudged = True
                _log("↻ Zoom requested or claimed with no zoom call — nudging")
                messages.append({"role": "assistant", "content": final_text})
                messages.append({"role": "user", "content": (
                    "The camera moves only through the `zoom` tool, and it was not called. "
                    "Call zoom(selection=...) on what the user asked to zoom to, then "
                    "answer from the tool results. Do not mention this note.")})
                continue
            # Fact check: every PDB id, accession and decimal number in the
            # reply has to appear in something this request actually saw — the
            # scene, the working context, the user's words or a tool result.
            # One chance to correct it; after that, flag what is unverified
            # rather than pass it off as fact.
            # S6: a plan step that never ran is not silently dropped. Twice at
            # most, the model is sent back to it; whatever is still open after
            # that is reported as not done in the checklist below.
            unrun = ([s for s in plan["steps"] if s["status"] == "pending" and s["tool"]]
                     if plan else [])
            if unrun and plan_nudges < 2 and turn < MAX_TURNS - 2:
                plan_nudges += 1
                _log(f"↻ Plan steps {[s['n'] for s in unrun]} never ran — nudging")
                messages.append({"role": "assistant", "content": final_text})
                messages.append({"role": "user", "content": (
                    "These plan steps have not run: "
                    + "; ".join(f"{s['n']}. {s['goal']}"
                                + (f" (`{s['tool']}`)" if s["tool"] else "") for s in unrun)
                    + ". Call the tools for them now. If one cannot be done, or is "
                    "answered in words, reply covering every step and say plainly which "
                    "ones were not done and why. Do not mention this note.")})
                continue
            unsupported = _unsupported_facts(final_text, _request_evidence())
            if unsupported and not fact_checked:
                fact_checked = True
                _log(f"🔎 Reply states unsupported facts {unsupported} — asking for a correction")
                _progress(status, "Double-checking the answer against the data…")
                messages.append({"role": "assistant", "content": final_text})
                messages.append({"role": "user", "content": (
                    f"Check failed: your reply states {', '.join(unsupported)}, which appear in "
                    "no tool result, Scene block or working context of this request. Rewrite "
                    "the reply using only facts from those. If you need data you do not have, "
                    "call the tool that provides it; if no tool can, say you cannot tell. Do "
                    "not mention this check.")})
                continue
            _log(f"💬 Agent: {final_text}")
            _progress(status, "Composing reply…")
            reply = _clean_reply(final_text) or ("Done: " + "; ".join(summary_parts))
            if unsupported:
                _log(f"⚠️ Unverified facts left in reply: {unsupported}", logging.WARNING)
                reply += ("\n\n_Not verified against any tool result: "
                          + ", ".join(unsupported) + " — treat with caution._")
            # P10: the 7B model paraphrases the ESM result without its caveats
            # and has called it "a significant decrease in stability" — so the
            # standard caveat rides on every reply that used the prediction.
            if any(r.ok and r.data.get("model_prediction")
                   for r in _ran("predict_mutation_effect")):
                reply += ("\n\n_ESM-2 model prediction from sequence statistics "
                          "(zero-shot masked-marginal score) — not an experimental "
                          "measurement and not a stability (ΔΔG) estimate._")
            # P13: same paraphrase risk for a function inferred from family
            # and domain matches — it must never read as the known function.
            pf = [r.data for r in _ran("protein_function") if r.ok]
            if any(d.get("inferred") for d in pf):
                basis = "sequence-family, domain and electronic GO matches"
                if any(d.get("structural_neighbours") for d in pf):
                    basis += ", or from structurally similar proteins (Foldseek)"
                reply += ("\n\n_UniProt has no curated function for this protein. Any role "
                          f"above marked as inferred comes from {basis} — a low-confidence "
                          "computational guess, not an established function._")
            elif any(d.get("automatic") for d in pf):
                reply += ("\n\n_This function text is UniProt's automatic (rule-based) "
                          "annotation for an unreviewed entry, not curator-reviewed._")
            # P9 follow-up: qwen2.5:7b called AlphaFold DB hits "known
            # structures" and dropped that no PDB entry matched — so the
            # provenance rides on every reply that used them.
            afdb = [r.data for n in ("find_structural_neighbors", "describe_fold")
                    for r in _ran(n) if "afdb_neighbours" in r.data]
            if afdb:
                no_pdb = any(d.get("no_pdb_match") for d in afdb)
                reply += ("\n\n_" + ("No experimental (PDB) structure matched confidently. "
                                     if no_pdb else "")
                          + "The AlphaFold DB neighbours are predicted models of reviewed "
                          "UniProt entries, not experimental structures, and have no "
                          "CATH/SCOP fold classification._")
            question = st.session_state.pop("foldseek_question", None)
            if question and any(r.data.get("needs_user_choice") for _, r in ran):
                reply = f"{reply}\n\n{question}"
            # P12: qwen2.5:7b paraphrased find_pockets' report into wrong
            # counts ("four druggable", listing three), wrong chains and
            # "occupied by hemoglobin subunits" for heme. A pocket-only
            # request gets the report itself; anything more keeps the
            # model's reply with the report's SUMMARY line attached verbatim.
            pocket_reports = [r for r in _ran("find_pockets") if r.ok]
            if pocket_reports:
                # A failed call that led to find_pockets (find_contacts on
                # 'pocket 30') still leaves a pocket-only answer.
                if all(n == "find_pockets" for n, r in ran if r.ok):
                    return pocket_reports[-1].summary + _plan_footer()
                if pocket_reports[-1].data.get("summary"):
                    reply += f"\n\n_fpocket: {pocket_reports[-1].data['summary']}_"
            # Same for salt bridges: the model miscounted pairs and called
            # same-chain pairs inter-chain, so the tool's own count rides along.
            sb = [r.data for r in _ran("detect_salt_bridges") if r.ok and r.data.get("summary")]
            if sb:
                reply += f"\n\n_Salt-bridge detector: {sb[-1]['summary']}_"
                # Short pair lists ride along too — asked to "list the salt
                # bridges between chains", the model repeated the count only.
                for label, key in (("Within one chain", "within"), ("Between chains", "between")):
                    if 0 < len(sb[-1][key]) <= 12:
                        reply += f"\n_{label}: {', '.join(sb[-1][key])}._"
            # Same for an Amber build: force field, charge and disulfide count.
            amber = [r.data for r in _ran("setup_amber") if r.ok and r.data.get("summary")]
            if amber and amber[-1]["summary"] not in reply:
                reply += f"\n\n_tleap: {amber[-1]['summary']}_"
            if (plan and "?" in final_text
                    and any(s["status"] != "done" and s["tool"] for s in plan["steps"])):
                st.session_state.plan_paused = {**plan, "asked_in_text": True}
            return reply + _plan_footer()

        tool_results = []

        # Asking and acting in the same breath means the model is unsure what
        # to act on — ask only.
        asks = [tc for tc in tool_calls if tc.get("function", {}).get("name") == "ask_user"]
        if asks:
            tool_calls = asks[:1]

        # Check if this batch contains a non-ball+stick show (affects select gate below)
        batch_has_rep_show = any(
            tc.get("function", {}).get("name") == "show"
            and _tc_args(tc).get("rep_type", "") != "ball+stick"
            for tc in tool_calls
        )

        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = _tc_args(tc)

            # Deterministic fallback for explicitly requested chain scopes.
            # Local models may select the correct tool but omit the chain
            # argument even when the user clearly specifies one.
            if name == "detect_salt_bridges" and not str(args.get("chain", "")).strip():
                chain_match = re.search(
                    r"\bchain\s+([A-Za-z0-9]+)\b",
                    user_prompt,
                    flags=re.IGNORECASE,
                )
                if chain_match:
                    args["chain"] = chain_match.group(1).upper()
                    _log(
                        f"🧭 Injected explicit salt-bridge chain scope: "
                        f"{args['chain']}"
                    )

            # Structure-list filters only when the user asked for them. With
            # chat history in context, qwen2.5:7b carried "7MN5 is Cryo-EM"
            # into a follow-up as method='Cryo-EM', max_resolution=3.5 and
            # silently hid 49 of HER2's 63 structures.
            if name == "protein_structures":
                unasked = []
                if args.get("method") and not re.search(
                        r"x-?ray|crystal|nmr|cryo|electron|\bem\b|method|techni", prompt_lower):
                    unasked.append("method")
                if args.get("max_resolution") and not re.search(
                        r"resolution|å|\bangstrom|\d\s*a\b|high[- ]res|sharp", prompt_lower):
                    unasked.append("max_resolution")
                if args.get("ligands_only") and not re.search(
                        r"ligand|bound|inhibitor|drug|compound|cofactor", prompt_lower):
                    unasked.append("ligands_only")
                for k in unasked:
                    args.pop(k, None)
                if unasked:
                    _log(f"🧭 Dropped unrequested protein_structures filters: {unasked}")

            # predict_mutation_effect: qwen2.5:7b dropped the substitution from
            # "mutating residue 44 to alanine" and "is H92A damaging" (sent
            # residue='44' / 'H92', no mutant), which silently turns a single-
            # mutation question into an all-19 scan; and it filled in chain
            # letters nobody said. Take both from the user's own words.
            # find_pockets: qwen2.5:7b sent chain='A' for "find the hidden
            # binding pocket" on single- and multi-chain structures alike,
            # which would silently drop every pocket not lined by chain A.
            # detect_salt_bridges: same habit, live — after a turn about chain B
            # it scoped "which salt bridges stabilize this protein" to chain B
            # and reported 9 pairs of 4HHB's 34.
            if name in ("find_pockets", "detect_salt_bridges"):
                ch = str(args.get("chain") or "").strip().upper()
                if ch and ch not in user_chains:
                    args.pop("chain")
                    _log(f"🧭 Dropped chain '{ch}' the user never named")
            if name == "predict_mutation_effect":
                if not str(args.get("mutant") or "").strip():
                    mut = _mutation_from_prompt(user_prompt, str(args.get("residue", "")))
                    if mut:
                        args["mutant"] = mut
                        _log(f"🧭 Injected mutant '{mut}' from the prompt")
                # The wild type the user wrote wins: for "is K48R tolerated"
                # qwen2.5:14b sent wildtype='L' (lysine → L), and the tool
                # rightly refused a claim the structure contradicts (S6 eval).
                num = re.search(r"\d+", str(args.get("residue", "")))
                said = [esm.parse_amino_acid(wt) for wt, n, mt in re.findall(
                            r"\b([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})\b", user_prompt)
                        if num and n == num.group()
                        and esm.parse_amino_acid(wt) and esm.parse_amino_acid(mt)]
                claimed = esm.parse_amino_acid(str(args.get("wildtype") or "")) or ""
                if said and claimed and claimed != said[0]:
                    args["wildtype"] = said[0]
                    _log(f"🧭 Wild type '{claimed}' replaced by the user's '{said[0]}'")
                ch = str(args.get("chain") or "").strip().upper()
                if ch and ch not in user_chains and not re.search(
                        rf"\b{re.escape(ch)}\s*[/:]\s*\d|\d\s*[/:]\s*{re.escape(ch)}\b",
                        user_prompt, re.I):
                    args.pop("chain")
                    _log(f"🧭 Dropped chain '{ch}' the user never named")

            # `show` replaces a layer of the same type and selection, colour
            # included. For a plain "load 1CRN" qwen2.5:14b added show(cartoon,
            # protein) with the default colour 'element', which swapped the
            # loaded rainbow cartoon for an element-coloured one nobody asked
            # for. Keep the drawn layer's colour unless the user named one.
            if name == "show" and not args.get("exclusive"):
                drawn = next((r for r in st.session_state.representations
                              if r["type"] == str(args.get("rep_type", "")).lower()
                              and r["selection"] == resolve_selection(args.get("selection", ""))),
                             None)
                want = str(args.get("color") or "element").lower()
                if (drawn and drawn.get("color") and drawn["color"] != want
                        and want not in prompt_lower
                        and not re.search(r"colou?r|paint|rainbow|spectrum", prompt_lower)):
                    args["color"] = drawn["color"]
                    _log(f"🧭 Kept the drawn {drawn['type']} colour '{drawn['color']}' "
                         f"(user named no colour)")

            # The user named one chain; a residue selection that dropped it
            # would act on that residue in every chain ("select residue 58 of
            # chain A" arrived as expression='resi 58').
            if len(user_chains) == 1:
                ch = next(iter(user_chains))
                if name == "select":
                    expr = str(args.get("expression", ""))
                    if re.search(r"\d", expr) and not re.search(
                            r"chain|segid|:[A-Za-z0-9]|/|@", expr, re.I):
                        args["expression"] = f"{expr} chain {ch}"
                        _log(f"🧭 Injected chain {ch} into select expression")
                elif name == "highlight":
                    tgt = str(args.get("target", ""))
                    if re.search(r"\d", tgt) and not re.search(r"chain|/|:", tgt, re.I):
                        args["target"] = ", ".join(f"{ch}/{p.strip()}"
                                                   for p in tgt.split(",") if p.strip())
                        _log(f"🧭 Injected chain {ch} into highlight target")

            # ── Gate: invented residue numbers ──────────────────────────────
            # "highlight the domain" came back as highlight(target='18-96'):
            # numbers nobody said and no tool returned. Any number in a
            # selection-like argument must come from the user's words, the
            # scene, or a tool result in this request; otherwise block and
            # have the model ask.
            if name not in NUMBER_GATE_EXEMPT:
                seen = set(re.findall(r"\d+", _request_evidence()))
                invented = sorted({n for k, v in args.items()
                                   if isinstance(v, str) and k not in NUMBER_GATE_FREE_ARGS
                                   for n in re.findall(r"\d+", v)} - seen, key=int)
                if invented:
                    _log(f"🚫 Blocked '{name}' — numbers {invented} not from user or any tool")
                    tool_results.append({"tool": name, "result": _blocked(
                        f"Blocked — {', '.join(invented)} did not come from the user or any "
                        "tool result. Do not guess residue numbers: call `ask_user` to ask "
                        "which residues/region they mean (offer concrete options if a tool "
                        "result lists them), or use a named selection without numbers.")})
                    continue

            # ── Gate: select ────────────────────────────────────────────────
            # Block select when: a show already fired this run, or a
            # rep-type show is in the same batch. The former third
            # condition — a keyword-based "show-only intent" guess — was
            # retired (P5 in todo.txt): the system prompt now states the
            # same rule directly and qwen2.5:7b follows it without a
            # hard block, verified across pure-"show" prompts, mixed
            # show+highlight prompts, and a "show the ligand" case.
            # With a plan, a select that serves its own open highlight step is
            # the user's request, not a stray add-on to a show (S6).
            plan_wants_select = bool(plan) and any(
                s["status"] == "pending" and s["tool"] in ("select", "highlight")
                for s in plan["steps"])
            if (name == "select" and (show_rep_fired or batch_has_rep_show)
                    and not plan_wants_select):
                _log(
                    f"🚫 Blocked 'select' — representation command; no highlight needed"
                )
                tool_results.append({"tool": name, "result": _blocked(
                    "Blocked — use 'show' for representations, 'select' for highlights.")})
                continue

            # ── Gate: destructive ───────────────────────────────────────────
            if name in DESTRUCTIVE_TOOLS and not hide_requested:
                _log(f"🚫 Blocked '{name}' — not requested")
                tool_results.append({"tool": name, "result": _blocked(
                    "Blocked — user did not request hiding.")})
                continue

            # The load/search gate (LOAD_TOOLS + a regex classifier for
            # "does this ambiguous verb mean load or analyze") was retired
            # (P5 in todo.txt): the system prompt's rule 0a1 (STAY ON THE
            # WORKING CONTEXT) already states this directly, and
            # qwen2.5:7b followed it without the regex backing it up —
            # verified across "find residues 1-500", "get the contacts",
            # "show me the binding site", "look up the salt bridges",
            # "search for hydrogen bonds" (none re-fetched the loaded
            # structure) and "load PDB 4HHB" (still loaded correctly).

            # ── Gate: write/side-effect ─────────────────────────────────────
            if name in WRITE_TOOLS and not write_requested:
                _log(f"🚫 Blocked '{name}' — not requested by user")
                tool_results.append({"tool": name, "result": _blocked(
                    "Blocked — user did not request this operation.")})
                continue

            # ── Gate: Foldseek online upload / database download ───────────
            if name == "find_structural_neighbors":
                entry_now = active_structure()
                if foldseek_online_requested and entry_now:
                    st.session_state.setdefault("foldseek_online_ok", set()).add(entry_now["path"])
                    args["where"] = "online"
                elif (args.get("where") == "online" and entry_now and entry_now["path"]
                      not in st.session_state.get("foldseek_online_ok", set())):
                    _log("🚫 Blocked online Foldseek — user did not agree to upload")
                    tool_results.append({"tool": name, "result": _blocked(
                        "Blocked — the user has not agreed to upload this structure. "
                        "Ask them: search online, or download the local database?")})
                    continue
            if name == "download_foldseek_database" and not foldseek_download_requested:
                _log("🚫 Blocked Foldseek database download — not requested")
                tool_results.append({"tool": name, "result": _blocked(
                    "Blocked — the user did not ask to download the database. Ask them first.")})
                continue

            # ── Plan: dependencies (S6) ─────────────────────────────────────
            # A step that needs an earlier step's result waits until that step
            # is done: in one batch the model highlighted 'heme_neighbors'
            # right after the select_within that failed to make it. Deferring
            # also when the dependency finished in the same batch serialised
            # every step (the planner chains them) and doubled latency, so a
            # blind guess there (setup_amber on '_PREP1CRN') is left to fail
            # and get its retry instead.
            step = _plan_step_for(plan, name) if plan else None
            if step:
                late = [d for d in plan["steps"] if d["n"] in step["deps"] and d["tool"]
                        and d["status"] != "done"]
                if late:
                    d = late[0]
                    _log(f"⏸ Deferred '{name}' — plan step {step['n']} needs step {d['n']}, "
                         f"which is {d['status']}")
                    tool_results.append({"tool": name, "result": _skipped(
                        f"Deferred — step {step['n']} needs step {d['n']}'s result, and step "
                        f"{d['n']} is {d['status']}. Do step {d['n']} first, then call this "
                        "again with arguments taken from its result.", ok=False)})
                    continue

            # ── Dedup: exact same call ──────────────────────────────────────
            sig = f"{name}:{json.dumps(args, sort_keys=True)}"
            if sig in called_sigs:
                _log(f"⏭ Skipped duplicate: {name}")
                # Worded as a failure when the first one failed: a bare
                # "Already called" let qwen2.5:14b report an ONIOM setup that
                # never ran as "completed successfully" (S3 eval).
                prior = failed_sigs.get(sig)
                tool_results.append({"tool": name, "result": (
                    _skipped(f"Skipped — this exact call already failed and nothing has "
                             f"changed since: {prior[:150]}", ok=False)
                    if prior else _skipped("Already called — skipped."))})
                continue

            # ── Dedup: ball+stick show for an already-selected NGL string ───
            if name == "show" and args.get("rep_type", "") == "ball+stick":
                ngl_candidate = resolve_selection(args.get("selection", ""))
                if ngl_candidate in selected_ngl_strs:
                    _log(f"⏭ Skipped 'show ball+stick' — already highlighted")
                    tool_results.append({"tool": name, "result": _skipped(
                        "Already highlighted by select — skipped.")})
                    continue

            # ── Dedup: select for an already-selected NGL string ────────────
            if name == "select":
                ngl_candidate = resolve_selection(args.get("expression", ""))
                if ngl_candidate in selected_ngl_strs:
                    _log(f"⏭ Skipped duplicate select — '{ngl_candidate}' already selected")
                    tool_results.append({"tool": name, "result": _skipped(
                        "Already selected — skipped.")})
                    continue

            called_sigs.add(sig)

            # Dispatch to the tool function
            dispatch = TOOL_DISPATCH.get(name)
            if not dispatch:
                _log(f"❓ Unknown tool requested: {name}", logging.WARNING)
            _progress(status, f"Running {name.replace('_', ' ')}…")
            t0 = time.monotonic()
            try:
                raw = dispatch(args) if dispatch else _fail(f"Unknown tool: {name}")
                if not isinstance(raw, ToolResult):
                    _log(f"⚠️ '{name}' returned a bare {type(raw).__name__}, not a "
                         "ToolResult — classified by the legacy text check", logging.WARNING)
                result = as_tool_result(raw)
            except Exception as e:
                result = _fail(f"Tool error: {e}")
                # The user only ever sees the one-line "Tool error: ..." string
                # above; the full traceback -- what actually broke inside the
                # tool -- only ever lands here, in the log.
                log.exception("Tool '%s' raised with args=%s", name, args)
            elapsed_ms = (time.monotonic() - t0) * 1000
            if elapsed_ms > 2000:
                log.warning("Tool '%s' took %.0f ms", name, elapsed_ms)

            # Track NGL strings that now have a ball+stick highlight
            if name == "select" and result.ok:
                ngl_str = st.session_state.selections.get(args.get("name", ""), "")
                if ngl_str:
                    selected_ngl_strs.add(ngl_str)

            # Mark that a representation-type show has fired this run
            if name == "show" and args.get("rep_type", "cartoon") != "ball+stick":
                show_rep_fired = True

            level = logging.INFO if result.ok else logging.ERROR
            if status is not None:
                status.write(f"{'✓' if result.ok else '✗'} {name.replace('_', ' ')}")
            _log(f"🔧 {name}({args}) → {result}", level)
            summary_parts.append(f"{name}: {result}")
            ran.append((name, result))
            _remember(name, args, result)
            if plan and not result.data.get("needs_user_choice"):
                _advance_plan(plan, name, result)
                _show_plan()
            tool_results.append({"tool": name, "result": result,
                                 "args": dict(args), "ms": round(elapsed_ms)})
            # A call that succeeds may have fixed what an earlier one lacked
            # (parameterize_ligand, then setup_amber again), so failed calls
            # become retryable; one that fails stays blocked until then.
            if not result.ok:
                failed_sigs[sig] = result.summary
            else:
                called_sigs.difference_update(failed_sigs)
                failed_sigs.clear()

        # Every call above appended exactly one result, in call order.
        for tc, r in zip(tool_calls, tool_results):
            res = r["result"]
            _trace({"turn": turn, "tool": r["tool"],
                    "args": r.get("args", _tc_args(tc)),
                    "status": "ran" if "ms" in r else res.data.get("status", "skipped"),
                    "ok": res.ok,
                    "failed": "ms" in r and not res.ok,
                    "result": res.summary[:800], "ms": r.get("ms"),
                    "data": json.loads(json.dumps(res.data, default=str)) if "ms" in r else {},
                    "files": res.files, "next_hints": res.next_hints})

        # A question for the user (ask_user, or a tool that found the request
        # ambiguous) ends the request here: the answer decides what runs next,
        # so letting the model continue would only mean a guess.
        if st.session_state.get("clarify"):
            _log(f"❔ Asking the user: {clarification_text()}")
            _progress(status, "Need a choice from you before going on.")
            # S6: the answer resumes this plan, and what already ran is reported
            # now — a question at step 6 used to drop steps 1-5's results.
            if plan and any(s["status"] == "done" for s in plan["steps"]):
                st.session_state.plan_paused = plan
                return (f"{clarification_text()}\n\nSo far:\n{_render_plan(plan)}")
            return clarification_text()

        results_text = "\n".join(f"[{r['tool']}]: {_for_model(r['result'])}"
                                  for r in tool_results)

        # Append tool results to the conversation. The system message is left
        # alone; the refreshed scene rides along with the results instead.
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})

        # After tool execution, strongly prefer summarization over additional tool calls.
        follow_up = (
            "Representation updated. Reply with one sentence confirming what changed. "
            "Do not call any more tools."
            if show_rep_fired else
            "Tool execution is complete. "
            "If the user's request has been satisfied, reply only with a short plain-text summary. "
            "Do not call additional tools. "
            "Only call another tool if the previous result explicitly reports missing information "
            "that prevents completion."
        )
        # Say which calls failed, in so many words. Left implicit, the model
        # read "could not make sense of '30'" and told the user the residues
        # were highlighted.
        if plan:
            follow_up = _plan_follow_up(plan)
        failed = [r for r in tool_results if not r["result"].ok]
        if failed:
            follow_up = (
                "These calls FAILED and changed nothing: "
                + "; ".join(f"{r['tool']} ({r['result'].summary[:120]})" for r in failed)
                + ". Never say they worked — and a failed call found nothing either way, "
                "so never report its absence of results as a finding ('no salt bridges "
                "were found'). Retry once with corrected arguments if the "
                "error shows how, otherwise tell the user plainly what failed and why. "
                + ("" if show_rep_fired and not plan else follow_up))
        messages.append({
            "role": "user",
            "content": f"Tool results:\n{results_text}\n\n{_state_block()}\n\n{follow_up}"
        })

    _log("⚠️ Max turns reached", logging.WARNING)
    _progress(status, "Stopping — reached the step limit for this request.")
    return "Done: " + "; ".join(summary_parts) + _plan_footer()


# ═══════════════════════════════════════════════════════════════════════════════
# NGL.js HTML renderer
# ═══════════════════════════════════════════════════════════════════════════════

def _viewer_payload() -> dict:
    """
    Describe the whole scene for the viewer as plain JSON-serialisable data.

    Building a data structure and handing it to one JS loop, rather than
    concatenating a block of JavaScript per structure, is what keeps the viewer
    code readable now that any number of structures can be open at once.

    Each structure carries either a `url` (fetched entries stream straight from
    RCSB) or `data` (the file's text, for anything local — an upload or a path
    the user gave). The local case has to be embedded: the browser cannot read
    a path on the server's disk, which is why local files never used to appear
    in the viewer at all.
    """
    bg = st.session_state.background
    reps = st.session_state.representations
    payload = []

    for s in structures():
        rep_list = []
        for rep in reps:
            if not rep.get("visible", True):        # layer hidden via the eye toggle
                continue
            if not rep_applies_to(rep, s["sid"]):   # layer scoped to another structure
                continue
            color = rep.get("color", "element")
            if color == BY_STRUCTURE:
                color = s["color"]
            # On white, a rainbow protein cartoon washes out; chain colouring reads better.
            elif (bg == "white" and color in ("spectrum", "residueindex")
                    and rep.get("selection") == "protein"):
                color = "chainname"
            rep_list.append({
                "type": rep["type"],
                "sele": rep["selection"],
                "color": color,
                "opacity": round(1.0 - float(rep.get("transparency", 0.0)), 3),
            })

        # "Active only" hides the rest of the scene without forgetting that
        # they were shown, so turning it off puts everything back as it was.
        shown = bool(s["visible"]) and not (
            st.session_state.active_only and s["sid"] != st.session_state.active_sid)

        entry = {
            "sid": s["sid"], "label": s["pdb_id"], "color": s["color"],
            "visible": shown, "matrix": s.get("matrix"),
            "reps": rep_list, "url": None, "data": None,
            # Measured distances are drawn by NGL's distance representation,
            # which resolves each end of a pair itself — so the viewer shows the
            # same geometry the measurement was taken from, not a copy of the
            # number that could drift out of step with it.
        }
        if s["source"] == "rcsb":
            entry["url"] = f"https://files.rcsb.org/download/{s['pdb_id']}.pdb"
        else:
            try:
                entry["data"] = Path(s["path"]).read_text(errors="replace")
            except Exception:
                entry["url"] = None     # unreadable file — the viewer skips it
        payload.append(entry)

    return {
        "bg": bg,
        "structures": payload,
        "lines": _dashed_lines(),
        "labels": _label_points(),
        # Vocabularies for the in-viewer style menu, so the toolbar offers
        # exactly the names the Representations panel does.
        "repTypes": NGL_REP_TYPES,
        "colorSchemes": [[k, v] for k, v in NGL_COLOR_SCHEMES.items()],
        "selPresets": [[k, v] for k, v in SELECTION_PRESETS.items()],
        "labelColors": [[k, v] for k, v in LABEL_COLORS.items()],
        "activeOnly": bool(st.session_state.active_only),
        # A zoom target is a selection resolved against the active structure,
        # so the camera has to be aimed at that component, not whichever
        # happens to be first in the list.
        "activeSid": st.session_state.active_sid or "",
        "camTarget": st.session_state.camera_target or "",
        "snapName": "_".join(x["pdb_id"] for x in structures()) + "_view.png",
        # The camera is remembered per *set* of structures: restoring a
        # single-structure viewpoint onto a superposed pair would drop the user
        # somewhere arbitrary in the new scene.
        "camKey": "ngl_cam_" + "_".join(s["pdb_id"] for s in structures()),
    }


def _hex_to_rgb(value: str):
    """'#FFD34D' → [1.0, 0.83, 0.30], the 0–1 triple NGL's Shape API wants."""
    v = value.lstrip("#")
    return [int(v[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


MEASUREMENT_COLOR = "#FFFFFF"


def _dashed_lines() -> list:
    """
    Every interaction and measurement as a dashed line in world coordinates.

    Drawn through NGL's Shape API rather than its distance representation: a
    distance representation renders a solid line, and dashes are the universal
    convention for a non-covalent contact — a solid stick between two residues
    reads as a bond that is not there. Shape takes dashedCylinder directly.

    Endpoints are resolved here, in Python, from the atom indices recorded when
    the interaction was found, with the structure's superposition transform
    applied. That keeps the drawing and the reported number derived from the
    same coordinates.
    """
    lines = []

    def endpoint(record, colour, label, kind):
        entry = find_structure(record["sid"])
        if not entry or "a_index" not in record:
            return
        try:
            atoms = _atoms_of(entry, in_common_frame=True)
            a = atoms["xyz"][record["a_index"]]
            b = atoms["xyz"][record["b_index"]]
        except (IndexError, KeyError, OSError):
            return
        lines.append({
            "a": [round(float(v), 3) for v in a],
            "b": [round(float(v), 3) for v in b],
            "color": _hex_to_rgb(colour),
            "label": label,
            "type": kind,
        })

    for record in st.session_state.interactions:
        endpoint(record, ixn.TYPE_COLORS.get(record["type"], "#CCCCCC"),
                 f"{record.get('distance', 0):.2f} Å", record["type"])

    for record in st.session_state.measurements:
        endpoint(record, MEASUREMENT_COLOR,
                 f"{record.get('closest', 0):.2f} Å", "measurement")

    return lines


def _label_points():
    """
    Each pinned label as a point in the shared coordinate space.

    The anchor is recomputed here rather than stored on the annotation, so a
    label stays on its region after a superposition moves the structure, and a
    label whose target no longer resolves (the residues were stripped by a
    preparation step, say) quietly drops out instead of floating in space.
    """
    out = []
    for a in st.session_state.annotations:
        entry = find_structure(a["sid"])
        if not entry:
            continue
        try:
            indices, _, err = _resolve_label_target(entry, a["target"])
            if err or not indices:
                continue
            atoms = _atoms_of(entry, in_common_frame=True)
            out.append({
                "id":    a["id"],
                "xyz":   _label_anchor(atoms, indices, a.get("offset", 0.0)),
                "text":  a["text"],
                "color": _hex_to_rgb(a["color"]),
                "hex":   a["color"],
                "where": a["desc"] or a["target"],
                "size":  float(a.get("size", 4.0)),
            })
        except (IndexError, KeyError, OSError, ValueError):
            continue
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Viewer component
# ═══════════════════════════════════════════════════════════════════════════════
# build_ngl_html() used to go straight to st.components.v1.html, which is a
# one-way iframe: Python could draw the scene but the scene could not answer.
# That is why every control had to live in the side panels — anything changed
# inside the viewer was thrown away by the next rerun, and reruns happen on
# every chat message and every widget click.
#
# viewer_component/ is a declared component, so the same generated HTML now
# runs somewhere that can call setComponentValue. The toolbar's pen and style
# menus post an event through window.paroraEmit, it arrives here as the
# component's return value, and handle_viewer_event() folds it into session
# state exactly as the equivalent panel button would.

_VIEWER_DIR = Path(__file__).parent / "viewer_component"
try:
    _viewer_component = st.components.v1.declare_component(
        "parora_viewer", path=str(_VIEWER_DIR))
except Exception:                      # missing directory — fall back below
    _viewer_component = None


def handle_viewer_event(event) -> bool:
    """
    Apply one edit made inside the viewer.

    Streamlit replays a component's last value on every rerun, so the event
    carries a counter and anything already applied is ignored — otherwise a
    single click would keep re-adding its label for the rest of the session.

    Returns:
        True when session state changed and the page should rerun.
    """
    if not isinstance(event, dict):
        return False
    seq = event.get("n")
    if not isinstance(seq, int) or seq <= st.session_state.viewer_seq:
        return False
    st.session_state.viewer_seq = seq

    kind = event.get("kind", "")
    entry = find_structure(event.get("sid", "")) or active_structure()

    if kind == "add_label":
        st.session_state.label_msg = tool_add_label(
            event.get("text", ""), event.get("target", ""),
            entry["pdb_id"] if entry else "",
            event.get("color", "yellow"), float(event.get("offset", 0) or 0))
        return True

    if kind == "edit_label":
        for a in st.session_state.annotations:
            if a["id"] == event.get("id"):
                text = (event.get("text") or "").strip()
                if text:
                    a["text"] = text
                if event.get("color"):
                    a["color"] = LABEL_COLORS.get(event["color"], a["color"])
                return True
        return False

    if kind == "delete_label":
        before = len(st.session_state.annotations)
        st.session_state.annotations = [
            a for a in st.session_state.annotations if a["id"] != event.get("id")]
        return len(st.session_state.annotations) != before

    if kind == "set_active":
        target = find_structure(event.get("sid", ""))
        if target and target["sid"] != st.session_state.active_sid:
            st.session_state.active_sid = target["sid"]
            _sync_active()
            return True
        return False

    if kind == "active_only":
        want = bool(event.get("value"))
        if want != st.session_state.active_only:
            st.session_state.active_only = want
            return True
        return False

    if kind == "toggle_structure":
        target = find_structure(event.get("sid", ""))
        if target:
            target["visible"] = not target.get("visible", True)
            return True
        return False

    if kind == "set_style":
        if not entry:
            return False
        sele = resolve_selection(event.get("selection", "all") or "all")
        rep_type = event.get("rep_type", "cartoon")
        if rep_type not in NGL_REP_TYPES:
            rep_type = "cartoon"
        scope = entry["sid"] if event.get("scope", "active") == "active" else "*"
        # Replace any layer of the same type on the same selection and scope,
        # so repeatedly nudging a slider does not stack up duplicate layers.
        # A layer that applies to everything is spelled None, "" or "*"
        # depending on where it came from; rep_applies_to treats all three the
        # same, so matching here has to as well.
        def _same_scope(rep):
            a = rep.get("sid") or "*"
            return (a if a != "" else "*") == scope
        st.session_state.representations = [
            r for r in st.session_state.representations
            if not (r["type"] == rep_type and r["selection"] == sele and _same_scope(r))
        ]
        st.session_state.representations.append({
            "id": uuid.uuid4().hex[:8],
            "type": rep_type,
            "selection": sele,
            "color": event.get("color", "residueindex"),
            "transparency": max(0.0, min(1.0, float(event.get("transparency", 0) or 0))),
            "visible": True,
            "sid": scope,
        })
        return True

    if kind == "toolbar_open":
        pane = (event.get("top"), event.get("sub"))
        if pane in TOOLBAR_DISPATCH and st.session_state.toolbar_open != pane:
            st.session_state.toolbar_open = pane
            return True
        return False

    if kind == "toggle_bg":
        st.session_state.background = (
            "white" if st.session_state.background == "black" else "black")
        return True

    return False


def build_ngl_html() -> str:
    """
    Generate the full HTML/JS snippet that renders the current scene in NGL.js.

    Every loaded structure is drawn in one shared coordinate space, each with
    the representation layers scoped to it, and each with its superposition
    transform applied through NGL's Component.setTransform. Applying the fit as
    a transform rather than rewriting coordinates is what lets a superposition
    be undone or recomputed without reloading anything.

    Camera orientation is saved to and restored from localStorage so the
    viewpoint survives Streamlit reruns.

    The block also carries its own navigation layer, because NGL's stock
    controls assume a three-button mouse: an overlay toolbar (rotate/move mode,
    zoom, fit, reset, spin, help), remapped mouse bindings so Shift+left-drag
    pans, keyboard navigation with the arrow keys, and double-click to zoom
    into a residue. The chosen drag mode is remembered in localStorage.

    Returns:
        HTML string to pass to st.components.v1.html(), or "" if nothing is
        loaded.
    """
    if not structures():
        return ""

    cfg = _viewer_payload()
    bg = cfg["bg"]
    bg_label = "☀️ Light" if bg == "black" else "\U0001f319 Dark"
    # "</" inside a <script> would end the block early, whatever the JSON says.
    cfg_json = json.dumps(cfg).replace("</", "<\\/")

    legend_rows = "".join(
        f'<button class="leg" data-sid="{s["sid"]}" title="Show or hide {s["pdb_id"]}">'
        f'<span class="dot" style="background:{s["color"]}"></span>{s["pdb_id"]}</button>'
        for s in structures()
    ) if len(structures()) > 1 else ""

    shown_types = {x["type"] for x in st.session_state.interactions}
    key_rows = [(label, color) for kind, label, color in ixn.INTERACTION_TYPES
                if kind in shown_types]
    if st.session_state.measurements:
        key_rows.append(("Measured", MEASUREMENT_COLOR))
    ixn_key = "".join(f'<span style="color:{color}"><i></i>{label}</span>'
                      for label, color in key_rows)

    # The HTML is built from a plain template with __TOKEN__ placeholders rather
    # than an f-string: the JS/CSS below is full of braces, and doubling every
    # one of them makes the block unreadable and easy to break.
    tpl = r"""
    <div id="wrap" tabindex="0" style="position:relative;width:100%;height:680px;
         border:1px solid #555;border-radius:8px;overflow:hidden;background:__BG__;outline:none;">
      <div id="toolbar">
        <div class="tbgrp">
          <button data-act="tb-load">Load</button>
          <div class="tbdrop" data-top="Load" hidden>
            <button data-act="tb-pick" data-top="Load" data-sub="Loaded">Loaded</button>
            <button data-act="tb-pick" data-top="Load" data-sub="Find by protein name">Find by protein name</button>
          </div>
        </div>
        <div class="tbgrp">
          <button data-act="tb-pick" data-top="Prepare">Prepare</button>
        </div>
        <div class="tbgrp">
          <button data-act="tb-style">Style</button>
          <div class="tbdrop" data-top="Style" hidden>
            <button data-act="tb-pick" data-top="Style" data-sub="Representations">Representations</button>
            <button data-act="tb-pick" data-top="Style" data-sub="Labels">Labels</button>
            <button data-act="tb-pick" data-top="Style" data-sub="Ray-traced figure">Ray-traced figure</button>
          </div>
        </div>
        <div class="tbgrp">
          <button data-act="tb-analyze">Analyze</button>
          <div class="tbdrop" data-top="Analyze" hidden>
            <button data-act="tb-pick" data-top="Analyze" data-sub="Measure">Measure</button>
            <button data-act="tb-pick" data-top="Analyze" data-sub="Interactions">Interactions</button>
            <button data-act="tb-pick" data-top="Analyze" data-sub="Highlight">Highlight</button>
            <button data-act="tb-pick" data-top="Analyze" data-sub="Summary">Summary</button>
            <button data-act="tb-pick" data-top="Analyze" data-sub="Sequence">Sequence</button>
          </div>
        </div>
        <div class="tbgrp">
          <button data-act="tb-simulate">Simulate</button>
          <div class="tbdrop" data-top="Simulate" hidden>
            <button data-act="tb-pick" data-top="Simulate" data-sub="MD &middot; Amber">MD &middot; Amber</button>
            <button data-act="tb-pick" data-top="Simulate" data-sub="MD &middot; GROMACS">MD &middot; GROMACS</button>
            <button data-act="tb-pick" data-top="Simulate" data-sub="MD &middot; Rosetta">MD &middot; Rosetta</button>
            <button data-act="tb-pick" data-top="Simulate" data-sub="QM">QM</button>
            <button data-act="tb-pick" data-top="Simulate" data-sub="QM/MM">QM/MM</button>
            <button data-act="tb-pick" data-top="Simulate" data-sub="Membrane">Membrane</button>
          </div>
        </div>
        <div class="tbgrp tbgrp-right">
          <button data-act="tb-bg" title="Switch the viewer background">__BG_LABEL__</button>
        </div>
      </div>
      <div id="viewport" style="width:100%;height:100%;"></div>

      <!-- Navigation toolbar: overlays the canvas so the structure can be moved
           without a three-button mouse. -->
      <div id="nav">
        <div class="grp">
          <button data-act="rotate" class="mode on" title="Left-drag spins the structure (R)">&#8635; Rotate</button>
          <button data-act="pan"    class="mode"    title="Left-drag slides the structure around (M)">&#10021; Move</button>
        </div>
        <div class="grp">
          <button data-act="zin"  title="Zoom in (+)">&#43;</button>
          <button data-act="zout" title="Zoom out (&#8722;)">&#8722;</button>
        </div>
        <div class="grp">
          <button data-act="fit"   title="Fit everything in view (F)">&#9974; Fit</button>
          <button data-act="reset" title="Forget the saved viewpoint and start over">&#8634; Reset</button>
          <button data-act="spin"  title="Toggle auto-spin (S)">&#9862; Spin</button>
        </div>
        <div class="grp">
          <button data-act="pen"   title="Label mode: click the structure to write on it (L)">&#9998; Label</button>
          <button data-act="style" title="Change how the active structure is drawn (Y)">&#127912; Style</button>
        </div>
        <div class="grp">
          <button data-act="solo"  title="Draw only the active structure (O)">&#9678; Active only</button>
        </div>
        <div class="grp">
          <button data-act="snap" title="Save exactly this view as a PNG (P)">&#128247; PNG</button>
          <button data-act="help" title="Show all mouse and keyboard controls">?</button>
        </div>
      </div>

      <!-- Label mode. Placing one is a click on the structure; the list below
           is how an existing label gets renamed or removed without leaving
           the picture. -->
      <div id="penbox" hidden>
        <div class="hd">&#9998; Label mode
          <button class="x" data-act="pen-close" title="Leave label mode">&times;</button></div>
        <div class="tip">Click anywhere on the structure to label that residue.</div>
        <div class="row">
          <label>Colour</label><select id="pen-color"></select>
          <label>Push out</label><input id="pen-offset" type="number" value="0" min="0" max="120" step="5">
        </div>
        <div id="pen-list"></div>
      </div>

      <!-- Style menu. Applies to the active structure by default, which is
           what "active only" is for: seeing the change on one structure
           without the others in the way. -->
      <div id="stylebox" hidden>
        <div class="hd">&#127912; Style
          <button class="x" data-act="style-close" title="Close">&times;</button></div>
        <div class="row"><label>Apply to</label><select id="st-scope">
          <option value="active">active structure</option>
          <option value="all">every structure</option></select></div>
        <div class="row"><label>Part</label><select id="st-sele"></select></div>
        <div class="row"><label>Draw as</label><select id="st-type"></select></div>
        <div class="row"><label>Colour</label><select id="st-color"></select></div>
        <div class="row"><label>Transparency</label>
          <input id="st-alpha" type="range" min="0" max="1" step="0.05" value="0">
          <span id="st-alpha-v">0.00</span></div>
        <div class="row"><button class="btn" data-act="style-apply">Apply</button></div>
      </div>

      <!-- One chip per structure; clicking one shows or hides it. Empty when
           only a single structure is loaded. -->
      <div id="legend">__LEGEND__</div>

      <!-- Colour key for detected interactions; empty when none are drawn. -->
      <div id="ixnkey">__IXNKEY__</div>

      <div id="hint">Drag&nbsp;= rotate &nbsp;·&nbsp; <b>Shift</b>+drag or right-drag&nbsp;= move &nbsp;·&nbsp;
           scroll&nbsp;= zoom &nbsp;·&nbsp; <b>L</b>&nbsp;= label &nbsp;·&nbsp; <b>Y</b>&nbsp;= style &nbsp;·&nbsp;
           <b>O</b>&nbsp;= active only</div>

      <!-- Snapshot overlay. The image is shown, not only downloaded: an
           embedded iframe may not be permitted to start a download, and a
           visible image can always be saved with right-click. -->
      <div id="snapbox" style="display:none;">
        <div class="row">
          <b>Snapshot</b>
          <span style="flex:1"></span>
          <a id="snaplink" class="btn" download>&#11015; Download</a>
          <button class="btn" data-act="snap-close">Close</button>
        </div>
        <img id="snapimg" alt="">
        <div id="snapnote"></div>
      </div>

      <div id="helpbox" style="display:none;">
        <b>Mouse</b>
        <table>
          <tr><td>Left drag</td><td>rotate &mdash; or slide, in <i>Move</i> mode</td></tr>
          <tr><td>Shift&nbsp;/&nbsp;Ctrl + left drag</td><td>move (pan)</td></tr>
          <tr><td>Right drag &middot; middle drag</td><td>move (pan)</td></tr>
          <tr><td>Ctrl + Shift + left drag</td><td>zoom</td></tr>
          <tr><td>Scroll wheel / two&#8209;finger</td><td>zoom</td></tr>
          <tr><td>Shift + scroll</td><td>depth of field / focus slab</td></tr>
          <tr><td>Click an atom</td><td>centre the camera on it</td></tr>
          <tr><td>Double-click an atom</td><td>zoom right in on that residue</td></tr>
          <tr><td>Double-click the background</td><td>fit everything back in view</td></tr>
          <tr><td>Hover</td><td>identify the atom / residue</td></tr>
          <tr><td>Right-click two atoms</td><td>measure the distance between them</td></tr>
        </table>
        <b>Keyboard</b> <span class="dim">&mdash; with the pointer over the viewer</span>
        <table>
          <tr><td>Arrow keys</td><td>move the structure</td></tr>
          <tr><td>Shift + arrows</td><td>rotate</td></tr>
          <tr><td>+ &nbsp;/&nbsp; &#8722;</td><td>zoom in / out</td></tr>
          <tr><td>R &nbsp;/&nbsp; M</td><td>rotate mode / move mode</td></tr>
          <tr><td>P</td><td>save this view as a PNG</td></tr>
          <tr><td>F &nbsp;/&nbsp; S</td><td>fit the <i>active</i> structure / auto-spin</td></tr>
          <tr><td>L</td><td>label mode &mdash; click the structure to write on it</td></tr>
          <tr><td>Y</td><td>style menu for the active structure</td></tr>
          <tr><td>O</td><td>draw only the active structure</td></tr>
          <tr><td>1 &hellip; 9</td><td>make that structure active</td></tr>
          <tr><td>Shift + 1 &hellip; 9</td><td>show or hide that structure</td></tr>
        </table>
        <b>Several structures at once</b>
        <table>
          <tr><td>Click a chip</td><td>make that structure the active one</td></tr>
          <tr><td>Click its coloured dot</td><td>show or hide it, without a reload</td></tr>
          <tr><td><i>Active only</i></td><td>hide everything but the active structure</td></tr>
          <tr><td><i>Fit</i> &nbsp;/&nbsp; <i>Reset</i></td><td>frame the active one / frame everything</td></tr>
        </table>
        <button data-act="help" class="close">Close</button>
      </div>
    </div>

    <style>
      #toolbar{position:absolute;top:0;left:0;width:100%;height:34px;z-index:12;
               display:flex;align-items:center;gap:4px;padding:0 8px;box-sizing:border-box;
               background:rgba(20,20,24,.85);border-bottom:1px solid rgba(255,255,255,.16);
               font:12px/1 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
      #toolbar button{background:transparent;border:1px solid rgba(255,255,255,.22);
               border-radius:5px;color:#e8e8ea;padding:4px 10px;cursor:pointer;font:inherit;}
      #toolbar button:hover{background:rgba(255,255,255,.18);}
      .tbgrp{position:relative;}
      .tbgrp-right{margin-left:auto;}
      .tbdrop{position:absolute;top:100%;left:0;z-index:13;display:flex;flex-direction:column;
              min-width:190px;background:rgba(20,20,24,.94);border:1px solid rgba(255,255,255,.2);
              border-radius:6px;padding:4px;box-shadow:0 6px 18px rgba(0,0,0,.4);}
      .tbdrop[hidden]{display:none;}
      .tbdrop button{background:transparent;border:0;color:#e8e8ea;text-align:left;
              padding:6px 8px;border-radius:4px;cursor:pointer;font:inherit;white-space:nowrap;}
      .tbdrop button:hover{background:rgba(255,255,255,.15);}
      #nav{position:absolute;top:42px;left:8px;display:flex;gap:8px;flex-wrap:wrap;
           font:12px/1 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;z-index:10;}
      #nav .grp{display:flex;background:rgba(20,20,24,.72);border:1px solid rgba(255,255,255,.16);
                border-radius:7px;overflow:hidden;}
      #nav button{background:transparent;border:0;border-right:1px solid rgba(255,255,255,.12);
                  color:#e8e8ea;padding:6px 10px;cursor:pointer;font:inherit;white-space:nowrap;}
      #nav .grp button:last-child{border-right:0;}
      #nav button:hover{background:rgba(255,255,255,.18);}
      #nav button.on{background:#2f6feb;color:#fff;}
      #penbox,#stylebox{position:absolute;top:78px;left:8px;z-index:20;width:264px;
              background:rgba(20,20,24,.94);border:1px solid rgba(255,255,255,.2);
              border-radius:8px;color:#e8e8ea;padding:8px 10px 10px;
              font:12px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
              box-shadow:0 6px 22px rgba(0,0,0,.45);}
      #stylebox{left:auto;right:8px;top:44px;}
      #penbox[hidden],#stylebox[hidden]{display:none;}
      #penbox .hd,#stylebox .hd{font-weight:600;margin-bottom:6px;display:flex;
              align-items:center;justify-content:space-between;}
      #penbox .x,#stylebox .x{background:transparent;border:0;color:#bbb;cursor:pointer;
              font-size:15px;line-height:1;padding:0 2px;}
      #penbox .tip{color:#aab;margin-bottom:7px;}
      #penbox .row,#stylebox .row{display:flex;align-items:center;gap:6px;margin-bottom:6px;}
      #penbox label,#stylebox label{flex:0 0 76px;color:#aab;}
      #penbox select,#penbox input,#stylebox select,#stylebox input{flex:1;min-width:0;
              background:#15151a;color:#e8e8ea;border:1px solid rgba(255,255,255,.22);
              border-radius:5px;padding:3px 5px;font:inherit;}
      #stylebox #st-alpha-v{flex:0 0 34px;text-align:right;color:#aab;}
      .btn{background:#2f6feb;border:0;color:#fff;border-radius:5px;padding:5px 12px;
              cursor:pointer;font:inherit;}
      #pen-list{max-height:150px;overflow:auto;margin-top:4px;}
      #pen-list .lbl{display:flex;align-items:center;gap:6px;padding:3px 0;
              border-top:1px solid rgba(255,255,255,.1);}
      #pen-list .lbl b{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;
              white-space:nowrap;font-weight:600;}
      #pen-list .lbl small{color:#8a8a97;flex:0 0 auto;max-width:88px;overflow:hidden;
              text-overflow:ellipsis;white-space:nowrap;}
      #pen-list .lbl button{background:transparent;border:0;color:#bbb;cursor:pointer;padding:0 3px;}
      #pen-list .lbl button:hover{color:#fff;}
      #wrap.penning #viewport{cursor:crosshair;}
      #legend .leg.active{outline:2px solid #2f6feb;outline-offset:-2px;}
      #legend{position:absolute;top:42px;right:8px;z-index:10;display:flex;flex-direction:column;
              gap:4px;align-items:flex-end;
              font:12px/1 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
      #legend .leg{display:flex;align-items:center;gap:7px;cursor:pointer;font:inherit;
                   background:rgba(20,20,24,.72);border:1px solid rgba(255,255,255,.16);
                   border-radius:7px;color:#e8e8ea;padding:6px 10px;}
      #legend .leg:hover{background:rgba(255,255,255,.18);}
      #legend .leg.off{opacity:.4;text-decoration:line-through;}
      #legend .dot{width:10px;height:10px;border-radius:50%;display:inline-block;
                   box-shadow:0 0 0 1px rgba(255,255,255,.35);}
      #ixnkey{position:absolute;bottom:8px;right:8px;z-index:10;display:flex;
              flex-direction:column;gap:3px;align-items:flex-end;
              font:11px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
      #ixnkey span{display:flex;align-items:center;gap:6px;color:#e8e8ea;
                   background:rgba(20,20,24,.66);padding:3px 8px;border-radius:5px;}
      #ixnkey i{width:14px;height:0;border-top:2px dashed currentColor;display:block;}
      #snapbox{position:absolute;inset:12px;z-index:30;overflow:auto;
               background:rgba(16,16,20,.97);border:1px solid rgba(255,255,255,.2);
               border-radius:8px;padding:10px 12px;color:#e8e8ea;
               font:12px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
      #snapbox .row{display:flex;align-items:center;gap:8px;margin-bottom:8px;}
      #snapbox .btn{background:#2f6feb;border:0;color:#fff;padding:5px 12px;
                    border-radius:5px;cursor:pointer;font:inherit;text-decoration:none;}
      #snapbox img{max-width:100%;border-radius:6px;
                   background:repeating-conic-gradient(#2a2a30 0% 25%,#1c1c22 0% 50%)
                              50%/16px 16px;}
      #snapnote{margin-top:8px;color:#b9bcc4;}
      #hint{position:absolute;bottom:8px;left:10px;z-index:10;pointer-events:none;
            font:11px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#e8e8ea;
            background:rgba(20,20,24,.6);padding:4px 8px;border-radius:5px;
            opacity:.85;transition:opacity .6s;}
      #hint.fade{opacity:0;}
      #helpbox{position:absolute;top:80px;left:8px;z-index:20;max-width:440px;
               background:rgba(20,20,24,.94);border:1px solid rgba(255,255,255,.18);
               border-radius:8px;padding:12px 14px;color:#e8e8ea;
               font:12px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
      #helpbox table{border-collapse:collapse;margin:6px 0 12px;}
      #helpbox td{padding:2px 14px 2px 0;vertical-align:top;}
      #helpbox td:first-child{color:#9fd0ff;white-space:nowrap;}
      #helpbox .dim{color:#9a9aa2;font-weight:normal;}
      #helpbox .close{background:#2f6feb;border:0;color:#fff;padding:5px 12px;
                      border-radius:5px;cursor:pointer;font:inherit;}
    </style>

    <script src="https://unpkg.com/ngl@2/dist/ngl.js"></script>
    <script>
    (function(){
        var CFG      = __CFG__;
        var CAM_KEY  = CFG.camKey;
        var MODE_KEY = "ngl_dragmode";
        var wrap = document.getElementById("wrap");

        function saveCam(stage) {
            try {
                var o = stage.viewerControls.getOrientation();
                localStorage.setItem(CAM_KEY, JSON.stringify(Array.from(o.elements)));
            } catch(e) {}
        }

        function restoreCam(stage) {
            try {
                var raw = localStorage.getItem(CAM_KEY);
                if (!raw) return false;
                var arr = JSON.parse(raw);
                var o = stage.viewerControls.getOrientation();
                o.elements.set(new Float32Array(arr));
                stage.viewerControls.orient(o);
                return true;
            } catch(e) { return false; }
        }

        var stage = new NGL.Stage("viewport", {backgroundColor: CFG.bg});
        var comps = {};          // sid -> loaded NGL component
        var spinning = false;
        var snapUrl = null;      // object URL of the most recent snapshot

        // ── Mouse bindings ───────────────────────────────────────────────────
        // NGL's stock preset only pans on right/middle-drag and puts zoom on
        // Shift+drag, which is unusable on a trackpad or one-button mouse.
        // Shift-drag is remapped to pan (the convention in most viewers) and
        // zoom moves to Ctrl+Shift; the scroll wheel already zooms.
        var MA = NGL.MouseActions, MC = stage.mouseControls;
        ["drag-left", "drag-right", "drag-middle",
         "drag-shift-left", "drag-ctrl-left", "drag-ctrl-shift-left"
        ].forEach(function(t){ MC.remove(t); });
        MC.add("drag-right",           MA.panDrag);
        MC.add("drag-middle",          MA.panDrag);
        MC.add("drag-shift-left",      MA.panDrag);
        MC.add("drag-ctrl-left",       MA.panDrag);
        MC.add("drag-ctrl-shift-left", MA.zoomDrag);

        function setMode(mode) {
            MC.remove("drag-left");
            MC.add("drag-left", mode === "pan" ? MA.panDrag : MA.rotateDrag);
            wrap.querySelectorAll("#nav .mode").forEach(function(b){
                b.classList.toggle("on", b.dataset.act === mode);
            });
            try { localStorage.setItem(MODE_KEY, mode); } catch(e) {}
        }
        var savedMode = "rotate";
        try { savedMode = localStorage.getItem(MODE_KEY) || "rotate"; } catch(e) {}
        setMode(savedMode === "pan" ? "pan" : "rotate");

        // ── Per-structure visibility ─────────────────────────────────────────
        // Toggled from the legend chips or the number keys. This is local to
        // the page on purpose: comparing two superposed structures means
        // flicking one on and off repeatedly, and a Streamlit rerun per click
        // would reload the scene every time.
        function toggleStructure(sid) {
            var comp = comps[sid];
            if (!comp) return;
            var on = !comp.visible;
            comp.setVisibility(on);
            var chip = wrap.querySelector('.leg[data-sid="' + sid + '"]');
            if (chip) chip.classList.toggle("off", !on);
        }
        wrap.addEventListener("click", function(e){
            var chip = e.target.closest(".leg");
            if (!chip) return;
            e.preventDefault();
            // The coloured dot keeps the old local show/hide — flicking a
            // structure on and off to compare it has to stay instant. The rest
            // of the chip makes that structure the active one, which is a real
            // state change and so does go back to Python.
            if (e.target.classList.contains("dot")) { toggleStructure(chip.dataset.sid); }
            else if (chip.dataset.sid !== CFG.activeSid) {
                emit({kind: "set_active", sid: chip.dataset.sid});
            }
        });

        // ── Bridge to Python ─────────────────────────────────────────────────
        // window.paroraEmit is installed by viewer_component/index.html. When
        // the viewer is mounted the old one-way way it is absent, and the
        // in-viewer editors simply do nothing rather than failing loudly.
        function emit(o){
            if (window.paroraEmit) { window.paroraEmit(o); }
            else { console.warn("PARORA viewer: no bridge; use the side panels."); }
        }

        // ── In-viewer Load/Prepare/Style/Analyze/Simulate toolbar ─────────────
        // Mirrors the Streamlit-native toolbar above the viewer: a top-level
        // click opens its sub-panel dropdown, and picking a sub-item emits the
        // same (top, sub) pair that toolbar_open holds, so handle_viewer_event
        // routes it through the identical TOOLBAR_DISPATCH/_toolbar_dialog path
        // — one dialog, reachable from either toolbar.
        function closeTbDrops(){
            wrap.querySelectorAll(".tbdrop").forEach(function(d){ d.hidden = true; });
        }
        function toggleTbDrop(top){
            var drop = wrap.querySelector('.tbdrop[data-top="' + top + '"]');
            if (!drop) return;
            var opening = drop.hidden;
            closeTbDrops();
            drop.hidden = !opening;
        }

        function sidOfComponent(c){
            for (var k in comps) { if (comps[k] === c) return k; }
            return CFG.activeSid || "";
        }

        function fillSelect(el, pairs, selected){
            if (!el) return;
            el.innerHTML = "";
            pairs.forEach(function(p){
                var o = document.createElement("option");
                o.value = p[1]; o.textContent = p[0];
                if (p[1] === selected) o.selected = true;
                el.appendChild(o);
            });
        }

        // ── A text box floating at the click, instead of window.prompt, which
        //    a sandboxed component iframe is allowed to suppress. ────────────
        function askAt(px, py, initial, cb){
            var old = wrap.querySelector(".askbox");
            if (old) old.remove();
            var box = document.createElement("input");
            box.className = "askbox";
            box.value = initial || "";
            box.placeholder = "label text, Enter to place";
            box.style.cssText = "position:absolute;z-index:40;left:" +
                Math.max(4, Math.min(px, wrap.clientWidth - 190)) + "px;top:" +
                Math.max(4, Math.min(py, wrap.clientHeight - 34)) + "px;width:180px;" +
                "background:#15151a;color:#e8e8ea;border:1px solid #2f6feb;border-radius:5px;" +
                "padding:5px 7px;font:12px -apple-system,BlinkMacSystemFont,sans-serif;";
            wrap.appendChild(box);
            box.focus(); box.select();
            box.addEventListener("keydown", function(ev){
                ev.stopPropagation();
                if (ev.key === "Enter") { var v = box.value.trim(); box.remove(); if (v) cb(v); }
                else if (ev.key === "Escape") { box.remove(); }
            });
            box.addEventListener("blur", function(){ setTimeout(function(){ box.remove(); }, 120); });
        }

        // ── Label mode ───────────────────────────────────────────────────────
        var penning = false;

        function renderPenList(){
            var host = document.getElementById("pen-list");
            if (!host) return;
            host.innerHTML = "";
            (CFG.labels || []).forEach(function(l){
                var row = document.createElement("div");
                row.className = "lbl";
                row.innerHTML =
                    '<span class="dot" style="background:' + (l.hex || "#FFD24A") + '"></span>' +
                    '<b></b><small></small>' +
                    '<button data-edit="1" title="Rename">&#9998;</button>' +
                    '<button data-del="1" title="Remove">&times;</button>';
                row.querySelector("b").textContent = l.text;
                row.querySelector("small").textContent = l.where || "";
                row.querySelector('[data-edit]').addEventListener("click", function(){
                    askAt(20, 90, l.text, function(v){
                        emit({kind: "edit_label", id: l.id, text: v});
                    });
                });
                row.querySelector('[data-del]').addEventListener("click", function(){
                    emit({kind: "delete_label", id: l.id});
                });
                host.appendChild(row);
            });
            if (!(CFG.labels || []).length) {
                host.innerHTML = '<div class="tip" style="padding-top:5px">No labels yet.</div>';
            }
        }

        function setPen(on){
            penning = on;
            wrap.classList.toggle("penning", on);
            var box = document.getElementById("penbox");
            if (box) box.hidden = !on;
            var b = wrap.querySelector('button[data-act="pen"]');
            if (b) b.classList.toggle("on", on);
            if (on) { setStyleBox(false); renderPenList(); }
        }

        // A click on the structure while in label mode writes on the residue
        // that was hit, in the same chain/residue spelling the rest of the app
        // uses, so the label survives a rerun as a real annotation rather than
        // as a floating point in space.
        stage.signals.clicked.add(function(pp){
            if (!penning) return;
            var atom = pp && (pp.atom || pp.closestBondAtom);
            if (!atom) return;
            var target = (atom.chainname ? atom.chainname + "/" : "") + atom.resno;
            var sid = sidOfComponent(pp.component);
            var m = pp.canvasPosition || {x: 40, y: 60};
            askAt(m.x + 12, m.y + 12, "", function(text){
                emit({kind: "add_label", sid: sid, target: target, text: text,
                      color: (document.getElementById("pen-color") || {}).value || "yellow",
                      offset: parseFloat((document.getElementById("pen-offset") || {}).value || 0)});
            });
        });

        // ── Style menu ───────────────────────────────────────────────────────
        function setStyleBox(on){
            var box = document.getElementById("stylebox");
            if (box) box.hidden = !on;
            var b = wrap.querySelector('button[data-act="style"]');
            if (b) b.classList.toggle("on", on);
            if (on) setPen(false);
        }

        (function initMenus(){
            fillSelect(document.getElementById("st-type"),
                       (CFG.repTypes || []).map(function(t){ return [t, t]; }), "cartoon");
            fillSelect(document.getElementById("st-color"), CFG.colorSchemes || [], "residueindex");
            fillSelect(document.getElementById("st-sele"), CFG.selPresets || [], "all");
            fillSelect(document.getElementById("pen-color"),
                       (CFG.labelColors || []).map(function(c){ return [c[0], c[0]]; }), "yellow");
            var a = document.getElementById("st-alpha"), av = document.getElementById("st-alpha-v");
            if (a && av) a.addEventListener("input", function(){ av.textContent = (+a.value).toFixed(2); });
            var solo = wrap.querySelector('button[data-act="solo"]');
            if (solo) solo.classList.toggle("on", !!CFG.activeOnly);
            var chip = wrap.querySelector('.leg[data-sid="' + CFG.activeSid + '"]');
            if (chip) chip.classList.add("active");
        })();

        // ── Double-click: zoom into the residue under the cursor, or, on empty
        //    background, fit the whole scene back in view. (Single-click
        //    re-centring and the hover tooltip are NGL defaults.)
        stage.mouseObserver.signals.doubleClicked.add(function(x, y){
            var pp = null;
            try { pp = stage.pickingControls.pick(x, y); } catch(e) {}
            var atom = pp ? (pp.atom || pp.closestBondAtom) : null;
            if (atom && pp.component) {
                var chain = atom.chainname ? (":" + atom.chainname) : "";
                pp.component.autoView(atom.resno + chain, 500);
            } else {
                stage.autoView(500);
            }
            setTimeout(function(){ saveCam(stage); }, 700);
        });

        // ── Toolbar ──────────────────────────────────────────────────────────
        var actions = {
            rotate: function(){ setMode("rotate"); },
            pan:    function(){ setMode("pan"); },
            zin:    function(){ stage.viewerControls.zoom(0.15); },
            zout:   function(){ stage.viewerControls.zoom(-0.15); },
            // Fit frames the *active* structure. With several loaded, fitting
            // the whole scene is what kept dragging a half-off-screen partner
            // back into the picture while the user was working on one of them.
            // Reset still frames everything.
            fit:    function(){
                        var c = comps[CFG.activeSid];
                        if (c && c.visible) { c.autoView(400); } else { stage.autoView(400); }
                    },
            reset:  function(){
                        try { localStorage.removeItem(CAM_KEY); } catch(e) {}
                        stage.autoView(400);
                    },
            spin:   function(){
                        spinning = !spinning;
                        stage.setSpin(spinning);
                        var b = wrap.querySelector('button[data-act="spin"]');
                        if (b) b.classList.toggle("on", spinning);
                    },
            pen:    function(){ setPen(!penning); },
            "pen-close": function(){ setPen(false); },
            style:  function(){ setStyleBox(document.getElementById("stylebox").hidden); },
            "style-close": function(){ setStyleBox(false); },
            "style-apply": function(){
                        emit({kind: "set_style",
                              sid: CFG.activeSid,
                              scope: document.getElementById("st-scope").value,
                              selection: document.getElementById("st-sele").value,
                              rep_type: document.getElementById("st-type").value,
                              color: document.getElementById("st-color").value,
                              transparency: parseFloat(document.getElementById("st-alpha").value)});
                    },
            solo:   function(){ emit({kind: "active_only", value: !CFG.activeOnly}); },
            help:   function(){
                        var h = document.getElementById("helpbox");
                        h.style.display = (h.style.display === "none") ? "block" : "none";
                    },
            // Exported by NGL itself, from the same renderer and the same
            // camera that drew the viewport. This is the only export that is
            // guaranteed to match what is on screen — the PyMOL path re-builds
            // the scene in a different program and cannot be identical.
            snap:   function(){
                        var box  = document.getElementById("snapbox");
                        var img  = document.getElementById("snapimg");
                        var link = document.getElementById("snaplink");
                        var note = document.getElementById("snapnote");
                        box.style.display = "block";
                        img.removeAttribute("src");
                        link.removeAttribute("href");
                        note.textContent = "Rendering…";
                        stage.makeImage({factor: 3, antialias: true,
                                         trim: false, transparent: false})
                            .then(function(blob){
                                if (snapUrl) { URL.revokeObjectURL(snapUrl); }
                                snapUrl = URL.createObjectURL(blob);
                                img.src = snapUrl;
                                link.href = snapUrl;
                                link.download = CFG.snapName;
                                note.textContent =
                                    "Exactly what the viewer shows, at 3× the on-screen "
                                    + "resolution. Click Download — or, if your browser "
                                    + "blocks that here, right-click the image and "
                                    + "choose Save image as…";
                            })
                            .catch(function(e){
                                note.textContent = "Could not render the image: " + e;
                            });
                    },
            "snap-close": function(){
                        document.getElementById("snapbox").style.display = "none";
                    },
            "tb-load":     function(){ toggleTbDrop("Load"); },
            "tb-style":    function(){ toggleTbDrop("Style"); },
            "tb-analyze":  function(){ toggleTbDrop("Analyze"); },
            "tb-simulate": function(){ toggleTbDrop("Simulate"); },
            "tb-pick":     function(btn){
                        closeTbDrops();
                        emit({kind: "toolbar_open", top: btn.dataset.top,
                              sub: btn.dataset.sub || null});
                    },
            "tb-bg":       function(){ emit({kind: "toggle_bg"}); }
        };
        wrap.addEventListener("click", function(e){
            var btn = e.target.closest("[data-act]");
            if (!btn || btn.tagName === "A") return;
            e.preventDefault();
            e.stopPropagation();
            actions[btn.dataset.act](btn);
            if (["zin", "zout", "fit", "reset"].indexOf(btn.dataset.act) >= 0) {
                setTimeout(function(){ saveCam(stage); }, 600);
            }
        }, true);

        // ── Keyboard navigation ──────────────────────────────────────────────
        // Bound on the iframe document, with focus grabbed when the pointer
        // enters the viewer, so the arrows work without hunting for focus.
        var PAN_STEP = 30, ROT_STEP = 15;
        wrap.addEventListener("mouseenter", function(){
            try { window.focus(); wrap.focus({preventScroll: true}); } catch(e) {}
        });
        document.addEventListener("keydown", function(e){
            var tc = stage.trackballControls, handled = true, k = e.key;
            if      (k === "ArrowLeft")  { e.shiftKey ? tc.rotate(-ROT_STEP, 0) : tc.pan(-PAN_STEP, 0); }
            else if (k === "ArrowRight") { e.shiftKey ? tc.rotate( ROT_STEP, 0) : tc.pan( PAN_STEP, 0); }
            else if (k === "ArrowUp")    { e.shiftKey ? tc.rotate(0, -ROT_STEP) : tc.pan(0, -PAN_STEP); }
            else if (k === "ArrowDown")  { e.shiftKey ? tc.rotate(0,  ROT_STEP) : tc.pan(0,  PAN_STEP); }
            else if (k === "+" || k === "=") { stage.viewerControls.zoom(0.15); }
            else if (k === "-" || k === "_") { stage.viewerControls.zoom(-0.15); }
            else if (k === "r" || k === "R") { setMode("rotate"); }
            else if (k === "m" || k === "M") { setMode("pan"); }
            else if (k === "f" || k === "F") { actions.fit(); }
            else if (k === "s" || k === "S") { actions.spin(); }
            else if (k === "p" || k === "P") { actions.snap(); }
            else if (k === "l" || k === "L") { actions.pen(); }
            else if (k === "y" || k === "Y") { actions.style(); }
            else if (k === "o" || k === "O") { actions.solo(); }
            else if (k >= "1" && k <= "9" && CFG.structures[+k - 1]) {
                // Shift+n hides or shows; plain n makes that structure active.
                var pick = CFG.structures[+k - 1].sid;
                if (e.shiftKey) { toggleStructure(pick); }
                else if (pick !== CFG.activeSid) { emit({kind: "set_active", sid: pick}); }
            }
            else { handled = false; }
            if (handled) { e.preventDefault(); saveCam(stage); }
        });

        // ── Load every structure into one shared coordinate space ────────────
        function load(s) {
            var src = s.data !== null
                ? new Blob([s.data], {type: "text/plain"})
                : s.url;
            if (!src) return Promise.resolve(null);
            return stage.loadFile(src, {ext: "pdb", name: s.label}).then(function(comp){
                comps[s.sid] = comp;
                // A superposition is applied as the component's transform, so
                // the file on disk keeps its deposited coordinates and the fit
                // can be undone by simply not setting one.
                if (s.matrix) {
                    comp.setTransform(new NGL.Matrix4().fromArray(s.matrix));
                }
                s.reps.forEach(function(r){
                    comp.addRepresentation(r.type,
                        {sele: r.sele, color: r.color, opacity: r.opacity});
                });
                comp.setVisibility(s.visible);
                return comp;
            }).catch(function(e){
                console.error("NGL load error for " + s.label + ":", e);
                return null;
            });
        }

        // ── Interactions and measurements, as dashed lines ───────────────────
        // One Shape for the whole scene. dashedCylinder chops each cylinder
        // into segments, which is what makes a contact read as a contact
        // rather than as a covalent bond.
        function drawLines() {
            if (!CFG.lines || !CFG.lines.length) return;
            var shape = new NGL.Shape("interactions", {
                dashedCylinder: true,
                radialSegments: 8,
                openEnded: true,
                labelParams: {attachment: "middle-center", fontSize: 32,
                              backgroundColor: "black", backgroundOpacity: 0.55,
                              borderColor: "black", borderWidth: 0.25}
            });
            var withLabels = CFG.lines.length <= 40;
            CFG.lines.forEach(function(l){
                var radius = (l.type === "measurement") ? 0.09 : 0.06;
                shape.addCylinder(l.a, l.b, l.color, radius);
                if (withLabels && l.label) {
                    shape.addText(
                        [(l.a[0] + l.b[0]) / 2,
                         (l.a[1] + l.b[1]) / 2,
                         (l.a[2] + l.b[2]) / 2],
                        l.color, 1.3, l.label);
                }
            });
            var comp = stage.addComponentFromObject(shape);
            comp.addRepresentation("buffer");
        }

        // ── Pinned text labels ───────────────────────────────────────────────
        // A separate Shape from the measurement lines: these want a larger
        // font and no dashed-cylinder settings, and labelParams is per-shape.
        function drawLabels() {
            if (!CFG.labels || !CFG.labels.length) return;
            var shape = new NGL.Shape("annotations", {
                labelParams: {attachment: "middle-center", fontSize: 52,
                              backgroundColor: "black", backgroundOpacity: 0.6,
                              borderColor: "black", borderWidth: 0.3,
                              fixedSize: false}
            });
            CFG.labels.forEach(function(l){
                shape.addText(l.xyz, l.color, l.size, l.text);
            });
            stage.addComponentFromObject(shape).addRepresentation("buffer");
        }

        Promise.all(CFG.structures.map(load)).then(function(){
            drawLines();
            drawLabels();
            if (CFG.camTarget) {
                var target = comps[CFG.activeSid] ||
                             (CFG.structures[0] && comps[CFG.structures[0].sid]);
                if (target) { target.autoView(CFG.camTarget, 500); }
                else { stage.autoView(); }
            } else if (!restoreCam(stage)) {
                stage.autoView();
            }
            // Persist the camera on interaction so Streamlit reruns come back
            // to the same viewpoint.
            var el = stage.viewer.renderer.domElement;
            el.addEventListener("mouseup",  function(){ saveCam(stage); });
            el.addEventListener("wheel",    function(){ saveCam(stage); }, {passive: true});
            el.addEventListener("touchend", function(){ saveCam(stage); });
            setTimeout(function(){
                document.getElementById("hint").classList.add("fade");
            }, 12000);
        });

        window.addEventListener("resize", function(){ stage.handleResize(); });
    })();
    </script>
    """

    return (tpl
            .replace("__BG__",     bg)
            .replace("__BG_LABEL__", bg_label)
            .replace("__LEGEND__", legend_rows)
            .replace("__IXNKEY__", ixn_key)
            .replace("__CFG__",    cfg_json))


# ═══════════════════════════════════════════════════════════════════════════════
# Structure manager UI
# ═══════════════════════════════════════════════════════════════════════════════

def _chain_options(path: str) -> list:
    """
    Chain ids present in a structure, for the superposition chain pickers.

    Parsed from the PDB records rather than through MDAnalysis so the picker
    still populates when MDAnalysis is missing — the same reasoning as the
    sequence browser.
    """
    try:
        chains = _load_residues(path, Path(path).stat().st_mtime)
    except Exception:
        return []
    # Keep file order — chain A first — and drop chains with no protein in them,
    # since a superposition is fitted on protein residues.
    return [c for c, residues in chains.items()
            if c != "_" and any(r["kind"] == "protein" for r in residues)]


def _fit_selection(chain: str) -> str:
    """MDAnalysis selection for the whole protein, or one chain of it."""
    return "protein" if chain == "All chains" else f"protein and segid {chain}"


def structure_manager_ui() -> None:
    """
    Load, hide, remove and superpose structures.

    The agent can drive all of this too, but a superposition has parameters a
    language model tends to guess at — which structure moves, which chain to
    fit on, how residues are paired — and getting one of them wrong produces a
    confident-looking, wrong overlay. These widgets set them explicitly.
    """
    st.markdown("#### 🧱 Structures")

    # ── Add another structure ────────────────────────────────────────────────
    a1, a2, a3 = st.columns([3, 1.2, 1.2])
    with a1:
        new_id = st.text_input(
            "Add a structure", value="", key="add_struct_id",
            placeholder="PDB accession (e.g. 1UBQ) or a path to a local .pdb",
            label_visibility="collapsed",
        )
    with a2:
        add_clicked = st.button("➕ Add", key="add_struct_btn",
                                use_container_width=True,
                                help="Load this alongside the structures already open")
    with a3:
        replace_clicked = st.button("↻ Replace", key="replace_struct_btn",
                                    use_container_width=True,
                                    help="Load this as the only structure in the scene")

    if add_clicked or replace_clicked:
        name = new_id.strip()
        if not name:
            st.warning("Enter a PDB accession or a file path first.")
        else:
            if Path(name).exists():
                msg = tool_load_local(name, replace=replace_clicked)
            elif replace_clicked:
                msg = tool_fetch_structure(name)
            else:
                msg = tool_add_structure(name)
            st.session_state.superpose_msg = None
            st.toast(msg)
            st.rerun()

    upload = st.file_uploader("Or upload a .pdb / .cif file", type=["pdb", "ent", "cif"],
                              key="struct_upload", label_visibility="collapsed")
    if upload is not None:
        dest = STRUCTURES_DIR / upload.name
        # Re-uploading the same file on every rerun would otherwise re-register
        # it endlessly; the registry dedupes by label, so only the write repeats.
        dest.write_bytes(upload.getbuffer())
        if not find_structure(Path(upload.name).stem):
            _recolor_for_comparison()
            register_structure(Path(upload.name).stem, dest, source="local")
            st.rerun()

    if not structures():
        return

    # ── Loaded structures ────────────────────────────────────────────────────
    st.caption("Loaded — the active one is what selections and analysis apply to")
    active = st.session_state.active_sid
    for s in structures():
        c0, c1, c2, c3, c4 = st.columns([0.5, 3.4, 1.3, 1.1, 0.7])
        with c0:
            st.markdown(
                f"<div style='width:14px;height:14px;border-radius:50%;margin-top:9px;"
                f"background:{s['color']};box-shadow:0 0 0 1px #8888'></div>",
                unsafe_allow_html=True)
        with c1:
            label = f"**{s['pdb_id']}**"
            if s["fit"]:
                label += f" · fitted on {s.get('fit_reference', '?')} — {s['fit']}"
            elif s["source"] == "local":
                label += " · local file"
            elif s["source"] == "alphafold":
                label += " · AlphaFold predicted model"
            st.markdown(label)
        with c2:
            vis = st.checkbox("Visible", value=s["visible"], key=f"svis_{s['sid']}")
            s["visible"] = vis
        with c3:
            if s["sid"] == active:
                st.caption("active")
            elif st.button("Make active", key=f"sact_{s['sid']}"):
                st.session_state.active_sid = s["sid"]
                _sync_active()
                st.rerun()
        with c4:
            if st.button("🗑", key=f"sdel_{s['sid']}", help=f"Remove {s['pdb_id']}"):
                drop_structure(s["sid"])
                st.rerun()

    # ── Superposition ────────────────────────────────────────────────────────
    if len(structures()) < 2:
        st.caption("Add a second structure to superimpose them.")
        return

    st.divider()
    st.markdown("##### 🎯 Superimpose")
    if not sup.available():
        st.caption("Superposition needs MDAnalysis, which is not importable here.")
        return

    names = [s["pdb_id"] for s in structures()]
    f1, f2, f3 = st.columns(3)
    with f1:
        mobile = st.selectbox("Move this one", names, index=len(names) - 1,
                              key="sp_mobile")
    with f2:
        ref_options = [n for n in names if n != mobile]
        reference = st.selectbox("Onto this one", ref_options, index=0, key="sp_ref")
    with f3:
        method = st.selectbox(
            "Match residues by", sup.methods(), index=0, key="sp_method",
            help="'auto' pairs residues by number and falls back to a sequence "
                 "alignment when the two structures are numbered differently.",
        )

    mob_entry = find_structure(mobile)
    ref_entry = find_structure(reference)
    g1, g2 = st.columns(2)
    with g1:
        mob_chains = ["All chains"] + _chain_options(mob_entry["path"])
        mob_chain = st.selectbox(f"{mobile} — fit on", mob_chains, key="sp_mob_chain")
    with g2:
        ref_chains = ["All chains"] + _chain_options(ref_entry["path"])
        ref_chain = st.selectbox(f"{reference} — fit on", ref_chains, key="sp_ref_chain")

    b1, b2, b3 = st.columns([1.4, 1.4, 2.2])
    with b1:
        if st.button("🎯 Superimpose", key="sp_go", type="primary",
                     use_container_width=True):
            with st.spinner(f"Fitting {mobile} onto {reference}…"):
                tool_superpose(mobile, reference, method,
                               _fit_selection(mob_chain), _fit_selection(ref_chain))
            st.rerun()
    with b2:
        if st.button("↩︎ Undo fits", key="sp_clear", use_container_width=True):
            tool_clear_superposition("all")
            st.rerun()
    with b3:
        # The viewer transforms the component in place, so the only way to get
        # superposed coordinates out of the app is to bake the matrix into a copy.
        if mob_entry.get("matrix"):
            out = STRUCTURES_DIR / f"{mobile}_on_{mob_entry.get('fit_reference', 'ref')}.pdb"
            ok, detail = sup.write_transformed(mob_entry["path"], mob_entry["matrix"], out)
            if ok:
                st.download_button("⬇︎ Superposed .pdb", data=out.read_bytes(),
                                   file_name=out.name, mime="chemical/x-pdb",
                                   key="sp_dl", use_container_width=True)
            else:
                st.caption(detail)

    if st.session_state.superpose_msg:
        st.info(st.session_state.superpose_msg)


# ═══════════════════════════════════════════════════════════════════════════════
# Structure summary UI
# ═══════════════════════════════════════════════════════════════════════════════

def _md_table(headers: list, rows: list) -> str:
    """
    Render a Markdown table.

    Markdown rather than st.dataframe on purpose: the point of this panel is
    that the numbers can be lifted straight out of it, and a Markdown table
    survives a copy-paste into an email, a notebook or a manuscript, while a
    rendered dataframe does not.
    """
    if not rows:
        return ""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        cells = [str(c).replace("|", "\\|") for c in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def structure_summary_ui() -> None:
    """
    Composition report for the active structure: what it holds, in tables.

    Answers the questions people actually open a structure to ask — how many
    residues, which are modified, what the ligands are and which of them are
    only there because of the crystallography — and offers the whole thing as
    CSV, JSON or plain text so it can be pasted into a notebook or a paper.
    """
    entry = active_structure()
    if not entry:
        return
    summary = structure_summary()
    if summary is None:
        st.caption("Could not read this structure's file.")
        return

    t = summary["totals"]

    # ── Overview ─────────────────────────────────────────────────────────────
    if summary.get("title"):
        st.markdown(f"**{summary.get('id') or entry['pdb_id']}** — {summary['title']}")
    meta = []
    if summary.get("method"):
        meta.append(srep.pretty_method(summary["method"]))
    if summary.get("resolution"):
        meta.append(f"{summary['resolution']} Å")
    if summary.get("models", 0) > 1:
        meta.append(f"{summary['models']}-model NMR ensemble (first model counted)")
    if summary.get("organisms"):
        meta.append("; ".join(srep.pretty_organism(o) for o in summary["organisms"]))
    if meta:
        st.caption(" · ".join(meta))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Chains", t["chains"])
    m2.metric("Standard residues", t["standard"],
              help="The 20 standard amino acids")
    m3.metric("Non-standard", t["nonstandard"],
              help="Every residue that is not one of the 20 amino acids — modified "
                   "residues, nucleotides, ligands, cofactors, ions, buffer components. "
                   "Water is counted separately.")
    m4.metric("Ligands + cofactors", t["ligands"] + t["cofactors"],
              help="Copies of bound ligands and cofactors, excluding ions and additives")
    m5.metric("Waters", t["waters"])

    # The one-glance answer, before the tables that back it up.
    st.info(srep.as_brief(summary).split("\n")[-1])

    # ── Chains ───────────────────────────────────────────────────────────────
    st.markdown("**Chains**")
    st.markdown(_md_table(
        ["Chain", "Type", "Residues", "Numbering", "Missing", "Breaks"],
        [[c["chain"], c["type"], c["observed"], f"{c['first']}–{c['last']}",
          c["missing"] if c["missing"] is not None else "—",
          ", ".join(f"{a}→{b}" for a, b, _ in c["gaps"]) or "none"]
         for c in summary["chains"]],
    ))
    if t["missing"]:
        st.caption(f"{t['missing']} residues are in the deposited construct (SEQRES) but "
                   f"have no coordinates — disordered loops and termini that the "
                   f"experiment could not resolve.")

    # ── Standard vs non-standard ─────────────────────────────────────────────
    st.markdown("**Residue composition**")
    breakdown = [["Standard amino acids", t["standard"], "the 20 standard amino acids"]]
    for key, label in srep.NONSTANDARD_KINDS:
        n = sum(e["count"] for e in summary["nonstandard"] if e["kind"] == key)
        if n:
            breakdown.append([label, n, "non-standard"])
    breakdown.append(["Water", t["waters"], "counted separately"])
    st.markdown(_md_table(["Category", "Count", "Counts as"], breakdown))
    st.caption("**Standard** means one of the 20 amino acids. Every other residue — "
               "modified amino acids, nucleotides, ligands, cofactors, ions and buffer "
               "components — is non-standard. Water is counted on its own line so it "
               "does not swamp the total.")

    # ── Non-standard residues ────────────────────────────────────────────────
    if summary["nonstandard"]:
        st.markdown(f"**Non-standard residues ({t['nonstandard']})**")
        st.markdown(_md_table(
            ["Code", "What it is", "Copies", "Chains", "Name"],
            [[e["code"], srep.KIND_LABELS[e["kind"]]
              + (f" (replaces {e['parent']})" if e["kind"] == "modified_aa"
                 and e.get("parent") else ""),
              e["count"], ", ".join(e["chains"]) or "—",
              srep.pretty_chemical(e.get("name", "")) or "—"]
             for e in summary["nonstandard"]],
        ))
        if any(e["kind"] == "additive" for e in summary["nonstandard"]):
            st.caption("Rows marked *Crystallization additive* (glycerol, sulfate, PEG, "
                       "buffers) are there because of how the crystal was grown or frozen, "
                       "not because of the biology. The categories are a curated lookup, "
                       "so check anything surprising against the PDB entry.")
    else:
        st.caption("No non-standard residues — only standard amino acids and water.")

    # ── Export ───────────────────────────────────────────────────────────────
    stem = summary.get("id") or entry["pdb_id"]
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("⬇︎ CSV", srep.as_csv(summary), f"{stem}_summary.csv",
                           "text/csv", key="sum_csv", use_container_width=True,
                           help="One flat table — opens in Excel or pandas")
    with e2:
        st.download_button("⬇︎ JSON", srep.as_json(summary), f"{stem}_summary.json",
                           "application/json", key="sum_json", use_container_width=True,
                           help="The full report, sequences included")
    with e3:
        st.download_button("⬇︎ Text", srep.as_text(summary), f"{stem}_summary.txt",
                           "text/plain", key="sum_txt", use_container_width=True,
                           help="The same report as plain text")

    with st.expander("Plain-text version (copy-paste)", expanded=False):
        st.code(srep.as_text(summary), language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement, interaction and highlight UI
# ═══════════════════════════════════════════════════════════════════════════════
# Three separate panels rather than one with nested tabs. "What are the salt
# bridges" and "highlight the binding site" are questions people arrive with,
# and burying them two levels deep behind a control named after distances is
# how a feature ends up invisible.

SPEC_HELP = ("Name residues as `12`, `A/12`, `ARG12`, `12.CA`, a ligand code like "
             "`BEN`, or `1UBQ/A/12` to reach across structures.")


def measurement_ui() -> None:
    """Distances, contact shells and angles between named residues."""
    if not active_structure():
        return
    st.caption(SPEC_HELP)

    tab_dist, tab_contacts, tab_angle, tab_tors = st.tabs(
        ["Distance", "Contacts", "Angle", "Dihedral"])

    with tab_dist:
        d1, d2, d3 = st.columns([2, 2, 1])
        with d1:
            spec_a = st.text_input("From", key="ms_a", placeholder="e.g. HIS57")
        with d2:
            spec_b = st.text_input("To", key="ms_b", placeholder="e.g. ASP102")
        with d3:
            st.write("")
            go = st.button("📏 Measure", key="ms_go", use_container_width=True,
                           type="primary")
        if go:
            if not spec_a.strip() or not spec_b.strip():
                st.warning("Name both residues first.")
            else:
                st.session_state.ms_result = tool_measure_distance(spec_a, spec_b)
                st.rerun()
        if st.session_state.get("ms_result"):
            st.code(st.session_state.ms_result, language="text")

    with tab_contacts:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            target = st.text_input("Around", key="ms_target", placeholder="e.g. BEN")
        with c2:
            radius = st.slider("Cutoff (Å)", 2.0, 10.0, 4.0, 0.5, key="ms_radius")
        with c3:
            st.write("")
            find = st.button("🔍 Find", key="ms_find", use_container_width=True,
                             type="primary")
        water = st.checkbox("Include water", value=False, key="ms_water")
        if find:
            if not target.strip():
                st.warning("Name a residue or ligand first.")
            else:
                st.session_state.ms_contacts = tool_find_contacts(target, radius, water)
                st.rerun()
        if st.session_state.get("ms_contacts"):
            st.code(st.session_state.ms_contacts, language="text")

    with tab_angle:
        a1, a2, a3, a4 = st.columns([2, 2, 2, 1])
        with a1:
            pa = st.text_input("From", key="ms_ang_a")
        with a2:
            pb = st.text_input("Vertex", key="ms_ang_b")
        with a3:
            pc = st.text_input("To", key="ms_ang_c")
        with a4:
            st.write("")
            go_angle = st.button("📐 Angle", key="ms_ang_go",
                                 use_container_width=True, type="primary")
        if go_angle:
            if not all(x.strip() for x in (pa, pb, pc)):
                st.warning("Name all three residues first.")
            else:
                st.session_state.ms_angle = tool_measure_angle(pa, pb, pc)
                st.rerun()
        if st.session_state.get("ms_angle"):
            st.code(st.session_state.ms_angle, language="text")

    with tab_tors:
        t1, t2, t3, t4, t5 = st.columns([2, 2, 2, 2, 1.2])
        specs = []
        for col, key, label in ((t1, "ms_t_a", "1st"), (t2, "ms_t_b", "2nd"),
                                (t3, "ms_t_c", "3rd"), (t4, "ms_t_d", "4th")):
            with col:
                specs.append(st.text_input(label, key=key))
        with t5:
            st.write("")
            go_tors = st.button("🌀 Torsion", key="ms_t_go",
                                use_container_width=True, type="primary")
        st.caption("Four points, in order along the torsion. Name atoms for a "
                   "backbone angle — phi of residue 57 is `56.C`, `57.N`, "
                   "`57.CA`, `57.C`.")
        if go_tors:
            if not all(x.strip() for x in specs):
                st.warning("Name all four points first.")
            else:
                st.session_state.ms_tors = tool_measure_dihedral(*specs)
                st.rerun()
        if st.session_state.get("ms_tors"):
            st.code(st.session_state.ms_tors, language="text")

    drawn = st.session_state.measurements
    if drawn:
        st.divider()
        st.caption(f"Drawn in the viewer ({len(drawn)}) — white dashed lines")
        for m in drawn:
            r1, r2 = st.columns([6, 1])
            with r1:
                st.markdown(f"`{m['a_label']}` ↔ `{m['b_label']}` — "
                            f"closest {m['closest']:.2f} Å")
            with r2:
                if st.button("✕", key=f"ms_del_{m['id']}", help="Remove this line"):
                    st.session_state.measurements = [
                        x for x in drawn if x["id"] != m["id"]
                    ]
                    st.rerun()
        if st.button("🧹 Clear all measurements", key="ms_clear"):
            st.session_state.measurements = []
            st.rerun()


def interactions_ui() -> None:
    """Detect salt bridges, hydrogen bonds and the rest, and draw them."""
    if not active_structure():
        return
    st.caption("Leave *Around* blank to scan the whole structure, or name a "
               "residue, ligand or chain to see only what it interacts with.")

    i1, i2 = st.columns([2, 2])
    with i1:
        ixn_target = st.text_input(
            "Around (optional)", key="ix_target",
            placeholder="BEN · A/57 · chain B — blank scans everything")
    with i2:
        labels = {ixn.TYPE_LABELS[k]: k for k, _, _ in ixn.INTERACTION_TYPES}
        picked = st.multiselect(
            "Types", list(labels),
            default=[ixn.TYPE_LABELS[k] for k in ixn.DEFAULT_TYPES],
            key="ix_types")
    i3, i4, i5 = st.columns([2, 1.2, 1.2])
    with i3:
        ixn_radius = st.slider("Override cutoff (Å) — 0 keeps the standard criteria",
                               0.0, 8.0, 0.0, 0.5, key="ix_radius")
    with i4:
        ixn_water = st.checkbox("Include water", value=False, key="ix_water")
    with i5:
        st.write("")
        scan = st.button("🔗 Find interactions", key="ix_go",
                         use_container_width=True, type="primary")
    if scan:
        st.session_state.interaction_msg = tool_find_interactions(
            ixn_target, ",".join(labels[p] for p in picked), ixn_radius, ixn_water)
        st.rerun()

    if st.session_state.get("interaction_msg"):
        st.code(st.session_state.interaction_msg, language="text")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🎨 Highlight these residues", key="ix_hl",
                         use_container_width=True):
                tool_highlight("interactions", "ball+stick", "element")
                st.rerun()
        with b2:
            if st.button("🧹 Clear interaction lines", key="ix_clear",
                         use_container_width=True):
                st.session_state.interactions = []
                st.session_state.interaction_msg = None
                st.rerun()


def highlight_ui() -> None:
    """Highlight residues, a ligand or a region, and zoom to them."""
    if not active_structure():
        return
    st.caption(SPEC_HELP)

    h1, h2, h3, h4 = st.columns([2.4, 1.4, 1.4, 1])
    with h1:
        hl_target = st.text_input(
            "Highlight", key="hl_target",
            placeholder="57 · A/57 · BEN · 57, 102, 195 · 100-120")
    with h2:
        hl_style = st.selectbox("Style", NGL_REP_TYPES,
                                index=NGL_REP_TYPES.index("ball+stick"),
                                key="hl_style")
    with h3:
        hl_color = st.selectbox("Colour", list(NGL_COLOR_SCHEMES), key="hl_color")
    with h4:
        st.write("")
        hl_go = st.button("🎨 Show", key="hl_go", use_container_width=True,
                          type="primary")
    if hl_go:
        if not hl_target.strip():
            st.warning("Name what to highlight first.")
        else:
            st.toast(tool_highlight(hl_target, hl_style,
                                    NGL_COLOR_SCHEMES[hl_color]))
            st.rerun()
    st.caption("A range like `100-120` highlights the whole stretch; "
               "`interactions` highlights every residue from the last scan.")


# ═══════════════════════════════════════════════════════════════════════════════
# Representation manager UI
# ═══════════════════════════════════════════════════════════════════════════════

def _color_label(value: str) -> str:
    """Reverse-lookup a display label for a stored NGL colour value."""
    for label, val in NGL_COLOR_SCHEMES.items():
        if val == value:
            return label
    return value


SCOPE_ALL = "All structures"


def _scope_labels() -> list:
    """Choices for a layer's structure scope, all-structures first."""
    return [SCOPE_ALL] + [s["pdb_id"] for s in structures()]


def _scope_to_sid(label: str):
    """Turn a scope label back into a sid, or None for 'All structures'."""
    if label == SCOPE_ALL:
        return None
    s = find_structure(label)
    return s["sid"] if s else None


def _sid_to_scope(sid) -> str:
    """Turn a stored sid back into its scope label."""
    if not sid:
        return SCOPE_ALL
    s = find_structure(sid)
    return s["pdb_id"] if s else SCOPE_ALL


def representation_manager_ui() -> None:
    """
    Direct controls for the representation stack.

    The agent can also manipulate representations, but it is a language model
    and will sometimes pick the wrong tool or argument. These widgets write to
    st.session_state.representations directly, so changing a style never
    depends on the model interpreting a sentence correctly.
    """
    _ensure_rep_ids()

    st.markdown("#### 🎨 Representations")

    # ── One-click scenes ─────────────────────────────────────────────────────
    st.caption("Quick styles — these replace the whole stack")
    preset_names = list(REP_PRESETS.keys())
    for row_start in range(0, len(preset_names), 4):
        cols = st.columns(4)
        for col, name in zip(cols, preset_names[row_start:row_start + 4]):
            with col:
                if st.button(name, key=f"preset_{name}", use_container_width=True):
                    st.session_state.representations = [
                        dict(r, id=uuid.uuid4().hex[:8], visible=True)
                        for r in REP_PRESETS[name]
                    ]
                    st.rerun()

    st.divider()

    # ── Add a layer ──────────────────────────────────────────────────────────
    # The structure column only appears with more than one structure loaded:
    # with a single one, "All structures" and its name mean the same thing and
    # the extra control is just noise.
    multi = len(structures()) > 1
    st.caption("Add a layer")
    if multi:
        a1, a2, a3, a5, a4 = st.columns([2, 2, 2, 1.6, 1])
    else:
        a1, a2, a3, a4 = st.columns([2, 2, 2, 1])
        a5 = None
    with a1:
        new_type = st.selectbox("Style", NGL_REP_TYPES, key="add_rep_type")
    with a2:
        # Named selections made by the agent are offered alongside the presets.
        sel_options = list(SELECTION_PRESETS.keys())
        sel_options += [f"★ {n}" for n in st.session_state.selections.keys()]
        sel_options.append("✏️ Custom…")
        sel_choice = st.selectbox("Selection", sel_options, key="add_rep_sel")
    with a3:
        new_color_label = st.selectbox("Colour", list(NGL_COLOR_SCHEMES.keys()),
                                       key="add_rep_color")
    if a5 is not None:
        with a5:
            scope_choice = st.selectbox("Structure", _scope_labels(),
                                        key="add_rep_scope")
    else:
        scope_choice = SCOPE_ALL
    with a4:
        st.write("")
        add_clicked = st.button("➕ Add", key="add_rep_btn", use_container_width=True)

    custom_sel = ""
    if sel_choice == "✏️ Custom…":
        custom_sel = st.text_input(
            "Custom NGL selection", value="",
            placeholder="e.g. :A and 50-80, or ATP, or @1,2,3",
            key="add_rep_custom",
        )
    new_opacity = st.slider("Opacity", 0.0, 1.0, 1.0, 0.05, key="add_rep_opacity")

    if add_clicked:
        if sel_choice == "✏️ Custom…":
            selection = custom_sel.strip()
        elif sel_choice.startswith("★ "):
            selection = st.session_state.selections.get(sel_choice[2:], "all")
        else:
            selection = SELECTION_PRESETS[sel_choice]

        if not selection:
            st.warning("Enter a custom selection first.")
        else:
            add_representation(new_type, selection,
                               NGL_COLOR_SCHEMES[new_color_label],
                               transparency=1.0 - new_opacity,
                               sid=_scope_to_sid(scope_choice))
            st.rerun()

    # ── Active layers ────────────────────────────────────────────────────────
    reps = [r for r in st.session_state.representations if not r.get("_preview")]
    if not reps:
        st.info("No representations. Pick a quick style or add a layer above.")
        return

    st.divider()
    st.caption(f"Active layers ({len(reps)}) — edits apply immediately")

    to_delete = None
    for i, rep in enumerate(reps):
        rid = rep["id"]
        if multi:
            c0, c1, c2, c3, c4, c6, c5 = st.columns([0.6, 2, 2.6, 2, 2, 1.6, 0.6])
        else:
            c0, c1, c2, c3, c4, c5 = st.columns([0.6, 2, 2.6, 2, 2, 0.6])
            c6 = None

        with c0:
            st.write("")
            visible = st.checkbox(
                "👁", value=rep.get("visible", True), key=f"vis_{rid}",
                help="Show or hide this layer without deleting it",
                label_visibility="collapsed",
            )
        with c1:
            rtype = st.selectbox(
                "Style", NGL_REP_TYPES,
                index=NGL_REP_TYPES.index(rep["type"]) if rep["type"] in NGL_REP_TYPES else 0,
                key=f"type_{rid}", label_visibility="collapsed",
            )
        with c2:
            rsel = st.text_input(
                "Selection", value=rep["selection"], key=f"sel_{rid}",
                label_visibility="collapsed",
            )
        with c3:
            labels = list(NGL_COLOR_SCHEMES.keys())
            current = _color_label(rep.get("color", "element"))
            rcolor_label = st.selectbox(
                "Colour", labels,
                index=labels.index(current) if current in labels else 0,
                key=f"color_{rid}", label_visibility="collapsed",
            )
        with c4:
            ropacity = st.slider(
                "Opacity", 0.0, 1.0,
                round(1.0 - float(rep.get("transparency", 0.0)), 2), 0.05,
                key=f"op_{rid}", label_visibility="collapsed",
            )
        if c6 is not None:
            with c6:
                labels = _scope_labels()
                current_scope = _sid_to_scope(rep.get("sid"))
                rscope = st.selectbox(
                    "Structure", labels,
                    index=labels.index(current_scope) if current_scope in labels else 0,
                    key=f"scope_{rid}", label_visibility="collapsed",
                )
        else:
            rscope = None
        with c5:
            st.write("")
            if st.button("🗑", key=f"del_{rid}", help="Remove this layer"):
                to_delete = i

        # Write the widget values straight back into session state.
        rep.update({
            "visible": visible, "type": rtype, "selection": rsel.strip() or "all",
            "color": NGL_COLOR_SCHEMES[rcolor_label],
            "transparency": round(1.0 - ropacity, 3),
        })
        if rscope is not None:
            rep["sid"] = _scope_to_sid(rscope)

    if to_delete is not None:
        victim = reps[to_delete]["id"]
        st.session_state.representations = [
            r for r in st.session_state.representations if r.get("id") != victim
        ]
        st.rerun()

    d1, d2 = st.columns(2)
    with d1:
        if st.button("🧹 Clear all layers", use_container_width=True):
            st.session_state.representations = []
            st.rerun()
    with d2:
        if st.button("👁 Toggle all visibility", use_container_width=True):
            any_hidden = any(not r.get("visible", True) for r in reps)
            for r in reps:
                r["visible"] = any_hidden
            st.rerun()


@st.cache_data(show_spinner=False)
def _load_residues(pdb_path: str, mtime: float, schema: int = squ.SCHEMA_VERSION):
    """
    Parse and cache a structure's residues.

    mtime busts the cache on edit, schema on a change to the shape
    parse_structure_residues returns — see _cached_summary for why the second
    one is needed.
    """
    return squ.parse_structure_residues(pdb_path)


def sequence_browser_ui() -> None:
    """
    Sequence viewer and residue picker.

    Lists the residues actually present in the coordinates, with their real
    deposited numbering, so the user can read off which residues they want and
    turn them into a named selection without guessing at numbering. Selections
    made here land in st.session_state.selections, which is the same place the
    agent's `select` tool writes, so they immediately appear in the
    representation manager's selection dropdown.
    """
    path = st.session_state.pdb_path
    if not path or not Path(path).exists():
        return

    try:
        chains = _load_residues(path, Path(path).stat().st_mtime)
    except Exception as e:
        st.warning(f"Could not read sequence from {Path(path).name}: {e}")
        return
    if not chains:
        st.info("No residues found in this structure.")
        return

    # ── Chain picker, labelled with what each chain actually contains ────────
    labels = {}
    for cid, residues in chains.items():
        c = squ.chain_summary(residues)
        bits = []
        if c["protein"]:
            bits.append(f"{c['protein']} aa")
        if c["nucleic"]:
            bits.append(f"{c['nucleic']} nt")
        if c["hetero"]:
            bits.append(f"{c['hetero']} het")
        if c["water"]:
            bits.append(f"{c['water']} wat")
        labels[f"Chain {cid} — {', '.join(bits) or 'empty'}"] = cid

    chosen_label = st.selectbox("Chain", list(labels.keys()), key="seq_chain")
    chain_id = labels[chosen_label]
    residues = chains[chain_id]
    polymer = [r for r in residues if r["kind"] in ("protein", "nucleic")]

    # ── The sequence itself ─────────────────────────────────────────────────
    lines = squ.format_sequence_lines(polymer)
    if lines:
        lo, hi = polymer[0]["resseq"], polymer[-1]["resseq"]
        st.caption(
            f"Residues {lo}–{hi} as deposited. The number at the start of each "
            "row is that row's first residue — use these numbers below."
        )
        st.code("\n".join(lines), language=None)
    else:
        st.caption("This chain has no polymer residues.")

    # ── Ligands and other hetero groups, which are usually what gets picked ──
    het = [r for r in residues if r["kind"] == "hetero"]
    if het:
        st.caption("Hetero groups in this chain (click a button to select one)")
        hcols = st.columns(min(len(het), 6))
        for col, r in zip(hcols, het[:6]):
            with col:
                tag = f"{r['resname']}{r['resseq']}"
                if st.button(tag, key=f"het_{chain_id}_{r['resseq']}",
                             use_container_width=True):
                    name = tag.lower()
                    st.session_state.selections[name] = squ.ranges_to_ngl(
                        [(r["resseq"], r["resseq"])], chain_id)
                    st.session_state.camera_target = st.session_state.selections[name]
                    add_representation("ball+stick",
                                       st.session_state.selections[name],
                                       "element", 0.0,
                                       sid=st.session_state.active_sid)
                    st.rerun()

    # ── Build a selection from residue numbers ──────────────────────────────
    st.markdown("**Select residues**")
    if polymer:
        lo, hi = polymer[0]["resseq"], polymer[-1]["resseq"]
        rng = st.slider("Residue range", int(lo), int(hi),
                        (int(lo), int(min(lo + 20, hi))), key="seq_range")
    else:
        rng = None

    spec = st.text_input(
        "…or type specific residues (overrides the slider)",
        value="", placeholder="e.g. 74-80, 95, 100-110", key="seq_spec",
    )

    ranges, errors = squ.parse_residue_spec(spec) if spec.strip() else ([], [])
    if errors:
        st.warning("Could not read: " + ", ".join(errors))
    if not ranges and rng:
        ranges = [(rng[0], rng[1])]

    picked = squ.residues_in_ranges(residues, ranges)
    ngl_sel = squ.ranges_to_ngl(ranges, chain_id)

    if picked:
        preview = "".join(r["one"] for r in picked if r["kind"] != "water")[:60]
        st.caption(f"{len(picked)} residue(s) · NGL selection `{ngl_sel}`"
                   + (f" · {preview}" if preview else ""))
    else:
        st.caption("No residues match that specification.")

    # Live highlight. The viewer is built after this function runs, so the
    # preview layer appears in the same interaction that changed the picker.
    live = st.checkbox("Highlight this selection in the viewer", value=True,
                       key="seq_live")
    st.session_state.representations = [
        r for r in st.session_state.representations if not r.get("_preview")
    ]
    if live and picked:
        st.session_state.representations.append({
            "id": "preview", "type": "licorice", "selection": ngl_sel,
            "color": "magenta", "transparency": 0.0, "visible": True,
            "sid": st.session_state.active_sid, "_preview": True,
        })

    s1, s2, s3 = st.columns([2, 1, 1])
    with s1:
        sel_name = st.text_input("Name", value=f"sel_{chain_id}{ranges[0][0] if ranges else ''}",
                                 key="seq_name", label_visibility="collapsed")
    with s2:
        save = st.button("💾 Save selection", key="seq_save", use_container_width=True)
    with s3:
        showit = st.button("👁 Show + zoom", key="seq_show", use_container_width=True)

    if (save or showit) and picked:
        name = (sel_name or f"sel_{chain_id}").strip()
        st.session_state.selections[name] = ngl_sel
        if showit:
            add_representation("licorice", ngl_sel, "element", 0.0,
                               sid=st.session_state.active_sid)
            st.session_state.camera_target = ngl_sel
        st.rerun()
    elif (save or showit) and not picked:
        st.warning("Nothing selected — adjust the range or specification first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PyMOL ray-tracing UI
# ═══════════════════════════════════════════════════════════════════════════════

def label_ui() -> None:
    """
    Pin text onto regions of the structure, for figures that read on their own.

    A colour key tells someone which colour is which; it does not tell them
    that the middle third of the picture is the part inside the membrane. This
    panel writes that on the picture. The target vocabulary is deliberately the
    same as the measurement tools' — a residue is named the same way here as
    everywhere else — plus the three membrane slabs, which only appear for a
    structure that has been oriented and so has bilayer planes to be relative to.
    """
    st.markdown("#### 🏷️ Labels")
    st.caption("Write a name onto a domain, a residue range or a chain, so the "
               "figure can be read without a key.")

    entry = active_structure()
    if not entry:
        st.info("Load a structure first.")
        return

    names = [x["pdb_id"] for x in structures()]
    c1, c2 = st.columns([1.4, 1])
    with c1:
        which = st.selectbox("Structure", names,
                             index=names.index(entry["pdb_id"]) if entry["pdb_id"] in names else 0,
                             key="lbl_struct")
    target_entry = find_structure(which) or entry
    planes = _structure_planes(target_entry)
    with c2:
        if planes:
            st.caption(f"Bilayer at z {planes[0]:.1f} → {planes[1]:.1f} Å "
                       f"({planes[1] - planes[0]:.1f} Å thick)")
        else:
            st.caption("No bilayer planes — orient it under Simulate → Membrane "
                       "to unlock the membrane targets.")

    presets = ["Custom…"]
    if planes:
        presets = ["Transmembrane (TM)", "Extracellular", "Intracellular"] + presets

    p1, p2 = st.columns([1.2, 1.6])
    with p1:
        preset = st.selectbox("Region", presets, key="lbl_preset")
    defaults_for = {
        "Transmembrane (TM)": ("TM", "transmembrane", 30.0),
        "Extracellular":      ("extracellular", "extracellular", 30.0),
        "Intracellular":      ("intracellular", "intracellular", 30.0),
    }
    d_text, d_target, d_offset = defaults_for.get(preset, ("", "", 30.0))
    with p2:
        if preset == "Custom…":
            target = st.text_input("What to label", value="", key="lbl_target",
                                   placeholder="100-250, A/100-250, chain A, ARG12")
        else:
            target = d_target
            st.text_input("What to label", value=d_target, key="lbl_target_ro",
                          disabled=True)

    t1, t2, t3 = st.columns([2, 1.2, 1.2])
    with t1:
        text = st.text_input("Label text", value=d_text, key="lbl_text",
                             placeholder="e.g. TM, NBD1, catalytic site")
    with t2:
        color = st.selectbox("Colour", list(LABEL_COLORS.keys()), key="lbl_color")
    with t3:
        offset = st.number_input("Push out (Å)", 0.0, 120.0, float(d_offset), 5.0,
                                 key="lbl_offset",
                                 help="Moves the text away from the centre of the "
                                      "region so it is not hidden inside the "
                                      "structure. 0 pins it exactly on the region.")

    if st.button("➕ Add label", key="lbl_add", type="primary",
                 disabled=not (text.strip() and target.strip())):
        st.session_state.label_msg = tool_add_label(
            text, target, target_entry["pdb_id"], color, offset)
        st.rerun()

    if st.session_state.label_msg:
        msg = st.session_state.label_msg
        (st.success if as_tool_result(msg).ok else st.warning)(msg)

    rows = st.session_state.annotations
    if not rows:
        st.caption("No labels yet.")
        return

    st.divider()
    st.markdown(f"**{len(rows)} label(s)**")
    for a in list(rows):
        owner = find_structure(a["sid"])
        r1, r2, r3 = st.columns([2.6, 1, 0.6])
        with r1:
            st.markdown(
                f'<span style="color:{a["color"]};font-weight:600">{a["text"]}</span>'
                f' — {a["desc"] or a["target"]}', unsafe_allow_html=True)
        with r2:
            st.caption(owner["pdb_id"] if owner else "structure gone")
        with r3:
            if st.button("✕", key=f"lbl_del_{a['id']}", help="Remove this label"):
                st.session_state.annotations = [
                    x for x in st.session_state.annotations if x["id"] != a["id"]]
                st.session_state.label_msg = None
                st.rerun()

    if st.button("Clear all labels", key="lbl_clear"):
        st.session_state.annotations = []
        st.session_state.label_msg = None
        st.rerun()


def render_ui() -> None:
    """
    Ray-trace the active structure with PyMOL for a publication figure.

    This is a re-render, not a screenshot, and the difference is worth stating
    plainly in the UI: PyMOL rebuilds the scene from the representation stack
    in its own renderer, with its own camera and its own idea of what
    "spectrum" or "hydrophobicity" colouring looks like. It will not match the
    viewer pixel for pixel and cannot be made to. The viewer's own 📷 PNG
    button is the export that does match.
    """
    st.info(
        "**This re-renders the scene in PyMOL — it will not look identical to "
        "the viewer.** The camera angle, the shading and the exact colours of "
        "the by-residue and by-property schemes are PyMOL's, not NGL's, and "
        "interaction dashes and measurements are not carried over.\n\n"
        "For a picture of **exactly** what you see on screen, use the "
        "**📷 PNG** button in the viewer's toolbar — that is NGL exporting its "
        "own canvas at 3× resolution. Use this tab when you want ray-traced "
        "shadows and ambient occlusion for a figure, and are happy to set the "
        "view up again here."
    )
    if len(structures()) > 1:
        st.caption(
            f"It also renders the **active** structure only "
            f"(`{st.session_state.pdb_id}`), with its superposition applied, "
            f"while the viewer shows all {len(structures())} together."
        )
    if not PYMOL_AVAILABLE:
        st.caption(
            "PyMOL backend not found. Set `PYMOL_PYTHON` to an interpreter that "
            "can `import pymol2` (e.g. a conda env with `pymol-open-source`)."
        )
    else:
        rc1, rc2, rc3 = st.columns([2, 2, 1])
        with rc1:
            quality = st.selectbox(
                "Quality", ["draft", "publication"], index=0, key="render_quality",
                help="Draft renders in seconds. Publication enables fine cartoon "
                     "sampling, surface quality and ambient occlusion — minutes, "
                     "especially with a transparent surface.",
            )
        with rc2:
            res = st.selectbox(
                "Resolution", ["1200x900", "1600x1200", "2400x1800", "900x700"],
                index=0, key="render_res",
            )
        with rc3:
            st.write("")
            go = st.button("Render", type="primary", key="render_btn")

        if go:
            w, h = (int(x) for x in res.split("x"))
            with st.spinner("Ray tracing with PyMOL — this can take a while..."):
                tool_render_image(quality=quality, width=w, height=h)
            st.rerun()

        if st.session_state.render_msg:
            if st.session_state.render_path:
                st.success(st.session_state.render_msg)
            else:
                st.error(st.session_state.render_msg)

        png = st.session_state.render_path
        if png and Path(png).exists():
            st.image(png, use_container_width=True)
            with open(png, "rb") as fh:
                st.download_button(
                    "⬇️ Download PNG", fh.read(),
                    file_name=Path(png).name, mime="image/png",
                    key="render_dl",
                )
            st.caption(pymol_render.describe_backend())


# ═══════════════════════════════════════════════════════════════════════════════
# Preparation UI
# ═══════════════════════════════════════════════════════════════════════════════

ISSUE_ICONS = {"blocker": "🛑", "warn": "⚠️", "note": "ℹ️"}


def prepare_ui() -> None:
    """
    Clean a deposited structure into something a simulation can start from.

    The panel follows the order the decisions actually have to be made in:
    what is wrong with this file, which state of it do you mean, what is the
    structure for, and only then the fine control. The diagnosis comes first
    because most of these problems are invisible until something downstream
    fails in a way that does not mention them — twenty NMR states become one
    silently, an alternate conformation becomes a clash, a glycerol becomes a
    ligand in someone's binding-site analysis.
    """
    st.markdown("#### 🧼 Prepare")
    st.caption("Keep one state of an NMR ensemble, resolve alternate conformations, "
               "drop the crystallography, add hydrogens — and see what still needs "
               "deciding before Amber or Rosetta will do anything sensible with it.")

    if not structures():
        st.info("Load a structure first.")
        return

    names = [x["pdb_id"] for x in structures()]
    active = st.session_state.active_sid
    index = next((i for i, x in enumerate(structures()) if x["sid"] == active), 0)
    target = st.selectbox("Structure", names, index=index, key="prep_target")
    entry, finding = inspection_for(target)
    if not finding:
        st.caption("Could not read this structure's file.")
        return

    # ── What is wrong with it ────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("States", finding["models"],
              help="Models in the file. An NMR entry is an ensemble.")
    m2.metric("Residues", finding["residues"])
    m3.metric("Waters", finding["waters"])
    m4.metric("Hydrogens", finding["hydrogens"])
    m5.metric("Alt. conformations", finding["altloc_residues"],
              help="Residues modelled in more than one position.")

    if finding["issues"]:
        blockers = sum(1 for i in finding["issues"] if i["level"] == "blocker")
        with st.expander(
                f"{len(finding['issues'])} thing(s) to look at"
                + (f" — {blockers} would break a simulation" if blockers else ""),
                expanded=bool(blockers)):
            for issue in finding["issues"]:
                st.markdown(f"{ISSUE_ICONS[issue['level']]} **{issue['title']}**")
                st.caption(issue["detail"])
    else:
        st.success("Nothing obvious needs fixing before simulation.")

    # ── Which state ──────────────────────────────────────────────────────────
    model = None
    if finding["models"] > 1:
        st.markdown("**Which state to keep**")
        ids = finding["model_ids"]
        rep = finding.get("representative_model")
        default = ids.index(rep) if rep in ids else 0
        c1, c2 = st.columns([1, 3])
        with c1:
            model = st.selectbox("Model", ids, index=default, key="prep_model")
        with c2:
            if rep:
                st.caption(f"The depositors nominated model {rep} as the best "
                           "representative conformer. The states are otherwise "
                           "equally valid — they are the spread of the data, not "
                           "a ranking, so if the question is about flexibility, "
                           "the answer is the ensemble and not any one of them.")
            else:
                st.caption("The states are equally valid; this entry nominates no "
                           "representative. Model 1 is the convention, and every "
                           "tool that does not ask uses it.")

    # ── What for ─────────────────────────────────────────────────────────────
    st.markdown("**What is it for**")
    keys = list(prp.PROFILES)
    labels = [prp.PROFILES[k]["label"] for k in keys]
    choice = st.selectbox("Profile", labels, key="prep_profile")
    profile = keys[labels.index(choice)]
    st.caption(prp.PROFILES[profile]["note"])

    settings = {k: v for k, v in prp.PROFILES[profile].items()
                if k not in ("label", "note")}

    with st.expander("Fine control", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            settings["keep_waters"] = st.checkbox(
                "Keep waters", value=settings["keep_waters"], key="prep_wat")
            settings["keep_ions"] = st.checkbox(
                "Keep ions", value=settings["keep_ions"], key="prep_ion")
            settings["keep_ligands"] = st.checkbox(
                "Keep ligands and cofactors", value=settings["keep_ligands"],
                key="prep_lig")
        with c2:
            settings["keep_additives"] = st.checkbox(
                "Keep crystallisation additives", value=settings["keep_additives"],
                key="prep_add",
                help="Glycerol, PEG, sulfate, buffer components — there because of "
                     "how the crystal was grown, not because of the biology.")
            settings["mse_to_met"] = st.checkbox(
                "Selenomethionine → methionine", value=settings["mse_to_met"],
                key="prep_mse")
            settings["cys_to_cyx"] = st.checkbox(
                "Rename disulfide cysteines to CYX", value=settings["cys_to_cyx"],
                key="prep_cyx", help="Amber naming. tleap will not form the bond "
                                     "between two residues called CYS.")
        with c3:
            settings["altloc"] = st.selectbox(
                "Alternate conformations", ["occupancy", "first", "A", "B", "keep"],
                index=0, key="prep_alt",
                help="'occupancy' keeps the best-occupied conformation of each atom, "
                     "which is what the crystallographer would pick.")
            settings["renumber"] = st.checkbox(
                "Renumber residues from 1", value=settings["renumber"],
                key="prep_renum",
                help="Needed for the generated leap script's disulfide bonds to "
                     "land on the right residues. The original numbering is kept "
                     "in the mapping table you can download below.")
        chain_ids = sorted({c["chain"] for c in (finding["chains"] or [])})
        chosen_chains = st.multiselect(
            "Chains to keep (all if empty)", chain_ids, default=[], key="prep_chains")

    h1, h2 = st.columns([1.4, 2.6])
    with h1:
        add_h = st.checkbox("Add hydrogens (reduce)", key="prep_addh",
                            disabled=not prp.reduce_available())
    with h2:
        if prp.reduce_available():
            st.caption("`reduce` places them and picks the flip state of every "
                       "Asn, Gln and His to satisfy hydrogen bonding — the part "
                       "that geometry alone gets wrong.")
        else:
            st.caption("`reduce` was not found; it comes with AmberTools.")

    # ── Run ──────────────────────────────────────────────────────────────────
    if st.button("🧼 Prepare structure", key="prep_go", type="primary"):
        with st.spinner(f"Preparing {target}…"):
            record, err = prepare_structure(
                target, profile, model=model,
                chains=chosen_chains or None, add_h=add_h,
                **{k: v for k, v in settings.items()
                   if k in ("keep_waters", "keep_ions", "keep_ligands",
                            "keep_additives", "mse_to_met", "cys_to_cyx",
                            "altloc", "renumber", "hydrogens")})
        if err:
            st.session_state.prepare_msg = err
        else:
            label = f"{target}_prep"
            reg = register_structure(label, record["path"], source="local")
            if entry["sid"] != reg["sid"]:
                entry["visible"] = False
            st.session_state.prepare_msg = (
                f"Prepared {target} and loaded it as {label}; {target} itself is "
                f"hidden so the two do not overlap.")
        st.rerun()

    if st.session_state.prepare_msg:
        st.info(st.session_state.prepare_msg)

    record = st.session_state.prepared.get(target)
    if not record:
        # Preparing makes the cleaned copy the active structure, so coming back
        # to this tab usually lands on the copy rather than the original it was
        # made from. Match on the file so the report and the downloads are
        # still here either way.
        here = Path(entry["path"]).resolve()
        record = next((r for r in st.session_state.prepared.values()
                       if Path(r["path"]).resolve() == here), None)
    if not record:
        return

    # ── What came out ────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"**Prepared — {Path(record['path']).name}**")
    st.code(prp.report_text(record["report"]), language="text")
    for note in record["report"].get("notes", []):
        st.warning(note)

    engine = "rosetta" if record["profile"] == "rosetta" else "amber"
    st.markdown(f"**Ready for {engine.title()}?**")
    for check in prp.readiness(record["finding"], engine):
        st.markdown(("✅ " if check["ok"] else "⚠️ ") + check["text"])
        if check["detail"]:
            st.caption(check["detail"])

    d1, d2, d3 = st.columns(3)
    stem = Path(record["path"]).stem
    with d1:
        st.download_button("⬇︎ Prepared .pdb", Path(record["path"]).read_bytes(),
                           f"{stem}.pdb", "chemical/x-pdb", key="prep_dl",
                           use_container_width=True)
    with d2:
        if engine == "amber":
            script = prp.tleap_script(record["path"], record["finding"])
            st.download_button("⬇︎ tleap script", script, f"{stem}.leap",
                               "text/plain", key="prep_leap",
                               use_container_width=True,
                               help="loadpdb, the disulfide bond commands, "
                                    "solvation and saveamberparm")
    with d3:
        mapping = record["report"].get("renumber_map") or []
        if mapping:
            text = "chain\toriginal\ticode\tnew\n" + "\n".join(
                f"{c}\t{old}\t{icode}\t{new}" for c, old, icode, new in mapping)
            st.download_button("⬇︎ Numbering map", text, f"{stem}_numbering.tsv",
                               "text/tab-separated-values", key="prep_map",
                               use_container_width=True,
                               help="Original residue numbers against the new ones, "
                                    "so results can be reported in the numbering "
                                    "everyone else uses.")

    if engine == "amber":
        with st.expander("The generated tleap script", expanded=False):
            st.code(prp.tleap_script(record["path"], record["finding"]),
                    language="text")

    st.caption("Next: the **Membrane** tab embeds the prepared structure in a "
               "bilayer — start it from the prepared copy, not the original.")


# ═══════════════════════════════════════════════════════════════════════════════
# Quantum chemistry UI
# ═══════════════════════════════════════════════════════════════════════════════

def _region_centre_ui(entry: dict, key_prefix: str,
                      include_water: bool = False) -> tuple:
    """
    The shared "what is the centre, and how far out" controls.

    Returns (centre codes, [(key, resname, distance)], selected keys).
    """
    candidates = sim.ligand_candidates(entry["path"])
    codes = [c["code"] for c in candidates]
    c1, c2 = st.columns([1.4, 1.6])
    with c1:
        centre = st.selectbox(
            "Centre on", codes or ["— no ligand in this structure —"],
            key=f"{key_prefix}_centre",
            help="The molecule the region is built around. Usually the ligand.")
    with c2:
        radius = st.slider("Include residues within (Å)", 0.0, 12.0, 4.0, 0.5,
                           key=f"{key_prefix}_radius",
                           help="Measured atom to atom. 3–4 Å is the first "
                                "contact shell; beyond about 6 Å a QM region "
                                "grows faster than it gets better.")
    if not codes:
        st.caption("No ligand here to centre on. Prepare a structure that keeps "
                   "its ligand, or pick residues by hand below.")
        return [], [], []

    near = qm.residues_near(entry["path"], center_codes=[centre], radius=radius) \
        if radius > 0 else []
    labels = {}
    for key, resname, distance in near:
        labels[f"{resname} {key[1]}{key[2]} · {distance} Å"] = key
    default = list(labels)
    chosen = st.multiselect(f"{len(near)} residues within {radius} Å — "
                            "untick anything you do not want",
                            list(labels), default=default,
                            key=f"{key_prefix}_residues")
    return [centre], near, [labels[c] for c in chosen]


def quantum_ui() -> None:
    """
    Cut a QM model out of the structure and write the input file.

    The panel is arranged around the two things that decide whether the
    calculation means anything — what is in the region and what its charge is —
    rather than around the level of theory, which is the part everyone
    remembers to check anyway.
    """
    st.markdown("#### 🔬 Quantum mechanics")
    st.caption("Take the ligand, or the ligand and the residues around it, out of "
               "the structure as a cluster model — hydrogens added where bonds "
               "were cut — and write a Gaussian, ORCA or Psi4 input.")

    if not structures():
        st.info("Load and prepare a structure first.")
        return
    name, entry = _simulation_target("qm")
    if not entry:
        return

    finding = prp.inspect(entry["path"])
    if not finding["hydrogens"]:
        st.warning("This structure has no hydrogens. A QM calculation on heavy "
                   "atoms alone is meaningless — go back to **Prepare** and add "
                   "them before cutting a region out of it.")

    st.markdown("**1 · The region**")
    codes, near, chosen = _region_centre_ui(entry, "qm", include_water=True)
    c1, c2 = st.columns(2)
    with c1:
        mode = st.radio("Include", ["Side chains only", "Whole residues"],
                        horizontal=True, key="qm_mode",
                        help="Side chains cut at the Cα–Cβ bond are the usual "
                             "cluster model: the backbone is rarely what the "
                             "chemistry is about and it roughly triples the cost.")
    with c2:
        st.caption("Waters are offered in the residue list above when they are "
                   "within the cutoff — a water bridging the ligand and the "
                   "protein is part of the chemistry, and the rest are not.")

    if st.button("✂️ Build the region", key="qm_build", type="primary",
                 disabled=not codes):
        build_qm_region(name, entry, codes, chosen, mode == "Side chains only")
        st.rerun()

    region = st.session_state.qm_region
    if not region or region.get("name") != name:
        return

    st.divider()
    st.markdown("**2 · What came out**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Atoms", len(region["atoms"]))
    m2.metric("Residues", len(region["residues"]))
    m3.metric("Link atoms", len(region["links"]),
              help="Hydrogens standing in for the bonds that were cut.")
    m4.metric("Charge", f"{region['charge']:+d}")
    st.code(qm.region_report(region), language="text")
    for warning in region["warnings"]:
        st.warning(warning)

    v1, v2 = st.columns([1, 3])
    with v1:
        if st.button("👁 Show it", key="qm_show", use_container_width=True):
            path = qm.region_pdb(region, SIMULATIONS_DIR / f"{name}_qm_region.pdb")
            reg = register_structure(f"{name}_QM", path, source="local")
            add_representation("ball+stick", "all", "element", 0.0, reg["sid"])
            st.rerun()
    with v2:
        st.caption("Worth looking at before spending anything on it: a residue "
                   "half in, or a link atom inside a ring, is obvious in three "
                   "dimensions and invisible in a list of coordinates.")

    st.markdown("**3 · The calculation**")
    d1, d2, d3 = st.columns(3)
    with d1:
        charge = st.number_input("Charge", -10, 10, int(region["charge"]),
                                 key="qm_charge",
                                 help="Computed from the residues in the region. "
                                      "The ligand's own charge is not in that sum "
                                      "— add it here.")
        multiplicity = st.number_input("Multiplicity", 1, 11, 1, key="qm_mult")
    with d2:
        method = st.selectbox("Method", list(qm.METHODS), index=1, key="qm_method")
        st.caption(qm.METHODS[method])
    with d3:
        basis = st.selectbox("Basis set", list(qm.BASIS_SETS), key="qm_basis")
        st.caption(qm.BASIS_SETS[basis])

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        job = st.selectbox("Job", list(qm.JOB_TYPES), index=2, key="qm_job")
    with e2:
        solvent = st.selectbox("Solvent", list(qm.SOLVENT_MODELS), key="qm_solvent")
    with e3:
        processors = st.number_input("Cores", 1, 128, 8, key="qm_cpu")
    with e4:
        memory = st.number_input("Memory (GB)", 1, 512, 16, key="qm_mem")
    st.caption(qm.JOB_TYPES[job] + "  " + qm.SOLVENT_MODELS[solvent])
    freeze = st.checkbox("Freeze the link atoms during optimisation", value=True,
                         key="qm_freeze",
                         help="A cluster model optimised with everything free "
                              "relaxes into a shape the protein would never allow, "
                              "because the protein is not there.")

    files = qm_input_files(region, charge, multiplicity, method, basis, job, solvent,
                           processors, memory, freeze)
    tabs = st.tabs(["Gaussian", "ORCA", "Psi4", "xyz"])
    for tab, key in zip(tabs, ("region.gjf", "region.inp", "region.psi4",
                               "region.xyz")):
        with tab:
            st.code(files[key][:4000], language="text")
    st.download_button("⬇︎ All input formats as a .zip", sim.bundle(files),
                       f"{name}_qm.zip", "application/zip", key="qm_zip",
                       use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# QM/MM (ONIOM) UI
# ═══════════════════════════════════════════════════════════════════════════════

def oniom_ui() -> None:
    """
    Build an ONIOM QM/MM input from a structure that already has a topology.

    The dependency on the Amber tab is the point rather than an inconvenience:
    Gaussian's MM layer needs an atom type and a partial charge for every atom
    in it, and the only honest source of those is a force field that has
    actually been applied to this system, ligand parameters and all.
    """
    st.markdown("#### 🧩 QM/MM (ONIOM)")
    st.caption("A quantum region inside a molecular-mechanics protein. The MM "
               "layer's atom types and charges come from the Amber topology built "
               "on the Simulate tab, so the electrostatics are the ones the force "
               "field actually assigns.")

    build = st.session_state.amber_build
    if not (build and build.get("ok")):
        st.info("Build an Amber topology first — **Simulate → Amber**. ONIOM needs "
                "the prmtop for the MM layer's types and charges, and the PDB leap "
                "wrote alongside it.")
        return

    leap_pdb, paired, err = oniom_pairing(build)
    if not leap_pdb:
        st.warning(err)
        return
    st.caption(f"Using `{Path(build['prmtop']).name}` and `{leap_pdb.name}` — "
               f"{build['atoms']:,} atoms.")
    if err:
        st.error(err)
        return

    st.markdown("**1 · The QM layer**")
    st.caption("Residue numbering here is leap's, which renumbers from 1 — the "
               "names still tell you which residues these are.")
    entry = {"path": str(leap_pdb)}
    codes, near, chosen = _region_centre_ui(entry, "on")
    c1, c2 = st.columns(2)
    with c1:
        mode = st.radio("Include", ["Side chains only", "Whole residues"],
                        horizontal=True, key="on_mode")
    with c2:
        sphere = st.slider("MM layer radius (Å)", 0.0, 30.0, 15.0, 1.0,
                           key="on_sphere",
                           help="How much of the protein and solvent to keep "
                                "around the QM layer. 0 keeps everything, which "
                                "for a solvated system is tens of thousands of "
                                "atoms and more than Gaussian wants.")

    if st.button("🧩 Assign layers", key="on_build", type="primary",
                 disabled=not codes):
        build_oniom_model(build, paired, codes, chosen, mode == "Side chains only",
                          sphere)
        st.rerun()

    model = st.session_state.oniom_model
    if not model:
        return

    layered, boundary = model["layered"], model["boundary"]
    st.divider()
    st.markdown("**2 · The layers**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("QM atoms", layered["high"])
    m2.metric("MM atoms", layered["low"])
    m3.metric("Left out", layered["dropped"])
    m4.metric("Link atoms", len(boundary))
    st.code(oniom.report(layered, boundary), language="text")

    st.markdown("**3 · The calculation**")
    d1, d2 = st.columns(2)
    with d1:
        method = st.selectbox("ONIOM method", list(oniom.ONIOM_METHODS),
                              key="on_method")
        st.caption(oniom.ONIOM_METHODS[method])
    with d2:
        embedding = st.selectbox("Embedding", list(oniom.EMBEDDING), key="on_embed")
        st.caption(oniom.EMBEDDING[embedding][1])

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        job = st.selectbox("Job", ["opt", "sp", "opt freq"], key="on_job")
    with e2:
        charge_high = st.number_input(
            "QM charge", -10, 10, int(round(layered["charge_high"])), key="on_chg")
    with e3:
        mult_high = st.number_input("QM multiplicity", 1, 11, 1, key="on_mult")
    with e4:
        processors = st.number_input("Cores", 1, 128, 8, key="on_cpu")
    freeze_mm = st.checkbox("Freeze the MM layer", value=True, key="on_freeze",
                            help="Almost always right for a first calculation: the "
                                 "QM region relaxes inside a protein held where the "
                                 "crystallographer put it, and the optimisation "
                                 "finishes this decade.")

    files = oniom_input_files(model, method, embedding, job, charge_high, mult_high,
                              processors, freeze_mm)
    text = files["oniom.gjf"]

    with st.expander("The input file (first 200 lines)", expanded=False):
        st.code("\n".join(text.splitlines()[:200]), language="text")
    st.caption(f"{len(text.splitlines()):,} lines — geometry, layers, link atoms "
               "and the connectivity taken from the topology's own bond list.")
    st.download_button("⬇︎ ONIOM input", sim.bundle(files),
                       f"{build.get('name', 'system')}_oniom.zip",
                       "application/zip", key="on_zip", use_container_width=True)
    st.caption("PDB2ONIOM and the WebMO API do this too; neither is available here "
               "(WebMO needs an account), so the input is written natively — which "
               "also means every line of it can be read and checked.")


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation setup UI
# ═══════════════════════════════════════════════════════════════════════════════

def _simulation_target(key_prefix: str):
    """
    The structure to set a calculation up from, preferring a prepared one.

    Defaulting to whatever is active would usually mean the raw crystal
    structure, and quietly building a topology from an unprepared file --
    twenty NMR states, alternate conformations, glycerol and all -- is the
    failure this whole tab exists to prevent.

    Args:
        key_prefix: Distinct per calling panel, and not optional. Streamlit
                    registers every widget key once per script run, and every
                    tab renders in that run whether or not it is the one on
                    screen -- so a shared helper with a hard-coded key raises
                    a duplicate-key error the moment a second panel calls it.
    """
    names = [x["pdb_id"] for x in structures()]
    if not names:
        return None, None
    default = default_simulation_name()
    choice = st.selectbox("Structure", names, index=names.index(default),
                          key=f"{key_prefix}_target",
                          help="Set this up from a prepared structure, not a raw "
                               "download — the Prepare tab is the step before this one.")
    entry = find_structure(choice)
    if not choice.upper().endswith(PREPARED_SUFFIXES):
        st.caption("⚠️ This looks like an unprepared structure. Run it through the "
                   "Prepare tab first: alternate conformations and extra NMR states "
                   "become clashes and nonsense here, not error messages.")
    return choice, entry


def _ligand_parameter_ui(name: str, entry: dict) -> list:
    """
    Parameterise every ligand in the structure. Returns [(code, mol2, frcmod)].

    The net charge is asked for and never guessed. antechamber accepts any
    number it is given and builds a molecule with that many electrons, so a
    carboxylate entered as neutral produces parameters for a species that does
    not exist, and nothing downstream notices.
    """
    candidates = sim.ligand_candidates(entry["path"])
    if not candidates:
        st.caption("No ligands or cofactors here — nothing needs parameters.")
        return []

    if not sim.rdkit_available():
        st.warning(
            "RDKit is not installed in this environment, so ligands are "
            "protonated with `reduce` instead. reduce places hydrogens from "
            "idealised geometry and can leave a molecule chemically incomplete "
            "— on benzamidine it adds seven of the eight, and antechamber then "
            "parameterises a species that does not exist. Install it with "
            "`pip install rdkit` and the bond orders come from the PDB chemical "
            "dictionary instead.")

    params = []
    for lig in candidates:
        code = lig["code"]
        key = (name, code)
        stored = st.session_state.ligand_params.get(key)
        info = sim.chem_component(code) or {}

        with st.container(border=True):
            head = f"**{code}** — {srep.pretty_chemical(lig['name']) or 'unknown'}"
            if lig["copies"] > 1:
                head += f" · {lig['copies']} copies"
            st.markdown(head)
            bits = [f"{lig['atoms']} atoms in the structure ({lig['formula']})"]
            if info.get("formula"):
                bits.append(f"the dictionary says {info['formula']}")
            if info.get("charge") is not None:
                bits.append(f"formal charge {info['charge']}")
            st.caption(" · ".join(bits))

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                charge = st.number_input("Net charge", -6, 6,
                                         int((stored or {}).get("charge",
                                                                info.get("charge") or 0)),
                                         key=f"chg_{name}_{code}")
            with c2:
                method = st.selectbox("Charges", ["bcc", "gas"], key=f"cm_{name}_{code}",
                                      help="bcc is AM1-BCC, the standard for GAFF. "
                                           "gas is Gasteiger — instant, and much "
                                           "cruder; useful only to check a setup runs.")
            with c3:
                if st.button(f"Parameterise {code}", key=f"par_{name}_{code}",
                             use_container_width=True,
                             disabled=not sim.tools_available()["antechamber"]):
                    with st.spinner(f"antechamber on {code} — AM1-BCC takes "
                                    f"seconds to minutes…"):
                        parameterize_ligand(name, entry, code, charge, method)
                    st.rerun()

            if stored:
                if stored.get("error"):
                    st.error(stored["error"])
                else:
                    st.success(f"{code} parameterised — {Path(stored['mol2']).name} "
                               f"and {Path(stored['frcmod']).name}")
                    note = (stored.get("log") or "").splitlines()
                    if note and "hydrogens were added" in note[0]:
                        st.caption(note[0] + " Check that this is the protonation "
                                   "state you meant: the dictionary's state is the "
                                   "deposited one, not necessarily the one at your pH.")
                    if stored["guessed"]:
                        with st.expander(f"{len(stored['guessed'])} parameters were "
                                         "guessed by parmchk2", expanded=False):
                            st.caption("parmchk2 had no data for these and "
                                       "substituted the closest thing it could find. "
                                       "Normal for an unusual ligand, and a reason to "
                                       "look at the geometry after minimisation.")
                            st.code("\n".join(stored["guessed"][:20]), language="text")
                    params.append((code, stored["mol2"], stored["frcmod"]))
    return params


def amber_ui() -> None:
    """
    Build an Amber topology, and the inputs to run it.

    Everything on this tab actually runs: antechamber and parmchk2 for the
    ligand parameters, tleap for the topology. What comes out is a directory
    you can submit, not a description of one -- which matters because the
    steps that go wrong in an Amber setup go wrong silently, and the only way
    to know the disulfides were made and the box came out neutral is to build
    it and look.
    """
    st.markdown("#### ⚛️ Amber")
    tools = sim.tools_available()
    if not tools["tleap"]:
        st.warning("tleap was not found, so a topology cannot be built here.")
        st.caption(mem.backend_report())
        st.caption("AmberTools provides all of this: "
                   "`conda create -n ambertools -c conda-forge ambertools`.")
        return
    missing = [k for k, v in tools.items() if not v]
    if missing:
        st.caption("Not found: " + ", ".join(missing)
                   + " — ligand parameterisation needs antechamber and parmchk2.")

    if not structures():
        st.info("Load and prepare a structure first.")
        return
    name, entry = _simulation_target("amb")
    if not entry:
        return

    # ── Ligands ──────────────────────────────────────────────────────────────
    st.markdown("**1 · Ligand parameters**")
    st.caption("No protein force field contains a drug, a cofactor or a modified "
               "nucleotide. GAFF2 plus AM1-BCC charges is the usual answer, and "
               "this runs it for you.")
    params = _ligand_parameter_ui(name, entry)

    # ── System ───────────────────────────────────────────────────────────────
    st.markdown("**2 · Force field, box and ions**")
    c1, c2, c3 = st.columns(3)
    with c1:
        ff = st.selectbox("Protein force field", list(sim.PROTEIN_FFS),
                          key="amb_ff")
        st.caption(sim.PROTEIN_FFS[ff]["note"])
    with c2:
        recommended = sim.PROTEIN_FFS[ff]["water"]
        waters_list = list(sim.WATER_MODELS)
        water = st.selectbox("Water model", waters_list,
                             index=waters_list.index(recommended), key="amb_water")
        if water != recommended:
            st.caption(f"⚠️ {ff} was parameterised against {recommended}. Pairing it "
                       f"with {water} is a choice worth being able to defend.")
        else:
            st.caption(sim.WATER_MODELS[water])
    with c3:
        shape = st.selectbox("Box", list(sim.BOX_SHAPES), key="amb_shape")
        st.caption(sim.BOX_SHAPES[shape][1])

    d1, d2, d3 = st.columns(3)
    with d1:
        buffer_a = st.number_input("Buffer (Å)", 8.0, 25.0, 12.0, 0.5,
                                   key="amb_buffer",
                                   help="Clearance between the solute and the box "
                                        "edge. Below about 10 Å the protein starts "
                                        "seeing its own periodic image.")
    with d2:
        salt = st.number_input("Salt (M)", 0.0, 1.0, 0.15, 0.01, key="amb_salt")
    with d3:
        pair = st.selectbox("Ions", list(sim.ION_PAIRS), key="amb_ions")
        cation, anion, ion_note = sim.ION_PAIRS[pair]
        st.caption(ion_note)

    source_pdb = entry["path"]
    charge = sim.estimate_charge(source_pdb)
    est_waters = sim.estimate_waters(source_pdb, buffer_a, shape)
    bulk = sim.ion_counts(est_waters, salt, cation, anion, charge)
    st.caption(f"Estimated: about {est_waters:,} waters, solute charge {charge:+d} "
               f"(so {abs(charge)} {anion if charge > 0 else cation} counter-ions), "
               f"plus {bulk.get(cation, 0)} {cation} / {bulk.get(anion, 0)} {anion} "
               f"of bulk salt. tleap's real numbers appear after the build.")

    # ── Build ────────────────────────────────────────────────────────────────
    st.markdown("**3 · Build the topology**")
    if st.button("⚙️ Run tleap", key="amb_build", type="primary",
                 disabled=not tools["tleap"]):
        with st.spinner("tleap is building the system…"):
            build_amber_system(name, entry, params, ff, water, shape, buffer_a,
                               salt, pair)
        st.rerun()

    build = st.session_state.amber_build
    if build and build.get("name") == name:
        if build["ok"]:
            st.success(f"Built {Path(build['prmtop']).name} — "
                       f"{build['atoms']:,} atoms, {build['residues']:,} residues, "
                       f"{build['waters']:,} waters, net charge "
                       f"{build['charge']:+.3f}, {build['disulfides']} disulfide(s), "
                       f"periodic box: {'yes' if build['box'] else 'no'}.")
            if abs(build["charge"]) > 0.01:
                st.warning(f"The system is not neutral ({build['charge']:+.3f}). "
                           "Ewald summation will add a uniform background charge to "
                           "compensate, which is a physical artefact you do not want "
                           "— check the ion counts and the ligand's charge.")
            if build["ions"]:
                st.caption("Ions: " + ", ".join(f"{n} × {c}"
                                                for c, n in build["ions"].items()))
        else:
            st.error("tleap did not produce a topology.")
        if build["errors"]:
            with st.expander(f"{len(build['errors'])} error line(s) from leap",
                             expanded=not build["ok"]):
                st.code("\n".join(build["errors"][:20]), language="text")
        with st.expander("The leap script and its log", expanded=False):
            st.code(build.get("script", ""), language="text")
            st.code(build["log"][-4000:], language="text")

    # ── MD inputs ────────────────────────────────────────────────────────────
    st.markdown("**4 · Run inputs**")
    is_membrane = bool(mem.membrane_planes(source_pdb))
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        temperature = st.number_input("Temperature (K)", 100.0, 400.0, 300.0, 5.0,
                                      key="amb_temp")
    with e2:
        pressure = st.number_input("Pressure (bar)", 0.5, 5.0, 1.0, 0.1,
                                   key="amb_press")
    with e3:
        ns = st.number_input("Production (ns)", 1.0, 10000.0, 100.0, 10.0,
                             key="amb_ns")
    with e4:
        engine = st.selectbox("Engine", ["pmemd.cuda", "pmemd.MPI", "sander"],
                              key="amb_engine")
    if is_membrane:
        st.caption("This structure has membrane planes, so the equilibration and "
                   "production inputs use semi-isotropic pressure coupling with the "
                   "surface tension set to zero — coupling a bilayer isotropically "
                   "squeezes it.")

    files = amber_run_files(entry, build, params, temperature, pressure, ns, engine)

    with st.expander("Preview the production input", expanded=False):
        st.code(files["05_prod.in"], language="text")
    st.download_button(
        "⬇︎ Everything as a .zip", sim.bundle(files),
        f"{name}_amber.zip", "application/zip", key="amb_zip",
        use_container_width=True,
        help="Topology, coordinates, ligand parameters, the five run inputs and "
             "a script that runs them in order.")
    if not (build and build.get("ok")):
        st.caption("The archive has the run inputs but no topology yet — build one "
                   "above and download again.")


def gromacs_ui() -> None:
    """
    GROMACS run-parameter files, and an honest account of the topology problem.

    The .mdp files are the easy half and are generated properly. Getting a
    topology into GROMACS is the hard half, and this app cannot do it: the two
    converters are Python libraries that are not installed here, and one of
    them is broken by the same NumPy 2 problem that breaks pdb4amber. Saying
    so, with the commands, is more use than a button that fails.
    """
    st.markdown("#### 🧲 GROMACS")
    if not structures():
        st.info("Load and prepare a structure first.")
        return
    name, entry = _simulation_target("gmx")
    if not entry:
        return

    is_membrane = bool(mem.membrane_planes(entry["path"]))
    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = st.number_input("Temperature (K)", 100.0, 400.0, 300.0, 5.0,
                                      key="gmx_temp")
    with c2:
        pressure = st.number_input("Pressure (bar)", 0.5, 5.0, 1.0, 0.1,
                                   key="gmx_press")
    with c3:
        ns = st.number_input("Production (ns)", 1.0, 10000.0, 100.0, 10.0,
                             key="gmx_ns")
    if is_membrane:
        st.caption("Membrane planes found, so the pressure coupling is "
                   "semi-isotropic — a bilayer coupled isotropically is squeezed "
                   "in the plane it is supposed to stay flat in.")

    files = gromacs_run_files(entry, temperature, pressure, ns)

    st.markdown("**Getting a topology**")
    st.code(sim.GROMACS_CONVERSION_NOTE, language="text")
    with st.expander("Preview md.mdp", expanded=False):
        st.code(files["md.mdp"], language="text")
    st.download_button("⬇︎ mdp files as a .zip", sim.bundle(files),
                       f"{name}_gromacs.zip", "application/zip", key="gmx_zip",
                       use_container_width=True)


def rosetta_ui() -> None:
    """
    Rosetta job files for ligand docking.

    Generated, not run: Rosetta is not installed in this environment, and a
    tab that pretends otherwise would be worse than one that says so. The
    files themselves are complete and commented, and the ligand parameters
    start from the same mol2 the Amber tab produced.
    """
    st.markdown("#### 🎲 Rosetta")
    st.caption("Rosetta is not installed here, so these are written out rather "
               "than run — a complete set of job files with the commands to "
               "execute them.")
    if not structures():
        st.info("Load and prepare a structure first.")
        return
    name, entry = _simulation_target("ros")
    if not entry:
        return

    prepared = st.session_state.prepared.get(name.replace("_PREP", ""))
    if not prepared or prepared.get("profile") != "rosetta":
        st.caption("⚠️ Prepare this with the **Rosetta** profile first: Rosetta "
                   "rebuilds hydrogens itself, and a mixture of deposited and "
                   "rebuilt hydrogens is the usual cause of duplicate-atom errors.")

    candidates = sim.ligand_candidates(entry["path"])
    codes = [c["code"] for c in candidates] or ["LIG"]
    c1, c2, c3 = st.columns(3)
    with c1:
        code = st.selectbox("Ligand", codes, key="ros_lig")
    with c2:
        chain = st.text_input("Ligand chain", value="X", max_chars=1, key="ros_chain")
    with c3:
        nstruct = st.number_input("Structures", 10, 10000, 100, 10, key="ros_n",
                                  help="Docking output is a distribution. A hundred "
                                       "is a redocking sanity check; a real search "
                                       "into an apo site wants thousands.")

    files = rosetta_job_files(name, entry, code, chain, nstruct)
    if f"{code}.mol2" in files:
        st.caption(f"The mol2 from the Amber tab is included — molfile_to_params.py "
                   f"can read it directly.")

    with st.expander("Preview dock.xml", expanded=False):
        st.code(files["dock.xml"], language="xml")
    with st.expander("Preview dock.options", expanded=False):
        st.code(files["dock.options"], language="text")
    st.download_button("⬇︎ Rosetta job files as a .zip", sim.bundle(files),
                       f"{name}_rosetta.zip", "application/zip", key="ros_zip",
                       use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Membrane UI
# ═══════════════════════════════════════════════════════════════════════════════

def _composition_picker() -> tuple:
    """
    The lipid composition controls. Returns (lower, upper, label).

    Presets first, custom underneath, because the question "what is in my
    membrane" has a defensible stock answer almost every time and a bespoke
    one rarely.
    """
    labels = [p["label"] for p in mem.PRESETS] + ["Custom…"]
    choice = st.selectbox("Composition", labels, key="mem_preset")

    if choice != "Custom…":
        preset = mem.PRESETS[labels.index(choice)]
        st.caption(preset["note"])
        st.caption("Composition: " + mem.describe_composition(preset["lower"],
                                                              preset["upper"]))
        return preset["lower"], preset["upper"], preset["label"]

    catalog = mem.lipid_catalog()
    names = {l["code"]: l["name"] for l in catalog}
    codes = sorted(names)
    if not codes:
        st.caption("The lipid list could not be read from packmol-memgen.")
        return [("POPC", 1)], None, "POPC"

    def leaflet(side: str, default: list, key: str):
        picked = st.multiselect(
            f"{side} leaflet lipids", codes, default=default, key=key,
            format_func=lambda c: f"{c} — {names.get(c, '')[:52]}")
        spec = []
        if picked:
            cols = st.columns(min(len(picked), 4))
            for i, code in enumerate(picked):
                with cols[i % len(cols)]:
                    spec.append((code, st.number_input(
                        f"{code} parts", min_value=1, max_value=100, value=1,
                        step=1, key=f"{key}_{code}")))
        return spec

    lower = leaflet("Lower (cytoplasmic)", ["POPC"], "mem_lower")
    asymmetric = st.checkbox("Different upper leaflet", key="mem_asym",
                             help="Real plasma membranes are asymmetric — "
                                  "phosphatidylserine inside, sphingomyelin outside.")
    upper = leaflet("Upper (extracellular)", ["POPC"], "mem_upper") if asymmetric else None
    if not lower:
        st.caption("Pick at least one lipid for the lower leaflet.")
        return [("POPC", 1)], None, "POPC"
    return lower, upper, "custom composition"


def _membrane_backend_ok() -> bool:
    """Explain the backend, and say whether a build is possible at all."""
    if mem.available():
        return True
    st.warning("PACKMOL-Memgen was not found, so lipids cannot be packed here.")
    st.caption(mem.backend_report())
    st.caption("It ships with AmberTools: `conda create -n ambertools -c conda-forge "
               "ambertools`. This app finds it automatically in a conda environment, "
               "or set PACKMOL_MEMGEN to its path. Orientation alone needs only "
               "MEMEMBED (also from AmberTools) or an entry in OPM.")
    return False


def membrane_ui() -> None:
    """
    Put a membrane protein in a bilayer: orient it, then pack lipids around it.

    The panel is deliberately two steps, because they cost three orders of
    magnitude apart. Orienting takes seconds and answers most of what people
    open this for -- where the membrane sits, whether the protein really spans
    it, which surface is buried. Packing a bilayer is a background job of
    minutes to hours, and is only worth starting once the orientation looks
    right, which is why nothing here packs anything until the orientation has
    been done and shown.
    """
    st.markdown("#### 🧪 Membrane")
    st.caption("Orient a transmembrane protein in the bilayer, then pack a lipid "
               "membrane of your chosen composition around it — the PACKMOL-Memgen "
               "route, driven from here.")

    if not structures():
        st.info("Load a structure first.")
        return

    names = [x["pdb_id"] for x in structures()]
    active = st.session_state.active_sid
    index = next((i for i, x in enumerate(structures()) if x["sid"] == active), 0)
    target = st.selectbox("Structure", names, index=index, key="mem_target")
    entry = find_structure(target)
    record = st.session_state.oriented.get(target)
    if not record and mem.is_oriented(entry["path"]):
        # A structure that already carries bilayer planes — an OPM download, or
        # the oriented copy this panel made a moment ago, which appears in this
        # dropdown under its own name — needs no orientation step at all.
        record = {"path": entry["path"], "source": "the file itself",
                  "info": {"planes": mem.membrane_planes(entry["path"])}}
        st.session_state.oriented[target] = record

    # ── Step 1: orientation ──────────────────────────────────────────────────
    st.markdown("**1 · Orientation in the bilayer**")
    opm = mem.opm_lookup(target)
    if opm:
        st.caption(f"OPM has this entry as *{opm['name']}*"
                   + (f", hydrophobic thickness {opm['thickness']} ± "
                      f"{opm.get('thickness_error', '?')} Å" if opm.get("thickness") else "")
                   + ". That is a curated, published orientation — prefer it.")
    elif mem.is_oriented(entry["path"]):
        st.caption("This file already carries membrane planes, so it is oriented.")
    else:
        st.caption(f"{target} is not in OPM, so the orientation has to be computed "
                   "with MEMEMBED — a knowledge-based fit, not an experimental "
                   "result. Check it before building on it.")

    o1, o2, o3 = st.columns([1.4, 1.2, 1.4])
    with o1:
        source = st.selectbox("Orientation from", ["auto", "opm", "memembed"],
                              key="mem_source",
                              help="auto uses OPM when the entry is in it and "
                                   "MEMEMBED otherwise.")
    with o2:
        n_ter = st.selectbox("N-terminus", ["in", "out"], key="mem_nter",
                             help="Which side of the membrane residue 1 starts on. "
                                  "Only used by MEMEMBED, and it flips the protein "
                                  "if you get it wrong.")
    with o3:
        barrel = st.checkbox("Beta barrel", key="mem_barrel",
                             help="For porins and other outer-membrane barrels, "
                                  "which MEMEMBED fits with a different model.")

    if st.button("🧭 Orient in membrane", key="mem_orient", type="primary"):
        with st.spinner(f"Orienting {target} — MEMEMBED takes a few seconds to a "
                        f"few minutes…"):
            record, err = orient_structure(target, source, n_ter, barrel)
        if err:
            st.session_state.membrane_msg = err
        else:
            label = _load_oriented(entry, record)
            st.session_state.membrane_msg = (
                f"Oriented {target} using {record['source']} and loaded it as "
                f"{label}; {target} itself is hidden so the two poses do not "
                f"overlap. The grey slab is the bilayer.")
            st.rerun()

    if record:
        info = record.get("info") or {}
        planes = info.get("planes")
        bits = [f"Oriented by **{record['source']}**"]
        if planes:
            bits.append(f"planes at z = {planes[0]:.1f} and {planes[1]:.1f} Å "
                        f"(thickness {planes[1] - planes[0]:.1f} Å)")
        if info.get("energy"):
            bits.append(f"MEMEMBED energy {info['energy']}")
        st.success(" · ".join(bits))
        if info.get("models", 0) > 1:
            st.caption(f"The file held {info['models']} models; only the first was "
                       "used — a bilayer cannot be packed around an ensemble.")

    if st.session_state.membrane_msg:
        st.info(st.session_state.membrane_msg)

    st.divider()

    # ── Step 2: composition and build ────────────────────────────────────────
    st.markdown("**2 · Lipid composition**")
    if not _membrane_backend_ok():
        return

    lower, upper, comp_label = _composition_picker()

    with st.expander("Box, salt and output options", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            saltcon = st.number_input("Salt (M)", 0.0, 1.0, 0.15, 0.05,
                                      key="mem_salt",
                                      help="0 turns salt off. Counter-ions to "
                                           "neutralise the system are added either way.")
            cation = st.selectbox("Cation", ["K+", "Na+"], key="mem_cation")
        with c2:
            water = st.number_input("Water layer (Å)", 5.0, 60.0, 17.5, 2.5,
                                    key="mem_water",
                                    help="Thickness of water above and below the "
                                         "membrane. Too thin and the protein sees "
                                         "its own periodic image.")
            dist = st.number_input("Box padding (Å)", 5.0, 40.0, 15.0, 1.0,
                                   key="mem_dist")
        with c3:
            patch = st.number_input("Patch size x/y (Å, 0 = auto)", 0.0, 300.0, 0.0,
                                    10.0, key="mem_patch",
                                    help="Fix the membrane patch width instead of "
                                         "sizing it from the protein.")
            charmm = st.checkbox("CHARMM naming", key="mem_charmm",
                                 help="Write CHARMM lipid names instead of Amber's. "
                                      "Only a subset of lipids supports this.")
        keepligs = st.checkbox("Keep ligands and cofactors", value=True,
                               key="mem_keepligs",
                               help="Carry hetero atoms through the build. Off "
                                    "means a bound drug or cofactor is discarded.")
        parametrize = st.checkbox("Also run tleap to make Amber topology",
                                  key="mem_param",
                                  help="Produces .prmtop/.inpcrd for simulation. "
                                       "Adds time and can fail on unusual ligands — "
                                       "check the log if it does.")

    lipids, ratios = mem.composition_args(lower, upper)
    st.caption(f"PACKMOL-Memgen arguments: `-l {lipids} -r {ratios}`")

    if mem.parmed_broken():
        st.warning("ParmEd cannot be imported in the AmberTools environment, so the "
                   "step that finds lipid tails threaded through aromatic rings will "
                   "be skipped — the build still finishes, but check the system by "
                   "eye before simulating it. Fix with "
                   "`conda run -n ambertools pip install -U parmed`.")

    job = st.session_state.membrane_job
    running = bool(job and job.get("status") == "running")

    b1, b2 = st.columns([1.6, 1])
    with b1:
        if st.button("🧱 Build membrane system", key="mem_build", type="primary",
                     disabled=running, use_container_width=True):
            if not record:
                st.session_state.membrane_msg = (
                    "Orient the protein first — packing lipids around an "
                    "arbitrarily rotated protein produces a membrane through "
                    "its side.")
                st.rerun()
            stamp = time.strftime("%Y%m%d-%H%M%S")
            job = mem.start_build(
                record["path"], MEMBRANES_DIR / f"{target}_{stamp}", lipids, ratios,
                label=target, preoriented=True, keep_ligands=keepligs,
                salt=saltcon > 0, salt_concentration=saltcon, salt_cation=cation,
                water_thickness=water, boundary_distance=dist, patch_xy=patch,
                charmm_output=charmm, parametrize=parametrize)
            st.session_state.membrane_job = job
            st.session_state.membrane_msg = (
                f"Started packing {comp_label} around {target}."
                if job["status"] == "running" else job["error"])
            st.rerun()
    with b2:
        st.caption("Minutes to hours. It runs in the background — you can keep "
                   "working, and it survives page reruns.")

    # ── Step 3: the running or finished job ──────────────────────────────────
    if not job:
        return

    st.divider()
    mem.poll(job)
    st.markdown(f"**3 · Build — {job['label']}**")
    minutes = job.get("elapsed", 0) / 60

    if job["status"] == "running":
        progress = job.get("progress")
        if progress is not None:
            st.progress(progress, text=f"{job['stage']} · {minutes:.1f} min elapsed")
        else:
            st.caption(f"{job['stage']} · {minutes:.1f} min elapsed")
        r1, r2, r3 = st.columns([1, 1, 2])
        with r1:
            if st.button("↻ Refresh", key="mem_refresh", use_container_width=True):
                st.rerun()
        with r2:
            if st.button("✕ Cancel", key="mem_cancel", use_container_width=True):
                mem.cancel(job)
                st.rerun()
        with r3:
            if st.checkbox("Auto-refresh every 5 s", key="mem_auto",
                           help="Re-runs the page while the build is going. Turn it "
                                "off if the viewer feels sluggish."):
                time.sleep(5)
                st.rerun()
    elif job["status"] == "failed":
        st.error(f"The build failed after {minutes:.1f} minutes: {job['error']}")
    elif job["status"] == "cancelled":
        st.warning("The build was cancelled.")
    else:
        summary = job.get("summary") or mem.system_summary(job["output"])
        st.success(f"Finished in {minutes:.1f} minutes.")
        st.write(mem.summary_text(summary))
        if summary.get("lipids"):
            st.markdown(_md_table(
                ["Component", "Count"],
                [[f"{c} (head group)", n] for c, n in summary["lipids"].items()]
                + [[c, n] for c, n in summary["sterols"].items()]
                + [["water", f"{summary['waters']:,}"]]
                + [[c, n] for c, n in summary["ions"].items()]))
            st.caption("Amber's Lipid21 splits every phospholipid into a head group "
                       "and two acyl tails, so lipids are counted by their phosphorus "
                       "atom rather than by residue — the residue count would be "
                       "three times too high.")

        l1, l2, l3 = st.columns(3)
        with l1:
            if st.button("👁 Load into viewer", key="mem_load",
                         use_container_width=True,
                         help="Loads a copy with the water removed — the full system "
                              "is mostly water and too heavy for the browser."):
                view_path, dropped = mem.viewer_copy(job["output"])
                size_mb = Path(view_path).stat().st_size / 1e6
                # A local file is embedded in the viewer's HTML as text, so its
                # size is paid on every rerun of the page. Past roughly fifteen
                # megabytes that stops being a slow viewer and starts being a
                # browser tab that does not come back.
                if size_mb > 15:
                    st.session_state.membrane_msg = (
                        f"The packed system is {size_mb:.0f} MB even without its "
                        f"water — too large to embed in the browser. It is on disk "
                        f"at {view_path}; open that in PyMOL or VMD instead.")
                else:
                    reg = register_structure(f"{job['label']}_membrane", view_path,
                                             source="local")
                    _membrane_layers(reg["sid"], planes=False)
                    st.session_state.membrane_msg = (
                        f"Loaded the packed system ({size_mb:.1f} MB) without its "
                        f"{dropped:,} water atoms.")
                st.rerun()
        with l2:
            try:
                st.download_button("⬇︎ System .pdb", Path(job["output"]).read_bytes(),
                                   f"{job['label']}_membrane.pdb", "chemical/x-pdb",
                                   key="mem_dl", use_container_width=True)
            except Exception:
                st.caption("The output file could not be read.")
        with l3:
            st.caption(f"On disk: `{job['output']}`")

    with st.expander("Build log and command", expanded=False):
        st.code(job["cmd"], language="bash")
        st.code(mem.log_tail(job, 30) or "(no output yet)", language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# Protein finder UI
# ═══════════════════════════════════════════════════════════════════════════════

def _protein_caption(prof: dict) -> str:
    """The identity line under the protein's name."""
    bits = [prof["accession"]]
    if prof["gene"]:
        bits.append(f"gene {prof['gene']}")
    bits.append(prof["organism"])
    bits.append(f"{prof['length']} aa")
    if prof["mass"]:
        bits.append(f"{prof['mass'] / 1000:.0f} kDa")
    bits.append("Swiss-Prot" if prof["reviewed"] else "unreviewed (TrEMBL)")
    return " · ".join(bits)


def _load_protein_structure(pdb_id: str, replace: bool) -> None:
    """Load a chosen entry from the protein table and refresh the page."""
    msg = tool_replace_scene(pdb_id) if replace else tool_add_structure(pdb_id)
    st.session_state.superpose_msg = None
    st.toast(msg)
    st.rerun()


def protein_finder_ui() -> None:
    """
    Search a protein by name, then choose which of its structures to open.

    The layer above the viewer. Everywhere else in this app you start from a
    PDB accession, which assumes the choice has already been made; this panel
    is where it gets made. A human protein typically has dozens of depositions
    that differ in what part of the sequence they contain, by which technique,
    at what resolution and with what bound — and those differences, not the
    accession codes, are what someone is actually choosing between. So the
    table leads with coverage and method, groups the entries by which region
    of the protein they solve, and says plainly when a structure is too large
    for the PDB file format this app reads.
    """
    st.markdown("#### 🔎 Find a protein")
    st.caption("Search UniProt by name, gene symbol or accession — then pick the "
               "structure you want from everything deposited for it.")

    c1, c2, c3 = st.columns([3, 1.3, 1])
    with c1:
        name = st.text_input(
            "Protein", value=st.session_state.protein_query, key="prot_query_in",
            placeholder="e.g. insulin receptor, EGFR, TP53, P06213",
            label_visibility="collapsed")
    with c2:
        species = st.selectbox("Organism", ["human", "mouse", "rat", "yeast",
                                            "E. coli", "any"], index=0,
                               key="prot_species", label_visibility="collapsed")
    with c3:
        go = st.button("Search", key="prot_go", type="primary",
                       use_container_width=True)

    if go:
        organism = "" if species == "any" else species
        with st.spinner(f"Looking up {name or '…'} in UniProt…"):
            hits, err = _cached_protein_search(name.strip(), organism, True,
                                               pacc.SCHEMA_VERSION)
        if err:
            st.session_state.protein_msg = err
        elif not hits:
            st.session_state.protein_msg = (
                f"No UniProt entry matches '{name}'"
                + (f" in {species}." if organism else ".")
                + " Try the gene symbol, or set the organism to *any*.")
        else:
            st.session_state.protein_msg = None
            st.session_state.protein_query = name.strip()
            st.session_state.protein_hits = hits
            with st.spinner(f"Collecting structures for {hits[0]['accession']}…"):
                prof, perr = _cached_protein_profile(hits[0]["accession"],
                                                     pacc.SCHEMA_VERSION)
            st.session_state.protein_profile = prof
            st.session_state.protein_msg = perr
            if prof:
                set_focus(prof)

    if st.session_state.protein_msg:
        st.warning(st.session_state.protein_msg)

    hits = st.session_state.protein_hits
    if not hits:
        return

    # ── Which entry? ─────────────────────────────────────────────────────────
    # A name almost always matches a family, and the member people mean is not
    # always the one UniProt ranks first, so the alternatives stay one click
    # away rather than being decided silently.
    if len(hits) > 1:
        labels = [f"{h['accession']} — {h['protein_name']}"
                  + (f" ({h['gene']})" if h["gene"] else "")
                  + f" · {h['n_structures']} structures" for h in hits]
        current = st.session_state.protein_profile
        index = next((i for i, h in enumerate(hits)
                      if current and h["accession"] == current["accession"]), 0)
        chosen = st.selectbox(f"{len(hits)} UniProt entries match — which one?",
                              labels, index=index, key="prot_pick")
        acc = hits[labels.index(chosen)]["accession"]
        if not current or current["accession"] != acc:
            with st.spinner(f"Collecting structures for {acc}…"):
                prof, perr = _cached_protein_profile(acc, pacc.SCHEMA_VERSION)
            st.session_state.protein_profile = prof
            st.session_state.protein_msg = perr
            if prof:
                set_focus(prof)
            st.rerun()

    prof = st.session_state.protein_profile
    if not prof:
        return

    # ── Identity ─────────────────────────────────────────────────────────────
    st.markdown(f"### {prof['protein_name']}")
    st.caption(_protein_caption(prof))
    if prof["alt_names"]:
        st.caption("Also called: " + ", ".join(prof["alt_names"][:5]))

    t = prof["totals"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Structures", t["structures"])
    m2.metric("X-ray", t["xray"])
    m3.metric("Cryo-EM", t["em"])
    m4.metric("NMR", t["nmr"])
    m5.metric("Sequence covered", f"{t['sequence_covered_pct']:.0f}%",
              help="Share of the canonical UniProt sequence that appears in at "
                   "least one deposited structure.")

    if not prof["structures"]:
        st.info(f"{prof['protein_name']} has no experimental structure in the PDB. "
                "A predicted model from AlphaFold DB may exist — this app loads "
                "experimental PDB entries only.")
        _protein_exports(prof)
        return

    st.info(pacc.as_brief(prof))
    if prof["note"]:
        st.caption(f"⚠️ {prof['note']}")

    # ── What part of the protein is solved ───────────────────────────────────
    with st.expander(f"Coverage — {len(prof['regions'])} region(s) of the sequence "
                     f"have structures", expanded=False):
        region_rows = pacc.region_table_rows(prof)
        st.markdown(_md_table(pacc.REGION_HEADERS, region_rows[:12]))
        if len(region_rows) > 12:
            st.caption(f"{len(region_rows) - 12} further regions, each with only a "
                       "handful of structures, are left out of this table — they are "
                       "still in the Region filter and the CSV.")
        st.caption("Structures are grouped when they cover substantially the same "
                   "stretch of sequence, so a domain construct and a full-length "
                   "deposition stay in different rows — that distinction is usually "
                   "the real choice.")
        gaps = t["uncovered_spans"]
        if gaps:
            st.caption("No structure covers: " +
                       ", ".join(f"{a}–{b}" for a, b in gaps[:8]) +
                       (" …" if len(gaps) > 8 else ""))
        if prof["domains"]:
            st.markdown("**Domains (UniProt)**")
            st.markdown(_md_table(["Domain", "Residues"],
                                  [[d["name"], f"{d['start']}–{d['end']}"]
                                   for d in prof["domains"]]))

    if prof["function"]:
        with st.expander("What this protein does (UniProt)", expanded=False):
            st.write(prof["function"])
            if prof["subunit"]:
                st.markdown("**Assembly**")
                st.write(prof["subunit"])
            if prof["isoforms"]:
                st.markdown("**Isoforms** — residue numbering in a structure follows "
                            "whichever isoform the construct came from")
                st.markdown(_md_table(
                    ["Isoform", "UniProt id", "Canonical"],
                    [[i["name"], i["id"], "✓" if i["canonical"] else ""]
                     for i in prof["isoforms"]]))

    # ── Filters ──────────────────────────────────────────────────────────────
    st.markdown("**Structures**")
    f1, f2, f3, f4 = st.columns([1.3, 1.5, 1.4, 1.4])
    with f1:
        methods = ["All"] + sorted(t["methods"])
        method = st.selectbox("Method", methods, key="prot_method")
    with f2:
        resolutions = [r["resolution"] for r in prof["structures"]
                       if r["resolution"] is not None]
        worst = max(resolutions) if resolutions else 10.0
        max_res = st.slider("Resolution at least (Å)", 1.0, float(max(2.0, worst)),
                            float(max(2.0, worst)), 0.1, key="prot_res",
                            help="Structures with no resolution (NMR) drop out as "
                                 "soon as this is tightened.")
    with f3:
        region_labels = ["Whole protein"] + [
            f"{r['label'][:28]} ({r['start']}–{r['end']})" for r in prof["regions"]]
        region_pick = st.selectbox("Region", region_labels, key="prot_region")
    with f4:
        order = st.selectbox("Best means", ["balanced", "coverage", "resolution"],
                             key="prot_order",
                             help="balanced: loadable, then more of the protein, "
                                  "then sharper. coverage: the most complete. "
                                  "resolution: the sharpest.")
    g1, g2, g3 = st.columns(3)
    with g1:
        ligands_only = st.checkbox("With a ligand bound", key="prot_lig",
                                   help="Ligands and cofactors only — crystallisation "
                                        "additives such as glycerol or sulfate do not "
                                        "count as a bound ligand.")
    with g2:
        wt_only = st.checkbox("Wild-type only", key="prot_wt",
                              help="Drop constructs carrying engineered mutations.")
    with g3:
        loadable_only = st.checkbox("Loadable here only", value=True, key="prot_loadable",
                                    help="Hide entries deposited as mmCIF only — too "
                                         "large for the PDB file format this app reads.")

    region = None
    if region_pick != "Whole protein":
        r = prof["regions"][region_labels.index(region_pick) - 1]
        region = (r["start"], r["end"])

    rows = pacc.filter_structures(
        prof["structures"],
        method="" if method == "All" else method,
        max_resolution=None if not resolutions or max_res >= max(2.0, worst) else max_res,
        ligands_only=ligands_only, wild_type_only=wt_only,
        region=region, loadable_only=loadable_only)
    ranked = pacc.rank_structures(rows, order)

    if not ranked:
        st.caption("No structure matches these filters.")
        _protein_exports(prof)
        return

    show_all = st.checkbox(f"Show all {len(ranked)}", key="prot_all",
                           value=len(ranked) <= 15)
    shown = ranked if show_all else ranked[:15]
    st.markdown(_md_table(pacc.STRUCTURE_HEADERS, pacc.structure_table_rows(shown)))
    st.caption(f"{len(ranked)} of {t['structures']} structures shown, best first. "
               "**Coverage** is the share of the canonical UniProt sequence the entry "
               "contains; **Mutations** counts engineered point mutations in the "
               "construct. A ⚠ marks an entry with no PDB-format file.")

    # ── Open one ─────────────────────────────────────────────────────────────
    st.markdown("**Open a structure**")
    o1, o2, o3 = st.columns([2.5, 1.2, 1.2])
    with o1:
        options = [f"{r['pdb_id']} — {r['method']}"
                   + (f", {r['resolution']:.2f} Å" if r["resolution"] is not None else "")
                   + f", {r['coverage_pct']:.0f}% coverage" for r in ranked]
        pick = st.selectbox("Structure", options, index=0, key="prot_open",
                            label_visibility="collapsed")
    chosen = ranked[options.index(pick)]
    with o2:
        if st.button("➕ Add", key="prot_add", use_container_width=True,
                     disabled=not chosen["pdb_format"],
                     help="Load it alongside whatever is already in the scene"):
            _load_protein_structure(chosen["pdb_id"], replace=False)
    with o3:
        if st.button("↻ Replace", key="prot_replace", use_container_width=True,
                     disabled=not chosen["pdb_format"],
                     help="Load it as the only structure in the scene"):
            _load_protein_structure(chosen["pdb_id"], replace=True)

    if not chosen["pdb_format"]:
        st.caption(f"{chosen['pdb_id']} has {chosen['atom_count'] or '?'} atoms and is "
                   "deposited as mmCIF only — the PDB file format cannot hold it, and "
                   "every reader in this app parses PDB records. Pick another entry, "
                   "or open a single chain of it from RCSB by hand.")
    else:
        detail = [f"**{chosen['pdb_id']}**"]
        if chosen["title"]:
            detail.append(chosen["title"])
        st.caption(" — ".join(detail))
        extra = []
        if chosen["partners"]:
            extra.append("in complex with " + ", ".join(chosen["partners"][:3]))
        lig = pacc.notable_ligands(chosen)
        if lig:
            extra.append("bound: " + ", ".join(f"{l['code']} ({l['name']})"
                                               for l in lig[:3]))
        if chosen["mutations"]:
            extra.append(f"{chosen['mutations']} engineered mutation(s)")
        if (chosen["models"] or 1) > 1:
            extra.append(f"{chosen['models']}-model ensemble")
        if chosen["r_free"] is not None:
            extra.append(f"R-free {chosen['r_free']}")
        if extra:
            st.caption(" · ".join(extra))

    _protein_exports(prof)


def _protein_exports(prof: dict) -> None:
    """CSV / FASTA / text downloads for the protein's structure table."""
    st.divider()
    e1, e2, e3 = st.columns(3)
    stem = prof["accession"]
    with e1:
        st.download_button("⬇︎ Structures CSV", pacc.as_csv(prof),
                           f"{stem}_structures.csv", "text/csv", key="prot_csv",
                           use_container_width=True,
                           help="One row per PDB entry — opens in Excel or pandas")
    with e2:
        st.download_button("⬇︎ Sequence FASTA", pacc.as_fasta(prof),
                           f"{stem}.fasta", "text/plain", key="prot_fasta",
                           use_container_width=True,
                           help="The canonical UniProt sequence")
    with e3:
        st.download_button("⬇︎ Report", pacc.as_text(prof), f"{stem}_report.txt",
                           "text/plain", key="prot_txt", use_container_width=True,
                           help="The whole profile as plain text")


# ═══════════════════════════════════════════════════════════════════════════════
# UI Layout
# ═══════════════════════════════════════════════════════════════════════════════

left, right = st.columns([1, 3])   # 25% chat/controls | 75% 3D viewer

with left:
    st.subheader("💬 Agent Chat")

    # Scrollable chat history container
    chat_container = st.container(height=420)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Working context ──────────────────────────────────────────────────────
    # The agent applies an unqualified request to whatever protein is pinned
    # here, so it has to be visible before the message is sent rather than
    # inferred afterwards from the reply. Switching subject is this button, not
    # a turn of phrase the model has to guess at.
    _f = st.session_state.focus
    if _f:
        _who = _f["protein_name"] + (f" ({_f['gene']})" if _f["gene"] else "")
        _loaded = [k for k in _f["entries"] if find_structure(k)]
        _c1, _c2 = st.columns([3, 1.2])
        with _c1:
            st.caption(f"🎯 Working on **{_who}** · `{_f['accession']}` · "
                       + (", ".join(f"`{k}`" for k in _loaded) if _loaded
                          else "no structure loaded yet"))
        with _c2:
            if st.button("New search", key="focus_clear", use_container_width=True,
                         help="Forget this protein. Requests stop applying to it and "
                              "the next protein you name starts a fresh lookup."):
                clear_focus()
                st.rerun()

    # A question the agent asked last turn: each choice as a button, so the
    # user can answer with a click. Typing a reply works just as well — see
    # resolve_clarification().
    _clicked = None
    _pending = st.session_state.get("clarify")
    if _pending:
        st.caption("❔ Pick one, or type your own answer:")
        for _i, _o in enumerate(_pending["options"], 1):
            if st.button(f"{_i}. {_o['label']}", key=f"clarify_opt_{_i}",
                         use_container_width=True):
                _clicked = _o["label"]

    if prompt := (st.chat_input("e.g. Load pdb id, color chain A red.") or _clicked):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # st.status() renders and updates live during run_agent() — unlike
        # st.spinner, it can carry a running trace of what the agent loop is
        # doing (thinking / running a tool / composing a reply), so the
        # screen-lock during a slow multi-turn request isn't a blank wait.
        with st.status("🤖 Working on it…", expanded=True) as status:
            reply = run_agent(prompt, status=status)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            status.update(label="✅ Done", state="complete", expanded=False)
        st.rerun()

    with st.expander("🔍 Agent Debug Logs", expanded=False):
        for log in st.session_state.debug_logs[-30:]:   # Show only the most recent 30 entries
            st.write(log)
        if not st.session_state.debug_logs:
            st.write("No logs yet")

    # Compact state summary below the chat
    if structures():
        st.divider()
        if len(structures()) == 1:
            st.markdown(f"**Structure:** `{st.session_state.pdb_id}`")
        else:
            st.markdown("**Structures:** " + ", ".join(
                (f"`{x['pdb_id']}`*" if x["sid"] == st.session_state.active_sid
                 else f"`{x['pdb_id']}`") for x in structures()) +
                " — *\\* active*")
            for x in structures():
                if x["fit"]:
                    st.caption(f"↳ {x['pdb_id']} on {x.get('fit_reference', '?')}: {x['fit']}")
        if st.session_state.selections:
            st.markdown("**Named Selections:**")
            for name, expr in st.session_state.selections.items():
                short = expr[:40] + "..." if len(expr) > 40 else expr
                st.markdown(f"- `{name}`: {short}")
        st.markdown(f"**Representations:** {len(st.session_state.representations)}")

    if st.button("🆕 Clear scene", type="secondary",
                 help="Unload every structure and start over"):
        # Reset all session state keys to their defaults
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()


# ── Toolbar: menu-bar mirror of the Load/Prepare/Style/Analyze/Simulate tabs ──
# Docked above the viewer rather than replacing the tabs below, so this can be
# sanity-checked side by side with the proven controls before the tabs go
# away. Streamlit has no native hover menu, so each top-level item is a
# click-to-open st.popover listing its sub-panels; picking one opens
# _toolbar_dialog as a modal, and closing it drops back to a script rerun that
# rebuilds the viewer from whatever the dialog just changed — the same
# session-state path the tabs already use, so the reflection in the view is
# automatic.
# Old Load/Prepare/Style/Analyze/Simulate tab strip below the viewer, kept
# only as a disabled sanity-check fallback now that the toolbar above the
# viewer covers the same panels. Flip to True to bring it back.
SHOW_LEGACY_TABS = False

TOOLBAR_DISPATCH = {
    ("Load", "Loaded"):               ("Load — Loaded structures", structure_manager_ui),
    ("Load", "Find by protein name"): ("Load — Find by protein name", protein_finder_ui),
    ("Prepare", None):                ("Prepare", prepare_ui),
    ("Style", "Representations"):     ("Style — Representations", representation_manager_ui),
    ("Style", "Labels"):              ("Style — Labels", label_ui),
    ("Style", "Ray-traced figure"):   ("Style — Ray-traced figure", render_ui),
    ("Analyze", "Measure"):           ("Analyze — Measure", measurement_ui),
    ("Analyze", "Interactions"):      ("Analyze — Interactions", interactions_ui),
    ("Analyze", "Highlight"):         ("Analyze — Highlight", highlight_ui),
    ("Analyze", "Summary"):           ("Analyze — Summary", structure_summary_ui),
    ("Analyze", "Sequence"):          ("Analyze — Sequence", sequence_browser_ui),
    ("Simulate", "MD · Amber"):       ("Simulate — MD — Amber", amber_ui),
    ("Simulate", "MD · GROMACS"):     ("Simulate — MD — GROMACS", gromacs_ui),
    ("Simulate", "MD · Rosetta"):     ("Simulate — MD — Rosetta", rosetta_ui),
    ("Simulate", "QM"):               ("Simulate — QM", quantum_ui),
    ("Simulate", "QM/MM"):            ("Simulate — QM/MM", oniom_ui),
    ("Simulate", "Membrane"):         ("Simulate — Membrane", membrane_ui),
}


def _forget_toolbar_dialog() -> None:
    st.session_state.toolbar_open = None


@st.dialog("PARORA", width="large", on_dismiss=_forget_toolbar_dialog)
def _toolbar_dialog() -> None:
    """
    The popup for whichever toolbar sub-option was just clicked.

    `on_dismiss` matters here: the dialog's own "x", Escape and click-outside
    all close it on the frontend only — by default nothing clears
    `toolbar_open` server-side, so the *next* rerun for any reason at all
    (even an unrelated button elsewhere on the page, like the light/dark
    toggle) sees it still set and pops the same dialog back up.
    """
    title, render_fn = TOOLBAR_DISPATCH[st.session_state.toolbar_open]
    st.subheader(title)
    render_fn()
    st.divider()
    if st.button("Close", key="toolbar_dialog_close"):
        _forget_toolbar_dialog()
        st.rerun()


def _tab_panel(pane: tuple, render_fn) -> None:
    """
    Render a tab's body, unless the toolbar dialog for that same pane is open.

    Both entry points call the identical `*_ui()` function, and Streamlit
    requires every widget key to be unique for the whole script run — calling
    a panel's function twice in one run (once here, once inside the open
    dialog) throws StreamlitDuplicateElementKey on its first hardcoded key.
    Skipping the tab's copy while its dialog twin is open keeps both entry
    points working without touching the widget keys inside every `*_ui()`.
    """
    if st.session_state.toolbar_open == pane:
        st.caption("Open in the toolbar dialog above.")
    else:
        render_fn()


with right:
    st.subheader("Interactive 3D Viewer")
    if structures():
        st.caption("**Load** a structure or find one by protein name, "
                   "**Prepare** it for calculation, **Style** the picture; "
                   "**Analyze** measures distances, interactions, composition "
                   "and sequence; **Simulate** sets up **MD** (Amber, GROMACS, "
                   "Rosetta), **QM**, **QM/MM** and the **Membrane** a "
                   "membrane protein needs first. The button at the right of "
                   "the viewer's own toolbar switches the background.")

    if structures():
        # The Load/Prepare/Style/Analyze/Simulate menu now lives inside the
        # viewer itself (build_ngl_html()'s #toolbar); it emits a "toolbar_open"
        # event that handle_viewer_event() folds into toolbar_open below, so
        # this dialog is the only piece of the old outer toolbar left here.
        if st.session_state.toolbar_open:
            _toolbar_dialog()

        # Reserve the viewer's slot now, but fill it at the end of this run:
        # the panels below mutate the structure registry and the representation
        # stack, and the viewer must be built from their post-edit state, not
        # the stale values this run started with.
        viewer_slot = st.container()

        # Every control sits in one always-visible tab row directly under the
        # viewer. They used to be a stack of collapsed expanders, which meant
        # that below a 700px viewer there was nothing on screen but headings —
        # a feature nobody scrolls to and expands is a feature that does not
        # exist.
        if SHOW_LEGACY_TABS:
            st.divider()
            # Five short labels, not eight with emoji. Streamlit scrolls a tab strip
            # that overflows its column, behind arrows small enough to miss — which
            # is exactly how "Interactions" and "Highlight" end up invisible on a
            # 75%-width column. Everything is one click deep at most.
            # Three groups rather than one long row: the things you do to a
            # structure (load it, clean it, style it), the things you measure on
            # it, and the calculations you set up from it. Every kind of
            # calculation -- molecular dynamics, quantum, QM/MM, and the membrane
            # a membrane protein needs before any of them -- lives under Simulate,
            # so the top row stays short enough to read at a glance instead of
            # scrolling behind arrows.
            panels = st.tabs(["Load", "Prepare", "Style", "Analyze", "Simulate"])
            with panels[0]:
                loaded_tab, finder_tab = st.tabs(["Loaded", "Find by protein name"])
                with loaded_tab:
                    _tab_panel(("Load", "Loaded"), structure_manager_ui)
                with finder_tab:
                    _tab_panel(("Load", "Find by protein name"), protein_finder_ui)
            with panels[1]:
                _tab_panel(("Prepare", None), prepare_ui)
            with panels[2]:
                style_tab, label_tab, render_tab = st.tabs(
                    ["Representations", "Labels", "Ray-traced figure"])
                with style_tab:
                    _tab_panel(("Style", "Representations"), representation_manager_ui)
                with label_tab:
                    _tab_panel(("Style", "Labels"), label_ui)
                with render_tab:
                    _tab_panel(("Style", "Ray-traced figure"), render_ui)
            with panels[3]:
                measure_tab, inter_tab, highlight_tab, summary_tab, sequence_tab = st.tabs(
                    ["Measure", "Interactions", "Highlight", "Summary", "Sequence"])
                with measure_tab:
                    _tab_panel(("Analyze", "Measure"), measurement_ui)
                with inter_tab:
                    _tab_panel(("Analyze", "Interactions"), interactions_ui)
                with highlight_tab:
                    _tab_panel(("Analyze", "Highlight"), highlight_ui)
                with summary_tab:
                    _tab_panel(("Analyze", "Summary"), structure_summary_ui)
                with sequence_tab:
                    _tab_panel(("Analyze", "Sequence"), sequence_browser_ui)
            with panels[4]:
                md_tab, qm_tab, qmmm_tab, membrane_tab = st.tabs(
                    ["MD", "QM", "QM/MM", "Membrane"])
                with md_tab:
                    amber_tab, gromacs_tab, rosetta_tab = st.tabs(
                        ["Amber", "GROMACS", "Rosetta"])
                    with amber_tab:
                        _tab_panel(("Simulate", "MD · Amber"), amber_ui)
                    with gromacs_tab:
                        _tab_panel(("Simulate", "MD · GROMACS"), gromacs_ui)
                    with rosetta_tab:
                        _tab_panel(("Simulate", "MD · Rosetta"), rosetta_ui)
                with qm_tab:
                    _tab_panel(("Simulate", "QM"), quantum_ui)
                with qmmm_tab:
                    _tab_panel(("Simulate", "QM/MM"), oniom_ui)
                with membrane_tab:
                    _tab_panel(("Simulate", "Membrane"), membrane_ui)

        with viewer_slot:
            # Mounted as a declared component so the toolbar's pen and style
            # menus can post changes back; falls back to the old one-way iframe
            # if the component directory is missing, in which case the in-viewer
            # editors stay inert and the side panels still do everything.
            if _viewer_component is not None:
                event = _viewer_component(html=build_ngl_html(), height=700,
                                          key="ngl_viewer", default=None)
                if handle_viewer_event(event):
                    st.rerun()
            else:
                st.components.v1.html(build_ngl_html(), height=700, scrolling=False)
    else:
        st.info("Load a structure to begin. Try: **\"Load the insulin receptor\"**, "
                "**\"Fetch 3pp0\"**, or **\"What structures are there for EGFR?\"** — "
                "or use the panels below.")
        empty_load, empty_find = st.tabs(["Load a structure", "Find by protein name"])
        with empty_load:
            structure_manager_ui()
        with empty_find:
            protein_finder_ui()

    if not MDA_AVAILABLE:
        st.warning("MDAnalysis not installed — B-factor/proximity selections disabled. Run: `pip install MDAnalysis`")
