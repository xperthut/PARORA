# =============================================================================
# Developer : Methun Kamruzzaman
# Date      : 2026-03-11
# Summary   : Full-featured agentic protein structure visualizer with
#             MCP-style multi-turn tool-calling via Ollama. Integrates
#             MDAnalysis for server-side structural analysis (B-factor
#             filtering, proximity selections, structure alignment, solvent
#             removal), maintains named selections and layered NGL.js
#             representations, and enforces a gate-based agent loop to prevent
#             redundant or destructive tool calls. Camera orientation is
#             persisted across reruns via localStorage.
# =============================================================================

import streamlit as st
import os
import json
import requests
import numpy as np
from pathlib import Path
from ollama import Client
from rcsbapi.search import TextQuery
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

st.set_page_config(page_title="PDB Agentic Visualizer", layout="wide")
st.title("🧬 Agentic Protein Visualizer")
st.caption("Natural language → multi-turn tool-calling agent → local MDAnalysis + NGL.js 3D")

# ── Ollama host resolution: prefer env var, fall back to Docker bridge ────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if os.path.exists("/.dockerenv"):
    OLLAMA_HOST = "http://host.docker.internal:11434"
ollama_client = Client(host=OLLAMA_HOST)
MODEL = "llama3.2:latest"

# ── Storage ───────────────────────────────────────────────────────────────────
# PDB files downloaded during the session are cached here to avoid re-fetching
STRUCTURES_DIR = Path("./structures")
STRUCTURES_DIR.mkdir(exist_ok=True)

# ── Session state defaults ────────────────────────────────────────────────────
# Initialise every key on first run; subsequent reruns leave existing values.
defaults = {
    "messages":      [],        # chat history shown in the left panel
    "debug_logs":    [],        # internal tool-call trace for the debug expander
    "pdb_id":        None,      # active PDB accession (e.g. "3PP0")
    "pdb_path":      None,      # local path of the downloaded .pdb file
    "universe":      None,      # MDAnalysis Universe object (lazy-loaded)
    "selections":    {},        # name → NGL selection string (named selections)
    "representations": [],      # list of {type, selection, color, transparency}
    "background":    "black",   # NGL viewer background colour
    "camera_target": None,      # NGL selection to zoom/focus on after load
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


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
    # Pass-through — assume it's already a valid NGL expression
    return sel_name_or_expr


# ═══════════════════════════════════════════════════════════════════════════════
# Tool implementations
# ═══════════════════════════════════════════════════════════════════════════════

def tool_search_pdb(search_term: str) -> str:
    """
    Search the RCSB PDB by free-text and return the top-ranked PDB accession.

    Args:
        search_term: Protein name or descriptive query string.

    Returns:
        PDB ID string, "No results found", or "Error: <detail>" on failure.
    """
    try:
        query_obj = TextQuery(value=search_term)
        results = query_obj(rows=1)
        # rcsbapi may return a list, generator, or Session object
        pdb_id = next(iter(results), None)
        return pdb_id if pdb_id else "No results found"
    except Exception as e:
        return f"Error: {e}"


def tool_fetch_structure(pdb_id: str) -> str:
    """
    Download a PDB structure from RCSB and cache it in STRUCTURES_DIR.

    Skips the download if the file already exists locally. Resets MDAnalysis
    universe, named selections, and representations so they reflect the new
    structure rather than a stale previous one.

    Args:
        pdb_id: 4-character PDB accession code (case-insensitive).

    Returns:
        Confirmation string with the local path, or an error message.
    """
    pdb_id = pdb_id.upper().strip()
    if st.session_state.pdb_id == pdb_id:
        return f"{pdb_id} is already loaded — no action needed."
    dest = STRUCTURES_DIR / f"{pdb_id}.pdb"
    if not dest.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
        except Exception as e:
            return f"Error downloading {pdb_id}: {e}"
    st.session_state.pdb_id = pdb_id
    st.session_state.pdb_path = str(dest)
    st.session_state.universe = None          # Force MDAnalysis reload on next access
    st.session_state.selections = {}
    st.session_state.representations = [
        {"type": "cartoon", "selection": "protein", "color": "spectrum", "transparency": 0.0}
    ]
    st.session_state.camera_target = None
    return f"Loaded {pdb_id} → {dest}"


def tool_load_local(filepath: str) -> str:
    """
    Load a PDB structure from a local file path into the viewer.

    Uses the file stem (without extension) as the PDB ID label. Resets
    universe, selections, and representations the same way as tool_fetch_structure.

    Args:
        filepath: Absolute or relative path to a .pdb file.

    Returns:
        Confirmation string, or an error if the file is not found.
    """
    p = Path(filepath)
    if not p.exists():
        return f"File not found: {filepath}"
    pdb_id = p.stem.upper()
    st.session_state.pdb_id = pdb_id
    st.session_state.pdb_path = str(p)
    st.session_state.universe = None
    st.session_state.selections = {}
    st.session_state.representations = [
        {"type": "cartoon", "selection": "protein", "color": "spectrum", "transparency": 0.0}
    ]
    return f"Loaded local file: {p.name}"


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
        # ligand / small molecules
        "ligand": "ligand",
        "small molecule": "ligand",
        "small molecules": "ligand",
        "organic": "organic",
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
    if expr_lower.startswith("chain "):
        chain = expression.split()[-1]
        return f":{chain}", f"chain {chain}"

    # "resn ATP" / "resname ATP" → NGL "ATP"
    if expr_lower.startswith("resn ") or expr_lower.startswith("resname "):
        resname = expression.split()[-1].upper()
        # Guard against hallucinated residue names that mean "protein"
        if resname in _FAKE_RESNAMES:
            return "protein", "standard residues (protein)"
        return resname, f"resname {resname}"

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
                    ngl = " or ".join(nonstd)
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

    return expression, expression


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
        "_sel_name": name,   # internal tag used for later removal by tool_hide
    })


def tool_select(name: str, expression: str) -> str:
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
                return (
                    f"Warning: '{expression}' matched 0 atoms in {st.session_state.pdb_id}. "
                    f"This structure may not contain that residue/selection. "
                    f"Nothing was highlighted."
                )
        except Exception:
            pass

    st.session_state.selections[name] = ngl
    _add_highlight(ngl, name)
    short = ngl[:60] + "..." if len(ngl) > 60 else ngl
    return f"Selection '{name}' ({label}) highlighted as ball+stick. NGL: {short}"


def tool_select_within(name: str, radius: float, target_selection: str) -> str:
    """Select all atoms within `radius` Å of `target_selection`, expanded to whole residues."""
    u = get_universe()
    if not u:
        # Approximate fallback when MDAnalysis is unavailable
        ngl = f"({resolve_selection(target_selection)}) or polymer"
        st.session_state.selections[name] = ngl
        return f"MDAnalysis unavailable; approximate selection stored"

    try:
        target_ngl = resolve_selection(target_selection)
        mda_target_expr = _ngl_to_mda_approx(target_ngl)
        target_ag = u.select_atoms(mda_target_expr)
        nearby = u.select_atoms(f"byres (around {radius} group target)", target=target_ag)
        ngl = mda_to_ngl_serial(nearby)
        st.session_state.selections[name] = ngl
        return f"Selection '{name}': {len(nearby.residues)} residues within {radius}Å of '{target_selection}'"
    except Exception as e:
        return f"Error: {e}"


def tool_select_by_bfactor(name: str, operator: str, threshold: float) -> str:
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
        return "MDAnalysis unavailable"
    try:
        ag = u.select_atoms(f"tempfactor {operator} {threshold}")
        ngl = mda_to_ngl_serial(ag)
        st.session_state.selections[name] = ngl
        return f"Selection '{name}': {len(ag)} atoms with B-factor {operator} {threshold}"
    except Exception as e:
        return f"Error: {e}"


def tool_show(rep_type: str, selection: str, color: str = "element") -> str:
    """
    Add or replace a visual representation layer for a given selection.

    Normalises PyMOL/VMD representation names to their NGL equivalents, resolves
    named selections to NGL strings, and deduplicates by removing any existing
    layer of the same type+selection before appending the new one.

    Args:
        rep_type : Representation style (e.g. "cartoon", "surface", "licorice").
        selection: Named selection key or raw NGL expression.
        color    : NGL color scheme or named color (default: "element").

    Returns:
        Confirmation string.
    """
    # Map PyMOL/VMD names → NGL.js equivalents
    rep_aliases = {
        "licorice": "ball+stick",
        "sticks":   "ball+stick",
        "spheres":  "spacefill",
        "lines":    "line",
        "dots":     "point",
    }
    rep_type = rep_aliases.get(rep_type.lower(), rep_type)

    # Named selections take priority; otherwise translate the expression to NGL
    sels = st.session_state.selections
    if selection in sels:
        ngl_sel = sels[selection]
    else:
        u = get_universe()
        ngl_sel, _ = _expression_to_ngl(selection, u)

    # Replace any existing representation of the same type+selection instead of stacking
    st.session_state.representations = [
        r for r in st.session_state.representations
        if not (r["type"] == rep_type and r["selection"] == ngl_sel)
    ]
    st.session_state.representations.append({
        "type": rep_type, "selection": ngl_sel,
        "color": color, "transparency": 0.0
    })
    return f"Showing {rep_type} for '{selection}' ({color})"


def tool_hide(selection: str) -> str:
    """
    Remove all representation layers for a selection name or NGL expression.

    Args:
        selection: Named selection key or raw NGL expression to hide.

    Returns:
        Status string with the number of layers removed.
    """
    ngl_sel = resolve_selection(selection)
    before = len(st.session_state.representations)
    st.session_state.representations = [
        r for r in st.session_state.representations
        # Match by internal selection-name tag OR by NGL string — not both simultaneously
        if r.get("_sel_name") != selection and r["selection"] != ngl_sel
    ]
    removed = before - len(st.session_state.representations)
    return f"Removed {removed} representation(s) for '{selection}'"


def tool_hide_all() -> str:
    """Remove every representation layer from the viewer, leaving a blank canvas."""
    st.session_state.representations = []
    return "All representations hidden"


def tool_show_all(rep_type: str = "cartoon") -> str:
    """
    Replace all current representations with a single full-structure view.

    Args:
        rep_type: NGL representation type to apply to all atoms (default: "cartoon").

    Returns:
        Confirmation string.
    """
    st.session_state.representations = [
        {"type": rep_type, "selection": "all", "color": "spectrum", "transparency": 0.0}
    ]
    return f"Showing {rep_type} for all atoms"


def tool_color(color: str, selection: str) -> str:
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
    ngl_sel = resolve_selection(selection)
    updated = 0
    for r in st.session_state.representations:
        if r["selection"] == ngl_sel:
            r["color"] = color
            updated += 1
    if updated == 0:
        # Add a cartoon layer with this color if nothing matches
        st.session_state.representations.append({
            "type": "cartoon", "selection": ngl_sel,
            "color": color, "transparency": 0.0
        })
    return f"Colored '{selection}' as {color}"


def tool_set_transparency(value: float, selection: str) -> str:
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
    return f"Set transparency {value} on {updated} representation(s) for '{selection}'"


def tool_measure_distance(atom1_sel: str, atom2_sel: str) -> str:
    """
    Measure the Euclidean distance between the centers of geometry of two selections.

    Requires MDAnalysis. Both selections are resolved (named or raw NGL) and
    approximated to MDAnalysis syntax before computing the center of geometry.

    Args:
        atom1_sel: First selection name or NGL expression.
        atom2_sel: Second selection name or NGL expression.

    Returns:
        Distance string in Ångstroms, or an error/unavailability message.
    """
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable — cannot measure distance"
    try:
        sel1 = u.select_atoms(_ngl_to_mda_approx(resolve_selection(atom1_sel)))
        sel2 = u.select_atoms(_ngl_to_mda_approx(resolve_selection(atom2_sel)))
        if len(sel1) == 0 or len(sel2) == 0:
            return "One or both selections are empty"
        c1 = sel1.center_of_geometry()
        c2 = sel2.center_of_geometry()
        dist = float(np.linalg.norm(c1 - c2))
        return f"Distance between '{atom1_sel}' and '{atom2_sel}': {dist:.2f} Å"
    except Exception as e:
        return f"Error: {e}"
def tool_measure_mda_distance(sel1: str, sel2: str) -> str:
    """
    Notebook-derived deterministic MDAnalysis distance tool.
    """
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable — cannot measure distance"

    try:
        return measure_distance_from_universe(u, sel1, sel2)
    except Exception as e:
        return f"Error measuring MDAnalysis distance: {e}"


def tool_align_structures(mobile_id: str, reference_id: str) -> str:
    """
    Structurally align two locally downloaded PDB structures by backbone RMSD.

    Both structures must already exist in STRUCTURES_DIR. The aligned mobile
    structure is written to <mobile_id>_aligned.pdb in the same directory.

    Args:
        mobile_id   : PDB ID of the structure to move.
        reference_id: PDB ID of the fixed reference structure.

    Returns:
        RMSD string and output file path, or an error/unavailability message.
    """
    mob_path = STRUCTURES_DIR / f"{mobile_id.upper()}.pdb"
    ref_path = STRUCTURES_DIR / f"{reference_id.upper()}.pdb"
    if not mob_path.exists() or not ref_path.exists():
        return f"Both structures must be downloaded first. Missing: " + \
               (str(mob_path) if not mob_path.exists() else str(ref_path))
    if not MDA_AVAILABLE:
        return "MDAnalysis unavailable"
    try:
        from MDAnalysis.analysis import align as mda_align
        mob = mda.Universe(str(mob_path))  # type: ignore[union-attr]
        ref = mda.Universe(str(ref_path))  # type: ignore[union-attr]
        result = mda_align.alignto(mob, ref, select="backbone")
        rmsd = result[1]
        out_path = STRUCTURES_DIR / f"{mobile_id.upper()}_aligned.pdb"
        mob.atoms.write(str(out_path))
        return f"Aligned {mobile_id} → {reference_id}: RMSD = {rmsd:.3f} Å. Saved as {out_path.name}"
    except Exception as e:
        return f"Alignment error: {e}"


def tool_zoom(selection: str) -> str:
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
    return f"Camera focused on '{selection}'"


def tool_set_background(color: str) -> str:
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
    return f"Background set to {c}"


def tool_save_structure(filename: str) -> str:
    """
    Copy the currently loaded PDB file to a new filename in STRUCTURES_DIR.

    Args:
        filename: Target filename (e.g. "my_protein.pdb").

    Returns:
        Confirmation string with the output path, or an error message.
    """
    path = st.session_state.pdb_path
    if not path or not Path(path).exists():
        return "No structure loaded"
    out = STRUCTURES_DIR / filename
    import shutil
    shutil.copy2(path, out)
    return f"Saved to {out}"


def tool_remove_solvent() -> str:
    """
    Strip all water molecules from the current structure and save the result.

    Uses MDAnalysis to select non-water atoms, writes them to
    <pdb_id>_no_solvent.pdb, and updates the session to point at the new file.

    Returns:
        Confirmation with the output filename, or an error/unavailability message.
    """
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable"
    try:
        no_water = u.select_atoms("not water")
        out_path = STRUCTURES_DIR / f"{st.session_state.pdb_id}_no_solvent.pdb"
        no_water.write(str(out_path))
        st.session_state.pdb_path = str(out_path)
        st.session_state.universe = None   # Force reload from the new de-solvated file
        return f"Removed solvent. Saved to {out_path.name}"
    except Exception as e:
        return f"Error: {e}"


def tool_add_hydrogens() -> str:
    """
    Placeholder for hydrogen addition (requires OpenBabel or RDKit).

    Returns an informative message instead of silently failing.
    """
    return "Hydrogen addition requires OpenBabel or RDKit (not installed). Install openbabel-python to enable."

def tool_summarize_chains() -> str:
    """Summarize chains/segments in the currently loaded structure."""
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable — cannot summarize chains"

    try:
        df = summarize_chains_from_universe(u)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error summarizing chains: {e}"


def tool_list_residues(max_rows: int = 200) -> str:
    """List residues in the currently loaded structure."""
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable — cannot list residues"

    try:
        df = list_residues_from_universe(u, max_rows=max_rows)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error listing residues: {e}"


def tool_bfactor_summary() -> str:
    """Summarize B-factor/tempfactor values for the loaded structure."""
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable — cannot summarize B-factors"

    try:
        df = bfactor_summary_from_universe(u)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error summarizing B-factors: {e}"


def tool_detect_contacts(
    sel1: str = "protein",
    sel2: str = "protein",
    cutoff: float = 4.0,
    max_rows: int = 100
) -> str:
    """Detect residue-level contacts between two selections."""
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable — cannot detect contacts"

    try:
        df = contact_detection_from_universe(
            u,
            sel1=sel1,
            sel2=sel2,
            cutoff=cutoff,
            max_rows=max_rows
        )
        return df.to_string(index=False)
    except Exception as e:
        return f"Error detecting contacts: {e}"


def tool_detect_salt_bridges(
    cutoff: float = 4.0,
    max_rows: int = 100
) -> str:
    """Detect candidate salt bridges using acidic O atoms and basic N atoms."""
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable — cannot detect salt bridges"

    try:
        df = salt_bridge_detection_from_universe(
            u,
            cutoff=cutoff,
            max_rows=max_rows
        )
        return df.to_string(index=False)
    except Exception as e:
        return f"Error detecting salt bridges: {e}"


def tool_detect_hydrogen_bonds(
    cutoff: float = 3.5,
    max_rows: int = 100
) -> str:
    """Detect candidate hydrogen bonds using a distance-only donor/acceptor screen."""
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable — cannot detect hydrogen bonds"

    try:
        df = hydrogen_bond_detection_from_universe(
            u,
            cutoff=cutoff,
            max_rows=max_rows
        )
        return df.to_string(index=False)
    except Exception as e:
        return f"Error detecting hydrogen bonds: {e}"

def tool_nearby_residues(selection: str, cutoff: float = 5.0, max_rows: int = 100) -> str:
    """Find residues near an MDAnalysis selection in the currently loaded structure."""
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable — cannot find nearby residues"

    try:
        df = nearby_residues_from_universe(
            u,
            selection=selection,
            cutoff=_safe_float(cutoff, 5.0),
            max_rows=_safe_int(max_rows, 100),
        )
        return df.to_string(index=False)
    except Exception as e:
        return f"Error finding nearby residues: {e}"


def tool_measure_angle(sel1: str, sel2: str, sel3: str) -> str:
    """Measure an angle between three MDAnalysis atom selections."""
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable — cannot measure angle"

    try:
        result = measure_angle_from_universe(u, sel1=sel1, sel2=sel2, sel3=sel3)
        return str(result)
    except Exception as e:
        return f"Error measuring angle: {e}"


def tool_measure_dihedral(sel1: str, sel2: str, sel3: str, sel4: str) -> str:
    """Measure a dihedral angle between four MDAnalysis atom selections."""
    u = get_universe()
    if not u:
        return "MDAnalysis unavailable — cannot measure dihedral"

    try:
        result = measure_dihedral_from_universe(
            u,
            sel1=sel1,
            sel2=sel2,
            sel3=sel3,
            sel4=sel4,
        )
        return str(result)
    except Exception as e:
        return f"Error measuring dihedral: {e}"


# ── NGL → MDAnalysis expression approximation ────────────────────────────────

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
    return ngl_sel                       # pass-through for unknown expressions


# ═══════════════════════════════════════════════════════════════════════════════
# Tool registry for Ollama
# ═══════════════════════════════════════════════════════════════════════════════

# TOOLS is the formal schema sent to Ollama; TOOL_DISPATCH maps names to callables.
TOOLS = [
    {
        "type": "function", "function": {
            "name": "search_pdb",
            "description": "Search RCSB PDB by protein name and return the top matching PDB ID",
            "parameters": {"type": "object", "properties": {
                "search_term": {"type": "string"}}, "required": ["search_term"]}
        }
    },
    {
        "type": "function", "function": {
            "name": "fetch_structure",
            "description": "Download a PDB structure from RCSB by ID and save it locally. Always call this to load a structure before any other operation.",
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
                "rep_type": {"type": "string", "enum": ["cartoon", "ball+stick", "surface", "spacefill", "ribbon", "licorice", "line", "point"]},
                "selection": {"type": "string", "description": "A selection name or NGL expression"},
                "color": {"type": "string", "default": "element"}
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
            "description": "Apply a color to a selection. Color can be a name (red, green, blue, white) or scheme (element, spectrum, chainname, residueindex, bfactor)",
            "parameters": {"type": "object", "properties": {
                "color": {"type": "string"},
                "selection": {"type": "string"}
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
            "description": "Measure the distance in Ångstroms between the centers of two selections",
            "parameters": {"type": "object", "properties": {
                "atom1_sel": {"type": "string"},
                "atom2_sel": {"type": "string"}
            }, "required": ["atom1_sel", "atom2_sel"]}
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
            "description": "Summarize available chains/segments in the currently loaded protein structure.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "list_residues",
            "description": "List residues in the currently loaded protein structure.",
            "parameters": {
                "type": "object",
                "properties": {
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
            "description": "Detect candidate salt bridges between acidic and basic residues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cutoff": {"type": "number", "default": 4.0},
                    "max_rows": {"type": "integer", "default": 100}
                }
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "detect_hydrogen_bonds",
            "description": "Detect candidate hydrogen bonds using a distance-based donor/acceptor screen.",
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
            "name": "measure_angle",
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
            "name": "measure_dihedral",
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
    }
]
# Lambda dispatch table maps tool names → callables with argument extraction

def _safe_int(value, default):
    if value is None:
        return default
    return int(value)


def _safe_float(value, default):
    if value is None:
        return default
    return float(value)

TOOL_DISPATCH = {
    "search_pdb": lambda a: tool_search_pdb(a.get("search_term", "")),
    "fetch_structure": lambda a: tool_fetch_structure(a.get("pdb_id", "")),
    "load_local": lambda a: tool_load_local(a.get("filepath", "")),
    "select": lambda a: tool_select(
        a.get("name", "sel"),
        a.get("expression", "all"),
    ),
    "select_within": lambda a: tool_select_within(
        a.get("name", "pocket"),
        a.get("radius", 5.0),
        a.get("target_selection", "ligand"),
    ),
    "select_by_bfactor": lambda a: tool_select_by_bfactor(
        a.get("name", "flex"),
        a.get("operator", ">"),
        a.get("threshold", 50.0),
    ),
    "show": lambda a: tool_show(
        a.get("rep_type", "cartoon"),
        a.get("selection", "protein"),
        a.get("color", "element"),
    ),
    "hide": lambda a: tool_hide(a.get("selection", "all")),
    "hide_all": lambda _: tool_hide_all(),
    "show_all": lambda a: tool_show_all(a.get("rep_type", "cartoon")),
    "color": lambda a: tool_color(
        a.get("color", "red"),
        a.get("selection", "all"),
    ),
    "set_transparency": lambda a: tool_set_transparency(
        a.get("value", 0.5),
        a.get("selection", "all"),
    ),

    # ---------- Analysis ----------
    "measure_distance": lambda a: tool_measure_distance(
        a.get("atom1_sel", ""),
        a.get("atom2_sel", ""),
    ),

    "measure_mda_distance": lambda a: tool_measure_mda_distance(
        a.get("sel1", ""),
        a.get("sel2", ""),
    ),

    "summarize_chains": lambda _: tool_summarize_chains(),

    "list_residues": lambda a: tool_list_residues(
       _safe_int(a.get("max_rows", 200), 200),
    ),

    "bfactor_summary": lambda _: tool_bfactor_summary(),

    "detect_contacts": lambda a: tool_detect_contacts(
        a.get("sel1", "protein"),
        a.get("sel2", "protein"),
        _safe_float(a.get("cutoff", 4.0), 4.0),
        _safe_int(a.get("max_rows", 100), 100),
    ),

    "detect_salt_bridges": lambda a: tool_detect_salt_bridges(
        _safe_float(a.get("cutoff", 4.0)),
        _safe_int(a.get("max_rows", 100)),
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

    "measure_angle": lambda a: tool_measure_angle(
        a.get("sel1", ""),
	a.get("sel2", ""),
	a.get("sel3", ""),
    ),

    "measure_dihedral": lambda a: tool_measure_dihedral(
        a.get("sel1", ""),
        a.get("sel2", ""),
        a.get("sel3", ""),
        a.get("sel4", ""),
    ),


    # ---------- Structure ----------
    "align_structures": lambda a: tool_align_structures(
        a.get("mobile_id", ""),
        a.get("reference_id", ""),
    ),
    "zoom": lambda a: tool_zoom(a.get("selection", "all")),
    "set_background": lambda a: tool_set_background(
        a.get("color", "black"),
    ),
    "save_structure": lambda a: tool_save_structure(
        a.get("filename", "output.pdb"),
    ),
    "remove_solvent": lambda _: tool_remove_solvent(),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-turn agent loop
# ═══════════════════════════════════════════════════════════════════════════════

def _system_prompt() -> str:
    """
    Build the system prompt injected at every agent turn.

    Includes the current structure, named selections, and active representation
    count so the LLM can reason about what already exists before deciding which
    tools to call.
    """
    pdb = st.session_state.pdb_id or "none"
    sels = ", ".join(st.session_state.selections.keys()) or "none"
    reps = len(st.session_state.representations)
    return (
        f"You are a protein structure analysis agent. "
        f"Current structure: {pdb}. Named selections: [{sels}]. Active representations: {reps}. "
        "Rules — follow exactly: "
        "1. Only call fetch_structure or search_pdb when the current structure is 'none'. "
        "2. Call the MINIMUM tools needed. Never repeat a tool with the same arguments. "
        "3. Use a short descriptive selection name (e.g. 'nonstandard', 'atp_res', 'chain_a') — never 'sel'. "
        "4. `show` vs `select` are MUTUALLY EXCLUSIVE for the same command: "
        "   - User says 'show <type> for X' → call ONLY `show`. NEVER also call `select`. "
        "   - User says 'select X' or 'highlight X' → call ONLY `select`. Only also call `show` if the user explicitly wants a different rep type (e.g. surface, cartoon) in addition. "
        "5. Expression rules for `select` and `show`: "
        "   - 'standard residues' or 'protein' → expression='protein' "
        "   - 'ligand' or 'small molecule' → expression='ligand' "
        "   - 'non-standard residues' → expression='non-standard residues' "
        "   - 'chain A' → expression='chain A' "
        "   - 'chains' or 'all chains' → expression='chains' "
        "   - NEVER invent residue names like STANDARD, CANONICAL, NORMAL — use 'protein' instead. "
        "6. Never call `hide` unless the user explicitly asked to hide something. "
        "7. After your tools have run, reply with a plain-text summary. Stop calling tools. "
	"8. For atom/residue distance analysis, prefer `measure_mda_distance` with MDAnalysis syntax. "
	"Example: sel1='segid A and resid 50 and name CA', sel2='segid A and resid 100 and name CA'. "
	"CA means alpha carbon atom name CA. "
	"9. For chain summaries, use `summarize_chains`. "
	"10. For residue tables, use `list_residues`. "
	"11. For B-factor summaries, use `bfactor_summary`. "
	"12. For contact detection, use `detect_contacts`. "
	"13. For salt bridge detection, use `detect_salt_bridges`; do not use `select_by_bfactor` for salt bridges. "
	"14. For hydrogen bond candidates, use `detect_hydrogen_bonds`."
    )


def _tc_args(tc: dict) -> dict:
    """Safely extract tool-call arguments as a dict regardless of whether they're a str or dict."""
    raw = tc.get("function", {}).get("arguments", {})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def run_agent(user_prompt: str) -> str:
    """
    Execute a gated, deduplicated multi-turn tool-calling loop for one user command.

    The loop runs up to MAX_TURNS iterations. On each turn it:
      1. Sends the message history (with refreshed system prompt) to Ollama.
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

    Returns:
        Final agent text summary or a concatenation of tool result strings.
    """
    st.session_state.debug_logs.append(f"📨 User: {user_prompt}")

    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt}
    ]

    prompt_lower = user_prompt.lower() 
    # Direct deterministic route for explicit MDAnalysis distance prompts
    if "sel1=" in prompt_lower and "sel2=" in prompt_lower:
        import re
        m1 = re.search(r"sel1=['\"]([^'\"]+)['\"]", user_prompt)
        m2 = re.search(r"sel2=['\"]([^'\"]+)['\"]", user_prompt)

        if m1 and m2:
            sel1 = m1.group(1)
            sel2 = m2.group(1)

            results = []
            if st.session_state.pdb_id is None:
                pdb_match = re.search(r"\b([0-9][A-Za-z0-9]{3})\b", user_prompt)
                if pdb_match:
                    results.append(tool_fetch_structure(pdb_match.group(1)))

            results.append(tool_measure_mda_distance(sel1, sel2))
            return "Done: " + "; ".join(results)

    if "salt bridge" in prompt_lower or "salt bridges" in prompt_lower:
        results = []
        if st.session_state.pdb_id is None:
            pdb_match = re.search(r"\b([0-9][A-Za-z0-9]{3})\b", user_prompt)
            if pdb_match:
                results.append(tool_fetch_structure(pdb_match.group(1)))
        results.append(tool_detect_salt_bridges())
        return "Done: " + "; ".join(results)

    # Gate: destructive tools only when user explicitly asked for hiding
    DESTRUCTIVE_TOOLS = {"hide_all", "hide"}
    hide_requested = any(w in prompt_lower for w in {"hide", "clear", "remove", "delete", "clean", "erase", "reset"})

    # Gate: load/search tools only when no structure is loaded or user asks for one
    LOAD_TOOLS = {"search_pdb", "fetch_structure", "load_local"}
    load_requested = (
        st.session_state.pdb_id is None
        or any(w in prompt_lower for w in {"load", "fetch", "download", "open", "get", "show me", "find"})
    )

    # Gate: write/side-effect tools only when explicitly requested by the user
    WRITE_TOOLS = {"save_structure", "remove_solvent", "align_structures"}
    write_requested = any(w in prompt_lower for w in {"save", "write", "align", "remove solvent", "remove water", "no water"})

    # Intent: purely a representation command → block all select calls for the entire run
    show_only_intent = (
        any(w in prompt_lower for w in {"show", "display", "render", "visualize", "view"})
        and not any(w in prompt_lower for w in {"select", "highlight", "find", "identify", "pick"})
        and not any(w in prompt_lower for w in {"save", "write", "load", "fetch", "download"})
    )

    MAX_TURNS = 8
    summary_parts = []
    called_sigs: set[str] = set()          # Tracks (name, args) pairs to avoid exact repeats
    selected_ngl_strs: set[str] = set()    # Tracks NGL strings that already have a highlight
    show_rep_fired = False                 # True once any non-ball+stick show has executed

    for _ in range(MAX_TURNS):
        try:
            response = ollama_client.chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                options={"temperature": 0.0}
            )
        except ConnectionError:
            err = f"Cannot reach Ollama at {OLLAMA_HOST}. Start Ollama with `ollama serve`."
            st.session_state.debug_logs.append(f"❌ {err}")
            return err
        except Exception as e:
            err = f"Ollama error: {e}"
            st.session_state.debug_logs.append(f"❌ {err}")
            return err

        msg = response.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        # No tool calls → agent is done; return its text reply
        if not tool_calls:
            final_text = msg.get("content", "").strip()
            st.session_state.debug_logs.append(f"💬 Agent: {final_text}")
            return final_text or ("Done: " + "; ".join(summary_parts))

        tool_results = []

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

            # ── Gate: select ────────────────────────────────────────────────
            # Block select when: show-only intent, a show already fired this run,
            # or a rep-type show is in the same batch.
            if name == "select" and (show_only_intent or show_rep_fired or batch_has_rep_show):
                st.session_state.debug_logs.append(
                    f"🚫 Blocked 'select' — representation command; no highlight needed"
                )
                tool_results.append({"tool": name, "result": "Blocked — use 'show' for representations, 'select' for highlights."})
                continue

            # ── Gate: destructive ───────────────────────────────────────────
            if name in DESTRUCTIVE_TOOLS and not hide_requested:
                st.session_state.debug_logs.append(f"🚫 Blocked '{name}' — not requested")
                tool_results.append({"tool": name, "result": "Blocked — user did not request hiding."})
                continue

            # ── Gate: load/search ───────────────────────────────────────────
            if name in LOAD_TOOLS and not load_requested:
                st.session_state.debug_logs.append(f"🚫 Blocked '{name}' — structure already loaded")
                tool_results.append({"tool": name, "result": f"Blocked — {st.session_state.pdb_id} is already loaded."})
                continue

            # ── Gate: write/side-effect ─────────────────────────────────────
            if name in WRITE_TOOLS and not write_requested:
                st.session_state.debug_logs.append(f"🚫 Blocked '{name}' — not requested by user")
                tool_results.append({"tool": name, "result": "Blocked — user did not request this operation."})
                continue

            # ── Dedup: exact same call ──────────────────────────────────────
            sig = f"{name}:{json.dumps(args, sort_keys=True)}"
            if sig in called_sigs:
                st.session_state.debug_logs.append(f"⏭ Skipped duplicate: {name}")
                tool_results.append({"tool": name, "result": "Already called — skipped."})
                continue

            # ── Dedup: ball+stick show for an already-selected NGL string ───
            if name == "show" and args.get("rep_type", "") == "ball+stick":
                ngl_candidate = resolve_selection(args.get("selection", ""))
                if ngl_candidate in selected_ngl_strs:
                    st.session_state.debug_logs.append(f"⏭ Skipped 'show ball+stick' — already highlighted")
                    tool_results.append({"tool": name, "result": "Already highlighted by select — skipped."})
                    continue

            # ── Dedup: select for an already-selected NGL string ────────────
            if name == "select":
                ngl_candidate = resolve_selection(args.get("expression", ""))
                if ngl_candidate in selected_ngl_strs:
                    st.session_state.debug_logs.append(f"⏭ Skipped duplicate select — '{ngl_candidate}' already selected")
                    tool_results.append({"tool": name, "result": "Already selected — skipped."})
                    continue

            called_sigs.add(sig)

            # Dispatch to the tool function
            dispatch = TOOL_DISPATCH.get(name)
            try:
                result = dispatch(args) if dispatch else f"Unknown tool: {name}"
            except Exception as e:
                result = f"Tool error: {e}"

            # Track NGL strings that now have a ball+stick highlight
            if name == "select" and "error" not in result.lower():
                ngl_str = st.session_state.selections.get(args.get("name", ""), "")
                if ngl_str:
                    selected_ngl_strs.add(ngl_str)

            # Mark that a representation-type show has fired this run
            if name == "show" and args.get("rep_type", "cartoon") != "ball+stick":
                show_rep_fired = True

            st.session_state.debug_logs.append(f"🔧 {name}({args}) → {result}")
            summary_parts.append(f"{name}: {result}")
            tool_results.append({"tool": name, "result": result})

        # Append tool results to the conversation and refresh the system prompt
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})
        results_text = "\n".join(f"[{r['tool']}]: {r['result']}" for r in tool_results)
        messages[0] = {"role": "system", "content": _system_prompt()}

        # After a representation show, request a short confirmation and stop calling tools
        follow_up = (
            "Representation updated. Reply with one sentence confirming what changed — NO MORE TOOL CALLS."
            if show_rep_fired else
            "Continue with any remaining steps. When all steps are done, reply with a short plain-text summary — no more tool calls."
        )
        messages.append({
            "role": "user",
            "content": f"Tool results:\n{results_text}\n\n{follow_up}"
        })

    st.session_state.debug_logs.append("⚠️ Max turns reached")
    return "Done: " + "; ".join(summary_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# NGL.js HTML renderer
# ═══════════════════════════════════════════════════════════════════════════════

def _js_str(s: str) -> str:
    """Escape a string for safe embedding inside a JS double-quoted string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_ngl_html() -> str:
    """
    Generate the full HTML/JS snippet that renders the current structure in NGL.js.

    Reads pdb_id, representations, background, and camera_target from session
    state to build a self-contained iframe-compatible block. Camera orientation
    is saved to and restored from localStorage so the user's viewpoint persists
    across Streamlit reruns. On white backgrounds, spectrum-colored protein
    cartoons are swapped to chainname coloring for better contrast.

    Returns:
        HTML string to pass to st.components.v1.html(), or "" if no structure
        is loaded.
    """
    pdb_id = st.session_state.pdb_id
    if not pdb_id:
        return ""

    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    bg = st.session_state.background
    reps = st.session_state.representations
    camera_target = st.session_state.camera_target
    cam_key = f"ngl_cam_{pdb_id}"   # Per-structure localStorage key for camera state

    reps_js = ""
    for rep in reps:
        t = _js_str(rep["type"])
        s = _js_str(rep["selection"])
        raw_color = rep.get("color", "element")
        # On white background swap spectrum protein cartoon to chainname for readability
        if bg == "white" and raw_color == "spectrum" and rep.get("selection") == "protein":
            raw_color = "chainname"
        c = _js_str(raw_color)
        opacity = round(1.0 - float(rep.get("transparency", 0.0)), 3)
        reps_js += (
            f'comp.addRepresentation("{t}", '
            f'{{sele: "{s}", color: "{c}", opacity: {opacity}}});\n'
        )

    zoom_js = f'comp.setSelection("{_js_str(camera_target)}");' if camera_target else ""

    return f"""
    <div style="width:100%;height:680px;border:1px solid #555;border-radius:8px;overflow:hidden;background:{bg};">
        <div id="viewport" style="width:100%;height:100%;"></div>
    </div>
    <script src="https://unpkg.com/ngl@2/dist/ngl.js"></script>
    <script>
    (function(){{
        var CAM_KEY = "{cam_key}";

        function saveCam(stage) {{
            try {{
                var o = stage.viewerControls.getOrientation();
                localStorage.setItem(CAM_KEY, JSON.stringify(Array.from(o.elements)));
            }} catch(e) {{}}
        }}

        function restoreCam(stage) {{
            try {{
                var raw = localStorage.getItem(CAM_KEY);
                if (!raw) return false;
                var arr = JSON.parse(raw);
                var o = stage.viewerControls.getOrientation();
                o.elements.set(new Float32Array(arr));
                stage.viewerControls.orient(o);
                return true;
            }} catch(e) {{ return false; }}
        }}

        var stage = new NGL.Stage("viewport", {{backgroundColor: "{bg}"}});

        stage.loadFile("{pdb_url}").then(function(comp){{
            {reps_js}
            {zoom_js}
            if (!restoreCam(stage)) {{
                comp.autoView();
            }}
            // Persist camera on every user interaction so reruns restore the same view
            var el = stage.viewer.renderer.domElement;
            el.addEventListener("mouseup",  function(){{ saveCam(stage); }});
            el.addEventListener("wheel",     function(){{ saveCam(stage); }}, {{passive: true}});
            el.addEventListener("touchend",  function(){{ saveCam(stage); }});
        }}).catch(function(e){{
            console.error("NGL load error:", e);
        }});

        window.addEventListener("resize", function(){{ stage.handleResize(); }});
    }})();
    </script>
    """


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

    if prompt := st.chat_input("e.g. Load 3pp0, select ATP, color chain A red..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Agent reasoning..."):
            reply = run_agent(prompt)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    with st.expander("🔍 Agent Debug Logs", expanded=False):
        for log in st.session_state.debug_logs[-30:]:   # Show only the most recent 30 entries
            st.write(log)
        if not st.session_state.debug_logs:
            st.write("No logs yet")

    # Compact state summary below the chat
    if st.session_state.pdb_id:
        st.divider()
        st.markdown(f"**Structure:** `{st.session_state.pdb_id}`")
        if st.session_state.selections:
            st.markdown("**Named Selections:**")
            for name, expr in st.session_state.selections.items():
                short = expr[:40] + "..." if len(expr) > 40 else expr
                st.markdown(f"- `{name}`: {short}")
        st.markdown(f"**Representations:** {len(st.session_state.representations)}")

    if st.button("🆕 New Structure", type="secondary"):
        # Reset all session state keys to their defaults
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

with right:
    title_col, btn_col = st.columns([5, 1])
    with title_col:
        st.subheader("Interactive 3D Viewer")
    with btn_col:
        # Toggle button label reflects the current background state
        is_dark = st.session_state.background == "black"
        label = "☀️ Light" if is_dark else "🌙 Dark"
        if st.button(label, key="bg_toggle"):
            st.session_state.background = "white" if is_dark else "black"
            st.rerun()

    if st.session_state.pdb_id:
        html = build_ngl_html()
        st.components.v1.html(html, height=700, scrolling=False)

        # Expandable table showing exactly what NGL.js will receive
        if st.session_state.representations:
            with st.expander("🎨 Active Representations (NGL selections)", expanded=False):
                rows = []
                for i, r in enumerate(st.session_state.representations):
                    sel = r["selection"]
                    short_sel = sel[:60] + "…" if len(sel) > 60 else sel
                    rows.append({
                        "#": i + 1,
                        "Type": r["type"],
                        "NGL Selection": short_sel,
                        "Color": r.get("color", "element"),
                        "Opacity": round(1.0 - r.get("transparency", 0.0), 2),
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Load a structure to begin. Try: **\"Load insulin\"** or **\"Fetch 3pp0\"**")

    if not MDA_AVAILABLE:
        st.warning("MDAnalysis not installed — B-factor/proximity selections disabled. Run: `pip install MDAnalysis`")
