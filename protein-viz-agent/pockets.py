# =============================================================================
# Summary   : Binding-pocket / druggability detection (fpocket).
#
#             Answers "where could something bind" from the fold itself,
#             independent of whether anything is bound there in this file.
#             fpocket (Le Guilloux, Schmidtke & Tufféry 2009) fills the
#             protein surface with alpha spheres (Voronoi-vertex spheres
#             touching four atoms, no atom inside), clusters them into
#             pockets and scores each one; its druggability score
#             (Schmidtke & Barril 2010, 0-1, >= 0.5 predicted druggable) is
#             a trained estimate of whether a drug-like molecule could bind
#             there with high affinity. Classic geometry — no ML model, no
#             GPU, no network.
#
#             fpocket runs on a protein-only copy of the structure (ligands,
#             ions and water stripped), so a site that happens to be occupied
#             in this file is detected on the same footing as an empty one,
#             and each pocket is then checked against the hetero groups the
#             deposited file actually has there: a pocket that holds the
#             bound heme is independent evidence the method finds real sites.
#
#             Optional external binary, discovered at runtime and reported
#             "unavailable" when missing — same pattern as DSSP/Foldseek
#             (FPOCKET_BIN env var, else PATH; run.sh also finds a conda env
#             whose name mentions fpocket).
#
#             No Streamlit import — same split as topology.py/
#             structure_search.py, so this is testable standalone and app.py
#             owns the @st.cache_data wrapping.
# =============================================================================

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import sequence_utils as squ
import structure_report as srep

SCHEMA_VERSION = 1

# fpocket's default minimum pocket size is 15 alpha spheres. A small protein
# (crambin, 46 residues) has no cluster that large, so a second pass at this
# size reports its shallow surface grooves instead of nothing — labelled as
# below the default threshold, never as a real pocket.
DEFAULT_MIN_SPHERES = 15
RELAXED_MIN_SPHERES = 6

# fpocket's own threshold for its druggability score (Schmidtke & Barril 2010).
DRUGGABLE = 0.5

# A hetero atom this close to an alpha-sphere centre sits inside the pocket.
# Alpha spheres are >= 3.4 Å in radius, so a bound ligand's atoms fall well
# within this of the centres that fill its cavity.
LIGAND_OVERLAP = 3.0
OCCUPIED_FRACTION = 0.25

_WATERS = {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"}

_INFO_KEYS = {
    "Score": "score",
    "Druggability Score": "drug_score",
    "Number of Alpha Spheres": "n_spheres",
    "Volume": "volume",
    "Hydrophobicity score": "hydrophobicity",
    "Polarity score": "polarity",
    "Proportion of polar atoms": "polar_atoms_pct",
    "Mean alp. sph. solvent access": "solvent_access",
    "Total SASA": "sasa",
}


# ── fpocket discovery ────────────────────────────────────────────────────────

_cached_bin = None


def find_fpocket(force_rescan: bool = False):
    """
    Locate a working `fpocket` binary: FPOCKET_BIN env var, then PATH.

    fpocket has no --version flag; run bare, it prints its usage banner
    (naming itself) and exits, which is enough to prove it executes.

    Returns:
        Path to fpocket, or None.
    """
    global _cached_bin
    if _cached_bin and not force_rescan:
        return _cached_bin
    for exe in (os.getenv("FPOCKET_BIN"), shutil.which("fpocket")):
        if not exe or not Path(exe).exists():
            continue
        try:
            r = subprocess.run([exe], capture_output=True, text=True, timeout=30)
            if "fpocket" in (r.stdout + r.stderr).lower():
                _cached_bin = exe
                return exe
        except Exception:
            continue
    return None


def fpocket_available() -> bool:
    """True if a working `fpocket` was found on this machine."""
    return find_fpocket() is not None


UNAVAILABLE = ("fpocket not installed — pocket detection unavailable. Set "
               "FPOCKET_BIN to an fpocket binary, or install one: "
               "conda create -n fpocket -c conda-forge fpocket")


# ── Input preparation ────────────────────────────────────────────────────────

def _split(path: str):
    """
    First model of a PDB file, split into (protein_lines, hetero_groups).

    protein_lines: ATOM records plus HETATM records of modified amino acids
    (MSE, SEP, ... — part of the chain, so part of the pocket walls), first
    alternate location only. hetero_groups: every other non-water hetero
    residue, {(chain, resseq, icode, resname): [(x, y, z), ...]}.
    """
    protein, hetero = [], {}
    modres = set()
    with open(path, errors="replace") as fh:
        for line in fh:
            rec = line[:6]
            if rec == "MODRES":
                modres.add(line[12:15].strip())
            elif rec.startswith("ENDMDL"):
                break
            elif rec in ("ATOM  ", "HETATM"):
                if line[16] not in (" ", "A", "1"):
                    continue
                resname = line[17:20].strip()
                if rec == "ATOM  " or resname in modres or (
                        resname in squ.AA3_TO_1 and resname not in _WATERS):
                    protein.append("ATOM  " + line[6:])
                    continue
                if resname in _WATERS or line[76:78].strip() == "H":
                    continue
                try:
                    xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except ValueError:
                    continue
                key = (line[21], int(line[22:26]), line[26].strip(), resname)
                hetero.setdefault(key, []).append(xyz)
    return protein, hetero


# ── Parsing fpocket's output ─────────────────────────────────────────────────

def _parse_info(path: Path) -> dict:
    """{pocket_number: {score, drug_score, ...}} from <name>_info.txt."""
    pockets, current = {}, None
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"^Pocket\s+(\d+)\s*:", line)
        if m:
            current = pockets.setdefault(int(m.group(1)), {})
            continue
        if current is None or ":" not in line:
            continue
        key, _, value = line.partition(":")
        field = _INFO_KEYS.get(key.strip())
        if field:
            try:
                current[field] = float(value.strip())
            except ValueError:
                pass
    return pockets


def _parse_residues(path: Path) -> list:
    """Lining residues (chain, resseq, icode, resname), in residue order."""
    seen = set()
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                seen.add((line[21], int(line[22:26]), line[26].strip(), line[17:20].strip()))
            except ValueError:
                continue
    return sorted(seen, key=lambda r: (r[0], r[1], r[2]))


def _parse_spheres(path: Path) -> list:
    """Alpha-sphere centres (x, y, z) from pocketN_vert.pqr."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("ATOM"):
            try:
                out.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    return out


def _occupants(spheres: list, hetero: dict) -> list:
    """
    Hetero groups of the deposited file that sit inside this pocket:
    [{label, resname, kind, n_in, n_atoms}], most-buried first.
    """
    found = []
    cut2 = LIGAND_OVERLAP ** 2
    for (chain, resseq, icode, resname), coords in hetero.items():
        n_in = sum(1 for a in coords
                   if any((a[0] - s[0]) ** 2 + (a[1] - s[1]) ** 2 + (a[2] - s[2]) ** 2
                          <= cut2 for s in spheres))
        # A glycan or ligand grazing the rim with one atom is not "in" the
        # pocket; a quarter of its atoms inside is.
        if n_in and n_in >= OCCUPIED_FRACTION * len(coords):
            found.append({"label": f"{resname} {chain}/{resseq}{icode}",
                          "resname": resname, "kind": srep.classify_component(resname),
                          "n_in": n_in, "n_atoms": len(coords)})
    found.sort(key=lambda o: (-o["n_in"] / o["n_atoms"], -o["n_atoms"]))
    return found


# ── Running fpocket ──────────────────────────────────────────────────────────

def _run(exe: str, pdb_file: Path, min_spheres, timeout: int):
    cmd = [exe, "-f", pdb_file.name]
    if min_spheres:
        cmd += ["-i", str(min_spheres)]
    r = subprocess.run(cmd, cwd=pdb_file.parent, capture_output=True,
                       text=True, timeout=timeout)
    out_dir = pdb_file.parent / f"{pdb_file.stem}_out"
    info = out_dir / f"{pdb_file.stem}_info.txt"
    if not info.exists():
        tail = (r.stderr or r.stdout).strip()[-300:]
        raise RuntimeError(f"fpocket produced no output (exit {r.returncode}): {tail}")
    return out_dir, _parse_info(info)


def find_pockets(pdb_path: str, timeout: int = 300) -> tuple:
    """
    Detect candidate binding pockets on a structure's own geometry.

    Args:
        pdb_path: Local .pdb file.
        timeout : Seconds before each fpocket run is killed.

    Returns:
        (ok, message, result). result = {"pockets": [...], "relaxed": bool,
        "hetero": [labels of stripped hetero groups]}; each pocket is
        {rank, score, drug_score, volume, n_spheres, hydrophobicity,
        polarity, solvent_access, residues, center, occupants}, in fpocket's
        own rank order (its pocket score). relaxed is True when the default
        run found nothing and the list comes from the RELAXED_MIN_SPHERES
        pass. Empty result when ok is False.
    """
    exe = find_fpocket()
    if exe is None:
        return False, UNAVAILABLE, {}
    if not Path(pdb_path).exists():
        return False, f"Structure file not found: {pdb_path}", {}

    protein, hetero = _split(pdb_path)
    if not protein:
        return False, "No protein atoms in this structure — nothing to search for pockets.", {}

    with tempfile.TemporaryDirectory(prefix="fpocket_") as tmp:
        pdb_file = Path(tmp) / "protein.pdb"
        pdb_file.write_text("".join(protein) + "END\n")
        relaxed = False
        try:
            out_dir, info = _run(exe, pdb_file, None, timeout)
            if not info:
                shutil.rmtree(out_dir, ignore_errors=True)
                out_dir, info = _run(exe, pdb_file, RELAXED_MIN_SPHERES, timeout)
                relaxed = True
        except subprocess.TimeoutExpired:
            return False, f"fpocket timed out after {timeout} s.", {}
        except Exception as e:
            return False, f"fpocket failed: {e}", {}

        pockets = []
        for num in sorted(info):
            atm = out_dir / "pockets" / f"pocket{num}_atm.pdb"
            vert = out_dir / "pockets" / f"pocket{num}_vert.pqr"
            if not atm.exists() or not vert.exists():
                continue
            spheres = _parse_spheres(vert)
            if not spheres:
                continue
            n = len(spheres)
            center = tuple(round(sum(s[i] for s in spheres) / n, 2) for i in range(3))
            p = {"rank": num, **info[num]}
            p.update(residues=_parse_residues(atm), center=center,
                     occupants=_occupants(spheres, hetero))
            pockets.append(p)

    stripped = sorted({f"{k[3]} {k[0]}/{k[1]}{k[2]}" for k in hetero})
    return True, "", {"pockets": pockets, "relaxed": relaxed, "hetero": stripped}
