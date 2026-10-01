# =============================================================================
# Developer : Abdullah Al Mamun, Methun Kamruzzaman
# Date      : 2026-09-11
# Summary   : Detection of non-covalent interactions salt bridges, hydrogen
#             bonds, disulfides, hydrophobic contacts, pi-stacking, cation-pi
#             and metal coordination from a structure's coordinates. The criterion
#             here is geometric and each one is spelled out in the constant that 
#             defines it, because "salt bridge" means different distances to 
#             different tools and a number with no stated cutoff is not a result 
#             anyone can reuse. The defaults are 4.0 A between oppositely
#             charged groups (Barlow & Thornton), 3.5 A between hydrogen-bond
#             donor and acceptor heavy atoms, 2.5 A for a disulfide.
#
#             Crystal structures almost never contain hydrogens, so hydrogen
#             bonds are detected from donor/acceptor heavy-atom distance alone.
#             That is the standard fallback and it over-predicts: a pair can be
#             3.2 A apart and still be geometrically incapable of bonding. When
#             hydrogens are present the D-H...A angle is checked as well, and
#             the result says which of the two was used. 
# =============================================================================

import math

import numpy as np

import sequence_utils as squ

SCHEMA_VERSION = 1

# ── Criteria ─────────────────────────────────────────────────────────────────
# Every cutoff the detectors use, in Ångstroms and degrees, in one place.

SALT_BRIDGE_MAX = 4.0        # Barlow & Thornton (1983), charged group atoms
HBOND_MAX = 3.5              # donor-acceptor heavy atom separation
HBOND_MIN_ANGLE = 120.0      # D-H...A, only checked when hydrogens are present
DISULFIDE_MAX = 2.5          # SG-SG
HYDROPHOBIC_MAX = 4.5        # apolar side-chain carbon pairs
STACK_CENTROID_MAX = 5.5     # aromatic ring centroids, face-to-face
STACK_PARALLEL_MAX = 30.0    # interplanar angle counted as parallel stacking
TSHAPE_CENTROID_MAX = 6.0    # edge-to-face rings sit further apart
TSHAPE_MIN_ANGLE = 60.0
CATION_PI_MAX = 6.0          # cation to ring centroid
CATION_PI_MAX_OFFSET = 45.0  # angle off the ring normal
METAL_COORD_MAX = 3.0        # metal to coordinating N/O/S

# ── Chemistry tables ─────────────────────────────────────────────────────────

# Formally charged side-chain atoms at physiological pH. Histidine is included
# as a cation because it is protonated often enough to matter, and flagged in
# the output so it can be discounted.
ANIONIC = {
    ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"),
}
CATIONIC = {
    ("LYS", "NZ"),
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("HIS", "ND1"), ("HIS", "NE2"),
}
CONDITIONAL_CATION = {"HIS"}     # only charged when protonated

# Hydrogen-bond donors: heavy atoms carrying a hydrogen. Backbone N is added
# for every residue except proline, which has no amide hydrogen.
DONORS = {
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("ASN", "ND2"), ("GLN", "NE2"),
    ("HIS", "ND1"), ("HIS", "NE2"),
    ("LYS", "NZ"), ("TRP", "NE1"),
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"),
    ("CYS", "SG"),
}
# Hydrogen-bond acceptors: lone-pair bearing atoms. Backbone O is added for
# every residue.
ACCEPTORS = {
    ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
    ("ASN", "OD1"), ("GLN", "OE1"),
    ("HIS", "ND1"), ("HIS", "NE2"),
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"),
    ("MET", "SD"), ("CYS", "SG"),
}
BACKBONE_DONOR = "N"
BACKBONE_ACCEPTOR = "O"

# Residues whose side chains are apolar enough for a carbon-carbon contact to
# read as a hydrophobic interaction.
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE",
                        "TRP", "PRO", "TYR", "CYS"}
BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}

# Aromatic rings, as the atoms that define the plane. Tryptophan's six-membered
# ring is used; histidine's imidazole is aromatic and participates in stacking.
AROMATIC_RINGS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
}
# The atom that carries the positive charge for cation-pi geometry.
CATION_PI_ATOMS = {("LYS", "NZ"), ("ARG", "CZ")}

METALS = {"ZN", "MG", "CA", "MN", "FE", "FE2", "CU", "CU1", "NI", "CO",
          "NA", "K", "CD", "HG", "PT", "MO", "W", "SR", "BA"}
COORDINATING_ELEMENTS = {"N", "O", "S"}

# Display order and colours, used by the report and by the viewer overlay.
INTERACTION_TYPES = [
    ("salt_bridge",  "Salt bridge",          "#FFD34D"),
    ("hbond",        "Hydrogen bond",        "#5AD2F4"),
    ("disulfide",    "Disulfide bond",       "#F2A93B"),
    ("pi_stacking",  "π–π stacking",         "#B07AD9"),
    ("cation_pi",    "Cation–π",             "#F06BA8"),
    ("metal",        "Metal coordination",   "#59C36A"),
    ("hydrophobic",  "Hydrophobic contact",  "#9AA0A6"),
]
TYPE_LABELS = {k: label for k, label, _ in INTERACTION_TYPES}
TYPE_COLORS = {k: color for k, _, color in INTERACTION_TYPES}
DEFAULT_TYPES = ["salt_bridge", "hbond", "disulfide", "pi_stacking",
                 "cation_pi", "metal"]


# ── Neighbour search ─────────────────────────────────────────────────────────

def _grid_pairs(coords_a, coords_b, cutoff):
    """
    Index pairs (i, j) with |a_i - b_j| <= cutoff, via a uniform grid.

    A full distance matrix would be simpler but quadratic, and a ribosome
    subunit has enough atoms to make that a several-gigabyte allocation. The
    grid keeps the whole-structure scans below a second.
    """
    if len(coords_a) == 0 or len(coords_b) == 0:
        return []

    cell = max(cutoff, 1e-6)
    origin = np.minimum(coords_a.min(axis=0), coords_b.min(axis=0))

    buckets = {}
    keys_b = np.floor((coords_b - origin) / cell).astype(int)
    for j, key in enumerate(map(tuple, keys_b)):
        buckets.setdefault(key, []).append(j)

    keys_a = np.floor((coords_a - origin) / cell).astype(int)
    cutoff2 = cutoff * cutoff
    out = []
    neighbourhood = [(dx, dy, dz)
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for i, key in enumerate(map(tuple, keys_a)):
        pa = coords_a[i]
        for off in neighbourhood:
            bucket = buckets.get((key[0] + off[0], key[1] + off[1], key[2] + off[2]))
            if not bucket:
                continue
            for j in bucket:
                d = coords_b[j] - pa
                d2 = float(d @ d)
                if d2 <= cutoff2:
                    out.append((i, j, math.sqrt(d2)))
    return out


# ── Atom bookkeeping ─────────────────────────────────────────────────────────

def _residue_key(atoms, i):
    return (atoms["chain"][i], atoms["resseq"][i], atoms["icode"][i])


def _residue_label(atoms, i):
    return (f"{atoms['resname'][i]} {atoms['resseq'][i]}{atoms['icode'][i]}"
            f" {atoms['chain'][i]}")


def _usable(atoms, i):
    """Skip alternate conformations beyond the first so pairs are not doubled."""
    return atoms["altloc"][i] in ("", "A")


def _select_indices(atoms, predicate):
    return [i for i in range(len(atoms["name"])) if _usable(atoms, i) and predicate(i)]


def _has_hydrogens(atoms):
    return any(e.upper() == "H" for e in atoms["element"])


def _record(atoms, i, j, kind, distance, note="", extra=None):
    """One interaction, as a flat dict the report and the viewer both consume."""
    rec = {
        "type": kind,
        "distance": round(float(distance), 2),
        "a_label": _residue_label(atoms, i), "a_atom": atoms["name"][i],
        "b_label": _residue_label(atoms, j), "b_atom": atoms["name"][j],
        "a_key": _residue_key(atoms, i), "b_key": _residue_key(atoms, j),
        "a_index": int(i), "b_index": int(j),
        "note": note,
    }
    if extra:
        rec.update(extra)
    return rec


# ── Detectors ────────────────────────────────────────────────────────────────

def _salt_bridges(atoms, cutoff):
    """Oppositely charged groups within the cutoff."""
    def anionic(i):
        rn, nm = atoms["resname"][i], atoms["name"][i]
        if (rn, nm) in ANIONIC or nm == "OXT":
            return True
        # A ligand's oxygens might be a carboxylate; protonation is unknown.
        return atoms["kind"][i] == "hetero" and atoms["element"][i].upper() == "O"

    def cationic(i):
        rn, nm = atoms["resname"][i], atoms["name"][i]
        if (rn, nm) in CATIONIC:
            return True
        return atoms["kind"][i] == "hetero" and atoms["element"][i].upper() == "N"

    neg = _select_indices(atoms, anionic)
    pos = _select_indices(atoms, cationic)
    if not neg or not pos:
        return []

    out = []
    for a, b, d in _grid_pairs(atoms["xyz"][neg], atoms["xyz"][pos], cutoff):
        i, j = neg[a], pos[b]
        if _residue_key(atoms, i) == _residue_key(atoms, j):
            continue
        notes = []
        if atoms["resname"][i] in CONDITIONAL_CATION or atoms["resname"][j] in CONDITIONAL_CATION:
            notes.append("histidine — only charged when protonated")
        if atoms["kind"][i] == "hetero" or atoms["kind"][j] == "hetero":
            notes.append("involves a ligand, so the charge is assumed")
        out.append(_record(atoms, i, j, "salt_bridge", d, "; ".join(notes)))
    return out


def _hydrogen_bonds(atoms, cutoff, with_hydrogens):
    """
    Donor/acceptor pairs within the cutoff.

    The peptide bond itself has to be excluded: the backbone O of residue i sits
    about 2.25 A from the N of residue i+1, well inside any hydrogen-bond
    cutoff, and reporting every one of them would bury the real bonds under one
    false positive per residue.
    """
    def is_donor(i):
        rn, nm = atoms["resname"][i], atoms["name"][i]
        if (rn, nm) in DONORS:
            return True
        if nm == BACKBONE_DONOR and atoms["kind"][i] == "protein" and rn != "PRO":
            return True
        return atoms["kind"][i] in ("hetero", "water") and \
            atoms["element"][i].upper() in ("N", "O")

    def is_acceptor(i):
        rn, nm = atoms["resname"][i], atoms["name"][i]
        if (rn, nm) in ACCEPTORS:
            return True
        if nm in (BACKBONE_ACCEPTOR, "OXT") and atoms["kind"][i] == "protein":
            return True
        return atoms["kind"][i] in ("hetero", "water") and \
            atoms["element"][i].upper() in ("N", "O")

    donors = _select_indices(atoms, is_donor)
    acceptors = _select_indices(atoms, is_acceptor)
    if not donors or not acceptors:
        return []

    hydrogens = None
    if with_hydrogens:
        hidx = _select_indices(atoms, lambda i: atoms["element"][i].upper() == "H")
        hydrogens = (hidx, atoms["xyz"][hidx]) if hidx else None

    out, seen = [], set()
    for a, b, d in _grid_pairs(atoms["xyz"][donors], atoms["xyz"][acceptors], cutoff):
        i, j = donors[a], acceptors[b]
        if i == j:
            continue
        ki, kj = _residue_key(atoms, i), _residue_key(atoms, j)
        if ki == kj:
            continue
        # The peptide bond: backbone N of one residue and backbone O of the
        # residue immediately before it, in the same chain.
        if (atoms["name"][i] == "N" and atoms["name"][j] == "O"
                and ki[0] == kj[0] and ki[1] - kj[1] == 1):
            continue
        pair = tuple(sorted((i, j)))
        if pair in seen:
            continue
        seen.add(pair)

        note = ""
        if hydrogens:
            # Find the hydrogen bonded to the donor and check D-H...A opens up.
            hidx, hxyz = hydrogens
            dv = hxyz - atoms["xyz"][i]
            close = np.nonzero((dv * dv).sum(axis=1) < 1.5 ** 2)[0]
            if len(close):
                best = -1.0
                for c in close:
                    v1 = atoms["xyz"][i] - hxyz[c]
                    v2 = atoms["xyz"][j] - hxyz[c]
                    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                    if denom:
                        ang = math.degrees(math.acos(
                            float(np.clip(np.dot(v1, v2) / denom, -1, 1))))
                        best = max(best, ang)
                if best >= 0 and best < HBOND_MIN_ANGLE:
                    continue
                note = f"D–H···A {best:.0f}°"
        out.append(_record(atoms, i, j, "hbond", d, note))
    return out


def _disulfides(atoms, cutoff):
    """Cysteine SG pairs close enough to be bonded."""
    # CYX/CYM: an Amber-prepared copy (prepare_structure) renames bonded
    # cysteines, and the check found "none" in trypsin's six (3PTB_PREP).
    sg = _select_indices(atoms, lambda i: atoms["resname"][i] in ("CYS", "CYX", "CYM")
                         and atoms["name"][i] == "SG")
    out, seen = [], set()
    for a, b, d in _grid_pairs(atoms["xyz"][sg], atoms["xyz"][sg], cutoff):
        i, j = sg[a], sg[b]
        if i >= j or _residue_key(atoms, i) == _residue_key(atoms, j):
            continue
        if (i, j) in seen:
            continue
        seen.add((i, j))
        out.append(_record(atoms, i, j, "disulfide", d))
    return out


def _hydrophobic(atoms, cutoff):
    """Apolar side-chain carbon contacts, reported once per residue pair."""
    def apolar_carbon(i):
        return (atoms["resname"][i] in HYDROPHOBIC_RESIDUES
                and atoms["element"][i].upper() == "C"
                and atoms["name"][i] not in BACKBONE_NAMES)

    carbons = _select_indices(atoms, apolar_carbon)
    best = {}
    for a, b, d in _grid_pairs(atoms["xyz"][carbons], atoms["xyz"][carbons], cutoff):
        i, j = carbons[a], carbons[b]
        if i >= j:
            continue
        ki, kj = _residue_key(atoms, i), _residue_key(atoms, j)
        if ki == kj:
            continue
        # Neighbours in sequence are always in contact; that is not an interaction.
        if ki[0] == kj[0] and abs(ki[1] - kj[1]) <= 1:
            continue
        key = tuple(sorted((ki, kj)))
        if key not in best or d < best[key]["distance"]:
            best[key] = _record(atoms, i, j, "hydrophobic", d)
    return list(best.values())


def _rings(atoms):
    """Aromatic rings as (residue index, centroid, unit normal, label)."""
    wanted = {}
    for i in range(len(atoms["name"])):
        rn = atoms["resname"][i]
        if rn not in AROMATIC_RINGS or not _usable(atoms, i):
            continue
        if atoms["name"][i] not in AROMATIC_RINGS[rn]:
            continue
        wanted.setdefault(_residue_key(atoms, i), []).append(i)

    rings = []
    for key, indices in wanted.items():
        if len(indices) < 5:
            continue                       # incomplete side chain
        pts = atoms["xyz"][indices]
        centroid = pts.mean(axis=0)
        # The ring plane's normal is the smallest singular vector of the
        # centred ring atoms — robust to a slightly non-planar model.
        _, _, vh = np.linalg.svd(pts - centroid)
        rings.append({"key": key, "centroid": centroid, "normal": vh[2],
                      "index": indices[0], "resname": atoms["resname"][indices[0]]})
    return rings


def _interplanar(n1, n2):
    """Angle between two ring planes, folded into 0-90 degrees."""
    cos = abs(float(np.clip(np.dot(n1, n2), -1, 1)))
    return math.degrees(math.acos(cos))


def _pi_stacking(atoms):
    """Face-to-face and edge-to-face aromatic pairs."""
    rings = _rings(atoms)
    out = []
    for a in range(len(rings)):
        for b in range(a + 1, len(rings)):
            r1, r2 = rings[a], rings[b]
            d = float(np.linalg.norm(r1["centroid"] - r2["centroid"]))
            if d > TSHAPE_CENTROID_MAX:
                continue
            angle = _interplanar(r1["normal"], r2["normal"])
            if d <= STACK_CENTROID_MAX and angle <= STACK_PARALLEL_MAX:
                geometry = f"parallel, planes {angle:.0f}° apart"
            elif angle >= TSHAPE_MIN_ANGLE:
                geometry = f"edge-to-face, planes {angle:.0f}° apart"
            else:
                continue
            out.append(_record(atoms, r1["index"], r2["index"],
                               "pi_stacking", d, geometry))
    return out


def _cation_pi(atoms):
    """Cationic groups sitting over the face of an aromatic ring."""
    rings = _rings(atoms)
    if not rings:
        return []
    cations = _select_indices(
        atoms, lambda i: (atoms["resname"][i], atoms["name"][i]) in CATION_PI_ATOMS)

    out = []
    for i in cations:
        p = atoms["xyz"][i]
        for ring in rings:
            if _residue_key(atoms, i) == ring["key"]:
                continue
            v = p - ring["centroid"]
            d = float(np.linalg.norm(v))
            if d > CATION_PI_MAX or d == 0:
                continue
            cos = abs(float(np.clip(np.dot(v / d, ring["normal"]), -1, 1)))
            offset = math.degrees(math.acos(cos))
            if offset > CATION_PI_MAX_OFFSET:
                continue          # beside the ring, not over its face
            out.append(_record(atoms, i, ring["index"], "cation_pi", d,
                               f"{offset:.0f}° off the ring normal"))
    return out


def _metal_coordination(atoms, cutoff):
    """Metal ions and the N/O/S atoms around them."""
    metals = _select_indices(
        atoms, lambda i: atoms["resname"][i].upper() in METALS
        and atoms["kind"][i] == "hetero")
    if not metals:
        return []
    ligands = _select_indices(
        atoms, lambda i: atoms["element"][i].upper() in COORDINATING_ELEMENTS)

    out = []
    for a, b, d in _grid_pairs(atoms["xyz"][metals], atoms["xyz"][ligands], cutoff):
        i, j = metals[a], ligands[b]
        if _residue_key(atoms, i) == _residue_key(atoms, j):
            continue
        note = "water" if atoms["kind"][j] == "water" else ""
        out.append(_record(atoms, i, j, "metal", d, note))
    return out


# ── Entry point ──────────────────────────────────────────────────────────────

def find(atoms, types=None, cutoffs=None, restrict_to=None, include_water=False):
    """
    Detect non-covalent interactions in a structure.

    Args:
        atoms       : Atom table from measure.read_atoms.
        types       : Interaction types to look for; None means DEFAULT_TYPES,
                      which is everything except hydrophobic contacts (there
                      are thousands of those in any protein and they drown
                      everything else unless asked for).
        cutoffs     : Optional {type: distance} overriding the defaults.
        restrict_to : Set of (chain, resseq, icode) residue keys; only
                      interactions involving one of them are returned.
        include_water: Keep interactions where one partner is a water.

    Returns:
        dict with "interactions" (list, sorted by type then distance),
        "counts" per type, "criteria" describing the cutoffs actually used,
        and "hydrogens" saying whether the file had any.
    """
    types = list(types or DEFAULT_TYPES)
    cutoffs = dict(cutoffs or {})
    with_h = _has_hydrogens(atoms)

    found = []
    if "salt_bridge" in types:
        found += _salt_bridges(atoms, cutoffs.get("salt_bridge", SALT_BRIDGE_MAX))
    if "hbond" in types:
        found += _hydrogen_bonds(atoms, cutoffs.get("hbond", HBOND_MAX), with_h)
    if "disulfide" in types:
        found += _disulfides(atoms, cutoffs.get("disulfide", DISULFIDE_MAX))
    if "pi_stacking" in types:
        found += _pi_stacking(atoms)
    if "cation_pi" in types:
        found += _cation_pi(atoms)
    if "metal" in types:
        found += _metal_coordination(atoms, cutoffs.get("metal", METAL_COORD_MAX))
    if "hydrophobic" in types:
        found += _hydrophobic(atoms, cutoffs.get("hydrophobic", HYDROPHOBIC_MAX))

    if not include_water:
        waters = squ.WATER_NAMES
        found = [f for f in found
                 if f["a_label"].split()[0] not in waters
                 and f["b_label"].split()[0] not in waters]

    if restrict_to is not None:
        keys = set(restrict_to)
        found = [f for f in found if f["a_key"] in keys or f["b_key"] in keys]

    order = {k: n for n, (k, _, _) in enumerate(INTERACTION_TYPES)}
    found.sort(key=lambda f: (order[f["type"]], f["distance"]))

    counts = {}
    for f in found:
        counts[f["type"]] = counts.get(f["type"], 0) + 1

    return {
        "interactions": found,
        "counts": counts,
        "hydrogens": with_h,
        "criteria": criteria_text(types, cutoffs, with_h),
    }


def criteria_text(types, cutoffs=None, with_hydrogens=False):
    """One line per detector saying exactly what was counted."""
    cutoffs = cutoffs or {}
    lines = []
    if "salt_bridge" in types:
        lines.append(f"salt bridge: charged N/O atoms within "
                     f"{cutoffs.get('salt_bridge', SALT_BRIDGE_MAX):.1f} Å")
    if "hbond" in types:
        line = (f"hydrogen bond: donor–acceptor heavy atoms within "
                f"{cutoffs.get('hbond', HBOND_MAX):.1f} Å")
        line += (f", D–H···A ≥ {HBOND_MIN_ANGLE:.0f}°" if with_hydrogens
                 else " (no hydrogens in the file, so distance only — this "
                      "over-predicts)")
        lines.append(line)
    if "disulfide" in types:
        lines.append(f"disulfide: SG–SG within "
                     f"{cutoffs.get('disulfide', DISULFIDE_MAX):.1f} Å")
    if "pi_stacking" in types:
        lines.append(f"π–π: ring centroids within {STACK_CENTROID_MAX:.1f} Å and "
                     f"planes ≤ {STACK_PARALLEL_MAX:.0f}° apart (parallel), or "
                     f"within {TSHAPE_CENTROID_MAX:.1f} Å and ≥ "
                     f"{TSHAPE_MIN_ANGLE:.0f}° (edge-to-face)")
    if "cation_pi" in types:
        lines.append(f"cation–π: cation within {CATION_PI_MAX:.1f} Å of a ring "
                     f"centroid and ≤ {CATION_PI_MAX_OFFSET:.0f}° off its normal")
    if "metal" in types:
        lines.append(f"metal coordination: N/O/S within "
                     f"{cutoffs.get('metal', METAL_COORD_MAX):.1f} Å of a metal")
    if "hydrophobic" in types:
        lines.append(f"hydrophobic: apolar side-chain carbons within "
                     f"{cutoffs.get('hydrophobic', HYDROPHOBIC_MAX):.1f} Å, "
                     f"non-adjacent residues")
    return lines


def residues_involved(found):
    """Residue keys touched by a list of interactions, for highlighting."""
    keys = []
    for f in found:
        for key in (f["a_key"], f["b_key"]):
            if key not in keys:
                keys.append(key)
    return keys
