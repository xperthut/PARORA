"""
analysis_tools.py

Notebook-derived deterministic protein structure analysis tools
for PARORA.

Originally developed in the Goal 1/2 Colab notebook and adapted
for the Streamlit application.
"""

import numpy as np
import pandas as pd


def summarize_chains_from_universe(u) -> pd.DataFrame:
    """
    Summarize chains/segments in the current MDAnalysis Universe.
    Adapted from notebook summarize_chains(shared).
    """
    rows = []
    for seg in u.segments:
        atoms = seg.atoms
        residues = atoms.residues
        rows.append({
            "chain_or_segment": seg.segid,
            "n_atoms": len(atoms),
            "n_residues": len(residues),
        })
    return pd.DataFrame(rows)

def list_residues_from_universe(
    u,
    chain=None,
    max_rows: int = 200
) -> pd.DataFrame:
    """List residues, optionally restricted to one chain or segment."""

    residues = u.residues

    if chain is not None:
        chain = str(chain).strip()
        if chain:
            residues = [
                res for res in residues
                if str(res.segid).strip().upper() == chain.upper()
            ]

    rows = []
    for res in residues[:max_rows]:
        rows.append({
            "resid": res.resid,
            "resname": res.resname,
            "chain_or_segment": res.segid,
            "n_atoms": len(res.atoms)
        })

    if chain and not rows:
        return pd.DataFrame([{
            "error": f"No residues found for chain or segment {chain}"
        }])

    return pd.DataFrame(rows)

def bfactor_summary_from_universe(u) -> pd.DataFrame:
    atoms = u.atoms

    if not hasattr(atoms, "tempfactors"):
        return pd.DataFrame([{"error": "No B-factor/tempfactor data found"}])

    return pd.DataFrame({
        "mean_bfactor": [float(np.mean(atoms.tempfactors))],
        "min_bfactor": [float(np.min(atoms.tempfactors))],
        "max_bfactor": [float(np.max(atoms.tempfactors))],
    })

def measure_distance_from_universe(u, sel1: str, sel2: str) -> str:
    ag1 = u.select_atoms(sel1)
    ag2 = u.select_atoms(sel2)

    if len(ag1) == 0 or len(ag2) == 0:
        return f"Empty selection: sel1 atoms={len(ag1)}, sel2 atoms={len(ag2)}"

    c1 = ag1.center_of_geometry()
    c2 = ag2.center_of_geometry()
    d = float(np.linalg.norm(c1 - c2))
    return f"Distance between [{sel1}] and [{sel2}] = {d:.2f} Å"

def nearby_residues_from_universe(u, target_selection: str, radius: float = 5.0) -> pd.DataFrame:
    target = u.select_atoms(target_selection)

    if len(target) == 0:
        return pd.DataFrame([{"error": "Target selection matched 0 atoms"}])

    nearby = u.select_atoms(f"byres around {radius} group target", target=target)

    rows = []
    for res in nearby.residues:
        rows.append({
            "resname": res.resname,
            "resid": res.resid,
            "chain_or_segment": res.segid,
            "n_atoms": len(res.atoms)
        })
    return pd.DataFrame(rows).drop_duplicates()

def single_atom_position_from_universe(u, selection: str):
    ag = u.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(f"Selection matched 0 atoms: {selection}")
    return ag.center_of_geometry(), len(ag)

def measure_angle_from_universe(u, sel1: str, sel2: str, sel3: str) -> str:
    """
    Measure angle sel1-sel2-sel3 using centers of geometry.
    """
    p1, n1 = single_atom_position_from_universe(u, sel1)
    p2, n2 = single_atom_position_from_universe(u, sel2)
    p3, n3 = single_atom_position_from_universe(u, sel3)

    v1 = p1 - p2
    v2 = p3 - p2
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return "Angle failed: duplicate/zero-length vectors."
    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))
    return f"Angle [{sel1}] - [{sel2}] - [{sel3}] = {angle:.2f}° using atom counts {n1}, {n2}, {n3}"

def measure_dihedral_from_universe(u, sel1: str, sel2: str, sel3: str, sel4: str) -> str:
    """
    Measure dihedral/torsion angle using centers of geometry.
    """
    p1, n1 = single_atom_position_from_universe(u, sel1)
    p2, n2 = single_atom_position_from_universe(u, sel2)
    p3, n3 = single_atom_position_from_universe(u, sel3)
    p4, n4 = single_atom_position_from_universe(u, sel4)

    b0 = -(p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3
    if np.linalg.norm(b1) == 0:
        return "Dihedral failed: middle vector has zero length."

    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    dih = np.degrees(np.arctan2(y, x))
    return f"Dihedral [{sel1}] - [{sel2}] - [{sel3}] - [{sel4}] = {dih:.2f}° using atom counts {n1}, {n2}, {n3}, {n4}"

def contact_detection_from_universe(u, sel1: str = "protein", sel2: str = "protein", cutoff: float = 4.0, max_rows: int = 100) -> pd.DataFrame:
    """
    Residue-level contact detection. Excludes same residue contacts.
    """
    ag1 = u.select_atoms(sel1)
    ag2 = u.select_atoms(sel2)

    if len(ag1) == 0 or len(ag2) == 0:
        return pd.DataFrame([{"error": f"Empty selection: sel1 atoms={len(ag1)}, sel2 atoms={len(ag2)}"}])

    pairs = []
    for r1 in ag1.residues:
        a1 = r1.atoms
        if len(a1) == 0:
            continue
        for r2 in ag2.residues:
            if r1.segid == r2.segid and r1.resid == r2.resid:
                continue
            a2 = r2.atoms
            if len(a2) == 0:
                continue
            dmat = np.linalg.norm(a1.positions[:, None, :] - a2.positions[None, :, :], axis=2)
            min_d = float(np.min(dmat))
            if min_d <= cutoff:
                pairs.append({
                    "residue_1": f"{r1.resname}{r1.resid}:{r1.segid}",
                    "residue_2": f"{r2.resname}{r2.resid}:{r2.segid}",
                    "min_distance_A": round(min_d, 3),
                    "cutoff_A": cutoff
                })
                if len(pairs) >= max_rows:
                    return pd.DataFrame(pairs)
    return pd.DataFrame(pairs) if pairs else pd.DataFrame([{"result": "No contacts found", "cutoff_A": cutoff}])

def salt_bridge_detection_from_universe(
    u,
    chain: str | None = None,
    cutoff: float = 4.0,
    max_rows: int = 100
) -> pd.DataFrame:
    """
    Detect geometry-based candidate salt bridges.

    Acidic ASP/GLU side-chain oxygen atoms are compared with basic
    LYS/ARG/HIS side-chain nitrogen atoms.

    When ``chain`` is provided, both interacting residues must belong
    to that chain or segment.
    """
    chain = str(chain).strip() if chain is not None else ""

    acidic_selection = (
        "(resname ASP GLU) and "
        "(name OD1 OD2 OE1 OE2)"
    )
    basic_selection = (
        "(resname LYS ARG HIS HSD HSE HSP) and "
        "(name NZ NH1 NH2 NE NE2 ND1)"
    )

    if chain:
        chain_selection = f"segid {chain}"
        acidic_selection = (
            f"({chain_selection}) and ({acidic_selection})"
        )
        basic_selection = (
            f"({chain_selection}) and ({basic_selection})"
        )

    acidic = u.select_atoms(acidic_selection)
    basic = u.select_atoms(basic_selection)

    if len(acidic) == 0 or len(basic) == 0:
        scope = f" in chain {chain}" if chain else ""
        return pd.DataFrame([{
            "error": (
                f"Missing acidic/basic atoms{scope}: "
                f"acidic={len(acidic)}, basic={len(basic)}"
            )
        }])

    rows = []

    for acidic_atom in acidic:
        for basic_atom in basic:
            # Exclude atoms belonging to the same residue.
            if (
                acidic_atom.segid == basic_atom.segid
                and acidic_atom.resid == basic_atom.resid
            ):
                continue

            distance = float(
                np.linalg.norm(
                    acidic_atom.position - basic_atom.position
                )
            )

            if distance <= cutoff:
                rows.append({
                    "acidic_residue": (
                        f"{acidic_atom.resname}"
                        f"{acidic_atom.resid}:"
                        f"{acidic_atom.segid}"
                    ),
                    "acidic_atom": acidic_atom.name,
                    "basic_residue": (
                        f"{basic_atom.resname}"
                        f"{basic_atom.resid}:"
                        f"{basic_atom.segid}"
                    ),
                    "basic_atom": basic_atom.name,
                    "distance_A": round(distance, 3),
                    "cutoff_A": cutoff,
                })

                if len(rows) >= max_rows:
                    return pd.DataFrame(rows)

    if rows:
        return pd.DataFrame(rows)

    scope = f" in chain {chain}" if chain else ""
    return pd.DataFrame([{
        "result": f"No salt bridges found{scope}",
        "cutoff_A": cutoff,
    }])

def hydrogen_bond_detection_from_universe(u, cutoff: float = 3.5, max_rows: int = 100) -> pd.DataFrame:
    """
    Lightweight donor/acceptor proximity screen.
    This is a demo-safe approximation, not a full geometric H-bond classifier.
    """
    donors = u.select_atoms("protein and (name N NE NH1 NH2 NZ ND1 NE2 OG OG1 OH SG)")
    acceptors = u.select_atoms("protein and (name O OD1 OD2 OE1 OE2 OG OG1 OH SD SG ND1 NE2)")

    if len(donors) == 0 or len(acceptors) == 0:
        return pd.DataFrame([{"error": f"Missing donor/acceptor atoms: donors={len(donors)}, acceptors={len(acceptors)}"}])

    rows = []
    for d_atom in donors:
        for a_atom in acceptors:
            if d_atom.segid == a_atom.segid and d_atom.resid == a_atom.resid:
                continue
            dist = float(np.linalg.norm(d_atom.position - a_atom.position))
            if dist <= cutoff:
                rows.append({
                    "donor_residue": f"{d_atom.resname}{d_atom.resid}:{d_atom.segid}",
                    "donor_atom": d_atom.name,
                    "acceptor_residue": f"{a_atom.resname}{a_atom.resid}:{a_atom.segid}",
                    "acceptor_atom": a_atom.name,
                    "distance_A": round(dist, 3),
                    "cutoff_A": cutoff,
                    "note": "distance-only screen"
                })
                if len(rows) >= max_rows:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"result": "No H-bond candidates found", "cutoff_A": cutoff}])
