# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-12
# Summary   : This module will be used to extract Quantum-mechanical model out 
#             of a crystal structure such as a ligand, or a ligand with the 
#             residues around it and write the input file for QM tools. A QM region
#             taken out of a protein has bonds running out of it into the rest
#             of the structure, and every one of them has to be dealt with:
#             left dangling they are radicals, and a calculation on a handful
#             of radicals converges to something, reports no error, and means
#             nothing. The standard answer is a link atom a hydrogen placed
#             along the broken bond  and that is what this module does,
#             finding the bonds by distance rather than trusting CONECT
#             records that deposited files frequently lack.
#
#             The second hard part is the charge. It is a single integer that
#             changes the answer completely, and it cannot be read off the
#             coordinates: it is the sum of the ionisation states of whatever
#             ended up in the region, plus the ligand's own charge, which only
#             the person doing the work knows. So it is computed from a
#             residue table, shown broken down residue by residue, and left
#             editable. A QM calculation run at the wrong charge is the most
#             expensive way there is to get a wrong number. Region selection is 
#             deliberately plain: everything within a distance of the ligand, or 
#             a list of residues chosen by hand, or both. Whole residues or side 
#             chains only, side chains cut at the CA-CB bond are the usual cluster
#             model, because the backbone is rarely what the chemistry is about 
#             and including it triples the computational cost.
# =============================================================================

from pathlib import Path

import measure as mz
import prepare as prp
import simulation as sim

SCHEMA_VERSION = 1

# Covalent radii, in angstrom, for the elements that turn up in a protein or a
# drug. Two atoms are bonded when they are closer than the sum of their radii
# plus a tolerance -- crude, but it is what everything from PyMOL to VMD does,
# and it does not depend on CONECT records that half the PDB omits.
COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07,
    "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39, "SE": 1.20, "B": 0.84,
    "SI": 1.11, "FE": 1.32, "ZN": 1.22, "MG": 1.41, "CA": 1.76, "MN": 1.39,
    "CU": 1.32, "NI": 1.24, "CO": 1.26, "NA": 1.66, "K": 2.03,
}
BOND_TOLERANCE = 0.45

# Where to put a link hydrogen: standard bond lengths to hydrogen, by the
# element whose bond was cut. The link atom goes along the original bond
# vector at this distance, which keeps the local geometry and the dipole
# roughly right without pretending to be the atom it replaced.
LINK_LENGTHS = {"C": 1.09, "N": 1.01, "O": 0.96, "S": 1.34, "P": 1.42}
DEFAULT_LINK_LENGTH = 1.09

# Backbone atom names, for side-chain-only models.
BACKBONE = {"N", "CA", "C", "O", "OXT", "H", "HA", "HA2", "HA3", "H1", "H2", "H3"}


def is_protein(resname: str, kind: str) -> bool:
    """
    True for an amino acid, including the force-field spellings of one.

    The atom reader classifies residues by name against the twenty standard
    amino acids, so a structure prepared for Amber -- where every disulfide
    cysteine is CYX and every histidine is HID, HIE or HIP -- has those
    residues classified as hetero. Left uncorrected, a side-chain-only QM
    region silently takes the whole backbone of every such residue with it and
    cuts two peptide bonds per residue instead of one C-C bond.
    """
    return kind == "protein" or resname.upper() in prp.FF_VARIANTS

# Charges of the ionisable residues at neutral pH. Reused from simulation.py
# so that the charge a QM region reports and the charge tleap builds into a
# topology come from one table rather than two that drift apart.
RESIDUE_CHARGES = sim.RESIDUE_CHARGES
ION_CHARGES = sim.ION_CHARGES

# Methods and basis sets worth offering, with what each is for. Opinionated on
# purpose: the menu is where someone picks a level of theory, and an unlabelled
# list of forty functionals helps nobody.
METHODS = {
    "B3LYP": "The default hybrid functional. Well understood, cheap, and poor "
             "at dispersion — pair it with an empirical correction.",
    "B3LYP-D3": "B3LYP with Grimme's dispersion correction. What most people "
                "mean by B3LYP for a binding-site model.",
    "M06-2X": "Good for non-covalent interactions and thermochemistry of "
              "organic molecules; expensive.",
    "wB97XD": "Range-separated with dispersion built in. A reliable default "
              "for ligand–protein interaction energies.",
    "PBE0": "Hybrid GGA, dependable for geometries and metals.",
    "HF": "Hartree-Fock. Rarely the answer on its own; useful as a reference.",
    "MP2": "Post-HF, correlated. Accurate for interactions and much more "
           "expensive; for single points rather than optimisation.",
}

BASIS_SETS = {
    "6-31G(d)": "The standard workhorse for geometry optimisation.",
    "6-311+G(d,p)": "Larger, with diffuse functions — needed for anions.",
    "def2-SVP": "Ahlrichs double-zeta; the usual starting point in ORCA.",
    "def2-TZVP": "Triple-zeta. What to use for energies once the geometry is done.",
    "cc-pVDZ": "Dunning double-zeta, for correlated methods.",
    "cc-pVTZ": "Dunning triple-zeta.",
}

JOB_TYPES = {
    "sp": "Single point — the energy of this geometry, nothing moved.",
    "opt": "Optimise the geometry. In a cluster model the cut atoms must be "
           "frozen or the model falls apart.",
    "opt freq": "Optimise, then frequencies — needed for thermochemistry and "
                "to prove the structure is a minimum.",
    "freq": "Frequencies at this geometry.",
}

SOLVENT_MODELS = {
    "none": "Gas phase. For a buried site this is often defensible; for an "
            "exposed one it is not.",
    "water": "Continuum water (SMD).",
    "protein-like": "Continuum with a low dielectric (ether, ~4) standing in "
                    "for a protein interior.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Region selection
# ═══════════════════════════════════════════════════════════════════════════════

def _radius(element: str) -> float:
    return COVALENT_RADII.get((element or "C").upper(), 0.77)


def residue_keys(atoms: dict) -> list:
    """Residue key per atom: (chain, resseq, icode)."""
    return list(zip(atoms["chain"], atoms["resseq"], atoms["icode"]))


def residues_near(pdb_path, center_codes=None, *, center_keys=None,
                  radius: float = 5.0, include_water: bool = False,
                  atoms: dict = None) -> list:
    """
    Residues with any atom within `radius` of the centre.

    Everything after the centre is keyword-only. Passing a radius as the third
    positional argument is the obvious mistake -- it binds to `center_keys`,
    which then fails somewhere unrelated with "'float' object is not iterable".

    Args:
        pdb_path    : Structure to read (ignored when `atoms` is given).
        center_codes: Component codes forming the centre, e.g. ["BEN"].
        center_keys : Residue keys forming the centre, as an alternative.
        radius      : Cutoff in angstrom, measured atom to atom.
        include_water: Count waters as residues that can be selected.
        atoms       : A pre-read atom table.

    Returns:
        [(key, resname, closest distance)], nearest first, excluding the centre.
    """
    import numpy as np

    atoms = atoms or mz.read_atoms(pdb_path)
    keys = residue_keys(atoms)
    codes = {c.strip().upper() for c in (center_codes or [])}
    wanted = set(center_keys or [])

    centre = [i for i, key in enumerate(keys)
              if (atoms["resname"][i].upper() in codes) or (key in wanted)]
    if not centre:
        return []

    centre_xyz = atoms["xyz"][centre]
    out = {}
    for i, key in enumerate(keys):
        if i in set(centre):
            continue
        if not include_water and atoms["kind"][i] == "water":
            continue
        d = float(np.min(np.linalg.norm(centre_xyz - atoms["xyz"][i], axis=1)))
        if d <= radius and (key not in out or d < out[key][1]):
            out[key] = (atoms["resname"][i], d)
    return sorted(((k, v[0], round(v[1], 2)) for k, v in out.items()),
                  key=lambda r: r[2])


def build_region(pdb_path, center_codes=None, center_keys=None,
                 residue_keys_wanted=None, side_chains_only: bool = False,
                 keep_backbone_of=None, atoms: dict = None) -> dict:
    """
    Assemble the atoms of a QM region and cap the bonds that leave it.

    Args:
        pdb_path          : Structure to cut the region out of.
        center_codes      : Component codes always included whole (the ligand).
        center_keys       : Residue keys always included whole.
        residue_keys_wanted: Additional residues to include.
        side_chains_only  : Include only side-chain atoms of the additional
                            residues, cutting the CA-CB bond. The usual cluster
                            model: the backbone is rarely what the chemistry is
                            about, and including it roughly triples the cost.
        keep_backbone_of  : Residue keys to include whole even in side-chain mode.
        atoms             : A pre-read atom table.

    Returns:
        {atoms: [...], links: [...], residues: [...], charge, formula, warnings}
        where each atom is {element, x, y, z, name, resname, key, link}.
    """
    import numpy as np

    atoms = atoms or mz.read_atoms(pdb_path)
    keys = residue_keys(atoms)
    codes = {c.strip().upper() for c in (center_codes or [])}
    centre_keys = set(center_keys or [])
    extra = set(residue_keys_wanted or [])
    whole = set(keep_backbone_of or [])

    selected, residues = [], {}
    for i, key in enumerate(keys):
        name = atoms["name"][i]
        resname = atoms["resname"][i].upper()
        is_centre = resname in codes or key in centre_keys
        if not (is_centre or key in extra):
            continue
        if (side_chains_only and not is_centre and key not in whole
                and is_protein(resname, atoms["kind"][i])):
            if name in BACKBONE or (name == "CB" and False):
                continue
            if resname == "GLY":
                continue        # nothing but backbone; a glycine side chain is H
        selected.append(i)
        residues.setdefault(key, resname)

    chosen = set(selected)
    links, warnings = [], []
    # A bond leaving the region is one between a selected atom and an
    # unselected one, found by distance. Hydrogens are skipped as partners:
    # a missing hydrogen is not a cut bond, it is a structure without hydrogens.
    for i in selected:
        if (atoms["element"][i] or "").upper() == "H":
            continue
        ri = _radius(atoms["element"][i])
        for j in range(len(keys)):
            if j in chosen or (atoms["element"][j] or "").upper() == "H":
                continue
            if atoms["kind"][j] == "water":
                continue
            d = float(np.linalg.norm(atoms["xyz"][i] - atoms["xyz"][j]))
            if d < ri + _radius(atoms["element"][j]) + BOND_TOLERANCE:
                links.append(_link_atom(atoms, i, j, d))

    out_atoms = []
    for i in selected:
        x, y, z = atoms["xyz"][i]
        out_atoms.append({
            "element": (atoms["element"][i] or atoms["name"][i][:1]).capitalize(),
            "x": float(x), "y": float(y), "z": float(z),
            "name": atoms["name"][i], "resname": atoms["resname"][i],
            "key": keys[i], "link": False, "index": i,
        })
    out_atoms.extend(links)

    charge, breakdown = region_charge(residues, side_chains_only)
    # Link atoms are hydrogens too, but placed by us: they must not count, or
    # an unprotonated X-ray file (3PTB) passes as protonated.
    if not any(a["element"] == "H" and not a["link"] for a in out_atoms):
        warnings.append(
            "There is not a single hydrogen in this region. A QM calculation on "
            "heavy atoms alone is meaningless — protonate the structure first "
            "(the Prepare tab does it with reduce).")
    metals = {a["element"].upper() for a in out_atoms} & {
        "FE", "ZN", "CU", "MN", "CO", "NI", "MG", "MO"}
    if metals:
        warnings.append(
            f"This region contains {', '.join(sorted(metals))}. The spin state is "
            "yours to choose — the default multiplicity of 1 is very likely wrong "
            "for an open-shell transition metal.")
    return {
        "atoms": out_atoms,
        "links": links,
        "residues": sorted(residues.items()),
        "charge": charge,
        "charge_breakdown": breakdown,
        "formula": _formula(out_atoms),
        "warnings": warnings,
        "source": str(pdb_path),
    }


def _link_atom(atoms: dict, inside: int, outside: int, distance: float) -> dict:
    """
    A hydrogen placed along a broken bond.

    Put at a standard bond length from the atom that stays, in the direction of
    the atom that goes. Replacing a carbon with a hydrogen changes the
    electronics of that bond, which is exactly why link atoms are placed on
    non-polar C-C bonds wherever there is a choice -- cutting through a
    peptide bond or next to a charged group is how a cluster model acquires an
    artefact nobody can see in the output.
    """
    import numpy as np

    keep = atoms["xyz"][inside]
    gone = atoms["xyz"][outside]
    direction = gone - keep
    norm = float(np.linalg.norm(direction))
    length = LINK_LENGTHS.get((atoms["element"][inside] or "C").upper(),
                              DEFAULT_LINK_LENGTH)
    position = keep + direction / (norm or 1.0) * length
    return {
        "element": "H", "x": float(position[0]), "y": float(position[1]),
        "z": float(position[2]), "name": "HL", "link": True,
        "resname": atoms["resname"][inside],
        "key": (atoms["chain"][inside], atoms["resseq"][inside],
                atoms["icode"][inside]),
        "replaces": {"element": atoms["element"][outside],
                     "name": atoms["name"][outside],
                     "resname": atoms["resname"][outside],
                     "distance": round(distance, 2)},
        "index": -1,
    }


def region_charge(residues: dict, side_chains_only: bool = False) -> tuple:
    """
    Formal charge of a set of residues, and the breakdown behind it.

    Ligands are not in the table and contribute nothing here -- their charge
    is a separate input, because the only way to know it is chemistry the
    coordinates do not carry.

    Returns:
        (total, [(key, resname, charge)]) for the residues that carry one.
    """
    total, breakdown = 0, []
    for key, resname in sorted(residues.items()):
        q = RESIDUE_CHARGES.get(resname.upper())
        if q is None:
            q = ION_CHARGES.get(resname.upper())
        if q:
            total += q
            breakdown.append((key, resname, q))
    return total, breakdown


def _formula(region_atoms: list) -> str:
    counts = {}
    for atom in region_atoms:
        el = atom["element"].capitalize()
        counts[el] = counts.get(el, 0) + 1
    order = ["C", "H", "N", "O"]
    keys = [k for k in order if k in counts] + sorted(k for k in counts
                                                      if k not in order)
    return "".join(f"{k}{counts[k] if counts[k] > 1 else ''}" for k in keys)


def region_pdb(region: dict, dest) -> str:
    """
    Write the region as a PDB so it can be loaded into the viewer and looked at.

    Worth doing before spending anything on the calculation: a region that
    looks wrong -- a residue half in, a link atom inside a ring -- is obvious
    in three dimensions and invisible in a list of coordinates.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, atom in enumerate(region["atoms"], start=1):
        chain = (atom["key"][0] or "A") if atom.get("key") else "A"
        resseq = atom["key"][1] if atom.get("key") else 1
        try:
            resseq = int(resseq)
        except (TypeError, ValueError):
            resseq = 1
        name = atom["name"][:4].ljust(4) if len(atom["name"]) >= 4 else \
            (" " + atom["name"]).ljust(4)
        lines.append(
            f"HETATM{i:>5} {name}{atom['resname'][:3]:>3} {chain}{resseq:>4}    "
            f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
            f"{1.0:6.2f}{0.0:6.2f}          {atom['element'].upper():>2}\n")
    dest.write_text("".join(lines) + "END\n")
    return str(dest)


# ═══════════════════════════════════════════════════════════════════════════════
# Input files
# ═══════════════════════════════════════════════════════════════════════════════

def _route_solvent(model: str, program: str) -> str:
    if model == "water":
        return {"gaussian": "scrf=(smd,solvent=water)",
                "orca": "CPCM(water)", "psi4": "water"}.get(program, "")
    if model == "protein-like":
        return {"gaussian": "scrf=(smd,solvent=diethylether)",
                "orca": "CPCM(diethylether)", "psi4": "diethylether"}.get(program, "")
    return ""


def gaussian_input(region: dict, charge: int = None, multiplicity: int = 1,
                   method: str = "B3LYP-D3", basis: str = "6-31G(d)",
                   job: str = "opt freq", solvent: str = "none",
                   processors: int = 8, memory_gb: int = 16,
                   freeze_links: bool = True, title: str = "") -> str:
    """
    A Gaussian input file for a QM region.

    freeze_links matters for anything but a single point: a cluster model
    optimised with everything free relaxes into a shape the protein would
    never allow, because the protein is not there. Freezing the link atoms --
    and, in practice, the atoms they attach to -- keeps the model in the
    geometry the crystal structure put it in.
    """
    charge = region["charge"] if charge is None else charge
    method_name = method.replace("-D3", "")
    dispersion = " EmpiricalDispersion=GD3" if method.endswith("-D3") else ""
    route = f"#p {method_name}/{basis} {job}{dispersion}"
    solvent_key = _route_solvent(solvent, "gaussian")
    if solvent_key:
        route += f" {solvent_key}"
    # No opt=modredundant: the freeze flags are the integer column in the
    # geometry block, which plain `opt` reads directly. Adding the keyword as
    # well asks Gaussian for a redundant-coordinate section that is not there.

    lines = [f"%NProcShared={processors}", f"%Mem={memory_gb}GB",
             "%Chk=region.chk", route, "",
             title or f"QM region from {Path(region['source']).name} — "
                      f"{region['formula']}", "",
             f"{charge} {multiplicity}"]
    for atom in region["atoms"]:
        frozen = -1 if (freeze_links and atom.get("link")) else 0
        if freeze_links and "opt" in job:
            lines.append(f" {atom['element']:<2} {frozen:>2} "
                         f"{atom['x']:12.6f}{atom['y']:12.6f}{atom['z']:12.6f}")
        else:
            lines.append(f" {atom['element']:<2} "
                         f"{atom['x']:12.6f}{atom['y']:12.6f}{atom['z']:12.6f}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


def orca_input(region: dict, charge: int = None, multiplicity: int = 1,
               method: str = "B3LYP-D3", basis: str = "def2-SVP",
               job: str = "opt freq", solvent: str = "none",
               processors: int = 8, memory_mb: int = 3000,
               freeze_links: bool = True) -> str:
    """An ORCA input for the same region."""
    charge = region["charge"] if charge is None else charge
    keywords = [method.replace("-D3", ""), basis]
    if method.endswith("-D3"):
        keywords.append("D3BJ")
    keywords += [k.upper() for k in job.split()]
    solvent_key = _route_solvent(solvent, "orca")
    if solvent_key:
        keywords.append(solvent_key)

    lines = [f"! {' '.join(keywords)}", f"%pal nprocs {processors} end",
             f"%maxcore {memory_mb}"]
    if freeze_links and "opt" in job:
        frozen = [i for i, a in enumerate(region["atoms"]) if a.get("link")]
        if frozen:
            lines += ["%geom Constraints"]
            lines += [f"    {{ C {i} C }}" for i in frozen]
            lines += ["  end", "end"]
    lines.append(f"* xyz {charge} {multiplicity}")
    for atom in region["atoms"]:
        lines.append(f"  {atom['element']:<2} {atom['x']:12.6f}"
                     f"{atom['y']:12.6f}{atom['z']:12.6f}")
    lines += ["*", ""]
    return "\n".join(lines)


def psi4_input(region: dict, charge: int = None, multiplicity: int = 1,
               method: str = "B3LYP-D3", basis: str = "def2-SVP",
               job: str = "opt", memory_gb: int = 16) -> str:
    """A Psi4 input for the same region."""
    charge = region["charge"] if charge is None else charge
    method_name = method.lower().replace("-d3", "-d3bj")
    task = {"sp": "energy", "opt": "optimize", "opt freq": "frequencies",
            "freq": "frequencies"}.get(job, "energy")
    lines = [f"memory {memory_gb} GB", "", "molecule region {",
             f"  {charge} {multiplicity}"]
    for atom in region["atoms"]:
        lines.append(f"  {atom['element']:<2} {atom['x']:12.6f}"
                     f"{atom['y']:12.6f}{atom['z']:12.6f}")
    lines += ["  no_reorient", "  no_com", "}", "",
              f"set basis {basis}", "set scf_type df", "",
              f"{task}('{method_name}')", ""]
    return "\n".join(lines)


def xyz_file(region: dict, comment: str = "") -> str:
    """The region as a plain .xyz, which every program on earth can read."""
    lines = [str(len(region["atoms"])),
             comment or f"{region['formula']} from {Path(region['source']).name}"]
    for atom in region["atoms"]:
        lines.append(f"{atom['element']:<2} {atom['x']:12.6f}"
                     f"{atom['y']:12.6f}{atom['z']:12.6f}")
    return "\n".join(lines) + "\n"


def region_report(region: dict) -> str:
    """What was cut, in words -- for the panel and for the input file header."""
    lines = [f"{len(region['atoms'])} atoms ({region['formula']}), "
             f"{len(region['residues'])} residues, "
             f"{len(region['links'])} link atoms, net charge {region['charge']:+d}."]
    if region["charge_breakdown"]:
        lines.append("Charged residues: " + ", ".join(
            f"{name}{key[1]}{key[2]} {q:+d}"
            for key, name, q in region["charge_breakdown"]))
    if region["links"]:
        cuts = {}
        for link in region["links"]:
            label = (f"{link['resname']}{link['key'][1]} "
                     f"→ {link['replaces']['resname']}"
                     f"{link['replaces']['name']}")
            cuts[label] = cuts.get(label, 0) + 1
        lines.append("Bonds cut: " + "; ".join(sorted(cuts)))
    for warning in region["warnings"]:
        lines.append("WARNING: " + warning)
    return "\n".join(lines)
