# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-12
# Summary   : This module is used to turn a prepared structure into a simulation 
#             to run for AMBER using Amber topology and MD inputs, GROMACS run files, 
#             and Rosetta job files including the ligand parameters that are
#             the usual reason none of it works.
#
#             prepare.py answers "is this structure sane". This module answers
#             "what does the engine need", which is a different and much more
#             opinionated list: a force field and a water model that were
#             parameterised together, a box big enough that the protein never
#             sees its own periodic image, counter-ions plus however much salt
#             the experiment had, and, for every ligand, a set of parameters
#             that no protein force field contains.
#
#             Two of those steps actually run here rather than being described.
#             antechamber and parmchk2 generate GAFF2 parameters and AM1-BCC
#             charges for a ligand pulled straight out of the crystal
#             structure, and tleap builds the topology -- both ship with
#             AmberTools and both work, which makes the difference between a
#             page of instructions and a directory of files you can submit.
#
#             What is generated rather than run is anything needing software
#             that is not here: GROMACS run parameters, and Rosetta job files.
#             Those are written as complete, commented inputs with the
#             commands to run them, and the module says plainly which of them
#             it could not check.
#
#             The numbers in the MD inputs are conventional starting points,
#             not results: 2 fs with SHAKE, 10 A cutoffs, Langevin at 1 ps-1,
#             Monte Carlo barostat. They are written into the files where they
#             can be read and changed, rather than hidden behind a preset.
# =============================================================================

import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import requests

import membrane as mem
import prepare as prp

SCHEMA_VERSION = 1

# Water is 55.5 M, so a box holding N water molecules holds N/55.5 mol-equivalents
# of volume; the ion count for a target concentration follows from that. It
# ignores the volume the solute displaces, which makes the real concentration
# slightly higher than asked for -- the usual correction people apply, and the
# reason the panel shows the number it is about to add rather than only the
# concentration.
WATER_MOLARITY = 55.5

# Protein force fields, with the water model each was parameterised against.
# Pairing ff19SB with TIP3P is a common and quietly wrong choice: ff19SB's
# backbone parameters were fitted with OPC, and using TIP3P gives back much of
# what ff19SB was built to fix.
PROTEIN_FFS = {
    "ff19SB": {"water": "opc", "note": "Current default. Fitted against OPC water; "
                                       "use it with OPC unless you have a reason not to."},
    "ff14SB": {"water": "tip3p", "note": "The long-standing workhorse, and what most "
                                         "published protocols used. Pairs with TIP3P."},
    "ff15ipq": {"water": "spce", "note": "Fitted self-consistently with SPC/E-b."},
}

WATER_MODELS = {
    "opc": "4-point, the most accurate common model; pairs with ff19SB.",
    "tip3p": "3-point, cheapest and most widely used; pairs with ff14SB.",
    "tip4pew": "4-point Ewald-corrected.",
    "spce": "3-point, extended simple point charge.",
}

BOX_SHAPES = {
    "octahedron": ("solvateoct", "A truncated octahedron holds the same buffer "
                                 "in about 70% of the water of a cube -- and the "
                                 "water is most of the cost of the simulation."),
    "cubic": ("solvatebox", "A cube. Simpler to reason about, and required by "
                            "some analysis tools, but roughly 40% more water "
                            "for the same clearance."),
}

# Ion pairs, with what they are for.
ION_PAIRS = {
    "Na+/Cl-": ("Na+", "Cl-", "Physiological default for most systems."),
    "K+/Cl-": ("K+", "Cl-", "Intracellular ionic composition; the usual choice "
                            "for membrane proteins and nucleic acids."),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Backend
# ═══════════════════════════════════════════════════════════════════════════════

def backend() -> dict:
    """AmberTools paths, reusing the discovery the membrane module already does."""
    return mem.find_backend()


def _amber_exe(name: str) -> str:
    """Path to an AmberTools executable, or "" if it is not installed."""
    home = backend().get("amberhome", "")
    if home:
        candidate = Path(home) / "bin" / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name) or ""


def tools_available() -> dict:
    """Which of the executables this module drives are actually present."""
    return {name: bool(_amber_exe(name))
            for name in ("tleap", "antechamber", "parmchk2", "sqm")}


def available() -> bool:
    """True when a topology can be built here."""
    return bool(_amber_exe("tleap"))


def _env() -> dict:
    """Environment with AMBERHOME set -- antechamber and tleap both need it."""
    env = dict(os.environ)
    home = backend().get("amberhome", "")
    if home:
        env["AMBERHOME"] = home
        env["PATH"] = str(Path(home) / "bin") + os.pathsep + env.get("PATH", "")
    return env


# ═══════════════════════════════════════════════════════════════════════════════
# Box and ions
# ═══════════════════════════════════════════════════════════════════════════════

def ion_counts(n_waters: int, concentration: float, cation: str = "Na+",
               anion: str = "Cl-", net_charge: float = 0.0,
               offset_counter_ions: bool = True) -> dict:
    """
    How many ions of bulk salt to add for a target concentration.

    Equal numbers of each, always. The counter-ions that neutralise the solute
    are a separate step that leap does first, and it leaves the system neutral;
    anything added afterwards has to be charge-balanced or it puts the charge
    straight back. Subtracting the counter-ions from one side of the salt --
    which looks like the right way to hit a concentration -- built a trypsin
    system here with 19 sodium and 21 chloride and a net charge of +6, and
    nothing in the leap output called that an error.

    The concentration correction is applied to the pair count instead: ions
    added for neutralisation already contribute to the ionic strength, so the
    number of bulk pairs is reduced by half of them.

    Args:
        n_waters     : Water molecules in the solvated box.
        concentration: Target salt concentration in molar.
        cation, anion: Ion names.
        net_charge   : Solute charge, i.e. how many counter-ions leap will add.
        offset_counter_ions: Count those towards the concentration.

    Returns:
        {ion: count}, equal for both species.
    """
    if not n_waters or concentration <= 0:
        return {}
    pairs = int(round(n_waters * concentration / WATER_MOLARITY))
    if offset_counter_ions:
        pairs -= int(round(abs(net_charge) / 2.0))
    if pairs <= 0:
        return {}
    return {cation: pairs, anion: pairs}


def estimate_waters(pdb_path, buffer_a: float = 12.0,
                    shape: str = "octahedron") -> int:
    """
    Rough number of waters a solvated box will hold.

    Only used to turn a salt concentration into an ion count before tleap has
    run. The box is the solute's bounding box grown by the buffer on all
    sides, minus the volume the solute itself occupies (taken as 1.21 A^3 per
    dalton, the usual protein partial specific volume), at 0.0334 waters per
    cubic angstrom. It is an estimate, and the panel says so -- after tleap
    runs, the real count is read from the topology instead.
    """
    xs, ys, zs, atoms = [], [], [], 0
    for line in prp.read_lines(pdb_path):
        if not prp._is_coord(line):
            continue
        try:
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
        except ValueError:
            continue
        atoms += 1
    if not xs:
        return 0
    dims = [max(v) - min(v) + 2 * buffer_a for v in (xs, ys, zs)]
    volume = dims[0] * dims[1] * dims[2]
    if shape == "octahedron":
        volume *= 0.77          # truncated octahedron inscribed in that box
    solute_volume = atoms * 10.0   # ~10 A^3 per heavy-ish atom, including H
    return max(0, int((volume - solute_volume) * 0.0334))


# ═══════════════════════════════════════════════════════════════════════════════
# Ligand parameters
# ═══════════════════════════════════════════════════════════════════════════════

def extract_residue(pdb_path, code: str, dest, chain: str = "",
                    keep_hydrogens: bool = True) -> tuple:
    """
    Pull one chemical component out of a structure into its own PDB.

    Args:
        pdb_path : The structure.
        code     : Residue code, e.g. "BEN".
        dest     : Where to write the extracted component.
        chain    : Restrict to one chain; empty means the first copy found.
        keep_hydrogens: Keep any hydrogens already present.

    Returns:
        (path, atom count, formula) or (None, 0, "") when it is not there.
    """
    code = code.strip().upper()
    wanted, first_key = [], None
    for line in prp.read_lines(pdb_path):
        if not prp._is_coord(line) or prp._resname(line) != code:
            continue
        if chain and line[21] != chain:
            continue
        key = prp._reskey(line)
        if first_key is None:
            first_key = key
        if key != first_key:
            continue
        if not keep_hydrogens and prp._is_hydrogen(line):
            continue
        wanted.append(line)
    if not wanted:
        return None, 0, ""

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # HETATM records renumbered from 1: antechamber reads the file as a single
    # molecule and is happier without the gaps the parent structure leaves.
    body = []
    for i, line in enumerate(wanted, start=1):
        body.append("HETATM" + f"{i:>5}" + line[11:])
    dest.write_text("".join(body) + "END\n")

    formula = {}
    for line in wanted:
        el = prp._element(line) or "?"
        formula[el] = formula.get(el, 0) + 1
    text = "".join(f"{el}{n if n > 1 else ''}"
                   for el, n in sorted(formula.items()))
    return str(dest), len(wanted), text


def strip_components(pdb_path, codes, dest) -> tuple:
    """
    Copy a structure without the named components.

    Used to take a parameterised ligand out of the PDB so it can be loaded
    from its mol2 instead: antechamber renames atoms when it reads a molecule
    with real bond orders, so the names in the mol2 and the names in the
    crystal structure no longer agree, and leap matches residues by name.

    Returns:
        (path, atoms removed).
    """
    wanted = {c.strip().upper() for c in codes}
    return str(dest), mem.strip_records(pdb_path, dest, wanted)


def mol2_with_coordinates(mol2_path, coordinates, dest) -> tuple:
    """
    A copy of a parameterised mol2 moved onto another copy's coordinates.

    A structure with four copies of the same ligand needs one unit per copy,
    each where its copy actually sits -- parameterising once and loading it
    four times would stack every copy in the same place. The parameters,
    types and charges are identical by construction, so only the coordinate
    columns change.

    Returns:
        (path, error). Refuses rather than guesses when the atom counts differ.
    """
    lines = Path(mol2_path).read_text(errors="replace").splitlines(True)
    start = next((i for i, l in enumerate(lines) if l.startswith("@<TRIPOS>ATOM")), None)
    end = next((i for i, l in enumerate(lines) if l.startswith("@<TRIPOS>BOND")), None)
    if start is None or end is None:
        return None, "That mol2 file has no atom block."
    block = lines[start + 1:end]
    if len(block) != len(coordinates):
        return None, (f"The mol2 has {len(block)} atoms and this copy has "
                      f"{len(coordinates)}; they are not the same molecule.")
    out = []
    for line, (x, y, z) in zip(block, coordinates):
        parts = line.split()
        if len(parts) < 6:
            return None, "Unexpected mol2 atom line."
        out.append(f"{parts[0]:>7} {parts[1]:<8}{x:10.4f}{y:10.4f}{z:10.4f} "
                   + " ".join(parts[5:]) + "\n")
    Path(dest).write_text("".join(lines[:start + 1] + out + lines[end:]))
    return str(dest), None


def component_copies(pdb_path, code: str) -> list:
    """Every copy of a component in a structure, as [(chain, resseq, icode)]."""
    code = code.strip().upper()
    keys = []
    for line in prp.read_lines(pdb_path):
        if prp._is_coord(line) and prp._resname(line) == code:
            key = prp._reskey(line)
            if key not in keys:
                keys.append(key)
    return keys


def ligand_candidates(pdb_path) -> list:
    """
    Components in a structure that need parameters, with their atom counts.

    Ions are left out: Amber has them. Additives are included but flagged,
    because a glycerol that someone deliberately kept still needs parameters
    like anything else.
    """
    finding = prp.inspect(pdb_path)
    out = []
    for entry in finding["hetero"]:
        if entry["kind"] == "ion":
            continue
        _, atoms, formula = extract_residue(
            pdb_path, entry["code"], Path(tempfile.gettempdir()) / "probe.pdb")
        out.append({"code": entry["code"], "kind": entry["kind"],
                    "name": entry.get("name", ""), "copies": entry["count"],
                    "atoms": atoms, "formula": formula})
    return out


# Atomic numbers, for the electron count that decides whether sqm can run at
# all. Only the elements that turn up in ligands.
ATOMIC_NUMBER = {
    "H": 1, "D": 1, "HE": 2, "LI": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17,
    "K": 19, "CA": 20, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29,
    "ZN": 30, "SE": 34, "BR": 35, "I": 53,
}

CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{code}"


def rdkit_available() -> bool:
    """True when RDKit can be imported."""
    try:
        import importlib
        return importlib.util.find_spec("rdkit") is not None
    except Exception:
        return False


@lru_cache(maxsize=128)
def chem_component(code: str):
    """
    What the PDB chemical dictionary says about a component.

    Returns {name, formula, charge, smiles} or None. The SMILES is the useful
    part: it carries the bond orders and the protonation state that a set of
    crystallographic coordinates does not, and without them there is no way to
    add hydrogens to a ligand correctly.
    """
    code = (code or "").strip().upper()
    if not code:
        return None
    try:
        r = requests.get(CHEMCOMP_URL.format(code=code), timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    info = data.get("chem_comp") or {}
    smiles = ""
    for desc in data.get("pdbx_chem_comp_descriptor") or []:
        if desc.get("type") == "SMILES_CANONICAL" and desc.get("program") == "CACTVS":
            smiles = desc.get("descriptor", "")
            break
        if desc.get("type") == "SMILES_CANONICAL" and not smiles:
            smiles = desc.get("descriptor", "")
    return {
        "code": code,
        "name": info.get("name", ""),
        "formula": (info.get("formula") or "").strip(),
        # RCSB's field is pdbx_formal_charge; "formal_charge" never exists,
        # so reading only that one reported every charge as unknown.
        "charge": info.get("pdbx_formal_charge", info.get("formal_charge")),
        "smiles": smiles,
        "weight": info.get("formula_weight"),
    }


def electron_count(pdb_path, net_charge: int = 0) -> int:
    """Total electrons in a molecule, which must be even for a closed shell."""
    total = 0
    for line in prp.read_lines(pdb_path):
        if prp._is_coord(line):
            total += ATOMIC_NUMBER.get(prp._element(line), 0)
    return total - int(net_charge)


def protonate_ligand(ligand_pdb, dest, code: str = "", smiles: str = "",
                     net_charge=None) -> tuple:
    """
    Add hydrogens to a ligand lifted out of a crystal structure.

    Necessary rather than optional: an X-ray ligand has no hydrogens at all,
    and antechamber hands it to sqm regardless, which fails on an odd electron
    count -- benzamidine out of 3PTB is C7N2, 55 electrons, and the run dies
    with "Cannot properly run sqm" and nothing about the real cause.

    Two routes, in order of how much chemistry they actually know:

      RDKit with the chemical dictionary's SMILES as a template. The SMILES
      carries the bond orders, which coordinates alone do not, so hydrogens go
      on in the right number at the right places. This is what gets
      benzamidine to C7H8N2 rather than the C7H7N2 that geometry alone
      produces.

      reduce, as a fallback. It places hydrogens from the dictionary's
      idealised geometry, which is right for most things and quietly
      incomplete for others: on benzamidine it adds seven of the eight,
      leaving an amidine nitrogen with one hydrogen where it should have two.

    Neither picks a protonation state on its own. The dictionary's state is
    the deposited one, and for anything ionisable that is a chemical decision
    -- benzamidine is neutral in the dictionary and an amidinium cation at
    pH 7. When the user states `net_charge` and it differs from the
    dictionary's, the RDKit route moves protons to reach it
    (_shift_charge) and says which atoms it changed.

    Returns:
        (path, hydrogens added, error, notes). The path is `dest`.
    """
    dest = Path(dest)
    before = sum(1 for l in prp.read_lines(ligand_pdb)
                 if prp._is_coord(l) and prp._is_hydrogen(l))

    if not smiles and code:
        info = chem_component(code)
        smiles = (info or {}).get("smiles", "")

    if smiles and rdkit_available():
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem import AllChem
            RDLogger.DisableLog("rdApp.*")
            mol = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=False, sanitize=False)
            template = Chem.MolFromSmiles(smiles)
            if mol is not None and template is not None:
                fixed = AllChem.AssignBondOrdersFromTemplate(template, mol)
                notes = []
                if net_charge is not None:
                    fixed, notes, err = _shift_charge(fixed, int(net_charge))
                    if err:
                        return None, 0, err, []
                fixed = Chem.AddHs(fixed, addCoords=True)
                Chem.SanitizeMol(fixed)
                Chem.MolToPDBFile(fixed, str(dest))
                # An SDF alongside the PDB, because it is the one that keeps
                # the bond orders -- and those are what antechamber needs in
                # order to type the molecule as the thing it actually is.
                Chem.MolToMolFile(fixed, str(Path(dest).with_suffix(".sdf")))
                after = sum(1 for l in prp.read_lines(dest)
                            if prp._is_coord(l) and prp._is_hydrogen(l))
                if after > before:
                    return str(dest), after - before, None, notes
        except Exception:
            pass        # fall through to reduce rather than failing outright

    path, added, err = prp.add_hydrogens(ligand_pdb, dest, flip=True, his=False)
    return path, added, err, []


def _shift_charge(mol, target: int):
    """
    Add or remove protons on a heavy-atom molecule (bond orders assigned, no
    hydrogens yet) until its formal charge is `target`.

    Only the textbook sites, most basic / most acidic first, so the choice is
    predictable and reported: amidine/guanidine imine N, then aliphatic
    amine N, then pyridine-type aromatic N for a proton; carboxylic, then
    phosphoric/sulfonic acid OH for its removal. Anything else (a charge the
    molecule has no such sites for) is an error, never a guess.

    Returns:
        (mol, notes, error).
    """
    from rdkit import Chem
    mol = Chem.RWMol(mol)
    mol.UpdatePropertyCache(strict=False)
    current = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    delta = target - current
    if delta == 0:
        return mol, [], None

    def label(a):
        info = a.GetPDBResidueInfo()
        return info.GetName().strip() if info else f"{a.GetSymbol()}{a.GetIdx() + 1}"

    def is_amide_n(a):
        for nb in a.GetNeighbors():
            if nb.GetSymbol() in ("C", "S", "P"):
                for b in nb.GetBonds():
                    o = b.GetOtherAtom(nb)
                    if o.GetSymbol() == "O" and b.GetBondTypeAsDouble() == 2:
                        return True
        return False

    def base_rank(a):
        if a.GetSymbol() != "N" or a.GetFormalCharge() != 0 or is_amide_n(a):
            return None
        double_c = [b.GetOtherAtom(a) for b in a.GetBonds()
                    if b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(a).GetSymbol() == "C"]
        if double_c and not a.GetIsAromatic():
            c = double_c[0]
            if any(n.GetSymbol() == "N" and n.GetIdx() != a.GetIdx() for n in c.GetNeighbors()):
                return 0                                      # amidine / guanidine
            return None                                       # plain imine: leave it
        if (not a.GetIsAromatic() and a.GetDegree() < 4
                and all(b.GetBondTypeAsDouble() == 1 for b in a.GetBonds())
                and not any(n.GetIsAromatic() for n in a.GetNeighbors())):
            return 1                                          # aliphatic amine
        if a.GetIsAromatic() and a.GetTotalNumHs() == 0 and a.GetDegree() == 2:
            return 2                                          # pyridine-type
        return None

    def acid_rank(a):
        if a.GetSymbol() != "O" or a.GetFormalCharge() != 0 or a.GetTotalNumHs() != 1:
            return None
        nbs = list(a.GetNeighbors())
        if len(nbs) != 1:
            return None
        x = nbs[0]
        has_oxo = any(b.GetOtherAtom(x).GetSymbol() == "O" and b.GetBondTypeAsDouble() == 2
                      for b in x.GetBonds())
        if not has_oxo:
            return None
        return {"C": 0, "P": 1, "S": 1}.get(x.GetSymbol())

    rank = base_rank if delta > 0 else acid_rank
    sites = sorted((r, a.GetIdx()) for a in mol.GetAtoms()
                   if (r := rank(a)) is not None)
    if len(sites) < abs(delta):
        kind = ("basic nitrogens to protonate" if delta > 0
                else "acidic OH groups to deprotonate")
        return None, [], (
            f"Cannot reach a net charge of {target:+d}: the dictionary form is "
            f"{current:+d}, which needs {abs(delta)} {kind}, and the molecule "
            f"has {len(sites)}. Check the charge, or supply a protonated ligand.")
    notes = []
    for _, idx in sites[:abs(delta)]:
        a = mol.GetAtomWithIdx(idx)
        if delta > 0:
            a.SetFormalCharge(1)
            a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
            notes.append(f"protonated {label(a)}")
        else:
            a.SetFormalCharge(-1)
            a.SetNumExplicitHs(0)
            notes.append(f"deprotonated {label(a)}")
        a.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    return mol.GetMol(), [
        f"To reach the stated net charge {target:+d} (dictionary form "
        f"{current:+d}): " + ", ".join(notes) + "."], None


def run_antechamber(ligand_pdb, out_dir, code: str, net_charge: int = 0,
                    charge_method: str = "bcc", atom_type: str = "gaff2",
                    multiplicity: int = 1, add_hydrogens: bool = True,
                    timeout: int = 3600) -> tuple:
    """
    Generate GAFF2 parameters and AM1-BCC charges for a ligand.

    The net charge is an input and not a guess. antechamber will accept any
    number it is given and produce a molecule with the wrong number of
    electrons without complaining, and there is no reliable way to read the
    charge off a PDB file -- protonation states are not in it. A carboxylate
    read as a neutral acid is a difference of one unit of charge on the thing
    the whole calculation is about.

    Args:
        ligand_pdb   : Single-component PDB from extract_residue().
        out_dir      : Working directory; antechamber litters, so give it its own.
        code         : Residue name to write into the parameters.
        net_charge   : Formal charge of the ligand.
        charge_method: "bcc" (AM1-BCC, the usual choice) or "gas" (Gasteiger,
                       fast and much cruder).
        atom_type    : "gaff2" or "gaff".
        multiplicity : Spin multiplicity, 1 for a closed shell.
        add_hydrogens: Protonate the ligand first when it has no hydrogens,
                       which an X-ray ligand never does.
        timeout      : Seconds. AM1-BCC on a large flexible ligand is minutes.

    Returns:
        (mol2 path, frcmod path, log text, error or None).
    """
    exe, chk = _amber_exe("antechamber"), _amber_exe("parmchk2")
    if not exe or not chk:
        return None, None, "", ("antechamber and parmchk2 were not found. They "
                                "ship with AmberTools.")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / f"{code}_in.pdb"
    shutil.copyfile(ligand_pdb, src)
    # electron_count/extract_formula read PDB records only; src may become
    # an SDF below, which they would count as 0 electrons ("is  with a net
    # charge of 1, which is -1 electrons" for 3PTB's benzamidine).
    count_pdb = src

    notes = []
    sdf = out_dir / f"{code}.sdf"
    has_h = any(prp._is_hydrogen(l) for l in prp.read_lines(src)
                if prp._is_coord(l))
    if add_hydrogens and not has_h:
        protonated, added, err, shift = protonate_ligand(
            src, out_dir / f"{code}_h.pdb", code=code, net_charge=net_charge)
        if err and err.startswith("Cannot reach"):
            return None, None, "", f"{code}: {err}"
        if err or not added:
            return None, None, "", (
                f"{code} has no hydrogens, and they could not be added "
                f"automatically ({err or 'reduce added none'}). antechamber will "
                "hand it to sqm with the wrong number of electrons. Protonate it "
                "first — the protonation state is a chemical decision, and for a "
                "ligand with an ionisable group it is the decision that matters "
                "most.")
        shutil.copyfile(protonated, src)
        count_pdb = src
        notes.append(f"{added} hydrogens were added to {code} before "
                     "parameterisation.")
        notes.extend(shift)
        if Path(str(protonated)).with_suffix(".sdf").exists():
            shutil.copyfile(Path(str(protonated)).with_suffix(".sdf"), sdf)
            src = sdf

    # sqm cannot run an open shell from antechamber, so an odd electron count
    # is a dead end -- and saying so here, with the arithmetic, beats the
    # "Cannot properly run sqm" that comes out of it two minutes later.
    electrons = electron_count(count_pdb, net_charge)
    if electrons % 2 and int(multiplicity) == 1:
        formula = extract_formula(count_pdb)
        return None, None, "\n".join(notes), (
            f"{code} as it stands is {formula} with a net charge of {net_charge}, "
            f"which is {electrons} electrons — an odd number, so it cannot be a "
            f"closed shell. Either the net charge is wrong, or a hydrogen is "
            f"missing. Check the protonation state against what the chemistry "
            f"says: the dictionary's state is the deposited one, not necessarily "
            f"the one at your pH.")
    mol2 = out_dir / f"{code}.mol2"
    frcmod = out_dir / f"{code}.frcmod"
    log = []

    # SDF rather than PDB whenever RDKit could write one. A PDB carries no
    # bond orders, so antechamber perceives them from distances -- and gets
    # benzamidine's benzene ring as cyclohexane, typing every aromatic carbon
    # c3 instead of ca. The parameters that come out are for a different
    # molecule, and nothing in the output says so.
    fmt = "mdl" if src.suffix.lower() in (".sdf", ".mol") else "pdb"
    cmd = [exe, "-i", src.name, "-fi", fmt, "-o", mol2.name, "-fo", "mol2",
           "-c", charge_method, "-nc", str(int(net_charge)), "-m", str(int(multiplicity)),
           "-at", atom_type, "-rn", code[:3], "-s", "2", "-pf", "y"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=str(out_dir), env=_env())
    except subprocess.TimeoutExpired:
        return None, None, "", (f"antechamber did not finish within {timeout} s. "
                                "AM1-BCC on a large ligand is slow; try the "
                                "Gasteiger charge method to check the setup first.")
    except Exception as e:
        return None, None, "", f"Could not run antechamber: {e}"
    log.append(proc.stdout + proc.stderr)
    if not mol2.exists() or mol2.stat().st_size == 0:
        return None, None, "\n".join(notes + log), _antechamber_error("\n".join(log))

    try:
        proc = subprocess.run([chk, "-i", mol2.name, "-f", "mol2", "-o", frcmod.name,
                               "-s", "2" if atom_type == "gaff2" else "1"],
                              capture_output=True, text=True, timeout=600,
                              cwd=str(out_dir), env=_env())
        log.append(proc.stdout + proc.stderr)
    except Exception as e:
        return str(mol2), None, "\n".join(notes + log), f"parmchk2 failed: {e}"
    if not frcmod.exists():
        return str(mol2), None, "\n".join(notes + log), "parmchk2 produced no frcmod."
    return str(mol2), str(frcmod), "\n".join(notes + log), None


def _antechamber_error(log: str) -> str:
    """
    The informative line out of an antechamber failure.

    The last line is almost never the useful one: antechamber reports "Cannot
    properly run sqm" whatever went wrong, while the sentence that explains it
    -- an odd electron count, an unrecognised element, a valence it could not
    satisfy -- is further up.
    """
    for marker in ("number of electrons is odd", "net charge", "Unknown element",
                   "cannot be assigned", "Weird atomic valence"):
        for line in log.splitlines():
            if marker.lower() in line.lower():
                return (line.strip()[:300] +
                        "  — check the net charge and that the ligand is protonated.")
    for marker in ("Error", "ERROR", "Fatal", "Cannot", "Unable"):
        for line in log.splitlines():
            if marker in line:
                return line.strip()[:300]
    return "antechamber produced no mol2 file; see the log."


def extract_formula(pdb_path) -> str:
    """Molecular formula of a single-component PDB, for reporting."""
    counts = {}
    for line in prp.read_lines(pdb_path):
        if prp._is_coord(line):
            el = prp._element(line) or "?"
            counts[el] = counts.get(el, 0) + 1
    order = ["C", "H", "N", "O"]
    keys = [k for k in order if k in counts] + sorted(k for k in counts if k not in order)
    return "".join(f"{k.capitalize()}{counts[k] if counts[k] > 1 else ''}" for k in keys)


def missing_parameters(frcmod_path) -> list:
    """
    Parameters parmchk2 had to guess, as a list of lines.

    Worth surfacing: a frcmod full of "ATTN, need revision" is parmchk2 saying
    it has no data for part of this molecule and has substituted something
    plausible. That is a normal outcome for an unusual ligand and a reason to
    check the geometry afterwards, not a reason to stop -- but it should never
    be a surprise discovered after the simulation.
    """
    try:
        text = Path(frcmod_path).read_text(errors="replace")
    except Exception:
        return []
    return [line.strip() for line in text.splitlines() if "ATTN" in line]


# ═══════════════════════════════════════════════════════════════════════════════
# tleap
# ═══════════════════════════════════════════════════════════════════════════════

def run_tleap(script_text: str, work_dir, extra_files=None,
              timeout: int = 1800) -> dict:
    """
    Run tleap on a script and report what it produced.

    Args:
        script_text: The leap input.
        work_dir   : Directory to run in; every path in the script is resolved
                     against it, so the structure and any ligand parameters
                     must be there.
        extra_files: Paths copied into the working directory first.
        timeout    : Seconds.

    Returns:
        {ok, prmtop, inpcrd, log, errors, warnings, atoms, waters, charge}.
    """
    exe = _amber_exe("tleap")
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if not exe:
        return {"ok": False, "log": "", "errors": ["tleap was not found."],
                "warnings": [], "prmtop": None, "inpcrd": None}
    for path in extra_files or []:
        src = Path(path)
        if src.exists() and src.parent.resolve() != work_dir:
            shutil.copyfile(src, work_dir / src.name)

    script = work_dir / "build.leap"
    script.write_text(script_text)
    try:
        proc = subprocess.run([exe, "-f", script.name], capture_output=True,
                              text=True, timeout=timeout, cwd=str(work_dir),
                              env=_env())
        log = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {"ok": False, "log": "", "errors": [f"tleap timed out after {timeout} s."],
                "warnings": [], "prmtop": None, "inpcrd": None}
    except Exception as e:
        return {"ok": False, "log": "", "errors": [f"Could not run tleap: {e}"],
                "warnings": [], "prmtop": None, "inpcrd": None}

    (work_dir / "leap.log").write_text(log)
    prmtop = next((p for p in work_dir.glob("*.prmtop") if p.stat().st_size), None)
    inpcrd = next((p for p in work_dir.glob("*.inpcrd") if p.stat().st_size), None)
    result = {
        "ok": bool(prmtop and inpcrd),
        "prmtop": str(prmtop) if prmtop else None,
        "inpcrd": str(inpcrd) if inpcrd else None,
        "log": log,
        "errors": [l.strip() for l in log.splitlines()
                   if "Error" in l or "FATAL" in l or "Fatal" in l],
        "warnings": [l.strip() for l in log.splitlines() if "Warning" in l],
        "dir": str(work_dir),
    }
    if prmtop:
        result.update(topology_summary(prmtop))
    return result


def topology_summary(prmtop_path) -> dict:
    """
    What a built topology contains, read straight out of the prmtop.

    Parsed rather than loaded through ParmEd, which is broken by NumPy 2 in
    current AmberTools builds -- the same reason prepare.py does its own PDB
    surgery. The format is simple enough that this is not a hardship: named
    sections, a Fortran format line, fixed-width fields.
    """
    try:
        text = Path(prmtop_path).read_text(errors="replace")
    except Exception:
        return {}

    def section(name):
        m = re.search(r"%FLAG " + name + r"\s*\n%FORMAT\((.*?)\)\s*\n(.*?)(?=%FLAG|\Z)",
                      text, re.S)
        if not m:
            return []
        width = int(re.search(r"[aIEF](\d+)", m.group(1)).group(1))
        return [line[i:i + width].strip()
                for line in m.group(2).splitlines()
                for i in range(0, len(line), width)
                if line[i:i + width].strip()]

    names = section("ATOM_NAME")
    labels = section("RESIDUE_LABEL")
    charges = section("CHARGE")
    pointers = section("POINTERS")
    bonds = [int(v) for v in section("BONDS_WITHOUT_HYDROGEN")]
    waters = sum(1 for r in labels if r in ("WAT", "HOH"))
    ions = {}
    for label in labels:
        if label in mem.ION_RESNAMES or label.endswith(("+", "-")):
            ions[label] = ions.get(label, 0) + 1
    # Amber stores charges scaled by 18.2223 so that energies come out in
    # kcal/mol; the net charge is only meaningful after dividing it back out.
    net = sum(float(c) for c in charges) / 18.2223 if charges else 0.0
    disulfides = sum(1 for i in range(0, len(bonds), 3)
                     if names[bonds[i] // 3] == "SG" and names[bonds[i + 1] // 3] == "SG")
    return {
        "atoms": len(names),
        "residues": len(labels),
        "waters": waters,
        "ions": ions,
        "charge": round(net, 3),
        "disulfides": disulfides,
        "box": bool(pointers and len(pointers) > 27 and pointers[27] != "0"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Amber MD inputs
# ═══════════════════════════════════════════════════════════════════════════════

# Deliberately written as readable files rather than built by a settings
# object: an mdin is the thing a reviewer asks to see, and it should be
# possible to read every number in it without consulting this source.

def amber_mdin(stage: str, membrane_system: bool = False, temperature: float = 300.0,
               pressure: float = 1.0, nanoseconds: float = 100.0,
               timestep: float = 0.002, restraint_mask: str = "@CA,C,N,O",
               restraint_weight: float = 10.0) -> str:
    """
    One stage of a standard Amber protocol.

    Stages: "min_solvent" (solute restrained, solvent relaxed), "min_all",
    "heat" (0 to T with restraints), "equil_npt" (restraints released),
    "production".
    """
    steps = int(nanoseconds * 1000 / timestep)
    ntp, extra = (2, "") if not membrane_system else (
        3, "  csurften = 3, gamma_ten = 0.0, ninterface = 2,\n")
    barostat = "  barostat = 2, mcbarint = 100,\n"

    if stage == "min_solvent":
        return f"""Minimise the solvent with the solute held in place
 &cntrl
  imin = 1, maxcyc = 5000, ncyc = 2500,
  ntb = 1, cut = 10.0,
  ntr = 1, restraintmask = '!:WAT,Na+,Cl-,K+ & !@H=',
  restraint_wt = {restraint_weight},
 /
"""
    if stage == "min_all":
        return """Minimise everything
 &cntrl
  imin = 1, maxcyc = 10000, ncyc = 5000,
  ntb = 1, cut = 10.0,
  ntr = 0,
 /
"""
    if stage == "heat":
        return f"""Heat from 0 to {temperature} K at constant volume, solute restrained
 &cntrl
  imin = 0, irest = 0, ntx = 1,
  nstlim = 50000, dt = {timestep},
  ntc = 2, ntf = 2,                      ! SHAKE on bonds to hydrogen
  ntb = 1, ntp = 0, cut = 10.0,
  ntt = 3, gamma_ln = 1.0,               ! Langevin thermostat
  tempi = 0.0, temp0 = {temperature},
  nmropt = 1,
  ntr = 1, restraintmask = '{restraint_mask}', restraint_wt = {restraint_weight},
  ntpr = 500, ntwx = 5000, ntwr = 5000,
  ig = -1,
 /
 &wt type = 'TEMP0', istep1 = 0, istep2 = 45000,
     value1 = 0.0, value2 = {temperature} /
 &wt type = 'END' /
"""
    if stage == "equil_npt":
        return f"""Equilibrate at constant pressure, releasing the restraints
 &cntrl
  imin = 0, irest = 1, ntx = 5,
  nstlim = 500000, dt = {timestep},
  ntc = 2, ntf = 2,
  ntb = 2, ntp = {ntp}, pres0 = {pressure}, taup = 2.0,
{barostat}{extra}  cut = 10.0,
  ntt = 3, gamma_ln = 1.0, temp0 = {temperature},
  ntr = 1, restraintmask = '{restraint_mask}', restraint_wt = 1.0,
  ntpr = 1000, ntwx = 10000, ntwr = 10000,
  ig = -1,
 /
"""
    return f"""Production, {nanoseconds} ns
 &cntrl
  imin = 0, irest = 1, ntx = 5,
  nstlim = {steps}, dt = {timestep},
  ntc = 2, ntf = 2,
  ntb = 2, ntp = {ntp}, pres0 = {pressure}, taup = 2.0,
{barostat}{extra}  cut = 10.0,
  ntt = 3, gamma_ln = 1.0, temp0 = {temperature},
  ntr = 0,
  ntpr = 5000, ntwx = 25000, ntwr = 50000,
  iwrap = 1, ig = -1,
 /
"""


AMBER_STAGES = [
    ("01_min_solvent.in", "min_solvent"),
    ("02_min_all.in", "min_all"),
    ("03_heat.in", "heat"),
    ("04_equil.in", "equil_npt"),
    ("05_prod.in", "production"),
]


def amber_run_script(prmtop: str, inpcrd: str, engine: str = "pmemd.cuda") -> str:
    """A shell script that runs the stages in order."""
    lines = ["#!/bin/bash", "# Generated by PARORA (simulation.py).",
             "# Check every input file before submitting this anywhere.",
             "set -euo pipefail", "",
             f"PRMTOP={prmtop}", f"ENGINE={engine}", "",
             f"$ENGINE -O -i 01_min_solvent.in -p $PRMTOP -c {inpcrd} "
             f"-r 01_min_solvent.rst7 -o 01_min_solvent.out -ref {inpcrd}",
             "$ENGINE -O -i 02_min_all.in -p $PRMTOP -c 01_min_solvent.rst7 "
             "-r 02_min_all.rst7 -o 02_min_all.out",
             "$ENGINE -O -i 03_heat.in -p $PRMTOP -c 02_min_all.rst7 "
             "-r 03_heat.rst7 -o 03_heat.out -x 03_heat.nc -ref 02_min_all.rst7",
             "$ENGINE -O -i 04_equil.in -p $PRMTOP -c 03_heat.rst7 "
             "-r 04_equil.rst7 -o 04_equil.out -x 04_equil.nc -ref 03_heat.rst7",
             "$ENGINE -O -i 05_prod.in -p $PRMTOP -c 04_equil.rst7 "
             "-r 05_prod.rst7 -o 05_prod.out -x 05_prod.nc",
             ""]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# GROMACS
# ═══════════════════════════════════════════════════════════════════════════════

GROMACS_CONVERSION_NOTE = """\
Getting an Amber topology into GROMACS needs a converter, and both of the
usual ones are Python libraries rather than command-line tools:

    ParmEd:   parmed -i <<< 'parm system.prmtop; loadRestrt system.inpcrd;
              outparm system.top system.gro'
    acpype:   acpype -p system.prmtop -x system.inpcrd

Neither is installed in the AmberTools environment this app found, and ParmEd
there is currently broken by NumPy 2 (it imports numpy.compat, which was
removed). Fix it with:

    conda run -n ambertools pip install -U parmed

The alternative is to skip Amber entirely and build the topology in GROMACS
itself with pdb2gmx, using the prepared PDB from the Prepare tab:

    gmx pdb2gmx -f prepared.pdb -o system.gro -water tip3p -ignh
    gmx editconf -f system.gro -o box.gro -c -d 1.2 -bt dodecahedron
    gmx solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top
    gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr
    gmx genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL \\
        -neutral -conc 0.15

pdb2gmx will not parameterise a ligand -- that still needs GAFF parameters
converted over, or a CHARMM-GUI / ATB-style service.
"""


def gromacs_mdp(stage: str, temperature: float = 300.0, pressure: float = 1.0,
                nanoseconds: float = 100.0, timestep: float = 0.002,
                membrane_system: bool = False) -> str:
    """One GROMACS run-parameter file: "em", "nvt", "npt" or "md"."""
    steps = int(nanoseconds * 1000 / timestep)
    coupling = "semiisotropic" if membrane_system else "isotropic"
    compress = "4.5e-5  4.5e-5" if membrane_system else "4.5e-5"
    ref_p = f"{pressure}     {pressure}" if membrane_system else f"{pressure}"

    if stage == "em":
        return """; Energy minimisation
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
"""
    common = f"""integrator      = md
dt              = {timestep}
nsteps          = {steps}
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
constraints     = h-bonds
constraint-algorithm = lincs
pbc             = xyz
tcoupl          = V-rescale
tc-grps         = Protein Non-Protein
tau-t           = 0.1     0.1
ref-t           = {temperature}     {temperature}
"""
    if stage == "nvt":
        return ("; NVT equilibration, position restraints on\n"
                "define          = -DPOSRES\n" + common +
                "pcoupl          = no\ngen-vel         = yes\n"
                f"gen-temp        = {temperature}\ngen-seed        = -1\n"
                "nstxout-compressed = 5000\n")
    if stage == "npt":
        return ("; NPT equilibration, position restraints on\n"
                "define          = -DPOSRES\n" + common +
                f"pcoupl          = C-rescale\npcoupltype      = {coupling}\n"
                f"tau-p           = 2.0\nref-p           = {ref_p}\n"
                f"compressibility = {compress}\n"
                "gen-vel         = no\ncontinuation    = yes\n"
                "nstxout-compressed = 5000\n")
    return ("; Production\n" + common +
            f"pcoupl          = C-rescale\npcoupltype      = {coupling}\n"
            f"tau-p           = 2.0\nref-p           = {ref_p}\n"
            f"compressibility = {compress}\n"
            "gen-vel         = no\ncontinuation    = yes\n"
            "nstxout-compressed = 25000\nnstenergy       = 5000\n")


GROMACS_STAGES = [("em.mdp", "em"), ("nvt.mdp", "nvt"),
                  ("npt.mdp", "npt"), ("md.mdp", "md")]


# ═══════════════════════════════════════════════════════════════════════════════
# Rosetta
# ═══════════════════════════════════════════════════════════════════════════════

def rosetta_params_command(code: str, atoms: int = 0) -> str:
    """The molfile_to_params.py command line for a ligand."""
    return (f"# Rosetta needs a .params file for {code}. From a mol2 or sdf with\n"
            f"# hydrogens and correct bond orders (the one antechamber wrote on the\n"
            f"# Amber tab will do):\n"
            f"#   $ROSETTA/main/source/scripts/python/public/molfile_to_params.py \\\n"
            f"#       -n {code} -p {code} --conformers-in-one-file {code}.mol2\n"
            f"# That writes {code}.params and {code}_0001.pdb; the PDB is the ligand\n"
            f"# with Rosetta's atom naming and is what you append to the receptor.\n")


def rosetta_ligand_docking_xml(code: str = "LIG", chain: str = "X") -> str:
    """
    A RosettaScripts protocol for ligand docking.

    The standard four-stage arrangement: a coarse low-resolution search over
    the box, then repeated rounds of high-resolution repacking and
    minimisation. Written out in full rather than as a preset because the
    numbers -- box size, move distances, cycles -- are what someone tuning a
    docking run actually changes.
    """
    return f"""<ROSETTASCRIPTS>
  <!-- Generated by PARORA (simulation.py). Ligand docking, RosettaLigand. -->
  <SCOREFXNS>
    <ScoreFunction name="ligand_soft_rep" weights="ligand_soft_rep">
      <Reweight scoretype="fa_elec" weight="0.42"/>
      <Reweight scoretype="hbond_bb_sc" weight="1.3"/>
      <Reweight scoretype="hbond_sc" weight="1.3"/>
      <Reweight scoretype="rama" weight="0.2"/>
    </ScoreFunction>
    <ScoreFunction name="hard_rep" weights="ligand">
      <Reweight scoretype="fa_intra_rep" weight="0.004"/>
      <Reweight scoretype="fa_elec" weight="0.42"/>
      <Reweight scoretype="hbond_bb_sc" weight="1.3"/>
      <Reweight scoretype="hbond_sc" weight="1.3"/>
      <Reweight scoretype="rama" weight="0.2"/>
    </ScoreFunction>
  </SCOREFXNS>

  <LIGAND_AREAS>
    <LigandArea name="docking_sidechain" chain="{chain}" cutoff="6.0"
                add_nbr_radius="true" all_atom_mode="true" minimize_ligand="10"/>
    <LigandArea name="final_sidechain" chain="{chain}" cutoff="6.0"
                add_nbr_radius="true" all_atom_mode="true"/>
    <LigandArea name="final_backbone" chain="{chain}" cutoff="7.0"
                add_nbr_radius="false" all_atom_mode="true" Calpha_restraints="0.3"/>
  </LIGAND_AREAS>

  <INTERFACE_BUILDERS>
    <InterfaceBuilder name="side_chain_for_docking" ligand_areas="docking_sidechain"/>
    <InterfaceBuilder name="side_chain_for_final" ligand_areas="final_sidechain"/>
    <InterfaceBuilder name="backbone" ligand_areas="final_backbone"
                      extension_window="3"/>
  </INTERFACE_BUILDERS>

  <MOVEMAP_BUILDERS>
    <MoveMapBuilder name="docking" sc_interface="side_chain_for_docking"
                    minimize_water="true"/>
    <MoveMapBuilder name="final" sc_interface="side_chain_for_final"
                    bb_interface="backbone" minimize_water="true"/>
  </MOVEMAP_BUILDERS>

  <MOVERS>
    <!-- Start from the ligand where it is; widen the box to search further. -->
    <Transform name="transform" chain="{chain}" box_size="5.0" move_distance="0.1"
               angle="5" cycles="500" repeats="1" temperature="5"/>
    <HighResDocker name="high_res_docker" cycles="6" repack_every_Nth="3"
                   scorefxn="ligand_soft_rep" movemap_builder="docking"/>
    <FinalMinimizer name="final" scorefxn="hard_rep" movemap_builder="final"/>
    <InterfaceScoreCalculator name="add_scores" chains="{chain}" scorefxn="hard_rep"/>
    <ParsedProtocol name="low_res_dock">
      <Add mover_name="transform"/>
    </ParsedProtocol>
    <ParsedProtocol name="high_res_dock">
      <Add mover_name="high_res_docker"/>
      <Add mover_name="final"/>
    </ParsedProtocol>
  </MOVERS>

  <PROTOCOLS>
    <Add mover_name="low_res_dock"/>
    <Add mover_name="high_res_dock"/>
    <Add mover_name="add_scores"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""


def rosetta_options(pdb_name: str, code: str = "LIG", nstruct: int = 100) -> str:
    """The flags file for a ligand docking run."""
    return f"""# Generated by PARORA (simulation.py).
-in:file:s {pdb_name}
-in:file:extra_res_fa {code}.params
-parser:protocol dock.xml

-nstruct {nstruct}
-packing:ex1
-packing:ex2
-packing:no_optH false
-packing:flip_HNQ true
-packing:ignore_ligand_chi true

-out:level 300
-out:file:scorefile dock_score.sc
-out:path:pdb output/
-overwrite

# Ligand docking scores are only comparable within one run. Rank by
# interface_delta_X, not by total_score, and look at the spread rather than
# the single best number.
"""


ROSETTA_README = """\
Running this
------------
1. Make the ligand parameters (see params_command.txt). Rosetta cannot read a
   ligand it has no .params file for, and the .params file must describe the
   same protonation state as the PDB you give it.
2. Append the Rosetta-named ligand PDB to the receptor:
       cat receptor.pdb LIG_0001.pdb > complex.pdb
3. Run:
       $ROSETTA/main/source/bin/rosetta_scripts.default.linuxgccrelease \\
           @dock.options
4. Rank the output by interface_delta_X in dock_score.sc, not by total_score.

Notes
-----
* The receptor here came out of the Prepare tab with the Rosetta profile:
  one model, no alternate conformations, no waters, hydrogens stripped so
  Rosetta rebuilds its own.
* Transform's box_size is how far the ligand may wander from where it starts.
  Starting from a crystallographic pose with a small box is redocking; a real
  docking run into an apo site wants a larger box and many more structures.
* None of this was run or checked here -- Rosetta is not installed in this
  environment, so these files are generated, not tested.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Bundling
# ═══════════════════════════════════════════════════════════════════════════════

def bundle(files: dict) -> bytes:
    """
    Zip a set of files into one downloadable archive.

    A simulation setup is a directory, not a file: topology, coordinates, five
    run inputs and a script that ties them together. Handing them over one
    download button at a time is how one of them gets missed.

    Values are either content or a file to copy in, and which one is never
    inferred: a `Path` is read from disk, a `str` or `bytes` is written as the
    content. Guessing by asking whether the string happens to name an existing
    file is the kind of shortcut that works until an mdin file arrives, at
    which point the filesystem is asked whether a 600-character namelist
    exists and answers with OSError.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            if content is None:
                continue
            if isinstance(content, Path):
                if content.exists():
                    archive.write(content, name)
            elif isinstance(content, (bytes, bytearray)):
                archive.writestr(name, content)
            else:
                archive.writestr(name, str(content))
    return buffer.getvalue()


# Formal charges of the ionisable residues at neutral pH, and of the ions that
# turn up in structures. The ions matter more than they look: trypsin's single
# structural calcium is +2, and leaving it out of the charge estimate is two
# counter-ions' worth of error before the topology is even built.
RESIDUE_CHARGES = {
    "ARG": 1, "LYS": 1, "HIP": 1, "ASP": -1, "GLU": -1, "CYM": -1, "TYM": -1,
    "ASH": 0, "GLH": 0, "HID": 0, "HIE": 0, "HIS": 0,
}

ION_CHARGES = {
    "NA": 1, "NA+": 1, "K": 1, "K+": 1, "LI": 1, "RB": 1, "CS": 1, "AG": 1,
    "CA": 2, "CA2+": 2, "MG": 2, "MG2+": 2, "ZN": 2, "ZN2+": 2, "MN": 2,
    "FE2": 2, "CU": 2, "CU1": 1, "NI": 2, "CO": 2, "CD": 2, "SR": 2, "BA": 2,
    "FE": 3, "AL": 3, "FE3": 3,
    "CL": -1, "CL-": -1, "BR": -1, "IOD": -1, "F": -1,
}


def residue_charge_table() -> dict:
    """Formal charges of the ionisable residues at neutral pH, for charge sums."""
    return dict(RESIDUE_CHARGES)


def estimate_charge(pdb_path) -> int:
    """
    Net formal charge of a structure at neutral pH, counted from its residues.

    Only a sanity check against what tleap reports: it assumes standard
    protonation, knows nothing about any ligand, and will be wrong for a
    structure with an unusual histidine or a metal site. Useful before the
    topology exists, which is when the ion counts have to be decided.
    """
    seen, total = set(), 0
    for line in prp.read_lines(pdb_path):
        if not prp._is_coord(line):
            continue
        key = prp._reskey(line)
        if key in seen:
            continue
        seen.add(key)
        name = prp._resname(line)
        total += RESIDUE_CHARGES.get(name, ION_CHARGES.get(name, 0))
    return total
