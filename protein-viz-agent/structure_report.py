# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-11
# Summary   : Plain-language composition report for a PDB structure such as chains,
#             residue counts, modified residues, ligands, cofactors, ions and
#             crystallization additives.
#
#             Everything here is read out of the PDB file itself, by column,
#             with no MDAnalysis, Biopython or network call. The deposited file
#             already carries the answers: MODRES names every modified residue
#             and the standard residue it stands for, HETNAM gives each
#             chemical component's full name, SEQRES gives the construct the
#             crystallographer put in the drop, and the ATOM records give what
#             was actually observed. Reading them directly means the report
#             works in any environment and always describes the file on disk.
#
#             The one judgement call is sorting hetero components into
#             cofactors, ions, crystallization additives and everything else.
#             That is a curated lookup, not something the file states, and it
#             exists because the single most common misreading of a structure
#             is treating a glycerol or a sulfate from the cryoprotectant as a
#             biologically meaningful ligand. Anything unrecognised is reported
#             as a plain ligand rather than guessed at.
# =============================================================================

import json
import math
import re

import sequence_utils as squ

# Bumped whenever summarize() changes the shape of what it returns. Callers
# that cache a summary must feed this into their cache key: Streamlit's
# st.cache_data keys on the decorated function's own code, so a wrapper that
# merely calls summarize() keeps serving dicts built by an older version of it
# long after this module has been edited, and the new formatters then fail on
# keys the old dict never had.
SCHEMA_VERSION = 2

# What "standard" means here: the 20 canonical amino acids, and nothing else.
# Everything else in the coordinates -- a modified amino acid, a nucleotide, a
# ligand, a cofactor, an ion, a buffer component -- is non-standard. That is
# broader than the crystallographic convention, where "non-standard residue"
# usually means only a MODRES-declared modified amino acid, and it is the
# reading this project wants: if it is not an amino acid, it is not standard.
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

# Unmodified DNA and RNA bases. Not amino acids, so still non-standard, but
# worth their own line rather than being lumped in with ligands.
STANDARD_NT = {"DA", "DC", "DG", "DT", "DU", "DI", "A", "C", "G", "U", "I", "T"}

# Labels for the non-standard breakdown, in the order they are reported.
NONSTANDARD_KINDS = [
    ("modified_aa", "Modified amino acid"),
    ("nucleotide",  "Nucleotide"),
    ("ligand",      "Ligand"),
    ("cofactor",    "Cofactor / coenzyme"),
    ("ion",         "Ion"),
    ("additive",    "Crystallization additive"),
]

# A peptide bond is about 1.33 Å. Anything beyond this between one residue's
# carbonyl carbon and the next residue's amide nitrogen is a genuine break in
# the modelled chain, whatever the residue numbering says.
PEPTIDE_BOND_MAX = 2.0

# ── Chemical component classification ────────────────────────────────────────
# Curated, not exhaustive. Codes are PDB Chemical Component Dictionary ids.

# Cofactors, coenzymes, nucleotides and prosthetic groups -- the hetero groups
# that are usually part of the biology rather than the crystallography.
COFACTORS = {
    "HEM", "HEC", "HEA", "HEB", "SRM", "DHE",              # hemes
    "NAD", "NAI", "NAP", "NDP", "NAX",                     # NAD(P)(H)
    "FAD", "FMN", "FDA", "RBF",                            # flavins
    "ATP", "ADP", "AMP", "ANP", "ACP", "AGS", "APC",       # adenine nucleotides
    "GTP", "GDP", "GMP", "GNP", "GSP", "G4P",              # guanine nucleotides
    "UTP", "UDP", "UMP", "CTP", "CDP", "CMP", "TTP", "TMP",
    "COA", "ACO", "MLC", "SCA", "CAA",                     # coenzyme A and thioesters
    "SAM", "SAH", "MTA",                                   # methyl donors
    "PLP", "PMP", "P5P",                                   # vitamin B6
    "TPP", "TDP", "THD",                                   # thiamine
    "BTN", "B12", "COB", "CNC", "B1Z",                     # biotin, cobalamin
    "MGD", "MOO", "MSS", "PCD",                            # molybdopterin
    "F43", "SF4", "FES", "FS4", "F3S", "CLF", "ICS",       # iron-sulfur clusters
    "PQQ", "TPQ", "LPA", "H4B", "BH4", "GSH", "GDS",
    "UQ1", "UQ2", "PLQ", "HQE",                            # quinones
    "CLA", "CHL", "BCL", "PHO",                            # chlorophylls
    "RET", "LUM", "FOL", "THF", "MTX",
}

# Monatomic ions and simple inorganic species.
IONS = {
    "ZN", "MG", "CA", "MN", "FE", "FE2", "CU", "CU1", "CO", "3CO", "NI",
    "CD", "HG", "PT", "AU", "AG", "PB", "BA", "SR", "CS", "RB", "LI",
    "NA", "K", "CL", "BR", "IOD", "F", "SE", "MO", "W", "V", "CR",
    "NH4", "OH",
    # Deliberately not here: OXY (dioxygen), CMO, NO. They are bound small
    # molecules -- the substrate, in oxymyoglobin -- and belong under ligands.
}

# Buffers, cryoprotectants, precipitants and detergents. Present because of how
# the crystal was grown or frozen, not because of what the protein does.
ADDITIVES = {
    "GOL", "EDO", "PGE", "PG4", "PG0", "1PE", "2PE", "P6G", "PEG", "7PE",
    "MPD", "MRD", "BU3", "IPA", "EOH", "MOH", "DMS", "DMF", "ACN",
    "SO4", "PO4", "2PO", "NO3", "AZI", "CAC", "SCN", "BCT", "CO3",
    "ACT", "ACY", "FMT", "CIT", "FLC", "TLA", "MLA", "MLI", "SIN", "OXL",
    "TRS", "EPE", "MES", "BIS", "TAM", "CHES", "PIN", "HEPES",
    "IMD", "BME", "DTT", "DTU", "MRC", "TCE",
    "BOG", "LDA", "C8E", "OCT", "LMT", "DDQ", "UND", "HEX", "PEE",
    "UNX", "UNL", "UNK",
}

CATEGORY_LABELS = {
    "cofactor": "Cofactor / coenzyme",
    "ion":      "Ion",
    "ligand":   "Ligand",
    "additive": "Crystallization additive",
}

CATEGORY_PLURALS = {
    "cofactor": "COFACTORS / COENZYMES",
    "ion":      "IONS",
    "ligand":   "LIGANDS",
    "additive": "CRYSTALLIZATION ADDITIVES",
}

# Order the component table is grouped in — most biologically interesting first.
CATEGORY_ORDER = ["ligand", "cofactor", "ion", "additive"]


def classify_component(code: str) -> str:
    """
    Sort a hetero component into cofactor, ion, additive or plain ligand.

    Anything not in the curated lists is called a ligand, which is the honest
    default: an unrecognised code is far more likely to be the compound the
    structure was solved with than a buffer nobody has heard of.
    """
    c = code.strip().upper()
    if c in COFACTORS:
        return "cofactor"
    if c in IONS:
        return "ion"
    if c in ADDITIVES:
        return "additive"
    return "ligand"


# ── PDB record parsing ───────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Collapse the runs of padding spaces PDB text fields are padded with."""
    return " ".join(text.split())


def _join_continuation(head: str, tail: str) -> str:
    """
    Append a wrapped HETNAM continuation to the name so far.

    A long chemical name is split across records at whatever column runs out,
    which may be mid-token. Joining with a space would put one inside
    "(trifluoromethyl)phenoxy"; joining without one would weld "calcium" to
    "ion". The break is mid-token whenever the name so far ends on a bond,
    bracket or punctuation character, or the continuation opens with one.
    """
    head, tail = head.rstrip(), tail.strip()
    if not head:
        return tail
    if not tail:
        return head
    if head[-1] in "-,([{)]}" or tail[0] in "-,)]},":
        return head + tail
    return head + " " + tail


def pretty_chemical(name: str) -> str:
    """
    Make a deposited ALL-CAPS chemical name readable without corrupting it.

    Lower-casing is right for chemical nomenclature ("benzamidine", not
    "Benzamidine"), except for the indicated-hydrogen locants written as a
    digit followed by a capital letter -- 5H-pyrrolo, 1H-indole -- which are
    restored. Title-casing, the obvious alternative, breaks every one of these
    and every embedded element symbol.
    """
    if not name:
        return ""
    lowered = name.lower()
    lowered = re.sub(r"(?<=\d)([a-z])(?=[-\]])", lambda m: m.group(1).upper(), lowered)
    # Standalone Roman numerals ("protoporphyrin IX") and element symbols
    # ("containing FE") are the two things lowercasing gets visibly wrong.
    # Both are matched only as whole words, and only for spellings that are not
    # also English words, so nothing else is touched.
    lowered = re.sub(r"\b(i{2,3}|iv|vi{0,3}|ix|xi{0,2})\b",
                     lambda m: m.group(1).upper(), lowered)
    return re.sub(r"\b(fe|zn|mg|mn|cu|ni|cd|hg|pt|au|ag|pb|se|mo|cl|br)\b",
                  lambda m: m.group(1).capitalize(), lowered)


# Acronyms that title-casing would ruin in an EXPDTA string.
_METHOD_ACRONYMS = ("NMR", "EM", "ESR", "EPR", "FRET", "SAXS", "SANS")


def pretty_method(name: str) -> str:
    """Title-case an experimental method while keeping its acronyms upper case."""
    pretty = name.title()
    for acronym in _METHOD_ACRONYMS:
        pretty = re.sub(rf"\b{acronym.title()}\b", acronym, pretty)
    return pretty


def pretty_organism(name: str) -> str:
    """Render a deposited organism name as a binomial: BOS TAURUS -> Bos taurus."""
    return name.capitalize() if name.isupper() else name


def _plural(n: int, word: str, plural: str = None) -> str:
    """'1 chain' / '2 chains', so the report reads like prose."""
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def _parse_header(path):
    """
    Read the descriptive records: title, method, resolution, organism.

    Every field is optional. Predicted models and trimmed files routinely carry
    none of them, so each one falls back to None rather than raising.
    """
    info = {"id": None, "title": "", "classification": None, "deposited": None,
            "method": None, "resolution": None, "organisms": []}
    title_parts, hetnam, seqres, modres = [], {}, {}, []

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            rec = line[:6]

            if rec == "HEADER":
                info["classification"] = _clean(line[10:50]) or None
                info["deposited"] = _clean(line[50:59]) or None
                info["id"] = _clean(line[62:66]) or None

            elif rec == "TITLE ":
                title_parts.append(line[10:80].rstrip())

            elif rec == "EXPDTA":
                info["method"] = _clean(line[10:79]) or None

            elif rec == "REMARK" and line[6:10].strip() == "2" and "RESOLUTION" in line:
                # Read the number after "RESOLUTION.", not the first float on
                # the line -- that one is the remark number itself, which made
                # every structure come back as 2 Å.
                m = re.search(r"RESOLUTION\.?\s+([0-9]*\.?[0-9]+)", line)
                if m:
                    info["resolution"] = float(m.group(1))

            elif rec == "SOURCE":
                text = line[10:79]
                if "ORGANISM_SCIENTIFIC:" in text:
                    name = text.split("ORGANISM_SCIENTIFIC:", 1)[1].strip().rstrip(";")
                    if name and name not in info["organisms"]:
                        info["organisms"].append(name)

            elif rec == "HETNAM":
                code = line[11:14].strip()
                hetnam[code] = _join_continuation(hetnam.get(code, ""), line[15:70])

            elif rec == "SEQRES":
                chain = line[11:12].strip() or "_"
                seqres.setdefault(chain, []).extend(line[19:70].split())

            elif rec == "MODRES":
                modres.append({
                    "code": line[12:15].strip(),
                    "chain": line[16:17].strip() or "_",
                    "parent": line[24:27].strip(),
                    "note": _clean(line[29:70]),
                })

    info["title"] = _clean(" ".join(title_parts))
    return info, {k: _clean(v) for k, v in hetnam.items()}, seqres, modres


def _parse_contents(path):
    """
    Walk the coordinate records once, counting residues and hetero components.

    Residues are de-duplicated by (chain, resseq, icode) so that a residue's
    many atoms, and its alternate conformations, count once.
    """
    chains = {}          # chain -> list of residue dicts, in file order
    het = {}             # component code -> {"count", "chains"}
    index = {}           # residue key -> its dict, for attaching backbone atoms
    waters = 0
    atoms = 0
    models = 0

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("MODEL "):
                # An NMR ensemble repeats the whole structure once per model.
                # Only the first is counted, but the records keep being read so
                # the report can say how large the ensemble actually is.
                models += 1
                continue
            if models > 1 or not line.startswith(("ATOM  ", "HETATM")):
                continue

            atoms += 1
            resname = line[17:20].strip()
            chain = line[21].strip() or "_"
            raw_seq = line[22:26].strip()
            icode = line[26].strip()
            if not raw_seq:
                continue
            try:
                resseq = int(raw_seq)
            except ValueError:
                continue

            key = (chain, resseq, icode, resname)
            first_time = key not in index

            kind = squ.classify_residue(resname)
            if kind == "water":
                if first_time:
                    index[key] = None
                    waters += 1
                continue
            if kind == "hetero":
                if first_time:
                    index[key] = None
                    slot = het.setdefault(resname, {"count": 0, "chains": []})
                    slot["count"] += 1
                    if chain not in slot["chains"]:
                        slot["chains"].append(chain)
                continue

            if first_time:
                residue = {"resseq": resseq, "icode": icode,
                           "resname": resname, "kind": kind, "N": None, "C": None}
                index[key] = residue
                chains.setdefault(chain, []).append(residue)
            residue = index[key]

            # Keep the backbone atoms needed to tell a real chain break from a
            # jump in the numbering. The first altloc encountered wins.
            name = line[12:16].strip()
            if name in ("N", "C") and residue[name] is None:
                try:
                    residue[name] = (float(line[30:38]), float(line[38:46]),
                                     float(line[46:54]))
                except ValueError:
                    pass

    return chains, het, waters, atoms, models


def _chain_rows(chains, seqres, modified_codes):
    """
    One row per polymer chain: type, how much was observed, and where it breaks.

    "Missing" compares SEQRES -- the construct that went into the drop -- with
    what actually has coordinates. Disordered loops and termini routinely fail
    to appear in the density, and a residue count that ignores that overstates
    what the structure really shows.

    Breaks are measured, not inferred from the numbering. Several protein
    families are deposited in a reference numbering that legitimately skips
    numbers -- the chymotrypsin numbering used by the serine proteases makes
    trypsin look like it has six breaks when the chain is continuous -- so a
    break is only counted where the peptide bond itself is missing.
    """
    rows = []
    for chain, residues in chains.items():
        if not residues:
            continue
        kinds = [r["kind"] for r in residues]
        ctype = "protein" if kinds.count("protein") >= kinds.count("nucleic") else "nucleic"
        numbers = [r["resseq"] for r in residues]

        gaps = []
        for prev, nxt in zip(residues, residues[1:]):
            if prev["C"] is None or nxt["N"] is None:
                continue                       # nucleic acid, or no backbone atoms
            d = math.dist(prev["C"], nxt["N"])
            if d > PEPTIDE_BOND_MAX:
                gaps.append((prev["resseq"], nxt["resseq"],
                             max(nxt["resseq"] - prev["resseq"] - 1, 0)))

        expected = len(seqres.get(chain, []))
        rows.append({
            "chain": chain,
            "type": ctype,
            "observed": len(residues),
            "expected": expected or None,
            "missing": max(expected - len(residues), 0) if expected else None,
            "first": min(numbers),
            "last": max(numbers),
            "modified": sum(1 for r in residues if r["resname"] in modified_codes),
            "gaps": gaps,
            "sequence": "".join(squ.one_letter(r["resname"]) for r in residues),
        })
    return rows


def summarize(pdb_path) -> dict:
    """
    Describe what a PDB file contains.

    Args:
        pdb_path: Path to a .pdb file.

    Returns:
        dict with keys:
          id, title, classification, method, resolution, deposited, organisms,
          models     : number of models (>1 means an NMR ensemble; only the
                       first is counted)
          chains     : per-chain rows — type, observed/expected residues,
                       missing count, numbering range, chain breaks, sequence
          totals     : residue and atom counts across the structure
          modified   : modified residues, each with the standard residue it
                       replaces
          components : hetero components, each with its full chemical name and
                       category (ligand / cofactor / ion / additive)
          nonstandard: every residue that is not one of the 20 standard amino
                       acids -- modified amino acids, nucleotides, ligands,
                       cofactors, ions and additives -- each tagged with which
                       of those it is. Water is excluded.
          waters     : number of water molecules
    """
    info, hetnam, seqres, modres = _parse_header(pdb_path)
    chains, het, waters, atoms, models = _parse_contents(pdb_path)

    # MODRES is the authoritative list of modified residues. Fall back to the
    # residue vocabulary for files that omit it: a bare MSE with no MODRES
    # record is still a selenomethionine.
    modified = {}
    for m in modres:
        slot = modified.setdefault(m["code"], {
            "code": m["code"], "parent": m["parent"],
            "name": hetnam.get(m["code"]) or m["note"] or "",
            "count": 0, "chains": [],
        })
        slot["count"] += 1
        if m["chain"] not in slot["chains"]:
            slot["chains"].append(m["chain"])

    modified_codes = set(modified)
    for residues in chains.values():
        for r in residues:
            code = r["resname"]
            if code in squ.AA3_TO_1 and code not in STANDARD_AA:
                if code not in modified:
                    modified[code] = {
                        "code": code, "parent": squ.AA3_TO_1.get(code, ""),
                        "name": hetnam.get(code, ""), "count": 0, "chains": [],
                    }
                modified_codes.add(code)

    # Count occurrences in the coordinates, which MODRES does not always match.
    for slot in modified.values():
        slot["count"] = 0
        slot["chains"] = []
    for chain, residues in chains.items():
        for r in residues:
            if r["resname"] in modified:
                slot = modified[r["resname"]]
                slot["count"] += 1
                if chain not in slot["chains"]:
                    slot["chains"].append(chain)

    components = []
    for code, data in het.items():
        components.append({
            "code": code,
            "name": hetnam.get(code, ""),
            "count": data["count"],
            "chains": data["chains"],
            "category": classify_component(code),
        })
    components.sort(key=lambda c: (CATEGORY_ORDER.index(c["category"]),
                                   -c["count"], c["code"]))

    chain_rows = _chain_rows(chains, seqres, modified_codes)
    total_polymer = sum(r["observed"] for r in chain_rows)

    # ── Standard vs non-standard ─────────────────────────────────────────────
    # Standard means one of the 20 amino acids. Everything else that occupies a
    # residue slot is non-standard, grouped so the headline number can always
    # be broken back down into what it is made of.
    standard_aa = 0
    nucleotides = {}
    for residues in chains.values():
        for r in residues:
            code = r["resname"]
            if code in STANDARD_AA:
                standard_aa += 1
            elif r["kind"] == "nucleic":
                slot = nucleotides.setdefault(code, {
                    "code": code, "count": 0, "chains": [], "kind": "nucleotide",
                    "name": "",
                    "parent": "",
                })
                slot["count"] += 1

    for chain, residues in chains.items():
        for r in residues:
            if r["resname"] in nucleotides and chain not in nucleotides[r["resname"]]["chains"]:
                nucleotides[r["resname"]]["chains"].append(chain)

    nonstandard = []
    for m in sorted(modified.values(), key=lambda x: -x["count"]):
        nonstandard.append({**m, "kind": "modified_aa"})
    nonstandard += sorted(nucleotides.values(), key=lambda x: -x["count"])
    for c in components:
        nonstandard.append({
            "code": c["code"], "count": c["count"], "chains": c["chains"],
            "name": c["name"], "parent": "", "kind": c["category"],
        })

    order = [k for k, _ in NONSTANDARD_KINDS]
    nonstandard.sort(key=lambda x: (order.index(x["kind"]), -x["count"], x["code"]))
    total_nonstandard = sum(x["count"] for x in nonstandard)

    return {
        **info,
        "models": models,
        "chains": chain_rows,
        "modified": sorted(modified.values(), key=lambda m: -m["count"]),
        "components": components,
        # Every non-amino-acid residue in one list — modified amino acids,
        # nucleotides and hetero components alike. Water is deliberately not in
        # here; hundreds of solvent molecules would bury everything else.
        "nonstandard": nonstandard,
        "waters": waters,
        "totals": {
            "chains": len(chain_rows),
            "residues": total_polymer + sum(c["count"] for c in components),
            "polymer": total_polymer,
            "standard": standard_aa,
            "nonstandard": total_nonstandard,
            "modified_aa": sum(x["count"] for x in nonstandard if x["kind"] == "modified_aa"),
            "nucleotides": sum(x["count"] for x in nonstandard if x["kind"] == "nucleotide"),
            "protein": sum(r["observed"] for r in chain_rows if r["type"] == "protein"),
            "nucleic": sum(r["observed"] for r in chain_rows if r["type"] == "nucleic"),
            "missing": sum(r["missing"] or 0 for r in chain_rows),
            # Counts below are copies of each component, not distinct codes.
            "ligands": sum(c["count"] for c in components if c["category"] == "ligand"),
            "cofactors": sum(c["count"] for c in components if c["category"] == "cofactor"),
            "ions": sum(c["count"] for c in components if c["category"] == "ion"),
            "additives": sum(c["count"] for c in components if c["category"] == "additive"),
            "waters": waters,
            "atoms": atoms,
        },
    }


# ── Chain identities ──────────────────────────────────────────────────────


def chain_molecules(pdb_path) -> list:
    """
    Which molecule each chain of a PDB file is, and which residues it observes.

    A complex names every chain's molecule in COMPND (MOL_ID blocks) and its
    UniProt accession in DBREF. Without this, "which chain is HER2" can only be
    guessed — and chain A is often not the protein that was searched for (in
    7MN5, chain A is HER3 and HER2 is chain B). The observed range comes from
    the ATOM records, not DBREF: DBREF describes the construct, which can span
    far more than the density resolved.

    Returns:
        One dict per polymer chain, in file order: chain, molecule (readable
        COMPND name, or ""), uniprot (accessions from DBREF), first/last
        observed residue number, residues (observed count).
    """
    names, uniprot, observed, order = {}, {}, {}, []
    mol_name, mol_chains, cont = {}, {}, ""
    try:
        fh = open(pdb_path, "r", errors="replace")
    except OSError:
        return []
    with fh:
        for line in fh:
            rec = line[:6]
            if rec == "COMPND":
                text = line[10:80].rstrip()
                # A field can wrap onto the next COMPND line with no key.
                if cont and ":" not in text.split(";")[0]:
                    text = cont + " " + text.strip()
                cont = "" if text.rstrip().endswith(";") else text
                m = re.match(r"\s*(MOL_ID|MOLECULE|CHAIN):\s*(.*?);?\s*$", text)
                if not m:
                    continue
                key, val = m.group(1), m.group(2).strip()
                if key == "MOL_ID":
                    mol = val
                elif key == "MOLECULE":
                    mol_name[mol] = val.rstrip(",")
                else:
                    mol_chains[mol] = [c.strip() for c in val.split(",") if c.strip()]
            elif rec == "DBREF ":
                chain = line[12:13].strip() or "_"
                if line[26:32].strip() == "UNP":
                    acc = line[33:41].strip()
                    if acc and acc not in uniprot.setdefault(chain, []):
                        uniprot[chain].append(acc)
            elif rec == "ATOM  ":
                chain = line[21:22].strip() or "_"
                try:
                    num = int(line[22:26])
                except ValueError:
                    continue
                if chain not in observed:
                    observed[chain] = {"first": num, "last": num, "seen": set()}
                    order.append(chain)
                o = observed[chain]
                o["first"], o["last"] = min(o["first"], num), max(o["last"], num)
                o["seen"].add((num, line[26:27]))
            elif rec == "ENDMDL":
                break               # the first model is enough to name chains
    for mol, chains in mol_chains.items():
        for c in chains:
            names[c] = mol_name.get(mol, "")
    return [{"chain": c,
             "molecule": pretty_chemical(names[c]) if names.get(c) else "",
             "uniprot": uniprot.get(c, []),
             "first": observed[c]["first"], "last": observed[c]["last"],
             "residues": len(observed[c]["seen"])} for c in order]


# ── Output formats ────────────────────────────────────────────────────────


KIND_LABELS = dict(NONSTANDARD_KINDS)

# Spelled out wherever a standard/non-standard count is reported, because the
# split is a convention and a bare number invites the wrong reading.
CONVENTION = ("standard = the 20 amino acids; every other residue — modified "
              "amino acids, nucleotides, ligands, cofactors, ions, buffer "
              "components — counts as non-standard. Water is counted separately.")


def _describe(entry: dict) -> str:
    """One line for a non-standard residue: code, count, what it is, its name."""
    line = f"{entry['code']} ×{entry['count']} — {KIND_LABELS[entry['kind']].lower()}"
    if entry["kind"] == "modified_aa" and entry.get("parent"):
        line += f" replacing {entry['parent']}"
    if entry.get("name"):
        line += f" — {pretty_chemical(entry['name'])}"
    if entry.get("chains"):
        line += f" (chain {', '.join(entry['chains'])})"
    return line


def as_brief(summary: dict) -> str:
    """
    Four lines: what it is, how big, and what the non-standard residues are.

    This is the default the agent answers with. A composition question is
    usually a quick orientation check, and the full report buries the two or
    three numbers being asked for under everything else.
    """
    t = summary["totals"]
    out = []

    head = summary.get("id") or "Structure"
    if summary.get("classification"):
        head += f" — {summary['classification'].lower()}"
    out.append(head)

    meta = []
    if summary.get("method"):
        meta.append(pretty_method(summary["method"]))
    if summary.get("resolution"):
        meta.append(f"{summary['resolution']} Å")
    if summary.get("organisms"):
        meta.append("; ".join(pretty_organism(o) for o in summary["organisms"]))
    if meta:
        out.append(" · ".join(meta))

    out.append(f"{_plural(t['chains'], 'chain')} · "
               f"{t['standard']} standard amino acids · "
               f"{t['nonstandard']} non-standard · "
               f"{t['waters']} waters")

    if summary["nonstandard"]:
        shown = summary["nonstandard"][:6]
        bits = []
        for e in shown:
            label = f"{e['code']} ×{e['count']} ({KIND_LABELS[e['kind']].lower()}"
            if e.get("name"):
                label += f", {pretty_chemical(e['name'])}"
            bits.append(label + ")")
        line = "Non-standard: " + "; ".join(bits)
        if len(summary["nonstandard"]) > len(shown):
            line += f"; and {len(summary['nonstandard']) - len(shown)} more"
        out.append(line)
    else:
        out.append("Non-standard: none — only standard amino acids and water")

    return "\n".join(out)


def as_text(summary: dict) -> str:
    """
    The full report as plain text.

    One table of non-standard residues with a category column, rather than a
    section per category: the question being answered is "what is standard and
    what is not", and splitting the answer across four headings makes it harder
    to read off, not easier.
    """
    t = summary["totals"]
    out = []

    head = summary.get("id") or "Structure"
    if summary.get("title"):
        # Deposited titles are stored upper case; left as they are rather than
        # title-cased, which would turn HER2 into Her2 and EGFR into Egfr.
        head += f" — {summary['title']}"
    out.append(head)

    meta = []
    if summary.get("classification"):
        meta.append(summary["classification"].lower())
    if summary.get("method"):
        meta.append(pretty_method(summary["method"]))
    if summary.get("resolution"):
        meta.append(f"{summary['resolution']} Å")
    if summary.get("models", 0) > 1:
        meta.append(f"{summary['models']} models in the ensemble (the first is counted)")
    if summary.get("organisms"):
        meta.append("; ".join(pretty_organism(o) for o in summary["organisms"]))
    if meta:
        out.append("  " + " · ".join(meta))

    out.append("")
    out.append("COUNTS")
    out.append(f"  Standard amino acids   {t['standard']}")
    out.append(f"  Non-standard residues  {t['nonstandard']}")
    for key, label in NONSTANDARD_KINDS:
        n = sum(e["count"] for e in summary["nonstandard"] if e["kind"] == key)
        if n:
            out.append(f"    {label:<26} {n}")
    out.append(f"  Water molecules        {t['waters']}")
    out.append(f"  Atoms                  {t['atoms']}")
    if t["missing"]:
        out.append(f"  Missing from the model {t['missing']}   "
                   f"(in the deposited construct but not resolved)")
    out.append(f"  [{CONVENTION}]")

    out.append("")
    out.append(f"CHAINS ({t['chains']})")
    for c in summary["chains"]:
        line = (f"  Chain {c['chain']}: {c['observed']} {c['type']} residues "
                f"numbered {c['first']}–{c['last']}")
        if c["missing"]:
            line += f", {c['missing']} not modelled"
        if c["gaps"]:
            spans = ", ".join(f"{a}→{b}" for a, b, _ in c["gaps"])
            line += f", {_plural(len(c['gaps']), 'break')} in the chain ({spans})"
        out.append(line)

    out.append("")
    if summary["nonstandard"]:
        out.append(f"NON-STANDARD RESIDUES ({t['nonstandard']})")
        for entry in summary["nonstandard"]:
            out.append("  " + _describe(entry))
        if any(e["kind"] == "additive" for e in summary["nonstandard"]):
            out.append("")
            out.append("  Note: crystallization additives (buffers, cryoprotectants, "
                       "precipitants) are")
            out.append("  present because of how the crystal was grown, not because of "
                       "the biology.")
    else:
        out.append("NON-STANDARD RESIDUES: none — only standard amino acids "
                   "and water are present")

    return "\n".join(out)


def as_rows(summary: dict) -> list:
    """
    Flatten the report into one table, for CSV export or a dataframe.

    Every row is (section, item, value, detail), so the whole report is a
    single flat table that opens in any spreadsheet and needs no unpacking.
    """
    t = summary["totals"]
    rows = [
        ("Overview", "PDB ID", summary.get("id") or "—", summary.get("title", "")),
        ("Overview", "Classification", summary.get("classification") or "—", ""),
        ("Overview", "Method", pretty_method(summary.get("method") or "") or "—",
         f"{summary['resolution']} Å resolution" if summary.get("resolution") else ""),
        ("Overview", "Organism",
         "; ".join(pretty_organism(o) for o in (summary.get("organisms") or [])) or "—", ""),
        ("Counts", "Chains", t["chains"], ""),
        ("Counts", "Standard amino acids", t["standard"], "the 20 standard amino acids"),
        ("Counts", "Non-standard residues", t["nonstandard"],
         "everything that is not one of the 20 amino acids; water excluded"),
    ]
    for key, label in NONSTANDARD_KINDS:
        n = sum(e["count"] for e in summary["nonstandard"] if e["kind"] == key)
        if n:
            rows.append(("Counts", label, n, ""))
    rows += [
        ("Counts", "Water molecules", t["waters"], ""),
        ("Counts", "Atoms", t["atoms"], ""),
        ("Counts", "Missing residues", t["missing"], "in SEQRES but not modelled"),
    ]
    for c in summary["chains"]:
        detail = f"residues {c['first']}–{c['last']}"
        if c["missing"]:
            detail += f", {c['missing']} missing"
        if c["gaps"]:
            detail += f", {_plural(len(c['gaps']), 'chain break')}"
        rows.append(("Chain " + c["chain"], c["type"], c["observed"], detail))
    for e in summary["nonstandard"]:
        detail = pretty_chemical(e.get("name", ""))
        if e["kind"] == "modified_aa" and e.get("parent"):
            detail = (f"replaces {e['parent']}"
                      + (f" — {detail}" if detail else ""))
        if e.get("chains"):
            detail += f" [chains {', '.join(e['chains'])}]"
        rows.append((KIND_LABELS[e["kind"]], e["code"], e["count"], detail))
    return [{"section": s, "item": i, "value": v, "detail": d} for s, i, v, d in rows]


def as_csv(summary: dict) -> str:
    """The flat table as CSV text."""
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["section", "item", "value", "detail"])
    writer.writeheader()
    writer.writerows(as_rows(summary))
    return buf.getvalue()


def as_json(summary: dict) -> str:
    """The full report as indented JSON, sequences included."""
    return json.dumps(summary, indent=2)
