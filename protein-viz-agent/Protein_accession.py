# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-11
# Summary   : The goal of this layer is to get a protein structures from different 
#             databases. We are integrating data from Uniprot, RCSB, and other 
#             databases. Typically many protein has dozens of depositions in RCSB.  
#             Opening them one at a time in the PDB web page to find out is the work 
#             this module removes.
#
#             UniProt is the spine, not RCSB text search. UniProt is the entity
#             that knows "this is the human insulin receptor, canonical length
#             1382" and cross-references every PDB entry that contains part of
#             that sequence, with the residue range each one covers. A text
#             search of the PDB answers a different question which titles
#             contain these words and quietly mixes in other species,
#             homologues and complexes the protein merely appears in.
#
#             RCSB's Data API then fills in what UniProt does not carry per
#             entry: title, release date, ligands, engineered mutation counts,
#             model counts, deposited size. One batched GraphQL request covers
#             every structure of the protein, and if it fails the table still
#             renders from UniProt alone rather than erroring out. Coverage is 
#             the column that decides most choices, so it is computed rather than 
#             quoted: residue ranges are merged per entry, expressed as a percentage 
#             of the canonical sequence, and labelled with the UniProt domains they 
#             overlap. Structures are then grouped into regions of the sequence, 
#             because "45 kinase-domain structures and 12 ectodomain structures" 
#             is the shape of the answer, not a flat list of 57 accessions.
# =============================================================================

import csv
import io
import json
import re
import urllib.parse
from functools import lru_cache

import requests

import structure_report as srep

# Bumped whenever profile() changes the shape of what it returns. Callers that
# cache a profile must feed this into their cache key: Streamlit's
# st.cache_data keys on the decorated function's own code, so a wrapper that
# merely calls profile() keeps serving dicts built by an older version of this
# module long after it has been edited.
SCHEMA_VERSION = 2

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/{acc}.json"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"

# UniProt's own accession grammar (P06213, A0A0B4J2D5). Recognising one lets a
# pasted accession be looked up as an accession instead of as free text, which
# otherwise matches every entry that merely cites it.
ACCESSION_RE = re.compile(
    r"[OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2}")
TIMEOUT = 30

# The fields UniProt is asked for. Requesting them explicitly keeps the
# response small -- a full entry for a well-studied human protein is megabytes
# of literature cross-references nothing here reads.
UNIPROT_FIELDS = ",".join([
    "accession", "id", "reviewed", "protein_name", "gene_names", "organism_name",
    "organism_id", "length", "sequence", "cc_function", "cc_alternative_products",
    "ft_domain", "ft_region", "ft_dna_bind", "ft_zn_fing", "xref_pdb",
    "protein_existence", "cc_subunit",
])

# Common model organisms, so callers can say "human" instead of a taxon id.
ORGANISMS = {
    "human": 9606, "homo sapiens": 9606,
    "mouse": 10090, "mus musculus": 10090,
    "rat": 10116, "yeast": 559292, "e. coli": 83333, "ecoli": 83333,
    "zebrafish": 7955, "fly": 7227, "drosophila": 7227,
    "c. elegans": 6239, "arabidopsis": 3702, "bovine": 9913, "any": None,
}

# How the experiment classes are named and ordered. UniProt writes "X-ray",
# "EM", "NMR", "Neutron", "Model"; RCSB writes them out in full. Both are
# folded onto these labels so one column can be filtered on.
METHOD_LABELS = {
    "X-RAY DIFFRACTION": "X-ray", "X-RAY": "X-ray", "X-RAY POWDER DIFFRACTION": "X-ray",
    "ELECTRON MICROSCOPY": "Cryo-EM", "EM": "Cryo-EM", "ELECTRON CRYSTALLOGRAPHY": "Cryo-EM",
    "SOLUTION NMR": "NMR", "SOLID-STATE NMR": "NMR", "NMR": "NMR",
    "NEUTRON DIFFRACTION": "Neutron", "NEUTRON": "Neutron",
    "FIBER DIFFRACTION": "Fiber", "SOLUTION SCATTERING": "SAXS",
    "THEORETICAL MODEL": "Model", "PREDICTED": "Model", "MODEL": "Model",
}

# Preference order when two structures cover the same thing equally well.
# Not a statement about which technique is better -- it is which file is the
# least surprising to open: one X-ray model of a domain, rather than an
# ensemble of twenty NMR models or a map-derived multi-chain assembly.
METHOD_RANK = {"X-ray": 0, "Neutron": 1, "Cryo-EM": 2, "NMR": 3,
               "Fiber": 4, "SAXS": 5, "Model": 6, "Other": 7}

# The PDB file format cannot express more than 99,999 atoms or more than 62
# chains, so entries above either limit are deposited as mmCIF only. Every
# reader in this app (sequence_utils, structure_report, measure) parses PDB
# records by column, so an entry that fails this test cannot be loaded here --
# worth saying in the table rather than discovering as a download error.
PDB_FORMAT_ATOM_LIMIT = 99999
PDB_FORMAT_CHAIN_LIMIT = 62

# Two structures belong to the same region when they overlap over at least
# this fraction of the LONGER of the two -- reciprocal overlap, not plain
# overlap. Plain overlap chains: a full-length cryo-EM structure overlaps both
# an ectodomain crystal form and an isolated kinase domain, so single-link
# merging collapses every structure of a multi-domain protein into one useless
# "residues 1-1382" group. Measuring against the longer span keeps a 300-
# residue domain construct out of the full-length group, which is the exact
# distinction someone picking a file is trying to see.
REGION_OVERLAP_FRACTION = 0.6

# UniProt Region features shorter than this are annotations, not structural
# units, and are left out of the domain list.
MIN_REGION_FEATURE = 30

# Which UniProt features count as "part of the protein" for labelling, in the
# order they are preferred as a label. A curated Domain beats a free-text
# Region: "Protein kinase" says what a construct is, "Interaction with CCAR2"
# says what a stretch of it does.
FEATURE_TYPES = ("Domain", "DNA binding", "Zinc finger", "Region")


# Evidence codes (ECO) UniProt attaches to text produced by automatic rules
# (UniRule/ARBA sequence-model matches, imported unreviewed assertions).
_AUTOMATIC_ECO = {"ECO:0000256", "ECO:0000259", "ECO:0000313", "ECO:0007829"}

# GO evidence codes. Experimental and curator-statement codes are recorded
# observations; the rest are computational transfers (by homology, phylogeny,
# InterPro mapping ...) and are predictions, however confident they look.
_GO_RECORDED = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
                "HTP", "HDA", "HMP", "HGI", "HEP", "TAS", "NAS", "IC"}

# Extra fields for function_evidence() only; kept out of UNIPROT_FIELDS so
# every ordinary lookup does not pay for the GO list.
_EVIDENCE_FIELDS = "accession,protein_name,cc_similarity,keyword,go,xref_interpro,ec"


class LookupFailed(Exception):
    """Raised internally by the fetchers; public functions return errors instead."""


# ═══════════════════════════════════════════════════════════════════════════════
# Network
# ═══════════════════════════════════════════════════════════════════════════════

def _http_json(method: str, url: str, **kwargs):
    """Perform a request and return parsed JSON, raising LookupFailed on failure."""
    try:
        r = requests.request(method, url, timeout=TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        raise LookupFailed(f"{url.split('/')[2]} returned {e.response.status_code}")
    except requests.RequestException as e:
        raise LookupFailed(f"Could not reach {url.split('/')[2]}: {e}")
    except ValueError:
        raise LookupFailed(f"{url.split('/')[2]} returned a malformed response")


@lru_cache(maxsize=64)
def _uniprot_search_raw(query: str, size: int) -> str:
    """Raw UniProt search, cached on the exact query string."""
    data = _http_json("GET", UNIPROT_SEARCH, params={
        "query": query, "fields": UNIPROT_FIELDS, "format": "json", "size": size})
    return json.dumps(data.get("results", []))


@lru_cache(maxsize=64)
def _uniprot_entry_raw(accession: str) -> str:
    """Raw UniProt entry by accession, cached."""
    data = _http_json("GET", UNIPROT_ENTRY.format(acc=urllib.parse.quote(accession)),
                      params={"fields": UNIPROT_FIELDS})
    return json.dumps(data)


_RCSB_QUERY = """
query($ids:[String!]!){
  entries(entry_ids:$ids){
    rcsb_id
    struct { title }
    exptl { method }
    rcsb_accession_info { initial_release_date }
    rcsb_entry_info {
      resolution_combined
      experimental_method
      deposited_model_count
      deposited_atom_count
      deposited_polymer_entity_instance_count
      polymer_entity_count_protein
    }
    refine { ls_R_factor_R_free }
    polymer_entities {
      rcsb_polymer_entity { pdbx_description }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids }
      entity_poly { rcsb_mutation_count }
    }
    nonpolymer_entities {
      nonpolymer_comp { chem_comp { id name } }
      rcsb_nonpolymer_entity_container_identifiers { auth_asym_ids }
    }
  }
}
"""


@lru_cache(maxsize=32)
def _rcsb_entries_raw(ids_csv: str) -> str:
    """
    Batched RCSB Data API lookup for up to a few dozen entries, cached.

    One request per chunk rather than one per entry: a protein with ninety
    structures would otherwise mean ninety round trips, which is slow enough
    that the panel would feel broken.
    """
    ids = [i for i in ids_csv.split(",") if i]
    data = _http_json("POST", RCSB_GRAPHQL,
                      json={"query": _RCSB_QUERY, "variables": {"ids": ids}})
    if data.get("errors"):
        raise LookupFailed(str(data["errors"][0].get("message", "GraphQL error")))
    return json.dumps((data.get("data") or {}).get("entries") or [])


# ═══════════════════════════════════════════════════════════════════════════════
# UniProt parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _names_of(desc: dict) -> tuple:
    """Return (recommended name, [alternative names]) from a proteinDescription."""
    rec = (desc.get("recommendedName") or {}).get("fullName", {}).get("value", "")
    alts = []
    for a in desc.get("alternativeNames") or []:
        v = (a.get("fullName") or {}).get("value")
        if v:
            alts.append(v)
    for s in (desc.get("recommendedName") or {}).get("shortNames") or []:
        if s.get("value"):
            alts.append(s["value"])
    # Chains cleaved out of a precursor ("Insulin receptor subunit alpha") are
    # named separately and are often what a structure actually contains.
    for c in desc.get("contains") or []:
        v = ((c.get("recommendedName") or {}).get("fullName") or {}).get("value")
        if v:
            alts.append(v)
    if not rec:
        sub = desc.get("submissionNames") or []
        if sub:
            rec = (sub[0].get("fullName") or {}).get("value", "")
    return rec, alts


def _genes_of(entry: dict) -> list:
    """Gene symbols, primary first, synonyms after."""
    out = []
    for g in entry.get("genes") or []:
        v = (g.get("geneName") or {}).get("value")
        if v:
            out.append(v)
        for s in g.get("synonyms") or []:
            if s.get("value"):
                out.append(s["value"])
    return out


def _function_of(entry: dict) -> str:
    """The FUNCTION comment as plain text, evidence tags stripped."""
    for c in entry.get("comments") or []:
        if c.get("commentType") == "FUNCTION":
            for t in c.get("texts") or []:
                if t.get("value"):
                    return re.sub(r"\s*\(PubMed:[^)]*\)", "", t["value"]).strip()
    return ""


def _function_is_automatic(entry: dict) -> bool:
    """
    True when every FUNCTION text rests only on automatic rules.

    Unreviewed (TrEMBL) entries often carry a FUNCTION comment written by
    UniRule/ARBA from a sequence-family match (ECO:0000256), not by a curator.
    It reads exactly like curated text, so it must be labelled as what it is.
    """
    codes = [e.get("evidenceCode", "")
             for c in entry.get("comments") or [] if c.get("commentType") == "FUNCTION"
             for t in c.get("texts") or [] for e in t.get("evidences") or []]
    return bool(codes) and all(c in _AUTOMATIC_ECO for c in codes)


def _subunit_of(entry: dict) -> str:
    """The SUBUNIT comment -- what this protein assembles with."""
    for c in entry.get("comments") or []:
        if c.get("commentType") == "SUBUNIT":
            for t in c.get("texts") or []:
                if t.get("value"):
                    return re.sub(r"\s*\(PubMed:[^)]*\)", "", t["value"]).strip()
    return ""


def _isoforms_of(entry: dict) -> list:
    """
    Splice isoforms declared in ALTERNATIVE PRODUCTS.

    Worth surfacing because the canonical sequence UniProt reports the length
    of is one isoform among several, and a structure's residue numbering
    follows whichever isoform the construct came from.
    """
    out = []
    for c in entry.get("comments") or []:
        if c.get("commentType") == "ALTERNATIVE PRODUCTS":
            for iso in c.get("isoforms") or []:
                out.append({
                    "id": (iso.get("isoformIds") or [""])[0],
                    "name": (iso.get("name") or {}).get("value", ""),
                    "canonical": iso.get("isoformSequenceStatus") == "Displayed",
                })
    return out


def _domains_of(entry: dict) -> list:
    """
    Named domains and regions, as {name, start, end}.

    UniProt's Region features range from real structural units down to
    "Important for interaction with IRS1" spanning a single residue. Only
    Regions long enough to be a piece of structure are kept, or the domain
    column fills up with annotations no construct was ever designed around.
    """
    out = []
    for f in entry.get("features") or []:
        if f.get("type") not in FEATURE_TYPES:
            continue
        loc = f.get("location") or {}
        start = (loc.get("start") or {}).get("value")
        end = (loc.get("end") or {}).get("value")
        desc = (f.get("description") or "").strip()
        # DNA-binding and zinc-finger features often carry no description at
        # all; the feature type is the name people would use anyway.
        if not desc and f["type"] in ("DNA binding", "Zinc finger"):
            desc = f["type"]
        if not (start and end and desc) or desc.lower().startswith("disordered"):
            continue
        if f["type"] == "Region" and int(end) - int(start) + 1 < MIN_REGION_FEATURE:
            continue
        out.append({"name": desc, "start": int(start), "end": int(end),
                    "kind": f["type"]})
    return sorted(out, key=lambda d: (d["start"], d["end"], d["name"]))


def _parse_chain_spec(spec: str) -> list:
    """
    Parse UniProt's PDB "Chains" property into segments.

    The property looks like "A/B=1005-1310" or "E=28-337, F=731-746": chain
    labels left of the equals sign share one residue range, and comma-separated
    groups are independent. Returns [(chain, start, end), ...].
    """
    segs = []
    for part in (spec or "").split(","):
        part = part.strip()
        if "=" not in part:
            continue
        chains, _, rng = part.partition("=")
        m = re.match(r"\s*(-?\d+)\s*-\s*(-?\d+)\s*$", rng)
        if not m:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        for ch in chains.split("/"):
            ch = ch.strip()
            if ch:
                segs.append((ch, start, end))
    return segs


def _merge_ranges(ranges: list) -> list:
    """Merge overlapping or abutting (start, end) pairs into disjoint spans."""
    if not ranges:
        return []
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _span_len(ranges: list) -> int:
    """Total residues covered by a list of disjoint spans."""
    return sum(b - a + 1 for a, b in ranges)


def _parse_resolution(value: str):
    """'2.70 A' -> 2.70; '-' or '' -> None."""
    m = re.search(r"(\d+(?:\.\d+)?)", value or "")
    return float(m.group(1)) if m else None


def _method_label(raw: str) -> str:
    """Fold a UniProt or RCSB method string onto one of the METHOD_LABELS."""
    key = (raw or "").strip().upper()
    if key in METHOD_LABELS:
        return METHOD_LABELS[key]
    for needle, label in (("X-RAY", "X-ray"), ("ELECTRON", "Cryo-EM"),
                          ("NMR", "NMR"), ("NEUTRON", "Neutron"),
                          ("SCATTERING", "SAXS"), ("MODEL", "Model")):
        if needle in key:
            return label
    return "Other" if key else "Unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Search
# ═══════════════════════════════════════════════════════════════════════════════

def organism_id(organism) -> int:
    """Resolve 'human', 9606 or 'Homo sapiens' to a taxon id (None = any)."""
    if organism is None or organism == "":
        return None
    if isinstance(organism, int):
        return organism
    text = str(organism).strip().lower()
    if text.isdigit():
        return int(text)
    return ORGANISMS.get(text, None)


def build_query(name: str, taxon: int = 9606, reviewed_only: bool = True) -> str:
    """
    Build the UniProt query string for a protein name typed by a person.

    A bare accession is matched as an accession; anything else is matched
    against protein name, gene name and the free-text index at once, because
    people type "EGFR", "epidermal growth factor receptor" and "erbB1"
    interchangeably and only one of those is a gene symbol.
    """
    text = (name or "").strip()
    clauses = []
    if ACCESSION_RE.fullmatch(text.upper()):
        # An accession names exactly one entry, so the organism and review
        # filters can only make it vanish -- pasting a mouse accession with the
        # organism left on "human" would otherwise report no such protein.
        return f"accession:{text.upper()}"
    else:
        safe = text.replace('"', " ").strip()
        quoted = f'"{safe}"'
        clauses.append(
            f"(protein_name:{quoted} OR gene:{quoted} OR gene_exact:{quoted} OR {quoted})")
    if taxon:
        clauses.append(f"organism_id:{taxon}")
    if reviewed_only:
        clauses.append("reviewed:true")
    return " AND ".join(clauses)


def search_proteins(name: str, organism="human", reviewed_only: bool = True,
                    limit: int = 25):
    """
    Find UniProt entries for a protein name.

    Reviewed (Swiss-Prot) entries only by default: an unreviewed search for a
    human protein returns dozens of TrEMBL fragments of the same gene, which is
    noise in a picker whose whole job is to be short. If nothing reviewed
    matches, the search is retried unreviewed rather than reporting nothing.

    Args:
        name         : Protein name, gene symbol or UniProt accession.
        organism     : 'human' (default), a taxon id, or None for any species.
        reviewed_only: Restrict to Swiss-Prot.
        limit        : Maximum entries to return.

    Returns:
        (hits, error) -- hits is a list of dicts from protein_row(), error is
        None on success or a message string on failure.
    """
    if not (name or "").strip():
        return [], "Type a protein name, gene symbol or UniProt accession."
    taxon = organism_id(organism)
    try:
        size = max(1, min(int(limit), 100))
        raw = json.loads(_uniprot_search_raw(build_query(name, taxon, reviewed_only), size))
        if not raw and reviewed_only:
            raw = json.loads(_uniprot_search_raw(build_query(name, taxon, False), size))
        if not raw and taxon:
            # Widening to every species beats reporting nothing: the organism
            # column and the profile header both name the species, so a hit
            # from another organism announces itself rather than passing as
            # the human protein.
            raw = json.loads(_uniprot_search_raw(build_query(name, None, True), size))
    except LookupFailed as e:
        return [], str(e)
    hits = [protein_row(e) for e in raw]
    _rank_hits(hits, name)
    return hits, None


def _rank_hits(hits: list, query: str) -> None:
    """
    Order search hits, in place, mostly by leaving UniProt's order alone.

    UniProt ranks its own results by relevance and does it well: "p53" puts
    TP53 first, "hemoglobin" puts HBB first. Re-sorting by structure count
    instead looks sensible and is not -- it answers "ubiquitin" with RBX1,
    which merely has the word in its name and ninety-nine structures. So the
    original order is preserved and only two things move: an entry whose gene
    symbol or name the query matches exactly goes to the top, and unreviewed
    entries fall to the bottom.
    """
    q = (query or "").strip().lower()

    def exactness(h):
        if h["gene"].lower() == q or h["accession"].lower() == q:
            return 0
        if h["protein_name"].lower() == q or q in {a.lower() for a in h["alt_names"]}:
            return 1
        if q in {g.lower() for g in h["gene_synonyms"]}:
            return 2
        return 3

    order = {id(h): i for i, h in enumerate(hits)}
    hits.sort(key=lambda h: (not h["reviewed"], exactness(h), order[id(h)]))


def protein_row(entry: dict) -> dict:
    """Condense one raw UniProt entry into the fields the protein table shows."""
    name, alts = _names_of(entry.get("proteinDescription") or {})
    genes = _genes_of(entry)
    pdb_ids = [x["id"] for x in entry.get("uniProtKBCrossReferences") or []
               if x.get("database") == "PDB"]
    return {
        "accession": entry.get("primaryAccession", ""),
        "entry_name": entry.get("uniProtkbId", ""),
        "protein_name": name,
        "alt_names": alts,
        "gene": genes[0] if genes else "",
        "gene_synonyms": genes[1:],
        "organism": (entry.get("organism") or {}).get("scientificName", ""),
        "taxon_id": (entry.get("organism") or {}).get("taxonId"),
        "length": (entry.get("sequence") or {}).get("length", 0),
        "mass": (entry.get("sequence") or {}).get("molWeight", 0),
        "reviewed": entry.get("entryType", "").startswith("UniProtKB reviewed"),
        "existence": entry.get("proteinExistence", ""),
        "n_structures": len(pdb_ids),
        "pdb_ids": pdb_ids,
        "function": _function_of(entry),
        "_entry": entry,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Structure table
# ═══════════════════════════════════════════════════════════════════════════════

def _uniprot_structures(entry: dict, length: int, domains: list) -> list:
    """
    One row per PDB cross-reference, from UniProt alone.

    This is the part that always works. Everything RCSB adds later is detail
    on top of rows that already carry the two things that decide a choice:
    which technique produced the structure, and which residues it contains.
    """
    rows = []
    for x in entry.get("uniProtKBCrossReferences") or []:
        if x.get("database") != "PDB":
            continue
        props = {p.get("key"): p.get("value") for p in x.get("properties") or []}
        segs = _parse_chain_spec(props.get("Chains", ""))
        ranges = _merge_ranges([(s, e) for _, s, e in segs])
        covered = _span_len(ranges)
        chains = sorted({c for c, _, _ in segs})
        rows.append({
            "pdb_id": x.get("id", "").upper(),
            "method": _method_label(props.get("Method", "")),
            "resolution": _parse_resolution(props.get("Resolution", "")),
            "chains": chains,
            "ranges": ranges,
            "start": ranges[0][0] if ranges else None,
            "end": ranges[-1][1] if ranges else None,
            "covered": covered,
            "coverage_pct": round(100.0 * covered / length, 1) if length and covered else 0.0,
            "domains": _domains_in(ranges, domains),
            # Filled in by _enrich(); defaults keep the table renderable without it.
            "title": "", "released": "", "models": None, "r_free": None,
            "ligands": [], "partners": [], "mutations": None,
            "atom_count": None, "chain_count": None, "pdb_format": True,
        })
    rows.sort(key=lambda r: r["pdb_id"])
    return rows


def _domains_in(ranges: list, domains: list) -> list:
    """
    Names of the domains a structure substantially contains.

    Half a domain is not that domain: a construct that clips the last thirty
    residues off a kinase domain is a different thing from one that holds it
    whole, and labelling both "Protein kinase" would hide exactly the
    difference this table exists to show. The threshold is 60% of the domain.
    """
    hits = []
    for d in domains:
        span = d["end"] - d["start"] + 1
        inside = sum(max(0, min(b, d["end"]) - max(a, d["start"]) + 1) for a, b in ranges)
        if span and inside >= 0.6 * span:
            hits.append(d)
    hits.sort(key=lambda d: (FEATURE_TYPES.index(d["kind"]), d["start"]))
    return list(dict.fromkeys(d["name"] for d in hits))


def _enrich(rows: list, protein_names: set, chunk: int = 40):
    """
    Add RCSB detail to UniProt rows in batches.

    Returns a note string when enrichment could not run, so the caller can say
    so in the panel instead of silently showing a table with empty columns.
    """
    if not rows:
        return ""
    by_id = {r["pdb_id"]: r for r in rows}
    ids = list(by_id)
    failures = []
    for i in range(0, len(ids), chunk):
        block = ids[i:i + chunk]
        try:
            entries = json.loads(_rcsb_entries_raw(",".join(block)))
        except LookupFailed as e:
            failures.append(str(e))
            continue
        for e in entries or []:
            row = by_id.get((e.get("rcsb_id") or "").upper())
            if row:
                _apply_rcsb(row, e, protein_names)
    if failures:
        return ("RCSB details unavailable (" + failures[0] + ") — the table shows "
                "UniProt's method, resolution and coverage only.")
    return ""


def _apply_rcsb(row: dict, e: dict, protein_names: set) -> None:
    """Merge one RCSB entry record into its UniProt-derived row."""
    info = e.get("rcsb_entry_info") or {}
    row["title"] = ((e.get("struct") or {}).get("title") or "").strip()
    methods = [_method_label(m.get("method", "")) for m in e.get("exptl") or []]
    if methods:
        # RCSB names the experiment in full and lists every method used for a
        # combined-technique entry; UniProt reports only the first.
        row["method"] = "+".join(dict.fromkeys(methods))
    res = info.get("resolution_combined") or []
    if res:
        row["resolution"] = round(float(res[0]), 2)
    row["models"] = info.get("deposited_model_count")
    row["atom_count"] = info.get("deposited_atom_count")
    row["chain_count"] = info.get("deposited_polymer_entity_instance_count")
    row["pdb_format"] = not (
        (row["atom_count"] or 0) > PDB_FORMAT_ATOM_LIMIT
        or (row["chain_count"] or 0) > PDB_FORMAT_CHAIN_LIMIT)
    refine = e.get("refine") or []
    if refine and refine[0].get("ls_R_factor_R_free") is not None:
        row["r_free"] = round(float(refine[0]["ls_R_factor_R_free"]), 3)
    date = ((e.get("rcsb_accession_info") or {}).get("initial_release_date") or "")
    row["released"] = date[:10]

    ours = set(row["chains"])
    partners, mutations = [], 0
    for pe in e.get("polymer_entities") or []:
        desc = ((pe.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "").strip()
        ids = set(((pe.get("rcsb_polymer_entity_container_identifiers") or {})
                   .get("auth_asym_ids")) or [])
        is_ours = bool(ours & ids) or desc.lower() in protein_names
        if is_ours:
            mutations = max(mutations, (pe.get("entity_poly") or {}).get("rcsb_mutation_count") or 0)
        elif desc:
            partners.append(desc)
    row["partners"] = list(dict.fromkeys(partners))
    row["mutations"] = mutations

    ligands = []
    for ne in e.get("nonpolymer_entities") or []:
        comp = ((ne.get("nonpolymer_comp") or {}).get("chem_comp") or {})
        code = (comp.get("id") or "").upper()
        if not code:
            continue
        kind = srep.classify_component(code)
        ligands.append({"code": code, "kind": kind,
                        "name": srep.pretty_chemical(comp.get("name", "") or "")})
    # Ligands and cofactors first, then ions, then the crystallography.
    order = {"ligand": 0, "cofactor": 1, "ion": 2, "additive": 3}
    row["ligands"] = sorted(ligands, key=lambda l: (order.get(l["kind"], 4), l["code"]))


def notable_ligands(row: dict) -> list:
    """Ligands and cofactors only -- what was bound on purpose."""
    return [l for l in row["ligands"] if l["kind"] in ("ligand", "cofactor")]


# ═══════════════════════════════════════════════════════════════════════════════
# Regions
# ═══════════════════════════════════════════════════════════════════════════════

def _regions(rows: list, domains: list, length: int) -> list:
    """
    Group structures into the stretches of sequence they cover.

    This is the answer to "which of these fifty files do I want": almost every
    multi-domain protein has been crystallised piecewise, and the first
    decision is which piece, not which accession.

    Structures are assigned to the first group whose defining construct they
    overlap over REGION_OVERLAP_FRACTION of the longer span. Groups are seeded
    in order of decreasing extent, so the full-length depositions define the
    full-length group and the domain constructs form their own, rather than
    being absorbed by whatever happened to be read first.
    """
    placed = [r for r in rows if r["start"] is not None]
    clusters = []
    for row in sorted(placed, key=lambda r: (-(r["end"] - r["start"] + 1), r["start"])):
        span = row["end"] - row["start"] + 1
        for c in clusters:
            lead = c["lead"]
            overlap = min(lead[1], row["end"]) - max(lead[0], row["start"]) + 1
            longer = max(lead[1] - lead[0] + 1, span)
            if overlap > 0 and overlap >= REGION_OVERLAP_FRACTION * longer:
                c["start"] = min(c["start"], row["start"])
                c["end"] = max(c["end"], row["end"])
                c["rows"].append(row)
                break
        else:
            clusters.append({"start": row["start"], "end": row["end"],
                             "lead": (row["start"], row["end"]), "rows": [row]})

    out = []
    for c in clusters:
        names = _domains_in([(c["start"], c["end"])], domains)
        label = ", ".join(names[:3]) if names else f"residues {c['start']}–{c['end']}"
        best = rank_structures(c["rows"])[0] if c["rows"] else None
        out.append({
            "start": c["start"], "end": c["end"], "label": label,
            "count": len(c["rows"]),
            "length": c["end"] - c["start"] + 1,
            "pct": round(100.0 * (c["end"] - c["start"] + 1) / length, 1) if length else 0.0,
            "methods": sorted({r["method"] for r in c["rows"]}),
            "best": best["pdb_id"] if best else None,
            "pdb_ids": [r["pdb_id"] for r in c["rows"]],
        })
    out.sort(key=lambda r: (-r["count"], r["start"]))
    return out


def rank_structures(rows: list, prefer: str = "balanced") -> list:
    """
    Order structures by how good a starting file they are.

    'balanced' (the default) reads as: prefer a file that can actually be
    loaded, then one that covers more of the protein, then a better
    resolution, then the least surprising technique. 'coverage' and
    'resolution' make one of those the only thing that matters, because
    "the most complete structure" and "the sharpest structure" are both
    questions people arrive with and they rarely have the same answer.
    """
    def res(r):
        return r["resolution"] if r["resolution"] is not None else 99.0

    if prefer == "coverage":
        key = lambda r: (not r["pdb_format"], -r["covered"], res(r))
    elif prefer == "resolution":
        key = lambda r: (not r["pdb_format"], res(r), -r["covered"])
    else:
        key = lambda r: (not r["pdb_format"], -round(r["coverage_pct"] / 10),
                         res(r), METHOD_RANK.get(r["method"].split("+")[0], 7),
                         -r["covered"])
    return sorted(rows, key=key)


# ═══════════════════════════════════════════════════════════════════════════════
# Profile
# ═══════════════════════════════════════════════════════════════════════════════

def profile(accession: str, enrich: bool = True):
    """
    Build the full protein profile: identity, sequence, every PDB structure.

    Args:
        accession: UniProt accession, e.g. "P06213".
        enrich   : Also query RCSB for titles, ligands, mutations and sizes.
                   Turning it off makes the lookup a single HTTP request.

    Returns:
        (profile dict, error). See the module header for what the dict holds.
    """
    acc = (accession or "").strip().upper()
    if not acc:
        return None, "No UniProt accession given."
    try:
        entry = json.loads(_uniprot_entry_raw(acc))
    except LookupFailed as e:
        return None, str(e)

    base = protein_row(entry)
    length = base["length"]
    domains = _domains_of(entry)
    rows = _uniprot_structures(entry, length, domains)

    note = ""
    if enrich and rows:
        names = {base["protein_name"].lower()} | {a.lower() for a in base["alt_names"]}
        note = _enrich(rows, names)

    ranked = rank_structures(rows)
    prof = {
        "schema": SCHEMA_VERSION,
        "accession": base["accession"],
        "entry_name": base["entry_name"],
        "protein_name": base["protein_name"],
        "alt_names": base["alt_names"],
        "gene": base["gene"],
        "gene_synonyms": base["gene_synonyms"],
        "organism": base["organism"],
        "taxon_id": base["taxon_id"],
        "reviewed": base["reviewed"],
        "existence": base["existence"],
        "length": length,
        "mass": base["mass"],
        "sequence": (entry.get("sequence") or {}).get("value", ""),
        "function": base["function"],
        "function_automatic": _function_is_automatic(entry),
        "subunit": _subunit_of(entry),
        "isoforms": _isoforms_of(entry),
        "domains": domains,
        "structures": rows,
        "regions": _regions(rows, domains, length),
        "recommended": ranked[0]["pdb_id"] if ranked else None,
        "sharpest": (min((r for r in rows if r["resolution"] is not None),
                         key=lambda r: r["resolution"])["pdb_id"]
                     if any(r["resolution"] is not None for r in rows) else None),
        "note": note,
    }
    prof["totals"] = _totals(rows, length)
    return prof, None


def _totals(rows: list, length: int) -> dict:
    """Counts for the metric row above the table."""
    methods = {}
    for r in rows:
        methods[r["method"]] = methods.get(r["method"], 0) + 1
    covered = _merge_ranges([rng for r in rows for rng in r["ranges"]])
    resolutions = [r["resolution"] for r in rows if r["resolution"] is not None]
    return {
        "structures": len(rows),
        "methods": dict(sorted(methods.items(), key=lambda kv: -kv[1])),
        "xray": sum(1 for r in rows if r["method"].startswith("X-ray")),
        "em": sum(1 for r in rows if "Cryo-EM" in r["method"]),
        "nmr": sum(1 for r in rows if "NMR" in r["method"]),
        "with_ligand": sum(1 for r in rows if notable_ligands(r)),
        "mutants": sum(1 for r in rows if (r["mutations"] or 0) > 0),
        "cif_only": sum(1 for r in rows if not r["pdb_format"]),
        "best_resolution": min(resolutions) if resolutions else None,
        "sequence_covered": _span_len(covered),
        "sequence_covered_pct": round(100.0 * _span_len(covered) / length, 1) if length else 0.0,
        "uncovered_spans": _gaps(covered, length),
    }


def _gaps(covered: list, length: int) -> list:
    """Stretches of the canonical sequence no deposited structure contains."""
    gaps, cursor = [], 1
    for a, b in covered:
        if a > cursor:
            gaps.append((cursor, a - 1))
        cursor = max(cursor, b + 1)
    if length and cursor <= length:
        gaps.append((cursor, length))
    return [g for g in gaps if g[1] - g[0] + 1 >= 10]


def lookup(name: str, organism="human", reviewed_only: bool = True, enrich: bool = True):
    """
    Name in, profile out -- the one call the rest of the app makes.

    Returns:
        (profile, hits, error). hits is the full candidate list so a caller can
        offer the alternatives when the top match is not the protein meant.
    """
    hits, err = search_proteins(name, organism, reviewed_only)
    if err:
        return None, [], err
    if not hits:
        where = f" in {organism}" if organism else ""
        return None, [], f"No UniProt entry matches '{name}'{where}."
    prof, err = profile(hits[0]["accession"], enrich=enrich)
    return prof, hits, err


# ═══════════════════════════════════════════════════════════════════════════════
# Function evidence (entries with no FUNCTION comment)
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=64)
def _uniprot_evidence_raw(accession: str) -> str:
    """Raw UniProt entry with the family/GO/InterPro fields, cached."""
    data = _http_json("GET", UNIPROT_ENTRY.format(acc=urllib.parse.quote(accession)),
                      params={"fields": _EVIDENCE_FIELDS})
    return json.dumps(data)


def function_evidence(accession: str):
    """
    Collect what UniProt still knows about an entry that has no FUNCTION text.

    About 40% of the UniProt entries that have a PDB structure carry no
    FUNCTION comment, yet nearly all of them are placed in a sequence family
    (InterPro, "Belongs to the ... family") and many carry GO terms. This
    splits that into what was observed (GO terms with experimental or curator
    evidence codes) and what was inferred by computation (family membership,
    InterPro, electronically transferred GO terms, EC numbers), so a caller
    can report the second as a low-confidence prediction and never at the
    confidence of curated text.

    Returns:
        (dict, error). dict keys: recorded_go / inferred_go (lists of
        (aspect, term, code)), family (list of SIMILARITY texts), interpro
        (list of (id, name)), ec (list), keywords (list of (category, name)).
    """
    acc = (accession or "").strip().upper()
    try:
        entry = json.loads(_uniprot_evidence_raw(acc))
    except LookupFailed as e:
        return None, str(e)

    recorded, inferred, interpro = [], [], []
    for x in entry.get("uniProtKBCrossReferences") or []:
        props = {p.get("key"): p.get("value", "") for p in x.get("properties") or []}
        if x.get("database") == "InterPro":
            interpro.append((x.get("id", ""), props.get("EntryName", "")))
        elif x.get("database") == "GO":
            aspect, _, term = props.get("GoTerm", "").partition(":")
            code = props.get("GoEvidenceType", "").split(":")[0]
            if aspect not in ("F", "P") or not term:
                continue   # cellular component is location, not function
            (recorded if code in _GO_RECORDED else inferred).append((aspect, term, code))

    family = [re.sub(r"\s*\(PubMed:[^)]*\)", "", t["value"]).strip()
              for c in entry.get("comments") or [] if c.get("commentType") == "SIMILARITY"
              for t in c.get("texts") or [] if t.get("value")]

    desc = entry.get("proteinDescription") or {}
    names = [desc.get("recommendedName") or {}] + list(desc.get("submissionNames") or [])
    ec = sorted({e["value"] for n in names for e in n.get("ecNumbers") or [] if e.get("value")})

    keywords = [(k.get("category", ""), k.get("name", ""))
                for k in entry.get("keywords") or []
                if k.get("category") in ("Molecular function", "Biological process", "Ligand")]

    return {"accession": acc, "recorded_go": recorded, "inferred_go": inferred,
            "family": family, "interpro": interpro, "ec": ec,
            "keywords": keywords}, None


# ═══════════════════════════════════════════════════════════════════════════════
# Filtering
# ═══════════════════════════════════════════════════════════════════════════════

def filter_structures(rows: list, method: str = "", max_resolution: float = None,
                      min_coverage: float = 0.0, ligands_only: bool = False,
                      wild_type_only: bool = False, region: tuple = None,
                      loadable_only: bool = False) -> list:
    """
    Narrow the structure table the way the panel's controls do.

    Args:
        method        : "X-ray", "NMR", "Cryo-EM" ... or "" for all.
        max_resolution: Keep structures at or better than this, in Å. Entries
                        with no resolution (NMR) are kept only when no
                        resolution limit is asked for.
        min_coverage  : Minimum percentage of the canonical sequence.
        ligands_only  : Keep only entries with a bound ligand or cofactor.
        wild_type_only: Drop entries with engineered mutations.
        region        : (start, end) -- keep entries overlapping this span.
        loadable_only : Drop entries with no PDB-format file.
    """
    out = []
    for r in rows:
        if method and method.lower() not in r["method"].lower():
            continue
        if max_resolution is not None:
            if r["resolution"] is None or r["resolution"] > max_resolution:
                continue
        if r["coverage_pct"] < min_coverage:
            continue
        if ligands_only and not notable_ligands(r):
            continue
        if wild_type_only and (r["mutations"] or 0) > 0:
            continue
        if loadable_only and not r["pdb_format"]:
            continue
        if region and r["start"] is not None:
            if min(region[1], r["end"]) - max(region[0], r["start"]) + 1 <= 0:
                continue
        out.append(r)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════════════════════

PROTEIN_HEADERS = ["Accession", "Entry", "Protein", "Gene", "Organism",
                   "Length", "Structures", "Reviewed"]

STRUCTURE_HEADERS = ["PDB", "Method", "Resolution (Å)", "Models", "Chains",
                     "Residues", "Coverage", "Region", "Ligands", "Mutations",
                     "Released", "Title"]


def protein_table_rows(hits: list) -> list:
    """Rows for PROTEIN_HEADERS."""
    return [[h["accession"], h["entry_name"], h["protein_name"], h["gene"] or "—",
             h["organism"], h["length"], h["n_structures"],
             "✓" if h["reviewed"] else "—"] for h in hits]


def structure_table_rows(rows: list, max_title: int = 70) -> list:
    """Rows for STRUCTURE_HEADERS."""
    out = []
    for r in rows:
        lig = notable_ligands(r)
        title = r["title"]
        if max_title and len(title) > max_title:
            title = title[:max_title - 1].rstrip() + "…"
        out.append([
            r["pdb_id"] + ("" if r["pdb_format"] else " ⚠"),
            r["method"],
            f"{r['resolution']:.2f}" if r["resolution"] is not None else "—",
            r["models"] if r["models"] else "—",
            "/".join(r["chains"]) or "—",
            f"{r['start']}–{r['end']}" if r["start"] is not None else "—",
            f"{r['coverage_pct']:.0f}%" if r["coverage_pct"] else "—",
            ", ".join(r["domains"][:2]) or "—",
            ", ".join(l["code"] for l in lig[:4]) or "—",
            r["mutations"] if r["mutations"] else ("—" if r["mutations"] is None else "0"),
            r["released"] or "—",
            title or "—",
        ])
    return out


def region_table_rows(prof: dict) -> list:
    """Rows for the region summary: what part of the protein is solved, and how."""
    return [[r["label"], f"{r['start']}–{r['end']}", r["count"],
             ", ".join(f"{m}" for m in r["methods"]), r["best"] or "—"]
            for r in prof["regions"]]


REGION_HEADERS = ["Region", "Residues", "Structures", "Methods", "Best pick"]


def as_brief(prof: dict) -> str:
    """One-paragraph answer: what this protein is and which file to open."""
    t = prof["totals"]
    if not t["structures"]:
        return (f"{prof['protein_name']} ({prof['accession']}, {prof['length']} aa) has "
                f"no experimental structure in the PDB.")
    bits = ", ".join(f"{n} {m}" for m, n in t["methods"].items())
    lead = (f"{prof['protein_name']} ({prof['accession']}, {prof['length']} aa) has "
            f"{t['structures']} PDB structures — {bits} — covering "
            f"{t['sequence_covered_pct']:.0f}% of the sequence in "
            f"{len(prof['regions'])} region(s).")
    best = next((r for r in prof["structures"] if r["pdb_id"] == prof["recommended"]), None)
    if best:
        detail = f"{best['method']}"
        if best["resolution"] is not None:
            detail += f", {best['resolution']:.2f} Å"
        lead += (f" Best starting file: {best['pdb_id']} ({detail}, residues "
                 f"{best['start']}–{best['end']}, {best['coverage_pct']:.0f}% coverage).")
    return lead


def as_text(prof: dict) -> str:
    """The whole profile as plain text, for pasting into notes or a paper."""
    L = []
    L.append(f"{prof['protein_name']}  [{prof['accession']} / {prof['entry_name']}]")
    L.append("=" * min(78, max(20, len(L[0]))))
    if prof["gene"]:
        L.append(f"Gene        : {prof['gene']}" +
                 (f" (also {', '.join(prof['gene_synonyms'][:4])})" if prof["gene_synonyms"] else ""))
    L.append(f"Organism    : {prof['organism']}")
    L.append(f"Length      : {prof['length']} aa" +
             (f", {prof['mass'] / 1000:.1f} kDa" if prof["mass"] else ""))
    L.append(f"Reviewed    : {'Swiss-Prot' if prof['reviewed'] else 'TrEMBL (unreviewed)'}")
    if prof["alt_names"]:
        L.append(f"Also called : {', '.join(prof['alt_names'][:6])}")
    if prof["isoforms"]:
        L.append("Isoforms    : " + ", ".join(
            f"{i['name']} ({i['id']}{', canonical' if i['canonical'] else ''})"
            for i in prof["isoforms"]))
    if prof["function"]:
        L.append("")
        L.append("Function")
        L.append("--------")
        L.append(prof["function"])
    L.append("")
    L.append(as_brief(prof))
    if prof["domains"]:
        L.append("")
        L.append("Domains")
        L.append("-------")
        for d in prof["domains"]:
            L.append(f"  {d['start']:>5}-{d['end']:<5}  {d['name']}")
    if prof["regions"]:
        L.append("")
        L.append("Structural coverage")
        L.append("-------------------")
        for r in prof["regions"]:
            L.append(f"  {r['start']:>5}-{r['end']:<5}  {r['count']:>3} structures  "
                     f"{r['label']}  (best: {r['best']})")
        for a, b in prof["totals"]["uncovered_spans"]:
            L.append(f"  {a:>5}-{b:<5}    no structure")
    if prof["structures"]:
        L.append("")
        L.append("Structures")
        L.append("----------")
        head = ["PDB", "Method", "Res", "Chains", "Residues", "Cov", "Ligands", "Title"]
        L.append("  " + "  ".join(h.ljust(w) for h, w in
                                  zip(head, [5, 9, 6, 10, 13, 5, 18, 40])))
        for r in rank_structures(prof["structures"]):
            lig = ",".join(l["code"] for l in notable_ligands(r)[:3]) or "-"
            cells = [r["pdb_id"], r["method"],
                     f"{r['resolution']:.2f}" if r["resolution"] is not None else "-",
                     "/".join(r["chains"])[:10] or "-",
                     f"{r['start']}-{r['end']}" if r["start"] is not None else "-",
                     f"{r['coverage_pct']:.0f}%", lig, (r["title"] or "")[:40]]
            L.append("  " + "  ".join(str(c).ljust(w) for c, w in
                                      zip(cells, [5, 9, 6, 10, 13, 5, 18, 40])))
    if prof["note"]:
        L.append("")
        L.append(f"Note: {prof['note']}")
    return "\n".join(L)


def as_csv(prof: dict) -> str:
    """The structure table as CSV -- one row per PDB entry."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["uniprot", "protein", "gene", "organism", "uniprot_length",
                "pdb_id", "method", "resolution_A", "models", "chains",
                "first_residue", "last_residue", "residues_covered",
                "coverage_percent", "domains", "ligands", "cofactors", "ions",
                "partners", "mutations", "pdb_format_available", "released", "title"])
    for r in rank_structures(prof["structures"]):
        kinds = {k: [l["code"] for l in r["ligands"] if l["kind"] == k]
                 for k in ("ligand", "cofactor", "ion")}
        w.writerow([prof["accession"], prof["protein_name"], prof["gene"],
                    prof["organism"], prof["length"], r["pdb_id"], r["method"],
                    "" if r["resolution"] is None else r["resolution"],
                    r["models"] or "", "/".join(r["chains"]),
                    r["start"] or "", r["end"] or "", r["covered"],
                    r["coverage_pct"], "; ".join(r["domains"]),
                    "; ".join(kinds["ligand"]), "; ".join(kinds["cofactor"]),
                    "; ".join(kinds["ion"]),
                    "; ".join(r["partners"][:5]),
                    "" if r["mutations"] is None else r["mutations"],
                    "yes" if r["pdb_format"] else "no", r["released"], r["title"]])
    return buf.getvalue()


def as_json(prof: dict) -> str:
    """The full profile as JSON, sequence included."""
    return json.dumps(prof, indent=2)


def as_fasta(prof: dict) -> str:
    """Canonical sequence in FASTA, wrapped at 60 columns."""
    seq = prof.get("sequence", "")
    head = (f">sp|{prof['accession']}|{prof['entry_name']} {prof['protein_name']} "
            f"OS={prof['organism']} GN={prof['gene']}")
    return head + "\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n"


def to_dataframe(rows: list, headers: list):
    """
    The table as a pandas DataFrame.

    pandas is imported here rather than at module scope: every other path
    through this module works without it, and this app runs in environments
    where the pandas/pyarrow stack is not importable. Returns None if it is
    not available.
    """
    try:
        import pandas as pd
    except Exception:
        return None
    return pd.DataFrame(rows, columns=headers)


def structures_dataframe(prof: dict, prefer: str = "balanced"):
    """The structure table as a DataFrame, ranked -- None if pandas is missing."""
    return to_dataframe(structure_table_rows(rank_structures(prof["structures"], prefer)),
                        STRUCTURE_HEADERS)


def proteins_dataframe(hits: list):
    """The search-hit table as a DataFrame -- None if pandas is missing."""
    return to_dataframe(protein_table_rows(hits), PROTEIN_HEADERS)


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "insulin receptor"
    prof_, hits_, err_ = lookup(query)
    if err_:
        print(err_)
    else:
        print(as_text(prof_))
        if len(hits_) > 1:
            print("\nOther matches: " + ", ".join(
                f"{h['accession']} ({h['gene'] or h['protein_name']})" for h in hits_[1:8]))
