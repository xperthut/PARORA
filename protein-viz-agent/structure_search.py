# =============================================================================
# Summary   : Structure-based fold/homology search (Foldseek).
#
#             Answers "what known structures does this look like" from the
#             3D coordinates alone, so it still works where topology.py's
#             CATH/SCOP lookup has nothing to key on — AlphaFold predicted
#             models, local files, very recent or genuinely novel folds.
#             Each hit is a real PDB chain that Foldseek aligned to the
#             query; callers look up that hit's own CATH/SCOP classification
#             (topology.lookup_classification) to say which known fold the
#             query resembles. Nothing is inferred from names.
#
#             Two optional pieces, both discovered at runtime and reported
#             "unavailable" when missing, same pattern as DSSP/AmberTools/
#             PyMOL:
#             - `foldseek` itself (single static binary; FOLDSEEK_BIN env
#               var, else PATH).
#             - a Foldseek reference database to search against (FOLDSEEK_DB
#               env var, else ./foldseek_db/pdb next to this file). The PDB
#               one is ~2.2 GB compressed and must be downloaded once, on
#               purpose — see run.sh / setup_tools.sh — never silently.
#
#             No Streamlit import — same split as topology.py/
#             interactions.py, so this is testable standalone and app.py owns
#             the @st.cache_data wrapping.
# =============================================================================

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1

DEFAULT_DB = Path(__file__).resolve().parent / "foldseek_db" / "pdb"

# Columns requested from easy-search, in order. alntmscore is the TM-score
# normalised by alignment length; prob is Foldseek's probability that query
# and target are homologous (same SCOP superfamily), which is the number
# that answers "same fold or not" most directly.
_FORMAT = ["query", "target", "fident", "alntmscore", "evalue", "prob",
           "qstart", "qend", "qlen", "tstart", "tend", "theader"]

# PDB100 target names look like "7ard-assembly1.cif.gz_A" (or plain
# "1abc_A" in custom databases); AlphaFold DB ones like "AF-P12345-F1-model_v4".
_PDB_TARGET = re.compile(r"^([0-9][A-Za-z0-9]{3})(?:[-_.][^_]*)?_([A-Za-z0-9]+)")
_AFDB_TARGET = re.compile(r"^AF-([A-Z0-9]+)-F\d+")

_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}


# ── Discovery ────────────────────────────────────────────────────────────────

_cached_bin = None


def find_foldseek(force_rescan: bool = False):
    """
    Locate a working `foldseek` binary: FOLDSEEK_BIN env var, then PATH.
    Verified by running `foldseek version`, not just path existence.

    Returns:
        Path to foldseek, or None.
    """
    global _cached_bin
    if _cached_bin and not force_rescan:
        return _cached_bin

    candidates = [os.getenv("FOLDSEEK_BIN"), shutil.which("foldseek")]
    for exe in candidates:
        if not exe or not Path(exe).exists():
            continue
        try:
            r = subprocess.run([exe, "version"], capture_output=True,
                               text=True, timeout=30)
            if r.returncode == 0:
                _cached_bin = exe
                return exe
        except Exception:
            continue
    return None


def find_database():
    """
    Locate a Foldseek database: FOLDSEEK_DB env var, else DEFAULT_DB.

    A Foldseek database is a path *prefix* ("…/pdb"), not a file — the
    prefix itself plus "<prefix>.dbtype" must exist.

    Returns:
        The database prefix as a string, or None.
    """
    for prefix in (os.getenv("FOLDSEEK_DB"), str(DEFAULT_DB)):
        if prefix and Path(prefix + ".dbtype").exists():
            return prefix
    return None


def availability() -> tuple:
    """
    (ok, message). ok is True only when both the binary and a database are
    present; message says which half is missing and how to get it.
    """
    exe, db = find_foldseek(), find_database()
    if exe and db:
        return True, ""
    missing = []
    if not exe:
        missing.append("the foldseek binary (conda create -n foldseek "
                       "-c conda-forge -c bioconda foldseek, or set FOLDSEEK_BIN)")
    if not db:
        missing.append("a Foldseek reference database (~2.2 GB for PDB: "
                       f"foldseek databases PDB {DEFAULT_DB} /tmp/fs, or set "
                       "FOLDSEEK_DB; `bash setup_tools.sh` does this)")
    return False, ("Structural similarity search unavailable — missing "
                   + " and ".join(missing) + ".")


# ── Query preparation ────────────────────────────────────────────────────────

def _write_query(pdb_path: str, chains, out_path: Path, max_chains: int) -> dict:
    """
    Copy the protein ATOM records of the wanted chains (first model only) to
    out_path. Ligands, waters and nucleic acids are dropped — Foldseek
    searches protein backbones, and a query file named "query.pdb" makes
    every result's query column read "query_<chain>".

    Chains with an identical residue sequence (a homo-oligomer's copies) are
    searched once, through the first of them — same answer, a fraction of
    the search time.

    Returns:
        {representative_chain: [every chain it stands for]}, in file order,
        at most max_chains representatives.
    """
    wanted = set(chains) if chains else None
    by_chain, seqs = {}, {}
    with open(pdb_path, errors="replace") as fh:
        for line in fh:
            rec = line[:6]
            if rec.startswith("ENDMDL"):
                break
            if rec not in ("ATOM  ", "HETATM"):
                continue
            if line[17:20].strip() not in _STANDARD_AA:
                continue
            ch = line[21].strip()
            if not ch or (wanted is not None and ch not in wanted):
                continue
            by_chain.setdefault(ch, []).append(line if line.endswith("\n") else line + "\n")
            if line[12:16].strip() == "CA":
                seqs.setdefault(ch, []).append(line[17:20])

    groups, rep_of_seq = {}, {}
    for ch in by_chain:
        seq = tuple(seqs.get(ch, ()))
        if len(seq) < 10:        # too short for a fold search to mean anything
            continue
        rep = rep_of_seq.get(seq)
        if rep is None:
            if len(groups) >= max_chains:
                continue
            rep_of_seq[seq] = rep = ch
            groups[rep] = []
        groups[rep].append(ch)

    lines = [l for rep in groups for l in by_chain[rep]]
    lines.append("END\n")
    out_path.write_text("".join(lines))
    return groups


# ── Search ───────────────────────────────────────────────────────────────────

def parse_target(target: str) -> dict:
    """
    Split a Foldseek target name into what it points at.

    Returns:
        {"kind": "pdb", "pdb_id": "7ard", "chain": "A"},
        {"kind": "afdb", "accession": "P12345"}, or
        {"kind": "other", "name": target}.
    """
    m = _AFDB_TARGET.match(target)
    if m:
        return {"kind": "afdb", "accession": m.group(1)}
    m = _PDB_TARGET.match(target)
    if m:
        # Assembly files can suffix a copied chain ("A-2"); SIFTS knows it as "A".
        return {"kind": "pdb", "pdb_id": m.group(1).lower(),
                "chain": m.group(2).split("-")[0]}
    return {"kind": "other", "name": target}


def search(pdb_path: str, chains=None, max_hits: int = 10,
           max_evalue: float = 1e-3, exclude_pdb_id: str = "",
           max_chains: int = 8, timeout: int = 900) -> tuple:
    """
    Run `foldseek easy-search` of a structure's protein chains against the
    configured database.

    Args:
        pdb_path  : Local .pdb file.
        chains    : Chain ids to search; None or empty for every protein chain.
        max_hits  : Hits kept per chain after filtering (best E-value first).
        max_evalue: Hits weaker than this are dropped — a weak hit is noise,
                    not evidence of a shared fold.
        exclude_pdb_id: The query's own PDB id, if it has one — its hits
                    against itself are dropped. For an AlphaFold model pass
                    its UniProt accession to drop its own AFDB entry.
        max_chains: Most distinct chains searched in one call — keeps a
                    50-chain complex from turning into a 50-query search.
        timeout   : Seconds before the subprocess is killed.

    Returns:
        (ok, message, hits_by_chain). hits_by_chain is keyed by the
        representative of each distinct-sequence chain group:
        {chain: {"chains": [every chain with that sequence],
                 "hits": [ {target, fident, tmscore, evalue, prob, qstart,
                            qend, qlen, tstart, tend, description,
                            **parse_target(target)} ]}}.
        Never raises — a missing binary/database, a crash or a timeout all
        come back as ok=False with a message.
    """
    ok, msg = availability()
    if not ok:
        return False, msg, {}
    if not Path(pdb_path).exists():
        return False, f"Structure file not found: {pdb_path}", {}

    exe, db = find_foldseek(), find_database()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        query = tmp / "query.pdb"
        groups = _write_query(pdb_path, chains, query, max_chains)
        if not groups:
            return False, "No protein chains (10+ residues) to search in this structure.", {}
        written = list(groups)

        out = tmp / "result.m8"
        cmd = [exe, "easy-search", str(query), db, str(out), str(tmp / "work"),
               "--format-output", ",".join(_FORMAT),
               "-e", str(max_evalue), "-v", "1"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"Foldseek timed out after {timeout}s", {}
        except Exception as e:
            return False, f"Foldseek failed to start: {e}", {}
        if r.returncode != 0 or not out.exists():
            err = (r.stderr or r.stdout or "").strip()[-500:]
            return False, f"Foldseek failed: {err}" if err else "Foldseek failed", {}

        hits = {ch: [] for ch in written}
        for line in out.read_text(errors="replace").splitlines():
            cols = line.split("\t")
            if len(cols) < len(_FORMAT):
                continue
            row = dict(zip(_FORMAT, cols))
            q_chain = row["query"].rsplit("_", 1)[-1] if "_" in row["query"] else ""
            if q_chain not in hits:
                if len(hits) == 1:   # single-chain query: naming may omit the chain
                    q_chain = written[0]
                else:
                    continue
            try:
                hit = {
                    "target": row["target"],
                    "fident": float(row["fident"]),
                    "tmscore": float(row["alntmscore"]),
                    "evalue": float(row["evalue"]),
                    "prob": float(row["prob"]),
                    "qstart": int(row["qstart"]), "qend": int(row["qend"]),
                    "qlen": int(row["qlen"]),
                    "tstart": int(row["tstart"]), "tend": int(row["tend"]),
                    "description": row["theader"].split(" ", 1)[-1].strip()
                                   if " " in row["theader"] else "",
                }
            except ValueError:
                continue
            hit.update(parse_target(row["target"]))
            hits[q_chain].append(hit)

    return True, "", _finalize(hits, groups, exclude_pdb_id, max_hits)


def _finalize(hits: dict, groups: dict, exclude_pdb_id: str, max_hits: int) -> dict:
    """
    Best hit per target entry only (an assembly file or identical chains
    otherwise fill the list with one structure), and never the query's own
    entry — "it resembles itself" is not an answer.
    """
    skip = (exclude_pdb_id or "").strip().lower()
    out = {}
    for ch in groups:
        best = {}
        for h in sorted(hits.get(ch, []), key=lambda h: h["evalue"]):
            key = h.get("pdb_id") or h.get("accession") or h["target"]
            if (skip and key.lower() == skip) or key in best:
                continue
            best[key] = h
        out[ch] = {"chains": groups[ch], "hits": list(best.values())[:max_hits]}
    return out


# ── Online search (search.foldseek.com) ──────────────────────────────────────
#
# Foldseek's public web server (Steinegger lab), no API key. Same databases
# as a local install without the multi-GB download — but it UPLOADS the
# query's coordinates to a third-party server, so app.py only calls this
# after the user has said yes to that in their own words. Confirmed live:
# POST /api/ticket (form fields q, mode, database[]), poll
# GET /api/ticket/{id} until COMPLETE, then GET /api/result/{id}/{i} gives
# one JSON result per query chain, headed "job_<chain>". In 3Di+AA mode the
# alignments carry prob/eval/seqId (percent) but no TM-score.

ONLINE_URL = "https://search.foldseek.com/api"
ONLINE_DB = "pdb100"


def search_online(pdb_path: str, chains=None, max_hits: int = 10,
                  max_evalue: float = 1e-3, exclude_pdb_id: str = "",
                  max_chains: int = 8, timeout: int = 300) -> tuple:
    """
    Same search as search(), run on search.foldseek.com instead of locally.
    Needs no foldseek binary and no database, only network access.

    Returns:
        Same (ok, message, hits_by_chain) shape as search(); "tmscore" is
        None for every hit (the web server's 3Di+AA mode does not report it).
    """
    import time
    import requests

    if not Path(pdb_path).exists():
        return False, f"Structure file not found: {pdb_path}", {}

    with tempfile.TemporaryDirectory() as tmp:
        query = Path(tmp) / "query.pdb"
        groups = _write_query(pdb_path, chains, query, max_chains)
        if not groups:
            return False, "No protein chains (10+ residues) to search in this structure.", {}
        payload = query.read_bytes()

    deadline = time.monotonic() + timeout
    try:
        r = requests.post(f"{ONLINE_URL}/ticket",
                          files={"q": ("query.pdb", payload)},
                          data={"mode": "3diaa", "database[]": [ONLINE_DB]},
                          timeout=60)
        r.raise_for_status()
        ticket = r.json()
        tid, status = ticket["id"], ticket.get("status")
        while status not in ("COMPLETE", "ERROR"):
            if time.monotonic() > deadline:
                return False, f"Online Foldseek search still queued after {timeout}s — try again later.", {}
            time.sleep(3)
            status = requests.get(f"{ONLINE_URL}/ticket/{tid}", timeout=30).json().get("status")
        if status == "ERROR":
            return False, "Online Foldseek search failed on the server.", {}

        hits = {ch: [] for ch in groups}
        for i in range(len(groups)):
            res = requests.get(f"{ONLINE_URL}/result/{tid}/{i}", timeout=60).json()
            header = ((res.get("queries") or [{}])[0].get("header") or "").split(" ", 1)[0]
            q_chain = header.rsplit("_", 1)[-1] if "_" in header else list(groups)[i]
            if q_chain not in hits:
                continue
            for db_res in res.get("results") or []:
                for block in db_res.get("alignments") or []:
                    for a in (block if isinstance(block, list) else [block]):
                        if float(a.get("eval", 1e9)) > max_evalue:
                            continue
                        target = a.get("target", "")
                        name, _, desc = target.partition(" ")
                        hit = {
                            "target": name,
                            "fident": float(a.get("seqId", 0)) / 100.0,
                            "tmscore": None,
                            "evalue": float(a["eval"]),
                            "prob": float(a.get("prob", 0)),
                            "qstart": int(a.get("qStartPos", 0)),
                            "qend": int(a.get("qEndPos", 0)),
                            "qlen": int(a.get("qLen", 0)),
                            "tstart": int(a.get("dbStartPos", 0)),
                            "tend": int(a.get("dbEndPos", 0)),
                            "description": desc.strip(),
                        }
                        hit.update(parse_target(name))
                        hits[q_chain].append(hit)
    except (requests.RequestException, ValueError, KeyError) as e:
        return False, f"Online Foldseek search failed: {e}", {}

    return True, "", _finalize(hits, groups, exclude_pdb_id, max_hits)


# ── Local database download (only ever started on the user's say-so) ────────

DOWNLOAD_GB = 2.2      # compressed download, PDB database
DISK_GB = 4.2          # unpacked on disk
_MIN_FREE_GB = 6.0     # download + unpack headroom


def download_status(prefix: str = str(DEFAULT_DB)) -> tuple:
    """
    ("ready" | "running" | "failed" | "none", detail) for the local PDB
    database at prefix.
    """
    d = Path(prefix).parent
    if Path(prefix + ".dbtype").exists():
        return "ready", prefix
    pid_file, exit_file = d / ".download_pid", d / ".download_exit"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), 0)
            return "running", str(d / "download.log")
        except (OSError, ValueError):
            pass
    if exit_file.exists() and exit_file.read_text().strip() not in ("", "0"):
        log = d / "download.log"
        tail = log.read_text(errors="replace")[-300:] if log.exists() else ""
        return "failed", tail
    return "none", ""


def start_download(prefix: str = str(DEFAULT_DB)) -> tuple:
    """
    Start `foldseek databases PDB` in the background, detached from the
    Streamlit process so a rerun does not kill it. Downloads into a staging
    directory and moves the files into place only on success, so a half-
    finished download is never mistaken for a usable database.

    Returns:
        (ok, message).
    """
    exe = find_foldseek()
    if not exe:
        return False, ("The foldseek binary is needed to build a local database: "
                       "conda create -n foldseek -c conda-forge -c bioconda foldseek")
    state, _ = download_status(prefix)
    if state == "ready":
        return True, f"Local Foldseek database already present at {prefix}."
    if state == "running":
        return True, "Local Foldseek database download already in progress."
    d = Path(prefix).parent
    d.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(d).free / 1e9
    if free_gb < _MIN_FREE_GB:
        return False, (f"Only {free_gb:.1f} GB free at {d}; the PDB database needs "
                       f"~{_MIN_FREE_GB:.0f} GB to download and unpack.")
    name = Path(prefix).name
    stage, work = d / ".staging", d / ".work"
    for f in (d / ".download_exit",):
        f.unlink(missing_ok=True)
    script = (
        'rm -rf "$STAGE" "$WORK" && mkdir -p "$STAGE" && '
        '"$FS" databases PDB "$STAGE/$NAME" "$WORK" && '
        'mv "$STAGE"/* "$DIR"/; rc=$?; rm -rf "$STAGE" "$WORK"; '
        'echo $rc > "$DIR/.download_exit"; rm -f "$DIR/.download_pid"'
    )
    env = dict(os.environ, FS=exe, STAGE=str(stage), WORK=str(work),
               NAME=name, DIR=str(d))
    with open(d / "download.log", "w") as log:
        p = subprocess.Popen(["bash", "-c", script], stdout=log, stderr=log,
                             env=env, start_new_session=True)
    (d / ".download_pid").write_text(str(p.pid))
    return True, (f"Started downloading the Foldseek PDB database (~{DOWNLOAD_GB} GB "
                  f"download, ~{DISK_GB} GB on disk) to {d}. It runs in the "
                  "background, usually a few minutes on a fast connection.")
