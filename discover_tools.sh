#!/bin/bash
# Optional-tool discovery (AmberTools, DSSP, Foldseek, fpocket, PyMOL, ESM-2):
# exports PACKMOL_MEMGEN, DSSP_BIN, FOLDSEEK_BIN/DB/AFDB, FPOCKET_BIN,
# PYMOL_PYTHON and ESM_PYTHON for whatever it finds. Sourced by run.sh, and by
# evals/run.py (which imports the exported variables) so the eval harness sees
# the same backends the app does. Variables already set are left alone.
# Installs nothing -- that is setup_tools.sh.

_PARORA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMBERTOOLS_ENV="${AMBERTOOLS_ENV:-ambertools}"

# Sourced from run.sh, conda is already set up; on its own, find conda.sh the
# same way run.sh does.
if ! type conda >/dev/null 2>&1; then
    for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
                     "/opt/homebrew/Caskroom/miniconda/base" "/opt/anaconda3" "/opt/miniconda3"; do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$candidate/etc/profile.d/conda.sh"
            break
        fi
    done
fi

# ── 3. AmberTools discovery (optional: prepare_structure / build_membrane / simulation, quantum, oniom) ──
# `conda env list` only prints a name for envs under the currently-preferred
# root; an env living under a *different* conda root (as ambertools does on
# this machine) is listed by path only, with no name column -- so match by
# either the name column or the path's own basename.
AMBER_ENV_PATH=$(conda env list | awk -v e="$AMBERTOOLS_ENV" \
    '{n=split($NF,parts,"/")} $1==e || parts[n]==e {print $NF; exit}')
if [ -n "$AMBER_ENV_PATH" ] && [ -x "$AMBER_ENV_PATH/bin/packmol-memgen" ]; then
    export PACKMOL_MEMGEN="$AMBER_ENV_PATH/bin/packmol-memgen"
    echo "AmberTools found -- PACKMOL_MEMGEN=$PACKMOL_MEMGEN"
else
    echo "AmberTools env '$AMBERTOOLS_ENV' not found -- prepare/membrane/simulation/QM tools will report"
    echo "unavailable rather than fail. Set it up with:"
    echo "  conda create -n $AMBERTOOLS_ENV --override-channels -c conda-forge ambertools -y"
    echo "or run 'bash setup_tools.sh' to be walked through installing this (and PyMOL/DSSP) interactively."
fi

# ── 3.5. DSSP discovery (optional: describe_fold's computed topology) ──────────
# mkdssp is a single static binary, so this is a plain PATH/conda-env-name scan
# like AmberTools' packmol-memgen, not the whole-interpreter scan PyMOL needs.
if [ -z "${DSSP_BIN:-}" ]; then
    if command -v mkdssp >/dev/null 2>&1; then
        export DSSP_BIN="$(command -v mkdssp)"
    else
        DSSP_ENV_PATH=$(conda env list | awk '$1 ~ /dssp/ {print $NF; exit}')
        if [ -n "$DSSP_ENV_PATH" ] && [ -x "$DSSP_ENV_PATH/bin/mkdssp" ]; then
            export DSSP_BIN="$DSSP_ENV_PATH/bin/mkdssp"
        fi
    fi
fi
if [ -n "${DSSP_BIN:-}" ]; then
    echo "DSSP found -- DSSP_BIN=$DSSP_BIN"
else
    echo "DSSP not found -- describe_fold's computed topology will report"
    echo "unavailable rather than fail (its CATH/SCOP lookup is unaffected, since"
    echo "that's a network call, not a DSSP one). Set up with:"
    echo "  conda create -n dssp -c conda-forge dssp"
    echo "  NOTE: conda-forge's dssp 4.x segfaults unpredictably on at least one"
    echo "  arm64 macOS machine this was tested on; dssp=3.1.4 was stable there"
    echo "  (conda install -n dssp -c conda-forge \"dssp=3\" if 4.x misbehaves)."
    echo "or run 'bash setup_tools.sh', which tries 4.x and falls back to 3.x for you"
    echo "if the smoke test fails on this machine."
fi

# ── 3.6. Foldseek discovery (optional: find_structural_neighbors, describe_fold fallback) ──
# Binary: FOLDSEEK_BIN, else PATH, else a conda env whose name mentions
# foldseek. Database: FOLDSEEK_DB, else protein-viz-agent/foldseek_db/pdb.
# The database is never downloaded from here -- ~2.2 GB is the user's call.
if [ -z "${FOLDSEEK_BIN:-}" ]; then
    if command -v foldseek >/dev/null 2>&1; then
        export FOLDSEEK_BIN="$(command -v foldseek)"
    else
        FS_ENV_PATH=$(conda env list | awk '$1 ~ /foldseek/ {print $NF; exit}')
        if [ -n "$FS_ENV_PATH" ] && [ -x "$FS_ENV_PATH/bin/foldseek" ]; then
            export FOLDSEEK_BIN="$FS_ENV_PATH/bin/foldseek"
        fi
    fi
fi
FS_DB="${FOLDSEEK_DB:-$_PARORA_ROOT/protein-viz-agent/foldseek_db/pdb}"
if [ -n "${FOLDSEEK_BIN:-}" ] && [ -f "$FS_DB.dbtype" ]; then
    export FOLDSEEK_DB="$FS_DB"
    echo "Foldseek found -- FOLDSEEK_BIN=$FOLDSEEK_BIN, FOLDSEEK_DB=$FOLDSEEK_DB"
else
    echo "Foldseek structural search not ready -- find_structural_neighbors will report"
    echo "unavailable rather than fail (describe_fold is otherwise unaffected)."
    [ -z "${FOLDSEEK_BIN:-}" ] && echo "  binary:   conda create -n foldseek -c conda-forge -c bioconda foldseek"
    [ -f "$FS_DB.dbtype" ] || echo "  database: foldseek databases PDB $FS_DB /tmp/fs   (~2.2 GB download, ~4.2 GB on disk)"
    echo "or run 'bash setup_tools.sh', which asks before installing or downloading either."
    echo "(The app itself also asks, the first time it's needed: search online at"
    echo "search.foldseek.com, which uploads the structure, or download the database.)"
fi
# Optional AlphaFold DB (Swiss-Prot) database: FOLDSEEK_AFDB, else
# protein-viz-agent/foldseek_db/afdb_swissprot. Never downloaded from here.
FS_AFDB="${FOLDSEEK_AFDB:-$_PARORA_ROOT/protein-viz-agent/foldseek_db/afdb_swissprot}"
if [ -f "$FS_AFDB.dbtype" ]; then
    export FOLDSEEK_AFDB="$FS_AFDB"
    echo "Foldseek AlphaFold DB (Swiss-Prot) found -- FOLDSEEK_AFDB=$FOLDSEEK_AFDB"
elif [ -n "${FOLDSEEK_BIN:-}" ]; then
    echo "Optional: AlphaFold DB (Swiss-Prot) Foldseek database not installed -- chains with no"
    echo "  PDB match get no predicted-model neighbours. foldseek databases Alphafold/Swiss-Prot"
    echo "  $FS_AFDB /tmp/fs   (~1.6 GB download, ~2.4 GB on disk)"
fi

# ── 3.7. fpocket discovery (optional: find_pockets) ────────────────────────────
# Single binary, same PATH/conda-env-name scan as DSSP.
if [ -z "${FPOCKET_BIN:-}" ]; then
    if command -v fpocket >/dev/null 2>&1; then
        export FPOCKET_BIN="$(command -v fpocket)"
    else
        FP_ENV_PATH=$(conda env list | awk '$1 ~ /fpocket/ {print $NF; exit}')
        if [ -n "$FP_ENV_PATH" ] && [ -x "$FP_ENV_PATH/bin/fpocket" ]; then
            export FPOCKET_BIN="$FP_ENV_PATH/bin/fpocket"
        fi
    fi
fi
if [ -n "${FPOCKET_BIN:-}" ]; then
    echo "fpocket found -- FPOCKET_BIN=$FPOCKET_BIN"
else
    echo "fpocket not found -- find_pockets (binding-pocket detection) will report"
    echo "unavailable rather than fail. Set up with:"
    echo "  conda create -n fpocket -c conda-forge fpocket"
    echo "or run 'bash setup_tools.sh'."
fi

# ── 4. PyMOL discovery (optional: render_image) ────────────────────────────────
PYMOL_ENV_PATH=$(conda env list | awk '$1 ~ /pymol/ {print $NF; exit}')
PYMOL_FOUND=false
if [ -z "${PYMOL_PYTHON:-}" ] && [ -n "$PYMOL_ENV_PATH" ] && [ -x "$PYMOL_ENV_PATH/bin/python" ]; then
    if "$PYMOL_ENV_PATH/bin/python" -c "import pymol2" >/dev/null 2>&1; then
        export PYMOL_PYTHON="$PYMOL_ENV_PATH/bin/python"
        echo "PyMOL found -- PYMOL_PYTHON=$PYMOL_PYTHON"
        PYMOL_FOUND=true
    fi
elif [ -n "${PYMOL_PYTHON:-}" ]; then
    PYMOL_FOUND=true
fi
if ! $PYMOL_FOUND; then
    echo "PyMOL not found -- render_image will report unavailable rather than fail."
    echo "Set it up with:"
    echo "  conda create -n pymol-render -c conda-forge pymol-open-source -y"
    echo "or run 'bash setup_tools.sh' to be walked through installing this (and AmberTools/DSSP) interactively."
fi

# ── 4.5. ESM-2 discovery (optional: predict_mutation_effect) ─────────────────
# esm_tools.py discovers this itself on first use; exporting it here just
# saves that probe and tells the user up front.
if [ -z "${ESM_PYTHON:-}" ]; then
    while IFS= read -r envpath; do
        [ -z "$envpath" ] && continue
        if [ -x "$envpath/bin/python" ] && "$envpath/bin/python" -c "import torch, transformers" >/dev/null 2>&1; then
            export ESM_PYTHON="$envpath/bin/python"
            break
        fi
    done < <(conda env list | awk '$1 ~ /esm|torch/ {print $NF}')
fi
if [ -n "${ESM_PYTHON:-}" ]; then
    echo "ESM-2 env found -- ESM_PYTHON=$ESM_PYTHON (model is checked on first use)"
else
    echo "ESM-2 not found -- predict_mutation_effect will report unavailable rather than fail."
    echo "Run 'bash setup_tools.sh' to create the env and download the model."
fi
