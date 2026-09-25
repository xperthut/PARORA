#!/bin/bash
# Check this machine for the OPTIONAL tools app.py's agent can use --
# AmberTools, PyMOL, DSSP and Foldseek (+ its reference database) -- and
# offer to install whichever are missing.
#
# None of these are required to run the app (run.sh already runs fine
# without any of them; the tools that need them just report "unavailable").
# This script exists for someone who wants the fuller feature set and would
# rather be asked than dig the install commands out of run.sh/requirements.txt
# themselves. It only ever touches its own named conda envs (ambertools,
# pymol-render, dssp, foldseek) and protein-viz-agent/foldseek_db/ (or
# $FOLDSEEK_DB) -- nothing here is required for those names, run.sh's
# existing discovery logic already looks for exactly these.
#
# Usage:
#   bash setup_tools.sh              # check each tool, prompt before installing
#   bash setup_tools.sh --yes        # check each tool, install whatever is missing, no prompts
#   bash setup_tools.sh --check-only # report status only, never install, never prompt
set -u

cd "$(dirname "$0")"

ASSUME_YES=false
CHECK_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --yes|-y) ASSUME_YES=true ;;
        --check-only) CHECK_ONLY=true ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (see --help)"
            exit 1
            ;;
    esac
done

# ── Locate conda ──────────────────────────────────────────────────────────────
# Same candidate list as run.sh's conda discovery, kept as its own copy for
# the same reason run.sh/deploy.sh/ollama.sh don't share code with each
# other -- each entry point (script) here is independent on purpose.
CONDA_ROOT=""
for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
                 "/opt/homebrew/Caskroom/miniconda/base" "/opt/anaconda3" "/opt/miniconda3"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_ROOT="$candidate"
        break
    fi
done

if [ -z "$CONDA_ROOT" ]; then
    echo "Could not find a conda installation. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo "All three tools below are installed as conda environments, so none"
    echo "of this script can do anything without conda."
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
echo "Using conda at $CONDA_ROOT"
echo

confirm() {
    # $1 = prompt text. Returns 0 (install) or 1 (skip).
    if $CHECK_ONLY; then
        return 1
    fi
    if $ASSUME_YES; then
        return 0
    fi
    if [ ! -t 0 ]; then
        echo "  (non-interactive, no terminal to prompt on -- skipping. Use --yes to install without asking.)"
        return 1
    fi
    local reply
    read -r -p "  $1 [y/N] " reply
    case "$reply" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

STATUS_AMBERTOOLS="not checked"
STATUS_PYMOL="not checked"
STATUS_DSSP="not checked"
STATUS_FOLDSEEK="not checked"
STATUS_FOLDSEEK_DB="not checked"

# ── 1. AmberTools ────────────────────────────────────────────────────────────
# Needed by: prepare_structure (hydrogens via reduce), build_membrane,
# simulation/quantum/oniom setup. Matches run.sh's own cross-conda-root
# matching (an env can live under a different conda root than the one this
# script found and still be picked up by name or path basename).
echo "── AmberTools ──────────────────────────────────────────────────────────"
AMBERTOOLS_ENV="ambertools"
AMBER_ENV_PATH=$(conda env list | awk -v e="$AMBERTOOLS_ENV" \
    '{n=split($NF,parts,"/")} $1==e || parts[n]==e {print $NF; exit}')
if [ -n "$AMBER_ENV_PATH" ] && [ -x "$AMBER_ENV_PATH/bin/packmol-memgen" ]; then
    echo "Found -- $AMBER_ENV_PATH"
    STATUS_AMBERTOOLS="found ($AMBER_ENV_PATH)"
else
    echo "Not found. Used by: prepare_structure (hydrogens), build_membrane,"
    echo "simulation/quantum/oniom tool setup."
    if confirm "Install AmberTools now? This downloads a conda-forge package (can take several minutes)."; then
        if conda create -n "$AMBERTOOLS_ENV" --override-channels -c conda-forge ambertools -y; then
            STATUS_AMBERTOOLS="installed"
            echo "AmberTools installed."
        else
            STATUS_AMBERTOOLS="install failed"
            echo "AmberTools install failed -- see the conda output above."
        fi
    else
        STATUS_AMBERTOOLS="skipped"
        echo "Skipped."
    fi
fi
echo

# ── 2. PyMOL ─────────────────────────────────────────────────────────────────
# Needed by: render_image (ray-traced publication figures). Checks every env
# whose name mentions pymol, not just the first match -- a name match alone
# doesn't guarantee `import pymol2` actually works (found on this machine
# during testing: one such env had the name but not the package).
echo "── PyMOL ────────────────────────────────────────────────────────────────"
PYMOL_FOUND=""
while IFS= read -r envpath; do
    [ -z "$envpath" ] && continue
    if [ -x "$envpath/bin/python" ] && "$envpath/bin/python" -c "import pymol2" >/dev/null 2>&1; then
        PYMOL_FOUND="$envpath"
        break
    fi
done < <(conda env list | awk '$1 ~ /pymol/ {print $NF}')

if [ -n "$PYMOL_FOUND" ]; then
    echo "Found -- $PYMOL_FOUND (import pymol2 works)"
    STATUS_PYMOL="found ($PYMOL_FOUND)"
else
    echo "Not found (or a pymol-named env exists but can't import pymol2)."
    echo "Used by: render_image (PyMOL ray-traced figures)."
    if confirm "Install PyMOL (open-source build) now? This downloads a conda-forge package."; then
        if conda create -n pymol-render -c conda-forge pymol-open-source -y; then
            STATUS_PYMOL="installed"
            echo "PyMOL installed."
        else
            STATUS_PYMOL="install failed"
            echo "PyMOL install failed -- see the conda output above."
        fi
    else
        STATUS_PYMOL="skipped"
        echo "Skipped."
    fi
fi
echo

# ── 3. DSSP ──────────────────────────────────────────────────────────────────
# Needed by: describe_fold's computed-topology half (its CATH/SCOP half is a
# plain HTTP call and needs nothing here). conda-forge's dssp 4.x has been
# observed to segfault unpredictably on at least one arm64 macOS machine
# (every input, including DSSP's own bundled test file); dssp=3.1.4 was
# stable there. Rather than assume that's true on every machine, this
# installs the current package first and smoke-tests it for real, only
# falling back to 3.x if the smoke test actually fails here.
echo "── DSSP ─────────────────────────────────────────────────────────────────"
dssp_bin=""
if [ -n "${DSSP_BIN:-}" ] && [ -x "${DSSP_BIN:-}" ]; then
    dssp_bin="$DSSP_BIN"
elif command -v mkdssp >/dev/null 2>&1; then
    dssp_bin="$(command -v mkdssp)"
else
    DSSP_ENV_PATH=$(conda env list | awk '$1 ~ /dssp/ {print $NF; exit}')
    if [ -n "$DSSP_ENV_PATH" ] && [ -x "$DSSP_ENV_PATH/bin/mkdssp" ]; then
        dssp_bin="$DSSP_ENV_PATH/bin/mkdssp"
    fi
fi

if [ -n "$dssp_bin" ] && "$dssp_bin" --version >/dev/null 2>&1; then
    echo "Found -- $dssp_bin ($("$dssp_bin" --version 2>&1 | head -1))"
    STATUS_DSSP="found ($dssp_bin)"
elif [ -n "$dssp_bin" ]; then
    echo "Found a binary at $dssp_bin but it fails to even run \`--version\` --"
    echo "broken install, not just missing."
    if confirm "Reinstall DSSP as dssp=3.1.4 (the version known to work) now?"; then
        if conda install -n dssp -c conda-forge "dssp=3" -y 2>/dev/null \
           || conda create -n dssp -c conda-forge "dssp=3" -y; then
            STATUS_DSSP="reinstalled as 3.x"
            echo "DSSP reinstalled as 3.x."
        else
            STATUS_DSSP="install failed"
            echo "DSSP install failed -- see the conda output above."
        fi
    else
        STATUS_DSSP="skipped (broken)"
        echo "Skipped."
    fi
else
    echo "Not found. Used by: describe_fold's computed secondary-structure"
    echo "topology (its CATH/SCOP classification lookup still works without it)."
    if confirm "Install DSSP now? This downloads a conda-forge package."; then
        if conda create -n dssp -c conda-forge dssp -y; then
            if "$CONDA_ROOT/envs/dssp/bin/mkdssp" --version >/dev/null 2>&1; then
                STATUS_DSSP="installed (4.x, smoke test passed)"
                echo "DSSP installed and smoke-tested OK."
            else
                echo "DSSP 4.x installed but fails to run \`--version\` on this machine"
                echo "(matches the known arm64 macOS segfault) -- falling back to 3.x."
                if conda install -n dssp -c conda-forge "dssp=3" -y; then
                    STATUS_DSSP="installed (fell back to 3.x)"
                    echo "DSSP 3.x installed."
                else
                    STATUS_DSSP="install failed"
                    echo "DSSP 3.x fallback install failed -- see the conda output above."
                fi
            fi
        else
            STATUS_DSSP="install failed"
            echo "DSSP install failed -- see the conda output above."
        fi
    else
        STATUS_DSSP="skipped"
        echo "Skipped."
    fi
fi
echo

# ── 4. Foldseek (binary + reference database) ───────────────────────────────
# Needed by: find_structural_neighbors, and describe_fold's fallback for
# chains with no CATH/SCOP classification of their own (AlphaFold models,
# local files, recent entries). Two separate pieces: the binary (small conda
# package) and a reference database to search against. The PDB database is
# ~2.2 GB to download, ~4.2 GB unpacked -- asked for separately and never
# fetched silently.
echo "── Foldseek ─────────────────────────────────────────────────────────────"
FOLDSEEK_DB_PREFIX="${FOLDSEEK_DB:-$PWD/protein-viz-agent/foldseek_db/pdb}"
fs_bin=""
if [ -n "${FOLDSEEK_BIN:-}" ] && [ -x "${FOLDSEEK_BIN:-}" ]; then
    fs_bin="$FOLDSEEK_BIN"
elif command -v foldseek >/dev/null 2>&1; then
    fs_bin="$(command -v foldseek)"
else
    FS_ENV_PATH=$(conda env list | awk '$1 ~ /foldseek/ {print $NF; exit}')
    if [ -n "$FS_ENV_PATH" ] && [ -x "$FS_ENV_PATH/bin/foldseek" ]; then
        fs_bin="$FS_ENV_PATH/bin/foldseek"
    fi
fi

if [ -n "$fs_bin" ] && "$fs_bin" version >/dev/null 2>&1; then
    echo "Binary found -- $fs_bin ($("$fs_bin" version 2>&1 | head -1))"
    STATUS_FOLDSEEK="found ($fs_bin)"
else
    echo "Binary not found. Used by: find_structural_neighbors, and describe_fold's"
    echo "fallback for structures with no CATH/SCOP classification of their own."
    if confirm "Install Foldseek now? This downloads a small conda-forge/bioconda package."; then
        if conda create -n foldseek -c conda-forge -c bioconda foldseek -y; then
            fs_bin="$CONDA_ROOT/envs/foldseek/bin/foldseek"
            STATUS_FOLDSEEK="installed"
            echo "Foldseek installed."
        else
            STATUS_FOLDSEEK="install failed"
            echo "Foldseek install failed -- see the conda output above."
        fi
    else
        STATUS_FOLDSEEK="skipped"
        echo "Skipped."
    fi
fi

if [ -f "$FOLDSEEK_DB_PREFIX.dbtype" ]; then
    echo "Database found -- $FOLDSEEK_DB_PREFIX"
    STATUS_FOLDSEEK_DB="found ($FOLDSEEK_DB_PREFIX)"
elif [ -n "$fs_bin" ] && [ -x "$fs_bin" ]; then
    echo "Reference database not found at $FOLDSEEK_DB_PREFIX."
    if confirm "Download the Foldseek PDB database now? ~2.2 GB download, ~4.2 GB on disk."; then
        mkdir -p "$(dirname "$FOLDSEEK_DB_PREFIX")"
        fs_tmp="$(mktemp -d)"
        if "$fs_bin" databases PDB "$FOLDSEEK_DB_PREFIX" "$fs_tmp"; then
            STATUS_FOLDSEEK_DB="downloaded ($FOLDSEEK_DB_PREFIX)"
            echo "Foldseek PDB database ready."
        else
            STATUS_FOLDSEEK_DB="download failed"
            echo "Database download failed -- see the output above."
        fi
        rm -rf "$fs_tmp"
    else
        STATUS_FOLDSEEK_DB="skipped"
        echo "Skipped. Later: foldseek databases PDB $FOLDSEEK_DB_PREFIX /tmp/fs"
    fi
else
    STATUS_FOLDSEEK_DB="not checked (no binary)"
fi
echo

# ── Summary ──────────────────────────────────────────────────────────────────
echo "── Summary ──────────────────────────────────────────────────────────────"
printf "  %-12s %s\n" "AmberTools" "$STATUS_AMBERTOOLS"
printf "  %-12s %s\n" "PyMOL" "$STATUS_PYMOL"
printf "  %-12s %s\n" "DSSP" "$STATUS_DSSP"
printf "  %-12s %s\n" "Foldseek" "$STATUS_FOLDSEEK"
printf "  %-12s %s\n" "Foldseek DB" "$STATUS_FOLDSEEK_DB"
echo
echo "Nothing further to do -- run.sh's own discovery already looks for these"
echo "exact conda env names (ambertools, a *pymol* name, dssp, foldseek) every time it"
echo "starts the app, so a freshly installed tool is picked up automatically"
echo "on the next \`bash run.sh\`."
