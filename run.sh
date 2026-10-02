#!/bin/bash
# PARORA — one command to set up and launch the full agent (app.py):
#
#     bash run.sh
#
# No options needed. Every launch checks everything the app uses and installs
# whatever is missing, without asking:
#   required  conda (Miniconda), the `parora` env + Python packages, Ollama
#             (installed and started), the model config.yaml names for app.py
#   optional  AmberTools, PyMOL, DSSP, fpocket, Foldseek + its PDB and
#             AlphaFold databases, ESM-2 (torch env + model checkpoint)
# The first run downloads ~15-20 GB and can take 30-60 minutes; later runs
# only check (a few seconds) and start the app. A missing optional tool never
# stops the launch — that feature reports "unavailable" inside the app.
# Install output goes to protein-viz-agent/logs/setup.log.
#
# Extras (never needed):  bash run.sh --check     report only, change nothing
#                         bash run.sh --port 8502 use another port
set -u

cd "$(dirname "$0")"
ROOT="$(pwd)"
APP_DIR="$ROOT/protein-viz-agent"
ENV_NAME="parora"
CHECK_ONLY=false
PORT=8501
while [ $# -gt 0 ]; do
    case "$1" in
        --check)   CHECK_ONLY=true ;;
        --port)    shift; PORT="${1:-}" ;;
        --port=*)  PORT="${1#--port=}" ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *)         echo "Unknown option: $1 — just run: bash run.sh"; exit 1 ;;
    esac
    shift
done
case "$PORT" in ''|*[!0-9]*) echo "--port needs a number"; exit 1 ;; esac

mkdir -p "$APP_DIR/logs"
SETUP_LOG="$APP_DIR/logs/setup.log"

# ── Output helpers ───────────────────────────────────────────────────────────
if [ -t 1 ]; then B=$'\e[1m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; N=$'\e[0m'
else B=""; G=""; Y=""; R=""; N=""; fi
SUMMARY=()          # "status|item|detail"
PROBLEMS=()         # required pieces still missing, with the manual fix
section() { echo; echo "${B}── $1 ${N}"; }
ok()      { echo "  ${G}✓${N} $1"; }
warn()    { echo "  ${Y}!${N} $1"; }
bad()     { echo "  ${R}✗${N} $1"; }
record()  { SUMMARY+=("$1|$2|$3"); }
problem() { PROBLEMS+=("$1"); }

# do_install "what" cmd... — runs one install step, output to the screen and
# setup.log; returns the command's own status. --check never installs.
do_install() {
    local what="$1"; shift
    if $CHECK_ONLY; then echo "    (--check: would install $what)"; return 1; fi
    echo "  → installing $what …"
    { echo; echo "=== $(date '+%F %T') $what: $*"; } >> "$SETUP_LOG"
    "$@" 2>&1 | tee -a "$SETUP_LOG"
    return "${PIPESTATUS[0]}"
}

free_gb() { df -Pk "$ROOT" | awk 'NR==2 {print int($4/1048576)}'; }
# room_for "what" GB — enough free disk for a large download (+5 GB spare)?
room_for() {
    local f; f=$(free_gb)
    if [ "$f" -lt $(( $2 + 5 )) ]; then
        warn "Not enough disk for $1 (needs ~$2 GB, ${f} GB free) — skipped"
        return 1
    fi
}

finish() {
    section "Summary"
    local row st item detail mark
    for row in "${SUMMARY[@]}"; do
        IFS='|' read -r st item detail <<<"$row"
        case "$st" in ok) mark="${G}✓${N}" ;; warn) mark="${Y}!${N}" ;; *) mark="${R}✗${N}" ;; esac
        printf "  %s %-16s %s\n" "$mark" "$item" "$detail"
    done
    if [ ${#PROBLEMS[@]} -gt 0 ]; then
        echo
        echo "${R}${B}PARORA cannot start yet.${N} Automatic setup could not fix:"
        local p; for p in "${PROBLEMS[@]}"; do echo "  • $p"; done
        echo "Details: protein-viz-agent/logs/setup.log. Then run again: bash run.sh"
        exit 1
    fi
}

OS="$(uname -s)"; ARCH="$(uname -m)"

# ═════════════════════════════════════════════════════════════════════════════
# REQUIRED
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. Platform ──────────────────────────────────────────────────────────────
section "Platform"
case "$OS" in
    Darwin) ok "macOS ($ARCH)" ;;
    Linux)  ok "Linux ($ARCH)" ;;
    *)      bad "$OS is not supported (macOS or Linux only)"; exit 1 ;;
esac
command -v curl >/dev/null 2>&1 || { bad "curl not found — install curl first"; exit 1; }
if [ "$OS" = Darwin ]; then
    RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
else
    RAM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
fi
ok "RAM ${RAM_GB} GB, $(free_gb) GB free disk"
if curl -s -o /dev/null --max-time 5 https://pypi.org; then
    ok "Internet reachable"
else
    warn "No internet — anything missing cannot be installed this run"
fi
record ok "Platform" "$OS $ARCH, ${RAM_GB} GB RAM, $(free_gb) GB free"

# ── 2. Conda ─────────────────────────────────────────────────────────────────
# A script does not read ~/.zshrc, and with two conda installs a bare `conda`
# can resolve to the wrong one — so source a real conda.sh explicitly.
section "Conda"
find_conda() {
    CONDA_ROOT=""
    local c
    for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
             "/opt/homebrew/Caskroom/miniconda/base" "/opt/anaconda3" "/opt/miniconda3"; do
        if [ -f "$c/etc/profile.d/conda.sh" ]; then CONDA_ROOT="$c"; return 0; fi
    done
    return 1
}
if ! find_conda; then
    bad "No conda installation found"
    case "$OS-$ARCH" in
        Darwin-arm64)  MC="MacOSX-arm64" ;;
        Darwin-x86_64) MC="MacOSX-x86_64" ;;
        Linux-x86_64)  MC="Linux-x86_64" ;;
        Linux-aarch64|Linux-arm64) MC="Linux-aarch64" ;;
        *) MC="" ;;
    esac
    if [ -n "$MC" ]; then
        tmp="$(mktemp -d)"
        do_install "Miniconda (~/miniconda3)" sh -c \
            "curl -fsSL -o '$tmp/mc.sh' https://repo.anaconda.com/miniconda/Miniconda3-latest-$MC.sh \
             && bash '$tmp/mc.sh' -b -p '$HOME/miniconda3'"
        rm -rf "$tmp"
        find_conda
    fi
fi
if [ -z "$CONDA_ROOT" ]; then
    record fail "Conda" "not installed"
    problem "Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    finish
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
ok "conda at $CONDA_ROOT"
record ok "Conda" "$CONDA_ROOT"

# Every env created here uses conda-forge only (--override-channels): a fresh
# Miniconda refuses the default channels until their terms are accepted.
conda_env_create() {    # conda_env_create NAME CHANNEL-ARGS... PACKAGES...
    local name="$1"; shift
    do_install "conda env '$name'" conda create -y -n "$name" --override-channels "$@"
}

# ── 3. Python environment ────────────────────────────────────────────────────
# Its binaries are called by absolute path: name-based `conda run -n` has
# picked the wrong root's envs/ on a machine with two conda installs.
section "Python environment '$ENV_NAME'"
ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
PY="$ENV_PREFIX/bin/python"
if [ ! -x "$PY" ]; then
    bad "Environment '$ENV_NAME' does not exist"
    do_install "conda env '$ENV_NAME' (parora.yml)" conda env create -f parora.yml
fi
if [ ! -x "$PY" ]; then
    record fail "Python env" "missing"
    problem "Create the environment: conda env create -f parora.yml"
    finish
fi
ok "$ENV_PREFIX ($("$PY" --version 2>&1))"
record ok "Python env" "$ENV_NAME ($("$PY" --version 2>&1 | awk '{print $2}'))"

# pip runs only when requirements.txt changed or an import is broken, so a
# normal launch does not wait on it.
REQ="$APP_DIR/requirements.txt"
STAMP="$ENV_PREFIX/.parora_requirements.sha"
REQ_SHA="$(shasum "$REQ" 2>/dev/null | awk '{print $1}')"
IMPORTS="import streamlit, ollama, yaml, requests, numpy, pandas, MDAnalysis, rcsbapi"
imports_ok() { "$PY" -c "$IMPORTS" >/dev/null 2>&1; }
synced()     { [ "$(cat "$STAMP" 2>/dev/null)" = "$REQ_SHA" ]; }
if imports_ok && synced; then
    ok "Python packages up to date"
else
    do_install "Python packages (requirements.txt)" "$ENV_PREFIX/bin/pip" install -q -r "$REQ" \
        && echo "$REQ_SHA" > "$STAMP"
fi
if imports_ok && synced; then
    record ok "Python packages" "requirements.txt installed"
elif imports_ok; then
    record warn "Python packages" "importable; requirements.txt sync pending"
else
    bad "Required Python packages are still missing"
    record fail "Python packages" "missing"
    problem "Install packages: $ENV_PREFIX/bin/pip install -r protein-viz-agent/requirements.txt"
fi

# Ollama host and the app model come from config.yaml (env vars win), the same
# way app.py reads them.
read -r OLLAMA_URL APP_MODEL < <("$PY" - <<EOF 2>/dev/null
import sys; sys.path.insert(0, "$APP_DIR")
from parora_config import get_config
c = get_config("app"); print(c["ollama_host"].rstrip("/"), c["model"])
EOF
)
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"; APP_MODEL="${APP_MODEL:-qwen2.5:14b}"

# ── 4. Ollama ────────────────────────────────────────────────────────────────
section "Ollama ($OLLAMA_URL)"
ollama_up() { curl -s -o /dev/null --max-time 3 "$OLLAMA_URL/api/version"; }
LOCAL_OLLAMA=false
case "$OLLAMA_URL" in *localhost*|*127.0.0.1*) LOCAL_OLLAMA=true ;; esac

# The desktop app and its own CLI first: on macOS the Homebrew `ollama`
# formula has shipped without the llama-server binary, so its `serve` cannot
# run models.
find_ollama() {
    OLLAMA_APP=""; OLLAMA_BIN=""
    local a
    for a in "/Applications/Ollama.app" "$HOME/Applications/Ollama.app"; do
        [ -d "$a" ] && OLLAMA_APP="$a" && break
    done
    if [ -n "$OLLAMA_APP" ] && [ -x "$OLLAMA_APP/Contents/Resources/ollama" ]; then
        OLLAMA_BIN="$OLLAMA_APP/Contents/Resources/ollama"
    elif command -v ollama >/dev/null 2>&1; then
        OLLAMA_BIN="$(command -v ollama)"
    fi
}
find_ollama

if ! ollama_up && $LOCAL_OLLAMA && [ -z "$OLLAMA_APP" ] && [ -z "$OLLAMA_BIN" ]; then
    bad "Ollama is not installed"
    if [ "$OS" = Darwin ]; then
        APPS="/Applications"; [ -w "$APPS" ] || { APPS="$HOME/Applications"; mkdir -p "$APPS"; }
        tmp="$(mktemp -d)"
        do_install "Ollama app ($APPS)" sh -c \
            "curl -fsSL -o '$tmp/ollama.zip' https://ollama.com/download/Ollama-darwin.zip \
             && unzip -q '$tmp/ollama.zip' -d '$APPS'"
        rm -rf "$tmp"
    else
        echo "  (the official installer uses sudo — enter your password if asked)"
        do_install "Ollama (official script)" sh -c "curl -fsSL https://ollama.com/install.sh | sh"
    fi
    find_ollama
fi

if ollama_up; then
    ok "Server running"
elif $LOCAL_OLLAMA && { [ -n "$OLLAMA_APP" ] || [ -n "$OLLAMA_BIN" ]; } && ! $CHECK_ONLY; then
    warn "Not running — starting it"
    if [ -n "$OLLAMA_APP" ]; then
        open -a "$OLLAMA_APP"
    else
        nohup "$OLLAMA_BIN" serve > "$APP_DIR/logs/ollama.log" 2>&1 &
        echo "    (ollama serve in the background, log: protein-viz-agent/logs/ollama.log)"
    fi
    for _ in $(seq 1 45); do ollama_up && break; sleep 1; done
    ollama_up && ok "Server started" || bad "Server did not come up within 45 s"
fi

if ollama_up; then
    VER=$(curl -s --max-time 3 "$OLLAMA_URL/api/version" | sed -n 's/.*"version":"\([^"]*\)".*/\1/p')
    record ok "Ollama" "running at $OLLAMA_URL${VER:+ (v$VER)}"
elif ! $LOCAL_OLLAMA; then
    bad "Cannot reach the Ollama server at $OLLAMA_URL"
    record fail "Ollama" "unreachable at $OLLAMA_URL"
    problem "Start Ollama on that host, or unset OLLAMA_HOST to use a local one"
elif [ -z "$OLLAMA_APP" ] && [ -z "$OLLAMA_BIN" ]; then
    record fail "Ollama" "not installed"
    problem "Install Ollama: https://ollama.com/download"
else
    bad "Ollama is not running"
    record fail "Ollama" "not running"
    if [ -n "$OLLAMA_APP" ]; then problem "Start Ollama: open -a Ollama"
    else problem "Start Ollama: ollama serve"; fi
fi

# ── 5. Model ─────────────────────────────────────────────────────────────────
section "Model"
have_model() {
    curl -s --max-time 5 "$OLLAMA_URL/api/tags" | "$PY" -c "
import json, sys
want = sys.argv[1]
names = {m['name'] for m in json.load(sys.stdin).get('models', [])}
sys.exit(0 if want in names or want + ':latest' in names else 1)" "$1" 2>/dev/null
}
pull_model() {
    if [ -n "$OLLAMA_BIN" ] && $LOCAL_OLLAMA; then
        "$OLLAMA_BIN" pull "$1"
    else    # remote server, or no CLI: the server pulls it itself
        curl -s -N "$OLLAMA_URL/api/pull" -d "{\"name\": \"$1\"}" | "$PY" -c "
import json, sys
last = ''
for line in sys.stdin:
    d = json.loads(line)
    if d.get('error'): print('   ', d['error']); sys.exit(1)
    if d.get('status', '') != last: last = d['status']; print('   ', last, flush=True)"
    fi
}
if ! ollama_up; then
    warn "Skipped — Ollama is not running"
    record fail "Model" "$APP_MODEL (not checked)"
else
    if have_model "$APP_MODEL"; then
        ok "$APP_MODEL"
    else
        bad "$APP_MODEL is not pulled"
        do_install "model $APP_MODEL (several GB)" pull_model "$APP_MODEL"
    fi
    if have_model "$APP_MODEL"; then
        record ok "Model" "$APP_MODEL"
    else
        record fail "Model" "$APP_MODEL missing"
        problem "Pull the model: ollama pull $APP_MODEL (or set PARORA_MODEL_APP to one you have)"
    fi
    case "$APP_MODEL" in
        *14b*) [ "$RAM_GB" -lt 16 ] && warn "$APP_MODEL needs ~12 GB of RAM; with ${RAM_GB} GB use PARORA_MODEL_APP=qwen2.5:7b bash run.sh" ;;
        *32b*) [ "$RAM_GB" -lt 32 ] && warn "$APP_MODEL needs ~22 GB of RAM; with ${RAM_GB} GB use a smaller model" ;;
    esac
fi

# ═════════════════════════════════════════════════════════════════════════════
# OPTIONAL TOOLS — installed when missing; a failure here never blocks launch.
# Env names match what the app (and discover_tools.sh, used by evals/run.py)
# looks for: ambertools, *pymol*, dssp, foldseek, fpocket, *esm*/*torch*.
# ═════════════════════════════════════════════════════════════════════════════
section "Optional tools"
# Envs by name, or by path basename: one living under a *different* conda
# root is listed by `conda env list` with its path only.
CONDA_ENVS="$(conda env list 2>/dev/null)"
env_path() {    # env_path REGEX — every env whose name or folder matches
    printf '%s\n' "$CONDA_ENVS" | awk -v e="$1" \
        '/^#/ || NF == 0 {next} {n = split($NF, p, "/")} $1 ~ e || p[n] ~ e {print $NF}'
}
refresh_envs() { CONDA_ENVS="$(conda env list 2>/dev/null)"; }
FS_DB="${FOLDSEEK_DB:-$APP_DIR/foldseek_db/pdb}"
FS_AFDB="${FOLDSEEK_AFDB:-$APP_DIR/foldseek_db/afdb_swissprot}"
ESM_MODEL_NAME="${ESM_MODEL:-facebook/esm2_t33_650M_UR50D}"

# Each finder sets the variable the app reads, or leaves it empty.
find_amber() {
    PACKMOL_MEMGEN=""
    local p
    for p in $(env_path '^ambertools$'); do
        [ -x "$p/bin/packmol-memgen" ] && { PACKMOL_MEMGEN="$p/bin/packmol-memgen"; return 0; }
    done
    return 1
}
find_pymol() {
    PYMOL_PYTHON=""
    local p
    for p in $(env_path 'pymol'); do
        [ -x "$p/bin/python" ] && "$p/bin/python" -c "import pymol2" >/dev/null 2>&1 \
            && { PYMOL_PYTHON="$p/bin/python"; return 0; }
    done
    return 1
}
# The binary must actually run: conda-forge DSSP 4.x has segfaulted on arm64 macOS.
find_dssp() {
    DSSP_BIN=""
    local c p
    for c in "$(command -v mkdssp 2>/dev/null)" $(for p in $(env_path 'dssp'); do echo "$p/bin/mkdssp"; done); do
        [ -n "$c" ] && [ -x "$c" ] && "$c" --version >/dev/null 2>&1 && { DSSP_BIN="$c"; return 0; }
    done
    return 1
}
find_foldseek() {
    FOLDSEEK_BIN=""
    local c p
    for c in "$(command -v foldseek 2>/dev/null)" $(for p in $(env_path 'foldseek'); do echo "$p/bin/foldseek"; done); do
        [ -n "$c" ] && [ -x "$c" ] && "$c" version >/dev/null 2>&1 && { FOLDSEEK_BIN="$c"; return 0; }
    done
    return 1
}
# fpocket has no --version; run bare it prints a usage banner naming itself.
find_fpocket() {
    FPOCKET_BIN=""
    local c p
    for c in "$(command -v fpocket 2>/dev/null)" $(for p in $(env_path 'fpocket'); do echo "$p/bin/fpocket"; done); do
        [ -n "$c" ] && [ -x "$c" ] && "$c" 2>&1 | grep -qi fpocket && { FPOCKET_BIN="$c"; return 0; }
    done
    return 1
}
find_esm() {
    ESM_PYTHON=""
    local p
    for p in $(env_path 'esm|torch'); do
        [ -x "$p/bin/python" ] && "$p/bin/python" -c "import torch, transformers" >/dev/null 2>&1 \
            && { ESM_PYTHON="$p/bin/python"; return 0; }
    done
    return 1
}
esm_model_cached() {
    [ -n "$ESM_PYTHON" ] && "$ESM_PYTHON" -c "from huggingface_hub import try_to_load_from_cache as t; import sys
sys.exit(0 if all(isinstance(t('$ESM_MODEL_NAME', f), str) for f in ('config.json', 'model.safetensors')) else 1)" \
        >/dev/null 2>&1
}
opt_result() {  # opt_result NAME FOUND-PATH
    if [ -n "$2" ]; then ok "$1 — $2"; record ok "$1" "$2"
    else bad "$1 — not available (that feature reports 'unavailable')"; record warn "$1" "not available"; fi
}

find_amber; find_pymol; find_dssp; find_foldseek; find_fpocket; find_esm
# Say up front when the long first-run downloads are coming.
todo=""
[ -z "$PACKMOL_MEMGEN" ] && todo="$todo AmberTools"
[ -z "$PYMOL_PYTHON" ]   && todo="$todo PyMOL"
[ -z "$DSSP_BIN" ]       && todo="$todo DSSP"
[ -z "$FPOCKET_BIN" ]    && todo="$todo fpocket"
[ -z "$FOLDSEEK_BIN" ]   && todo="$todo Foldseek"
[ -f "$FS_DB.dbtype" ]   || todo="$todo Foldseek-PDB-database(~4 GB)"
[ -f "$FS_AFDB.dbtype" ] || todo="$todo AlphaFold-database(~2.4 GB)"
esm_model_cached         || todo="$todo ESM-2(~5 GB)"
if [ -n "$todo" ] && ! $CHECK_ONLY; then
    warn "To install:$todo — can take a while (up to an hour the first time); log: logs/setup.log"
fi

# AmberTools — prepare_structure (hydrogens), membranes, Amber/QM/ONIOM setup.
if [ -z "$PACKMOL_MEMGEN" ]; then
    conda_env_create ambertools -c conda-forge ambertools
    refresh_envs; find_amber
fi
opt_result "AmberTools" "$PACKMOL_MEMGEN"

# PyMOL — render_image (ray-traced figures).
if [ -z "$PYMOL_PYTHON" ]; then
    conda_env_create pymol-render -c conda-forge pymol-open-source
    refresh_envs; find_pymol
fi
opt_result "PyMOL" "$PYMOL_PYTHON"

# DSSP — describe_fold's computed topology. Current 4.x first; if it does not
# run here, 3.x (stable where 4.x segfaulted).
if [ -z "$DSSP_BIN" ]; then
    [ -n "$(env_path '^dssp$')" ] || conda_env_create dssp -c conda-forge dssp
    refresh_envs
    find_dssp || { do_install "DSSP 3.x (4.x does not run here)" \
                       conda install -y -n dssp --override-channels -c conda-forge "dssp=3"
                   find_dssp; }
fi
opt_result "DSSP" "$DSSP_BIN"

# fpocket — find_pockets.
if [ -z "$FPOCKET_BIN" ]; then
    conda_env_create fpocket -c conda-forge fpocket
    refresh_envs; find_fpocket
fi
opt_result "fpocket" "$FPOCKET_BIN"

# Foldseek — find_structural_neighbors, describe_fold fallback: the binary,
# then the PDB database and the AlphaFold (Swiss-Prot) database.
if [ -z "$FOLDSEEK_BIN" ]; then
    conda_env_create foldseek -c conda-forge -c bioconda foldseek
    refresh_envs; find_foldseek
fi
opt_result "Foldseek" "$FOLDSEEK_BIN"
fs_database() {   # fs_database LABEL NAME PREFIX GB
    if [ ! -f "$3.dbtype" ] && [ -n "$FOLDSEEK_BIN" ] && room_for "$1" "$4"; then
        mkdir -p "$(dirname "$3")"
        local tmp; tmp="$(mktemp -d)"
        do_install "$1 (~$4 GB)" "$FOLDSEEK_BIN" databases "$2" "$3" "$tmp"
        rm -rf "$tmp"
    fi
    if [ -f "$3.dbtype" ]; then opt_result "$1" "$3"; else opt_result "$1" ""; fi
}
fs_database "Foldseek PDB DB" PDB "$FS_DB" 5
fs_database "AlphaFold DB" "Alphafold/Swiss-Prot" "$FS_AFDB" 3

# ESM-2 — predict_mutation_effect: torch + transformers env, then the
# checkpoint into the Hugging Face cache (the app runs it offline).
if [ -z "$ESM_PYTHON" ] && room_for "ESM-2 env" 3; then
    conda_env_create esm -c conda-forge python=3.11 pytorch transformers
    refresh_envs; find_esm
fi
opt_result "ESM-2 env" "$ESM_PYTHON"
if [ -n "$ESM_PYTHON" ]; then
    if ! esm_model_cached && room_for "ESM-2 model" 3; then
        do_install "ESM-2 model $ESM_MODEL_NAME (~2.6 GB)" "$ESM_PYTHON" -c \
            "from huggingface_hub import snapshot_download
snapshot_download('$ESM_MODEL_NAME', allow_patterns=['*.json', '*.txt', 'model.safetensors'])"
    fi
    if esm_model_cached; then opt_result "ESM-2 model" "$ESM_MODEL_NAME"
    else opt_result "ESM-2 model" ""; fi
fi

# What the app reads (unset = that tool reports "unavailable").
[ -n "$PACKMOL_MEMGEN" ] && export PACKMOL_MEMGEN
[ -n "$PYMOL_PYTHON" ]   && export PYMOL_PYTHON
[ -n "$DSSP_BIN" ]       && export DSSP_BIN
[ -n "$FPOCKET_BIN" ]    && export FPOCKET_BIN
[ -n "$ESM_PYTHON" ]     && export ESM_PYTHON
if [ -n "$FOLDSEEK_BIN" ]; then
    export FOLDSEEK_BIN
    [ -f "$FS_DB.dbtype" ]   && export FOLDSEEK_DB="$FS_DB"
    [ -f "$FS_AFDB.dbtype" ] && export FOLDSEEK_AFDB="$FS_AFDB"
fi

# ── Port ─────────────────────────────────────────────────────────────────────
section "Port"
port_busy() {
    if command -v lsof >/dev/null 2>&1; then lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
    else (echo > "/dev/tcp/127.0.0.1/$1") 2>/dev/null; fi
}
if port_busy "$PORT"; then
    warn "Port $PORT is in use — PARORA may already be running at http://localhost:$PORT"
    while port_busy "$PORT"; do PORT=$((PORT + 1)); done
fi
ok "Port $PORT"
record ok "Port" "$PORT"

finish

# ── Launch ───────────────────────────────────────────────────────────────────
if $CHECK_ONLY; then
    echo; echo "Everything required is in place. Start PARORA with: bash run.sh"
    exit 0
fi
cd "$APP_DIR"
mkdir -p structures membranes prepared logs
echo
echo "${B}Starting PARORA${N} → ${B}http://localhost:$PORT${N}   (Ctrl+C to stop)"
echo "Logs: protein-viz-agent/logs/parora.log"
exec "$ENV_PREFIX/bin/streamlit" run app.py --server.headless=true --server.port="$PORT"
