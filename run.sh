#!/bin/bash
# Set up (if needed) and launch PARORA's full agent (app.py) locally, with the
# conda environment, Ollama models, and AmberTools discovery all wired up.
# This is the path that reaches every tool -- Docker (deploy.sh) only runs
# server.py's 3-tool subset and never bundles AmberTools/PyMOL.
set -e

cd "$(dirname "$0")"

ENV_NAME="parora"
AMBERTOOLS_ENV="ambertools"

# ── 1. Conda environment ──────────────────────────────────────────────────────
# A plain script invocation doesn't source ~/.zshrc or ~/.bashrc, so `conda`
# is not guaranteed to be the shell function conda init set up interactively
# -- on a machine with more than one conda install (Miniconda + a Homebrew
# cask, say) a bare `conda` here can silently resolve to the wrong one and
# "not find" environments that exist under the other root. Locate a real
# conda.sh and source it explicitly so this script always uses the same
# installation, regardless of how it was invoked.
CONDA_ROOT=""
for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
                 "/opt/homebrew/Caskroom/miniconda/base" "/opt/anaconda3" "/opt/miniconda3"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_ROOT="$candidate"
        break
    fi
done

if [ -z "$CONDA_ROOT" ]; then
    echo "Could not find a conda installation. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
echo "Using conda at $CONDA_ROOT"

ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"

if [ -x "$ENV_PREFIX/bin/python" ]; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' from parora.yml..."
    conda env create -f parora.yml
fi

# From here on, call this environment's own binaries by absolute path rather
# than `conda run -n`/`conda activate` -- on a machine with more than one
# conda root, name-based resolution has proven to silently pick the wrong
# root's envs/ directory even after sourcing the intended conda.sh above.
echo "Syncing Python dependencies from protein-viz-agent/requirements.txt..."
"$ENV_PREFIX/bin/pip" install -q -r protein-viz-agent/requirements.txt

# ── 2. Ollama + models ─────────────────────────────────────────────────────────
if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not found. Install it first: https://ollama.com/download"
    exit 1
fi

if ! curl -s -o /dev/null --max-time 2 http://localhost:11434; then
    echo "Ollama does not appear to be running. Start it (the desktop app, or 'ollama serve') and re-run this script."
    exit 1
fi

for model in qwen2.5:14b llama3.2; do
    if ! ollama list | awk '{print $1}' | grep -qx "$model" && ! ollama list | awk '{print $1}' | grep -qx "${model}:latest"; then
        echo "Pulling $model..."
        ollama pull "$model"
    fi
done

# Optional tools: offer to install missing ones on first launch only.
# Asks before each (multi-GB) download; skips silently without a terminal.
# Re-run any time with: bash setup_tools.sh
if [ ! -f .tools_checked ]; then
    bash setup_tools.sh && touch .tools_checked
fi

# ── 3. Optional-tool discovery (AmberTools, DSSP, Foldseek, fpocket, PyMOL, ESM-2) ──
# Shared with evals/run.py so the eval harness sees the same backends.
# shellcheck disable=SC1091
source discover_tools.sh

# ── 5. Run the full agent ──────────────────────────────────────────────────────
cd protein-viz-agent
mkdir -p structures membranes prepared logs
echo "Logs: protein-viz-agent/logs/parora.log (tail -f it in another terminal to watch live)"
exec "$ENV_PREFIX/bin/streamlit" run app.py --server.headless=true
