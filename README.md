<img src="logo/banner.png" alt="PARORA — Protein Agentic Rendering &amp; Observation for Residue Analysis" width="100%" style="max-width:100%;display:block;"/>

# PARORA — Protein Agentic Rendering & Observation for Residue Analysis

> A natural language interface for protein structure exploration — powered by a local agentic AI pipeline that queries the RCSB Protein Data Bank, performs residue-level structural analysis, and renders interactive 3D models directly in your browser.

PARORA puts a conversational interface in front of the full protein structure analysis workflow. Describe what you want in plain English and a local LLM agent — running entirely on your machine via Ollama — autonomously searches the RCSB PDB, downloads the structure, and builds layered WebGL visualizations through NGL.js.

The production server streams each action to the browser as an SSE event, so the 3D viewer updates progressively — structure loads first, then each representation appears one by one. LLM tool calls are deduplicated server-side before execution, colour is extracted directly from natural language when the model omits it, and existing representation attributes (colour, opacity) are preserved when only one property is changed. The full analysis pipeline, from database query to rendered 3D model, runs offline after a one-time model download.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Platform Support](#platform-support) *(macOS, Linux/servers, Windows)*
- [Prerequisites](#prerequisites)
  - [1. Docker](#1-docker-for-containerized-deployment)
  - [2. Ollama](#2-ollama-local-llm-runtime)
  - [3. Conda](#3-conda-for-runsh-the-recommended-local-path) *(for `run.sh`, the recommended local path)*
  - [4. Python 3.12](#4-python-312-for-fully-manual-local-dev-only) *(fully manual local dev only)*
  - [5. Optional structural-biology tools](#5-optional-structural-biology-tools-ambertools--pymol--dssp--foldseek) *(AmberTools / PyMOL / DSSP / Foldseek)*
- [Getting Started](#getting-started)
  - [Option A: Docker (Recommended for most users)](#option-a-docker-recommended-for-most-users)
  - [Option B: `run.sh` (Recommended for local development)](#option-b-runsh-recommended-for-local-development)
  - [Option C: Fully manual](#option-c-fully-manual)
- [App Variants](#app-variants)
- [Agent Tools Reference](#agent-tools-reference)
  - [Data Loading](#data-loading)
  - [Selections](#selections)
  - [Visualization](#visualization)
  - [Analysis](#analysis)
- [Supported Representations](#supported-representations)
- [Example Interactions](#example-interactions)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Natural language interface** — type "Show me hemoglobin" or "Highlight the ATP binding site as ball and stick" and the agent handles everything
- **Progressive SSE streaming** — each action streams to the browser as it executes; the 3D viewer updates one representation at a time, not all at once
- **Server-side deduplication** — redundant LLM tool calls for the same selection are resolved before any execution, picking the rep type the user explicitly named
- **Robust NGL selection mapping** — normalises natural language ("non-standard residues", "chains", "ligand") and corrects model quirks (e.g. `:protein` → `protein`) before any call reaches NGL.js
- **Color and opacity preservation** — explicit colours are extracted from the prompt when the model omits them; changing opacity alone never reverts a previously set colour
- **Advanced structural analysis** — MDAnalysis-powered B-factor filtering, proximity selections, solvent removal, and backbone RMSD alignment (full `app.py` variant)
- **Interactive 3D viewer** — WebGL-based NGL.js renderer with rotation, zoom, dark/light toggle, and layered representation support
- **Fully local inference** — powered by Ollama; no API keys, no data leaves your machine
- **Docker-ready** — one command to build and run in a containerized environment

---

## Architecture

<img src="logo/arch.png" alt="PARORA Architecture" width="100%" style="max-width:100%;display:block;"/>

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Frontend | Vanilla JS + NGL.js (single-page, no framework) |
| LLM Runtime | [Ollama](https://ollama.com/) — `qwen2.5:7b` (`server.py`, `app.py`) and `llama3.2` (`app_lite.py`) |
| Structural Analysis | [MDAnalysis](https://www.mdanalysis.org/), plus optional [AmberTools](https://ambermd.org/AmberTools.php) / [PyMOL](https://pymol.org/) / [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/) |
| 3D Visualization | [NGL.js v2](https://nglviewer.org/) (WebGL via CDN) |
| PDB Data Source | [RCSB PDB API](https://www.rcsb.org/) (`rcsb-api`), [UniProt](https://www.uniprot.org/), [PDBe SIFTS](https://www.ebi.ac.uk/pdbe/docs/sifts/) |
| Language | Python 3.12 |
| Deployment | Docker, or natively via conda (`run.sh`) — see [Platform Support](#platform-support) |

---

## Platform Support

| Platform | Docker path | Local (`run.sh` / manual) path |
| --- | --- | --- |
| **macOS** | ✅ Fully supported (Docker Desktop) | ✅ Fully supported |
| **Linux (incl. servers)** | ✅ Fully supported | ✅ Fully supported — this is the actual production/server path |
| **Windows** | ✅ Supported via Docker Desktop (requires WSL 2) | ⚠️ Only via WSL 2 — see below |

`run.sh`, `deploy.sh`, `ollama.sh`, and `setup_tools.sh` are all `#!/bin/bash` scripts — there is no native `.bat`/`.ps1` equivalent for any of them. On Windows, either use the Docker path (which runs Linux inside a container regardless of the host OS) or open a WSL 2 terminal and treat it as Linux for everything else in this README.

The four optional structural-biology tools ([AmberTools / PyMOL / DSSP / Foldseek](#5-optional-structural-biology-tools-ambertools--pymol--dssp--foldseek)) are auto-discovered from conda environments in Unix-style locations (`~/miniconda3`, `/opt/...`, and — for PyMOL only — `/Applications/PyMOL.app` on macOS). This works on macOS, Linux, and inside WSL 2; on native Windows you would need to set `PACKMOL_MEMGEN` / `PYMOL_PYTHON` / `DSSP_BIN` / `FOLDSEEK_BIN` by hand if you have them installed there.

---

## Prerequisites

Before running PARORA, ensure the following are installed on your system. Which of these you actually need depends on which [Getting Started](#getting-started) option you pick — Docker only needs #1 and #2; the recommended local path (`run.sh`) needs #2 and #3; a fully manual setup needs #2 and #4. #5 is optional everywhere.

---

### 1. Docker (for containerized deployment)

Docker is required only if you plan to run PARORA via the Docker option. The Docker image builds and runs **`app.py`**, the full 54-tool Streamlit agent, on **port 8501** — not the lighter FastAPI `server.py` on port 8000.

**macOS** — Download and install **Docker Desktop for Mac** (supports both Intel and Apple Silicon):
[https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

After installation, launch Docker Desktop from your Applications folder and wait for the whale icon to appear in the menu bar.

**Windows** — Download and install **Docker Desktop for Windows**:
[https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

> **Windows requirement:** Docker Desktop requires **WSL 2** (Windows Subsystem for Linux). The installer will prompt you to enable it automatically. If not, run the following in PowerShell as Administrator, then restart your machine:
>
> ```powershell
> wsl --install
> ```

**Linux** — Install Docker Engine via your distribution's package manager or the [official instructions](https://docs.docker.com/engine/install/). This is the standard way to run PARORA on a Linux server.

**Verify:**

```bash
docker --version
```

---

### 2. Ollama (local LLM runtime)

**macOS** — Download the `.dmg` from [https://ollama.com/download](https://ollama.com/download), open it, and drag Ollama to Applications. Launch it once — it registers as a background menu bar service automatically.

Alternatively, install via Homebrew:

```bash
brew install ollama
```

**Linux** — Install with the official script, then it runs as a systemd service:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows** — Download the `.exe` installer from [https://ollama.com/download](https://ollama.com/download) and run it. Ollama installs as a background Windows service and appears in the system tray.

**Pull the required models (all platforms)** — after installation, open a terminal and run:

```bash
ollama pull qwen2.5:7b   # used by server.py and app.py
ollama pull llama3.2     # used by app_lite.py
```

Or use the provided script, which pulls both (works on macOS/Linux/WSL):

```bash
bash ollama.sh
```

`run.sh` also pulls whichever of these two isn't already present, so this step is optional if you're using the `run.sh` path below.

**Verify:**

```bash
ollama list   # should show qwen2.5:7b and llama3.2
```

> **Note:** `qwen2.5:7b` is ~4.7 GB and `llama3.2` is ~2.0 GB. A one-time internet connection is required for this step (and for `run.sh`'s conda/pip setup, and for `describe_fold`'s CATH/SCOP lookup at runtime). Structure search and everything else runs entirely offline once models are pulled — no API keys, nothing sent to a cloud LLM provider.

---

### 3. Conda (for `run.sh`, the recommended local path)

`run.sh` (repo root) is the preferred way to run the full `app.py` agent outside Docker — it creates a `parora` conda environment from `parora.yml`, installs everything in `requirements.txt` into it, pulls the required Ollama models, auto-discovers the optional tools below, and launches Streamlit — all in one command. It needs a conda installation already present; it does **not** install conda itself.

**macOS / Linux** — install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), [Miniforge](https://github.com/conda-forge/miniforge), or Anaconda. `run.sh` looks for it at `~/miniconda3`, `~/anaconda3`, `~/miniforge3`, `~/mambaforge`, or (macOS Homebrew cask / common system paths) `/opt/homebrew/Caskroom/miniconda/base`, `/opt/anaconda3`, `/opt/miniconda3` — any of these work.

**Windows** — use WSL 2 and follow the Linux instructions inside it; `run.sh` is a bash script (see [Platform Support](#platform-support)).

**Verify:**

```bash
conda --version
```

---

### 4. Python 3.12 (for fully manual local dev only)

Only needed if you'd rather skip conda entirely and manage a plain virtualenv/pip install yourself (Option C below).

**macOS** — Download from [https://www.python.org/downloads/](https://www.python.org/downloads/) and run the `.pkg`, or install via Homebrew:

```bash
brew install python@3.12
```

**Linux** — Install via your distribution's package manager, e.g. `sudo apt install python3.12 python3.12-venv` (Debian/Ubuntu), or from [python.org](https://www.python.org/downloads/).

**Windows** — Download the installer from [https://www.python.org/downloads/](https://www.python.org/downloads/) and run the `.exe`.

> **Important:** On the first screen of the installer, check **"Add Python to PATH"** before clicking Install. Without this, `python` and `pip` will not be recognized in the terminal.

**Verify:**

```bash
# macOS / Linux
python3 --version

# Windows
python --version
```

Should report `3.12.x`.

---

### 5. Optional structural-biology tools (AmberTools / PyMOL / DSSP / Foldseek)

The full `app.py` agent has four features that depend on external tools it does **not** bundle and does **not** require — without them, the agent still runs, and the tools that need them just report "unavailable" instead of failing:

| Tool | Unlocks | Without it |
| --- | --- | --- |
| **AmberTools** | Hydrogen addition, membrane building, MD/QM input generation (`prepare_structure`, `build_membrane`, simulation/quantum/oniom tools) | Those specific tools report unavailable; everything else works |
| **PyMOL** (open-source build) | Ray-traced publication-quality figure rendering (`render_image`) | That tool reports unavailable |
| **DSSP** | Computed secondary-structure topology strings in `describe_fold` | `describe_fold` still returns CATH/SCOP fold classification (a network lookup, no dependency needed) — just not the computed topology half |
| **Foldseek** + a reference database | Structural similarity search (`find_structural_neighbors`): which known PDB structures this one resembles in 3D, with each hit's own CATH/SCOP fold — also `describe_fold`'s fallback for AlphaFold models and unclassified entries | The app **asks you** whether to search online at search.foldseek.com (uploads the structure; no binary needed) or download the local database — it never does either on its own |

These only matter for the full `app.py` agent; `server.py` and `app_lite.py` never use them.

**Easiest way to set these up** — from the repo root, after conda is installed:

```bash
bash setup_tools.sh              # checks each tool, asks before installing anything missing
bash setup_tools.sh --yes        # installs whatever's missing without asking
bash setup_tools.sh --check-only # just reports what's found, installs nothing
```

It only touches its own conda environments (`ambertools`, `pymol-render`, `dssp`, `foldseek`) plus `protein-viz-agent/foldseek_db/`, and never installs or downloads anything without asking, unless you pass `--yes`.

> **Foldseek: local or online, your choice.** With no local database, the first structural-similarity question gets a reply asking which you want: *search online* (uploads that structure's coordinates to the public search.foldseek.com server — fine for published structures, not for confidential ones; consent is remembered per structure for the session) or *download the database* (starts the download in the background from inside the app). Neither happens without your say-so.
>
> **Foldseek database size:** the Foldseek binary is small, but it needs a reference database to search. The PDB one is **~2.2 GB to download and ~4.2 GB on disk**, so `setup_tools.sh` asks about it separately and `run.sh` never downloads it. Manual equivalent: `foldseek databases PDB protein-viz-agent/foldseek_db/pdb /tmp/fs`, or set `FOLDSEEK_DB` to a database you already have (any Foldseek database prefix works, e.g. an AlphaFold-DB/Swiss-Prot one).

> **Known issue:** conda-forge's `dssp` package (4.x) has been observed to crash unpredictably on at least one arm64 macOS machine. `setup_tools.sh` installs the current version first, actually tests it, and automatically falls back to `dssp=3` if the test fails — so this is handled for you either way.

`run.sh` auto-discovers all four by conda environment name every time it starts the app, so once installed (by `setup_tools.sh` or manually), no further configuration is needed. To point at a non-standard install location instead, set `PACKMOL_MEMGEN` / `PYMOL_PYTHON` / `DSSP_BIN` / `FOLDSEEK_BIN` / `FOLDSEEK_DB` yourself (see [Environment Variables](#environment-variables)).

---

## Getting Started

### Option A: Docker (Recommended for most users)

Requires [Docker](#1-docker-for-containerized-deployment) and [Ollama](#2-ollama-local-llm-runtime) installed. Runs the full `app.py` agent — AmberTools/PyMOL stay optional and are not bundled in the image (see [Environment Variables](#environment-variables) if you want to point the container at env-based tools running on the host).

**Step 1** — Pull the LLM models (first time only):

```bash
bash ollama.sh
```

**Step 2** — Build and run the container, from the repo root:

```bash
bash deploy.sh
```

**Step 3** — Open your browser at `http://localhost:8501`.

The script automatically removes any previous container, rebuilds the image, and mounts local `structures/`, `membranes/`, `prepared/`, and `logs/` directories so downloaded/generated files persist between runs.

---

### Option B: `run.sh` (Recommended for local development)

The preferred way to run the full agent **outside** Docker — one command sets up the conda environment, installs Python dependencies, pulls Ollama models, auto-discovers optional tools, and launches Streamlit. Requires [Ollama](#2-ollama-local-llm-runtime) and [conda](#3-conda-for-runsh-the-recommended-local-path) installed (macOS/Linux, or WSL 2 on Windows — see [Platform Support](#platform-support)).

```bash
bash run.sh
```

That's it — open your browser at `http://localhost:8501` once it prints the Streamlit URL. Optionally, run `bash setup_tools.sh` first if you also want AmberTools/PyMOL/DSSP/Foldseek (see [Optional structural-biology tools](#5-optional-structural-biology-tools-ambertools--pymol--dssp--foldseek)) — `run.sh` will pick them up automatically either way, before or after.

---

### Option C: Fully manual

Requires [Python 3.12](#4-python-312-for-fully-manual-local-dev-only) and [Ollama](#2-ollama-local-llm-runtime) installed. Use this if you'd rather skip conda and manage dependencies yourself, or if you only want the lightweight `server.py`/`app_lite.py` entry points (neither needs AmberTools/PyMOL/DSSP/Foldseek at all).

**Step 1** — Pull the LLM models (first time only):

```bash
bash ollama.sh
```

**Step 2** — Install Python dependencies:

```bash
cd protein-viz-agent
pip install -r requirements.txt
```

**Step 3** — Run one of the three entry points:

```bash
# FastAPI server — 3 tools, SSE streaming, persistent NGL viewer, no page reloads
uvicorn server:app --reload

# Full-featured Streamlit agent — 55 tools, MDAnalysis, AmberTools/PyMOL/DSSP/Foldseek-backed analysis
streamlit run app.py

# Lite Streamlit agent — same 3 tools as server.py, for debugging the agent loop
streamlit run app_lite.py
```

**Step 4** — Open your browser:

- FastAPI (`server.py`): `http://localhost:8000`
- Streamlit (`app.py` / `app_lite.py`): `http://localhost:8501`

---

## App Variants

| File | Description |
| --- | --- |
| `server.py` | FastAPI server — 3 tools (`search_pdb`, `set_pdb`, `add_representation`), SSE streaming, server-side tool-call deduplication, NGL selection normalisation, color extraction from natural language, in-place representation updates. Docker's own default before it switched to `app.py`; still available manually. |
| `app.py` | Streamlit full-featured agent — **55 tools** spanning structure loading, selections, visualization, MDAnalysis-backed analysis, interaction detection, measurement, structure prep, membrane building, MD/QM input generation, PyMOL ray-traced rendering, and fold/topology classification and Foldseek structural similarity search. This is what Docker and `run.sh` both run. |
| `app_lite.py` | Streamlit lite agent — same 3-tool set as `server.py`, for debugging the agent loop without the full pipeline. |

---

## Agent Tools Reference

The LLM agent has access to the following tools, which it calls autonomously based on your instructions:

### Data Loading

| Tool | Description |
| --- | --- |
| `search_pdb(term)` | Free-text search of RCSB PDB; returns the top matching accession ID |
| `fetch_structure(pdb_id)` | Downloads the PDB file from RCSB and caches it locally |
| `load_local(filepath)` | Loads a PDB structure from a local file path |

### Selections

| Tool | Description |
| --- | --- |
| `select(name, expression)` | Creates a named selection using an NGL or MDAnalysis expression |
| `select_within(name, radius, target)` | Selects all residues within a given Å radius of a named selection |
| `select_by_bfactor(name, operator, threshold)` | Selects atoms by B-factor value (`>`, `<`, `==`) |

### Visualization

| Tool | Description |
| --- | --- |
| `show(rep_type, selection, color)` | Adds a representation layer to a selection |
| `hide(selection)` | Removes representations for a selection |
| `hide_all()` | Clears all representation layers |
| `show_all(rep_type)` | Applies a representation to all atoms |
| `color(color, selection)` | Recolors a selection; supports named colors and schemes |
| `set_transparency(value, selection)` | Sets transparency (`0.0` = opaque, `1.0` = invisible) |
| `zoom(selection)` | Focuses and zooms the camera on a selection |
| `set_background(color)` | Sets viewer background (`black`, `white`, `grey`) |

### Analysis

| Tool | Description |
| --- | --- |
| `measure_distance(sel1, sel2)` | Returns the inter-centroid distance in Ångstroms |
| `align_structures(mobile_id, ref_id)` | Aligns two structures by backbone RMSD |
| `remove_solvent()` | Strips all water molecules from the loaded structure |
| `save_structure(filename)` | Saves the current structure to `structures/` |

---

## Supported Representations

| Type | Description |
| --- | --- |
| `cartoon` | Secondary structure ribbons (helices, sheets, coils) |
| `ball+stick` | Atoms as spheres connected by bond sticks |
| `surface` | Molecular surface mesh (solvent-accessible) |
| `ribbon` | Smooth backbone trace |
| `spacefill` | Van der Waals spheres |
| `licorice` | Bonds only, no atom spheres |
| `point` | Lightweight dot cloud |

**Selection targets:** `protein`, `ligand`, `hetero`, `water`, `nonstandard`, specific residue names (e.g., `ATP`, `HEM`), named selections, or any NGL selection expression.

**Color schemes:** `element`, `spectrum`, `chainname`, `residueindex`, `bfactor`, or any named color (`red`, `cyan`, `white`, …).

---

## Example Interactions

```text
"Show me the structure of insulin"
"Load 3PP0 and display it as a cartoon colored by chain"
"Highlight all residues within 5 Å of the ATP ligand as ball and stick"
"Select residues with B-factor above 60 and color them red"
"Measure the distance between the active site and the allosteric pocket"
"Align hemoglobin and myoglobin and show the RMSD"
"Remove all water molecules and save the cleaned structure"
"Show the protein surface with 30% transparency"
"Zoom into the heme binding site"
```

---

## Project Structure

```text
PARORA/
├── logo/                         # Banner/logo images used in this README
├── protein-viz-agent/
│   ├── server.py                 # FastAPI server — 3-tool agent
│   ├── app_lite.py               # Streamlit lite agent — same 3 tools as server.py
│   ├── app.py                    # Streamlit full agent — 55 tools (Docker/run.sh default)
│   ├── templates/index.html      # server.py's single-page UI (vanilla JS + NGL.js)
│   ├── viewer_component/         # app.py's NGL viewer as a declared Streamlit component
│   ├── config.yaml               # Per-entry-point model config (read by parora_config.py)
│   ├── parora_config.py          # Shared model/Ollama config loader
│   ├── parora_logging.py         # Shared logging setup (logs/parora.log)
│   ├── rag_grounding.py          # Few-shot prompt grounding (app.py only)
│   ├── rag_examples.json         # Grounding examples rag_grounding.py reads
│   ├── Protein_accession.py      # RCSB + UniProt protein lookup
│   ├── sequence_utils.py         # Dependency-free PDB sequence parsing
│   ├── structure_report.py       # Dependency-free composition reports
│   ├── interactions.py           # Salt bridges, H-bonds, disulfides, stacking, metals
│   ├── measure.py                # Distance/angle/dihedral measurement
│   ├── superpose.py              # RMSD structure superposition
│   ├── topology.py               # DSSP + CATH/SCOP fold/topology classification
│   ├── structure_search.py       # Foldseek structural similarity search
│   ├── prepare.py                # Structure prep (hydrogens, states, cleanup)
│   ├── membrane.py               # OPM/MEMEMBED orientation + PACKMOL-Memgen packing
│   ├── simulation.py             # Amber/GROMACS/Rosetta input generation
│   ├── quantum.py / oniom.py     # QM region extraction / Gaussian QM-MM
│   ├── pymol_render.py / pymol_worker.py  # PyMOL ray-traced rendering
│   ├── analysis_tools.py         # Notebook-derived deterministic analysis
│   ├── requirements.txt          # Python dependencies (pip)
│   ├── Dockerfile                # Container configuration (builds app.py)
│   ├── structures/ membranes/ prepared/ logs/  # Runtime dirs, gitignored, created on demand
│   ├── foldseek_db/              # Optional Foldseek reference database, gitignored
│   └── simulations/              # Generated MD/QM job files
├── run.sh                        # Preferred local launcher — conda env, models, tool discovery, then app.py
├── setup_tools.sh                # Interactive AmberTools/PyMOL/DSSP/Foldseek checker & installer
├── deploy.sh                     # Docker build & run script (port 8501)
├── ollama.sh                     # Pulls both required Ollama models
└── parora.yml                    # Conda env spec used by run.sh
```

---

## Environment Variables

None of these are required — every entry point works with its documented defaults. Set these to override behavior or point at a non-standard tool install.

| Variable | Applies to | Default | Description |
| --- | --- | --- | --- |
| `OLLAMA_HOST` | all | `http://localhost:11434` | Ollama server URL. `app.py`/`server.py`/`app_lite.py` all auto-switch to `http://host.docker.internal:11434` when they detect they're running inside a container, even without setting this explicitly. |
| `PARORA_TEMPERATURE` | all | `0.0` (from `config.yaml`) | Sampling temperature. |
| `PARORA_NUM_CTX` | all | `16384` | Ollama context window, in tokens. |
| `PARORA_KEEP_ALIVE` | all | `30m` | How long Ollama keeps the model loaded in memory. |
| `PARORA_MODEL_SERVER` | `server.py` | `qwen2.5:7b` | Override just this entry point's model. |
| `PARORA_MODEL_APP_LITE` | `app_lite.py` | `llama3.2:latest` | Override just this entry point's model. |
| `PARORA_MODEL_APP` | `app.py` | `qwen2.5:7b` | Override just this entry point's model. |
| `PARORA_LOG_LEVEL` | all | `INFO` | Logging verbosity. |
| `PARORA_LOG_DIR` | all | `protein-viz-agent/logs/` | Where `parora.log` is written. |
| `PACKMOL_MEMGEN` | `app.py` | auto-discovered | Path to AmberTools' `packmol-memgen`, if it lives somewhere `run.sh`/`setup_tools.sh` wouldn't find on their own. |
| `PYMOL_PYTHON` | `app.py` | auto-discovered | Path to a Python interpreter that can `import pymol2`. |
| `DSSP_BIN` | `app.py` | auto-discovered | Path to an `mkdssp` binary. |
| `FOLDSEEK_BIN` | `app.py` | auto-discovered | Path to a `foldseek` binary. |
| `FOLDSEEK_DB` | `app.py` | `protein-viz-agent/foldseek_db/pdb` | Foldseek database prefix to search against. |

---

## Troubleshooting

**Ollama connection refused**
Ensure Ollama is running. On macOS, the installer registers it as a background service; on Linux it runs as a systemd service after `curl -fsSL https://ollama.com/install.sh | sh`. Verify with `ollama list`; if it's not running, launch the desktop app or run `ollama serve`.

**Model not found**
Run `bash ollama.sh` to pull both `qwen2.5:7b` and `llama3.2` before starting any entry point.

**`run.sh` says "Could not find a conda installation"**
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Miniforge/Anaconda) first — `run.sh` looks for it at `~/miniconda3`, `~/anaconda3`, `~/miniforge3`, `~/mambaforge`, or a few common system paths, but doesn't install conda itself. See [Conda](#3-conda-for-runsh-the-recommended-local-path).

**Docker can't reach Ollama**
On macOS, the container connects to `host.docker.internal:11434` automatically. On Linux, add `--add-host=host.docker.internal:host-gateway` to the `docker run` command in `deploy.sh`.

**Slow responses**
`qwen2.5:7b` runs on CPU by default if no compatible GPU is detected. For faster inference on Apple Silicon, ensure the Ollama version supports Metal acceleration (included by default in recent Ollama releases). The model requires approximately 6 GB of memory to run.

**MDAnalysis not available**
The `app.py` full-featured agent gracefully degrades if MDAnalysis fails to import. Re-install with `pip install MDAnalysis` in your environment.

**AmberTools / PyMOL / DSSP / Foldseek report "unavailable"**
Expected if you haven't installed them — none are required. Run `bash setup_tools.sh` to check what's present and install what's missing, interactively. See [Optional structural-biology tools](#5-optional-structural-biology-tools-ambertools--pymol--dssp--foldseek).

**DSSP crashes / `describe_fold`'s computed topology never returns**
conda-forge's `dssp` package (4.x) has been observed to segfault unpredictably on at least one arm64 macOS machine, on every input. `setup_tools.sh` handles this automatically (installs 4.x, smoke-tests it, falls back to `dssp=3` if the test fails); if you installed DSSP manually and hit this, run `conda install -n dssp -c conda-forge "dssp=3" -y` yourself.

**Running natively on Windows (no WSL, no Docker)**
Not supported directly — `run.sh`/`deploy.sh`/`ollama.sh`/`setup_tools.sh` are all bash scripts. Use Docker Desktop, or WSL 2 for the local path. See [Platform Support](#platform-support).

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.

---

Built by **Methun Kamruzzaman** · Powered by [Ollama](https://ollama.com), [FastAPI](https://fastapi.tiangolo.com), [NGL.js](https://nglviewer.org), and [MDAnalysis](https://www.mdanalysis.org)
