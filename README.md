<img src="logo/banner.png" alt="PARORA — Protein Agentic Rendering &amp; Observation for Residue Analysis" width="100%" style="max-width:100%;display:block;"/>

# PARORA — Protein Agentic Rendering & Observation for Residue Analysis

> A natural language interface for protein structure exploration — powered by a local agentic AI pipeline that queries the RCSB Protein Data Bank, performs residue-level structural analysis, and renders interactive 3D models directly in your browser.

PARORA puts a conversational interface in front of the full protein structure analysis workflow. Describe what you want in plain English and a multi-turn LLM agent — running entirely on your machine via Ollama — autonomously searches the RCSB PDB, downloads the structure, applies MDAnalysis-powered residue selections, and builds layered WebGL visualizations through NGL.js.

Ask it to isolate residues within 5 Å of a ligand binding site, filter by B-factor to reveal flexible regions, measure inter-atom distances, or align two structures by backbone RMSD. Every operation is driven by a set of 18 structured agent tools — no scripting, no command-line flags, no manual data wrangling. The full analysis pipeline, from database query to rendered 3D model, runs offline after a one-time model download.

---

## Features

- **Natural language interface** — type "Show me hemoglobin" or "Highlight the ATP binding site as ball and stick" and the agent handles everything
- **Autonomous multi-turn tool calling** — the LLM agent orchestrates a rich set of structural tools across multiple reasoning steps
- **Advanced structural analysis** — MDAnalysis-powered B-factor filtering, proximity selections, solvent removal, and backbone RMSD alignment
- **Named selections** — define, reuse, and layer selections by name across the entire session
- **Interactive 3D viewer** — WebGL-based NGL.js renderer with rotation, zoom, camera persistence, and layered representation support
- **Fully local inference** — powered by Ollama; no API keys, no data leaves your machine
- **Docker-ready** — one command to build and run in a containerized environment

---

## Architecture

```text
User Input (Natural Language)
        │
        ▼
   Ollama LLM (llama3.2)          ← runs locally via Ollama
        │
        ▼
  Multi-Turn Tool Loop
        │
        ├── search_pdb(term)       → RCSB PDB text search → PDB ID
        ├── fetch_structure(id)    → Download .pdb file from RCSB
        ├── load_local(path)       → Load a local PDB file
        ├── select(name, expr)     → Create a named selection
        ├── select_within(…)       → Proximity-based selection (Å radius)
        ├── select_by_bfactor(…)   → B-factor threshold selection
        ├── show / hide / color    → Manage representation layers
        ├── measure_distance(…)    → Inter-atom distance in Ångstroms
        ├── align_structures(…)    → Backbone RMSD structural alignment
        ├── zoom / set_background  → Camera and scene controls
        └── save_structure / remove_solvent / add_hydrogens
        │
        ▼
  Streamlit Session State
        │
        ▼
  NGL.js 3D Renderer (WebGL in browser)
```

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| UI Framework | [Streamlit](https://streamlit.io/) |
| LLM Runtime | [Ollama](https://ollama.com/) with `llama3.2:latest` |
| Structural Analysis | [MDAnalysis](https://www.mdanalysis.org/) |
| 3D Visualization | [NGL.js v2](https://nglviewer.org/) (WebGL via CDN) |
| PDB Data Source | [RCSB PDB API](https://www.rcsb.org/) (`rcsb-api`) |
| Language | Python 3.12 |
| Deployment | Docker |

---

## Prerequisites

Before running PARORA, ensure the following are installed on your system.

### 1. Docker (for containerized deployment)

Download and install Docker Desktop from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/).

Verify installation:

```bash
docker --version
```

### 2. Ollama (local LLM runtime)

Download and install Ollama from [https://ollama.com/download](https://ollama.com/download). The installer registers Ollama as a background service — no need to run `ollama serve` manually after installation.

Verify installation:

```bash
ollama --version
```

Pull the required model:

```bash
bash ollama.sh
# Pulls llama3.2:latest and confirms Ollama is ready at http://localhost:11434
```

> **Note:** The model download is approximately 2 GB. A one-time internet connection is required for this step only. All subsequent inference runs entirely offline.

### 3. Python 3.12 (for local development only)

Download from [https://www.python.org/downloads/](https://www.python.org/downloads/) or install via your system package manager.

Verify installation:

```bash
python3 --version   # should report 3.12.x
```

---

## Getting Started

### Option A: Docker (Recommended)

Requires Docker and Ollama installed (see Prerequisites above).

**Step 1** — Pull the LLM model (first time only):

```bash
bash ollama.sh
```

**Step 2** — Build and run the container:

```bash
bash deploy.sh
```

**Step 3** — Open your browser at `http://localhost:8501`.

The script automatically removes any previous container, rebuilds the image, and mounts a local `structures/` directory so downloaded PDB files persist between runs.

---

### Option B: Local Development

Requires Python 3.12 and Ollama installed (see Prerequisites above).

**Step 1** — Pull the LLM model (first time only):

```bash
bash ollama.sh
```

**Step 2** — Install Python dependencies:

```bash
cd protein-viz-agent
pip install -r requirements.txt
```

**Step 3** — Run the application:

```bash
# Full-featured version (recommended) — MDAnalysis, named selections, agent debug log
streamlit run app.py

# Simplified MCP-style version — lightweight three-tool agent
streamlit run app_mcp.py
```

**Step 4** — Open your browser at `http://localhost:8501`.

---

## App Variants

| File | Description |
| --- | --- |
| `app.py` | Full-featured agent — MDAnalysis structural analysis, 18 tools, named selections, B-factor filtering, distance measurement, structure alignment, camera persistence, agent debug panel |
| `app_mcp.py` | Simplified MCP-style agent — three core tools (search, load, represent); Docker default |

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
├── logo/
│   ├── banner.png               # Project banner image
│   ├── logo.png                 # Logo with text
│   └── logo_notext.png          # Logo without text
├── protein-viz-agent/
│   ├── app.py                   # Full-featured agent (recommended)
│   ├── app_mcp.py               # Simplified MCP-style agent (Docker default)
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Container configuration
│   └── structures/              # Local PDB file cache
├── deploy.sh                    # Docker build & run script
└── ollama.sh                    # Ollama model setup script
```

---

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL. Automatically set to `http://host.docker.internal:11434` when running inside Docker on macOS. |

---

## Troubleshooting

**Ollama connection refused**
Ensure Ollama is running. On macOS, the installer registers it as a background service; verify with `ollama list`. If not running, launch the Ollama desktop app or run `ollama serve`.

**Model not found**
Run `bash ollama.sh` to pull the `llama3.2:latest` model before starting the app.

**Docker can't reach Ollama**
On macOS, the container connects to `host.docker.internal:11434` automatically. On Linux, you may need to add `--add-host=host.docker.internal:host-gateway` to the `docker run` command in `deploy.sh`.

**Slow responses**
`llama3.2` runs on CPU by default if no compatible GPU is detected. For faster inference on Apple Silicon, ensure the Ollama version supports Metal acceleration (included by default in recent Ollama releases).

**MDAnalysis not available**
The `app.py` full-featured agent gracefully degrades if MDAnalysis fails to import. Re-install with `pip install MDAnalysis` in your environment.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.

---

Built by **Methun Kamruzzaman** · Powered by [Ollama](https://ollama.com), [Streamlit](https://streamlit.io), [NGL.js](https://nglviewer.org), and [MDAnalysis](https://www.mdanalysis.org)
