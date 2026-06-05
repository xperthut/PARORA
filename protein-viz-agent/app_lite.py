# =============================================================================
# Developer : Methun Kamruzzaman
# Date      : 2026-06-04
# Summary   : PARORA lite — a three-tool agentic protein visualizer.
#             Exposes search_pdb, set_pdb, and add_representation to an
#             Ollama agent via native tool-calling. Structures are fetched
#             from RCSB PDB and rendered as interactive 3D models via NGL.js.
#             This is the Docker default; see app.py for the full-featured
#             variant with MDAnalysis and 18 tools.
# =============================================================================

import streamlit as st
import os
import json
from pathlib import Path
from ollama import Client
from rcsbapi.search import TextQuery

# ── Logo resolution: works in both Docker (/app/logo/) and local dev (../logo/)
_HERE = Path(__file__).parent
_LOGO = _HERE / "logo" / "logo.png"
if not _LOGO.exists():
    _LOGO = _HERE.parent / "logo" / "logo.png"
_LOGO_NOTEXT = _HERE / "logo" / "logo_notext.png"
if not _LOGO_NOTEXT.exists():
    _LOGO_NOTEXT = _HERE.parent / "logo" / "logo_notext.png"

st.set_page_config(
    page_title="PARORA",
    page_icon=str(_LOGO_NOTEXT) if _LOGO_NOTEXT.exists() else "🧬",
    layout="wide"
)

# ── Branding header ───────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    if _LOGO.exists():
        st.image(str(_LOGO), width=120)
    else:
        st.markdown("### 🧬")
with col_title:
    st.title("PARORA")
    st.caption("Protein Agentic Rendering & Observation for Residue Analysis")

st.divider()

# ── Ollama host resolution: prefer env var, fall back to Docker bridge ────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if os.path.exists("/.dockerenv"):
    OLLAMA_HOST = "http://host.docker.internal:11434"
ollama_client = Client(host=OLLAMA_HOST)

# ── Session state ─────────────────────────────────────────────────────────────
# pdb_id          : currently loaded PDB accession (e.g. "3PP0")
# representations : ordered list of NGL representation layers for the viewer
if "pdb_id" not in st.session_state:
    st.session_state.pdb_id = None
if "representations" not in st.session_state:
    st.session_state.representations = []

model = "llama3.2:latest"


# ====================== AGENT TOOLS ======================

def tool_search_pdb(search_term: str) -> str:
    """
    Search the RCSB PDB database by free-text and return the top matching PDB ID.

    Uses the rcsbapi TextQuery interface to retrieve the single best result.
    Returns "No results" when the query yields nothing, or an error string on
    network/API failure.

    Args:
        search_term: Free-text description or protein name (e.g. "human insulin").

    Returns:
        PDB accession string (e.g. "3I40"), "No results", or "Error: <detail>".
    """
    try:
        query_obj = TextQuery(value=search_term)
        session = query_obj(rows=1)       # Only fetch the top-ranked result
        pdb_id = next(session, None)
        return pdb_id if pdb_id else "No results"
    except Exception as e:
        return f"Error: {e}"


def tool_set_pdb(pdb_id: str) -> str:
    """
    Load a PDB structure into the viewer by setting session state.

    Uppercases the accession, stores it, and resets representations to a
    default spectrum-colored cartoon covering the whole protein chain.

    Args:
        pdb_id: 4-character PDB accession code (case-insensitive).

    Returns:
        Confirmation string indicating which structure was loaded.
    """
    st.session_state.pdb_id = pdb_id.upper()
    # Reset to a single default cartoon layer whenever a new structure is loaded
    st.session_state.representations = [{"type": "cartoon", "selection": "protein", "color": "spectrum"}]
    return f"Loaded {pdb_id}"


def tool_add_representation(rep_type: str, selection: str, color: str = "element") -> str:
    """
    Append an additional NGL representation layer to the current structure.

    Layers are cumulative — calling this multiple times stacks representations
    without removing existing ones, enabling combined views (e.g. cartoon
    backbone + ball+stick ligands).

    Args:
        rep_type : NGL representation type ("cartoon", "ball+stick", "surface",
                   "spacefill", or "ribbon").
        selection: NGL selection string or atom group (e.g. "ligand", "protein",
                   "ATP", ":A").
        color    : NGL color scheme or named color (default: "element").

    Returns:
        Confirmation string describing the added layer.
    """
    st.session_state.representations.append({
        "type": rep_type,
        "selection": selection,
        "color": color
    })
    return f"Added {rep_type} for {selection} ({color})"


# ── Formal tool schema exposed to the Ollama agent ───────────────────────────
# Each entry follows the OpenAI-compatible function-calling format that Ollama
# uses to decide which tool to invoke and with what arguments.
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_pdb",
            "description": "Search RCSB PDB by name and return the top PDB ID",
            "parameters": {
                "type": "object",
                "properties": {"search_term": {"type": "string"}},
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_pdb",
            "description": "Load a specific PDB ID as the current structure",
            "parameters": {
                "type": "object",
                "properties": {"pdb_id": {"type": "string"}},
                "required": ["pdb_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_representation",
            "description": "Add a visual representation layer (cartoon, ligand, ATP, surface, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "rep_type": {"type": "string", "enum": ["cartoon", "ball+stick", "surface", "spacefill", "ribbon"]},
                    "selection": {"type": "string"},
                    "color": {"type": "string"}
                },
                "required": ["rep_type", "selection"]
            }
        }
    }
]


# ====================== AGENT ======================

def run_agent(prompt: str):
    """
    Run a single-turn agentic loop for a given user prompt.

    Sends the prompt plus the current viewer state to the Ollama model with
    the tool schema attached. Iterates over any tool calls returned and
    dispatches them to the corresponding tool functions, updating session
    state in place. The viewer is then re-rendered by Streamlit on the next
    rerun cycle.

    Args:
        prompt: Natural-language command from the user.
    """
    messages = [
        {
            "role": "system",
            "content": (
                f"You are PARORA, a protein structure visualization agent. "
                f"Current PDB: {st.session_state.pdb_id or 'none'}. "
                "Use the provided tools to load structures and add representations. "
                "Keep all previous layers unless the user asks for a new structure."
            )
        },
        {"role": "user", "content": prompt}
    ]

    response = ollama_client.chat(
        model=model,
        messages=messages,
        tools=tools,
        options={"temperature": 0.0}   # Deterministic tool selection
    )

    tool_calls = response.get("message", {}).get("tool_calls", [])

    for tool_call in tool_calls:
        func = tool_call.get("function", {})
        name = func.get("name")
        arguments = func.get("arguments", "{}")

        # Ollama may return arguments as a JSON string or as a dict directly
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except Exception:
                args = {}
        else:
            args = arguments if isinstance(arguments, dict) else {}

        # Dispatch to the matching tool function
        if name == "search_pdb":
            result = tool_search_pdb(args.get("search_term", ""))
            st.info(f"🔍 Search result: {result}")
            # Auto-load the found structure when no structure is currently active
            if result and result not in ["No results", ""] and not st.session_state.pdb_id:
                tool_set_pdb(result)
        elif name == "set_pdb":
            tool_set_pdb(args.get("pdb_id", ""))
        elif name == "add_representation":
            tool_add_representation(
                args.get("rep_type", "ball+stick"),
                args.get("selection", "ligand"),
                args.get("color", "element")
            )


# ====================== 25% / 75% HORIZONTAL LAYOUT ======================

left, right = st.columns([1, 3])   # 25% controls | 75% 3D viewer

with left:
    st.subheader("Agent Controls")
    prompt = st.text_area("Your command (one at a time)",
                          value="Download the structure of 3pp0", height=120)

    if st.button("🚀 Run Agent", type="primary"):
        with st.spinner("Agent reasoning + calling tools..."):
            run_agent(prompt)
            st.rerun()

    st.divider()

    # Show the currently loaded structure and active representation layers
    if st.session_state.pdb_id:
        st.subheader(f"Current: **{st.session_state.pdb_id}**")
        for i, rep in enumerate(st.session_state.representations):
            st.write(f"• **{rep['type']}** — {rep['selection']} ({rep.get('color','default')})")
    else:
        st.info("Load a structure first")

    if st.button("🆕 Explore New Structure", type="secondary"):
        # Clear all state to start fresh with a different protein
        st.session_state.pdb_id = None
        st.session_state.representations = []
        st.rerun()

with right:
    st.subheader("Interactive 3D Visualization")
    if st.session_state.pdb_id:
        pdb = st.session_state.pdb_id
        # Build one addRepresentation JS call per active layer
        reps_js = "".join(
            f'comp.addRepresentation("{rep["type"]}", {{sele: "{rep["selection"]}", color: "{rep.get("color", "element")}"}}); '
            for rep in st.session_state.representations
        )

        html = f"""
        <div style="width:100%; height:720px; border:1px solid #555; border-radius:8px; overflow:hidden; background:black;">
            <div id="viewport" style="width:100%; height:100%;"></div>
        </div>
        <script src="https://unpkg.com/ngl@2/dist/ngl.js"></script>
        <script>
            (function() {{
                var stage = new NGL.Stage("viewport", {{backgroundColor: "black"}});
                stage.loadFile("https://files.rcsb.org/download/{pdb}.pdb")
                    .then(function (comp) {{
                        {reps_js}
                        comp.autoView();
                    }});
                window.addEventListener("resize", () => stage.handleResize());
            }})();
        </script>
        """
        st.components.v1.html(html, height=750, scrolling=False)
    else:
        st.info("Run a command on the left panel")
