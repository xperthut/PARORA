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

import re
import streamlit as st
import os
import json
from pathlib import Path
from ollama import Client
from rcsbapi.search import TextQuery

# ── Asset resolution: works in both Docker (/app/logo/) and local dev (../logo/)
_HERE = Path(__file__).parent
_LOGO_NOTEXT = _HERE / "logo" / "logo_notext.png"
if not _LOGO_NOTEXT.exists():
    _LOGO_NOTEXT = _HERE.parent / "logo" / "logo_notext.png"

st.set_page_config(
    page_title="PARORA",
    page_icon=str(_LOGO_NOTEXT) if _LOGO_NOTEXT.exists() else "🧬",
    layout="wide"
)

# ── Compact header: logo + product name + one-line description ────────────────
_LOGO = _HERE / "logo" / "logo.png"
if not _LOGO.exists():
    _LOGO = _HERE.parent / "logo" / "logo.png"

col_logo, col_title = st.columns([1, 10])
with col_logo:
    if _LOGO.exists():
        st.image(str(_LOGO), width=72)
    else:
        st.markdown("### 🧬")
with col_title:
    st.markdown(
        "<div style='display:flex; flex-direction:column; justify-content:center; height:72px;'>"
        "<span style='font-size:1.6rem; font-weight:700; line-height:1.2;'>PARORA</span>"
        "<span style='font-size:0.85rem; color:#888;'>Protein Agentic Rendering &amp; Observation for Residue Analysis</span>"
        "</div>",
        unsafe_allow_html=True
    )

st.markdown(
    "<style>[data-testid='stToolbar'] { display: none; }</style>",
    unsafe_allow_html=True
)

st.divider()

# ── Ollama host resolution: prefer env var, fall back to Docker bridge ────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if os.path.exists("/.dockerenv"):
    OLLAMA_HOST = "http://host.docker.internal:11434"
ollama_client = Client(host=OLLAMA_HOST)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "messages":        [],        # chat history shown in the left panel
    "debug_logs":      [],        # internal tool-call trace for the debug expander
    "pdb_id":          None,      # currently loaded PDB accession (e.g. "3PP0")
    "representations": [],        # ordered list of NGL representation layers
    "background":      "black",   # NGL viewer background colour
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

model = "llama3.2:latest"


# ====================== AGENT TOOLS ======================

def _normalize_ngl_selection(sel: str) -> str:
    """Map natural-language chain/residue references to valid NGL selection syntax."""
    s = sel.strip()
    # "chain A" / "chain: A" → ":A"
    m = re.match(r'^chain[:\s]+([A-Za-z0-9])$', s, re.IGNORECASE)
    if m:
        return f":{m.group(1).upper()}"
    # bare single uppercase letter assumed to be a chain ID → ":A"
    if re.match(r'^[A-Z]$', s):
        return f":{s}"
    # plural / generic "chains" → "protein"
    if s.lower() in ("chain", "chains", "all chains", "all protein chains"):
        return "protein"
    return s


def tool_search_pdb(search_term: str) -> str:
    try:
        query_obj = TextQuery(value=search_term)
        session = query_obj(rows=1)
        pdb_id = next(session, None)
        return pdb_id if pdb_id else "No results"
    except Exception as e:
        return f"Error: {e}"


def tool_set_pdb(pdb_id: str) -> str:
    st.session_state.pdb_id = pdb_id.upper()
    st.session_state.representations = [{"type": "cartoon", "selection": "protein", "color": "spectrum"}]
    return f"Loaded {pdb_id}"


def tool_add_representation(rep_type: str, selection: str, color: str = "element") -> str:
    color = color or "element"
    selection = _normalize_ngl_selection(selection)
    st.session_state.representations.append({
        "type": rep_type,
        "selection": selection,
        "color": color
    })
    return f"Added {rep_type} for {selection} ({color})"


# ── Formal tool schema exposed to the Ollama agent ───────────────────────────
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
            "description": (
                "Add a visual representation layer to the current structure. "
                "selection must use NGL syntax: ':A' for chain A, 'protein' for all chains, "
                "'hetero' for ligands/cofactors, 'water' for water, or a 3-letter residue code. "
                "Never pass a bare chain letter without the leading colon."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rep_type": {
                        "type": "string",
                        "enum": ["cartoon", "ball+stick", "licorice", "surface", "spacefill", "ribbon", "line"]
                    },
                    "selection": {"type": "string", "description": "NGL selection string (e.g. ':A', 'protein', 'hetero', 'ATP')"},
                    "color": {"type": "string", "description": "Color name or scheme (e.g. 'red', 'element', 'spectrum', 'chainname')"}
                },
                "required": ["rep_type", "selection"]
            }
        }
    }
]


# ====================== AGENT ======================

def run_agent(prompt: str) -> str:
    st.session_state.debug_logs.append(f"📨 User: {prompt}")

    messages = [
        {
            "role": "system",
            "content": (
                f"You are PARORA, a protein structure visualization agent. "
                f"Current PDB: {st.session_state.pdb_id or 'none'}. "
                "Use the provided tools to load structures and add representations. "
                "Keep all previous layers unless the user asks for a new structure. "
                "NGL selection syntax rules — always follow these exactly: "
                "chain A → ':A', chain B → ':B', all protein chains → 'protein', "
                "heteroatoms/ligands → 'hetero', water → 'water', "
                "specific residue name → its 3-letter code (e.g. 'ATP', 'HEM'). "
                "Never pass a bare letter like 'A' as a selection; always prefix chains with ':'."
            )
        },
        {"role": "user", "content": prompt}
    ]

    response = ollama_client.chat(
        model=model,
        messages=messages,
        tools=tools,
        options={"temperature": 0.0}
    )

    tool_calls = response.get("message", {}).get("tool_calls", [])
    summary_parts = []

    for tool_call in tool_calls:
        func = tool_call.get("function", {})
        name = func.get("name")
        arguments = func.get("arguments", "{}")

        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except Exception:
                args = {}
        else:
            args = arguments if isinstance(arguments, dict) else {}

        if name == "search_pdb":
            result = tool_search_pdb(args.get("search_term", ""))
            st.info(f"🔍 Search result: {result}")
            if result and result not in ["No results", ""] and not st.session_state.pdb_id:
                tool_set_pdb(result)
                summary_parts.append(f"Loaded {result}")
            else:
                summary_parts.append(f"search_pdb: {result}")
        elif name == "set_pdb":
            result = tool_set_pdb(args.get("pdb_id", ""))
            summary_parts.append(result)
        elif name == "add_representation":
            result = tool_add_representation(
                args.get("rep_type", "ball+stick"),
                args.get("selection", "ligand"),
                args.get("color", "element")
            )
            summary_parts.append(result)
        else:
            result = f"Unknown tool: {name}"
            summary_parts.append(result)

        st.session_state.debug_logs.append(f"🔧 {name}({args}) → {result}")

    final_text = response.get("message", {}).get("content", "").strip()
    reply = final_text or ("Done: " + "; ".join(summary_parts)) if summary_parts else "No action taken."
    st.session_state.debug_logs.append(f"💬 Agent: {reply}")
    return reply


# ── NGL HTML renderer ─────────────────────────────────────────────────────────

def build_ngl_html() -> str:
    pdb_id = st.session_state.pdb_id
    if not pdb_id:
        return ""

    bg = st.session_state.background
    reps = st.session_state.representations

    reps_js = "".join(
        f'comp.addRepresentation("{rep["type"]}", {{sele: "{rep["selection"]}", color: "{rep.get("color", "element")}"}}); '
        for rep in reps
    )

    return f"""
    <div style="width:100%; height:680px; border:1px solid #555; border-radius:8px; overflow:hidden; background:{bg};">
        <div id="viewport" style="width:100%; height:100%;"></div>
    </div>
    <script src="https://unpkg.com/ngl@2/dist/ngl.js"></script>
    <script>
        (function() {{
            var stage = new NGL.Stage("viewport", {{backgroundColor: "{bg}"}});
            stage.loadFile("https://files.rcsb.org/download/{pdb_id}.pdb")
                .then(function (comp) {{
                    {reps_js}
                    comp.autoView();
                }});
            window.addEventListener("resize", () => stage.handleResize());
        }})();
    </script>
    """


# ====================== 25% / 75% HORIZONTAL LAYOUT ======================

left, right = st.columns([1, 3])   # 25% chat/controls | 75% 3D viewer

with left:
    st.subheader("💬 Agent Chat")

    # Scrollable chat history container
    chat_container = st.container(height=420)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("e.g. Load 3pp0, show ATP as ball+stick..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Agent reasoning..."):
            reply = run_agent(prompt)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    with st.expander("🔍 Agent Debug Logs", expanded=False):
        for log in st.session_state.debug_logs[-30:]:
            st.write(log)
        if not st.session_state.debug_logs:
            st.write("No logs yet")

    # Compact state summary below the chat
    if st.session_state.pdb_id:
        st.divider()
        st.markdown(f"**Structure:** `{st.session_state.pdb_id}`")
        st.markdown(f"**Representations:** {len(st.session_state.representations)}")

    if st.button("🆕 New Structure", type="secondary"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

with right:
    title_col, btn_col = st.columns([5, 1])
    with title_col:
        st.subheader("Interactive 3D Viewer")
    with btn_col:
        is_dark = st.session_state.background == "black"
        label = "☀️ Light" if is_dark else "🌙 Dark"
        if st.button(label, key="bg_toggle"):
            st.session_state.background = "white" if is_dark else "black"
            st.rerun()

    if st.session_state.pdb_id:
        html = build_ngl_html()
        st.components.v1.html(html, height=700, scrolling=False)

        if st.session_state.representations:
            with st.expander("🎨 Active Representations (NGL selections)", expanded=False):
                rows = []
                for i, r in enumerate(st.session_state.representations):
                    rows.append({
                        "#": i + 1,
                        "Type": r["type"],
                        "Selection": r["selection"],
                        "Color": r.get("color", "element"),
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Load a structure to begin. Try: **\"Load insulin\"** or **\"Fetch 3pp0\"**")
