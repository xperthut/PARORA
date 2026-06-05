# =============================================================================
# Developer : Methun Kamruzzaman
# Summary   : PARORA — FastAPI backend.
#             Agent runs server-side; frontend receives structured "actions"
#             and drives NGL.js directly, so the 3D viewer never reloads.
# Run       : uvicorn server:app --reload
# =============================================================================

import re
import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ollama import Client
from rcsbapi.search import TextQuery

_HERE = Path(__file__).parent
_LOGO_DIR = _HERE / "logo"
if not _LOGO_DIR.exists():
    _LOGO_DIR = _HERE.parent / "logo"

app = FastAPI()
if _LOGO_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_LOGO_DIR)), name="static")
templates = Jinja2Templates(directory=str(_HERE / "templates"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = Client(host=OLLAMA_HOST)
MODEL = "qwen2.5:7b"

# Single in-memory session (swap for Redis + session cookie for multi-user)
_session: dict = {"pdb_id": None, "representations": []}


# ── NGL selection normaliser ───────────────────────────────────────────────────

_NL_TO_NGL: dict[str, str] = {
    "ligand": "hetero", "ligands": "hetero",
    "het": "hetero", "hetatm": "hetero",
    "heteroatom": "hetero", "heteroatoms": "hetero",
    "non-standard residue": "hetero", "non-standard residues": "hetero",
    "nonstandard residue": "hetero", "nonstandard residues": "hetero",
    "non standard residue": "hetero", "non standard residues": "hetero",
    "non-standard": "hetero", "nonstandard": "hetero",
    "standard residue": "protein", "standard residues": "protein",
    "amino acid": "protein", "amino acids": "protein",
    "backbone only": "backbone",
    "water molecule": "water", "water molecules": "water", "solvent": "water",
    "all": "*", "everything": "*",
}

_VALID_REP_TYPES: set[str] = {
    "cartoon", "ball+stick", "licorice", "surface", "spacefill",
    "ribbon", "line", "backbone", "rope", "tube", "trace",
    "helixorient", "hyperball", "contact", "base", "label",
}


def _normalize_ngl_selection(sel: str) -> str:
    s = sel.strip()
    mapped = _NL_TO_NGL.get(s.lower())
    if mapped:
        return mapped
    m = re.match(r'^chain[:\s]+([A-Za-z0-9])$', s, re.IGNORECASE)
    if m:
        return f":{m.group(1).upper()}"
    if re.match(r'^[A-Z]$', s):
        return f":{s}"
    if s.lower() in ("chain", "chains", "all chains", "all protein chains"):
        return "protein"
    return s


# ── Tool implementations ───────────────────────────────────────────────────────

def tool_search_pdb(search_term: str) -> str:
    try:
        pdb_id = next(TextQuery(value=search_term)(rows=1), None)
        return pdb_id or "No results"
    except Exception as e:
        return f"Error: {e}"


def tool_set_pdb(pdb_id: str) -> dict:
    _session["pdb_id"] = pdb_id.upper()
    _session["representations"] = [
        {"type": "cartoon", "selection": "protein", "color": "spectrum", "opacity": 1.0}
    ]
    return {"action": "load_pdb", "pdb_id": _session["pdb_id"]}


def tool_add_representation(rep_type: str, selection: str, color: str = "element", opacity: float = 1.0) -> dict:
    color = color or "element"
    if rep_type not in _VALID_REP_TYPES:
        rep_type = "ball+stick"
    selection = _normalize_ngl_selection(selection)

    # Handle legacy "transparent" color value from older prompts
    if color.lower() in ("transparent", "translucent", "semi-transparent", "semitransparent"):
        opacity = 0.3
        existing = next(
            (r for r in _session["representations"]
             if r["type"] == rep_type and r["selection"] == selection),
            None,
        )
        color = existing["color"] if existing else "element"

    opacity = max(0.0, min(1.0, float(opacity)))
    rep = {"type": rep_type, "selection": selection, "color": color, "opacity": opacity}

    for i, r in enumerate(_session["representations"]):
        if r["type"] == rep_type and r["selection"] == selection:
            _session["representations"][i] = rep
            return {"action": "add_rep", **rep}

    _session["representations"].append(rep)
    return {"action": "add_rep", **rep}


# ── Tool schema (identical to app_lite.py) ────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_pdb",
            "description": "Search RCSB PDB by name and return the top PDB ID",
            "parameters": {
                "type": "object",
                "properties": {"search_term": {"type": "string"}},
                "required": ["search_term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_pdb",
            "description": "Load a specific PDB ID as the current structure",
            "parameters": {
                "type": "object",
                "properties": {"pdb_id": {"type": "string"}},
                "required": ["pdb_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_representation",
            "description": (
                "Add or update a visual representation layer on the current structure. "
                "selection must use NGL syntax: ':A' for chain A, 'protein' for all chains, "
                "'hetero' for ligands/non-standard residues, 'water' for water, "
                "or a 3-letter residue code (e.g. 'ATP'). "
                "Never pass a bare chain letter without the leading colon. "
                "To update an existing layer's color, opacity, or style, call with the SAME "
                "rep_type and selection — it replaces the layer in-place. "
                "For transparency use opacity=0.3; to restore full opacity use opacity=1.0."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rep_type": {
                        "type": "string",
                        "enum": ["cartoon", "ball+stick", "licorice", "surface", "spacefill", "ribbon", "line"],
                    },
                    "selection": {"type": "string", "description": "NGL selection (e.g. ':A', 'protein', 'hetero', 'ATP')"},
                    "color": {"type": "string", "description": "Color name or scheme (e.g. 'red', 'green', 'spectrum', 'chainname', 'element')"},
                    "opacity": {"type": "number", "description": "Opacity 0.0 (invisible) to 1.0 (fully opaque). Default 1.0. Use 0.3 for a transparent/ghost effect."},
                },
                "required": ["rep_type", "selection"],
            },
        },
    },
]


# ── Agent ─────────────────────────────────────────────────────────────────────

def run_agent(prompt: str) -> tuple[str, list[dict]]:
    """Returns (reply_text, list_of_actions).

    Actions are structured dicts the browser applies directly to the NGL stage:
      {"action": "load_pdb", "pdb_id": "3PP0"}
      {"action": "add_rep",  "type": "cartoon", "selection": ":A", "color": "red"}
    """
    messages = [
        {
            "role": "system",
            "content": (
                f"You are PARORA, a protein structure visualization agent. "
                f"Current PDB: {_session['pdb_id'] or 'none'}. "
                "NGL selection syntax — follow these rules exactly:\n"
                "  'chain A' or 'chain A only' → ':A'\n"
                "  'chain B' or 'chain B only' → ':B'\n"
                "  'chains', 'all chains', 'both chains', 'protein', 'whole protein', "
                "  'standard residues', 'amino acids' → 'protein'\n"
                "  IMPORTANT: if the user says 'chains' (plural) without naming a specific letter, "
                "  ALWAYS use 'protein', never ':A'.\n"
                "  non-standard residues / ligands / heteroatoms → 'hetero'\n"
                "  water → 'water'  |  specific residue → 3-letter code (e.g. 'ATP', 'HEM')\n"
                "  Never pass a bare letter like 'A'; always prefix chains with ':'.\n"
                "Representation rules:\n"
                "  - To CHANGE color, style, or opacity of an existing layer, call add_representation "
                "with the SAME rep_type and selection — it replaces the layer in-place.\n"
                "  - To make something transparent/ghost: use opacity=0.3.\n"
                "  - To restore full opacity: use opacity=1.0.\n"
                "  - Do NOT add duplicate layers for the same type+selection.\n"
                "  - Only call set_pdb to load a new structure."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    response = ollama_client.chat(
        model=MODEL, messages=messages, tools=TOOLS, options={"temperature": 0.0}
    )

    actions: list[dict] = []
    summary: list[str] = []

    for call in response.get("message", {}).get("tool_calls", []):
        func = call.get("function", {})
        name = func.get("name")
        raw_args = func.get("arguments", "{}")
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})

        if name == "search_pdb":
            pdb_id = tool_search_pdb(args.get("search_term", ""))
            if pdb_id not in ("No results", ""):
                action = tool_set_pdb(pdb_id)
                actions.append(action)
                summary.append(f"Loaded {pdb_id}")
        elif name == "set_pdb":
            action = tool_set_pdb(args.get("pdb_id", ""))
            actions.append(action)
            summary.append(f"Loaded {args.get('pdb_id', '')}")
        elif name == "add_representation":
            raw_opacity = args.get("opacity")
            opacity = float(raw_opacity) if raw_opacity is not None else 1.0
            sel = args.get("selection", "protein")
            # Safety net: if the LLM returned a specific chain but the user
            # clearly meant all chains, override to "protein".
            _all_chains_words = re.compile(
                r'\b(chains|all chains|both chains|whole protein|all protein|protein chains)\b',
                re.IGNORECASE,
            )
            _specific_chain = re.compile(r'^chain\s+[A-Za-z]$', re.IGNORECASE)
            if (_all_chains_words.search(prompt)
                    and not _specific_chain.search(prompt)
                    and re.match(r'^:[A-Z]$', sel)):
                sel = "protein"
            action = tool_add_representation(
                args.get("rep_type", "ball+stick"),
                sel,
                args.get("color") or "chainname",
                opacity,
            )
            actions.append(action)
            summary.append(f"Added {action['type']} for {action['selection']} ({action['color']}, opacity={action['opacity']})")

    text = response.get("message", {}).get("content", "").strip()
    reply = text or ("Done: " + "; ".join(summary)) or "No action taken."
    return reply, actions


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    ico = _LOGO_DIR / "favicon.ico"
    return FileResponse(str(ico), media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    reply, actions = run_agent(body.get("prompt", ""))
    return JSONResponse({"reply": reply, "actions": actions})


@app.post("/api/reset")
async def reset():
    _session["pdb_id"] = None
    _session["representations"] = []
    return {"ok": True}


@app.get("/api/state")
async def state():
    return JSONResponse(_session)
