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
from fastapi.responses import HTMLResponse, JSONResponse
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
MODEL = "llama3.2:latest"

# Single in-memory session (swap for Redis + session cookie for multi-user)
_session: dict = {"pdb_id": None, "representations": []}


# ── NGL selection normaliser (same logic as app_lite.py) ──────────────────────

def _normalize_ngl_selection(sel: str) -> str:
    s = sel.strip()
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
        {"type": "cartoon", "selection": "protein", "color": "spectrum"}
    ]
    return {"action": "load_pdb", "pdb_id": _session["pdb_id"]}


def tool_add_representation(rep_type: str, selection: str, color: str = "chainname") -> dict:
    color = color or "chainname"
    selection = _normalize_ngl_selection(selection)
    rep = {"type": rep_type, "selection": selection, "color": color}
    _session["representations"].append(rep)
    # Return the action so the browser can apply it directly to the live stage
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
                "Add a visual representation layer. "
                "NGL selections: ':A' for chain A, 'protein' for all chains, "
                "'hetero' for ligands, 'water' for water, or a 3-letter residue code. "
                "Never pass a bare chain letter without the leading colon."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rep_type": {
                        "type": "string",
                        "enum": ["cartoon", "ball+stick", "licorice", "surface", "spacefill", "ribbon", "line"],
                    },
                    "selection": {"type": "string"},
                    "color": {"type": "string"},
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
                "Use the provided tools to load structures and add representations. "
                "Keep all previous layers unless the user asks for a new structure. "
                "NGL selection syntax: ':A' chain A, 'protein' all chains, "
                "'hetero' ligands, 'water' water. Never use a bare letter for a chain."
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
            action = tool_add_representation(
                args.get("rep_type", "ball+stick"),
                args.get("selection", "protein"),
                args.get("color") or "chainname",
            )
            actions.append(action)
            summary.append(f"Added {action['type']} for {action['selection']} ({action['color']})")

    text = response.get("message", {}).get("content", "").strip()
    reply = text or ("Done: " + "; ".join(summary)) or "No action taken."
    return reply, actions


# ── Routes ────────────────────────────────────────────────────────────────────

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
