FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY protein-viz-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Full-featured Streamlit agent (app.py) and every local module it imports
# unconditionally. Note: PyMOL ray tracing (pymol_render.py/pymol_worker.py)
# and the AmberTools-backed tools (membrane build, hydrogen prep, simulation,
# QM/QM-MM) stay optional -- they shell out to a PyMOL/AmberTools conda env
# discovered at runtime (PYMOL_PYTHON / PACKMOL_MEMGEN / AMBERHOME), which
# this image does not bundle. Same for DSSP and Foldseek (DSSP_BIN /
# FOLDSEEK_BIN); with no local Foldseek database the app asks the user
# before any online search, so the image still works without them.
# Everything else works out of the box.
COPY protein-viz-agent/app.py .
COPY protein-viz-agent/parora_logging.py .
COPY protein-viz-agent/parora_config.py .
COPY protein-viz-agent/config.yaml .
COPY protein-viz-agent/analysis_tools.py .
COPY protein-viz-agent/sequence_utils.py .
COPY protein-viz-agent/structure_report.py .
COPY protein-viz-agent/interactions.py .
COPY protein-viz-agent/measure.py .
COPY protein-viz-agent/superpose.py .
COPY protein-viz-agent/prepare.py .
COPY protein-viz-agent/oniom.py .
COPY protein-viz-agent/quantum.py .
COPY protein-viz-agent/simulation.py .
COPY protein-viz-agent/membrane.py .
COPY protein-viz-agent/Protein_accession.py .
COPY protein-viz-agent/pymol_render.py .
COPY protein-viz-agent/pymol_worker.py .
COPY protein-viz-agent/topology.py .
COPY protein-viz-agent/structure_search.py .
COPY protein-viz-agent/rag_grounding.py .
COPY protein-viz-agent/rag_examples.json .
COPY protein-viz-agent/viewer_component/ ./viewer_component/
COPY logo/ ./logo/
RUN mkdir -p /app/structures /app/membranes /app/prepared /app/logs

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
