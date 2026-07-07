"""
analysis_tools.py

Notebook-derived deterministic protein structure analysis tools
for PARORA.

Originally developed in the Goal 1/2 Colab notebook and adapted
for the Streamlit application.
"""

import numpy as np
import pandas as pd


def summarize_chains_from_universe(u) -> pd.DataFrame:
    """
    Summarize chains/segments in the current MDAnalysis Universe.
    Adapted from notebook summarize_chains(shared).
    """
    rows = []
    for seg in u.segments:
        atoms = seg.atoms
        residues = atoms.residues
        rows.append({
            "chain_or_segment": seg.segid,
            "n_atoms": len(atoms),
            "n_residues": len(residues),
        })
    return pd.DataFrame(rows)
