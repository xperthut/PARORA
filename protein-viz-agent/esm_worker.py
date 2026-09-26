# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-25
# Summary   : ESM-2 masked-marginal scoring worker. Runs inside a
#             torch + transformers interpreter (NOT the Streamlit env), reads
#             a JSON job on stdin, scores one masked position, and prints the
#             result on a single RESULT_MARKER line.
#
#             Invoked as a subprocess by esm_tools.py so that torch never has
#             to be installed alongside Streamlit (or in the Docker image).
#             Runs offline (HF_HUB_OFFLINE, set by the caller): the checkpoint
#             must already be in the Hugging Face cache — no data leaves the
#             machine, and nothing is downloaded silently.
#
#             Keep this file Python 3.10-compatible, like pymol_worker.py.
# =============================================================================

import json
import sys

RESULT_MARKER = "__ESM_RESULT__"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# ESM-2 was trained on crops of at most 1024 tokens, BOS/EOS included.
MAX_RESIDUES = 1022


def _emit(obj):
    print(RESULT_MARKER + json.dumps(obj))
    sys.stdout.flush()


def _window(n, center, size):
    """[start, end) of a length-`size` window over n residues, around center."""
    if n <= size:
        return 0, n
    start = max(0, min(center - size // 2, n - size))
    return start, start + size


def score(job):
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    tok = AutoTokenizer.from_pretrained(job["model"])
    model = EsmForMaskedLM.from_pretrained(job["model"])
    model.eval()
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    model.to(device)

    # One entry per residue: a one-letter code, or null for a position the
    # structure does not observe (filled with <mask> so numbering stays put).
    seq = job["sequence"]
    pos = int(job["index"])
    start, end = _window(len(seq), pos, MAX_RESIDUES)

    ids = [tok.cls_token_id]
    for i in range(start, end):
        if i == pos or seq[i] is None:
            ids.append(tok.mask_token_id)
        elif seq[i] in AMINO_ACIDS:
            ids.append(tok.convert_tokens_to_ids(seq[i]))
        else:
            ids.append(tok.unk_token_id)
    ids.append(tok.eos_token_id)

    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0, 1 + pos - start]
    aa_ids = [tok.convert_tokens_to_ids(a) for a in AMINO_ACIDS]
    # Renormalised over the 20 standard amino acids; a log-ratio between two
    # of them is identical either way, but the probabilities read cleanly.
    logp = torch.log_softmax(logits[aa_ids].float(), dim=-1).cpu().tolist()
    return {
        "ok": True,
        "device": device,
        "window": [start, end],
        "log_probs": dict(zip(AMINO_ACIDS, logp)),
    }


def main():
    try:
        job = json.loads(sys.stdin.read())
        _emit(score(job))
    except Exception as exc:
        _emit({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})


if __name__ == "__main__":
    main()
