"""
Small on-device retrieval store for few-shot tool-call grounding (P4 in
todo.txt).

The original plan called for sentence-transformers + chromadb/FAISS. That
was tried and dropped: sentence-transformers pulls in torch as a transitive
dependency (~130MB), which conflicts with this repo's documented preference
for avoiding heavy deps (see CLAUDE.md's "Dependency-free PDB parsing" and
"Optional-dependency-via-subprocess pattern" sections) and would meaningfully
grow the Docker image built by deploy.sh. At the scale this actually needs —
a few dozen short example prompts — plain TF-IDF + cosine similarity over
word tokens, using only the stdlib and re, does the same retrieval job with
no new dependency and no model download.

Examples live in rag_examples.json as (prompt, grounding) pairs: real
successful tool calls where logs/parora.log had them, and literal
translations of the deterministic-workflow/regex-shortcut logic they replace
for prompts that never reach that log line (a regex pre-router intercepts
them before any tool executes, so no real log line exists to mine).
"""
import json
import math
import re
from pathlib import Path

_EXAMPLES_PATH = Path(__file__).parent / "rag_examples.json"
_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Filler words common enough to overlap between an unrelated prompt (e.g. a
# greeting) and some example purely by chance, without carrying any of the
# domain vocabulary that should drive a match.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "for", "to", "of", "and", "or", "this", "that",
    "there", "it", "its", "with", "as", "do", "does", "did", "not",
    "what", "which", "how",
})

# Below this cosine similarity, a retrieved example shares too little
# vocabulary with the prompt to be useful grounding — drop it rather than
# padding the context with noise.
_MIN_SCORE = 0.15


def _tokenize(text: str) -> list:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class _TfidfIndex:
    def __init__(self, examples: list):
        self.examples = examples
        docs_tokens = [_tokenize(ex["prompt"]) for ex in examples]

        doc_freq = {}
        for tokens in docs_tokens:
            for term in set(tokens):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        n_docs = len(docs_tokens) or 1
        self.idf = {
            term: math.log((1 + n_docs) / (1 + count)) + 1
            for term, count in doc_freq.items()
        }
        self.doc_vectors = [self._vectorize(tokens) for tokens in docs_tokens]

    def _vectorize(self, tokens: list) -> dict:
        term_freq = {}
        for term in tokens:
            term_freq[term] = term_freq.get(term, 0) + 1

        vec = {}
        norm_sq = 0.0
        for term, count in term_freq.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            weight = count * idf
            vec[term] = weight
            norm_sq += weight * weight

        norm = math.sqrt(norm_sq) or 1.0
        return {term: weight / norm for term, weight in vec.items()}

    def query(self, text: str, k: int) -> list:
        qvec = self._vectorize(_tokenize(text))
        if not qvec:
            return []

        scored = []
        for i, dvec in enumerate(self.doc_vectors):
            small, big = (qvec, dvec) if len(qvec) <= len(dvec) else (dvec, qvec)
            sim = sum(weight * big.get(term, 0.0) for term, weight in small.items())
            if sim >= _MIN_SCORE:
                scored.append((sim, i))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self.examples[i] for _, i in scored[:k]]


_index = None
_index_load_failed = False


def _get_index():
    global _index, _index_load_failed
    if _index is not None or _index_load_failed:
        return _index
    try:
        with open(_EXAMPLES_PATH, encoding="utf-8") as f:
            examples = json.load(f)
        _index = _TfidfIndex(examples)
    except Exception:
        _index_load_failed = True
        _index = None
    return _index


def retrieve_examples(prompt: str, k: int = 4) -> list:
    """The k example (prompt, grounding) pairs most similar to `prompt`, or [] if none clear the similarity floor."""
    index = _get_index()
    if index is None:
        return []
    return index.query(prompt, k)


def format_grounding(prompt: str, k: int = 4) -> str:
    """Few-shot grounding block for the user message, or '' when no example is a close enough match."""
    examples = retrieve_examples(prompt, k=k)
    if not examples:
        return ""
    lines = ["Similar past requests and the correct way to handle them:"]
    for ex in examples:
        lines.append(f"- \"{ex['prompt']}\" -> {ex['grounding']}")
    return "\n".join(lines)
