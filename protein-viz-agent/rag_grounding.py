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

# An example scoring below this fraction of the top hit's score is treated
# as noise relative to that hit, not a second relevant pattern — see the
# comment in _TfidfIndex.query for the failure this prevents.
_RELATIVE_FLOOR = 0.5

# A learned example (S11) rides along only when it shares most of the
# prompt's weighted vocabulary — a paraphrase, not a topic neighbour. Set
# against the 2026-10-22 merge regressions: "describe the folding topology of
# this domain" vs "describe the prepared structure" (one shared word) must
# stay out.
_LEARNED_MIN = 0.6


def _tokenize(text: str) -> list:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class _TfidfIndex:
    """
    Hand-written examples (no "source" key) and learned ones (S11: harvested
    by evals/curate_examples.py, "source" set) are kept apart. The first
    alone set the IDF and fill the k slots exactly as before learned examples
    existed. A learned example only adds one line after them, and only when
    it is a close match (_LEARNED_MIN). Merging the two pools let every new
    example move every IDF and crowd hand-written reply guidance out of the
    k slots — the 2026-10-22 merge test lost 3 eval cases that way.
    """

    def __init__(self, examples: list):
        self.examples = examples
        docs_tokens = [_tokenize(ex["prompt"]) for ex in examples]
        self.learned = [bool(ex.get("source")) for ex in examples]

        doc_freq = {}
        for tokens, learned in zip(docs_tokens, self.learned):
            if learned:
                continue
            for term in set(tokens):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        n_docs = (len(docs_tokens) - sum(self.learned)) or 1
        self.idf = {
            term: math.log((1 + n_docs) / (1 + count)) + 1
            for term, count in doc_freq.items()
        }
        # A word only learned examples use weighs as a word seen once would.
        # Only learned-side vectors carry it, so hand-written scores never see it.
        rare = math.log((1 + n_docs) / 2) + 1
        self.learned_idf = dict(self.idf)
        for tokens, learned in zip(docs_tokens, self.learned):
            if learned:
                for term in tokens:
                    self.learned_idf.setdefault(term, rare)
        self.doc_vectors = [self._vectorize(tokens, self.learned_idf if learned else self.idf)
                            for tokens, learned in zip(docs_tokens, self.learned)]

    def _vectorize(self, tokens: list, idf_table: dict = None) -> dict:
        idf_table = self.idf if idf_table is None else idf_table
        term_freq = {}
        for term in tokens:
            term_freq[term] = term_freq.get(term, 0) + 1

        vec = {}
        norm_sq = 0.0
        for term, count in term_freq.items():
            idf = idf_table.get(term)
            if idf is None:
                continue
            weight = count * idf
            vec[term] = weight
            norm_sq += weight * weight

        norm = math.sqrt(norm_sq) or 1.0
        return {term: weight / norm for term, weight in vec.items()}

    def query(self, text: str, k: int) -> list:
        tokens = _tokenize(text)
        found = self._query_hand(self._vectorize(tokens), k)
        if any(self.learned):
            lvec = self._vectorize(tokens, self.learned_idf)
            best = max(((self._sim(lvec, dvec), i) for i, dvec in enumerate(self.doc_vectors)
                        if self.learned[i]), default=(0.0, None))
            if best[0] >= _LEARNED_MIN:
                found.append(self.examples[best[1]])
        return found

    @staticmethod
    def _sim(qvec: dict, dvec: dict) -> float:
        small, big = (qvec, dvec) if len(qvec) <= len(dvec) else (dvec, qvec)
        return sum(weight * big.get(term, 0.0) for term, weight in small.items())

    def _query_hand(self, qvec: dict, k: int) -> list:
        if not qvec:
            return []

        scored = []
        for i, dvec in enumerate(self.doc_vectors):
            if self.learned[i]:
                continue
            sim = self._sim(qvec, dvec)
            if sim >= _MIN_SCORE:
                scored.append((sim, i))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        # A weak match riding along next to a strong one is worse than no
        # match at all: it doesn't add a second relevant pattern, it adds
        # an unrelated example's numbers to the context. One dominant hit
        # (e.g. an exact-prompt match) should stand alone; several
        # comparably-strong hits (e.g. two salt-bridge phrasings) should
        # all survive. Cutting anything under half the top score gets both
        # right — confirmed against this store's own examples, not just
        # reasoned about: "highlight residue 5" kept only its one genuine
        # match once this was added, instead of also dragging in an
        # unrelated dihedral example that happened to repeat "57" four
        # times and which qwen2.5:7b then echoed into a real tool call
        # instead of the "5" actually in the prompt.
        if scored:
            floor = max(_MIN_SCORE, scored[0][0] * _RELATIVE_FLOOR)
            scored = [pair for pair in scored if pair[0] >= floor]

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


def _stem(term: str) -> str:
    # Just enough to meet "bonds"/"bond", "bridges"/"bridge", "colored"/"color";
    # a real stemmer would be a dependency for a few dozen tool documents.
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if term.endswith(suffix) and len(term) - len(suffix) >= 3 and not term.endswith("ss"):
            return term[: len(term) - len(suffix)] + repl
    return term


class DocRanker:
    """
    TF-IDF ranking of named documents, several per name allowed (S10: per
    tool its name, its description with argument descriptions, and each
    example prompt that uses it, kept apart so a long description does not
    drown the one word that names the tool). A name scores its best document.
    Same scheme as the example store, plus light stemming: a tool description
    says "bond", the user says "bonds".
    """

    def __init__(self, docs: list):
        self.names = [n for n, _ in docs]
        tokens = [self._tokens(text) for _, text in docs]
        df = {}
        for toks in tokens:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        n = len(tokens) or 1
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vectors = [self._vector(toks) for toks in tokens]

    @staticmethod
    def _tokens(text: str) -> list:
        return [_stem(t) for t in _tokenize(text.replace("_", " "))]

    def _vector(self, tokens: list) -> dict:
        tf = {}
        for t in tokens:
            if t in self.idf:
                tf[t] = tf.get(t, 0) + 1
        # Sublinear tf: a long description repeating "structure" ten times
        # should not outrank a short one that says it once.
        vec = {t: (1 + math.log(c)) * self.idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / norm for t, w in vec.items()}

    def scores(self, text: str) -> dict:
        """Best cosine similarity of `text` to each name's documents."""
        q = self._vector(self._tokens(text))
        best = {}
        for name, vec in zip(self.names, self.vectors):
            sim = sum(w * vec.get(t, 0.0) for t, w in q.items())
            best[name] = max(best.get(name, 0.0), sim)
        return best


def all_examples() -> list:
    """Every (prompt, grounding) pair in the example store, or [] when it failed to load."""
    index = _get_index()
    return list(index.examples) if index is not None else []


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
    lines = [
        "Similar past requests and the correct way to handle them (for the "
        "pattern only — always use the exact residue numbers, chains and "
        "names from THIS prompt, never a number from one of these examples):"
    ]
    for ex in examples:
        lines.append(f"- \"{ex['prompt']}\" -> {ex['grounding']}")
    return "\n".join(lines)
