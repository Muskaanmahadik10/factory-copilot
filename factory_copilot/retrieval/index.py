"""Hybrid retrieval: BM25 (exact terms like 'filter dP', 'SOP-C-202') fused with TF-IDF cosine (softer term overlap),
then a reranker that rewards section-level matches and document diversity.

Chunks are SOP *sections*, so every hit carries a precise citation (SOP id + section heading). The Index interface
is small on purpose: an embedding backend can replace `_scores` without touching the tools or the agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from factory_copilot.config import DATA_DIR

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

SYNONYMS = {
    "dp": "differential pressure", "temp": "temperature", "vib": "vibration", "um": "micrometre",
    "rul": "remaining useful life", "wo": "work order",
}


def _stem(t: str) -> str:
    """Tiny suffix stripper so 'approves', 'approval' and 'approved' share a stem. Ids like 'sop-c-201' are untouched."""
    if len(t) < 5 or any(ch.isdigit() for ch in t):
        return t
    for suf in ("ations", "ation", "ings", "ing", "ives", "ive", "als", "al", "ers", "er", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    out: list[str] = []
    for t in toks:
        out.append(_stem(t))
        if t in SYNONYMS:
            out.extend(_stem(w) for w in SYNONYMS[t].split())
    return out


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_id} §{self.section}"


@dataclass
class SearchHit:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {"citation": self.chunk.citation, "doc_id": self.chunk.doc_id, "title": self.chunk.title, "section": self.chunk.section, "text": self.chunk.text, "score": round(self.score, 4)}


def load_chunks(sop_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(sop_dir.glob("SOP-*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        head = lines[0].lstrip("# ").strip()
        doc_id, _, title = head.partition(":")
        section, buf = None, []
        for line in lines[1:] + ["## __end__"]:
            if line.startswith("## "):
                if section and "".join(buf).strip():
                    chunks.append(Chunk(doc_id.strip(), title.strip(), section, " ".join(buf).strip()))
                section, buf = line[3:].strip(), []
            else:
                buf.append(line.strip())
    return chunks


class HybridIndex:
    def __init__(self, chunks: list[Chunk], bm25_weight: float = 0.6):
        self.chunks = chunks
        self.w = bm25_weight
        self._docs = [f"{c.doc_id} {c.title} {c.section} {c.text}" for c in chunks]
        self._bm25 = BM25Okapi([tokenize(d) for d in self._docs])
        self._tfidf = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2), min_df=1, token_pattern=None)
        self._tfidf_matrix = self._tfidf.fit_transform(self._docs)

    def _scores(self, query: str) -> np.ndarray:
        bm = np.array(self._bm25.get_scores(tokenize(query)), dtype=float)
        bm = bm / (bm.max() + 1e-9)
        q = self._tfidf.transform([query])
        cos = np.asarray((self._tfidf_matrix @ q.T).todense()).ravel()
        cos = cos / (cos.max() + 1e-9)
        return self.w * bm + (1 - self.w) * cos

    def search(self, query: str, k: int = 4, diversity: bool = True) -> list[SearchHit]:
        s = self._scores(query)
        # Rerank: exact phrase (bigram) matches, 'Symptoms'/'Diagnosis' boost for symptom-like queries, exact SOP id mentions.
        ql = query.lower()
        q_toks = tokenize(query)
        q_bigrams = {f"{a} {b}" for a, b in zip(q_toks, q_toks[1:])}
        for i, c in enumerate(self.chunks):
            if q_bigrams:
                d_toks = tokenize(self._docs[i])
                d_bigrams = {f"{a} {b}" for a, b in zip(d_toks, d_toks[1:])}
                s[i] += 0.25 * len(q_bigrams & d_bigrams)
            if c.doc_id.lower() in ql:
                s[i] += 1.0
            if any(w in ql for w in ("rising", "falling", "high", "low", "spike", "drift", "above", "below", "trend")) and c.section.split(" ", 1)[-1] in ("Symptoms", "Diagnosis"):
                s[i] += 0.15
        order = np.argsort(-s)
        hits: list[SearchHit] = []
        seen_docs: dict[str, int] = {}
        for i in order:
            if s[i] <= 0:
                break
            c = self.chunks[i]
            if diversity and seen_docs.get(c.doc_id, 0) >= 2:
                continue
            seen_docs[c.doc_id] = seen_docs.get(c.doc_id, 0) + 1
            hits.append(SearchHit(c, float(s[i])))
            if len(hits) >= k:
                break
        return hits


def build_index(sop_dir: Path | None = None) -> HybridIndex:
    sop_dir = sop_dir or DATA_DIR / "sops"
    if not sop_dir.exists():
        from factory_copilot.data.sops import write_sops
        write_sops(sop_dir)
    return HybridIndex(load_chunks(sop_dir))


if __name__ == "__main__":
    idx = build_index()
    for q in ["coolant temperature rising and pressure falling", "position error spikes", "filter dP high", "who approves a work order"]:
        print(q, "->", json.dumps([h.chunk.citation for h in idx.search(q, 3)]))
