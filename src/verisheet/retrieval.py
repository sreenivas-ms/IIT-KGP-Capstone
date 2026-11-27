"""BGE-small-en-v1.5 semantic retrieval over governed business definitions and tool docs.

Pipeline:
    approved docs --(build_approved_corpus)--> chunks --(DefinitionRetriever.build_index)-->
    FAISS IndexFlatIP + manifest.json --(search / retrieve_context)--> RetrievedContext

Safety: only allowlisted documentation is indexed (never gold answers, raw/uploaded data or
secrets); retrieved text is advisory reference material and is never executed or obeyed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from verisheet.embeddings import EmbeddingServiceError, create_encoder, embedding_provider
from verisheet.schemas import RetrievedContext, RetrievedItem

logger = logging.getLogger(__name__)

# BGE v1.5 convention: prefix *queries* (not passages) with this instruction for retrieval.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

MANIFEST_NAME = "manifest.json"
FAISS_NAME = "index.faiss"
EMBEDDINGS_NAME = "embeddings.npy"
MANIFEST_FORMAT = 1

DEFAULT_TOP_K = 5
MAX_TOP_K = 20
MAX_QUERY_CHARS = 1000
CHUNK_MAX_CHARS = 600
MAX_DOC_BYTES = 1_000_000
RRF_K = 60  # standard reciprocal-rank-fusion constant
LOW_SCORE_THRESHOLD = 0.5  # BGE cosine below this rarely indicates a relevant definition
SEARCH_MODES = ("hybrid", "dense", "bm25")

ALLOWED_SUFFIXES = {".md", ".txt"}
# Directory names that may hold gold answers, uploaded/raw workbook data or build output.
BLOCKED_DIR_NAMES = {
    "gold", "raw", "derived", "uploads", "quarantine", "artifacts", "results", "traces",
    ".git", ".venv", "venv", "node_modules", "__pycache__",
}
_BLOCKED_NAME_RE = re.compile(r"(^\.env)|secret|credential|password|gold|\.(pem|key|p12)$", re.I)
_SECRET_RE = re.compile(
    r"(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+=]{12,}"
    r"|sk-[A-Za-z0-9]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}",
    re.I,
)
_GOLD_PATH_RE = re.compile(r"(^|[/\\])gold([/\\#]|$)|gold_|_gold", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+(?=[A-Z`(])")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how", "i",
    "in", "is", "it", "of", "on", "or", "the", "this", "to", "was", "what", "when", "which",
    "with", "we", "our", "my", "should", "can", "me", "you",
}


class IndexNotBuiltError(RuntimeError):
    """Raised when search/load is attempted before an index exists or when it is inconsistent."""


# --------------------------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------------------------


class DefinitionRetriever:
    """Local embedding and FAISS retrieval service.

    The corpus should contain approved README/data dictionary/metric definitions/tool docs only.
    It must exclude gold answers, gold scripts, uploaded arbitrary cell text and secrets.
    Use `BAAI/bge-small-en-v1.5`; apply the model's documented query/document prompting convention.

    Attributes:
        mode: ``"hybrid"`` (dense + BM25 fused with RRF, default), ``"dense"`` or ``"bm25"``.
        corpus_version: hash of model + chunk ids/content of the loaded index, or ``None``.
        backend: ``"faiss"`` (IndexFlatIP) or ``"numpy"`` (exact, same ranking); see
            ``_default_backend``. ``embeddings.npy`` is always saved so either can load an index.
        embedding_provider: ``"api"`` (shared capstone embedding service) or ``"local"``
            (sentence-transformers); see ``verisheet.embeddings``.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", index_dir: str = "artifacts/index"):
        """Configure the embedding provider (API or local) and load a saved index if present.

        The encoder is loaded lazily on first embed so that loading/inspecting an index (and
        tests that inject an encoder) do not require the model download.
        """
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.mode = "hybrid"
        self.embedding_provider = embedding_provider()
        self.backend = _default_backend(self.embedding_provider)
        self.corpus_version: str | None = None
        self._encoder: Any = None
        self._faiss_index: Any = None
        self._embeddings: np.ndarray | None = None
        self._items: list[dict[str, str]] = []
        self._bm25: _BM25 | None = None
        if (self.index_dir / MANIFEST_NAME).exists():
            self.load()

    # -- public API -------------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._embeddings is not None and bool(self._items)

    def build_index(self, documents: Iterable[dict[str, str]]) -> None:
        """Embed approved documents and persist a FAISS index plus a provenance manifest.

        Each document requires `item_id`, `source_path`, `title`, `content`, and optional
        `source_type`. IDs and metadata must remain aligned with vector positions.
        """
        items = _validate_documents(documents)
        vectors = self._embed([_passage_text(item) for item in items], is_query=False)
        version = corpus_version_for(self.model_name, items)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / EMBEDDINGS_NAME, vectors)
        faiss = _try_import_faiss() if self.backend == "faiss" else None
        if faiss is not None:
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            faiss.write_index(index, str(self.index_dir / FAISS_NAME))
        manifest = {
            "format": MANIFEST_FORMAT,
            "model_name": self.model_name,
            "embedding_provider": self.embedding_provider,
            "query_instruction": QUERY_INSTRUCTION,
            "dimension": int(vectors.shape[1]),
            "corpus_version": version,
            "item_count": len(items),
            "items": [
                {"position": pos, **item, "content_sha256": _sha(item["content"])}
                for pos, item in enumerate(items)
            ],
        }
        _atomic_write_text(self.index_dir / MANIFEST_NAME, json.dumps(manifest, indent=2))
        self._activate(items, vectors, version)
        logger.info(
            "retrieval index built: corpus_version=%s items=%d dim=%d dir=%s",
            version, len(items), vectors.shape[1], self.index_dir,
        )

    def search(self, query: str, top_k: int = 5) -> RetrievedContext:
        """Return top-k relevant, source-attributed definitions for a query.

        Results must include similarity scores and source paths. Retrieval is advisory context;
        do not turn retrieved natural-language content into executable commands.

        Ranking uses ``self.mode``; the reported ``score`` is always the dense cosine
        similarity between query and chunk (comparable across modes).
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {SEARCH_MODES}")
        top_k = min(top_k, MAX_TOP_K)
        if not self.is_ready:
            self.load()

        text = query.strip()[:MAX_QUERY_CHARS]
        query_vector = self._embed([text], is_query=True)[0]
        cosine = self._embeddings @ query_vector  # vectors are L2-normalised
        pool = min(len(self._items), max(top_k * 4, 20))

        rankings: list[list[int]] = []
        if self.mode in ("hybrid", "dense"):
            rankings.append(self._dense_rank(query_vector, cosine, pool))
        if self.mode in ("hybrid", "bm25"):
            rankings.append(self._bm25.rank(text)[:pool])
        order = _reciprocal_rank_fusion(rankings) if len(rankings) > 1 else rankings[0]

        hits = [
            RetrievedItem(
                item_id=self._items[pos]["item_id"],
                source_path=self._items[pos]["source_path"],
                title=self._items[pos]["title"],
                content=self._items[pos]["content"],
                score=round(float(cosine[pos]), 4),
                source_type=self._items[pos]["source_type"],
            )
            for pos in order[:top_k]
        ]
        warnings: list[str] = []
        if not hits:
            warnings.append("NO_RETRIEVAL_MATCH: no approved definition matched the query.")
        elif max(h.score for h in hits) < LOW_SCORE_THRESHOLD:
            warnings.append(
                f"LOW_RETRIEVAL_CONFIDENCE: best similarity {max(h.score for h in hits):.2f} is "
                f"below {LOW_SCORE_THRESHOLD:.2f}; retrieved definitions may not apply."
            )
        # Never log the query text itself: it may contain customer names or other PII.
        logger.debug("retrieval search: chars=%d mode=%s hits=%s", len(text), self.mode,
                     [h.item_id for h in hits])
        return RetrievedContext(query=text, items=hits, corpus_version=self.corpus_version or "",
                                warnings=warnings)

    def load(self) -> None:
        """Load an existing FAISS index and metadata manifest from `index_dir`."""
        manifest_path = self.index_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise IndexNotBuiltError(
                f"No retrieval index found in '{self.index_dir}'. Build it first with: "
                "python scripts/build_index.py"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexNotBuiltError(f"Retrieval manifest is unreadable: {exc}") from exc
        if manifest.get("format") != MANIFEST_FORMAT:
            raise IndexNotBuiltError("Retrieval manifest format is unsupported; rebuild the index.")
        if manifest.get("model_name") != self.model_name:
            raise IndexNotBuiltError(
                f"Index was built with '{manifest.get('model_name')}' but the retriever uses "
                f"'{self.model_name}'; rebuild the index."
            )

        built_with = manifest.get("embedding_provider")
        if built_with and built_with != self.embedding_provider:
            # Same model weights either way, but tiny numeric differences are possible.
            logger.warning("index built with %s embeddings, querying with %s; rebuild recommended",
                           built_with, self.embedding_provider)
        items = [
            {k: entry[k] for k in ("item_id", "source_path", "title", "content", "source_type")}
            for entry in sorted(manifest["items"], key=lambda e: e["position"])
        ]
        if corpus_version_for(self.model_name, items) != manifest.get("corpus_version"):
            raise IndexNotBuiltError("Manifest content does not match its corpus_version; rebuild.")

        embeddings_path = self.index_dir / EMBEDDINGS_NAME
        if not embeddings_path.exists():
            raise IndexNotBuiltError(f"Missing {EMBEDDINGS_NAME} in '{self.index_dir}'; rebuild.")
        vectors = np.load(embeddings_path, allow_pickle=False).astype(np.float32)
        if vectors.shape != (len(items), manifest.get("dimension")):
            raise IndexNotBuiltError("Embedding matrix does not align with manifest items; rebuild.")
        self._activate(items, vectors, manifest["corpus_version"])
        logger.info("retrieval index loaded: corpus_version=%s items=%d dir=%s",
                    self.corpus_version, len(items), self.index_dir)

    # -- internals --------------------------------------------------------------------------

    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            # Shared capstone embedding API when configured, else local sentence-transformers.
            self._encoder = create_encoder(self.model_name)
        return self._encoder

    def _embed(self, texts: list[str], is_query: bool) -> np.ndarray:
        if is_query:
            texts = [QUERY_INSTRUCTION + t for t in texts]
        vectors = self.encoder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise RuntimeError("Encoder returned an unexpected embedding shape.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def _activate(self, items: list[dict[str, str]], vectors: np.ndarray, version: str) -> None:
        self._items = items
        self._embeddings = np.ascontiguousarray(vectors, dtype=np.float32)
        self.corpus_version = version
        self._bm25 = _BM25([_passage_text(item) for item in items])
        self._faiss_index = None
        faiss = _try_import_faiss() if self.backend == "faiss" else None
        if faiss is not None:
            faiss_path = self.index_dir / FAISS_NAME
            index = faiss.read_index(str(faiss_path)) if faiss_path.exists() else None
            if index is None or index.ntotal != len(items):
                index = faiss.IndexFlatIP(self._embeddings.shape[1])
                index.add(self._embeddings)
            self._faiss_index = index

    def _dense_rank(self, query_vector: np.ndarray, cosine: np.ndarray, pool: int) -> list[int]:
        if self._faiss_index is not None:
            _, ids = self._faiss_index.search(query_vector.reshape(1, -1), pool)
            return [int(i) for i in ids[0] if i >= 0]
        # Exact numpy search: identical ranking to IndexFlatIP (used on macOS / without faiss).
        return [int(i) for i in np.argsort(-cosine, kind="stable")[:pool]]


# --------------------------------------------------------------------------------------------
# Corpus building
# --------------------------------------------------------------------------------------------


def build_approved_corpus(data_dictionary: str, tool_docs_dir: str, prompt_docs_dir: str) -> list[dict[str, str]]:
    """Collect and chunk allowlisted documentation for indexing, excluding gold material.

    Args:
        data_dictionary: path to the approved data dictionary / metric definitions (required).
        tool_docs_dir: directory of approved tool documentation (``.md``/``.txt``; optional).
        prompt_docs_dir: directory of approved prompt/rules documentation (optional).
    Returns:
        Chunk dictionaries with ``item_id``, ``source_path``, ``title``, ``content`` and
        ``source_type``, in deterministic order.
    Raises:
        FileNotFoundError: the data dictionary does not exist.
        ValueError: the data dictionary itself is in a blocked location (e.g. gold).
    """
    return collect_corpus(data_dictionary, tool_docs_dir, prompt_docs_dir, max_chars=CHUNK_MAX_CHARS)


def collect_corpus(
    data_dictionary: str | Path,
    tool_docs_dir: str | Path | None,
    prompt_docs_dir: str | Path | None,
    max_chars: int = CHUNK_MAX_CHARS,
) -> list[dict[str, str]]:
    """``build_approved_corpus`` with a configurable chunk size (used by chunking experiments)."""
    dictionary = Path(data_dictionary)
    if not dictionary.is_file():
        raise FileNotFoundError(f"Data dictionary not found: {dictionary}")
    if _blocked_reason(dictionary):
        raise ValueError(f"Data dictionary path is not allowlisted: {_blocked_reason(dictionary)}")

    sources: list[tuple[Path, str]] = [(dictionary, "business_definition")]
    for directory, source_type in ((tool_docs_dir, "tool_documentation"),
                                   (prompt_docs_dir, "prompt_documentation")):
        if directory is None:
            continue
        root = Path(directory)
        if not root.is_dir():
            logger.warning("approved docs directory missing, skipped: %s", root)
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            reason = _blocked_reason(path, root=root)
            if reason:
                logger.info("corpus file skipped (%s): %s", reason, path.name)
                continue
            sources.append((path, source_type))

    chunks: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    skipped = 0
    for path, source_type in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks_for_file, dropped = _chunk_document(text, path.stem, max_chars)
        skipped += dropped
        for title, anchor, content in chunks_for_file:
            base = f"{_slug(path.stem)}-{anchor}" if anchor else _slug(path.stem)
            item_id = base
            n = 2
            while item_id in seen_ids:
                item_id = f"{base}-{n}"
                n += 1
            seen_ids.add(item_id)
            chunks.append({
                "item_id": item_id,
                "source_path": _display_path(path) + (f"#{anchor}" if anchor else ""),
                "title": title,
                "content": content,
                "source_type": source_type,
            })
    if skipped:
        logger.warning("corpus paragraphs dropped for secret/PII patterns: %d", skipped)
    return chunks


def corpus_version_for(model_name: str, items: list[dict[str, str]]) -> str:
    """Deterministic corpus fingerprint: changes when the model or any chunk changes."""
    digest = hashlib.sha256(model_name.encode())
    for item in items:
        for key in ("item_id", "source_path", "title", "content", "source_type"):
            digest.update(b"\x1f" + item[key].encode("utf-8"))
        digest.update(b"\x1e")
    return f"sha256:{digest.hexdigest()[:16]}"


def _chunk_document(
    text: str, fallback_title: str, max_chars: int,
) -> tuple[list[tuple[str, str, str]], int]:
    """Split markdown into (title, anchor, content) chunks along headings, paragraphs and bullets.

    Units (paragraphs / list items) are packed into chunks of at most ``max_chars`` so that each
    chunk stays topically narrow (one or a few rules) for precise retrieval. Units containing
    secret-like or e-mail patterns are dropped individually. Returns (chunks, dropped_units).
    """
    sections: list[tuple[list[str], list[str]]] = []  # (heading path, lines)
    headings: list[tuple[int, str]] = []
    current: list[str] = []
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            sections.append(([h for _, h in headings], current))
            level = len(match.group(1))
            headings = [(lvl, h) for lvl, h in headings if lvl < level] + [(level, match.group(2))]
            current = []
        else:
            current.append(line)
    sections.append(([h for _, h in headings], current))

    chunks: list[tuple[str, str, str]] = []
    dropped = 0
    for path, lines in sections:
        units = _split_units(lines)
        safe = [u for u in units if not (_SECRET_RE.search(u) or _EMAIL_RE.search(u))]
        dropped += len(units) - len(safe)
        units = safe
        if not units:
            continue
        title = " > ".join(path) if path else fallback_title
        anchor = _slug(path[-1]) if path else ""
        for content in _pack_units(units, max_chars):
            chunks.append((title, anchor, content))
    return chunks, dropped


def _split_units(lines: list[str]) -> list[str]:
    units: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            unit = " ".join(part.strip() for part in buffer).strip()
            if unit:
                units.append(unit)
            buffer.clear()

    for line in lines:
        if not line.strip():
            flush()
        elif _LIST_ITEM_RE.match(line):
            flush()
            buffer.append(line)
        else:
            buffer.append(line)
    flush()
    return units


def _pack_units(units: list[str], max_chars: int) -> list[str]:
    pieces: list[str] = []
    for unit in units:
        if len(unit) <= max_chars:
            pieces.append(unit)
            continue
        # Oversized paragraph: split on sentence boundaries, hard-split only giant sentences.
        for sentence in _SENTENCE_SPLIT_RE.split(unit):
            while len(sentence) > max_chars:
                cut = sentence.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                pieces.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                pieces.append(sentence)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        # Merge a tiny trailing piece into its predecessor rather than indexing a fragment.
        if chunks and len(current) < max_chars // 4:
            chunks[-1] = f"{chunks[-1]}\n{current}"
        else:
            chunks.append(current)
    return chunks


def _blocked_reason(path: Path, root: Path | None = None) -> str | None:
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return "extension not allowlisted"
    if _BLOCKED_NAME_RE.search(path.name):
        return "blocked file name"
    resolved = path.resolve()
    parts = {p.lower() for p in resolved.parts}
    if parts & BLOCKED_DIR_NAMES:
        return "blocked directory"
    if root is not None and not resolved.is_relative_to(root.resolve()):
        return "outside approved directory (symlink?)"
    if path.stat().st_size > MAX_DOC_BYTES:
        return "file too large"
    return None


# --------------------------------------------------------------------------------------------
# Canonical entrypoint
# --------------------------------------------------------------------------------------------


def retrieve_context(
    query: str,
    workbook_profile,
    retriever: DefinitionRetriever | None = None,
) -> RetrievedContext:
    """Canonical retrieval entrypoint used by Planner and Orchestrator.

    Args:
        query: User question or metric/definition lookup string.
        workbook_profile: WorkbookProfile used to optionally scope applicable definitions.
        retriever: Optional injected DefinitionRetriever for tests/application dependency injection.
    Returns:
        RetrievedContext: ranked source-attributed definitions and corpus version.

    Behaviour:
        - Definitions that mention sheets/columns of the uploaded workbook are moved up by at
          most two ranks (scoping); scores stay the raw cosine similarity.
        - A missing/broken index or missing embedding library does not crash the agent loop:
          an empty context with a ``RETRIEVAL_UNAVAILABLE`` warning is returned instead.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    try:
        active = retriever if retriever is not None else _default_retriever()
        context = active.search(query, top_k=DEFAULT_TOP_K * 2)
    except (IndexNotBuiltError, EmbeddingServiceError, ImportError, OSError) as exc:
        logger.warning("retrieval unavailable: %s", type(exc).__name__)
        return RetrievedContext(
            query=query.strip()[:MAX_QUERY_CHARS],
            corpus_version="unavailable",
            warnings=[f"RETRIEVAL_UNAVAILABLE: {exc}"],
        )

    terms = _profile_terms(workbook_profile)
    items = context.items
    if terms:
        ranked = sorted(
            enumerate(items),
            # 1.1 (not 1) per match so a boosted item passes, rather than ties, its neighbour.
            key=lambda pair: pair[0] - 1.1 * min(2, _term_matches(pair[1], terms)),
        )
        items = [item for _, item in ranked]
    return context.model_copy(update={"items": items[:DEFAULT_TOP_K]})


@lru_cache(maxsize=1)
def _default_retriever() -> DefinitionRetriever:
    from verisheet.config import get_settings

    retriever = DefinitionRetriever(index_dir=str(Path(get_settings().artifact_dir) / "index"))
    if not retriever.is_ready:
        retriever.load()  # raises IndexNotBuiltError with build instructions
    return retriever


def _profile_terms(workbook_profile) -> set[str]:
    if workbook_profile is None:
        return set()
    names: list[str] = []
    for sheet in getattr(workbook_profile, "sheets", []) or []:
        names.append(sheet.name)
        names.extend(column.name for column in sheet.columns)
    terms = {_normalise_name(n) for n in names if isinstance(n, str)}
    return {t for t in terms if len(t) >= 3 and t not in {"data", "sheet1"}}


def _term_matches(item: RetrievedItem, terms: set[str]) -> int:
    haystack = f" {_normalise_name(item.title + ' ' + item.content)} "
    return sum(1 for term in terms if f" {term} " in haystack)


def _normalise_name(text: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return " ".join(_TOKEN_RE.findall(spaced.lower().replace("_", " ")))


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


class _BM25:
    """Minimal Okapi BM25 for exact-term matching (column names, codes like InvoiceNo)."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(_tokens(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def rank(self, query: str) -> list[int]:
        q_terms = set(_tokens(query))
        scores: list[tuple[float, int]] = []
        for pos, (doc, length) in enumerate(zip(self.docs, self.lengths)):
            score = 0.0
            for term in q_terms & doc.keys():
                tf = doc[term]
                norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                score += self.idf[term] * tf * (self.k1 + 1) / norm
            if score > 0:
                scores.append((score, pos))
        return [pos for _, pos in sorted(scores, key=lambda s: (-s[0], s[1]))]


def _reciprocal_rank_fusion(rankings: list[list[int]]) -> list[int]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, pos in enumerate(ranking):
            fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + rank + 1)
    return [pos for pos, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]


def _tokens(text: str) -> list[str]:
    return [t for t in _normalise_name(text).split() if t not in _STOPWORDS]


def _validate_documents(documents: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in documents:
        item = {}
        for key in ("item_id", "source_path", "title", "content"):
            value = doc.get(key) if isinstance(doc, dict) else None
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"document is missing a non-empty '{key}'")
            item[key] = value
        item["source_type"] = str(doc.get("source_type") or "business_definition")
        if item["item_id"] in seen:
            raise ValueError(f"duplicate item_id: {item['item_id']}")
        if _GOLD_PATH_RE.search(item["source_path"]):
            raise ValueError("gold material must never be indexed")
        seen.add(item["item_id"])
        items.append(item)
    if not items:
        raise ValueError("cannot build an index from an empty corpus")
    return items


def _passage_text(item: dict[str, str]) -> str:
    return f"{item['title']}\n{item['content']}"


def _slug(text: str) -> str:
    return "-".join(_TOKEN_RE.findall(text.lower())) or "section"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _default_backend(provider: str = "local") -> str:
    """Vector search backend: ``VERISHEET_VECTOR_BACKEND`` env var, else faiss except on macOS.

    On macOS the faiss-cpu and torch pip wheels each bundle their own ``libomp``; once FAISS has
    run, a later torch import in the same process aborts with "OMP: Error #15". Even with the
    embedding API, other code in the process (tests, tools) may load torch later, so macOS
    always defaults to exact numpy inner-product search, which ranks identically to
    ``IndexFlatIP``. Linux/Docker use FAISS. ``provider`` is kept for API compatibility.
    """
    choice = os.environ.get("VERISHEET_VECTOR_BACKEND", "").strip().lower()
    if choice in ("faiss", "numpy"):
        return choice
    if sys.platform == "darwin":
        return "numpy"
    return "faiss" if _try_import_faiss() is not None else "numpy"


def _try_import_faiss():
    try:
        import faiss  # noqa: PLC0415

        return faiss
    except ImportError:
        return None
