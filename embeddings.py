import os
import re
import pickle
import hashlib
import argparse
import json
from typing import Optional

# Avoid occasional crashes/oversubscription on CPU-only environments.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# CORPUS_PATH = os.environ.get("CORPUS_PATH", os.path.join("html", "cleaned_text"))
CORPUS_PATH = os.environ.get("CORPUS_PATH", os.path.join("html", "eecs_text_bs_rewritten.jsonl"))

MAX_CHARS = 900
OVERLAP = 150

MAX_FILES = None  # Set to int for debugging; None indexes all files.

CACHE_DIR = "embeddings_cache"
DEFAULT_INDEX_NAME = "embeddings_only_index"
CACHE_META_PATH = os.path.join(CACHE_DIR, f"{DEFAULT_INDEX_NAME}.pkl")
CACHE_FAISS_PATH = os.path.join(CACHE_DIR, f"{DEFAULT_INDEX_NAME}.faiss")

_chunks: list[tuple[str, str]] = [] # list of (filename, chunk_text)
_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)   
            print(f"Loaded embedding model {EMBEDDING_MODEL} from local cache")
        except Exception as e:
            print(f"Embedding model {EMBEDDING_MODEL} not found locally, downloading...")
            _model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"Downloaded embedding model {EMBEDDING_MODEL} to local cache")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
    """
    Embed many texts using the local embedding model.
    Returns shape (n, d) float32, L2-normalized.
    """
    model = _get_model()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(embs, dtype=np.float32)


def get_embedding(text: str) -> np.ndarray:
    """
    Embed a single string using an allowed local model (SentenceTransformers).
    Returns a float32 numpy vector.
    """
    return embed_texts([text], batch_size=1, show_progress_bar=False)[0]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_jsonl_corpus(corpus_path: str) -> bool:
    return os.path.isfile(corpus_path) and corpus_path.lower().endswith(".jsonl")


def _iter_corpus_docs(corpus_path: str, max_files: Optional[int]):
    """
    Yield (doc_id, text) pairs from either:
    - a directory of cleaned .txt files, or
    - a .jsonl file with one document per line.
    """
    if _is_jsonl_corpus(corpus_path):
        emitted = 0
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, 1):
                if max_files is not None and emitted >= max_files:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = str(obj.get("text", "")).strip()
                if not text:
                    continue
                url = str(obj.get("url", "")).strip()
                doc_id = url if url else f"{os.path.basename(corpus_path)}:{line_no}"
                yield (doc_id, text)
                emitted += 1
        return

    files = [f for f in os.listdir(corpus_path) if f.endswith(".txt") and "_pdf" not in f]
    files.sort()
    if max_files is not None:
        files = files[:max_files]
    for fn in files:
        path = os.path.join(corpus_path, fn)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            yield (fn, f.read())


def _build_cache_key(corpus_path: str, max_files: Optional[int], max_chars: int, overlap: int) -> dict:
    if _is_jsonl_corpus(corpus_path):
        return {
            "embedding_model": EMBEDDING_MODEL,
            "max_chars": max_chars,
            "overlap": overlap,
            "max_files": max_files,
            "corpus_type": "jsonl",
            "corpus_path": corpus_path,
            "corpus_sha256": _sha256_file(corpus_path),
        }

    files = [f for f in os.listdir(corpus_path) if f.endswith(".txt") and "_pdf" not in f]
    files.sort()
    if max_files is not None:
        files = files[:max_files]
    return {
        "embedding_model": EMBEDDING_MODEL,
        "max_chars": max_chars,
        "overlap": overlap,
        "max_files": max_files,
        "corpus_type": "txt_dir",
        "corpus_path": corpus_path,
        "files": {fn: _sha256_file(os.path.join(corpus_path, fn)) for fn in files},
    }


def _chunk_text(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    """Simple, robust chunking: sliding window over normalized text."""

    s = re.sub(r"\s+", " ", text).strip()
    if not s:
        return []

    chunks = []
    step = max(1, max_chars - overlap)
    for start in range(0, len(s), step):
        end = min(len(s), start + max_chars)
        chunk = s[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(s):
            break
    return chunks


def _load_index(index_dir: str, index_name: str, cache_key: dict) -> bool:
    global _chunks, _index
    meta_path = os.path.join(index_dir, f"{index_name}.pkl")
    faiss_path = os.path.join(index_dir, f"{index_name}.faiss")

    if not (os.path.exists(meta_path) and os.path.exists(faiss_path)):
        return False

    try:
        with open(meta_path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return False

    # If cache_key is None, skip validation (for when html not sent in submission)
    if cache_key is not None and obj.get("cache_key") != cache_key:
        return False

    try:
        _index = faiss.read_index(faiss_path)
    except Exception:
        return False

    raw_chunks = obj.get("chunks", []) # Expected format: list[(filename, text)]
    if not isinstance(raw_chunks, list):
        return False
    if raw_chunks and not (isinstance(raw_chunks[0], (tuple, list)) and len(raw_chunks[0]) == 2):
        return False
    _chunks = [(str(fn), str(txt)) for fn, txt in raw_chunks]

    return True


def _save_index(index_dir: str, index_name: str, cache_key: dict):
    global _index
    os.makedirs(index_dir, exist_ok=True)
    if _index is None:
        raise RuntimeError("Index is not built; cannot save cache.")
    faiss.write_index(_index, os.path.join(index_dir, f"{index_name}.faiss"))
    with open(os.path.join(index_dir, f"{index_name}.pkl"), "wb") as f:
        pickle.dump({"cache_key": cache_key, "chunks": _chunks}, f)


def build_index(
    corpus_path: str = CORPUS_PATH,
    index_dir: str = CACHE_DIR,
    index_name: str = DEFAULT_INDEX_NAME,
    max_files: Optional[int] = MAX_FILES,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
    batch_size: int = 64,
) -> None:
    """
    OFFLINE STEP (run once): chunk + embed all documents and write FAISS index + metadata.
    Runtime search should call `load_index(...)` once and then `search(...)` for each query.
    """
    global _chunks, _index
    if not (os.path.isdir(corpus_path) or _is_jsonl_corpus(corpus_path)):
        print(f"Warning: corpus '{corpus_path}' not found, index is empty.")
        _chunks = []
        _index = None
        return

    corpus_type = "JSONL" if _is_jsonl_corpus(corpus_path) else "directory"
    print("Starting dense embedding index build...", flush=True)
    print(f"  corpus_path={corpus_path}", flush=True)
    print(f"  corpus_type={corpus_type}", flush=True)
    print(f"  embedding_model={EMBEDDING_MODEL}", flush=True)
    print(
        f"  chunking=max_chars:{max_chars}, overlap:{overlap}, batch_size:{batch_size}, max_files:{max_files}",
        flush=True,
    )

    cache_key = _build_cache_key(
        corpus_path=corpus_path,
        max_files=max_files,
        max_chars=max_chars,
        overlap=overlap,
    )

    if _load_index(index_dir=index_dir, index_name=index_name, cache_key=cache_key):
        print("Loaded dense FAISS index from cache.", flush=True)
        return

    if _is_jsonl_corpus(corpus_path):
        source_msg = f"JSONL corpus '{corpus_path}'"
    else:
        files = [f for f in os.listdir(corpus_path) if f.endswith(".txt") and "_pdf" not in f]
        files.sort()
        if max_files is not None:
            files = files[:max_files]
        source_msg = f"{len(files)} files in '{corpus_path}'"
    print(f"Building dense FAISS index from {source_msg}...", flush=True)

    # Stream embedding/indexing to avoid holding all chunk texts in RAM at once.
    # Note: we still store chunk texts in metadata for prompting; this can be large.
    built: list[tuple[str, str]] = []
    pending_texts: list[str] = []
    pending_chunks: list[tuple[str, str]] = []

    max_total_chunks_env = os.environ.get("MAX_TOTAL_CHUNKS")
    max_total_chunks = int(max_total_chunks_env) if max_total_chunks_env else None

    _index = None
    dim: Optional[int] = None

    def flush_batch() -> None:
        nonlocal dim, pending_texts, pending_chunks
        global _index
        if not pending_texts:
            return
        embs = embed_texts(pending_texts, batch_size=batch_size, show_progress_bar=False)
        if dim is None:
            dim = int(embs.shape[1])
            _index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
        assert _index is not None
        _index.add(embs)
        built.extend(pending_chunks)
        pending_texts = []
        pending_chunks = []

    for i, (doc_id, text) in enumerate(_iter_corpus_docs(corpus_path, max_files=max_files)):
        if i % 200 == 0:
            print(f"  processed_docs={i} chunks={len(built)}", flush=True)
        for chunk_text in _chunk_text(text, max_chars=max_chars, overlap=overlap):
            if max_total_chunks is not None and (len(built) + len(pending_chunks)) >= max_total_chunks:
                break
            pending_chunks.append((doc_id, chunk_text))
            pending_texts.append(chunk_text)
            if len(pending_texts) >= batch_size:
                flush_batch()
        if max_total_chunks is not None and len(built) >= max_total_chunks:
            break

    flush_batch()

    _chunks = built
    if not _chunks or _index is None:
        _index = None
        print("Index built: 0 chunks.", flush=True)
        return

    print(f"Index built: {len(_chunks)} chunks.", flush=True)
    _save_index(index_dir=index_dir, index_name=index_name, cache_key=cache_key)


def load_index(
    corpus_path: str = CORPUS_PATH,
    index_dir: str = CACHE_DIR,
    index_name: str = DEFAULT_INDEX_NAME,
    max_files: Optional[int] = MAX_FILES,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
) -> None:
    """
    RUNTIME STEP: Load index + metadata. This must be called before evaluation queries.
    """
    cache_key = None
    if os.path.isdir(corpus_path) or _is_jsonl_corpus(corpus_path):
        cache_key = _build_cache_key(
            corpus_path=corpus_path,
            max_files=max_files,
            max_chars=max_chars,
            overlap=overlap,
        )
    ok = _load_index(index_dir=index_dir, index_name=index_name, cache_key=cache_key)
    if not ok:
        raise RuntimeError(
            f"Index not found or cache key mismatch. Build it first:\n"
            f"  python3 embeddings.py build --corpus_path {corpus_path}"
        )


def search(query, top_k=5):
    global _index
    if not _chunks or _index is None:
        build_index()
    if not _chunks or _index is None:
        return []

    q_emb = get_embedding(query).reshape(1, -1)
    scores, idxs = _index.search(q_emb, int(top_k))
    out = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0 or idx >= len(_chunks):
            continue
        fn, txt = _chunks[idx]
        out.append((txt, fn))
    return out


def _main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Offline: build and save FAISS index")
    p_build.add_argument("--corpus_path", default=CORPUS_PATH)
    p_build.add_argument("--index_dir", default=CACHE_DIR)
    p_build.add_argument("--index_name", default=DEFAULT_INDEX_NAME)
    p_build.add_argument("--max_files", type=int, default=None)
    p_build.add_argument("--max_chars", type=int, default=MAX_CHARS)
    p_build.add_argument("--overlap", type=int, default=OVERLAP)
    p_build.add_argument("--batch_size", type=int, default=64)

    p_query = sub.add_parser("query", help="Runtime: load index and run a single query")
    p_query.add_argument("question")
    p_query.add_argument("--top_k", type=int, default=5)
    p_query.add_argument("--corpus_path", default=CORPUS_PATH)
    p_query.add_argument("--index_dir", default=CACHE_DIR)
    p_query.add_argument("--index_name", default=DEFAULT_INDEX_NAME)
    p_query.add_argument("--max_files", type=int, default=None)
    p_query.add_argument("--max_chars", type=int, default=MAX_CHARS)
    p_query.add_argument("--overlap", type=int, default=OVERLAP)

    args = parser.parse_args()

    if args.cmd == "build":
        build_index(
            corpus_path=args.corpus_path,
            index_dir=args.index_dir,
            index_name=args.index_name,
            max_files=args.max_files,
            max_chars=args.max_chars,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )
        return 0

    if args.cmd == "query":
        load_index(
            corpus_path=args.corpus_path,
            index_dir=args.index_dir,
            index_name=args.index_name,
            max_files=args.max_files,
            max_chars=args.max_chars,
            overlap=args.overlap,
        )
        hits = search(args.question, top_k=args.top_k)
        for text, fn in hits:
            print(f"[{fn}] {text}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_main())