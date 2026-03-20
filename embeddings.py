import os
import re
import pickle
import hashlib
import argparse
from typing import Optional

# Avoid occasional crashes/oversubscription on CPU-only environments.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "google/embeddinggemma-300m")
CLEANED_TEXT_DIR = os.environ.get("CLEANED_TEXT_DIR", os.path.join("html", "cleaned_text"))

MAX_CHARS = 900
OVERLAP = 150

MAX_FILES = None  # Set to int for debugging; None indexes all files.

CACHE_DIR = "embeddings_cache"
DEFAULT_INDEX_NAME = "embeddings_only_index"
CACHE_META_PATH = os.path.join(CACHE_DIR, f"{DEFAULT_INDEX_NAME}.pkl")
CACHE_FAISS_PATH = os.path.join(CACHE_DIR, f"{DEFAULT_INDEX_NAME}.faiss")
MAX_INDEX_PART_BYTES = int(os.environ.get("INDEX_PART_MAX_BYTES", str(100_000_000 - 1)))  # <10MB decimal safety

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


def _load_index(index_dir: str, index_name: str, cache_key: dict) -> None:
    global _chunks, _index
    meta_path = os.path.join(index_dir, f"{index_name}.pkl")
    faiss_path = os.path.join(index_dir, f"{index_name}.faiss")

    # Backward compatible: prefer single-file artifacts if present; otherwise
    # reconstruct from split parts created by newer versions.
    has_single = os.path.exists(meta_path) and os.path.exists(faiss_path)
    faiss_parts = _list_part_paths(index_dir=index_dir, index_name=index_name, artifact="faiss")
    pkl_parts = _list_part_paths(index_dir=index_dir, index_name=index_name, artifact="pkl")
    has_parts = bool(faiss_parts) and bool(pkl_parts)
    should_resave_legacy = has_single and not has_parts
    if not (has_single or has_parts):
        raise FileNotFoundError(
            f"Index artifacts not found in '{index_dir}' for index_name='{index_name}'. "
            f"Expected either legacy '{index_name}.faiss' + '{index_name}.pkl' or split parts."
        )

    try:
        if has_single:
            with open(meta_path, "rb") as f:
                obj = pickle.load(f)
        else:
            tmp_meta_path = os.path.join(index_dir, f"{index_name}.pkl.recon.tmp.pkl")
            try:
                _stitch_parts_to_file(index_dir=index_dir, index_name=index_name, artifact="pkl", out_path=tmp_meta_path)
                with open(tmp_meta_path, "rb") as f:
                    obj = pickle.load(f)
            finally:
                try:
                    os.remove(tmp_meta_path)
                except OSError:
                    pass
    except Exception as e:
        raise RuntimeError(f"Failed to load index metadata for '{index_name}' from cache.") from e

    # If cache_key is None, skip validation (for when html not sent in submission)
    if cache_key is not None and obj.get("cache_key") != cache_key:
        raise RuntimeError(f"Index cache key mismatch for '{index_name}'. Cache needs rebuild.")

    try:
        if has_single:
            _index = faiss.read_index(faiss_path)
        else:
            tmp_faiss_path = os.path.join(index_dir, f"{index_name}.faiss.recon.tmp.faiss")
            try:
                _stitch_parts_to_file(index_dir=index_dir, index_name=index_name, artifact="faiss", out_path=tmp_faiss_path)
                _index = faiss.read_index(tmp_faiss_path)
            finally:
                try:
                    os.remove(tmp_faiss_path)
                except OSError:
                    pass
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index for '{index_name}' from cache.") from e

    raw_chunks = obj.get("chunks", []) # Expected format: list[(filename, text)]
    if not isinstance(raw_chunks, list):
        raise RuntimeError(f"Invalid cache metadata format for '{index_name}': chunks must be a list.")
    if raw_chunks and not (isinstance(raw_chunks[0], (tuple, list)) and len(raw_chunks[0]) == 2):
        raise RuntimeError(f"Invalid cache metadata format for '{index_name}': chunk entries must be pairs.")
    _chunks = [(str(fn), str(txt)) for fn, txt in raw_chunks]

    if should_resave_legacy:
        print(
            "Detected legacy monolithic embeddings cache; splitting into <100MB parts for upload...",
            flush=True,
        )
        try:
            _split_file_to_parts(
                in_path=faiss_path,
                index_dir=index_dir,
                index_name=index_name,
                artifact="faiss",
                max_bytes=MAX_INDEX_PART_BYTES,
            )
            _split_file_to_parts(
                in_path=meta_path,
                index_dir=index_dir,
                index_name=index_name,
                artifact="pkl",
                max_bytes=MAX_INDEX_PART_BYTES,
            )

            # Remove legacy monolithic artifacts to keep submission size small.
            for p in [faiss_path, meta_path]:
                try:
                    os.remove(p)
                except OSError:
                    pass

            print("Legacy cache split/resaved into parts.", flush=True)
        except Exception as e:
            # Loading still succeeded; splitting is just a cache-format upgrade.
            print(f"Warning: failed to split legacy cache into parts: {e}", flush=True)


def _save_index(index_dir: str, index_name: str, cache_key: dict):
    global _index
    os.makedirs(index_dir, exist_ok=True)
    if _index is None:
        raise RuntimeError("Index is not built; cannot save cache.")

    # Write full artifacts to temp files, then split them into <100MB parts.
    tmp_faiss_path = os.path.join(index_dir, f"{index_name}.faiss.save.tmp.faiss")
    tmp_pkl_path = os.path.join(index_dir, f"{index_name}.pkl.save.tmp.pkl")
    try:
        faiss.write_index(_index, tmp_faiss_path)
        _split_file_to_parts(
            in_path=tmp_faiss_path,
            index_dir=index_dir,
            index_name=index_name,
            artifact="faiss",
            max_bytes=MAX_INDEX_PART_BYTES,
        )

        with open(tmp_pkl_path, "wb") as f:
            pickle.dump({"cache_key": cache_key, "chunks": _chunks}, f)
        _split_file_to_parts(
            in_path=tmp_pkl_path,
            index_dir=index_dir,
            index_name=index_name,
            artifact="pkl",
            max_bytes=MAX_INDEX_PART_BYTES,
        )

        # Remove legacy monolithic artifacts (to ensure submission upload stays small).
        legacy_faiss_path = os.path.join(index_dir, f"{index_name}.faiss")
        legacy_pkl_path = os.path.join(index_dir, f"{index_name}.pkl")
        for p in [legacy_faiss_path, legacy_pkl_path]:
            try:
                os.remove(p)
            except OSError:
                pass
    finally:
        for p in [tmp_faiss_path, tmp_pkl_path]:
            try:
                os.remove(p)
            except OSError:
                pass


def _list_part_paths(index_dir: str, index_name: str, artifact: str) -> list[str]:
    """
    List part files matching:
      {index_name}.{artifact}.partNNN
    and return them sorted by part number.
    """
    prefix = f"{index_name}.{artifact}.part"
    parts: list[tuple[int, str]] = []
    try:
        files = os.listdir(index_dir)
    except FileNotFoundError:
        return []

    for fn in files:
        if not fn.startswith(prefix):
            continue
        part_str = fn[len(prefix) :]
        if not part_str.isdigit():
            continue
        parts.append((int(part_str), os.path.join(index_dir, fn)))

    parts.sort(key=lambda x: x[0])
    return [p for _, p in parts]


def _split_file_to_parts(
    in_path: str,
    index_dir: str,
    index_name: str,
    artifact: str,
    max_bytes: int,
) -> None:
    """
    Split `in_path` into numbered parts, each <= `max_bytes`.

    Part files are:
      {index_name}.{artifact}.part000, part001, ...
    """
    # Clean old parts (prevents mixing parts from different index builds).
    for part_path in _list_part_paths(index_dir=index_dir, index_name=index_name, artifact=artifact):
        try:
            os.remove(part_path)
        except OSError:
            pass

    size = os.path.getsize(in_path)
    if size == 0:
        return

    num_parts = (size + max_bytes - 1) // max_bytes
    with open(in_path, "rb") as src:
        for part_idx in range(num_parts):
            part_path = os.path.join(index_dir, f"{index_name}.{artifact}.part{part_idx:03d}")
            remaining = min(max_bytes, size - (part_idx * max_bytes))
            with open(part_path, "wb") as dst:
                while remaining > 0:
                    chunk = src.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    dst.write(chunk)
                    remaining -= len(chunk)


def _stitch_parts_to_file(index_dir: str, index_name: str, artifact: str, out_path: str) -> None:
    part_paths = _list_part_paths(index_dir=index_dir, index_name=index_name, artifact=artifact)
    if not part_paths:
        raise FileNotFoundError(
            f"No split parts found for index '{index_name}' artifact '{artifact}' in '{index_dir}'."
        )

    with open(out_path, "wb") as out_f:
        for part_path in part_paths:
            with open(part_path, "rb") as in_f:
                while True:
                    chunk = in_f.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)


def build_index(
    corpus_dir: str = CLEANED_TEXT_DIR,
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
    if not os.path.isdir(corpus_dir):
        print(f"Warning: '{corpus_dir}' directory not found, index is empty.")
        _chunks = []
        _index = None
        return

    files = [f for f in os.listdir(corpus_dir) if f.endswith(".txt") and "_pdf" not in f]
    files.sort()
    if max_files is not None:
        files = files[:max_files]

    cache_key = {
        "embedding_model": EMBEDDING_MODEL,
        "max_chars": max_chars,
        "overlap": overlap,
        "max_files": max_files,
        "files": {fn: _sha256_file(os.path.join(corpus_dir, fn)) for fn in files},
    }

    try:
        if _load_index(index_dir=index_dir, index_name=index_name, cache_key=cache_key):
            print("Loaded dense FAISS index from cache.")
            return
    except FileNotFoundError:
        pass
    except Exception as e:
        # Any cache load failure should trigger a rebuild.
        print(f"Cache load failed ({e}); rebuilding dense FAISS index...", flush=True)

    print(f"Building dense FAISS index from {len(files)} files...", flush=True)

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

    for i, fn in enumerate(files):
        if i % 200 == 0:
            print(f"  processed_files={i}/{len(files)} chunks={len(built)}", flush=True)

        path = os.path.join(corpus_dir, fn)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        for chunk_text in _chunk_text(text, max_chars=max_chars, overlap=overlap):
            if max_total_chunks is not None and (len(built) + len(pending_chunks)) >= max_total_chunks:
                break
            pending_chunks.append((fn, chunk_text))
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
    corpus_dir: str = CLEANED_TEXT_DIR,
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
    if os.path.isdir(corpus_dir):
        files = [f for f in os.listdir(corpus_dir) if f.endswith(".txt") and "_pdf" not in f]
        files.sort()
        if max_files is not None:
            files = files[:max_files]
        cache_key = {
            "embedding_model": EMBEDDING_MODEL,
            "max_chars": max_chars,
            "overlap": overlap,
            "max_files": max_files,
            "files": {fn: _sha256_file(os.path.join(corpus_dir, fn)) for fn in files},
        }
    _load_index(index_dir=index_dir, index_name=index_name, cache_key=cache_key)
    print("Loaded document embeddings index.")


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
    p_build.add_argument("--corpus_dir", default=CLEANED_TEXT_DIR)
    p_build.add_argument("--index_dir", default=CACHE_DIR)
    p_build.add_argument("--index_name", default=DEFAULT_INDEX_NAME)
    p_build.add_argument("--max_files", type=int, default=None)
    p_build.add_argument("--max_chars", type=int, default=MAX_CHARS)
    p_build.add_argument("--overlap", type=int, default=OVERLAP)
    p_build.add_argument("--batch_size", type=int, default=64)

    p_query = sub.add_parser("query", help="Runtime: load index and run a single query")
    p_query.add_argument("question")
    p_query.add_argument("--top_k", type=int, default=5)
    p_query.add_argument("--corpus_dir", default=CLEANED_TEXT_DIR)
    p_query.add_argument("--index_dir", default=CACHE_DIR)
    p_query.add_argument("--index_name", default=DEFAULT_INDEX_NAME)
    p_query.add_argument("--max_files", type=int, default=None)
    p_query.add_argument("--max_chars", type=int, default=MAX_CHARS)
    p_query.add_argument("--overlap", type=int, default=OVERLAP)

    args = parser.parse_args()

    if args.cmd == "build":
        build_index(
            corpus_dir=args.corpus_dir,
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
            corpus_dir=args.corpus_dir,
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