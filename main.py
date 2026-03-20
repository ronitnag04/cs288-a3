import argparse
import concurrent.futures as cf
import os
import sys
from typing import Callable, Optional

import embeddings
from tqdm import tqdm


def _get_llm_callable() -> Callable[..., str]:
    """
    Select LLM backend.

    - Default: use `llm.py` (OpenRouter wrapper) for autograder compliance.
    - Local dev: set LLM_BACKEND=bedrock to use `llm_local.py` (Amazon Bedrock).
    """
    backend = os.environ.get("LLM_BACKEND", "").strip().lower()
    if backend in {"bedrock", "aws", "local"}:
        from llm_local import call_llm as call_llm_local  # local-only dependency

        return call_llm_local

    from llm import call_llm as call_llm_openrouter

    return call_llm_openrouter


def _format_prompt(question: str, hits: list[tuple[str, str]]) -> tuple[str, str]:
    context_blocks = []
    for i, (chunk_text, filename) in enumerate(hits, 1):
        context_blocks.append(f"[{i}] source={filename}\n{chunk_text}")
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no retrieved context)"

    system = (
        "You are a factual QA assistant. Answer using ONLY the provided context.\n"
        "Return a short answer (ideally a short span, under ~10 words).\n"
        "If the answer is not in the context, return Unknown.\n"
        "Do not include citations, explanations, or newlines."
    )
    user = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    return system, user


def _postprocess_answer(s: str) -> str:
    if s is None:
        return "No Answer"
    s = str(s).strip()
    if not s:
        return "No Stripped Answer"
    # Force single line for Gradescope format.
    s = s.splitlines()[0].strip()
    # Strip surrounding quotes.
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s if s else "No Stripped Unquoted Answer"


def answer_question(
    question: str,
    llm_call: Callable[..., str],
    top_k: int = 5,
    timeout_s: float = 20.0,
    model: Optional[str] = None,
) -> str:
    hits = embeddings.search(question, top_k=top_k)
    system_prompt, user_prompt = _format_prompt(question, hits)

    def _call() -> str:
        kwargs = {
            "query": user_prompt,
            "system_prompt": system_prompt,
            "max_tokens": 64,
            "temperature": 0.0,
            "timeout": int(timeout_s),
        }
        if model:
            kwargs["model"] = model
        return llm_call(**kwargs)

    try:
        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call)
            return _postprocess_answer(fut.result(timeout=timeout_s + 2.0))
    except Exception:
        return "Error"


def _read_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n") for ln in f.readlines()]


def _write_answers(path: str, answers: list[str]) -> None:
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for a in answers:
            f.write(_postprocess_answer(a).replace("\n", " ") + "\n")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("questions_path")
    p.add_argument("predictions_out_path")
    p.add_argument("--index_dir", default="embeddings_cache")
    p.add_argument("--index_name", default="embeddings_only_index")
    p.add_argument("--corpus_path", default=os.path.join("html", "eecs_text_bs_rewritten.jsonl"))
    p.add_argument("--top_k", type=int, default=int(os.environ.get("TOP_K", "5")))
    p.add_argument("--timeout_s", type=float, default=float(os.environ.get("TIMEOUT_S", "20")))
    p.add_argument("--llm_model", default=os.environ.get("LLM_MODEL", ""))
    args = p.parse_args(argv)

    # Load retrieval index once (offline artifacts).
    embeddings.load_index(
        corpus_path=args.corpus_path,
        index_dir=args.index_dir,
        index_name=args.index_name,
        max_files=None,
    )

    llm_call = _get_llm_callable()
    model = args.llm_model.strip() or None

    questions = _read_questions(args.questions_path)
    answers: list[str] = []
    for q in tqdm(questions, desc="Answering Questions", unit="q",):
        answers.append(answer_question( q, llm_call, args.top_k, args.timeout_s, model))

    _write_answers(args.predictions_out_path, answers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
