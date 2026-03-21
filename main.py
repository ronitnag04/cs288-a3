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
    if backend in {"bedrock"}:
        print("Using local LLM backend with Amazon Bedrock")
        from llm_local import call_llm as call_llm_local  # local-only dependency
        return call_llm_local

    print("Using OpenRouter LLM backend")
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
        "If there are multiple possible answers or people, don't return all options, just return the best or first one."
        "If the answer is a list of people, don't return all people, just return the best or first one."
        "Numerical answers should be returned as numbers, not words."
        "If the question can be answered with a Yes or No answer, return Yes or No.\n"
        "If the question is abstractive (requiring counting or simple arithmetic), follow these steps: 1. Extract all the numbers and labels from the text, 2. Explicitly write out the equations needed, and 3. Calculate the result for the answer.\n"
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


def _format_error_answer(err: Exception) -> str:
    name = err.__class__.__name__.lower()
    msg = str(err).strip().replace("\n", " ")
    if "timeout" in name or "timed out" in msg.lower():
        label = "Error: Timeout"
    elif "auth" in name or "credential" in msg.lower() or "permission" in msg.lower():
        label = "Error: Auth"
    elif "connection" in name or "connection" in msg.lower() or "endpoint" in msg.lower():
        label = "Error: Connection"
    elif "response" in name:
        label = "Error: Response"
    elif "config" in name or "valueerror" in name or "assertionerror" in name:
        label = "Error: Config"
    else:
        label = "Error: Runtime"

    if msg:
        short = msg[:120] + ("..." if len(msg) > 120 else "")
        return f"{label} - {short}"
    return label


def _default_verbose_log_path(predictions_out_path: str) -> str:
    base_path = os.path.splitext(predictions_out_path)[0]
    return base_path + '.log'

def _init_verbose_log(path: str, top_k: int) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Verbose QA log\n# top_k={top_k}\n\n")


def _append_verbose_log(
    path: str,
    question_idx: int,
    question: str,
    hits: list[dict[str, object]],
    answer: str,
) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"=== Question {question_idx} ===\n")
        f.write(f"Question: {question}\n")
        f.write("Top-k documents:\n")
        if not hits:
            f.write("  (no retrieved context)\n")
        for i, hit in enumerate(hits, 1):
            doc_id = str(hit.get("doc_id", "unknown"))
            score = float(hit.get("score", 0.0))
            text = str(hit.get("text", ""))
            f.write(f"  [{i}] doc_id={doc_id} score={score:.6f}\n")
            f.write(f"  text: {text}\n")
        f.write(f"Answer: {answer}\n\n")


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
    except Exception as e:
        return _format_error_answer(e)


def answer_question_with_context(
    question: str,
    llm_call: Callable[..., str],
    top_k: int = 5,
    timeout_s: float = 20.0,
    model: Optional[str] = None,
) -> tuple[str, list[dict[str, object]]]:
    scored_hits = embeddings.search_with_scores(question, top_k=top_k)
    hits_for_prompt = [(str(h["text"]), str(h["doc_id"])) for h in scored_hits]
    system_prompt, user_prompt = _format_prompt(question, hits_for_prompt)

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
            return _postprocess_answer(fut.result(timeout=timeout_s + 2.0)), scored_hits
    except Exception as e:
        return _format_error_answer(e), scored_hits


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
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--timeout_s", type=float, default=20)
    p.add_argument("--llm_model", default="")
    p.add_argument("--verbose", action="store_true", help="Write per-question retrieval + answer log")
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
    verbose_log_path = _default_verbose_log_path(args.predictions_out_path)
    if args.verbose:
        print(f"Verbose log path: {verbose_log_path}")
        _init_verbose_log(verbose_log_path, args.top_k)

    for i, q in enumerate(tqdm(questions, desc="Answering Questions", unit="q"), 1):
        if args.verbose:
            answer, scored_hits = answer_question_with_context(q, llm_call, args.top_k, args.timeout_s, model)
            answers.append(answer)
            _append_verbose_log(verbose_log_path, i, q, scored_hits, answer)
        else:
            answers.append(answer_question(q, llm_call, args.top_k, args.timeout_s, model))

    _write_answers(args.predictions_out_path, answers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
