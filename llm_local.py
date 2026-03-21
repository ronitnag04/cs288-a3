import os
from typing import Optional

ALLOWED_MODELS = [
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-8b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2"
]

DEFAULT_MODEL = ALLOWED_MODELS[0]


def call_llm(
    query: str,
    system_prompt: str = "",
    model: Optional[str] = DEFAULT_MODEL,
    max_tokens: int = 64,
    temperature: float = 0.0,
    timeout: int = 30,
) -> str:
    """
    Local-development LLM backend using Amazon Bedrock.
    Set `LLM_BACKEND=bedrock` to use this backend instead of llm.py (used by autograder).
    """
    assert model in ALLOWED_MODELS, (
        f"Model '{model}' is not allowed. Allowed models: {ALLOWED_MODELS}"
    )

    try:
        import boto3  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "boto3 is required for Bedrock local development. Install it locally (pip/conda)."
        ) from e

    client = boto3.client("bedrock-runtime")

    # Prefer the newer Converse API when available.
    if hasattr(client, "converse"):
        messages = [{"role": "user", "content": [{"text": query}]}]
        kwargs = {}
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]
        resp = client.converse(
            modelId=model,
            messages=messages,
            inferenceConfig={
                "maxTokens": int(max_tokens),
                "temperature": float(temperature),
            },
            **kwargs,
        )
        parts = resp.get("output", {}).get("message", {}).get("content", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return ("".join(texts)).strip()

    raise RuntimeError(
        "Your boto3/bedrock-runtime client does not support the Converse API. "
        "Upgrade boto3 or adjust llm_local.py to use invoke_model for your chosen model."
    )

