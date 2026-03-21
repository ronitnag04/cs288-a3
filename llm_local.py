import os
from typing import Optional

ALLOWED_MODELS = [
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-8b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2"
]

DEFAULT_MODEL = ALLOWED_MODELS[0]


class LLMError(RuntimeError):
    """Base error for local LLM backend failures."""


class LLMTimeoutError(LLMError):
    """Request timed out."""


class LLMConnectionError(LLMError):
    """Network/endpoint connection failed."""


class LLMAuthError(LLMError):
    """Credentials or authorization failed."""


class LLMServiceError(LLMError):
    """Bedrock service returned an error."""


class LLMResponseError(LLMError):
    """Response payload was missing expected content."""


class LLMConfigError(LLMError):
    """Local backend is misconfigured."""


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
    if model not in ALLOWED_MODELS:
        raise LLMConfigError(
            f"Model '{model}' is not allowed. Allowed models: {ALLOWED_MODELS}"
        )

    try:
        import boto3  # type: ignore
    except Exception as e:  # pragma: no cover
        raise LLMConfigError(
            "boto3 is required for Bedrock local development. Install it locally (pip/conda)."
        ) from e

    client = boto3.client("bedrock-runtime")
    try:
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            ConnectTimeoutError,
            EndpointConnectionError,
            NoCredentialsError,
            ReadTimeoutError,
        )
    except Exception:  # pragma: no cover
        BotoCoreError = Exception  # type: ignore[assignment]
        ClientError = Exception  # type: ignore[assignment]
        ConnectTimeoutError = Exception  # type: ignore[assignment]
        EndpointConnectionError = Exception  # type: ignore[assignment]
        NoCredentialsError = Exception  # type: ignore[assignment]
        ReadTimeoutError = Exception  # type: ignore[assignment]

    # Prefer the newer Converse API when available.
    if hasattr(client, "converse"):
        messages = [{"role": "user", "content": [{"text": query}]}]
        kwargs = {}
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]
        try:
            resp = client.converse(
                modelId=model,
                messages=messages,
                inferenceConfig={
                    "maxTokens": int(max_tokens),
                    "temperature": float(temperature),
                },
                **kwargs,
            )
        except (ReadTimeoutError, ConnectTimeoutError, TimeoutError) as e:
            raise LLMTimeoutError(f"Bedrock request timed out: {e}") from e
        except (NoCredentialsError,) as e:
            raise LLMAuthError(f"AWS credentials error: {e}") from e
        except EndpointConnectionError as e:
            raise LLMConnectionError(f"Bedrock endpoint connection failed: {e}") from e
        except ClientError as e:
            err = e.response.get("Error", {}) if hasattr(e, "response") else {}
            code = str(err.get("Code", ""))
            msg = str(err.get("Message", e))
            if code in {"UnrecognizedClientException", "InvalidSignatureException", "AccessDeniedException"}:
                raise LLMAuthError(f"Bedrock auth/permission error [{code}]: {msg}") from e
            raise LLMServiceError(f"Bedrock client error [{code or 'Unknown'}]: {msg}") from e
        except BotoCoreError as e:
            raise LLMServiceError(f"Bedrock SDK error: {e}") from e
        except Exception as e:
            raise LLMServiceError(f"Unexpected Bedrock error: {e}") from e

        parts = resp.get("output", {}).get("message", {}).get("content", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        out = ("".join(texts)).strip()
        if not out:
            raise LLMResponseError(f"Bedrock response missing text content: {resp}")
        return out

    raise LLMConfigError(
        "Your boto3/bedrock-runtime client does not support the Converse API. "
        "Upgrade boto3 or adjust llm_local.py to use invoke_model for your chosen model."
    )

