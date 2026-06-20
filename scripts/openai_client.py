#!/usr/bin/env python3
import base64
import json
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any

from common import append_jsonl_path, utc_now


def load_image_as_data_url(image_path) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    with path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def response_to_dict(response: Any) -> dict:
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump(mode="json")
        except TypeError:
            return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if isinstance(response, dict):
        return response
    try:
        return json.loads(response.model_dump_json())
    except Exception:  # noqa: BLE001
        return {"repr": repr(response)}


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    raw = response_to_dict(response)
    if isinstance(raw.get("output_text"), str):
        return raw["output_text"]
    chunks = []
    for item in raw.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks)


def extract_finish_reason(raw_response: dict) -> str | None:
    status = raw_response.get("status")
    if status == "incomplete":
        details = raw_response.get("incomplete_details") or {}
        return details.get("reason") or "incomplete"
    if status and status not in {"completed", "queued", "in_progress"}:
        return status
    return None


def extract_output_tokens(raw_response: dict) -> int | None:
    usage = raw_response.get("usage") or {}
    value = usage.get("output_tokens")
    return value if isinstance(value, int) else None


def is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = exc.__class__.__name__.lower()
    retryable_names = ("timeout", "ratelimit", "rate_limit", "apierror", "apiconnection")
    return any(token in name for token in retryable_names)


def log_jsonl(path, row: dict) -> None:
    if path:
        append_jsonl_path(Path(path), row)


def _load_openai_client(api_key_env: str, timeout: int):
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Python package 'openai' is not installed. Install it with: pip install openai") from exc

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is not set")
    return OpenAI(api_key=api_key, timeout=timeout, max_retries=0)


def _call_responses(
    *,
    input_payload: list[dict],
    model: str,
    api_key_env: str,
    max_output_tokens: int,
    reasoning_effort: str = "xhigh",
    raw_log_path=None,
    error_log_path=None,
    timeout: int = 300,
    max_retries: int = 6,
) -> dict:
    try:
        client = _load_openai_client(api_key_env, timeout)
    except Exception as exc:  # noqa: BLE001
        error = {
            "created_at": utc_now(),
            "model": model,
            "error": str(exc),
            "retryable": False,
        }
        log_jsonl(error_log_path, error)
        return {
            "output_text": "",
            "raw_text": "",
            "raw_response": None,
            "finish_reason": None,
            "output_tokens": None,
            "error": str(exc),
        }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
                max_output_tokens=max_output_tokens,
                store=False,
                input=input_payload,
            )
            raw = response_to_dict(response)
            text = extract_output_text(response)
            finish_reason = extract_finish_reason(raw)
            result = {
                "output_text": text,
                "raw_text": text,
                "raw_response": raw,
                "finish_reason": finish_reason,
                "output_tokens": extract_output_tokens(raw),
                "error": None,
            }
            log_jsonl(
                raw_log_path,
                {
                    "created_at": utc_now(),
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "max_output_tokens": max_output_tokens,
                    "status": raw.get("status"),
                    "finish_reason": finish_reason,
                    "raw_response": raw,
                },
            )
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            retryable = is_retryable_error(exc)
            log_jsonl(
                error_log_path,
                {
                    "created_at": utc_now(),
                    "model": model,
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "retryable": retryable,
                    "error_type": exc.__class__.__name__,
                    "error": last_error,
                },
            )
            if not retryable or attempt >= max_retries:
                break
            sleep_seconds = min(90.0, (2**attempt) + random.uniform(0.0, 1.0))
            time.sleep(sleep_seconds)

    return {
        "output_text": "",
        "raw_text": "",
        "raw_response": None,
        "finish_reason": None,
        "output_tokens": None,
        "error": last_error or "unknown_openai_error",
    }


def call_gpt55_vision(
    image_path,
    prompt,
    model,
    api_key_env,
    max_output_tokens,
    reasoning_effort="xhigh",
    raw_log_path=None,
    error_log_path=None,
    timeout: int = 300,
    max_retries: int = 6,
) -> dict:
    image_data_url = load_image_as_data_url(image_path)
    input_payload = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt,
                },
                {
                    "type": "input_image",
                    "image_url": image_data_url,
                },
            ],
        }
    ]
    return _call_responses(
        input_payload=input_payload,
        model=model,
        api_key_env=api_key_env,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        raw_log_path=raw_log_path,
        error_log_path=error_log_path,
        timeout=timeout,
        max_retries=max_retries,
    )


def call_gpt55_text(
    prompt,
    model,
    api_key_env,
    max_output_tokens,
    reasoning_effort="xhigh",
    raw_log_path=None,
    error_log_path=None,
    timeout: int = 300,
    max_retries: int = 6,
) -> dict:
    input_payload = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt,
                }
            ],
        }
    ]
    return _call_responses(
        input_payload=input_payload,
        model=model,
        api_key_env=api_key_env,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        raw_log_path=raw_log_path,
        error_log_path=error_log_path,
        timeout=timeout,
        max_retries=max_retries,
    )
