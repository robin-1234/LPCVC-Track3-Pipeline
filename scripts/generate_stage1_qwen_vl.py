#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from common import PROMPT_KEYS, append_jsonl, read_jsonl, utc_now, validate_stage1_text
from local_qwen_vl_client import LocalQwenVLClient, LocalVLMError


def load_prompts(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    if list(prompts.keys()) != PROMPT_KEYS:
        raise SystemExit(f"Stage1 prompt config must contain keys {PROMPT_KEYS}: {path}")
    neutral_prefix = (
        "You are performing forensic analysis for AI-generated image detection. "
        "Think carefully internally before answering, but do not output hidden chain-of-thought.\n\n"
    )
    cleaned = {}
    for key, prompt in prompts.items():
        prompt = prompt.replace("GPT-5.5", "the local vision-language model").replace("xhigh", "careful")
        if "{LABEL}" not in prompt:
            raise SystemExit(f"Stage1 prompt {key} missing {{LABEL}} placeholder.")
        cleaned[key] = neutral_prefix + prompt
    return cleaned


def render_prompt(template: str, label: str) -> str:
    return template.replace("{LABEL}", label)


def index_valid_prompt_rows(rows: list[dict]) -> dict:
    valid = {}
    for row in rows:
        if row.get("prompt_valid"):
            valid[(row.get("id"), row.get("prompt_id"))] = row
    return valid


def parse_args():
    parser = argparse.ArgumentParser(description="Run local Qwen-VL Stage1 smoke annotation.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--max_new_tokens_stage1", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompt_config", default="configs/prompts/stage1_prompts.json")
    args = parser.parse_args()
    if args.max_new_tokens_stage1 != 500:
        parser.error("Stage1 max_new_tokens is fixed at 500. Use --max_new_tokens_stage1 500.")
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    prompt_path = Path(args.prompt_config)
    if not prompt_path.exists():
        prompt_path = Path(__file__).resolve().parents[1] / args.prompt_config
    prompts = load_prompts(prompt_path)

    raw_path = output_dir / "stage1_raw.jsonl"
    valid_path = output_dir / "stage1_valid.jsonl"
    failed_path = logs_dir / "stage1_failed.jsonl"
    error_path = logs_dir / "local_vlm_errors.jsonl"

    raw_rows = read_jsonl(raw_path) if args.resume else []
    valid_prompt_rows = index_valid_prompt_rows(read_jsonl(valid_path) if args.resume else [])
    raw_seen = {(row.get("id"), row.get("prompt_id")) for row in raw_rows if row.get("prompt_id")}

    try:
        client = LocalQwenVLClient(args.model_path)
    except LocalVLMError as exc:
        with error_path.open("a", encoding="utf-8") as f:
            append_jsonl(f, {"stage": "load", "model_path": args.model_path, "error": str(exc), "created_at": utc_now()})
        raise SystemExit(str(exc)) from exc

    mode = "a" if args.resume else "w"
    with raw_path.open(mode, encoding="utf-8") as raw_f, valid_path.open(mode, encoding="utf-8") as valid_f, failed_path.open(mode, encoding="utf-8") as failed_f, error_path.open("a", encoding="utf-8") as error_f:
        for sample in tqdm(read_jsonl(manifest_path), desc="Qwen-VL Stage1", unit="image"):
            for prompt_id in PROMPT_KEYS:
                key = (sample["id"], prompt_id)
                if args.resume and (key in valid_prompt_rows or key in raw_seen):
                    continue
                prompt = render_prompt(prompts[prompt_id], sample["label"])
                try:
                    output = client.generate(prompt, image_path=sample["image_path"], max_new_tokens=args.max_new_tokens_stage1)
                    error = None
                except LocalVLMError as exc:
                    output = ""
                    error = str(exc)
                    append_jsonl(error_f, {"stage": "stage1", "id": sample["id"], "prompt_id": prompt_id, "error": error, "created_at": utc_now()})
                    if "OOM" in error:
                        raise SystemExit(error) from exc
                valid, errors, warnings = validate_stage1_text(output, max_output_tokens=args.max_new_tokens_stage1)
                row = {
                    **sample,
                    "model": "Qwen3-VL-8B-Instruct",
                    "model_path": str(Path(args.model_path).expanduser().resolve()),
                    "prompt_id": prompt_id,
                    "attempt": 0,
                    "prompt": prompt,
                    "raw_text": output,
                    "prompt_valid": valid and error is None,
                    "validation_errors": errors if error is None else ["local_vlm_error"],
                    "validation_warnings": warnings,
                    "error": error,
                    "created_at": utc_now(),
                }
                append_jsonl(raw_f, row)
                if row["prompt_valid"]:
                    append_jsonl(valid_f, row)
                else:
                    append_jsonl(failed_f, row)
                raw_f.flush()
                valid_f.flush()
                failed_f.flush()
                error_f.flush()


if __name__ == "__main__":
    main()
