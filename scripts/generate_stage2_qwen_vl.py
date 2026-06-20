#!/usr/bin/env python3
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from common import PROMPT_KEYS, append_jsonl, normalized_json_string, read_jsonl, utc_now, validate_stage2_json
from local_qwen_vl_client import LocalQwenVLClient, LocalVLMError


def group_stage1_prompt_rows(rows: list[dict]) -> dict:
    grouped = {}
    for row in rows:
        sample_id = row.get("id")
        prompt_id = row.get("prompt_id")
        if prompt_id in PROMPT_KEYS and row.get("prompt_valid", True):
            grouped.setdefault(sample_id, {})[prompt_id] = row
        elif isinstance(row.get("stage1"), dict):
            grouped.setdefault(sample_id, {})
            for key in PROMPT_KEYS:
                if key in row["stage1"]:
                    grouped[sample_id][key] = {
                        **row,
                        "prompt_id": key,
                        "raw_text": row["stage1"][key]["output"],
                        "prompt": row["stage1"][key].get("prompt", ""),
                    }
    return grouped


def render_prompt(template: str, sample: dict, stage1_rows: dict) -> str:
    return (
        template.replace("{LABEL}", sample["label"])
        .replace("{STAGE1_OUTPUT_1}", stage1_rows["prompt1"]["raw_text"].strip())
        .replace("{STAGE1_OUTPUT_2}", stage1_rows["prompt2"]["raw_text"].strip())
        .replace("{STAGE1_OUTPUT_3}", stage1_rows["prompt3"]["raw_text"].strip())
    )


def make_valid_row(sample: dict, row: dict, parsed_json: dict, warnings: list[str]) -> dict:
    return {
        "id": sample["id"],
        "image_path": sample["image_path"],
        "label": sample["label"],
        "label_id": sample["label_id"],
        "source_dataset": sample.get("source_dataset", "SIDSET"),
        "model": "Qwen3-VL-8B-Instruct",
        "model_path": row["model_path"],
        "attempt": row["attempt"],
        "stage2_json": parsed_json,
        "stage2_json_string": normalized_json_string(parsed_json),
        "validation_warnings": warnings,
        "validated_at": utc_now(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run local Qwen-VL Stage2 JSON synthesis.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--max_new_tokens_stage2", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompt_file", default="configs/prompts/stage2_prompt.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {row["id"]: row for row in read_jsonl(output_dir / "manifest.jsonl")}
    stage1_grouped = group_stage1_prompt_rows(read_jsonl(output_dir / "stage1_valid.jsonl"))
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        prompt_path = Path(__file__).resolve().parents[1] / args.prompt_file
    template = prompt_path.read_text(encoding="utf-8")

    raw_path = output_dir / "stage2_raw.jsonl"
    valid_path = output_dir / "stage2_valid.jsonl"
    invalid_path = output_dir / "stage2_invalid.jsonl"
    error_path = logs_dir / "local_vlm_errors.jsonl"
    retry_path = logs_dir / "stage2_retry.jsonl"

    existing_valid = {row["id"] for row in read_jsonl(valid_path)} if args.resume else set()

    try:
        client = LocalQwenVLClient(args.model_path)
    except LocalVLMError as exc:
        with error_path.open("a", encoding="utf-8") as f:
            append_jsonl(f, {"stage": "load", "model_path": args.model_path, "error": str(exc), "created_at": utc_now()})
        raise SystemExit(str(exc)) from exc

    mode = "a" if args.resume else "w"
    with raw_path.open(mode, encoding="utf-8") as raw_f, valid_path.open(mode, encoding="utf-8") as valid_f, invalid_path.open(mode, encoding="utf-8") as invalid_f, error_path.open("a", encoding="utf-8") as error_f, retry_path.open("a", encoding="utf-8") as retry_f:
        for sample_id, sample in tqdm(manifest.items(), desc="Qwen-VL Stage2", unit="image"):
            if sample_id in existing_valid:
                continue
            stage1_rows = stage1_grouped.get(sample_id, {})
            if not all(key in stage1_rows for key in PROMPT_KEYS):
                append_jsonl(invalid_f, {**sample, "validation_errors": ["missing_stage1_valid_prompt_outputs"], "created_at": utc_now()})
                continue
            prompt = render_prompt(template, sample, stage1_rows)
            final_row = None
            for attempt in range(3):
                retry_prompt = prompt
                if attempt:
                    retry_prompt += "\n\nPrevious output was invalid. Return ONLY corrected parseable JSON in the exact schema."
                try:
                    output = client.generate(retry_prompt, image_path=None, max_new_tokens=args.max_new_tokens_stage2)
                    error = None
                except LocalVLMError as exc:
                    output = ""
                    error = str(exc)
                    append_jsonl(error_f, {"stage": "stage2", "id": sample_id, "attempt": attempt, "error": error, "created_at": utc_now()})
                    if "OOM" in error:
                        raise SystemExit(error) from exc
                valid, parsed_json, _, errors, warnings = validate_stage2_json(output, sample["label"]) if error is None else (False, None, "", ["local_vlm_error"], [])
                final_row = {
                    **sample,
                    "model": "Qwen3-VL-8B-Instruct",
                    "model_path": str(Path(args.model_path).expanduser().resolve()),
                    "attempt": attempt,
                    "prompt": retry_prompt,
                    "raw_text": output,
                    "valid": valid,
                    "validation_errors": errors,
                    "validation_warnings": warnings,
                    "error": error,
                    "created_at": utc_now(),
                }
                append_jsonl(raw_f, final_row)
                raw_f.flush()
                if valid:
                    append_jsonl(valid_f, make_valid_row(sample, final_row, parsed_json, warnings))
                    valid_f.flush()
                    break
                if attempt < 2:
                    append_jsonl(retry_f, {"id": sample_id, "attempt": attempt, "next_attempt": attempt + 1, "validation_errors": errors, "created_at": utc_now()})
                    retry_f.flush()
            if not final_row or not final_row["valid"]:
                append_jsonl(invalid_f, final_row or {**sample, "validation_errors": ["stage2_failed"], "created_at": utc_now()})
                invalid_f.flush()


if __name__ == "__main__":
    main()
