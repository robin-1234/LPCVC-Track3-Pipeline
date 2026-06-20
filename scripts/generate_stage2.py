#!/usr/bin/env python3
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from common import append_jsonl, append_jsonl_path, normalized_json_string, read_jsonl, utc_now, validate_stage2_json
from openai_client import call_gpt55_text


MAX_ATTEMPTS = 3


def render_prompt(template: str, sample: dict) -> str:
    stage1 = sample["stage1"]
    return (
        template.replace("{LABEL}", sample["label"])
        .replace("{STAGE1_OUTPUT_1}", stage1["prompt1"]["output"])
        .replace("{STAGE1_OUTPUT_2}", stage1["prompt2"]["output"])
        .replace("{STAGE1_OUTPUT_3}", stage1["prompt3"]["output"])
    )


def make_valid_row(sample: dict, record: dict, parsed_json: dict) -> dict:
    return {
        "id": sample["id"],
        "image_path": sample["image_path"],
        "label": sample["label"],
        "label_id": sample["label_id"],
        "source_dataset": sample.get("source_dataset", "SIDSET"),
        "model": record["model"],
        "reasoning_effort": record.get("reasoning_effort", "xhigh"),
        "attempt": record["attempt"],
        "stage2_json": parsed_json,
        "stage2_json_string": normalized_json_string(parsed_json),
        "validation_warnings": record.get("validation_warnings", []),
        "validated_at": utc_now(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage2 JSON synthesis with OpenAI Responses API.")
    parser.add_argument("--sidset_root", default=None)
    parser.add_argument("--output_dir", default="output_sidset_thinkfake_gpt55_xhigh")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_real", type=int, default=1000)
    parser.add_argument("--num_fake", type=int, default=1000)
    parser.add_argument("--max_output_tokens_stage1", type=int, default=500)
    parser.add_argument("--max_output_tokens_stage2", type=int, default=1200)
    parser.add_argument("--reasoning_effort", default="xhigh")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--prompt_file", default="configs/prompts/stage2_prompt.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    logs_dir = output_dir / "logs"
    stage1_valid_path = output_dir / "stage1_valid.jsonl"
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        prompt_path = Path(__file__).resolve().parents[1] / args.prompt_file
    if not stage1_valid_path.exists():
        raise SystemExit(f"Missing Stage1 valid file: {stage1_valid_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Missing Stage2 prompt file: {prompt_path}")

    logs_dir.mkdir(parents=True, exist_ok=True)
    stage1_rows = read_jsonl(stage1_valid_path)
    prompt_template = prompt_path.read_text(encoding="utf-8")

    raw_path = output_dir / "stage2_raw.jsonl"
    valid_path = output_dir / "stage2_valid.jsonl"
    invalid_path = output_dir / "stage2_invalid.jsonl"
    retry_path = logs_dir / "stage2_retry.jsonl"
    api_errors_path = logs_dir / "api_errors.jsonl"

    raw_rows = read_jsonl(raw_path) if args.resume else []
    raw_by_id: dict[str, list[dict]] = {}
    for row in raw_rows:
        raw_by_id.setdefault(row.get("id"), []).append(row)
    for rows in raw_by_id.values():
        rows.sort(key=lambda item: item.get("attempt", -1))

    existing_valid_ids = {row["id"] for row in read_jsonl(valid_path)} if args.resume else set()
    mode = "a" if args.resume else "w"
    processed = 0

    with (
        raw_path.open(mode, encoding="utf-8") as raw_f,
        valid_path.open(mode, encoding="utf-8") as valid_f,
        invalid_path.open(mode, encoding="utf-8") as invalid_f,
        retry_path.open(mode, encoding="utf-8") as retry_f,
    ):
        for sample in tqdm(stage1_rows, desc="Stage2", unit="image"):
            sample_id = sample["id"]
            if sample_id in existing_valid_ids:
                continue

            previous = raw_by_id.get(sample_id, [])
            recovered_valid = False
            for row in previous:
                if row.get("error"):
                    continue
                valid, parsed_json, _, errors, warnings = validate_stage2_json(row.get("raw_text", ""), sample["label"])
                if valid:
                    row["validation_errors"] = errors
                    row["validation_warnings"] = warnings
                    append_jsonl(valid_f, make_valid_row(sample, row, parsed_json))
                    existing_valid_ids.add(sample_id)
                    recovered_valid = True
                    break
            if recovered_valid:
                processed += 1
                continue

            start_attempt = previous[-1].get("attempt", -1) + 1 if previous else 0
            final_record = None
            for attempt in range(start_attempt, MAX_ATTEMPTS):
                prompt = render_prompt(prompt_template, sample)
                result_error_logged = False
                try:
                    result = call_gpt55_text(
                        prompt=prompt,
                        model=args.model,
                        api_key_env=args.api_key_env,
                        max_output_tokens=args.max_output_tokens_stage2,
                        reasoning_effort=args.reasoning_effort,
                        error_log_path=api_errors_path,
                        timeout=args.timeout,
                    )
                except Exception as exc:  # noqa: BLE001
                    result_error_logged = True
                    append_jsonl_path(
                        api_errors_path,
                        {
                            "id": sample_id,
                            "attempt": attempt,
                            "created_at": utc_now(),
                            "error": str(exc),
                        },
                    )
                    result = {
                        "output_text": "",
                        "raw_text": "",
                        "raw_response": None,
                        "finish_reason": None,
                        "output_tokens": None,
                        "error": str(exc),
                    }
                if result.get("error"):
                    if not result_error_logged:
                        append_jsonl_path(
                            api_errors_path,
                            {
                                "id": sample_id,
                                "attempt": attempt,
                                "created_at": utc_now(),
                                "error": result.get("error"),
                            },
                        )
                    valid = False
                    parsed_json = None
                    errors = ["api_error"]
                    warnings = []
                else:
                    valid, parsed_json, _, errors, warnings = validate_stage2_json(result.get("raw_text", ""), sample["label"])

                final_record = {
                    "id": sample_id,
                    "image_path": sample["image_path"],
                    "label": sample["label"],
                    "label_id": sample["label_id"],
                    "source_dataset": sample.get("source_dataset", "SIDSET"),
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "max_output_tokens": args.max_output_tokens_stage2,
                    "attempt": attempt,
                    "prompt": prompt,
                    **result,
                    "valid": valid,
                    "validation_errors": errors,
                    "validation_warnings": warnings,
                    "created_at": utc_now(),
                }
                append_jsonl(raw_f, final_record)
                raw_by_id.setdefault(sample_id, []).append(final_record)

                if valid:
                    append_jsonl(valid_f, make_valid_row(sample, final_record, parsed_json))
                    existing_valid_ids.add(sample_id)
                    break
                if attempt < MAX_ATTEMPTS - 1:
                    append_jsonl(
                        retry_f,
                        {
                            "id": sample_id,
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "validation_errors": errors,
                            "created_at": utc_now(),
                        },
                    )

            if sample_id not in existing_valid_ids:
                append_jsonl(
                    invalid_f,
                    final_record
                    or {
                        **sample,
                        "validation_errors": ["stage2_not_valid_after_retries"],
                        "created_at": utc_now(),
                    },
                )

            processed += 1
            if processed % 50 == 0:
                raw_f.flush()
                valid_f.flush()
                invalid_f.flush()
                retry_f.flush()

        raw_f.flush()
        valid_f.flush()
        invalid_f.flush()
        retry_f.flush()

    print(f"[done] Stage2 processed={processed}, valid_total={len(existing_valid_ids)}")


if __name__ == "__main__":
    main()
