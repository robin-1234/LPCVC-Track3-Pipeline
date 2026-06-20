#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from common import PROMPT_KEYS, append_jsonl, append_jsonl_path, read_jsonl, utc_now, validate_stage1_text
from openai_client import call_gpt55_vision


MAX_ATTEMPTS = 3


def load_prompts(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    if list(prompts.keys()) != PROMPT_KEYS:
        raise SystemExit(f"Stage1 prompt config must contain keys in order {PROMPT_KEYS}: {path}")
    for prompt_id, prompt in prompts.items():
        if "{LABEL}" not in prompt:
            raise SystemExit(f"Stage1 prompt {prompt_id} is missing {{LABEL}}.")
    return prompts


def render_prompt(template: str, label: str) -> str:
    return template.replace("{LABEL}", label)


def raw_indexes(raw_rows: list[dict]) -> tuple[dict, dict]:
    records_by_pair: dict[tuple[str, str], list[dict]] = {}
    valid_outputs: dict[tuple[str, str], dict] = {}
    for row in raw_rows:
        sample_id = row.get("id")
        prompt_id = row.get("prompt_id")
        if not sample_id or prompt_id not in PROMPT_KEYS:
            continue
        records_by_pair.setdefault((sample_id, prompt_id), []).append(row)
        if row.get("prompt_valid") and (sample_id, prompt_id) not in valid_outputs:
            valid_outputs[(sample_id, prompt_id)] = row
    for rows in records_by_pair.values():
        rows.sort(key=lambda item: item.get("attempt", -1))
    return records_by_pair, valid_outputs


def next_attempt(records_by_pair: dict, sample_id: str, prompt_id: str) -> int:
    rows = records_by_pair.get((sample_id, prompt_id), [])
    if not rows:
        return 0
    return max(row.get("attempt", -1) for row in rows) + 1


def make_valid_row(sample: dict, valid_outputs: dict, model: str, reasoning_effort: str) -> dict:
    stage1 = {}
    attempts = []
    for prompt_id in PROMPT_KEYS:
        row = valid_outputs[(sample["id"], prompt_id)]
        attempts.append(row.get("attempt", 0))
        stage1[prompt_id] = {
            "prompt": row["prompt"],
            "output": row["raw_text"].strip(),
        }
    return {
        "id": sample["id"],
        "image_path": sample["image_path"],
        "label": sample["label"],
        "label_id": sample["label_id"],
        "source_dataset": sample.get("source_dataset", "SIDSET"),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "attempt": max(attempts) if attempts else 0,
        "stage1": stage1,
        "validated_at": utc_now(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage1 forensic reasoning with OpenAI Responses API.")
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
    parser.add_argument("--prompt_config", default="configs/prompts/stage1_prompts.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_output_tokens_stage1 != 500:
        raise SystemExit("Stage1 max_output_tokens is fixed at 500. Use --max_output_tokens_stage1 500.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    logs_dir = output_dir / "logs"
    manifest_path = output_dir / "manifest.jsonl"
    prompt_path = Path(args.prompt_config)
    if not prompt_path.exists():
        prompt_path = Path(__file__).resolve().parents[1] / args.prompt_config
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Missing Stage1 prompt config: {prompt_path}")

    logs_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(prompt_path)
    manifest = read_jsonl(manifest_path)

    raw_path = output_dir / "stage1_raw.jsonl"
    valid_path = output_dir / "stage1_valid.jsonl"
    failed_path = output_dir / "stage1_failed.jsonl"
    retry_path = logs_dir / "stage1_retry.jsonl"
    failed_log_path = logs_dir / "stage1_failed.jsonl"
    api_errors_path = logs_dir / "api_errors.jsonl"

    raw_rows = read_jsonl(raw_path) if args.resume else []
    records_by_pair, valid_outputs = raw_indexes(raw_rows)
    existing_valid_ids = {row["id"] for row in read_jsonl(valid_path)} if args.resume else set()
    mode = "a" if args.resume else "w"
    processed = 0

    with (
        raw_path.open(mode, encoding="utf-8") as raw_f,
        valid_path.open(mode, encoding="utf-8") as valid_f,
        failed_path.open(mode, encoding="utf-8") as failed_f,
        retry_path.open(mode, encoding="utf-8") as retry_f,
        failed_log_path.open(mode, encoding="utf-8") as failed_log_f,
    ):
        for sample in tqdm(manifest, desc="Stage1", unit="image"):
            sample_id = sample["id"]
            if sample_id in existing_valid_ids:
                continue

            image_path = Path(sample["image_path"]).expanduser()
            if not image_path.exists():
                failed_record = {
                    **sample,
                    "validation_errors": ["image_not_found"],
                    "created_at": utc_now(),
                }
                append_jsonl(failed_f, failed_record)
                append_jsonl(failed_log_f, failed_record)
                processed += 1
                continue

            if all((sample_id, prompt_id) in valid_outputs for prompt_id in PROMPT_KEYS):
                append_jsonl(valid_f, make_valid_row(sample, valid_outputs, args.model, args.reasoning_effort))
                existing_valid_ids.add(sample_id)
                processed += 1
                continue

            final_errors = []
            for attempt_round in range(MAX_ATTEMPTS):
                attempted_this_round = False
                for prompt_id in PROMPT_KEYS:
                    if (sample_id, prompt_id) in valid_outputs:
                        continue
                    attempt = next_attempt(records_by_pair, sample_id, prompt_id)
                    if attempt >= MAX_ATTEMPTS:
                        continue
                    if attempt != attempt_round and attempt_round < attempt:
                        continue

                    attempted_this_round = True
                    prompt = render_prompt(prompts[prompt_id], sample["label"])
                    result_error_logged = False
                    try:
                        result = call_gpt55_vision(
                            image_path=image_path,
                            prompt=prompt,
                            model=args.model,
                            api_key_env=args.api_key_env,
                            max_output_tokens=args.max_output_tokens_stage1,
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
                                "prompt_id": prompt_id,
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
                    valid, errors, warnings = validate_stage1_text(
                        result.get("raw_text"),
                        result.get("finish_reason"),
                        result.get("output_tokens"),
                        args.max_output_tokens_stage1,
                    )
                    if result.get("error"):
                        if not result_error_logged:
                            append_jsonl_path(
                                api_errors_path,
                                {
                                    "id": sample_id,
                                    "prompt_id": prompt_id,
                                    "attempt": attempt,
                                    "created_at": utc_now(),
                                    "error": result.get("error"),
                                },
                            )
                        valid = False
                        errors = ["api_error"]
                    raw_record = {
                        "id": sample_id,
                        "image_path": sample["image_path"],
                        "label": sample["label"],
                        "label_id": sample["label_id"],
                        "source_dataset": sample.get("source_dataset", "SIDSET"),
                        "model": args.model,
                        "reasoning_effort": args.reasoning_effort,
                        "max_output_tokens": args.max_output_tokens_stage1,
                        "prompt_id": prompt_id,
                        "attempt": attempt,
                        "prompt": prompt,
                        **result,
                        "prompt_valid": valid,
                        "validation_errors": errors,
                        "validation_warnings": warnings,
                        "created_at": utc_now(),
                    }
                    append_jsonl(raw_f, raw_record)
                    records_by_pair.setdefault((sample_id, prompt_id), []).append(raw_record)
                    if valid:
                        valid_outputs[(sample_id, prompt_id)] = raw_record

                if all((sample_id, prompt_id) in valid_outputs for prompt_id in PROMPT_KEYS):
                    append_jsonl(valid_f, make_valid_row(sample, valid_outputs, args.model, args.reasoning_effort))
                    existing_valid_ids.add(sample_id)
                    final_errors = []
                    break

                final_errors = []
                for prompt_id in PROMPT_KEYS:
                    if (sample_id, prompt_id) in valid_outputs:
                        continue
                    rows = records_by_pair.get((sample_id, prompt_id), [])
                    if rows:
                        final_errors.extend(f"{prompt_id}: {err}" for err in rows[-1].get("validation_errors", []))
                    else:
                        final_errors.append(f"{prompt_id}: missing_output")
                if attempt_round < MAX_ATTEMPTS - 1 and attempted_this_round:
                    append_jsonl(
                        retry_f,
                        {
                            "id": sample_id,
                            "attempt": attempt_round,
                            "next_attempt": attempt_round + 1,
                            "validation_errors": final_errors,
                            "created_at": utc_now(),
                        },
                    )

            if sample_id not in existing_valid_ids:
                failed_record = {
                    **sample,
                    "validation_errors": final_errors or ["stage1_not_valid_after_retries"],
                    "created_at": utc_now(),
                }
                append_jsonl(failed_f, failed_record)
                append_jsonl(failed_log_f, failed_record)

            processed += 1
            if processed % 50 == 0:
                raw_f.flush()
                valid_f.flush()
                failed_f.flush()
                retry_f.flush()
                failed_log_f.flush()

        raw_f.flush()
        valid_f.flush()
        failed_f.flush()
        retry_f.flush()
        failed_log_f.flush()

    print(f"[done] Stage1 processed={processed}, valid_total={len(existing_valid_ids)}")


if __name__ == "__main__":
    main()
