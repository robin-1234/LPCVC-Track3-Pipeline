#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import (
    count_invalid_reasons,
    normalized_json_string,
    read_jsonl,
    utc_now,
    validate_stage2_json,
    write_json,
    write_jsonl,
)


def valid_row_from_raw(row: dict, parsed_json: dict, warnings: list[str]) -> dict:
    return {
        "id": row["id"],
        "image_path": row["image_path"],
        "label": row["label"],
        "label_id": row["label_id"],
        "source_dataset": row.get("source_dataset", "SIDSET"),
        "model": row.get("model"),
        "reasoning_effort": row.get("reasoning_effort", "xhigh"),
        "attempt": row.get("attempt"),
        "stage2_json": parsed_json,
        "stage2_json_string": normalized_json_string(parsed_json),
        "validation_warnings": warnings,
        "validated_at": utc_now(),
    }


def raw_text_from_valid_row(row: dict) -> str:
    if isinstance(row.get("stage2_json_string"), str):
        return row["stage2_json_string"]
    if isinstance(row.get("stage2_json"), dict):
        return normalized_json_string(row["stage2_json"])
    return row.get("raw_text", "")


def parse_args():
    parser = argparse.ArgumentParser(description="Strictly revalidate Stage2 JSON outputs.")
    parser.add_argument("--output_dir", default="output_sidset_thinkfake_gpt55_xhigh")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_jsonl(output_dir / "manifest.jsonl")
    manifest_by_id = {row["id"]: row for row in manifest}
    raw_rows = read_jsonl(output_dir / "stage2_raw.jsonl")
    source_rows = raw_rows if raw_rows else read_jsonl(output_dir / "stage2_valid.jsonl")
    source_kind = "stage2_raw" if raw_rows else "stage2_valid"

    grouped = {}
    for row in source_rows:
        grouped.setdefault(row.get("id"), []).append(row)

    valid_rows = []
    invalid_rows = []
    seen_valid_ids = set()

    for sample_id, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x.get("attempt", -1))
        manifest_row = manifest_by_id.get(sample_id)
        final_invalid = None
        for row in rows:
            label = manifest_row["label"] if manifest_row else row.get("label")
            raw_text = row.get("raw_text", "") if source_kind == "stage2_raw" else raw_text_from_valid_row(row)
            valid, parsed_json, _, errors, warnings = validate_stage2_json(raw_text, label)
            if not manifest_row:
                valid = False
                errors = errors + ["sample_missing_from_manifest"]
            if valid:
                merged = {**row, **manifest_row}
                valid_rows.append(valid_row_from_raw(merged, parsed_json, warnings))
                seen_valid_ids.add(sample_id)
                final_invalid = None
                break
            final_invalid = {
                **row,
                "validation_errors": errors,
                "validation_warnings": warnings,
                "revalidated_at": utc_now(),
            }
        if final_invalid is not None:
            invalid_rows.append(final_invalid)

    missing_stage2 = [row for row in manifest if row["id"] not in grouped]
    for row in missing_stage2:
        invalid_rows.append({**row, "validation_errors": ["missing_stage2_output"], "revalidated_at": utc_now()})

    write_jsonl(output_dir / "stage2_valid.jsonl", valid_rows)
    write_jsonl(output_dir / "stage2_invalid.jsonl", invalid_rows)

    report = {
        "source": source_kind,
        "total_manifest": len(manifest),
        "total_stage2_raw": len(raw_rows),
        "total_valid": len(valid_rows),
        "total_invalid": len(invalid_rows),
        "invalid_reasons_count": count_invalid_reasons(invalid_rows),
        "missing_stage2_count": len(missing_stage2),
        "real_valid_count": sum(1 for row in valid_rows if row.get("label") == "REAL"),
        "fake_valid_count": sum(1 for row in valid_rows if row.get("label") == "AI-GENERATED"),
        "validated_at": utc_now(),
    }
    write_json(logs_dir / "validation_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
