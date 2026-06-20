#!/usr/bin/env python3
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROMPT_KEYS = ["prompt1", "prompt2", "prompt3"]
VALID_LABELS = {"REAL", "AI-GENERATED"}

CRITERIA = [
    "Lighting & Shadows Consistency",
    "Edges & Boundaries",
    "Texture & Resolution",
    "Perspective & Spatial Relationships",
    "Physical & Common Sense Logic",
    "Text & Symbols",
    "Human & Biological Structure Integrity",
    "Material & Object Details",
]

TOP_LEVEL_KEYS = ["per_criterion", "overall_likelihood"]
CRITERION_KEYS = ["criterion", "evidence", "aigc score"]
OVERALL_VALUES = {"Real", "Uncertain", "AI-Generated"}

LABEL_ONLY_OUTPUTS = {
    "real",
    "fake",
    "ai-generated",
    "ai generated",
    "this image is real",
    "this image is fake",
    "this image is ai-generated",
    "this image is ai generated",
    "the image is real",
    "the image is fake",
    "the image is ai-generated",
    "the image is ai generated",
}

TRUNCATION_FINAL_WORDS = {
    "and",
    "or",
    "but",
    "because",
    "with",
    "without",
    "while",
    "although",
    "however",
    "therefore",
    "to",
    "of",
    "for",
    "from",
    "in",
    "on",
    "by",
    "as",
    "at",
    "into",
}

TRUNCATION_FINAL_PHRASES = {
    "due to",
    "such as",
    "including",
    "based on",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")


def append_jsonl(file_obj, row: dict) -> None:
    file_obj.write(json_dumps(row) + "\n")


def append_jsonl_path(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        append_jsonl(f, row)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalized_json_string(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def normalize_dir_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def is_label_only(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = normalized.strip(" .!?\t\r\n\"'")
    if normalized in LABEL_ONLY_OUTPUTS:
        return True
    words = normalized.split()
    label_words = {"real", "fake", "ai-generated", "ai", "generated", "image", "this", "is", "the"}
    return 0 < len(words) <= 6 and all(word in label_words for word in words)


def has_incomplete_bullet(text: str) -> bool:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return True
    last_line = lines[-1]
    if re.match(r"^[-*]\s*$", last_line):
        return True
    if re.match(r"^[-*]\s+\S+", last_line) and last_line[-1] not in ".!?\"')]}":
        return True
    if re.match(r"^\d+[.)]\s*$", last_line):
        return True
    if re.match(r"^\d+[.)]\s+\S+", last_line) and last_line[-1] not in ".!?\"')]}":
        return True
    return False


def looks_truncated(
    text: str,
    finish_reason: str | None = None,
    output_tokens: int | None = None,
    max_output_tokens: int | None = None,
) -> bool:
    stripped = text.strip()
    if finish_reason in {"length", "max_tokens", "max_output_tokens", "token_limit"}:
        return True
    if not stripped:
        return True
    if stripped.endswith("..."):
        return True
    if stripped[-1] in ",:":
        return True

    lower = re.sub(r"\s+", " ", stripped.lower()).rstrip(" .!?\"')]}")
    if any(lower.endswith(phrase) for phrase in TRUNCATION_FINAL_PHRASES):
        return True
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", lower)
    if words and words[-1] in TRUNCATION_FINAL_WORDS:
        return True
    if has_incomplete_bullet(stripped):
        return True
    if output_tokens is not None and max_output_tokens:
        near_limit = output_tokens >= int(max_output_tokens * 0.95)
        if near_limit and stripped[-1] not in ".!?\"')]}":
            return True
    return False


def validate_stage1_text(
    text,
    finish_reason: str | None = None,
    output_tokens: int | None = None,
    max_output_tokens: int = 500,
) -> tuple[bool, list[str], list[str]]:
    errors = []
    warnings = []
    if not isinstance(text, str):
        return False, ["output_not_string"], warnings
    stripped = text.strip()
    if not stripped:
        errors.append("empty_output")
    if len(stripped) < 30:
        errors.append("too_short_lt_30_chars")
    if is_label_only(stripped):
        errors.append("label_only_or_short_answer")
    if stripped.startswith("{") or stripped.startswith("["):
        errors.append("json_like_output_not_allowed")
    if "```" in stripped:
        errors.append("markdown_code_fence_not_allowed")
    if looks_truncated(stripped, finish_reason, output_tokens, max_output_tokens):
        errors.append("suspected_truncation_or_incomplete_sentence")
    return len(errors) == 0, errors, warnings


def parse_stage2_json(raw_text: str) -> tuple[dict | None, str, list[str]]:
    if not isinstance(raw_text, str):
        return None, "", ["raw_output_not_string"]
    candidate = raw_text.strip()
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, candidate, [f"json_parse_error: {exc}"]
    return obj, candidate, []


def validate_stage2_json(raw_text: str, label: str) -> tuple[bool, dict | None, str, list[str], list[str]]:
    obj, candidate, errors = parse_stage2_json(raw_text)
    warnings = []
    if errors:
        return False, obj, candidate, errors, warnings

    if label not in VALID_LABELS:
        errors.append("invalid_manifest_label")
    if not isinstance(obj, dict):
        errors.append("top_level_not_object")
        return False, obj, candidate, errors, warnings
    if list(obj.keys()) != TOP_LEVEL_KEYS:
        errors.append("top_level_keys_or_order_invalid")

    per_criterion = obj.get("per_criterion")
    if not isinstance(per_criterion, list):
        errors.append("per_criterion_not_list")
    elif len(per_criterion) != len(CRITERIA):
        errors.append(f"per_criterion_length_{len(per_criterion)}_not_8")
    else:
        for idx, item in enumerate(per_criterion):
            if not isinstance(item, dict):
                errors.append(f"per_criterion[{idx}]_not_object")
                continue
            if list(item.keys()) != CRITERION_KEYS:
                errors.append(f"per_criterion[{idx}]_keys_or_order_invalid")
            criterion = item.get("criterion")
            if criterion != CRITERIA[idx]:
                errors.append(f"per_criterion[{idx}]_criterion_mismatch")
            evidence = item.get("evidence")
            if not isinstance(evidence, str):
                errors.append(f"per_criterion[{idx}]_evidence_not_string")
            elif not evidence.strip():
                errors.append(f"per_criterion[{idx}]_evidence_empty")
            score = item.get("aigc score")
            if isinstance(score, bool) or not isinstance(score, int):
                errors.append(f"per_criterion[{idx}]_aigc_score_not_int")
            elif score not in {0, 1}:
                errors.append(f"per_criterion[{idx}]_aigc_score_not_0_or_1")

    overall = obj.get("overall_likelihood")
    if not isinstance(overall, str):
        errors.append("overall_likelihood_not_string")
    elif overall not in OVERALL_VALUES:
        errors.append("overall_likelihood_invalid_value")

    scores = []
    if isinstance(per_criterion, list):
        for item in per_criterion:
            if isinstance(item, dict):
                score = item.get("aigc score")
                if isinstance(score, int) and not isinstance(score, bool):
                    scores.append(score)

    if label == "REAL" and overall != "Real":
        errors.append("label_REAL_requires_overall_Real")
    if label == "AI-GENERATED":
        if overall != "AI-Generated":
            errors.append("label_AI_GENERATED_requires_overall_AI_Generated")
        if 1 not in scores:
            errors.append("label_AI_GENERATED_requires_at_least_one_aigc_score_1")

    return len(errors) == 0, obj, candidate, errors, warnings


def count_invalid_reasons(invalid_rows: list[dict]) -> dict:
    counter = Counter()
    for row in invalid_rows:
        for reason in row.get("validation_errors", []):
            counter[reason] += 1
    return dict(counter)
