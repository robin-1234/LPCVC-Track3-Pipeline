#!/usr/bin/env python3
import argparse
import io
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


REAL_TERMS = {"real", "authentic", "genuine"}
SYNTHETIC_TERMS = {
    "synthetic",
    "generated",
    "ai-generated",
    "ai_generated",
    "aigenerated",
    "full-synthetic",
    "full_synthetic",
    "fully-synthetic",
    "fully_synthetic",
}
TAMPERED_TERMS = {"tampered", "manipulated", "edited"}


def json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except ModuleNotFoundError:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)
        f.write("\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def normalize_label_name(value) -> str:
    text = str(value).strip().lower()
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def label_tokens(value) -> set[str]:
    normalized = normalize_label_name(value)
    tokens = set(normalized.split("_"))
    tokens.add(normalized)
    tokens.add(normalized.replace("_", "-"))
    return {token for token in tokens if token}


def classify_label_name(label_name: str) -> str | None:
    tokens = label_tokens(label_name)
    if tokens & REAL_TERMS:
        return "real"
    if tokens & TAMPERED_TERMS:
        return "tampered"
    if tokens & SYNTHETIC_TERMS:
        return "synthetic"
    if "ai" in tokens and "generated" in tokens:
        return "synthetic"
    return None


def hf_access_error_message(exc: Exception, hf_dataset: str) -> str:
    return (
        f"Failed to access Hugging Face dataset {hf_dataset!r}: {exc}\n"
        "If the dataset requires authentication or gated access, run:\n"
        "  huggingface-cli login\n"
        "or set:\n"
        "  export HF_TOKEN=\"your_hf_token\""
    )


def import_hf():
    try:
        import datasets
        from datasets import ClassLabel, Image, Value, get_dataset_config_names, get_dataset_split_names, load_dataset, load_dataset_builder
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency. Install required packages first:\n"
            "  python -m pip install -U datasets huggingface_hub pillow tqdm"
        ) from exc
    return {
        "datasets": datasets,
        "ClassLabel": ClassLabel,
        "Image": Image,
        "Value": Value,
        "get_dataset_config_names": get_dataset_config_names,
        "get_dataset_split_names": get_dataset_split_names,
        "hf_hub_download": hf_hub_download,
        "load_dataset": load_dataset,
        "load_dataset_builder": load_dataset_builder,
    }


def feature_to_schema(feature) -> dict:
    schema = {
        "type": feature.__class__.__name__,
        "repr": repr(feature),
    }
    if hasattr(feature, "dtype"):
        schema["dtype"] = feature.dtype
    if hasattr(feature, "names") and feature.names is not None:
        schema["names"] = list(feature.names)
    if hasattr(feature, "num_classes"):
        schema["num_classes"] = feature.num_classes
    return schema


def detect_image_column(features, sample_rows: list[dict], hf_types: dict) -> str:
    image_feature_columns = [
        name for name, feature in features.items() if isinstance(feature, hf_types["Image"]) or feature.__class__.__name__ == "Image"
    ]
    if len(image_feature_columns) == 1:
        return image_feature_columns[0]
    if len(image_feature_columns) > 1:
        preferred = [name for name in image_feature_columns if normalize_label_name(name) in {"image", "img", "photo", "picture"}]
        if len(preferred) == 1:
            return preferred[0]
        scores = {name: image_column_score(name, sample_rows) for name in image_feature_columns}
        best_score = max(scores.values())
        best = [name for name, score in scores.items() if score == best_score and score > 0]
        if len(best) == 1:
            return best[0]
        raise SystemExit(f"Ambiguous image columns detected from schema: {image_feature_columns}; sample_scores={scores}")

    image_like = []
    for name in features:
        for row in sample_rows:
            value = row.get(name)
            if value is None:
                continue
            if is_image_like_value(value):
                image_like.append(name)
                break
    if len(image_like) == 1:
        return image_like[0]
    if len(image_like) > 1:
        raise SystemExit(f"Ambiguous image columns detected from samples: {image_like}")
    raise SystemExit("Could not detect image column from Hugging Face dataset schema.")


def image_column_score(column: str, sample_rows: list[dict]) -> int:
    score = 0
    normalized = normalize_label_name(column)
    if normalized in {"image", "img", "photo", "picture"}:
        score += 10
    if normalized in {"mask", "segmentation", "alpha"}:
        score -= 10
    for row in sample_rows:
        value = row.get(column)
        if value is None:
            continue
        if hasattr(value, "mode") and hasattr(value, "size"):
            mode = str(value.mode).upper()
            if mode in {"RGB", "RGBA", "CMYK", "YCbCr".upper()}:
                score += 5
            elif mode in {"L", "P", "1"}:
                score += 1
            else:
                score += 2
        elif is_image_like_value(value):
            score += 2
    return score


def is_image_like_value(value) -> bool:
    if isinstance(value, (bytes, bytearray)):
        return True
    if isinstance(value, dict) and ("bytes" in value or "path" in value):
        return True
    if isinstance(value, str):
        lower = value.lower().split("?")[0]
        return lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    return hasattr(value, "save") and hasattr(value, "convert")


def is_scalar_label_value(value) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    return isinstance(value, (str, int))


def detect_label_column(features, sample_rows: list[dict], image_column: str, hf_types: dict) -> str:
    class_label_columns = [
        name for name, feature in features.items() if isinstance(feature, hf_types["ClassLabel"]) or feature.__class__.__name__ == "ClassLabel"
    ]
    if len(class_label_columns) == 1:
        return class_label_columns[0]
    if len(class_label_columns) > 1:
        raise SystemExit(f"Ambiguous ClassLabel columns detected: {class_label_columns}")

    candidate_scores = label_column_candidate_scores(features, sample_rows, image_column, hf_types)
    low_cardinality = list(candidate_scores)
    if len(low_cardinality) == 1:
        return low_cardinality[0]
    if len(low_cardinality) > 1:
        best_score = max(candidate_scores.values())
        best = [name for name, score in candidate_scores.items() if score == best_score and score > 0]
        if len(best) == 1:
            return best[0]
        raise SystemExit(
            f"Ambiguous low-cardinality label columns detected: {low_cardinality}; candidate_scores={candidate_scores}"
        )
    raise SystemExit("Could not detect label column from Hugging Face dataset schema.")


def label_column_candidate_scores(features, sample_rows: list[dict], image_column: str, hf_types: dict) -> dict:
    image_columns = set(image_columns_from_features(features, hf_types))
    candidate_scores = {}
    for name in features:
        if name == image_column or name in image_columns:
            continue
        values = []
        for row in sample_rows:
            if name not in row:
                continue
            value = row[name]
            if not is_scalar_label_value(value):
                values = []
                break
            values.append(value)
        unique = {str(value) for value in values}
        if values and 1 < len(unique) <= 20:
            candidate_scores[name] = label_column_score(name, values, sample_rows, image_columns)
    return candidate_scores


def label_column_score(column: str, values: list, sample_rows: list[dict], image_columns: set[str]) -> int:
    score = 0
    normalized = normalize_label_name(column)
    if normalized in {"label", "labels", "class", "classes", "category", "target"}:
        score += 10
    unique_values = {str(value) for value in values}
    if 1 < len(unique_values) <= 10:
        score += 2
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        score += 1

    value_to_categories = defaultdict(set)
    for row in sample_rows:
        value = row.get(column)
        if value is None:
            continue
        try:
            label_name = explicit_label_name_from_row(row, column, image_columns)
        except SystemExit:
            label_name = None
        category = classify_label_name(label_name) if label_name is not None else None
        if category:
            value_to_categories[str(value)].add(category)
    if value_to_categories and all(len(categories) == 1 for categories in value_to_categories.values()):
        score += 3
    if len({next(iter(categories)) for categories in value_to_categories.values() if categories}) > 1:
        score += 3
    return score


def image_columns_from_features(features, hf_types: dict) -> list[str]:
    return [
        name for name, feature in features.items() if isinstance(feature, hf_types["Image"]) or feature.__class__.__name__ == "Image"
    ]


def cast_image_columns_decode_false(dataset, image_columns: list[str], hf_types: dict):
    if not hasattr(dataset, "cast_column"):
        return dataset
    for column in image_columns:
        try:
            dataset = dataset.cast_column(column, hf_types["Image"](decode=False))
        except Exception:  # noqa: BLE001
            pass
    return dataset


def load_sample_rows(args, split_name: str, hf_types: dict, features, limit: int = 64) -> list[dict]:
    try:
        dataset = hf_types["load_dataset"](
            args.hf_dataset,
            split=split_name,
            streaming=True,
            token=os.environ.get("HF_TOKEN"),
        )
    except TypeError:
        dataset = hf_types["load_dataset"](
            args.hf_dataset,
            split=split_name,
            streaming=True,
            use_auth_token=os.environ.get("HF_TOKEN"),
        )
    dataset = cast_image_columns_decode_false(dataset, image_columns_from_features(features, hf_types), hf_types)
    rows = []
    for row in dataset:
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def load_streaming_split(args, split_name: str, image_columns: list[str], hf_types: dict):
    try:
        dataset = hf_types["load_dataset"](
            args.hf_dataset,
            split=split_name,
            streaming=True,
            token=os.environ.get("HF_TOKEN"),
        )
    except TypeError:
        dataset = hf_types["load_dataset"](
            args.hf_dataset,
            split=split_name,
            streaming=True,
            use_auth_token=os.environ.get("HF_TOKEN"),
        )
    return cast_image_columns_decode_false(dataset, image_columns, hf_types)


def get_hf_overview(args, hf_types: dict) -> dict:
    try:
        configs = hf_types["get_dataset_config_names"](args.hf_dataset, token=os.environ.get("HF_TOKEN"))
    except TypeError:
        configs = hf_types["get_dataset_config_names"](args.hf_dataset, use_auth_token=os.environ.get("HF_TOKEN"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(hf_access_error_message(exc, args.hf_dataset)) from exc
    config = configs[0] if configs else None

    try:
        splits = hf_types["get_dataset_split_names"](args.hf_dataset, token=os.environ.get("HF_TOKEN"))
    except TypeError:
        splits = hf_types["get_dataset_split_names"](args.hf_dataset, use_auth_token=os.environ.get("HF_TOKEN"))
    except Exception:
        splits = []

    try:
        builder = hf_types["load_dataset_builder"](args.hf_dataset, token=os.environ.get("HF_TOKEN"))
    except TypeError:
        builder = hf_types["load_dataset_builder"](args.hf_dataset, use_auth_token=os.environ.get("HF_TOKEN"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(hf_access_error_message(exc, args.hf_dataset)) from exc

    if not splits and builder.info.splits:
        splits = list(builder.info.splits.keys())
    if not splits:
        splits = ["train"]

    features = builder.info.features
    return {
        "configs": configs,
        "selected_config": config,
        "splits": splits,
        "builder": builder,
        "features": features,
    }


def read_dataset_card_label_mapping(args, hf_types: dict) -> dict:
    try:
        readme_path = hf_types["hf_hub_download"](
            repo_id=args.hf_dataset,
            repo_type="dataset",
            filename="README.md",
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception:
        return {}

    text = Path(readme_path).read_text(encoding="utf-8", errors="replace")
    mapping = {}
    for match in re.finditer(r"^\s*-\s*(\d+)\s*:\s*([^\n]+)$", text, flags=re.MULTILINE):
        raw_value = match.group(1)
        label_text = match.group(2).strip()
        category = classify_label_name(label_text)
        if category is None:
            continue
        mapping[raw_value] = {
            "label_name": representative_label_name(label_text, category),
            "label_text": label_text,
            "category": category,
            "source": "huggingface_dataset_card",
        }
    return mapping


def label_name_for_value(value, label_feature, label_value_mapping: dict | None = None) -> str | None:
    if label_value_mapping and str(value) in label_value_mapping:
        return label_value_mapping[str(value)]["label_name"]
    if label_feature is not None and label_feature.__class__.__name__ == "ClassLabel":
        try:
            return label_feature.int2str(int(value))
        except Exception:  # noqa: BLE001
            if isinstance(value, str):
                return value
            return None
    if isinstance(value, str):
        return value
    return None


def explicit_label_name_from_row(row: dict, label_column: str, image_columns: set[str]) -> str | None:
    matches = []
    for column, value in row.items():
        if column == label_column or column in image_columns:
            continue
        if not isinstance(value, str):
            continue
        category = classify_label_name(value)
        if category is None:
            continue
        matches.append((category, representative_label_name(value, category)))
    categories = {category for category, _ in matches}
    if len(categories) > 1:
        raise SystemExit(f"Conflicting explicit label cues in metadata row: {matches}")
    if not matches:
        return None
    return matches[0][1]


def representative_label_name(value: str, category: str) -> str:
    normalized = normalize_label_name(value)
    terms = sorted(label_tokens(value), key=len, reverse=True)
    if category == "real":
        for term in terms:
            if term in REAL_TERMS:
                return term
        return "real"
    if category == "tampered":
        for term in terms:
            if term in TAMPERED_TERMS:
                return term
        return "tampered"
    for term in terms:
        if term in SYNTHETIC_TERMS:
            return term
    if "ai" in normalized and "generated" in normalized:
        return "ai_generated"
    return "synthetic"


def original_label_for_manifest(value):
    if isinstance(value, (str, int)):
        return value
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return str(value)


def inspect_dataset(args) -> tuple[dict, dict, dict]:
    hf_types = import_hf()
    overview = get_hf_overview(args, hf_types)
    features = overview["features"]
    splits = overview["splits"]
    card_label_mapping = read_dataset_card_label_mapping(args, hf_types)
    sample_rows = load_sample_rows(args, splits[0], hf_types, features)
    if not sample_rows:
        raise SystemExit(f"No rows found in dataset {args.hf_dataset!r} split {splits[0]!r}.")

    image_column = detect_image_column(features, sample_rows, hf_types)
    image_columns = image_columns_from_features(features, hf_types)
    image_candidate_scores = {name: image_column_score(name, sample_rows) for name in image_columns}
    label_column = detect_label_column(features, sample_rows, image_column, hf_types)
    label_candidate_scores = label_column_candidate_scores(features, sample_rows, image_column, hf_types)
    label_feature = features[label_column]

    feature_schema = {name: feature_to_schema(feature) for name, feature in features.items()}
    hf_schema = {
        "hf_dataset": args.hf_dataset,
        "configs": overview["configs"],
        "selected_config": overview["selected_config"],
        "splits": splits,
        "features": feature_schema,
        "image_columns": image_columns,
        "image_column_candidates": image_columns,
        "image_column_candidate_scores": image_candidate_scores,
        "image_column": image_column,
        "label_column_candidates": list(label_candidate_scores),
        "label_column_candidate_scores": label_candidate_scores,
        "label_column": label_column,
        "label_feature": feature_to_schema(label_feature),
        "dataset_card_label_mapping": card_label_mapping,
    }
    return (
        hf_schema,
        {
            "hf_types": hf_types,
            "features": features,
            "label_feature": label_feature,
            "label_value_mapping": card_label_mapping,
        },
        overview,
    )


def scan_counts(args, hf_schema: dict, context: dict) -> dict:
    hf_types = context["hf_types"]
    label_feature = context["label_feature"]
    label_value_mapping = context.get("label_value_mapping", {})
    image_column = hf_schema["image_column"]
    image_columns = set(hf_schema.get("image_columns", [image_column]))
    label_column = hf_schema["label_column"]

    label_counts = Counter()
    category_counts = Counter()
    raw_label_examples = {}
    unknown_labels = set()
    label_mapping = {}
    raw_label_to_category = {}
    split_counts = Counter()
    total_rows = 0

    for split in hf_schema["splits"]:
        dataset = load_streaming_split(args, split, list(image_columns), hf_types)
        for split_index, row in enumerate(tqdm(dataset, desc=f"Scan {split}", unit="row")):
            if args.scan_only and not args.full_count and split_index >= args.scan_limit:
                break
            total_rows += 1
            split_counts[split] += 1
            raw_label = row.get(label_column)
            label_name = label_name_for_value(raw_label, label_feature, label_value_mapping)
            if label_name is None:
                label_name = explicit_label_name_from_row(row, label_column, image_columns)
            label_key = str(raw_label)
            label_counts[label_key] += 1
            raw_label_examples.setdefault(label_key, original_label_for_manifest(raw_label))

            category = classify_label_name(label_name) if label_name is not None else None
            if category is None:
                unknown_labels.add(label_key)
                category_counts["unknown"] += 1
            else:
                previous = raw_label_to_category.get(label_key)
                if previous is not None and previous != category:
                    raise SystemExit(
                        f"Conflicting label mapping for raw label {label_key!r}: {previous!r} vs {category!r}."
                    )
                raw_label_to_category[label_key] = category
                category_counts[category] += 1
                label_mapping[label_key] = {
                    "raw_label_example": raw_label_examples[label_key],
                    "label_name": label_name,
                    "category": category,
                    "source": label_value_mapping.get(label_key, {}).get("source", "metadata_or_feature"),
                }

    if unknown_labels:
        available = sorted(label_counts)
        raise SystemExit(
            "Dataset label naming is not explicit enough to safely map labels. "
            f"Available labels: {available}"
        )

    return {
        "total_rows": total_rows,
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "label_mapping": label_mapping,
        "real_candidate_count": category_counts["real"],
        "synthetic_fake_candidate_count": category_counts["synthetic"],
        "tampered_count": category_counts["tampered"],
        "unknown_count": category_counts["unknown"],
        "include_tampered": args.include_tampered,
        "scan_mode": "full_count" if args.full_count else "quick_scan",
        "scan_limit_per_split": None if args.full_count else args.scan_limit,
        "is_full_dataset_count": bool(args.full_count),
    }


def reservoir_add(heap: list, max_size: int, priority: float, candidate: dict) -> None:
    import heapq

    entry = (priority, candidate["source_original_index"], candidate)
    if len(heap) < max_size:
        heapq.heappush(heap, entry)
    elif priority > heap[0][0]:
        heapq.heapreplace(heap, entry)


def select_candidates(args, hf_schema: dict, context: dict) -> tuple[list[dict], list[dict], dict]:
    hf_types = context["hf_types"]
    label_feature = context["label_feature"]
    label_value_mapping = context.get("label_value_mapping", {})
    image_column = hf_schema["image_column"]
    image_columns = set(hf_schema.get("image_columns", [image_column]))
    label_column = hf_schema["label_column"]
    rng = random.Random(args.seed)
    real_heap = []
    fake_heap = []
    category_counts = Counter()
    label_counts = Counter()
    label_mapping = {}
    global_index = 0

    for split in hf_schema["splits"]:
        dataset = load_streaming_split(args, split, list(image_columns), hf_types)
        for split_index, row in enumerate(tqdm(dataset, desc=f"Select {split}", unit="row")):
            raw_label = row.get(label_column)
            label_name = label_name_for_value(raw_label, label_feature, label_value_mapping)
            if label_name is None:
                label_name = explicit_label_name_from_row(row, label_column, image_columns)
            category = classify_label_name(label_name) if label_name is not None else None
            label_key = str(raw_label)
            label_counts[label_key] += 1
            if category is None:
                global_index += 1
                continue
            category_counts[category] += 1
            label_mapping[label_key] = {
                "raw_label_example": original_label_for_manifest(raw_label),
                "label_name": label_name,
                "category": category,
                "source": label_value_mapping.get(label_key, {}).get("source", "metadata_or_feature"),
            }
            candidate = {
                "source_split": split,
                "source_split_index": split_index,
                "source_original_index": global_index,
                "source_original_label": original_label_for_manifest(raw_label),
                "source_label_name": label_name,
                "category": category,
            }
            priority = rng.random()
            if category == "real":
                reservoir_add(real_heap, args.num_real, priority, candidate)
            elif category == "synthetic" or (args.include_tampered and category == "tampered"):
                reservoir_add(fake_heap, args.num_fake, priority, candidate)
            global_index += 1

    if len(real_heap) < args.num_real or len(fake_heap) < args.num_fake:
        raise SystemExit(
            f"Not enough official SID-Set candidates. Required real={args.num_real} fake={args.num_fake}, "
            f"found real={category_counts['real']} synthetic={category_counts['synthetic']} "
            f"tampered={category_counts['tampered']} include_tampered={args.include_tampered}."
        )

    selected_real = [entry[2] for entry in sorted(real_heap, key=lambda item: item[2]["source_original_index"])]
    selected_fake = [entry[2] for entry in sorted(fake_heap, key=lambda item: item[2]["source_original_index"])]
    report = {
        "label_counts": dict(label_counts),
        "label_mapping": label_mapping,
        "real_candidate_count": category_counts["real"],
        "synthetic_fake_candidate_count": category_counts["synthetic"],
        "tampered_count": category_counts["tampered"],
        "selected_real": len(selected_real),
        "selected_fake": len(selected_fake),
        "include_tampered": args.include_tampered,
    }
    return selected_real, selected_fake, report


def image_to_rgb(value):
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: python -m pip install -U pillow") from exc

    if hasattr(value, "convert") and hasattr(value, "save"):
        image = value
    elif isinstance(value, (bytes, bytearray)):
        image = Image.open(io.BytesIO(value))
    elif isinstance(value, dict):
        if value.get("bytes") is not None:
            image = Image.open(io.BytesIO(value["bytes"]))
        elif value.get("path"):
            path = str(value["path"])
            if path.startswith(("http://", "https://")):
                with urllib.request.urlopen(path, timeout=120) as response:
                    image = Image.open(io.BytesIO(response.read()))
            else:
                image = Image.open(path)
        else:
            raise ValueError("image_dict_missing_bytes_or_path")
    elif isinstance(value, str):
        if value.startswith(("http://", "https://")):
            with urllib.request.urlopen(value, timeout=120) as response:
                image = Image.open(io.BytesIO(response.read()))
        else:
            image = Image.open(value)
    else:
        raise ValueError(f"unsupported_image_value_type: {type(value).__name__}")

    if image.mode in {"RGBA", "LA"} or ("transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def save_selected_images(args, hf_schema: dict, context: dict, selected_real: list[dict], selected_fake: list[dict]) -> list[dict]:
    hf_types = context["hf_types"]
    image_column = hf_schema["image_column"]
    output_root = Path(args.output_root).expanduser().resolve()
    real_dir = output_root / "real"
    fake_dir = output_root / "fake"
    metadata_dir = output_root / "metadata"
    log_path = metadata_dir / "download_log.jsonl"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    targets = {}
    manifest_by_global_index = {}
    for idx, candidate in enumerate(selected_real, start=1):
        sample_id = f"sidset_real_{idx:06d}"
        path = real_dir / f"{sample_id}.jpg"
        targets[candidate["source_original_index"]] = {**candidate, "sample_id": sample_id, "path": path, "label": "REAL", "label_id": 0}
    for idx, candidate in enumerate(selected_fake, start=1):
        sample_id = f"sidset_fake_{idx:06d}"
        path = fake_dir / f"{sample_id}.jpg"
        targets[candidate["source_original_index"]] = {**candidate, "sample_id": sample_id, "path": path, "label": "AI-GENERATED", "label_id": 1}

    remaining = set(targets)
    for split in hf_schema["splits"]:
        if not remaining:
            break
        dataset = load_streaming_split(args, split, [image_column], hf_types)
        for split_index, row in enumerate(tqdm(dataset, desc=f"Download {split}", unit="row")):
            remove_index = None
            for global_index in list(remaining):
                candidate = targets[global_index]
                if candidate["source_split"] == split and candidate["source_split_index"] == split_index:
                    output_path = candidate["path"]
                    if args.resume and output_path.exists() and output_path.stat().st_size > 0:
                        append_jsonl(
                            log_path,
                            {
                                "id": candidate["sample_id"],
                                "image_path": str(output_path),
                                "status": "skipped_existing",
                                "source_original_index": candidate["source_original_index"],
                                "source_original_label": candidate["source_original_label"],
                            },
                        )
                    else:
                        image = image_to_rgb(row[image_column])
                        image.save(output_path, format="JPEG", quality=95)
                        append_jsonl(
                            log_path,
                            {
                                "id": candidate["sample_id"],
                                "image_path": str(output_path),
                                "status": "saved",
                                "source_original_index": candidate["source_original_index"],
                                "source_original_label": candidate["source_original_label"],
                            },
                        )
                    manifest_by_global_index[global_index] = {
                        "id": candidate["sample_id"],
                        "image_path": str(output_path),
                        "label": candidate["label"],
                        "label_id": candidate["label_id"],
                        "source_dataset": "SID-Set",
                        "source_hf_dataset": args.hf_dataset,
                        "source_original_index": candidate["source_original_index"],
                        "source_original_label": candidate["source_original_label"],
                    }
                    remove_index = global_index
                    break
            if remove_index is not None:
                remaining.remove(remove_index)
            if not remaining:
                break

    if remaining:
        missing = [targets[index]["sample_id"] for index in sorted(remaining)]
        raise SystemExit(f"Failed to materialize selected images: {missing[:20]} total_missing={len(missing)}")

    return [manifest_by_global_index[index] for index in sorted(manifest_by_global_index)]


def save_one_image(row: dict, image_column: str, output_path: Path) -> None:
    image = image_to_rgb(row[image_column])
    image.save(output_path, format="JPEG", quality=95)


def direct_stream_download(args, hf_schema: dict, context: dict) -> tuple[list[dict], dict]:
    hf_types = context["hf_types"]
    label_feature = context["label_feature"]
    label_value_mapping = context.get("label_value_mapping", {})
    image_column = hf_schema["image_column"]
    image_columns = list(set(hf_schema.get("image_columns", [image_column])) | {image_column})
    label_column = hf_schema["label_column"]

    output_root = Path(args.output_root).expanduser().resolve()
    real_dir = output_root / "real"
    fake_dir = output_root / "fake"
    metadata_dir = output_root / "metadata"
    log_path = metadata_dir / "download_log.jsonl"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest_real = []
    manifest_fake = []
    label_counts = Counter()
    category_counts = Counter()
    label_mapping = {}
    unknown_labels = set()
    global_index = 0

    for split in hf_schema["splits"]:
        if len(manifest_real) >= args.num_real and len(manifest_fake) >= args.num_fake:
            break
        dataset = load_streaming_split(args, split, image_columns, hf_types)
        for split_index, row in enumerate(tqdm(dataset, desc=f"Download {split}", unit="row")):
            if len(manifest_real) >= args.num_real and len(manifest_fake) >= args.num_fake:
                break

            raw_label = row.get(label_column)
            label_name = label_name_for_value(raw_label, label_feature, label_value_mapping)
            if label_name is None:
                label_name = explicit_label_name_from_row(row, label_column, set(image_columns))
            category = classify_label_name(label_name) if label_name is not None else None
            label_key = str(raw_label)
            label_counts[label_key] += 1

            if category is None:
                unknown_labels.add(label_key)
                global_index += 1
                continue

            category_counts[category] += 1
            label_mapping[label_key] = {
                "raw_label_example": original_label_for_manifest(raw_label),
                "label_name": label_name,
                "category": category,
                "source": label_value_mapping.get(label_key, {}).get("source", "metadata_or_feature"),
            }

            should_save_real = category == "real" and len(manifest_real) < args.num_real
            should_save_fake = (
                len(manifest_fake) < args.num_fake
                and (category == "synthetic" or (args.include_tampered and category == "tampered"))
            )
            if not should_save_real and not should_save_fake:
                global_index += 1
                continue

            if should_save_real:
                sample_index = len(manifest_real) + 1
                sample_id = f"sidset_real_{sample_index:06d}"
                output_path = real_dir / f"{sample_id}.jpg"
                label = "REAL"
                label_id = 0
            else:
                sample_index = len(manifest_fake) + 1
                sample_id = f"sidset_fake_{sample_index:06d}"
                output_path = fake_dir / f"{sample_id}.jpg"
                label = "AI-GENERATED"
                label_id = 1

            if args.resume and output_path.exists() and output_path.stat().st_size > 0:
                status = "skipped_existing"
            else:
                save_one_image(row, image_column, output_path)
                status = "saved"

            append_jsonl(
                log_path,
                {
                    "id": sample_id,
                    "image_path": str(output_path),
                    "status": status,
                    "source_split": split,
                    "source_split_index": split_index,
                    "source_original_index": global_index,
                    "source_original_label": original_label_for_manifest(raw_label),
                    "source_label_name": label_name,
                },
            )
            manifest_row = {
                "id": sample_id,
                "image_path": str(output_path),
                "label": label,
                "label_id": label_id,
                "source_dataset": "SID-Set",
                "source_hf_dataset": args.hf_dataset,
                "source_original_index": global_index,
                "source_original_label": original_label_for_manifest(raw_label),
            }
            if should_save_real:
                manifest_real.append(manifest_row)
            else:
                manifest_fake.append(manifest_row)
            global_index += 1

    if unknown_labels:
        raise SystemExit(
            "Dataset label naming is not explicit enough to safely map labels. "
            f"Available observed labels: {sorted(label_counts)}"
        )
    if len(manifest_real) < args.num_real or len(manifest_fake) < args.num_fake:
        raise SystemExit(
            f"Not enough official SID-Set samples reached before stream ended. "
            f"Required real={args.num_real} fake={args.num_fake}, "
            f"saved real={len(manifest_real)} fake={len(manifest_fake)}. "
            f"Observed categories: {dict(category_counts)} include_tampered={args.include_tampered}."
        )

    report = {
        "label_counts_until_complete": dict(label_counts),
        "label_mapping": label_mapping,
        "observed_category_counts_until_complete": dict(category_counts),
        "selected_real": len(manifest_real),
        "selected_fake": len(manifest_fake),
        "include_tampered": args.include_tampered,
        "download_strategy": "stream_until_requested_counts_then_stop",
        "image_decode_policy": "Image columns are cast to decode=False; only selected rows are decoded for saving.",
    }
    return manifest_real + manifest_fake, report


def parse_args():
    parser = argparse.ArgumentParser(description="Download a balanced 2K SID-Set root from Hugging Face.")
    parser.add_argument("--hf_dataset", default="saberzl/SID_Set")
    parser.add_argument("--output_root", default="dataset/SID_Set_2k")
    parser.add_argument("--num_real", type=int, default=1000)
    parser.add_argument("--num_fake", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_tampered", type=parse_bool, default=False)
    parser.add_argument("--scan_only", action="store_true")
    parser.add_argument("--scan_limit", type=int, default=2000)
    parser.add_argument("--full_count", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    hf_schema, context, _ = inspect_dataset(args)
    write_json(metadata_dir / "hf_schema.json", hf_schema)
    if args.scan_only:
        scan_report = scan_counts(args, hf_schema, context)
        hf_schema["label_mapping"] = scan_report["label_mapping"]
        write_json(metadata_dir / "hf_schema.json", hf_schema)

        dataset_scan = {
            "hf_dataset": args.hf_dataset,
            "output_root": str(output_root),
            "splits": hf_schema["splits"],
            "image_column_candidates": hf_schema.get("image_column_candidates", []),
            "image_column_candidate_scores": hf_schema.get("image_column_candidate_scores", {}),
            "image_column": hf_schema["image_column"],
            "label_column_candidates": hf_schema.get("label_column_candidates", []),
            "label_column_candidate_scores": hf_schema.get("label_column_candidate_scores", {}),
            "label_column": hf_schema["label_column"],
            "observed_labels": sorted(scan_report["label_counts"]),
            **scan_report,
        }
        write_json(metadata_dir / "dataset_scan.json", dataset_scan)
        print(json.dumps(dataset_scan, ensure_ascii=False, indent=2, default=json_default))
        if args.full_count:
            print("[done] scan_only full_count completed; no images downloaded.")
        else:
            print("[done] scan_only quick scan completed; no images downloaded. Counts are from the scanned subset only.")
        return

    hf_schema["label_mapping"] = hf_schema.get("dataset_card_label_mapping", {})
    write_json(metadata_dir / "hf_schema.json", hf_schema)
    manifest, selection_report = direct_stream_download(args, hf_schema, context)
    write_jsonl(metadata_dir / "sidset_2k_manifest.jsonl", manifest)
    write_json(
        metadata_dir / "download_summary.json",
        {
            "hf_dataset": args.hf_dataset,
            "output_root": str(output_root),
            "num_real": args.num_real,
            "num_fake": args.num_fake,
            "seed": args.seed,
            **selection_report,
            "manifest_path": str(metadata_dir / "sidset_2k_manifest.jsonl"),
        },
    )
    print(f"[done] Wrote {len(manifest)} images and manifest rows under {output_root}")


if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
    except SystemExit as exc:
        sys.stdout.flush()
        sys.stderr.flush()
        code = exc.code if isinstance(exc.code, int) else 1
        os._exit(code)
