#!/usr/bin/env python3
import argparse
import heapq
import io
import json
import random
import shutil
from pathlib import Path


LABEL_TO_TYPE = {
    0: "real",
    1: "full_synthetic",
    2: "tampered",
}


def append_jsonl(file_obj, row: dict) -> None:
    file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_to_int(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower().replace("-", "_").replace(" ", "_")
        if lower in {"0", "real", "real_image"}:
            return 0
        if lower in {"1", "full_synthetic", "fullsynthetic", "synthetic", "ai_generated"}:
            return 1
        if lower in {"2", "tampered", "manipulated"}:
            return 2
    return None


def get_label(item: dict):
    for key in ["label", "labels", "class", "target"]:
        if key in item:
            label = label_to_int(item[key])
            if label in LABEL_TO_TYPE:
                return label
    return None


def get_image_value(item: dict):
    for key in ["image", "img", "jpg", "png"]:
        if key in item:
            return item[key]
    for value in item.values():
        if hasattr(value, "save") or isinstance(value, (bytes, bytearray, str, dict)):
            return value
    return None


def to_pil_image(value):
    from PIL import Image

    if hasattr(value, "save") and hasattr(value, "convert"):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    raise ValueError("unsupported_image_value")


def verify_image(path: Path) -> tuple[bool, str | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            image.convert("RGB")
        if width <= 0 or height <= 0:
            return False, "non_positive_dimensions"
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def reservoir_add(heap: list, max_size: int, priority: float, index: int, item: dict) -> None:
    entry = (priority, index, item)
    if len(heap) < max_size:
        heapq.heappush(heap, entry)
    elif priority > heap[0][0]:
        heapq.heapreplace(heap, entry)


def select_candidates(args):
    from datasets import load_dataset

    rng = random.Random(args.seed)
    pool_size = max(max(args.num_real, args.num_fake) * 3, max(args.num_real, args.num_fake) + 100)
    real_heap = []
    full_synthetic_heap = []
    tampered_heap = []

    try:
        dataset = load_dataset(args.hf_dataset, split=args.split, streaming=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load Hugging Face dataset {args.hf_dataset!r} split {args.split!r}. "
            "No fallback dataset will be used."
        ) from exc

    for index, item in enumerate(dataset):
        label = get_label(item)
        if label not in LABEL_TO_TYPE:
            continue
        slim_item = {
            "image": get_image_value(item),
            "original_sidset_label": label,
            "original_sidset_type": LABEL_TO_TYPE[label],
            "source_index": index,
        }
        priority = rng.random()
        if label == 0:
            reservoir_add(real_heap, pool_size, priority, index, slim_item)
        elif label == 1:
            reservoir_add(full_synthetic_heap, pool_size, priority, index, slim_item)
        elif label == 2:
            reservoir_add(tampered_heap, pool_size, priority, index, slim_item)

    selected_real = [entry[2] for entry in sorted(real_heap, reverse=True)]
    full = [entry[2] for entry in sorted(full_synthetic_heap, reverse=True)]
    tampered = [entry[2] for entry in sorted(tampered_heap, reverse=True)]
    if args.prefer_fake_type != "full_synthetic":
        raise RuntimeError("Only --prefer_fake_type full_synthetic is supported for SIDSET.")
    selected_fake = full + tampered
    return selected_real, selected_fake, {"real_pool": len(real_heap), "full_synthetic_pool": len(full), "tampered_pool": len(tampered)}


def clean_output_dirs(output_root: Path) -> tuple[Path, Path, Path]:
    real_dir = output_root / "real"
    fake_dir = output_root / "fake"
    logs_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in [real_dir, fake_dir]:
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
        directory.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return real_dir, fake_dir, logs_dir


def save_class(candidates: list[dict], out_dir: Path, prefix: str, label: str, target: int, error_f) -> list[dict]:
    rows = []
    for candidate in candidates:
        if len(rows) >= target:
            break
        sample_id = f"sidset_{prefix}_{len(rows):06d}"
        out_path = out_dir / f"{sample_id}.jpg"
        try:
            image = to_pil_image(candidate["image"])
            image.save(out_path, format="JPEG", quality=95)
            ok, reason = verify_image(out_path)
            if not ok:
                out_path.unlink(missing_ok=True)
                raise ValueError(reason)
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                error_f,
                {
                    "source_index": candidate.get("source_index"),
                    "label": candidate.get("original_sidset_label"),
                    "type": candidate.get("original_sidset_type"),
                    "error": str(exc),
                },
            )
            continue
        rows.append(
            {
                "id": sample_id,
                "image_path": str(out_path.resolve()),
                "label": label,
                "label_id": 0 if label == "REAL" else 1,
                "source_dataset": "SIDSET",
                "original_sidset_label": candidate["original_sidset_label"],
                "original_sidset_type": candidate["original_sidset_type"],
            }
        )
    return rows


def write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Download and verify a balanced SIDSET 2K root.")
    parser.add_argument("--output_root", default="/workspace/SID_Set_2k_verified")
    parser.add_argument("--num_real", type=int, default=1000)
    parser.add_argument("--num_fake", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer_fake_type", default="full_synthetic")
    parser.add_argument("--hf_dataset", default="saberzl/SID_Set")
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    real_dir, fake_dir, logs_dir = clean_output_dirs(output_root)
    error_path = logs_dir / "download_errors.jsonl"

    real_candidates, fake_candidates, pool_report = select_candidates(args)
    with error_path.open("w", encoding="utf-8") as error_f:
        real_rows = save_class(real_candidates, real_dir, "real", "REAL", args.num_real, error_f)
        fake_rows = save_class(fake_candidates, fake_dir, "fake", "AI-GENERATED", args.num_fake, error_f)

    if len(real_rows) != args.num_real or len(fake_rows) != args.num_fake:
        raise SystemExit(
            f"Failed to produce requested verified SIDSET root. "
            f"real={len(real_rows)}/{args.num_real}, fake={len(fake_rows)}/{args.num_fake}. "
            f"See {error_path}."
        )

    manifest = real_rows + fake_rows
    random.Random(args.seed).shuffle(manifest)
    write_jsonl(output_root / "manifest.jsonl", manifest)
    report = {
        "hf_dataset": args.hf_dataset,
        "split": args.split,
        "seed": args.seed,
        "num_real": len(real_rows),
        "num_fake": len(fake_rows),
        "prefer_fake_type": args.prefer_fake_type,
        "label_mapping": {
            "0": "REAL / real",
            "1": "AI-GENERATED / full_synthetic",
            "2": "AI-GENERATED / tampered fallback",
        },
        **pool_report,
        "output_root": str(output_root),
    }
    write_json(output_root / "download_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
