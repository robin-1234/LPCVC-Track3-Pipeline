#!/usr/bin/env python3
import argparse
import random
import re
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from common import IMAGE_EXTS, is_image, normalize_dir_name, read_jsonl, utc_now, write_json, write_jsonl


NOT_FOUND_MESSAGE = (
    "SIDSET real/fake folders not found. Please provide a valid SIDSET root containing real and fake images."
)

OUTPUT_FILES = [
    "manifest.jsonl",
    "stage1_raw.jsonl",
    "stage1_valid.jsonl",
    "stage1_failed.jsonl",
    "stage2_raw.jsonl",
    "stage2_valid.jsonl",
    "stage2_invalid.jsonl",
    "train_sft.jsonl",
]

LOG_FILES = [
    "stage1_retry.jsonl",
    "stage1_failed.jsonl",
    "stage2_retry.jsonl",
    "api_errors.jsonl",
]


def scan_sidset(root: Path) -> dict:
    all_dirs = []
    direct_counts: dict[Path, int] = {}
    recursive_counts: dict[Path, int] = {}
    image_files = []

    for directory in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
        all_dirs.append(directory)
        direct_counts[directory] = 0
        recursive_counts[directory] = 0

    for path in sorted(root.rglob("*")):
        if not is_image(path):
            continue
        resolved = path.resolve()
        image_files.append(resolved)
        parent = path.parent
        direct_counts[parent] = direct_counts.get(parent, 0) + 1
        current = parent
        while True:
            recursive_counts[current] = recursive_counts.get(current, 0) + 1
            if current == root:
                break
            current = current.parent

    directory_rows = []
    for directory in all_dirs:
        try:
            relative_path = "." if directory == root else str(directory.relative_to(root))
        except ValueError:
            relative_path = str(directory)
        directory_rows.append(
            {
                "path": str(directory.resolve()),
                "relative_path": relative_path,
                "dir_name": directory.name,
                "normalized_name": normalize_dir_name(directory.name),
                "direct_image_count": direct_counts.get(directory, 0),
                "recursive_image_count": recursive_counts.get(directory, 0),
            }
        )

    return {
        "sidset_root": str(root.resolve()),
        "supported_extensions": sorted(IMAGE_EXTS),
        "directory_count": len(directory_rows),
        "image_count": len(image_files),
        "directories": directory_rows,
    }


def class_dirs(scan: dict, class_name: str) -> list[Path]:
    rows = []
    for row in scan["directories"]:
        if row["normalized_name"] == class_name and row["recursive_image_count"] > 0:
            rows.append(Path(row["path"]))
    return rows


def collect_images(directories: list[Path]) -> list[Path]:
    seen = set()
    images = []
    for directory in sorted(directories):
        for path in sorted(directory.rglob("*")):
            if not is_image(path):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            images.append(resolved)
    return images


def load_download_metadata(root: Path) -> dict:
    metadata_path = root / "metadata" / "sidset_2k_manifest.jsonl"
    rows = read_jsonl(metadata_path)
    return {str(Path(row["image_path"]).resolve()): row for row in rows if row.get("image_path")}


def sample_images(images: list[Path], target: int, seed: int) -> list[Path]:
    ordered = sorted(images, key=lambda path: path.name)
    if len(ordered) == target:
        return ordered
    sampled = random.Random(seed).sample(ordered, target)
    return sorted(sampled, key=lambda path: path.name)


def id_from_filename(image_path: Path, label: str, fallback_index: int) -> str:
    stem = image_path.stem
    if label == "REAL" and re.fullmatch(r"sidset_real_\d{6}", stem):
        return stem
    if label == "AI-GENERATED" and re.fullmatch(r"sidset_fake_\d{6}", stem):
        return stem
    prefix = "real" if label == "REAL" else "fake"
    return f"sidset_{prefix}_{fallback_index:06d}"


def manifest_row(sample_id: str, image_path: Path, label: str, metadata_by_path: dict | None = None) -> dict:
    resolved = str(image_path.resolve())
    source = metadata_by_path.get(resolved, {}) if metadata_by_path else {}
    row = {
        "id": sample_id,
        "image_path": resolved,
        "label": label,
        "label_id": 0 if label == "REAL" else 1,
        "source_dataset": source.get("source_dataset", "SID-Set"),
    }
    for key in ["source_hf_dataset", "source_original_index", "source_original_label"]:
        if key in source:
            row[key] = source[key]
    return row


def touch_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        (output_dir / name).touch(exist_ok=True)
    for name in LOG_FILES:
        (logs_dir / name).touch(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Scan SIDSET and prepare a balanced manifest.")
    parser.add_argument("--sidset_root", required=True)
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
    return parser.parse_args()


def main():
    args = parse_args()
    sidset_root = Path(args.sidset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    logs_dir = output_dir / "logs"
    manifest_path = output_dir / "manifest.jsonl"

    if not sidset_root.exists() or not sidset_root.is_dir():
        raise SystemExit(f"SIDSET root does not exist or is not a directory: {sidset_root}")

    touch_outputs(output_dir)
    scan = scan_sidset(sidset_root)
    real_dirs = class_dirs(scan, "real")
    fake_dirs = class_dirs(scan, "fake")
    real_images = collect_images(real_dirs)
    fake_images = collect_images(fake_dirs)
    overlap = sorted(set(real_images) & set(fake_images))

    scan_report = {
        **scan,
        "real_candidate_dirs": [str(path) for path in real_dirs],
        "fake_candidate_dirs": [str(path) for path in fake_dirs],
        "found_real_images": len(real_images),
        "found_fake_images": len(fake_images),
        "overlap_count": len(overlap),
        "created_at": utc_now(),
    }
    write_json(logs_dir / "dataset_scan.json", scan_report)

    print(f"[scan] SIDSET root: {sidset_root}")
    print(f"[scan] real dirs: {[str(path) for path in real_dirs]}")
    print(f"[scan] fake dirs: {[str(path) for path in fake_dirs]}")
    print(f"[scan] real images: {len(real_images)}")
    print(f"[scan] fake images: {len(fake_images)}")

    if not real_dirs or not fake_dirs:
        raise SystemExit(NOT_FOUND_MESSAGE)
    if overlap:
        raise SystemExit("SIDSET real/fake image overlap detected. Refusing to sample ambiguous labels.")
    if len(real_images) < args.num_real or len(fake_images) < args.num_fake:
        raise SystemExit(
            f"Not enough SIDSET images. Required real={args.num_real} fake={args.num_fake}, "
            f"found real={len(real_images)} fake={len(fake_images)}."
        )
    if args.resume and manifest_path.exists() and manifest_path.stat().st_size > 0:
        print(f"[resume] Existing manifest kept: {manifest_path}")
        return

    metadata_by_path = load_download_metadata(sidset_root)
    sampled_real = sample_images(real_images, args.num_real, args.seed)
    sampled_fake = sample_images(fake_images, args.num_fake, args.seed)

    rows = []
    for idx, image in enumerate(tqdm(sampled_real, desc="Sample REAL", unit="image"), start=1):
        sample_id = id_from_filename(image, "REAL", idx)
        rows.append(manifest_row(sample_id, image, "REAL", metadata_by_path))
    for idx, image in enumerate(tqdm(sampled_fake, desc="Sample FAKE", unit="image"), start=1):
        sample_id = id_from_filename(image, "AI-GENERATED", idx)
        rows.append(manifest_row(sample_id, image, "AI-GENERATED", metadata_by_path))

    write_jsonl(manifest_path, rows)
    write_json(
        logs_dir / "prepare_summary.json",
        {
            "seed": args.seed,
            "num_real": args.num_real,
            "num_fake": args.num_fake,
            "total": len(rows),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens_stage1": args.max_output_tokens_stage1,
            "max_output_tokens_stage2": args.max_output_tokens_stage2,
            "manifest_path": str(manifest_path),
            "created_at": utc_now(),
        },
    )
    print(f"[done] Wrote manifest: {manifest_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
