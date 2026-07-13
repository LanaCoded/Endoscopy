from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_IMAGE_ROOT = Path("Test/Images")
DEFAULT_ANNOT_ROOT = Path("Test/Annotations")
DEFAULT_OUTPUT_DIR = Path("dataset")


def parse_annotation(annotation_path: Path) -> Tuple[Optional[int], Optional[Tuple[int, int, int, int]]]:
    """Parse annotation text into a class label and optional bounding box."""
    try:
        lines = [line.strip() for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        return None, None

    if not lines:
        return None, None

    first_line = lines[0]
    if first_line == "0":
        return 0, None

    if first_line == "1" and len(lines) > 1:
        parts = lines[1].split()
        if len(parts) == 4:
            try:
                x1, y1, x2, y2 = map(int, parts)
                return 1, (x1, y1, x2, y2)
            except ValueError:
                pass

    return None, None


def collect_records(image_root: Path, annot_root: Path) -> List[dict]:
    """Create a list of dataset records by pairing images with annotations."""
    records: List[dict] = []

    for image_path in sorted(image_root.rglob("*.jpg")):
        rel_path = image_path.relative_to(image_root)
        annotation_path = annot_root / rel_path.with_suffix(".txt")
        label, bbox = parse_annotation(annotation_path)

        if label is None:
            continue

        records.append(
            {
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
                "label": label,
                "bbox": bbox,
                "source_folder": rel_path.parts[0],
            }
        )

    return records


def stratified_split(records: List[dict], train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, List[dict]]:
    """Split records into train/val/test with a class-balanced approach."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(42)
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    for label, label_records in grouped.items():
        rng.shuffle(label_records)
        count = len(label_records)
        train_count = int(count * train_ratio)
        val_count = int(count * val_ratio)
        test_count = count - train_count - val_count

        splits["train"].extend(label_records[:train_count])
        splits["val"].extend(label_records[train_count : train_count + val_count])
        splits["test"].extend(label_records[train_count + val_count :])

    for split_name in splits:
        rng.shuffle(splits[split_name])

    return splits


def write_csv(output_path: Path, rows: List[dict]) -> None:
    fieldnames = ["image_path", "annotation_path", "label", "x1", "y1", "x2", "y2", "source_folder"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            bbox = row["bbox"] or ("", "", "", "")
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "annotation_path": row["annotation_path"],
                    "label": row["label"],
                    "x1": bbox[0] if bbox else "",
                    "y1": bbox[1] if bbox else "",
                    "x2": bbox[2] if bbox else "",
                    "y2": bbox[3] if bbox else "",
                    "source_folder": row["source_folder"],
                }
            )


def write_summary(output_dir: Path, records: List[dict], splits: Dict[str, List[dict]]) -> None:
    summary = {
        "total_records": len(records),
        "label_counts": dict(Counter(record["label"] for record in records)),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train/val/test dataset manifest from the Endoscopy ML image folders.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="Root directory containing JPG images.")
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOT_ROOT, help="Root directory containing TXT annotations.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to save the dataset CSV files.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(args.image_root, args.annotation_root)
    splits = stratified_split(records)

    write_csv(output_dir / "train.csv", splits["train"])
    write_csv(output_dir / "val.csv", splits["val"])
    write_csv(output_dir / "test.csv", splits["test"])
    write_summary(output_dir, records, splits)

    print(f"Built dataset files in {output_dir}")
    print(f"Total records: {len(records)}")
    print(f"Label counts: {dict(Counter(record['label'] for record in records))}")
    print("Split counts:")
    for name, rows in splits.items():
        print(f"  {name}: {len(rows)}")


if __name__ == "__main__":
    main()
