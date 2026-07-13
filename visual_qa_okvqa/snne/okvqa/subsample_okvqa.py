#!/usr/bin/env python3
"""Subsample OK-VQA dataset, save images and metadata CSV.

Downloads from HuggingFace lmms-lab/OK-VQA, balances across question_type,
then saves images + metadata.csv matching the VQAv2 CSV format.

Usage:
  python snne/okvqa/subsample_okvqa.py --n 300 --out_dir snne/okvqa
"""

import argparse
import csv
import io
import os
import random
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm
from PIL import Image as PILImage


def to_pil(img_field) -> PILImage.Image:
    """Convert HF Image feature to PIL.Image."""
    if isinstance(img_field, PILImage.Image):
        return img_field
    if isinstance(img_field, dict):
        b = img_field.get("bytes")
        p = img_field.get("path")
        if b is not None:
            return PILImage.open(io.BytesIO(b))
        if p and os.path.exists(p):
            return PILImage.open(p)
    if isinstance(img_field, str) and os.path.exists(img_field):
        return PILImage.open(img_field)
    raise ValueError("Unsupported image payload; cannot decode to PIL.Image")


def main():
    ap = argparse.ArgumentParser(description="Subsample OK-VQA dataset")
    ap.add_argument("--split", default="val2014")
    ap.add_argument("--out_dir", default="snne/okvqa")
    ap.add_argument("--n", type=int, default=300, help="Total examples to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_scan", type=int, default=50000,
                    help="Upper bound on items to scan")
    args = ap.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_csv = out_dir / "metadata.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading lmms-lab/OK-VQA ({args.split}) with streaming...")
    ds = load_dataset("lmms-lab/OK-VQA", split=args.split, streaming=True)

    # Collect all examples (OK-VQA has ~5046, small enough)
    all_examples = []
    for ex in tqdm(ds, desc=f"Scanning {args.split}"):
        all_examples.append(ex)
        if len(all_examples) >= args.max_scan:
            break

    print(f"Scanned {len(all_examples)} total examples.")

    # Group by question_type for balanced sampling
    groups = defaultdict(list)
    for ex in all_examples:
        qt = ex.get("question_type", "unknown")
        groups[qt].append(ex)

    print(f"Found {len(groups)} question types:")
    for k, v in sorted(groups.items()):
        print(f"  {k}: {len(v)} examples")

    # Balanced sampling: distribute n evenly across groups
    n = min(args.n, len(all_examples))
    per_group = n // len(groups)
    remainder = n % len(groups)

    selected = []
    group_keys = sorted(groups.keys())
    for i, key in enumerate(group_keys):
        group_n = per_group + (1 if i < remainder else 0)
        group_examples = groups[key]
        random.shuffle(group_examples)
        selected.extend(group_examples[:group_n])

    random.shuffle(selected)
    print(f"Selected {len(selected)} examples after balancing.")

    # Save images and build CSV rows
    rows = []
    for ex in tqdm(selected, desc="Saving images"):
        question_id = int(ex.get("question_id", 0))
        image_id = int(ex.get("image_id", 0))
        question = ex.get("question", "")
        question_type = ex.get("question_type", "")

        # OK-VQA has 'answers' field with list of answer dicts
        answers_raw = ex.get("answers", [])
        if isinstance(answers_raw, list):
            answer_texts = []
            for a in answers_raw:
                if isinstance(a, dict):
                    answer_texts.append(a.get("answer", str(a)))
                else:
                    answer_texts.append(str(a))
            answers_joined = "; ".join(answer_texts)
            # Pick the most common answer as multiple_choice_answer
            if answer_texts:
                from collections import Counter
                mc_answer = Counter(answer_texts).most_common(1)[0][0]
            else:
                mc_answer = ""
        else:
            answers_joined = str(answers_raw)
            mc_answer = str(answers_raw)

        # Save image
        try:
            img = to_pil(ex["image"]).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not decode image for qid={question_id}: {e}")
            continue

        img_fn = f"OKVQA_{args.split}_{image_id:012d}_{question_id}.jpg"
        img_path = img_dir / img_fn
        try:
            img.save(img_path, format="JPEG", quality=90)
        except Exception as e:
            print(f"Warning: Could not save image for qid={question_id}: {e}")
            continue

        rows.append({
            "split": args.split,
            "answer_type": "other",  # OK-VQA doesn't distinguish yes/no/number
            "question_id": question_id,
            "image_id": image_id,
            "question_type": question_type,
            "question": question,
            "multiple_choice_answer": mc_answer,
            "answers_joined": answers_joined,
            "image_path": str(img_path),
        })

    # Write CSV (same format as VQAv2)
    fieldnames = [
        "split", "answer_type", "question_id", "image_id",
        "question_type", "question", "multiple_choice_answer",
        "answers_joined", "image_path",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Saved {len(rows)} rows to {out_csv}")
    print(f"Images dir: {img_dir.resolve()}")

    # Print distribution
    from collections import Counter
    dist = Counter(r["question_type"] for r in rows)
    print("Distribution by question_type:")
    for k, v in sorted(dist.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
