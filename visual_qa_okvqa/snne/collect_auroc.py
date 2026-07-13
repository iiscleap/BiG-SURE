#!/usr/bin/env python3
"""Collect AUROC values from .err log files.

Scans recursively from a root directory for files ending in `.err`, extracts
`Dataset: ...`, `Model: ...` and `AUROC: ...` occurrences and writes a CSV
summary to the `--out` path (default: `snne/auroc_summary.csv`).

Usage:
    python snne/collect_auroc.py --root . --out snne/auroc_summary.csv

"""
import argparse
import csv
import re
from pathlib import Path


DATASET_RE = re.compile(r"Dataset:\s*(.+)", re.IGNORECASE)
MODEL_RE = re.compile(r"Model:\s*(.+)", re.IGNORECASE)
AUROC_RE = re.compile(r"AUROC:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
VANILLA_RUN_RE = re.compile(r"Vanilla run:\s*(\S+)", re.IGNORECASE)


def parse_err_file(path: Path):
    """Parse a single .err file and yield (dataset, model, auroc, run_name).

    There can be multiple dataset/model/AUROC blocks in a single file; we yield
    a row for each AUROC found, pairing it with the most recent Dataset/Model
    seen above it in the file.
    """
    dataset = None
    model = None
    run_name = None

    rows = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                m = DATASET_RE.search(line)
                if m:
                    dataset = m.group(1).strip()
                    continue

                m = MODEL_RE.search(line)
                if m:
                    model = m.group(1).strip()
                    continue

                m = VANILLA_RUN_RE.search(line)
                if m:
                    run_name = m.group(1).strip()
                    continue

                m = AUROC_RE.search(line)
                if m:
                    auroc = float(m.group(1))
                    rows.append({
                        "dataset": dataset or "",
                        "model": model or "",
                        "auroc": auroc,
                        "run": run_name or "",
                        "source": str(path)
                    })
    except Exception:
        # ignore any parse/read errors for robustness
        return []

    return rows


def collect(root: Path):
    rows = []
    for p in sorted(root.rglob("*.err")):
        parsed = parse_err_file(p)
        if parsed:
            rows.extend(parsed)
    return rows


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "dataset", "auroc", "run", "source"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "model": r.get("model", ""),
                "dataset": r.get("dataset", ""),
                "auroc": r.get("auroc", ""),
                "run": r.get("run", ""),
                "source": r.get("source", ""),
            })


def main():
    parser = argparse.ArgumentParser(description="Collect AUROC values from .err logs")
    parser.add_argument("--root", type=Path, default=Path("."), help="Root directory to scan")
    parser.add_argument("--out", type=Path, default=Path("snne/auroc_summary.csv"), help="Output CSV path")
    args = parser.parse_args()

    rows = collect(args.root)
    if not rows:
        print("No AUROC entries found under", args.root)
        return 1

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
