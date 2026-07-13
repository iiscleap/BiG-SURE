"""Maintain the Text QA generation manifest used by downstream stages."""

import csv
import os
import tempfile
from pathlib import Path

import fcntl


FIELDS = ["dataset", "model", "seed", "vanilla_run_dir", "rephrased_run_dir"]


def record_wandb_run(manifest_path, dataset, model, seed, stage, wandb_files_dir):
    """Insert or update one generation run in the shared TSV manifest."""
    if not manifest_path:
        return
    if stage not in {"vanilla", "rephrased"}:
        raise ValueError(f"Unsupported manifest stage: {stage}")

    manifest = Path(manifest_path).expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest.with_suffix(manifest.suffix + ".lock")
    key = (str(dataset), str(model), str(seed))
    run_dir = str(Path(wandb_files_dir).resolve().parent)

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows = []
        if manifest.exists():
            with manifest.open(newline="") as source:
                rows = list(csv.DictReader(source, delimiter="\t"))

        row = next(
            (item for item in rows if tuple(item.get(field, "") for field in FIELDS[:3]) == key),
            None,
        )
        if row is None:
            row = dict.fromkeys(FIELDS, "")
            row.update(zip(FIELDS[:3], key))
            rows.append(row)
        row[f"{stage}_run_dir"] = run_dir
        rows.sort(key=lambda item: (item["dataset"], item["model"], int(item["seed"])))

        fd, temporary_name = tempfile.mkstemp(dir=manifest.parent, prefix=".runs-", text=True)
        try:
            with os.fdopen(fd, "w", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=FIELDS, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary_name, manifest)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
