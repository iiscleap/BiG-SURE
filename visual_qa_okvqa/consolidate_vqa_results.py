#!/usr/bin/env python3
"""Consolidate the maintained OKVQA baseline and BiG-SURE outputs."""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "consolidated" / "consolidated_vqa_pivot.csv"
MODELS = {
    "llava-v1.6-mistral-7b-hf": ("llava-v1.6-mistral-7b", "llava7b"),
    "Pixtral-12B-2409": ("pixtral-12b", "pixtral12b"),
    "Qwen3-VL-8B-Instruct": ("qwen3-vl-8b", "qwen8b"),
}
RESULT_DIRS = {
    "snne": ROOT / "snne_results",
    "kle": ROOT / "kle_results",
    "graph": ROOT / "graph_baseline_results",
    "blackbox": ROOT / "blackbox_se_results",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def metric_files(result_dir, model):
    if not result_dir.is_dir():
        return []
    files = result_dir.glob(f"okvqa_{model}_10generations*.csv")
    candidates = sorted(
        path for path in files
        if "_per_example" not in path.name and "_per_sample" not in path.name
    )
    by_seed = {}
    for path in candidates:
        match = re.search(r"_seed(\d+)\.csv$", path.name)
        key = match.group(1) if match else path.name
        previous = by_seed.get(key)
        if previous is None or len(path.name) < len(previous.name):
            by_seed[key] = path
    return sorted(by_seed.values())


def read_frames(paths):
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
            frame["source_file"] = path.name
            frames.append(frame)
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mean_column(frame, mask=None, column="auroc"):
    if frame.empty or column not in frame:
        return np.nan
    values = frame.loc[mask, column] if mask is not None else frame[column]
    values = pd.to_numeric(values, errors="coerce").dropna()
    values = values[values >= 0]
    return float(values.mean()) if not values.empty else np.nan


def load_accuracy(model):
    values = []
    paths = sorted((ROOT / "outputs" / "responses" / "vanilla").glob(
        f"{model}_seed*/vqa_accuracy.json"
    ))
    for path in paths:
        try:
            values.append(float(json.loads(path.read_text())["accuracy"]))
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
    return (float(np.mean(values)), len(values)) if values else (np.nan, 0)


def load_spectral(short_model):
    paths = sorted((ROOT / "outputs" / "spectral_energy").glob(
        f"okvqa_{short_model}_seed*/summary_*.csv"
    ))
    frame = read_frames(paths)
    if frame.empty:
        return {}, 0

    output = {}
    for keys, group in frame.groupby(
        ["similarity", "score_mode", "combine", "weighting_scheme"], dropna=False
    ):
        similarity, score_mode, combine, weighting = keys
        name = f"bigsure_{similarity}_{score_mode}_{combine}_{weighting}"
        output[name] = mean_column(group)
    seeds = {path.parent.name.split("_seed", 1)[1].split("_", 1)[0] for path in paths}
    return output, len(seeds)


def consolidate_model(model, display_name, short_model):
    accuracy, accuracy_seeds = load_accuracy(model)
    row = {
        "dataset": "okvqa",
        "model": display_name,
        "accuracy_metric": "official_vqa_accuracy",
        "accuracy": accuracy,
        "accuracy_seeds": accuracy_seeds,
    }

    snne_files = metric_files(RESULT_DIRS["snne"], model)
    snne = read_frames(snne_files)
    if not snne.empty:
        base = (snne["method"] == "snne") & (snne["temperature"] == 1.0)
        row["snne_entailment"] = mean_column(
            snne, base & (snne["similarity"] == "entailment_sim")
        )
        row["snne_lexical"] = mean_column(
            snne, base & (snne["similarity"] == "lexical_sim")
        )
    row["snne_seeds"] = len(snne_files)

    graph_files = metric_files(RESULT_DIRS["graph"], model)
    graph = read_frames(graph_files)
    if not graph.empty:
        for method, group in graph.groupby("method"):
            normalized = "eccentricity" if str(method).startswith("eccentricity") else str(method)
            row[f"graph_{normalized}"] = mean_column(group)
    row["graph_seeds"] = len(graph_files)

    kle_files = metric_files(RESULT_DIRS["kle"], model)
    kle = read_frames(kle_files)
    if not kle.empty:
        heat_mask = kle["method"].astype(str).str.contains("heat.*kernel", regex=True)
        row["kle_heat_avg"] = mean_column(kle, heat_mask)
    row["kle_seeds"] = len(kle_files)

    blackbox_files = metric_files(RESULT_DIRS["blackbox"], model)
    blackbox = read_frames(blackbox_files)
    if not blackbox.empty:
        row["blackbox_semantic_entropy"] = mean_column(
            blackbox, blackbox["method"] == "blackbox_se"
        )
    row["blackbox_seeds"] = len(blackbox_files)

    spectral, spectral_seeds = load_spectral(short_model)
    row.update(spectral)
    row["bigsure_seeds"] = spectral_seeds
    return row


def main():
    rows = [
        consolidate_model(model, display, short)
        for model, (display, short) in MODELS.items()
    ]
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number]).columns
    average = {column: np.nan for column in frame.columns}
    average.update({"dataset": "okvqa", "model": "AVG", "accuracy_metric": "official_vqa_accuracy"})
    for column in numeric:
        average[column] = frame[column].mean()
    frame = pd.concat([frame, pd.DataFrame([average])], ignore_index=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(frame.to_string(index=False))
    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
