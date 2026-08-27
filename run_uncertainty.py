#!/usr/bin/env python3
"""Compute BiG-SURE and baseline uncertainty scores from a portable CSV."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "text_qa"))

from snne.uncertainty.utils.entropy_utils import (  # noqa: E402
    get_degreeuq,
    get_spectral_eigv,
    snne,
)


LOGGER = logging.getLogger("csv_uncertainty")
DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
ALL_MEASURES = (
    "bigsure",
    "semantic_entropy",
    "predictive_entropy",
    "num_semantic_sets",
    "lexical_similarity",
    "graph_degree",
    "graph_eigenvalue",
    "snne",
)
NLI_MEASURES = {
    "bigsure",
    "semantic_entropy",
    "num_semantic_sets",
    "graph_degree",
    "graph_eigenvalue",
    "snne",
}


def parse_json_list(value: object, column: str, row_id: str) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"row {row_id}: {column} is not a valid JSON array") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"row {row_id}: {column} must be a JSON array")
    return parsed


def clean_texts(values: Iterable[object], column: str, row_id: str) -> list[str]:
    texts = [str(value).strip() for value in values]
    if any(not text for text in texts):
        LOGGER.warning("row %s: replacing an empty %s item with an explicit placeholder", row_id, column)
        texts = [text or "<empty response>" for text in texts]
    return texts


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w ]+", "", text, flags=re.UNICODE)


def lexical_pair_similarity(left: str, right: str) -> float:
    left_norm, right_norm = normalize_text(left), normalize_text(right)
    if left_norm == right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    left_tokens, right_tokens = set(left_norm.split()), set(right_norm.split())
    token_union = left_tokens | right_tokens
    token_jaccard = len(left_tokens & right_tokens) / len(token_union) if token_union else 1.0
    n = 3
    left_grams = {left_norm[i:i+n] for i in range(max(1, len(left_norm) - n + 1))}
    right_grams = {right_norm[i:i+n] for i in range(max(1, len(right_norm) - n + 1))}
    gram_union = left_grams | right_grams
    gram_jaccard = len(left_grams & right_grams) / len(gram_union) if gram_union else 1.0
    score = 0.5 * SequenceMatcher(None, left_norm, right_norm).ratio()
    score += 0.25 * token_jaccard + 0.25 * gram_jaccard
    return float(np.clip(score, 0.0, 1.0))


def lexical_matrix(texts: list[str]) -> np.ndarray:
    matrix = np.eye(len(texts), dtype=np.float32)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            matrix[i, j] = matrix[j, i] = lexical_pair_similarity(texts[i], texts[j])
    return matrix


class NLIModel:
    def __init__(self, model_name: str, device: str, batch_size: int):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        LOGGER.info("Loading NLI model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.entailment_index = self._label_index("entail")
        self.neutral_index = self._label_index("neutral")
        self.contradiction_index = self._label_index("contrad")

    def _label_index(self, fragment: str) -> int:
        labels = {int(i): str(label).lower() for i, label in self.model.config.id2label.items()}
        matches = [i for i, label in labels.items() if fragment in label]
        if len(matches) != 1:
            raise ValueError(f"Cannot identify {fragment!r} label from {labels}")
        return matches[0]

    def probabilities(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        if len(premises) != len(hypotheses):
            raise ValueError("NLI premise and hypothesis batches have different lengths")
        chunks = []
        for start in range(0, len(premises), self.batch_size):
            encoded = self.tokenizer(
                premises[start:start + self.batch_size],
                hypotheses[start:start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits
            raw = torch.softmax(logits, dim=-1).cpu().numpy()
            chunks.append(raw[:, [self.contradiction_index, self.neutral_index, self.entailment_index]])
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3))

    def bidirectional_matrix(self, rows: list[str], columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
        premises = [row for row in rows for _ in columns]
        hypotheses = [column for _ in rows for column in columns]
        shape = (len(rows), len(columns), 3)
        forward = self.probabilities(premises, hypotheses).reshape(shape)
        backward = self.probabilities(hypotheses, premises).reshape(shape)
        return forward, backward


def semantic_ids(forward: np.ndarray, backward: np.ndarray) -> list[int]:
    count = forward.shape[0]
    labels_forward = np.argmax(forward, axis=2)
    labels_backward = np.argmax(backward, axis=2)
    ids = [-1] * count
    next_id = 0
    for i in range(count):
        if ids[i] >= 0:
            continue
        ids[i] = next_id
        for j in range(i + 1, count):
            if ids[j] >= 0:
                continue
            pair = (labels_forward[i, j], labels_backward[i, j])
            if 0 not in pair and pair != (1, 1):
                ids[j] = next_id
        next_id += 1
    return ids


def cluster_entropy(ids: list[int]) -> float:
    counts = np.asarray(list(Counter(ids).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def semantic_probability_entropy(ids: list[int], probabilities: list[float]) -> float:
    probabilities_array = np.asarray(probabilities, dtype=float)
    probabilities_array /= probabilities_array.sum()
    cluster_probabilities = np.asarray([
        probabilities_array[np.asarray(ids) == cluster].sum() for cluster in sorted(set(ids))
    ])
    return float(max(0.0, -np.sum(cluster_probabilities * np.log(cluster_probabilities + 1e-12))))


def cluster_distribution(texts: list[str], threshold: float = 0.7) -> np.ndarray:
    parent = list(range(len(texts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if lexical_pair_similarity(texts[i], texts[j]) >= threshold:
                union(i, j)
    counts = Counter(find(i) for i in range(len(texts)))
    values = np.asarray(list(counts.values()), dtype=float)
    return values / values.sum()


def entropy_confidence_weights(texts: list[str], group_ids: list[object]) -> np.ndarray:
    weights_by_group = {}
    for group_id in dict.fromkeys(group_ids):
        group_texts = [text for text, current_id in zip(texts, group_ids) if current_id == group_id]
        distribution = cluster_distribution(group_texts)
        if len(distribution) <= 1:
            weights_by_group[group_id] = 1.0
        else:
            entropy = -np.sum(distribution * np.log(distribution + 1e-12))
            weights_by_group[group_id] = 1.0 - float(entropy / np.log(len(distribution)))
    return np.asarray([weights_by_group[group_id] for group_id in group_ids])


def bigsure_uncertainty(
    low_temperature: list[str], rephrased: list[str], rephrase_ids: list[object], nli: NLIModel
) -> float:
    forward, backward = nli.bidirectional_matrix(low_temperature, rephrased)
    weights_matrix = np.minimum(forward[:, :, 2], backward[:, :, 2])
    column_weights = entropy_confidence_weights(rephrased, rephrase_ids)
    weighted = weights_matrix @ np.diag(np.sqrt(column_weights))
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    confidence = np.linalg.norm(singular_values) / math.sqrt(weighted.shape[0] * weighted.shape[1])
    return float(1.0 - confidence)


def parse_measures(raw: str) -> list[str]:
    measures = list(ALL_MEASURES) if raw.strip().lower() == "all" else [item.strip() for item in raw.split(",")]
    unknown = sorted(set(measures) - set(ALL_MEASURES))
    if unknown:
        raise ValueError(f"Unknown measures: {', '.join(unknown)}")
    return list(dict.fromkeys(measures))


def score_row(row: pd.Series, measures: list[str], nli: NLIModel | None) -> dict[str, float]:
    row_id = str(row["id"])
    sampled = clean_texts(parse_json_list(row.get("sampled_responses"), "sampled_responses", row_id), "sampled_responses", row_id)
    probabilities = [float(value) for value in parse_json_list(row.get("sampled_probabilities"), "sampled_probabilities", row_id)]
    scores: dict[str, float] = {}

    needs_sampled = set(measures) - {"bigsure"}
    if needs_sampled and not sampled:
        raise ValueError(f"row {row_id}: sampled_responses is required")
    if {"predictive_entropy", "semantic_entropy"} & set(measures):
        if len(probabilities) != len(sampled) or any(value <= 0 or value > 1 for value in probabilities):
            raise ValueError(f"row {row_id}: sampled_probabilities must align with responses and lie in (0, 1]")

    nli_forward = nli_backward = None
    ids = None
    if set(measures) & (NLI_MEASURES - {"bigsure"}):
        assert nli is not None
        nli_forward, nli_backward = nli.bidirectional_matrix(sampled, sampled)
        ids = semantic_ids(nli_forward, nli_backward)

    if "bigsure" in measures:
        assert nli is not None
        low = clean_texts(parse_json_list(row.get("low_temperature_responses"), "low_temperature_responses", row_id), "low_temperature_responses", row_id)
        rephrased = clean_texts(parse_json_list(row.get("rephrased_responses"), "rephrased_responses", row_id), "rephrased_responses", row_id)
        rephrase_ids = parse_json_list(row.get("rephrase_ids"), "rephrase_ids", row_id)
        if not low or not rephrased or len(rephrased) != len(rephrase_ids):
            raise ValueError(f"row {row_id}: BiG-SURE requires aligned low-temperature/rephrased inputs")
        if len(low) != 3 or len(set(rephrase_ids)) != 5 or len(rephrased) != 50:
            LOGGER.warning("row %s differs from paper shape (3 low-temperature, 5 x 10 rephrased)", row_id)
        scores["bigsure"] = bigsure_uncertainty(low, rephrased, rephrase_ids, nli)

    if "predictive_entropy" in measures:
        scores["predictive_entropy"] = float(-np.mean(np.log(np.asarray(probabilities))))
    if "semantic_entropy" in measures:
        assert ids is not None
        scores["semantic_entropy"] = semantic_probability_entropy(ids, probabilities)
    if "num_semantic_sets" in measures:
        assert ids is not None
        scores["num_semantic_sets"] = float(len(set(ids)))
    if "lexical_similarity" in measures:
        matrix = lexical_matrix(sampled)
        scores["lexical_similarity"] = float(1.0 - (matrix.sum() - len(sampled)) / (len(sampled) * max(1, len(sampled) - 1)))
    if {"graph_degree", "graph_eigenvalue", "snne"} & set(measures):
        assert nli_forward is not None and nli_backward is not None and ids is not None
        forward_labels = np.argmax(nli_forward, axis=2)
        backward_labels = np.argmax(nli_backward, axis=2)
        entailment_matrix = (
            (forward_labels == 2).astype(float)
            + (backward_labels == 2).astype(float)
            + 0.5 * (forward_labels == 1).astype(float)
            + 0.5 * (backward_labels == 1).astype(float)
        ) / 2.0
        np.fill_diagonal(entailment_matrix, 1.0)
        if "graph_degree" in measures:
            scores["graph_degree"] = float(get_degreeuq(entailment_matrix)[0])
        if "graph_eigenvalue" in measures:
            scores["graph_eigenvalue"] = float(get_spectral_eigv(entailment_matrix))
        if "snne" in measures:
            scores["snne"] = float(snne(entailment_matrix, ids, exclude_diagonal=False).item())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input CSV")
    parser.add_argument("--output", required=True, type=Path, help="Per-example output CSV")
    parser.add_argument("--measures", default="all", help="Comma-separated measures or 'all'")
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test limit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    measures = parse_measures(args.measures)
    frame = pd.read_csv(args.input)
    if "id" not in frame.columns:
        raise ValueError("Input CSV must contain an id column")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("Input CSV contains duplicate ids")
    if args.max_rows is not None:
        frame = frame.head(args.max_rows)
    nli = NLIModel(args.nli_model, args.device, args.batch_size) if set(measures) & NLI_MEASURES else None

    output_rows = []
    for index, row in frame.iterrows():
        LOGGER.info("Scoring row %d/%d (id=%s)", index + 1, len(frame), row["id"])
        output = {key: row[key] for key in ("id", "question", "correct") if key in frame.columns}
        output.update(score_row(row, measures, nli))
        output_rows.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_frame = pd.DataFrame(output_rows)
    output_frame.to_csv(args.output, index=False)
    LOGGER.info("Wrote %s", args.output)

    if "correct" in output_frame and output_frame["correct"].notna().all():
        correct = output_frame["correct"].astype(float)
        if set(correct.unique()) <= {0.0, 1.0} and correct.nunique() == 2:
            summary = [{"measure": measure, "error_auroc": roc_auc_score(1.0 - correct, output_frame[measure])} for measure in measures]
            summary_path = args.output.with_name(f"{args.output.stem}_summary.csv")
            pd.DataFrame(summary).to_csv(summary_path, index=False)
            LOGGER.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
