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
IMAGE_AUGMENTATIONS = ("contrast", "blur", "rotate", "shift", "noise", "masking", "bw")
MULTIMODAL_SAMPLES_PER_INPUT = 10


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected True or False")


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


def parse_image_augmentations(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(IMAGE_AUGMENTATIONS)
    augmentations = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(augmentations) - set(IMAGE_AUGMENTATIONS))
    if unknown:
        raise ValueError(
            f"Unknown image augmentations: {', '.join(unknown)}. "
            f"Choose from {', '.join(IMAGE_AUGMENTATIONS)}"
        )
    if not augmentations:
        raise ValueError("--image_augs must select at least one augmentation")
    return list(dict.fromkeys(augmentations))


def balanced_augmentation_counts(augmentations: list[str], total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(augmentations))
    return {
        augmentation: base + (index < remainder)
        for index, augmentation in enumerate(augmentations)
    }


def select_multimodal_responses(
    row: pd.Series,
    row_id: str,
    input_aug: bool,
    image_augmentations: list[str],
) -> tuple[list[str], list[object]]:
    response_column = "augmented_responses" if "augmented_responses" in row.index else "rephrased_responses"
    id_column = "image_augmentation_ids" if "image_augmentation_ids" in row.index else "augmentation_ids"
    responses = clean_texts(
        parse_json_list(row.get(response_column), response_column, row_id), response_column, row_id
    )
    augmentation_ids = [
        str(value).strip().lower()
        for value in parse_json_list(row.get(id_column), id_column, row_id)
    ]
    if len(responses) != len(augmentation_ids):
        raise ValueError(f"row {row_id}: {response_column} and {id_column} must align")

    if input_aug:
        rephrase_ids = parse_json_list(row.get("rephrase_ids"), "rephrase_ids", row_id)
        if len(rephrase_ids) != len(responses):
            raise ValueError(f"row {row_id}: rephrase_ids must align with multimodal responses")
    else:
        rephrase_ids = ["direct"] * len(responses)

    quotas = balanced_augmentation_counts(image_augmentations, MULTIMODAL_SAMPLES_PER_INPUT)
    selected_responses: list[str] = []
    selected_group_ids: list[object] = []
    for group_id in dict.fromkeys(rephrase_ids):
        for augmentation, required in quotas.items():
            candidates = [
                response
                for response, current_group, current_augmentation in zip(
                    responses, rephrase_ids, augmentation_ids
                )
                if current_group == group_id and current_augmentation == augmentation
            ]
            if len(candidates) < required:
                raise ValueError(
                    f"row {row_id}: group {group_id!r} needs {required} {augmentation} "
                    f"responses but only {len(candidates)} are available"
                )
            selected_responses.extend(candidates[:required])
            selected_group_ids.extend([group_id] * required)
    return selected_responses, selected_group_ids


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


def score_row(
    row: pd.Series,
    measures: list[str],
    nli: NLIModel | None,
    input_aug: bool,
    multimodal: bool,
    image_augmentations: list[str],
) -> dict[str, float]:
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
        if multimodal:
            rephrased, rephrase_ids = select_multimodal_responses(
                row, row_id, input_aug, image_augmentations
            )
        elif input_aug:
            rephrased = clean_texts(parse_json_list(row.get("rephrased_responses"), "rephrased_responses", row_id), "rephrased_responses", row_id)
            rephrase_ids = parse_json_list(row.get("rephrase_ids"), "rephrase_ids", row_id)
        else:
            rephrased = sampled
            rephrase_ids = ["direct"] * len(rephrased)
        if not low or not rephrased or len(rephrased) != len(rephrase_ids):
            raise ValueError(f"row {row_id}: BiG-SURE requires aligned low-temperature/augmented inputs")
        expected_groups = 5 if input_aug else 1
        expected_responses = expected_groups * MULTIMODAL_SAMPLES_PER_INPUT
        if len(low) != 3 or len(set(rephrase_ids)) != expected_groups or len(rephrased) != expected_responses:
            LOGGER.warning(
                "row %s differs from expected shape (3 low-temperature, %d x 10 augmented)",
                row_id,
                expected_groups,
            )
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
    parser.add_argument(
        "--input_aug", "--input-aug", type=parse_bool, default=True,
        help="Use paraphrased input generations for BiG-SURE (default: True)",
    )
    parser.add_argument(
        "--multimodal", type=parse_bool, default=False,
        help="Read image paths and image-augmentation generation metadata (default: False)",
    )
    parser.add_argument(
        "--image_augs", "--image-augs", "--input_augs", dest="image_augs", default="all",
        help="Comma-separated image augmentations, or 'all' (used with --multimodal True)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    measures = parse_measures(args.measures)
    image_augmentations = parse_image_augmentations(args.image_augs)
    frame = pd.read_csv(args.input)
    if "id" not in frame.columns:
        raise ValueError("Input CSV must contain an id column")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("Input CSV contains duplicate ids")
    if args.max_rows is not None:
        frame = frame.head(args.max_rows)
    if args.multimodal:
        if "image_path" not in frame.columns:
            raise ValueError("Multimodal CSV input must contain an image_path column")
        missing_images = []
        for value in frame["image_path"]:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = args.input.parent / path
            if not path.is_file():
                missing_images.append(str(value))
        if missing_images:
            preview = ", ".join(missing_images[:3])
            raise FileNotFoundError(f"Multimodal CSV references missing images: {preview}")
    nli = NLIModel(args.nli_model, args.device, args.batch_size) if set(measures) & NLI_MEASURES else None

    output_rows = []
    for index, row in frame.iterrows():
        LOGGER.info("Scoring row %d/%d (id=%s)", index + 1, len(frame), row["id"])
        output = {key: row[key] for key in ("id", "question", "image_path", "correct") if key in frame.columns}
        output.update(
            score_row(
                row, measures, nli, args.input_aug, args.multimodal, image_augmentations
            )
        )
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
