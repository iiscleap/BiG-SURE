#!/usr/bin/env python3
"""Precompute DeBERTa entailment matrices for Text QA BiG-SURE runs."""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def resolve_run(run_dir, wandb_base_dir):
    path = Path(run_dir)
    if not path.is_absolute():
        path = Path(wandb_base_dir) / path
    if path.is_file():
        return path
    direct = path / "validation_generations.pkl"
    return direct if direct.exists() else path / "files" / "validation_generations.pkl"


def response_text(item):
    if isinstance(item, dict):
        return item.get("response")
    if isinstance(item, (list, tuple)) and item:
        return item[0]
    return item if isinstance(item, str) else None


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def validate_and_group(vanilla, rephrased, expected_low_t, expected_rephrasings,
                       expected_high_t_per_rephrasing, subsample_high_t):
    if not isinstance(vanilla, dict) or not vanilla:
        raise ValueError("Vanilla generations must be a non-empty dictionary.")
    if not isinstance(rephrased, dict) or not rephrased:
        raise ValueError("Rephrased generations must be a non-empty dictionary.")

    by_original = defaultdict(list)
    for rephrased_id, entry in rephrased.items():
        if not isinstance(entry, dict) or entry.get("original_id") is None:
            raise ValueError(f"Rephrased sample {rephrased_id!r} has no original_id.")
        responses = [response_text(item) for item in entry.get("responses", [])]
        if len(responses) != expected_high_t_per_rephrasing or any(not text for text in responses):
            raise ValueError(
                f"Rephrased sample {rephrased_id!r} must contain exactly "
                f"{expected_high_t_per_rephrasing} non-empty stochastic responses; "
                f"found {len(responses)}."
            )
        by_original[str(entry["original_id"])].append((str(rephrased_id), entry, responses))

    vanilla_ids = {str(question_id) for question_id in vanilla}
    rephrased_ids = set(by_original)
    if vanilla_ids != rephrased_ids:
        missing = sorted(vanilla_ids - rephrased_ids)[:5]
        extra = sorted(rephrased_ids - vanilla_ids)[:5]
        raise ValueError(
            "Vanilla/rephrased question IDs do not match. "
            f"Missing rephrased IDs: {missing}; unexpected IDs: {extra}."
        )

    for raw_id, entry in vanilla.items():
        question_id = str(raw_id)
        if not isinstance(entry, dict):
            raise ValueError(f"Vanilla sample {question_id!r} is not a dictionary.")
        low_texts = [response_text(item) for item in entry.get("low_temp_answers", [])]
        if len(low_texts) != expected_low_t or any(not text for text in low_texts):
            raise ValueError(
                f"Vanilla sample {question_id!r} must contain exactly {expected_low_t} "
                f"non-empty low-temperature responses; found {len(low_texts)}."
            )
        if len(by_original[question_id]) != expected_rephrasings:
            raise ValueError(
                f"Question {question_id!r} must have exactly {expected_rephrasings} "
                f"rephrasings; found {len(by_original[question_id])}."
            )
        available_high_t = expected_rephrasings * expected_high_t_per_rephrasing
        if subsample_high_t > available_high_t:
            raise ValueError(
                f"--subsample_high_t={subsample_high_t} exceeds the {available_high_t} "
                "validated rephrased responses available per question."
            )
    return by_original


def batched_probabilities(tokenizer, model, device, premises, hypotheses, batch_size):
    probabilities = []
    for start in range(0, len(premises), batch_size):
        inputs = tokenizer(
            premises[start:start + batch_size],
            hypotheses[start:start + batch_size],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities.extend(functional.softmax(logits, dim=1).cpu().numpy())
    return np.asarray(probabilities, dtype=np.float32)


def probability_matrices(tokenizer, model, device, low_texts, high_texts, batch_size):
    forward_pairs = [(low, high) for low in low_texts for high in high_texts]
    backward_pairs = [(high, low) for low in low_texts for high in high_texts]
    forward = batched_probabilities(
        tokenizer, model, device,
        [pair[0] for pair in forward_pairs], [pair[1] for pair in forward_pairs], batch_size,
    )
    backward = batched_probabilities(
        tokenizer, model, device,
        [pair[0] for pair in backward_pairs], [pair[1] for pair in backward_pairs], batch_size,
    )
    return (
        forward.reshape(len(low_texts), len(high_texts), 3),
        backward.reshape(len(low_texts), len(high_texts), 3),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla_run_dir", required=True)
    parser.add_argument("--rephrased_run_dir", required=True)
    parser.add_argument("--wandb_base_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--subsample_high_t", type=int, default=50)
    parser.add_argument("--expected_low_t", type=int, default=3)
    parser.add_argument("--expected_rephrasings", type=int, default=5)
    parser.add_argument("--expected_high_t_per_rephrasing", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nli_model", default="microsoft/deberta-v2-xlarge-mnli")
    args = parser.parse_args()

    vanilla_path = resolve_run(args.vanilla_run_dir, args.wandb_base_dir)
    rephrased_path = resolve_run(args.rephrased_run_dir, args.wandb_base_dir)
    if not vanilla_path.exists() or not rephrased_path.exists():
        raise FileNotFoundError(f"Missing generation pickle: {vanilla_path} or {rephrased_path}")

    vanilla = load_pickle(vanilla_path)
    rephrased = load_pickle(rephrased_path)
    by_original = validate_and_group(
        vanilla,
        rephrased,
        args.expected_low_t,
        args.expected_rephrasings,
        args.expected_high_t_per_rephrasing,
        args.subsample_high_t,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device).eval()
    archive = {}
    question_ids = []

    for raw_id, vanilla_entry in vanilla.items():
        question_id = str(raw_id)
        low_texts = [response_text(item) for item in vanilla_entry["low_temp_answers"]]
        high_texts = []
        paraphrase_indices = []
        for paraphrase_index, (_, _, responses) in enumerate(by_original[question_id]):
            high_texts.extend(responses)
            paraphrase_indices.extend([paraphrase_index] * len(responses))
        high_texts = high_texts[:args.subsample_high_t]
        paraphrase_indices = paraphrase_indices[:args.subsample_high_t]

        probs_fwd, probs_bwd = probability_matrices(
            tokenizer, model, device, low_texts, high_texts, args.batch_size
        )
        prefix = f"q_{question_id}_"
        archive[f"{prefix}probs_fwd"] = probs_fwd
        archive[f"{prefix}probs_bwd"] = probs_bwd
        archive[f"{prefix}low_texts"] = np.asarray(low_texts, dtype=object)
        archive[f"{prefix}high_texts"] = np.asarray(high_texts, dtype=object)
        archive[f"{prefix}paraphrase_indices"] = np.asarray(paraphrase_indices, dtype=np.int16)
        archive[f"{prefix}m"] = np.asarray([len(low_texts)], dtype=np.int32)
        archive[f"{prefix}n"] = np.asarray([len(high_texts)], dtype=np.int32)
        accuracy = vanilla_entry.get("greedy_answer", vanilla_entry.get("most_likely_answer", {})).get("accuracy")
        if accuracy is not None:
            archive[f"{prefix}accuracy"] = np.asarray([float(accuracy)], dtype=np.float32)
        question_ids.append(question_id)

    if not question_ids:
        raise ValueError("No matched original/rephrased examples with usable responses.")
    archive["question_ids"] = np.asarray(question_ids, dtype=object)
    archive["schema_version"] = np.asarray([2], dtype=np.int16)
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **archive)
    print(f"Saved {len(question_ids)} entailment matrices to {output}")


if __name__ == "__main__":
    main()
