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
    return path / "files" / "validation_generations.pkl"


def response_text(item):
    if isinstance(item, dict):
        return item.get("response")
    if isinstance(item, (list, tuple)) and item:
        return item[0]
    return item if isinstance(item, str) else None


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nli_model", default="microsoft/deberta-v2-xlarge-mnli")
    args = parser.parse_args()

    vanilla_path = resolve_run(args.vanilla_run_dir, args.wandb_base_dir)
    rephrased_path = resolve_run(args.rephrased_run_dir, args.wandb_base_dir)
    if not vanilla_path.exists() or not rephrased_path.exists():
        raise FileNotFoundError(f"Missing generation pickle: {vanilla_path} or {rephrased_path}")

    vanilla = load_pickle(vanilla_path)
    rephrased = load_pickle(rephrased_path)
    by_original = defaultdict(list)
    for entry in rephrased.values():
        original_id = entry.get("original_id")
        if original_id is None:
            continue
        by_original[str(original_id)].append(entry)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device).eval()
    archive = {}
    question_ids = []

    for raw_id, vanilla_entry in vanilla.items():
        question_id = str(raw_id)
        low_texts = [response_text(item) for item in vanilla_entry.get("low_temp_answers", [])]
        low_texts = [text for text in low_texts if text]
        high_texts = []
        for entry in by_original.get(question_id, []):
            high_texts.extend(response_text(item) for item in entry.get("responses", []))
        high_texts = [text for text in high_texts if text][:args.subsample_high_t]
        if not low_texts or not high_texts:
            continue

        probs_fwd, probs_bwd = probability_matrices(
            tokenizer, model, device, low_texts, high_texts, args.batch_size
        )
        prefix = f"q_{question_id}_"
        archive[f"{prefix}probs_fwd"] = probs_fwd
        archive[f"{prefix}probs_bwd"] = probs_bwd
        archive[f"{prefix}low_texts"] = np.asarray(low_texts, dtype=object)
        archive[f"{prefix}high_texts"] = np.asarray(high_texts, dtype=object)
        archive[f"{prefix}m"] = np.asarray([len(low_texts)], dtype=np.int32)
        archive[f"{prefix}n"] = np.asarray([len(high_texts)], dtype=np.int32)
        accuracy = vanilla_entry.get("greedy_answer", vanilla_entry.get("most_likely_answer", {})).get("accuracy")
        if accuracy is not None:
            archive[f"{prefix}accuracy"] = np.asarray([float(accuracy)], dtype=np.float32)
        question_ids.append(question_id)

    if not question_ids:
        raise ValueError("No matched original/rephrased examples with usable responses.")
    archive["question_ids"] = np.asarray(question_ids, dtype=object)
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **archive)
    print(f"Saved {len(question_ids)} entailment matrices to {output}")


if __name__ == "__main__":
    main()
