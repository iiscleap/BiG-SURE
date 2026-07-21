#!/usr/bin/env python3
"""
Compute multilingual graph baseline uncertainty measures.

Data extraction uses multilingual JSON format via multilingual_utils.
Uncertainty computation (num_set, lexical_sim, sum_eigv, degree_mat,
eccentricity) is IDENTICAL to compute_graph_baselines.py — only the
data-level extraction changes.

Usage:
    python compute_multilingual_graph_baselines.py \
        --vanilla_json /path/to/vanilla/generate.json \
        --sampling_json /path/to/sampling/generate.json \
        --output_dir ./graph_baseline_results \
        --dataset sciq --model_name llama3 \
        --languages en hi fr
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import evaluate
from rouge_score import tokenizers

# SNNE repo uncertainty utilities reused from the consolidated text QA package.
BIGSURE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BIGSURE_ROOT / "text_qa"))
from snne.uncertainty.utils.eval_utils import auroc, auarc, aucpr, is_binary_list
from snne.uncertainty.utils.entropy_utils import (
    compute_lexical_similarity,
    get_spectral_eigv,
    get_degreeuq,
    get_eccentricity,
)

# Multilingual data utilities
from multilingual_utils import (
    load_json_data,
    get_languages,
    extract_vanilla_data,
    extract_sampling_data,
    calculate_prem_accuracy,
    MultilingualEntailmentDeberta,
    get_semantic_ids_using_entailment,
    validate_generation_pair,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Similarity matrix builders (data-extraction layer only)
# ---------------------------------------------------------------------------

def build_lexical_similarity_matrix(responses: List[str]) -> np.ndarray:
    """Build NxN RougeL similarity matrix from a list of sampled responses."""
    n = len(responses)
    rouge = evaluate.load('rouge', keep_in_memory=True)
    tokenizer_fn = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize
    sim_mat = np.zeros((n, n))
    for i in range(n):
        sim_mat[i, i] = 1.0
        for j in range(i + 1, n):
            try:
                score = rouge.compute(
                    predictions=[responses[i]],
                    references=[responses[j]],
                    rouge_types=['rougeL'],
                    tokenizer=tokenizer_fn,
                )['rougeL']
            except Exception:
                score = 0.0
            sim_mat[i, j] = score
            sim_mat[j, i] = score
    return sim_mat


def build_entailment_similarity_matrix(
    responses: List[str],
    model: MultilingualEntailmentDeberta,
) -> np.ndarray:
    """Build NxN entailment similarity matrix in [0, 1].

    Weight formula mirrors the original SNNE pipeline:
        weight(i,j) = I(i->j == entail) + I(j->i == entail)
                    + 0.5*I(i->j == neutral) + 0.5*I(j->i == neutral)
    Normalised to [0, 1] by dividing by 2.
    """
    n = len(responses)
    sim_mat = np.zeros((n, n))
    for i in range(n):
        sim_mat[i, i] = 1.0
        for j in range(i + 1, n):
            impl_ij, _ = model.check_implication(responses[i], responses[j])
            impl_ji, _ = model.check_implication(responses[j], responses[i])
            weight = (
                int(impl_ij == 2) + int(impl_ji == 2)
                + 0.5 * int(impl_ij == 1)
                + 0.5 * int(impl_ji == 1)
            )
            sim = weight / 2.0  # normalise to [0, 1]
            sim_mat[i, j] = sim
            sim_mat[j, i] = sim
    return sim_mat


# ---------------------------------------------------------------------------
# Per-language metric computation (uncertainty logic identical to reference)
# ---------------------------------------------------------------------------

def compute_metrics_for_language(
    vanilla_data: List[Dict],
    sampling_data: List[Dict],
    language: str,
    nli_model: MultilingualEntailmentDeberta,
    num_generations: int,
    metric: str = 'prem',
) -> pd.DataFrame:
    """Run all graph baseline methods for one language.

    The computation block is an exact port of compute_graph_baselines.py;
    only data extraction at the top differs.
    """
    v_questions, v_answers, v_outputs, v_probs = extract_vanilla_data(vanilla_data, language)
    s_questions, s_answers, s_outputs, s_probs = extract_sampling_data(sampling_data, language)

    # ── Data extraction (multilingual-specific) ──────────────────────────────
    correctness: List[float] = []
    list_num_sets: List[int] = []
    list_lex_sim_mat: List[np.ndarray] = []
    list_entail_sim_mat: List[np.ndarray] = []

    for idx in tqdm(range(len(v_questions)), desc=f"Building matrices [{language}]"):
        responses = s_outputs[idx][:num_generations]

        # Greedy (vanilla) output
        vanilla_out_raw = v_outputs[idx]
        if isinstance(vanilla_out_raw, list) and vanilla_out_raw:
            vanilla_output = vanilla_out_raw[0]
        else:
            vanilla_output = str(vanilla_out_raw) if vanilla_out_raw else ""

        ground_truth = v_answers[idx]

        # Correctness label
        if metric in ('claude', 'gemini'):
            is_correct = float(vanilla_data[idx].get('accuracy', {}).get(language, 0.0))
        else:
            is_correct = float(calculate_prem_accuracy(vanilla_output, ground_truth))
        correctness.append(is_correct)

        if not responses or len(responses) < 2:
            list_num_sets.append(1)
            list_lex_sim_mat.append(np.ones((1, 1)))
            list_entail_sim_mat.append(np.ones((1, 1)))
            continue

        # Semantic IDs for num_set (same clustering as reference)
        sem_ids = get_semantic_ids_using_entailment(
            responses, nli_model, strict_entailment=False
        )
        list_num_sets.append(len(set(sem_ids)))

        # Similarity matrices (data-extraction layer)
        list_lex_sim_mat.append(build_lexical_similarity_matrix(responses))
        list_entail_sim_mat.append(build_entailment_similarity_matrix(responses, nli_model))

    # ── Uncertainty computation — IDENTICAL to compute_graph_baselines.py ────
    validation_is_false = [1.0 - c for c in correctness]
    is_binary = is_binary_list(validation_is_false)

    list_method_name: List[str] = []
    list_auroc_scores: List[float] = []
    list_auarc_scores: List[float] = []
    list_aucpr_scores: List[float] = []

    ## Num sets
    list_method_name.append('num_set')
    list_auroc_scores.append(auroc(validation_is_false, list_num_sets) if is_binary else -1)
    list_auarc_scores.append(auarc(list_num_sets, correctness))
    list_aucpr_scores.append(aucpr(list_num_sets, correctness))

    ## Lexical similarity
    list_lex_sim: List[float] = []
    for idx in tqdm(range(len(correctness)), desc=f"Lexical sim [{language}]"):
        list_lex_sim.append(-compute_lexical_similarity(list_lex_sim_mat[idx]))

    list_method_name.append('lexical_sim')
    list_auroc_scores.append(auroc(validation_is_false, list_lex_sim) if is_binary else -1)
    list_auarc_scores.append(auarc(list_lex_sim, correctness))
    list_aucpr_scores.append(aucpr(list_lex_sim, correctness))

    ## Sum of eigenvalues (uses entailment similarity)
    list_sum_eigv: List[float] = []
    for idx in tqdm(range(len(correctness)), desc=f"Sum eigv [{language}]"):
        list_sum_eigv.append(get_spectral_eigv(list_entail_sim_mat[idx]))

    list_method_name.append('sum_eigv')
    list_auroc_scores.append(auroc(validation_is_false, list_sum_eigv) if is_binary else -1)
    list_auarc_scores.append(auarc(list_sum_eigv, correctness))
    list_aucpr_scores.append(aucpr(list_sum_eigv, correctness))

    ## Degree matrix (uses entailment similarity)
    list_degree_mat: List[float] = []
    for idx in tqdm(range(len(correctness)), desc=f"Degree mat [{language}]"):
        list_degree_mat.append(get_degreeuq(list_entail_sim_mat[idx])[0])

    list_method_name.append('degree_mat')
    list_auroc_scores.append(auroc(validation_is_false, list_degree_mat) if is_binary else -1)
    list_auarc_scores.append(auarc(list_degree_mat, correctness))
    list_aucpr_scores.append(aucpr(list_degree_mat, correctness))

    ## Eccentricity (uses entailment similarity)
    eigv_threshold = 0.9
    list_eccentricity: List[float] = []
    for idx in tqdm(range(len(correctness)), desc=f"Eccentricity [{language}]"):
        list_eccentricity.append(get_eccentricity(list_entail_sim_mat[idx])[0])

    list_method_name.append(f'eccentricity_thr{eigv_threshold}')
    list_auroc_scores.append(auroc(validation_is_false, list_eccentricity) if is_binary else -1)
    list_auarc_scores.append(auarc(list_eccentricity, correctness))
    list_aucpr_scores.append(aucpr(list_eccentricity, correctness))
    # ─────────────────────────────────────────────────────────────────────────

    df = pd.DataFrame({
        'method': list_method_name,
        'auroc': list_auroc_scores,
        'auarc': list_auarc_scores,
        'prr': list_aucpr_scores,
        'language': language,
        'accuracy': np.mean(correctness),
    })
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Multilingual graph baseline uncertainty measures'
    )
    parser.add_argument('--vanilla_json', required=True,
                        help='Path to vanilla inference JSON')
    parser.add_argument('--sampling_json', required=True,
                        help='Path to sampling inference JSON')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save per-language CSVs')
    parser.add_argument('--languages', nargs='+', default=None,
                        help='Languages to process (default: all available)')
    parser.add_argument('--dataset', default='dataset',
                        help='Dataset name (used in output filename)')
    parser.add_argument('--model_name', default='model',
                        help='Model name (used in output filename)')
    parser.add_argument('--num_generations', type=int, default=10,
                        help='Number of sampled generations per question')
    parser.add_argument('--suffix', default='',
                        help='Optional suffix for output filenames')
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument('--metric', choices=['prem', 'claude', 'gemini'], default='prem',
                        help='Accuracy metric: prem (default), claude, or gemini')
    parser.add_argument('--filter_ids_from', default=None,
                        help='Rephrased JSON path; filter to original question IDs only')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MULTILINGUAL GRAPH BASELINES")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset} | Model: {args.model_name}")
    logger.info(f"Metric: {args.metric} | Generations: {args.num_generations}")

    # Load data
    vanilla_data = load_json_data(args.vanilla_json)
    sampling_data = load_json_data(args.sampling_json)

    # Optional: filter to original (non-rephrased) question IDs
    if args.filter_ids_from:
        with open(args.filter_ids_from, 'r', encoding='utf-8') as f:
            rdata = json.load(f)
        valid_ids = {str(item['question_id']) for item in rdata if 'original_id' not in item}
        vanilla_data = [d for d in vanilla_data if str(d.get('question_id', '')) in valid_ids]
        sampling_data = [d for d in sampling_data if str(d.get('question_id', '')) in valid_ids]
        logger.info(f"Filtered to {len(vanilla_data)} samples from {args.filter_ids_from}")

    available_languages = get_languages(vanilla_data)
    languages = args.languages if args.languages else available_languages
    logger.info(f"Languages to process: {languages}")
    validate_generation_pair(
        vanilla_data, sampling_data, languages,
        num_sampling_outputs=args.num_generations,
        require_accuracy=args.metric in ('claude', 'gemini')
    )

    nli_model = MultilingualEntailmentDeberta()
    os.makedirs(args.output_dir, exist_ok=True)

    all_dfs: List[pd.DataFrame] = []

    for lang in languages:
        logger.info(f"\n{'='*40}")
        logger.info(f"Language: {lang}")
        logger.info(f"{'='*40}")

        df = compute_metrics_for_language(
            vanilla_data, sampling_data, lang, nli_model,
            num_generations=args.num_generations,
            metric=args.metric,
        )
        all_dfs.append(df)

        # Per-language CSV (naming mirrors VQA convention)
        lang_csv = os.path.join(
            args.output_dir,
            f"{args.dataset}_{args.model_name}_{args.num_generations}generations"
            f"{args.suffix}_{lang}_seed{args.random_seed}.csv"
        )
        df.to_csv(lang_csv, index=False)
        logger.info(f"Saved: {lang_csv}")
        print(df.to_string(index=False))

    # Combined CSV across all languages
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.model_name}_{args.num_generations}generations"
        f"{args.suffix}_seed{args.random_seed}.csv"
    )
    combined_df.to_csv(combined_csv, index=False)
    logger.info(f"\nCombined results saved: {combined_csv}")
    logger.info("DONE")
    return 0


if __name__ == "__main__":
    exit(main())
