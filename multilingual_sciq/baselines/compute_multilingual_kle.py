#!/usr/bin/env python3
"""
Compute multilingual Kernel Language Entropy (KLE) uncertainty measures.

Data extraction uses multilingual JSON format via multilingual_utils.
All uncertainty computation functions (get_kernels, all_graph_entropies,
full_sem_unc_plus_klu, all_semantic_entropies, all_semantic_entropies_diag,
compute_metrics) are IDENTICAL to compute_kle.py — only the data-level
extraction changes.

Usage:
    python compute_multilingual_kle.py \
        --vanilla_json /path/to/vanilla/generate.json \
        --sampling_json /path/to/sampling/generate.json \
        --output_dir ./kle_results \
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

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# SNNE repo utilities reused from the consolidated text QA package.
BIGSURE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BIGSURE_ROOT / "text_qa"))
from snne.kle.core import vn_entropy, normalize_kernel
from snne.kle.kernels import heat_kernel, matern_kernel
from snne.uncertainty.utils.eval_utils import auroc, auarc, aucpr, is_binary_list
from snne.uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from snne.uncertainty.uncertainty_measures.kernel_uncertainty import (
    get_entailment_graph,
    get_semantic_ids_graph,
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
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KLE hyperparameters — IDENTICAL to compute_kle.py
# ---------------------------------------------------------------------------
ALPHAS_RANGE = np.arange(0, 1.01, 0.1)
HEAT_T_RANGE = np.arange(0.1, 0.71, 0.1)
MATERN_KAPPA_RANGE = [1.0, 2.0, 3.0]
MATERN_NU_RANGE = [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Entailment model adapter (data-extraction layer)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# KLE computation functions — IDENTICAL to compute_kle.py
# ---------------------------------------------------------------------------

def get_from_sem_to_sentence_id(ordered_ids):
    from_sem_to_sentence_id = defaultdict(list)
    for i, el in enumerate(ordered_ids):
        from_sem_to_sentence_id[el].append(i)
    return from_sem_to_sentence_id


def reorder_by_semantic_ids(graph, semantic_ids, ordered_sem_ids):
    from_sem_to_sentence_id = get_from_sem_to_sentence_id(semantic_ids)
    new_graph = nx.Graph()
    for sem_id in ordered_sem_ids:
        for sent_id in from_sem_to_sentence_id[sem_id]:
            new_graph.add_node(sent_id)
    new_graph.add_edges_from(graph.edges)
    return new_graph


def get_kernels(graph):
    kernels = {}
    for t in HEAT_T_RANGE:
        kernels[f"heat_t={t:.2}"] = heat_kernel(graph, t=t)
        kernels[f"heatn_t={t:.2}"] = heat_kernel(graph, t=t, norm_lapl=True)
    for kappa in MATERN_KAPPA_RANGE:
        for nu in MATERN_NU_RANGE:
            kernels[f"matern_kappa={kappa:.2}_nu={nu:.2}"] = matern_kernel(graph, kappa=kappa, nu=nu)
            kernels[f"maternn_kappa={kappa:.2}_nu={nu:.2}"] = matern_kernel(graph, kappa=kappa, nu=nu, norm_lapl=True)
    return kernels


def all_graph_entropies(graph):
    kernels = get_kernels(graph)
    results = []
    for kernel_name, kernel in kernels.items():
        for scale in [True, False]:
            kernel_entropy = vn_entropy(kernel, scale=scale)
            postfix = "_s" if scale else ""
            results.append((f'{kernel_name}_kernel_entropy{postfix}', kernel_entropy))
    return results


def get_block_diagonal_sem_kernel(log_likelihoods_per_sem_id, semantic_ids, ordered_sem_ids):
    from_sem_to_sentence_id = get_from_sem_to_sentence_id(semantic_ids)
    blocks = []
    for i, sem_id in enumerate(ordered_sem_ids):
        block_size = len(from_sem_to_sentence_id[sem_id])
        blocks.append(
            torch.exp(torch.tensor(log_likelihoods_per_sem_id[sem_id]))
            * torch.ones((block_size, block_size))
            / block_size
        )
    return torch.block_diag(*blocks)


def full_sem_unc_plus_klu(graph, log_likelihoods_per_sem_id, semantic_ids, ordered_sem_ids):
    graph = reorder_by_semantic_ids(graph, semantic_ids, ordered_sem_ids)
    block_diag_sem_kernel = get_block_diagonal_sem_kernel(
        log_likelihoods_per_sem_id=log_likelihoods_per_sem_id,
        semantic_ids=semantic_ids,
        ordered_sem_ids=ordered_sem_ids,
    )
    alphas = ALPHAS_RANGE
    results = []
    kernels = get_kernels(graph)
    for kernel_name, kernel in kernels.items():
        for alpha in alphas:
            kernel = normalize_kernel(kernel) / kernel.shape[0]
            avg_kernel = alpha * torch.tensor(kernel) + (1 - alpha) * block_diag_sem_kernel
            avg_kernel = avg_kernel.numpy()
            success = False
            for jitter in [0, 1e-16, 1e-12]:
                try:
                    results.append((
                        f"full_klu_{kernel_name}_alpha_{alpha:.2}",
                        vn_entropy(avg_kernel, normalize=False, scale=False, jitter=jitter),
                    ))
                    success = True
                    if jitter > 0:
                        logging.warning(f"Had to use jitter for numerical stability: {jitter}")
                    break
                except Exception:
                    continue
            if not success:
                raise ValueError(f"Unable to calculate VNE for kernel {avg_kernel}")
    return results


def all_semantic_entropies(semantic_graph, log_likelihoods_per_sem_id):
    sem_entropies = torch.diag(torch.exp(torch.tensor(log_likelihoods_per_sem_id)))
    alphas = ALPHAS_RANGE
    results = []
    kernels = get_kernels(semantic_graph)
    for kernel_name, kernel in kernels.items():
        for alpha in alphas:
            kernel = normalize_kernel(kernel) / kernel.shape[0]
            avg_kernel = alpha * torch.tensor(kernel) + (1 - alpha) * sem_entropies
            avg_kernel = avg_kernel.numpy()
            results.append((
                f"semantic_kernel_{kernel_name}_alpha_{alpha:.2}",
                vn_entropy(avg_kernel, normalize=False, scale=False),
            ))
    return results


def all_semantic_entropies_diag(semantic_graph, log_likelihoods_per_sem_id):
    sem_entropies = torch.exp(torch.tensor(log_likelihoods_per_sem_id))
    results = []
    kernels = get_kernels(semantic_graph)
    for kernel_name, kernel in kernels.items():
        kernel = normalize_kernel(kernel) / kernel.shape[0]
        kernel_prod = torch.tensor(kernel) * sem_entropies
        kernel_sum = torch.tensor(kernel) + sem_entropies
        results.append((
            f"semantic_kernel_prod_{kernel_name}",
            vn_entropy(kernel_prod, normalize=True, scale=False),
        ))
        results.append((
            f"semantic_kernel_sum_{kernel_name}",
            vn_entropy(kernel_sum, normalize=True, scale=False),
        ))
    return results


# ---------------------------------------------------------------------------
# Per-language metric computation (identical logic to compute_kle.py)
# ---------------------------------------------------------------------------

def compute_metrics_for_language(
    vanilla_data: List[Dict],
    sampling_data: List[Dict],
    language: str,
    nli_model: MultilingualEntailmentDeberta,
    num_generations: int,
    metric: str = 'prem',
) -> pd.DataFrame:
    """Compute all KLE metrics for one language.

    The computation block mirrors compute_kle.py's compute_metrics() exactly;
    only data extraction at the top is multilingual-specific.
    """
    v_questions, v_answers, v_outputs, v_probs = extract_vanilla_data(vanilla_data, language)
    s_questions, s_answers, s_outputs, s_probs = extract_sampling_data(sampling_data, language)

    # ── Data extraction (multilingual-specific) ──────────────────────────────
    validation_is_true: List[float] = []
    list_responses: List[List[str]] = []
    list_generation_log_likelihoods: List[List[float]] = []
    list_semantic_ids: List[List[int]] = []

    for idx in tqdm(range(len(v_questions)), desc=f"Extracting data [{language}]"):
        responses = s_outputs[idx][:num_generations]
        probs = s_probs[idx][:num_generations]

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
        validation_is_true.append(is_correct)

        list_responses.append(responses)

        # Convert per-generation probabilities to log-likelihoods
        log_liks = np.log(np.array(probs, dtype=float) + 1e-10).tolist()
        list_generation_log_likelihoods.append(log_liks)

        # Semantic IDs via multilingual NLI clustering
        if responses:
            sem_ids = get_semantic_ids_using_entailment(
                responses, nli_model, strict_entailment=False
            )
        else:
            sem_ids = []
        list_semantic_ids.append(sem_ids)

    # ── Uncertainty computation — IDENTICAL to compute_kle.py ────────────────
    validation_is_false = [1.0 - v for v in validation_is_true]
    is_binary = is_binary_list(validation_is_false)
    entropies: Dict[str, List[float]] = defaultdict(list)

    for idx in tqdm(range(len(validation_is_true)), desc=f"KLE [{language}]"):
        responses = list_responses[idx]
        log_liks_agg = list_generation_log_likelihoods[idx][:num_generations]
        semantic_ids = list_semantic_ids[idx][:num_generations]

        if not responses or len(semantic_ids) == 0:
            continue

        unique_ids, log_likelihood_per_semantic_id = logsumexp_by_id(
            semantic_ids,
            log_liks_agg,
            agg='sum_normalized',
            return_unique_ids=True,
        )

        # Unweighted entailment graph
        graph = get_entailment_graph(
            responses, model=nli_model, example=None, is_weighted=False
        )
        for k, value in all_graph_entropies(graph):
            entropies[k].append(value)

        # Manually-weighted entailment graph
        weighted_graph = get_entailment_graph(
            responses, model=nli_model, example=None, is_weighted=True
        )
        for k, value in all_graph_entropies(weighted_graph):
            entropies[f"weighted_{k}"].append(value)

        # DeBERTa-probability-weighted graph (uses entailment confidence)
        weighted_graph_deberta = get_entailment_graph(
            responses, model=nli_model, example=None,
            is_weighted=True, weight_strategy="deberta"
        )
        for k, value in all_graph_entropies(weighted_graph_deberta):
            entropies[f"weighted_deberta_{k}"].append(value)

        # Semantic graph (KLE with log-likelihood prior)
        semantic_graph = get_semantic_ids_graph(
            responses,
            semantic_ids=semantic_ids,
            ordered_ids=unique_ids,
            model=nli_model,
            example=None,
        )
        for k, value in all_semantic_entropies(semantic_graph, log_likelihood_per_semantic_id):
            entropies[k].append(value)

        for k, value in all_semantic_entropies_diag(semantic_graph, log_likelihood_per_semantic_id):
            entropies[k].append(value)

        # Full KLE (graph kernel + semantic prior)
        for k, value in full_sem_unc_plus_klu(
            weighted_graph, log_likelihood_per_semantic_id,
            semantic_ids=semantic_ids, ordered_sem_ids=unique_ids
        ):
            entropies[k].append(value)

        for k, value in full_sem_unc_plus_klu(
            weighted_graph_deberta, log_likelihood_per_semantic_id,
            semantic_ids=semantic_ids, ordered_sem_ids=unique_ids
        ):
            entropies[f"deberta_{k}"].append(value)

    # Collect AUROC / AUARC / AUCPR — identical to compute_kle.py
    list_method_name: List[str] = []
    list_auroc_scores: List[float] = []
    list_auarc_scores: List[float] = []
    list_aucpr_scores: List[float] = []
    list_is_blackbox: List[bool] = []

    for entropy_name, entropy_list in entropies.items():
        _auroc = auroc(validation_is_false, entropy_list) if is_binary else -1
        _auarc = auarc(entropy_list, validation_is_true)
        _aucpr = aucpr(entropy_list, validation_is_true)
        list_method_name.append(entropy_name)
        list_auroc_scores.append(_auroc)
        list_auarc_scores.append(_auarc)
        list_aucpr_scores.append(_aucpr)
        list_is_blackbox.append(
            'semantic' not in entropy_name and 'full_klu' not in entropy_name
        )
    # ─────────────────────────────────────────────────────────────────────────

    df = pd.DataFrame({
        'method': list_method_name,
        'auroc': list_auroc_scores,
        'auarc': list_auarc_scores,
        'prr': list_aucpr_scores,
        'is_blackbox': list_is_blackbox,
        'language': language,
        'accuracy': np.mean(validation_is_true),
    })
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Multilingual KLE uncertainty measures'
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
    logger.info("MULTILINGUAL KLE")
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

    # Load model
    nli_model = MultilingualEntailmentDeberta()
    logger.info("Entailment model loading complete.")

    os.makedirs(args.output_dir, exist_ok=True)
    all_dfs: List[pd.DataFrame] = []

    for lang in languages:
        logger.info(f"\n{'='*40}")
        logger.info(f"Language: {lang}")
        logger.info(f"{'='*40}")

        df = compute_metrics_for_language(
            vanilla_data, sampling_data, lang,
            nli_model=nli_model,
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

        # Quick summary
        summary = df[['method', 'auroc', 'auarc', 'prr']].to_string(index=False)
        print(f"\n--- {lang} ---\n{summary}\n")

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
