#!/usr/bin/env python3
import os
import sys
import json
import math
import pickle
import argparse
import logging
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from snne.uncertainty.utils.normalization_utils import quantile_power_normalize
from snne.uncertainty.utils.eval_utils import auarc as compute_auarc, aucpr as compute_aucpr
from snne.utils.vanilla_accuracy import load_vanilla_accuracy_dict

# Re-use existing functions from the weighted spectral energy script
from snne.compute_spectral_energy_weighted_all_modes_rephrased import (
    load_accuracy_from_json,
    load_rephrased_generations,
    organize_by_original_id,
    extract_all_low_t_from_rephrasings,
    extract_all_high_t_with_paraphrase_indices,
    subsample_answers,
    load_results_npz,
    probs_to_weights_matrix,
    compute_similarity_matrix,
    compute_paraphrase_weights,
    compute_weighted_spectral_energy_from_W
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TASK_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_RESULTS_DIR = str(TASK_ROOT / "outputs" / "accuracy")

def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def compute_strawman_baselines(S, weights=None):
    M, N = S.shape
    if M == 0 or N == 0:
        return {}
    
    # Default to uniform weights if none provided
    if weights is None:
        weights = np.ones(N)
    else:
        weights = np.array(weights)
    
    # Use N (total probes) as the denominator instead of sum(weights).
    # This aligns the baseline with BiG-SURE's global normalization logic,
    # ensuring that low paraphrase weights correctly penalize the global confidence.
    denom = float(N)
    res = {}
    eps = 1e-12
    
    # Baseline 1: Weighted Average cross-temperature semantic similarity
    col_means = np.mean(S, axis=0) # shape (N,)
    avg_sim = float(np.sum(col_means * weights) / denom)
    res['avg_sim'] = avg_sim
    res['unc_avg_sim'] = 1.0 - avg_sim
    
    # Baseline 2: Weighted Average squared similarity
    # Mathematically equivalent to BiG-SURE confidence squared (Frobenius norm identity).
    col_sq_means = np.mean(S ** 2, axis=0) # shape (N,)
    avg_sq_sim = float(np.sum(col_sq_means * weights) / denom)
    res['avg_sq_sim'] = avg_sq_sim
    res['unc_avg_sq_sim'] = 1.0 - avg_sq_sim
    
    # Baseline 3: Weighted Max similarity to anchors
    max_anchor_sims = np.max(S, axis=0) # shape (N,)
    mean_max_sim = float(np.sum(max_anchor_sims * weights) / denom)
    res['max_sim'] = mean_max_sim
    res['unc_max_sim'] = 1.0 - mean_max_sim
    
    # Baseline 4: Weighted Degree-like statistic
    col_degrees = np.sum(S, axis=0) # shape (N,)
    mean_col_degree = float(np.sum(col_degrees * weights) / denom)
    res['mean_col_degree'] = mean_col_degree
    res['unc_mean_col_degree'] = 1.0 / (1.0 + mean_col_degree)
    
    # Baseline 5: Weighted Anchor-probe entropy
    col_sums = np.sum(S, axis=0)
    entropies = []
    normalized_entropies = []
    for j in range(N):
        if col_sums[j] > eps:
            p = S[:, j] / col_sums[j]
            H = -np.sum(p * np.log(p + eps))
        else:
            H = np.log(M) if M > 0 else 0.0 # Uniform if 0
        entropies.append(H)
        normalized_entropies.append(H / (np.log(M) + eps) if M > 1 else 0.0)
        
    mean_entropy = float(np.sum(np.array(entropies) * weights) / denom)
    mean_norm_entropy = float(np.sum(np.array(normalized_entropies) * weights) / denom)
    res['mean_entropy'] = mean_entropy
    res['mean_norm_entropy'] = mean_norm_entropy
    res['unc_mean_entropy'] = mean_norm_entropy # higher entropy = more uncertain
    
    # Baseline 6: Weighted Nearest-anchor margin
    margins = []
    for j in range(N):
        col_vals = S[:, j]
        if M >= 2:
            sorted_vals = np.sort(col_vals)[::-1]
            margin = sorted_vals[0] - sorted_vals[1]
        elif M == 1:
            margin = col_vals[0]
        else:
            margin = 0.0
        margins.append(margin)
    mean_margin = float(np.sum(np.array(margins) * weights) / denom)
    res['mean_margin'] = mean_margin
    res['unc_mean_margin'] = 1.0 - mean_margin # larger margin = more certain
    
    return res

def compute_all_baselines_with_precomputed(generations, precomputed_entailments, output_dir,
                                           score_mode="baseline_relaxed", combine="min",
                                           weighting_scheme="entropy_confidence",
                                           divergence_measure="js", sim_threshold=0.7,
                                           k_low_t=None, subsample_high_t=None, subsample_seed=0,
                                           accuracy_dict=None, self_similarity=False):
    by_original = organize_by_original_id(generations)
    logger.info(f"Found {len(by_original)} original questions")
    generation_ids = set(by_original)
    entailment_ids = set(precomputed_entailments)
    if generation_ids != entailment_ids:
        raise ValueError('Rephrased generations and entailment archive contain different question IDs.')
    if accuracy_dict is not None and set(map(str, accuracy_dict)) != entailment_ids:
        raise ValueError('Vanilla accuracy labels and entailment archive contain different question IDs.')
    
    graph_results = {}
    
    for orig_id in tqdm(sorted(by_original.keys()), desc="Processing"):
        precomp = precomputed_entailments[str(orig_id)]
        
        W_full = probs_to_weights_matrix(
            precomp['probs_fwd'],
            precomp['probs_bwd'],
            score_mode,
            combine
        )
        
        if self_similarity:
            W = W_full
            weights = np.ones(W.shape[1])
        else:
            rephrasings = by_original[orig_id]
            generated_high_t, generated_paraphrase_indices = extract_all_high_t_with_paraphrase_indices(rephrasings)
            all_high_t = precomp['high_texts']
            paraphrase_indices = precomp['paraphrase_indices']

            m_precomputed, n_precomputed = W_full.shape
            if precomp['m'] != m_precomputed or precomp['n'] != n_precomputed:
                raise ValueError(f'Entailment dimensions are inconsistent for question {orig_id!r}.')
            if len(all_high_t) != n_precomputed or len(paraphrase_indices) != n_precomputed:
                raise ValueError(f'Entailment metadata is inconsistent for question {orig_id!r}.')
            if generated_high_t[:n_precomputed] != all_high_t:
                raise ValueError(f'Rephrased responses no longer match entailments for question {orig_id!r}.')
            if generated_paraphrase_indices[:n_precomputed] != paraphrase_indices:
                raise ValueError(f'Rephrasing order no longer matches for question {orig_id!r}.')
            K_low = int(k_low_t) if k_low_t else m_precomputed
            K_high = int(subsample_high_t) if subsample_high_t else n_precomputed
            if K_low <= 0 or K_low > m_precomputed:
                raise ValueError(f'k_low_t must be in [1, {m_precomputed}], got {K_low}.')
            if K_high <= 0 or K_high > n_precomputed:
                raise ValueError(f'subsample_high_t must be in [1, {n_precomputed}], got {K_high}.')

            if K_low < m_precomputed and K_low > 0:
                W_current = W_full[:K_low, :]
            else:
                W_current = W_full

            if K_high < n_precomputed and K_high > 0:
                rnd = random.Random(subsample_seed)
                sampled_indices = sorted(rnd.sample(range(n_precomputed), K_high))
                W = W_current[:, sampled_indices]
                high_texts_for_weight = [all_high_t[i] for i in sampled_indices]
                para_indices_sampled = [paraphrase_indices[i] for i in sampled_indices]
            else:
                W = W_current
                high_texts_for_weight = all_high_t
                para_indices_sampled = paraphrase_indices

            weights = compute_paraphrase_weights(
                high_texts_for_weight, para_indices_sampled,
                weighting_scheme, divergence_measure, sim_threshold
            )
            if len(weights) != W.shape[1]:
                raise ValueError(f'Column weights do not match entailments for question {orig_id!r}.')

        # 1. BiG-SURE scores (Total, Top-1, Top-2)
        m_curr, n_curr = W.shape
        D_sqrt = np.sqrt(np.array(weights))
        W_weighted = W * D_sqrt
        
        try:
            _, S, _ = np.linalg.svd(W_weighted, full_matrices=False)
        except np.linalg.LinAlgError:
            S = np.zeros(min(m_curr, n_curr))
            
        max_energy_norm = math.sqrt(m_curr * n_curr)
        eps = 1e-10
        
        # Total Frobenius norm (Standard BiG-SURE)
        conf_total = float(np.linalg.norm(S) / (max_energy_norm + eps))
        # Top-1 singular value
        conf_top1 = float((S[0] if len(S) >= 1 else 0.0) / (max_energy_norm + eps))
        # Top-2 singular values (Frobenius norm of rank-2)
        conf_top2 = float((np.linalg.norm(S[:2]) if len(S) >= 2 else 0.0) / (max_energy_norm + eps))

        # Distributional Spectral Metrics
        energy_sq_total = np.sum(S**2) + 1e-12
        probs = (S**2) / energy_sq_total
        spec_entropy = -np.sum(probs * np.log(probs + 1e-12))
        spec_entropy_norm = float(spec_entropy / np.log(len(S)) if len(S) > 1 else 0.0)
        stable_rank = float(energy_sq_total / (S[0]**2 + 1e-12) if len(S) >= 1 else 1.0)
        spec_gap = float((S[0] - S[1]) if len(S) >= 2 else (S[0] if len(S) == 1 else 0.0))
        top1_ratio = float((S[0]**2) / energy_sq_total if len(S) >= 1 else 0.0)
        top2_ratio = float(np.sum(S[:2]**2) / energy_sq_total if len(S) >= 2 else 1.0)

        # Strawman baselines (now with weights)
        straw_res = compute_strawman_baselines(W, weights)

        if accuracy_dict is not None and str(orig_id) in accuracy_dict:
            accuracy = accuracy_dict[str(orig_id)]
        else:
            accuracy = precomp.get('accuracy')
            if accuracy is None and not self_similarity:
                reph0_data = by_original[orig_id][0].get('data', {})
                if 'most_likely_answer' in reph0_data:
                    accuracy = reph0_data['most_likely_answer'].get('accuracy')
                if accuracy is None:
                    accuracy = reph0_data.get('accuracy')
                if accuracy is not None:
                    try:
                        accuracy = float(accuracy)
                    except:
                        pass

        combined_res = {
            'accuracy': accuracy,
            'unc_big_sure': 1.0 - conf_total,
            'conf_big_sure': conf_total,
            'conf_big_sure_top1': conf_top1,
            'conf_big_sure_top2': conf_top2,
            'spec_entropy': spec_entropy_norm,
            'stable_rank': stable_rank,
            'spec_gap': spec_gap,
            'top1_ratio': top1_ratio,
            'top2_ratio': top2_ratio,
            'm': m_curr, 'n': n_curr
        }
        combined_res.update(straw_res)
        graph_results[str(orig_id)] = combined_res

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_pickle(graph_results, output_path / f"results_baselines_{score_mode}_{combine}.pkl")
    
    return graph_results

def evaluate_baselines(graph_results, dataset=None):
    question_ids = sorted(graph_results.keys())
    valid_qids = [qid for qid in question_ids if graph_results[qid].get('accuracy') is not None]
    
    if not valid_qids:
        return {}
    
    accuracies = np.array([graph_results[qid]['accuracy'] for qid in valid_qids])
    if len(accuracies) == 0 or len(set(accuracies)) <= 1 or dataset == "fr-en":
        return {}
        
    metrics_to_eval = [
        ('big_sure', 'conf_big_sure', True), 
        ('big_sure_top1', 'conf_big_sure_top1', True),
        ('big_sure_top2', 'conf_big_sure_top2', True),
        ('spec_entropy', 'spec_entropy', False),
        ('stable_rank', 'stable_rank', False),
        ('spec_gap', 'spec_gap', True),
        ('top1_ratio', 'top1_ratio', True),
        ('top2_ratio', 'top2_ratio', True),
        ('avg_sim', 'avg_sim', True),
        ('avg_sq_sim', 'avg_sq_sim', True),
        ('max_sim', 'max_sim', True),
        ('mean_col_degree', 'mean_col_degree', True),
        ('mean_entropy', 'mean_norm_entropy', False), 
        ('mean_margin', 'mean_margin', True) 
    ]
    
    eps = 1e-10
    auc_results = {}
    for short_name, metric_key, is_conf in metrics_to_eval:
        raw_vals = np.array([graph_results[qid][metric_key] for qid in valid_qids])
        
        if short_name.startswith('big_sure'):
            uncs_raw = -np.log(raw_vals + eps)
        elif is_conf:
            uncs_raw = np.max(raw_vals) - raw_vals + 1e-5
        else:
            uncs_raw = raw_vals - np.min(raw_vals) + 1e-5
            
        uncs_normalized = quantile_power_normalize(uncs_raw)
        
        auc = roc_auc_score(accuracies, uncs_normalized)
        auc_results[f"auroc_unc_{short_name}"] = float(auc)
        
        auarc_val = compute_auarc(uncs_normalized.tolist(), accuracies.tolist())
        aucpr_val = compute_aucpr(uncs_normalized.tolist(), accuracies.tolist())
        
        auc_results[f"auarc_unc_{short_name}"] = float(auarc_val) if auarc_val is not None else 0.0
        auc_results[f"aucpr_unc_{short_name}"] = float(aucpr_val) if aucpr_val is not None else 0.0
        
    return auc_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--wandb_base_dir', type=str, default=str(TASK_ROOT / 'outputs' / 'wandb'))
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--entailments_file', type=str, required=True)
    parser.add_argument('--score_mode', type=str, default='baseline_relaxed')
    parser.add_argument('--combine', type=str, default='min')
    parser.add_argument('--weighting_scheme', type=str, default='entropy_confidence')
    parser.add_argument('--sim_threshold', type=float, default=0.5)
    parser.add_argument('--k_low_t', type=int, default=3)
    parser.add_argument('--subsample_high_t', type=int, default=10)
    parser.add_argument('--subsample_seed', type=int, default=2000)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--metric', type=str, default='squad')
    parser.add_argument('--accuracy_dir', type=str, default=ACCURACY_RESULTS_DIR)
    parser.add_argument('--seed', type=int, default=None,
                        help='Generation seed used to identify the vanilla SQuAD labels.')
    parser.add_argument('--vanilla_run_dir', type=str, default=None,
                        help='Explicit vanilla run directory used for SQuAD labels.')
    parser.add_argument('--non_rephrased', action='store_true')
    args = parser.parse_args()
    
    run_path = Path(args.wandb_base_dir) / args.run_dir
    
    accuracy_dict = None
    if args.metric == 'squad':
        vanilla_model_query = args.model_name
        if args.seed is not None:
            vanilla_model_query = f"{args.model_name}_seed{args.seed}"
        accuracy_dict, _ = load_vanilla_accuracy_dict(
            args.dataset,
            vanilla_model_query,
            vanilla_run_dir=args.vanilla_run_dir,
            wandb_base_dir=args.wandb_base_dir,
        )
    
    generations = load_rephrased_generations(run_path)
    precomputed_entailments = load_results_npz(args.entailments_file)
    
    graph_results = compute_all_baselines_with_precomputed(
        generations, precomputed_entailments, args.output_dir,
        score_mode=args.score_mode, combine=args.combine,
        weighting_scheme=args.weighting_scheme,
        sim_threshold=args.sim_threshold,
        k_low_t=args.k_low_t, subsample_high_t=args.subsample_high_t,
        subsample_seed=args.subsample_seed,
        accuracy_dict=accuracy_dict,
        self_similarity=args.non_rephrased
    )
    
    auc_results = evaluate_baselines(graph_results, args.dataset)
    
    summary = {
        'dataset': args.dataset, 
        'model_name': args.model_name,
        'score_mode': args.score_mode,
        'combine': args.combine,
        'weighting_scheme': args.weighting_scheme,
        'metric': args.metric,
        'non_rephrased': args.non_rephrased,
        'n_questions': len(graph_results)
    }
    summary.update(auc_results)
    
    output_path = Path(args.output_dir)
    csv_name = f"summary_baselines_{args.score_mode}_{args.combine}.csv"
    pd.DataFrame([summary]).to_csv(output_path / csv_name, index=False)
    
    logger.info(f"Results saved to {output_path / csv_name}")
    logger.info(f"AUC Results: {auc_results}")

if __name__ == "__main__":
    main()
