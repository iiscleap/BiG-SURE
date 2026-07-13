#!/usr/bin/env python3
"""
WEIGHTED Spectral Energy for OKVQA dataset.

This script computes WEIGHTED spectral energy (SVD-based confidence) using
precomputed DeBERTa entailment probabilities for OKVQA.

Key features:
- Uses precomputed entailments from okvqa_entailments/*.npz files
- Loads accuracy from okvqa_accuracy_results/*.json files
- OKVQA accuracy is continuous [0,1], binarized at 0.5 threshold (same as VQA)
- Supports paraphrase-aware column weighting

Similarity Measures:
- deberta: Use precomputed DeBERTa entailment probabilities (default)
- rouge: Use ROUGE-L similarity between response texts
- jaccard: Use Jaccard lexical similarity

Weighting Schemes:
- none: No weighting (uniform weights = 1.0 for all columns)
- exp_consistency: exp(-divergence) - rewards consistency
- inverse_consistency: 1/(1+divergence) - rewards consistency
- linear_consistency: max(0, 1-divergence) - rewards consistency
- entropy_confidence: 1 - H(paraphrase) - rewards internal confidence
- hybrid: exp(-div) * (1-H) - combines consistency and confidence
- dominance: max(prob) - rewards single dominant answer
- inverse_cluster_count: 1/K - rewards fewer clusters

Scoring modes (for deberta similarity):
- baseline_relaxed: (1-p_contra_fwd)×(1-p_contra_bwd)×(1-p_neutral_fwd×p_neutral_bwd)
- baseline_strict: p_entail_fwd × p_entail_bwd
- entail_prob: p_entail with combine (min/geom_mean/mean)
- entail_over_noncontrad: p_entail/(p_entail+p_neutral) with combine
- noncontrad_prob: (p_entail+p_neutral) with combine

Usage:
    python compute_spectral_energy_weighted_okvqa.py \
        --entailments_file okvqa_entailments/gemma-3-12b_perturbed.npz \
        --accuracy_file okvqa_accuracy_results/gemma-3-12b_okvqa_accuracy.json \
        --output_dir okvqa_spectral_energy_results/gemma \
        --score_mode baseline_relaxed \
        --weighting_scheme entropy_confidence
"""

import os
import sys
import re
import json
import math
import pickle
import argparse
import logging
import random
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import shared normalization function
from snne.compute_vanilla_highT_entailment import quantile_power_normalize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== SIMILARITY FUNCTIONS ====================

def normalize_text(s: str) -> str:
    """Normalize text for similarity computation."""
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def char_ngrams(text: str, n: int = 3):
    """Get character n-grams from text."""
    t = normalize_text(text)
    if not t:
        return set()
    if len(t) < n:
        return {t}
    return {t[i:i+n] for i in range(len(t) - n + 1)}


def jaccard_set(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / (len(a | b) + 1e-12)


def jaccard_similarity(a: str, b: str) -> float:
    """
    Hybrid lexical similarity combining:
    - SequenceMatcher ratio (50%)
    - Token Jaccard (25%)
    - Trigram Jaccard (25%)
    """
    a0, b0 = normalize_text(a), normalize_text(b)
    if not a0 and not b0:
        return 1.0
    if not a0 or not b0:
        return 0.0
    if a0 == b0:
        return 1.0

    sm = SequenceMatcher(None, a0, b0).ratio()
    tok = jaccard_set(set(a0.split()), set(b0.split()))
    tri = jaccard_set(char_ngrams(a0, 3), char_ngrams(b0, 3))
    sim = 0.5 * sm + 0.25 * tri + 0.25 * tok
    return float(max(0.0, min(1.0, sim)))


def rouge_l_similarity(a: str, b: str) -> float:
    """
    Compute ROUGE-L F1 similarity using SequenceMatcher LCS approximation.
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    
    return SequenceMatcher(None, a_norm.split(), b_norm.split()).ratio()


# ==================== CLUSTERING & DIVERGENCE FUNCTIONS ====================

def _clusters_connected_components(texts, sim_threshold=0.7, sim_func=jaccard_similarity):
    """Return (assignments, cluster_sizes, n, K). Clustering via connected components."""
    high = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
    n = len(high)
    if n == 0:
        return [], [], 0, 0

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i in range(n):
        for j in range(i + 1, n):
            if sim_func(high[i], high[j]) >= sim_threshold:
                union(i, j)

    roots = [find(i) for i in range(n)]
    root_to_cid = {}
    assignments = []
    for r in roots:
        if r not in root_to_cid:
            root_to_cid[r] = len(root_to_cid)
        assignments.append(root_to_cid[r])

    counts = Counter(assignments)
    K = len(counts)
    cluster_sizes = [counts[cid] for cid in range(K)]
    return assignments, cluster_sizes, n, K


def _compute_distribution_from_clusters(texts, sim_threshold=0.7, sim_func=jaccard_similarity):
    """Cluster texts by lexical similarity and return cluster-size distribution."""
    if not texts:
        return np.array([])
    
    assignments, cluster_sizes, n, K = _clusters_connected_components(texts, sim_threshold=sim_threshold, sim_func=sim_func)
    if n == 0 or K == 0:
        return np.array([1.0])
    
    dist = np.array(cluster_sizes, dtype=float) / float(n)
    return dist


def _jensen_shannon_divergence(P, Q):
    """Compute Jensen-Shannon divergence between two probability distributions."""
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)
    P = P / (P.sum() + 1e-12)
    Q = Q / (Q.sum() + 1e-12)
    
    max_len = max(len(P), len(Q))
    if len(P) < max_len:
        P = np.pad(P, (0, max_len - len(P)), mode='constant')
    if len(Q) < max_len:
        Q = np.pad(Q, (0, max_len - len(Q)), mode='constant')
    
    M = 0.5 * (P + Q)
    eps = 1e-12
    kl_pm = np.sum(P * np.log((P + eps) / (M + eps)))
    kl_qm = np.sum(Q * np.log((Q + eps) / (M + eps)))
    js = 0.5 * kl_pm + 0.5 * kl_qm
    
    return float(js)


def _kl_divergence(P, Q):
    """Compute KL divergence KL(P||Q)."""
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)
    P = P / (P.sum() + 1e-12)
    Q = Q / (Q.sum() + 1e-12)
    
    max_len = max(len(P), len(Q))
    if len(P) < max_len:
        P = np.pad(P, (0, max_len - len(P)), mode='constant')
    if len(Q) < max_len:
        Q = np.pad(Q, (0, max_len - len(Q)), mode='constant')
        
    eps = 1e-12
    kl = np.sum(P * np.log((P + eps) / (Q + eps)))
    return float(kl)


# ==================== UTILITY FUNCTIONS ====================

def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_accuracy_json(accuracy_path):
    """
    Load accuracy file from vqa_accuracy_results JSON.
    Returns dict mapping question ID -> accuracy (float, 0.0 to 1.0)
    """
    with open(accuracy_path, 'r') as f:
        data = json.load(f)
    
    accuracy_dict = {}
    per_example = data.get('per_example', {})
    
    for qid, example_data in per_example.items():
        accuracy_dict[str(qid)] = float(example_data.get('accuracy', 0.0))
    
    return accuracy_dict


def load_results_npz(filepath):
    """Load precomputed entailment results from compressed numpy archive."""
    data = np.load(filepath, allow_pickle=True)
    
    # Extract question IDs from keys
    question_ids = set()
    for key in data.keys():
        if key.startswith('q_') and '_probs_fwd' in key:
            qid = key.replace('q_', '').replace('_probs_fwd', '')
            question_ids.add(qid)
    
    question_ids = sorted(list(question_ids))
    
    results = {}
    for qid in question_ids:
        prefix = f"q_{qid}_"
        try:
            results[qid] = {
                'probs_fwd': data[f"{prefix}probs_fwd"],
                'probs_bwd': data[f"{prefix}probs_bwd"],
                'low_texts': data[f"{prefix}low_texts"].tolist(),
                'high_texts': data[f"{prefix}high_texts"].tolist(),
                'm': int(data[f"{prefix}m"][0]),
                'n': int(data[f"{prefix}n"][0])
            }
            # Load paraphrase indices if available
            if f"{prefix}paraphrase_indices" in data.files:
                results[qid]['paraphrase_indices'] = data[f"{prefix}paraphrase_indices"].tolist()
            else:
                # Fallback: no paraphrase info, all same group
                results[qid]['paraphrase_indices'] = [0] * results[qid]['n']
            if f"{prefix}accuracy" in data.files:
                results[qid]['accuracy'] = float(data[f"{prefix}accuracy"][0])
        except KeyError as e:
            logger.warning(f"Missing key for question {qid}: {e}")
            continue
    
    return results


# ==================== WEIGHT MATRIX COMPUTATION ====================

def probs_to_weights_matrix(probs_fwd, probs_bwd, score_mode, combine):
    """Convert precomputed probability matrices to edge weights (for deberta)."""
    eps = 1e-10
    
    def probs_to_weight(probs, mode):
        p_contradiction = probs[:, :, 0]
        p_neutral = probs[:, :, 1]
        p_entail = probs[:, :, 2]
        
        if mode == "entail_prob":
            return p_entail
        elif mode == "entail_over_noncontrad":
            return p_entail / (p_entail + p_neutral + eps)
        elif mode == "noncontrad_prob":
            return p_entail + p_neutral
        elif mode == "baseline_relaxed":
            return (p_entail > 0.5).astype(np.float32)
        elif mode == "baseline_strict":
            return (np.argmax(probs, axis=2) == 2).astype(np.float32)
        else:
            raise ValueError(f"Unknown score_mode: {mode}")
    
    W_forward = probs_to_weight(probs_fwd, score_mode)
    W_backward = probs_to_weight(probs_bwd, score_mode)
    
    if combine == "min":
        W = np.minimum(W_forward, W_backward)
    elif combine == "geom_mean":
        W = np.sqrt(W_forward * W_backward + eps)
    elif combine == "mean":
        W = 0.5 * (W_forward + W_backward)
    else:
        raise ValueError(f"Unknown combine mode: {combine}")
    
    return W.astype(np.float32)


def compute_similarity_matrix(low_texts, high_texts, similarity):
    """Compute bipartite similarity matrix using specified similarity function."""
    m, n = len(low_texts), len(high_texts)
    W = np.zeros((m, n), dtype=np.float32)
    
    if similarity == "jaccard":
        sim_func = jaccard_similarity
    elif similarity == "rouge":
        sim_func = rouge_l_similarity
    else:
        raise ValueError(f"Unknown similarity: {similarity}")
    
    for i in range(m):
        for j in range(n):
            W[i, j] = sim_func(low_texts[i], high_texts[j])
    
    return W


# ==================== PARAPHRASE WEIGHTING ====================

def compute_paraphrase_weights(high_texts, paraphrase_indices, weighting_scheme, 
                                divergence_measure="js", sim_threshold=0.7):
    """
    Compute weights for each high-temp answer based on its paraphrase's divergence.
    
    Args:
        high_texts: list of high-temp answer strings
        paraphrase_indices: list of paraphrase indices for each high-temp answer
        weighting_scheme: strategy for computing weights (use "none" for uniform weights)
        divergence_measure: "js" or "kl"
        sim_threshold: threshold for clustering
    
    Returns:
        numpy array of weights (one per high-temp answer)
    """
    n = len(high_texts)
    if n == 0:
        return np.array([])
    
    # No weighting - return uniform weights
    if weighting_scheme == "none":
        return np.ones(n)
    
    # Compute global distribution
    dist_all = _compute_distribution_from_clusters(high_texts, sim_threshold=sim_threshold)
    
    weights = []
    
    for j in range(n):
        para_idx = paraphrase_indices[j]
        
        # Get all high-temp answers from this paraphrase
        paraphrase_high = [high_texts[k] for k in range(n) if paraphrase_indices[k] == para_idx]
        
        if not paraphrase_high:
            weights.append(1.0)
            continue
        
        # Compute distribution for this paraphrase
        dist_paraphrase = _compute_distribution_from_clusters(paraphrase_high, sim_threshold=sim_threshold)
        
        # Compute divergence
        if divergence_measure == "kl":
            div_val = _kl_divergence(dist_all, dist_paraphrase)
        else:
            div_val = _jensen_shannon_divergence(dist_all, dist_paraphrase)
        
        # Calculate weight based on scheme
        w_val = 1.0
        
        if weighting_scheme == "exp_consistency":
            w_val = np.exp(-float(div_val))
            
        elif weighting_scheme == "inverse_consistency":
            w_val = 1.0 / (1.0 + float(div_val))
            
        elif weighting_scheme == "linear_consistency":
            w_val = max(0.0, 1.0 - float(div_val))
            
        elif weighting_scheme == "entropy_confidence":
            H_raw = -np.sum(dist_paraphrase * np.log(dist_paraphrase + 1e-12))
            K_eff = len(dist_paraphrase)
            if K_eff <= 1:
                H_norm = 0.0
            else:
                H_norm = H_raw / (np.log(K_eff) + 1e-12)
            w_val = 1.0 - float(H_norm)

        elif weighting_scheme == "dominance":
            if len(dist_paraphrase) > 0:
                w_val = float(np.max(dist_paraphrase))
            else:
                w_val = 0.0

        elif weighting_scheme == "inverse_cluster_count":
            K_eff = len(dist_paraphrase)
            if K_eff > 0:
                w_val = 1.0 / float(K_eff)
            else:
                w_val = 1.0

        elif weighting_scheme == "hybrid":
            w_consist = np.exp(-float(div_val))
            H_raw = -np.sum(dist_paraphrase * np.log(dist_paraphrase + 1e-12))
            K_eff = len(dist_paraphrase)
            if K_eff <= 1:
                H_norm = 0.0
            else:
                H_norm = H_raw / (np.log(K_eff) + 1e-12)
            w_conf = 1.0 - float(H_norm)
            w_val = w_consist * w_conf

        else:
            w_val = np.exp(-float(div_val))

        weights.append(w_val)
    
    return np.array(weights)


# ==================== WEIGHTED SPECTRAL ENERGY COMPUTATION ====================

def compute_weighted_spectral_energy_from_W(W, weights):
    """Compute weighted spectral energy from weight matrix W and column weights."""
    m, n = W.shape
    
    if m == 0 or n == 0:
        return None
    
    # Apply column weights: W_weighted = W @ D^0.5
    D_sqrt = np.sqrt(np.array(weights))
    D_sqrt_diag = np.diag(D_sqrt)
    W_weighted = W @ D_sqrt_diag
    
    try:
        _, S, _ = np.linalg.svd(W_weighted, full_matrices=False)
    except np.linalg.LinAlgError:
        S = np.zeros(min(m, n))
    
    energy = np.linalg.norm(S)
    max_energy = math.sqrt(m * n)
    eps = 1e-10
    
    conf = energy / (max_energy + eps)
    unc = 1.0 - conf
    
    return float(conf), float(unc), int(m), int(n)


def compute_weighted_spectral_energy_with_precomputed(
    precomputed_entailments,
    output_dir,
    score_mode="baseline_relaxed",
    combine="min",
    weighting_scheme="entropy_confidence",
    divergence_measure="js",
    sim_threshold=0.7,
    subsample_high_t=None,
    subsample_seed=0,
    accuracy_dict=None
):
    """Compute weighted spectral energy using precomputed deberta entailment probabilities."""
    logger.info(f"Processing {len(precomputed_entailments)} questions")
    logger.info(f"Score mode: {score_mode}, Combine: {combine}")
    logger.info(f"Weighting scheme: {weighting_scheme}, Divergence: {divergence_measure}")
    if subsample_high_t:
        logger.info(f"Subsampling to {subsample_high_t} high-T from precomputed")
    
    graph_results = {}
    
    for qid in tqdm(sorted(precomputed_entailments.keys()), desc="Processing"):
        precomp = precomputed_entailments[qid]
        
        # Get high-temp texts and paraphrase indices for weight computation
        all_high_t = precomp['high_texts']
        paraphrase_indices = precomp.get('paraphrase_indices', [0] * len(all_high_t))
        
        W_full = probs_to_weights_matrix(
            precomp['probs_fwd'],
            precomp['probs_bwd'],
            score_mode,
            combine
        )

        n_precomputed = W_full.shape[1]
        K_high = int(subsample_high_t) if subsample_high_t else n_precomputed
        
        if K_high < n_precomputed and K_high > 0:
            rnd = random.Random(subsample_seed)
            # Sample indices
            all_indices = list(range(n_precomputed))
            sampled_indices = sorted(rnd.sample(all_indices, K_high))
            
            # Subsample W, high_texts, and paraphrase_indices
            W = W_full[:, sampled_indices]
            high_texts = [all_high_t[i] for i in sampled_indices] if len(all_high_t) >= n_precomputed else all_high_t
            para_indices_sampled = [paraphrase_indices[i] for i in sampled_indices] if len(paraphrase_indices) >= n_precomputed else [0] * K_high
        else:
            W = W_full
            high_texts = all_high_t
            para_indices_sampled = paraphrase_indices

        
        # Compute weights using paraphrase-aware weighting
        weights = compute_paraphrase_weights(
            high_texts,
            para_indices_sampled,
            weighting_scheme,
            divergence_measure,
            sim_threshold
        )
        
        # Ensure weights match W columns
        if len(weights) != W.shape[1]:
            weights = np.ones(W.shape[1])
        
        result = compute_weighted_spectral_energy_from_W(W, weights)
        if result is None:
            continue
        
        confidence, uncertainty, m, n = result
        
        # Get accuracy (prefer external dict, fallback to precomputed)
        if accuracy_dict is not None and str(qid) in accuracy_dict:
            accuracy = accuracy_dict[str(qid)]
        else:
            accuracy = precomp.get('accuracy')
        
        graph_results[str(qid)] = {
            'accuracy': accuracy,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'spectral_energy_weighted': confidence,
            'W_mean': float(W.mean()),
            'weights_mean': float(weights.mean()),
            'm': m, 'n': n
        }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_pickle(graph_results, output_path / f"results_{score_mode}_{combine}_{weighting_scheme}.pkl")
    
    return graph_results


def compute_weighted_spectral_energy_with_similarity(
    precomputed_entailments,
    output_dir,
    similarity,
    weighting_scheme="entropy_confidence",
    divergence_measure="js",
    sim_threshold=0.7,
    subsample_high_t=None,
    subsample_seed=0,
    accuracy_dict=None
):
    """Compute weighted spectral energy using text-based similarity (jaccard or rouge)."""
    logger.info(f"Processing {len(precomputed_entailments)} questions")
    logger.info(f"Using {similarity} similarity")
    logger.info(f"Weighting scheme: {weighting_scheme}, Divergence: {divergence_measure}")
    if subsample_high_t:
        logger.info(f"Subsampling to {subsample_high_t} high-T")
    
    graph_results = {}
    
    for qid in tqdm(sorted(precomputed_entailments.keys()), desc="Processing"):
        precomp = precomputed_entailments[qid]
        
        low_texts = precomp['low_texts']
        all_high_t = precomp['high_texts']
        paraphrase_indices = precomp.get('paraphrase_indices', [0] * len(all_high_t))
        
        # Subsample high texts
        n_high = len(all_high_t)
        K_high = int(subsample_high_t) if subsample_high_t else n_high
        
        if K_high < n_high and K_high > 0:
            rnd = random.Random(subsample_seed)
            all_indices = list(range(n_high))
            sampled_indices = sorted(rnd.sample(all_indices, K_high))
            
            high_texts = [all_high_t[i] for i in sampled_indices]
            para_indices_sampled = [paraphrase_indices[i] for i in sampled_indices]
        else:
            high_texts = all_high_t
            para_indices_sampled = paraphrase_indices
        
        if len(low_texts) == 0 or len(high_texts) == 0:
            continue
        
        # Compute similarity matrix
        W = compute_similarity_matrix(low_texts, high_texts, similarity)
        
        # Compute weights using paraphrase-aware weighting
        weights = compute_paraphrase_weights(
            high_texts,
            para_indices_sampled,
            weighting_scheme,
            divergence_measure,
            sim_threshold
        )
        
        result = compute_weighted_spectral_energy_from_W(W, weights)
        if result is None:
            continue
        
        confidence, uncertainty, m, n = result
        
        # Get accuracy (prefer external dict, fallback to precomputed)
        if accuracy_dict is not None and str(qid) in accuracy_dict:
            accuracy = accuracy_dict[str(qid)]
        else:
            accuracy = precomp.get('accuracy')
        
        graph_results[str(qid)] = {
            'accuracy': accuracy,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'spectral_energy_weighted': confidence,
            'W_mean': float(W.mean()),
            'weights_mean': float(weights.mean()),
            'm': m, 'n': n
        }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_pickle(graph_results, output_path / f"results_{similarity}_{weighting_scheme}.pkl")
    
    return graph_results


def load_vanilla_example_metadata(vanilla_pkl_path):
    """Load question / greedy answer text from vanilla validation_generations.pkl."""
    if not vanilla_pkl_path:
        return {}
    with open(vanilla_pkl_path, 'rb') as f:
        data = pickle.load(f)
    metadata = {}
    for qid, example in data.items():
        metadata[str(qid)] = {
            'question': example.get('question', ''),
            'answer': example.get('most_likely_answer', {}).get('response', ''),
        }
    return metadata


def build_spectral_method_name(similarity, score_mode, combine, weighting_scheme):
    if similarity == 'deberta':
        return f'spectral_{similarity}_{score_mode}_{combine}_{weighting_scheme}'
    return f'spectral_{similarity}_{weighting_scheme}'


def compute_auroc(graph_results, output_dir, score_mode, combine, weighting_scheme, similarity=None, metric_threshold=0.5):
    """
    Compute AUROC using uncertainty to predict correctness.
    
    For VQA, accuracy can be continuous (0.0 to 1.0). We binarize at 0.5 threshold.
    Returns (auroc, per_question_auroc_scores) where scores are used for AUROC ranking.
    """
    question_ids = sorted(graph_results.keys())
    
    confidences = np.array([graph_results[qid]['confidence'] for qid in question_ids])
    accuracies = [graph_results[qid].get('accuracy') for qid in question_ids]
    
    valid_mask = [a is not None for a in accuracies]
    valid_confidences = confidences[valid_mask]
    valid_accuracies = np.array([a for a in accuracies if a is not None])
    valid_qids = [qid for qid, ok in zip(question_ids, valid_mask) if ok]
    
    per_question_auroc_scores = {qid: np.nan for qid in question_ids}
    
    if len(valid_accuracies) == 0:
        logger.warning("No valid accuracy labels found")
        return None, per_question_auroc_scores
    
    binary_accuracies = (valid_accuracies >= metric_threshold).astype(float)
    
    eps = 1e-10
    uncertainties_raw = -np.log(valid_confidences + eps)
    confidence_normalized = quantile_power_normalize(uncertainties_raw)
    for qid, score in zip(valid_qids, confidence_normalized):
        per_question_auroc_scores[qid] = float(score)
    
    auroc = None
    if len(binary_accuracies) > 0 and len(set(binary_accuracies)) > 1:
        auroc = roc_auc_score(binary_accuracies, confidence_normalized)
    
    sim_str = f", Similarity: {similarity}" if similarity else ""
    auroc_str = f"{auroc:.4f}" if auroc else "N/A"
    logger.info(f"Score: {score_mode}, Combine: {combine}, Weighting: {weighting_scheme}{sim_str} -> AUROC: {auroc_str}")
    
    overall_accuracy = float(valid_accuracies.mean())
    binary_accuracy = float(binary_accuracies.mean())
    logger.info(f"Overall accuracy (mean): {overall_accuracy:.4f}")
    logger.info(f"Binary accuracy (>={metric_threshold}): {binary_accuracy:.4f} ({len(valid_accuracies)} questions)")
    
    return auroc, per_question_auroc_scores


def save_per_example_spectral_csv(
    graph_results,
    output_path,
    method_name,
    vanilla_metadata,
    per_question_auroc_scores,
    metric_threshold=0.5,
):
    """Save per-question uncertainties alongside question, answer, and labels."""
    rows = []
    for qid in sorted(graph_results.keys()):
        r = graph_results[qid]
        meta = vanilla_metadata.get(str(qid), {}) if vanilla_metadata else {}
        acc = r.get('accuracy')
        label_binary = (acc >= metric_threshold) if acc is not None else None
        rows.append({
            'question_id': str(qid),
            'question': meta.get('question', ''),
            'answer': meta.get('answer', ''),
            'label': acc,
            'label_binary': label_binary,
            'method': method_name,
            'uncertainty': r.get('uncertainty'),
            'confidence': r.get('confidence'),
            'uncertainty_auroc_score': per_question_auroc_scores.get(qid, np.nan),
        })
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info(f"Saved per-example uncertainties: {output_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description='Weighted Spectral Energy for OKVQA')
    
    # Input files
    parser.add_argument('--entailments_file', type=str, required=True,
                       help='Path to precomputed entailments .npz file')
    parser.add_argument('--accuracy_file', type=str, default=None,
                       help='Path to accuracy JSON file (optional, uses npz if not provided)')
    parser.add_argument('--vanilla_pkl', type=str, default=None,
                       help='Path to vanilla validation_generations.pkl (question/answer text)')
    parser.add_argument('--metric', type=str, default='vqa_acc',
                       help='Metric name (vqa_acc binarized at metric_threshold for AUROC)')
    parser.add_argument('--metric_threshold', type=float, default=0.5,
                       help='Threshold for binarizing continuous VQA accuracy')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    
    # Similarity measure
    parser.add_argument('--similarity', type=str, default='deberta',
                       choices=['deberta', 'rouge', 'jaccard'],
                       help='Similarity measure for W matrix')
    
    # Scoring mode (only used for deberta similarity)
    parser.add_argument('--score_mode', type=str, default='baseline_relaxed',
                       choices=['baseline_relaxed', 'baseline_strict',
                                'entail_prob', 'entail_over_noncontrad', 'noncontrad_prob'],
                       help='Scoring mode for W matrix (deberta only)')
    parser.add_argument('--combine', type=str, default='min',
                       choices=['min', 'geom_mean', 'mean'],
                       help='Method to combine forward/backward scores (deberta only)')
    
    # Weighting parameters
    parser.add_argument('--weighting_scheme', type=str, default='entropy_confidence',
                       choices=['none', 'exp_consistency', 'inverse_consistency', 'linear_consistency',
                                'entropy_confidence', 'hybrid', 'dominance', 'inverse_cluster_count'],
                       help='Weighting scheme for column weights (use "none" for no weighting)')
    parser.add_argument('--divergence_measure', type=str, default='js',
                       choices=['js', 'kl'],
                       help='Divergence measure for computing weights')
    parser.add_argument('--sim_threshold', type=float, default=0.5,
                       help='Similarity threshold for clustering')
    
    # Subsampling
    parser.add_argument('--subsample_high_t', type=int, default=None,
                       help='Subsample high-T responses (None = use all)')
    parser.add_argument('--subsample_seed', type=int, default=2000,
                       help='Random seed for subsampling')
    
    # Metadata
    parser.add_argument('--model_name', type=str, default='',
                       help='Model name for logging')
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"WEIGHTED SPECTRAL ENERGY - OKVQA")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Entailments: {args.entailments_file}")
    logger.info(f"Similarity: {args.similarity}")
    if args.similarity == "deberta":
        logger.info(f"Score mode: {args.score_mode}, Combine: {args.combine}")
    logger.info(f"Weighting scheme: {args.weighting_scheme}, Divergence: {args.divergence_measure}")
    logger.info(f"{'='*60}\n")
    
    # Load precomputed entailments
    entailments_path = Path(args.entailments_file)
    if not entailments_path.exists():
        logger.error(f"Entailments file not found: {entailments_path}")
        return 1
    
    precomputed_entailments = load_results_npz(entailments_path)
    logger.info(f"Loaded {len(precomputed_entailments)} precomputed questions")
    
    # Load accuracy dict from JSON file (optional)
    accuracy_dict = None
    if args.accuracy_file:
        accuracy_path = Path(args.accuracy_file)
        if accuracy_path.exists():
            accuracy_dict = load_accuracy_json(accuracy_path)
            logger.info(f"Loaded {len(accuracy_dict)} accuracy labels from JSON file")
        else:
            logger.warning(f"Accuracy file not found: {accuracy_path}")
    
    # Compute weighted spectral energy based on similarity measure
    if args.similarity == "deberta":
        graph_results = compute_weighted_spectral_energy_with_precomputed(
            precomputed_entailments,
            args.output_dir,
            score_mode=args.score_mode,
            combine=args.combine,
            weighting_scheme=args.weighting_scheme,
            divergence_measure=args.divergence_measure,
            sim_threshold=args.sim_threshold,
            accuracy_dict=accuracy_dict,
            subsample_high_t=args.subsample_high_t,
            subsample_seed=args.subsample_seed
        )
    else:
        graph_results = compute_weighted_spectral_energy_with_similarity(
            precomputed_entailments,
            args.output_dir,
            args.similarity,
            weighting_scheme=args.weighting_scheme,
            divergence_measure=args.divergence_measure,
            sim_threshold=args.sim_threshold,
            accuracy_dict=accuracy_dict,
            subsample_high_t=args.subsample_high_t,
            subsample_seed=args.subsample_seed
        )
    
    if not graph_results:
        logger.error("No results computed")
        return 1
    
    vanilla_metadata = load_vanilla_example_metadata(args.vanilla_pkl)
    if vanilla_metadata:
        logger.info(f"Loaded question/answer metadata for {len(vanilla_metadata)} examples from vanilla pkl")

    method_name = build_spectral_method_name(
        args.similarity, args.score_mode, args.combine, args.weighting_scheme
    )

    # Compute AUROC
    auroc, per_question_auroc_scores = compute_auroc(
        graph_results, args.output_dir,
        args.score_mode, args.combine, args.weighting_scheme,
        args.similarity if args.similarity != "deberta" else None,
        metric_threshold=args.metric_threshold,
    )

    if args.similarity == "deberta":
        per_example_name = f"per_example_{args.score_mode}_{args.combine}_{args.weighting_scheme}.csv"
    else:
        per_example_name = f"per_example_{args.similarity}_{args.weighting_scheme}.csv"
    save_per_example_spectral_csv(
        graph_results,
        str(Path(args.output_dir) / per_example_name),
        method_name,
        vanilla_metadata,
        per_question_auroc_scores,
        metric_threshold=args.metric_threshold,
    )
    
    # Save summary
    output_path = Path(args.output_dir)
    summary = {
        'model_name': args.model_name,
        'similarity': args.similarity,
        'score_mode': args.score_mode if args.similarity == 'deberta' else 'N/A',
        'combine': args.combine if args.similarity == 'deberta' else 'N/A',
        'weighting_scheme': args.weighting_scheme,
        'divergence_measure': args.divergence_measure,
        'auroc': auroc,
        'num_questions': len(graph_results)
    }
    
    if args.similarity == "deberta":
        csv_name = f"summary_{args.score_mode}_{args.combine}_{args.weighting_scheme}.csv"
    else:
        csv_name = f"summary_{args.similarity}_{args.weighting_scheme}.csv"
    
    pd.DataFrame([summary]).to_csv(output_path / csv_name, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"AUROC: {auroc:.4f}" if auroc else "AUROC: N/A")
    logger.info(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
