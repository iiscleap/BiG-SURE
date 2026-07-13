#!/usr/bin/env python3
"""
WEIGHTED Spectral Energy with ALL scoring modes for REPHRASED runs.

This script computes WEIGHTED spectral energy (SVD-based confidence) using various
similarity measures for the bipartite weight matrix W, with paraphrase-aware
column weighting.

Key difference from unweighted version:
- Each column (high-temp answer) is weighted based on how its paraphrase's
  answer distribution diverges from the global distribution
- W_weighted = W @ D^0.5 where D is diagonal matrix of weights

Similarity Measures:
- deberta: Use precomputed DeBERTa entailment probabilities (default)
- rouge: Use ROUGE-L similarity between response texts
- jaccard: Use Jaccard lexical similarity (hybrid of SequenceMatcher, token, trigram)

Weighting Schemes:
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

Accuracy Metrics:
- squad: Use vanilla accuracy from SNNE (default)
- gemini: Use Gemini-evaluated accuracy from accuracy_results JSON
- is_correct: Use is_correct accuracy from accuracy_results JSON
- is_correct_revised: Use is_correct_revised accuracy from accuracy_results JSON

Weighted Spectral Energy Computation:
1. Build W matrix from similarity measure
2. Compute paraphrase-aware weights for each column
3. Apply weights: W_weighted = W @ D^0.5
4. Compute SVD: W_weighted = U @ diag(S) @ Vh
5. Energy = Frobenius norm of singular values = ||S||_2
6. Confidence = energy / sqrt(m * n)  (normalized to [0, 1])
7. Uncertainty = 1.0 - confidence
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
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from snne.uncertainty.utils.normalization_utils import quantile_power_normalize
from snne.uncertainty.utils.eval_utils import auarc as compute_auarc, aucpr as compute_aucpr
from snne.utils.vanilla_accuracy import load_vanilla_accuracy_dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default accuracy results directory
TASK_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_RESULTS_DIR = str(TASK_ROOT / "outputs" / "accuracy")


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
    ROUGE-L = 2 * LCS / (len(a) + len(b))
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    
    # Use SequenceMatcher ratio as ROUGE-L approximation
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
    """
    Cluster texts by lexical similarity and return cluster-size distribution.
    
    Args:
        texts: list of text strings
        sim_threshold: similarity threshold for clustering
    
    Returns:
        numpy array of cluster sizes (normalized to sum to 1)
    """
    if not texts:
        return np.array([])
    
    assignments, cluster_sizes, n, K = _clusters_connected_components(texts, sim_threshold=sim_threshold, sim_func=sim_func)
    if n == 0 or K == 0:
        return np.array([1.0])  # Single cluster with all items
    
    # Normalize cluster sizes to get distribution
    dist = np.array(cluster_sizes, dtype=float) / float(n)
    return dist


def _jensen_shannon_divergence(P, Q):
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    """
    # Normalize to ensure they are probability distributions
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)
    P = P / (P.sum() + 1e-12)
    Q = Q / (Q.sum() + 1e-12)
    
    # Ensure same length (pad with zeros if needed)
    max_len = max(len(P), len(Q))
    if len(P) < max_len:
        P = np.pad(P, (0, max_len - len(P)), mode='constant')
    if len(Q) < max_len:
        Q = np.pad(Q, (0, max_len - len(Q)), mode='constant')
    
    # M = average distribution
    M = 0.5 * (P + Q)
    
    # Avoid log(0)
    eps = 1e-12
    
    # KL(P||M)
    kl_pm = np.sum(P * np.log((P + eps) / (M + eps)))
    
    # KL(Q||M)
    kl_qm = np.sum(Q * np.log((Q + eps) / (M + eps)))
    
    # JS divergence
    js = 0.5 * kl_pm + 0.5 * kl_qm
    
    return float(js)


def _kl_divergence(P, Q):
    """Compute KL divergence KL(P||Q)."""
    # Normalize
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)
    P = P / (P.sum() + 1e-12)
    Q = Q / (Q.sum() + 1e-12)
    
    # Ensure same length
    max_len = max(len(P), len(Q))
    if len(P) < max_len:
        P = np.pad(P, (0, max_len - len(P)), mode='constant')
    if len(Q) < max_len:
        Q = np.pad(Q, (0, max_len - len(Q)), mode='constant')
        
    eps = 1e-12
    # KL(P||Q) = sum P * log(P/Q)
    kl = np.sum(P * np.log((P + eps) / (Q + eps)))
    return float(kl)


# ==================== UTILITY FUNCTIONS ====================

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_accuracy_from_json(dataset: str, model_name: str, metric: str, 
                            accuracy_dir: str = ACCURACY_RESULTS_DIR):
    """Load accuracy labels from JSON files in accuracy_results directory."""
    filename = f"{dataset}_{model_name}_{metric}_accuracy.json"
    filepath = Path(accuracy_dir) / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Accuracy file not found: {filepath}\n"
            f"Expected file for metric='{metric}', dataset='{dataset}', model='{model_name}'"
        )
    
    logger.info(f"Loading {metric} accuracy from: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', {})
    metadata = data.get('metadata', {})
    
    accuracy_dict = {}
    accuracy_key = f"{metric}_accuracy"
    
    for sample_id, sample_data in samples.items():
        if accuracy_key in sample_data:
            accuracy_dict[str(sample_id)] = int(sample_data[accuracy_key])
        elif 'accuracy' in sample_data:
            accuracy_dict[str(sample_id)] = int(sample_data['accuracy'])
    
    overall_acc = metadata.get('accuracy', sum(accuracy_dict.values()) / len(accuracy_dict) if accuracy_dict else 0.0)
    logger.info(f"Loaded {len(accuracy_dict)} accuracy labels (mean: {overall_acc:.4f})")
    
    return accuracy_dict, overall_acc


def load_rephrased_generations(wandb_run_dir):
    """Load rephrased generations from wandb run directory."""
    p = Path(wandb_run_dir)
    validation_pkl = p if p.is_file() and p.suffix == '.pkl' else p / "files" / "validation_generations.pkl"
    if not validation_pkl.exists():
        raise FileNotFoundError(f"Rephrased generations not found: {validation_pkl}")
    generations = load_pickle(validation_pkl)
    logger.info(f"Loaded {len(generations)} rephrased samples")
    return generations


def organize_by_original_id(generations):
    """Organize rephrased generations by original question ID.

    When loading vanilla generations (no original_id), the full key is used as
    the orig_id so it aligns with the IDs stored in the precomputed entailments.
    The old fallback (split on '_') only worked for plain integer keys and broke
    for TriviaQA's compound string keys like 'bb_1013--62/62_852710.txt#0_0'.
    """
    by_original = defaultdict(list)
    for rephrased_id, data in generations.items():
        if not isinstance(data, dict):
            continue
        orig_id = data.get('original_id')
        if orig_id is None:
            # Use the full key directly; this correctly handles both plain integer
            # vanilla keys (e.g. "123" for SVAMP/NQ) and compound string keys
            # (e.g. "bb_1013--62/62_852710.txt#0_0" for TriviaQA).
            orig_id = str(rephrased_id)
        by_original[str(orig_id)].append({'rephrased_id': rephrased_id, 'data': data})
    return by_original


def _extract_response_text(item):
    """Extract text from various response formats."""
    if item is None:
        return None
    if isinstance(item, (tuple, list)):
        return item[0] if len(item) > 0 else None
    if isinstance(item, dict):
        return item.get('response') or item.get('answer')
    if isinstance(item, str):
        return item
    return None


def extract_all_low_t_from_rephrasings(rephrasings):
    """Extract all low-temperature responses from rephrasings."""
    all_low_t = []
    for reph in rephrasings:
        data = reph['data']
        if isinstance(data.get('low_temp_answers'), (list, tuple)):
            for x in data['low_temp_answers']:
                txt = _extract_response_text(x)
                if txt:
                    all_low_t.append(txt)
    return all_low_t


def extract_all_high_t_with_paraphrase_indices(rephrasings):
    """Extract all high-temperature responses with their paraphrase indices."""
    all_high_t = []
    paraphrase_indices = []
    
    for para_idx, reph in enumerate(rephrasings):
        data = reph['data']
        if isinstance(data.get('responses'), (list, tuple)):
            for resp_tuple in data['responses']:
                txt = _extract_response_text(resp_tuple)
                if txt:
                    all_high_t.append(txt)
                    paraphrase_indices.append(para_idx)
    return all_high_t, paraphrase_indices


def subsample_answers(answers, k, seed=None):
    """Subsample k answers from list."""
    if len(answers) <= k:
        return answers, list(range(len(answers)))
    rnd = random.Random(seed) if seed else random
    indices = list(range(len(answers)))
    if seed:
        sampled_indices = rnd.sample(indices, k)
    else:
        sampled_indices = indices[:k]
    return [answers[i] for i in sampled_indices], sampled_indices


def load_results_npz(filepath):
    """Load precomputed entailment results from compressed numpy archive."""
    data = np.load(filepath, allow_pickle=True)
    question_ids = data['question_ids'].tolist()
    
    results = {}
    for qid in question_ids:
        prefix = f"q_{qid}_"
        results[qid] = {
            'probs_fwd': data[f"{prefix}probs_fwd"],
            'probs_bwd': data[f"{prefix}probs_bwd"],
            'low_texts': data[f"{prefix}low_texts"].tolist(),
            'high_texts': data[f"{prefix}high_texts"].tolist(),
            'm': int(data[f"{prefix}m"][0]),
            'n': int(data[f"{prefix}n"][0])
        }
        if f"{prefix}accuracy" in data:
            results[qid]['accuracy'] = float(data[f"{prefix}accuracy"][0])
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
        weighting_scheme: strategy for computing weights
        divergence_measure: "js" or "kl"
        sim_threshold: threshold for clustering
    
    Returns:
        numpy array of weights (one per high-temp answer)
    """
    n = len(high_texts)
    if n == 0:
        return np.array([])
    
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
            # REWARD consistency: exp(-div)
            w_val = np.exp(-float(div_val))
            
        elif weighting_scheme == "inverse_consistency":
            # REWARD consistency: 1 / (1 + div)
            w_val = 1.0 / (1.0 + float(div_val))
            
        elif weighting_scheme == "linear_consistency":
            # REWARD consistency: max(0, 1 - div)
            w_val = max(0.0, 1.0 - float(div_val))
            
        elif weighting_scheme == "entropy_confidence":
            # REWARD internal confidence: 1 - H(paraphrase)
            H_raw = -np.sum(dist_paraphrase * np.log(dist_paraphrase + 1e-12))
            K_eff = len(dist_paraphrase)
            if K_eff <= 1:
                H_norm = 0.0
            else:
                H_norm = H_raw / (np.log(K_eff) + 1e-12)
            w_val = 1.0 - float(H_norm)

        elif weighting_scheme == "dominance":
            # REWARD single dominant answer: max(prob)
            if len(dist_paraphrase) > 0:
                w_val = float(np.max(dist_paraphrase))
            else:
                w_val = 0.0

        elif weighting_scheme == "inverse_cluster_count":
            # REWARD fewer clusters: 1 / K
            K_eff = len(dist_paraphrase)
            if K_eff > 0:
                w_val = 1.0 / float(K_eff)
            else:
                w_val = 1.0

        elif weighting_scheme == "hybrid":
            # Combine consistency and internal confidence
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
            # Fallback to exp_consistency
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


def compute_weighted_spectral_energy_with_precomputed(generations, precomputed_entailments, output_dir,
                                                       score_mode="baseline_relaxed", combine="min",
                                                       weighting_scheme="entropy_confidence",
                                                       divergence_measure="js", sim_threshold=0.7,
                                                       k_low_t=None, subsample_high_t=None, subsample_seed=0,
                                                       accuracy_dict=None, self_similarity=False, same_input_probes=False):
    """Compute weighted spectral energy using precomputed deberta entailment probabilities."""
    by_original = organize_by_original_id(generations)
    logger.info(f"Found {len(by_original)} original questions")
    logger.info(f"Using precomputed entailments (Score mode: {score_mode}, Combine: {combine})")
    logger.info(f"Weighting scheme: {weighting_scheme}, Divergence: {divergence_measure}")
    if subsample_high_t:
        logger.info(f"Subsampling to {subsample_high_t} high-T from precomputed")
    if self_similarity:
        logger.info("Self-similarity mode: square matrix, uniform column weights (low_low or high_high)")
    
    graph_results = {}
    
    for orig_id in tqdm(sorted(by_original.keys()), desc="Processing"):
        precomp = precomputed_entailments.get(str(orig_id))
        if precomp is None:
            continue
        
        W_full = probs_to_weights_matrix(
            precomp['probs_fwd'],
            precomp['probs_bwd'],
            score_mode,
            combine
        )
        
        if self_similarity:
            # Square self-similarity matrix (low_low or high_high): use directly with uniform weights.
            # No paraphrase structure exists, so column weighting is uniform.
            W = W_full
            weights = np.ones(W.shape[1])
        else:
            rephrasings = by_original[orig_id]
            # Get paraphrase indices for high-temp answers
            all_high_t, paraphrase_indices = extract_all_high_t_with_paraphrase_indices(rephrasings)

            m_precomputed, n_precomputed = W_full.shape
            K_low = int(k_low_t) if k_low_t else m_precomputed
            K_high = int(subsample_high_t) if subsample_high_t else n_precomputed

            # Subsample low_t (rows) deterministic first K_low
            if K_low < m_precomputed and K_low > 0:
                W_current = W_full[:K_low, :]
            else:
                W_current = W_full

            # Subsample high_t (cols) randomly
            if K_high < n_precomputed and K_high > 0:
                rnd = random.Random(2000)
                if same_input_probes and len(paraphrase_indices) >= n_precomputed:
                    indices_by_para = defaultdict(list)
                    for i, p_idx in enumerate(paraphrase_indices[:n_precomputed]):
                        indices_by_para[p_idx].append(i)
                    
                    # Choose a paraphrase index that has at least K_high samples, or just the one with max samples
                    valid_paras = [p for p, idxs in indices_by_para.items() if len(idxs) >= K_high]
                    if valid_paras:
                        chosen_para = valid_paras[0]
                        pool = indices_by_para[chosen_para]
                    else:
                        chosen_para = max(indices_by_para.keys(), key=lambda p: len(indices_by_para[p]))
                        pool = indices_by_para[chosen_para]
                    
                    if len(pool) >= K_high:
                        sampled_indices = sorted(rnd.sample(pool, K_high))
                    else:
                        sampled_indices = sorted(pool)
                else:
                    sampled_indices = sorted(rnd.sample(range(n_precomputed), K_high))
                W = W_current[:, sampled_indices]
                # Subsample high texts and indices
                if len(all_high_t) >= n_precomputed:
                    high_texts_for_weight = [all_high_t[i] for i in sampled_indices]
                    para_indices_sampled = [paraphrase_indices[i] for i in sampled_indices]
                else:
                    # Fallback: subsample precomp['high_texts'] with the same indices
                    precomp_high = precomp.get('high_texts', [])
                    if len(precomp_high) >= n_precomputed:
                        high_texts_for_weight = [precomp_high[i] for i in sampled_indices]
                    else:
                        high_texts_for_weight = precomp_high[:K_high] if len(precomp_high) >= K_high else precomp_high
                    para_indices_sampled = [0] * len(high_texts_for_weight)
            else:
                W = W_current
                if len(all_high_t) >= n_precomputed:
                    high_texts_for_weight = all_high_t
                    para_indices_sampled = paraphrase_indices
                else:
                    high_texts_for_weight = precomp.get('high_texts', [])
                    para_indices_sampled = [0] * len(high_texts_for_weight)

            # Compute paraphrase-aware column weights
            weights = compute_paraphrase_weights(
                high_texts_for_weight, para_indices_sampled,
                weighting_scheme, divergence_measure, sim_threshold
            )

            # Ensure weights match W columns
            if len(weights) != W.shape[1]:
                weights = np.ones(W.shape[1])

        result = compute_weighted_spectral_energy_from_W(W, weights)
        if result is None:
            continue

        confidence, uncertainty, m, n = result

        if accuracy_dict is not None and str(orig_id) in accuracy_dict:
            accuracy = accuracy_dict[str(orig_id)]
        else:
            # Try to get accuracy from precomp or generations
            accuracy = precomp.get('accuracy')

            if accuracy is None and not self_similarity:
                # Check rephrasings (e.g. for translation/rougel) - only in bipartite mode
                reph0_data = by_original[orig_id][0].get('data', {})
                if 'most_likely_answer' in reph0_data:
                    accuracy = reph0_data['most_likely_answer'].get('accuracy')
                if accuracy is None:
                    accuracy = reph0_data.get('accuracy')
                if accuracy is not None:
                    try:
                        accuracy = float(accuracy)
                    except (ValueError, TypeError):
                        pass

        graph_results[str(orig_id)] = {
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


def compute_weighted_spectral_energy_with_similarity(generations, output_dir, similarity,
                                                      weighting_scheme="entropy_confidence",
                                                      divergence_measure="js", sim_threshold=0.7,
                                                      k_low_t=3, subsample_high_t=10, subsample_seed=2000,
                                                      accuracy_dict=None, non_rephrased=False, same_input_probes=False):
    """Compute weighted spectral energy using text-based similarity (jaccard or rouge)."""
    by_original = organize_by_original_id(generations)
    logger.info(f"Found {len(by_original)} original questions")
    logger.info(f"Using {similarity} similarity")
    logger.info(f"Weighting scheme: {weighting_scheme}, Divergence: {divergence_measure}")
    
    graph_results = {}
    
    for orig_id in tqdm(sorted(by_original.keys()), desc="Processing"):
        rephrasings = by_original[orig_id]
        
        all_low_t = extract_all_low_t_from_rephrasings(rephrasings)
        all_high_t, paraphrase_indices = extract_all_high_t_with_paraphrase_indices(rephrasings)
        
        # Subsample
        sampled_low_t, _ = subsample_answers(all_low_t, k_low_t, seed=subsample_seed)
        
        if same_input_probes and len(all_high_t) > 0:
            rnd = random.Random(subsample_seed)
            indices_by_para = defaultdict(list)
            for i, p_idx in enumerate(paraphrase_indices):
                indices_by_para[p_idx].append(i)
            valid_paras = [p for p, idxs in indices_by_para.items() if len(idxs) >= subsample_high_t]
            if valid_paras:
                chosen_para = valid_paras[0]
                pool = indices_by_para[chosen_para]
            else:
                chosen_para = max(indices_by_para.keys(), key=lambda p: len(indices_by_para[p]))
                pool = indices_by_para[chosen_para]
            if len(pool) >= subsample_high_t:
                sampled_high_indices = sorted(rnd.sample(pool, subsample_high_t))
            else:
                sampled_high_indices = sorted(pool)
            sampled_high_t = [all_high_t[i] for i in sampled_high_indices]
        else:
            sampled_high_t, sampled_high_indices = subsample_answers(all_high_t, subsample_high_t, seed=subsample_seed)
        sampled_para_indices = [paraphrase_indices[i] for i in sampled_high_indices]
        
        if len(sampled_low_t) == 0 or len(sampled_high_t) == 0:
            continue
        
        # Compute similarity matrix
        W = compute_similarity_matrix(sampled_low_t, sampled_high_t, similarity)
        
        # Compute weights
        if non_rephrased:
            weights = np.ones(W.shape[1])
        else:
            weights = compute_paraphrase_weights(
                sampled_high_t, sampled_para_indices,
                weighting_scheme, divergence_measure, sim_threshold
            )
        
        result = compute_weighted_spectral_energy_from_W(W, weights)
        if result is None:
            continue
        
        confidence, uncertainty, m, n = result
        
        if accuracy_dict is not None and str(orig_id) in accuracy_dict:
            accuracy = accuracy_dict[str(orig_id)]
        else:
            # Fallback for translation/rougel
            accuracy = None
            if rephrasings:
                first_data = rephrasings[0].get('data', {})
                if isinstance(first_data, dict):
                    if 'most_likely_answer' in first_data and isinstance(first_data['most_likely_answer'], dict):
                        accuracy = first_data['most_likely_answer'].get('accuracy')
                    if accuracy is None:
                        accuracy = first_data.get('accuracy')
                
                if accuracy is not None:
                     try:
                        accuracy = float(accuracy)
                     except (ValueError, TypeError):
                        pass
        
        graph_results[str(orig_id)] = {
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


def compute_metrics(graph_results, output_dir, score_mode, combine, weighting_scheme, similarity=None, dataset=None):
    """Compute AUROC, AUARC, and AUCPR from spectral energy results."""
    question_ids = sorted(graph_results.keys())
    
    valid_qids = [qid for qid in question_ids if graph_results[qid].get('accuracy') is not None]
    
    if not valid_qids:
        logger.warning("No valid accuracies found for metric computation")
        return None, None, None
    
    confidences = np.array([graph_results[qid]['confidence'] for qid in valid_qids])
    accuracies = np.array([graph_results[qid]['accuracy'] for qid in valid_qids])
    
    eps = 1e-10
    uncertainties_raw = -np.log(confidences + eps)
    uncertainties_normalized = quantile_power_normalize(uncertainties_raw)
    
    # AUROC
    auroc = None
    if dataset != "fr-en":
        if len(accuracies) > 0 and len(set(accuracies)) > 1:
            auroc = roc_auc_score(accuracies, uncertainties_normalized)
    
    # AUARC: expects (y_score, y_true) where y_score=uncertainty, y_true=accuracy
    auarc_val = compute_auarc(uncertainties_normalized.tolist(), accuracies.tolist())
    
    # AUCPR (PRR): expects (y_score, y_true) where y_score=uncertainty, y_true=accuracy
    aucpr_val = compute_aucpr(uncertainties_normalized.tolist(), accuracies.tolist())
    
    sim_str = f", Similarity: {similarity}" if similarity else ""
    auroc_str = f"{auroc:.4f}" if auroc else "N/A"
    auarc_str = f"{auarc_val:.4f}" if auarc_val is not None else "N/A"
    aucpr_str = f"{aucpr_val:.4f}" if aucpr_val is not None else "N/A"
    logger.info(f"Score: {score_mode}, Combine: {combine}, Weighting: {weighting_scheme}{sim_str} -> AUROC: {auroc_str}, AUARC: {auarc_str}, AUCPR: {aucpr_str}")
    
    return auroc, auarc_val, aucpr_val


def main():
    parser = argparse.ArgumentParser(description='Weighted Spectral Energy - All Modes (Rephrased)')
    
    parser.add_argument('--run_dir', type=str, required=True,
                       help='WandB run directory with rephrased generations')
    parser.add_argument('--wandb_base_dir', type=str, default='./malay/uncertainty/wandb',
                       help='Base directory for WandB runs')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    
    # Similarity measure
    parser.add_argument('--similarity', type=str, default='deberta',
                       choices=['deberta', 'rouge', 'jaccard'],
                       help='Similarity measure for W matrix')
    
    # Precomputed entailments (required for deberta similarity)
    parser.add_argument('--entailments_file', type=str, default=None,
                       help='Path to precomputed entailments .npz file (required for deberta)')
    
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
                       choices=['exp_consistency', 'inverse_consistency', 'linear_consistency',
                                'entropy_confidence', 'hybrid', 'dominance', 'inverse_cluster_count'],
                       help='Weighting scheme for paraphrase-aware column weights')
    parser.add_argument('--divergence_measure', type=str, default='js',
                       choices=['js', 'kl'],
                       help='Divergence measure for computing weights')
    parser.add_argument('--sim_threshold', type=float, default=0.5,
                       help='Similarity threshold for clustering')
    
    # Subsampling
    parser.add_argument('--k_low_t', type=int, default=3,
                       help='Number of low-T responses (for rouge/jaccard)')
    parser.add_argument('--subsample_high_t', type=int, default=None,
                       help='Subsample high-T responses (None = use all)')
    parser.add_argument('--subsample_seed', type=int, default=2000,
                       help='Random seed for subsampling')
    parser.add_argument('--self_similarity', action='store_true',
                       help='Treat precomputed entailments as self-similarity (square matrix) and use uniform column weights')
    parser.add_argument('--non_rephrased', action='store_true',
                       help='Run on non-rephrased npz files, using uniform column weights since there are no paraphrases')
    parser.add_argument('--same_input_probes', action='store_true',
                       help='If set, high-temp samples are chosen entirely from a single paraphrase to test same-input vs mixed-input probes')
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name for accuracy lookup')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name for accuracy lookup')
    
    # Accuracy metric selection
    parser.add_argument('--metric', type=str, default='squad',
                       choices=['squad', 'gemini', 'is_correct', 'is_correct_revised', 'rougel'],
                       help='Accuracy metric to use')
    parser.add_argument('--accuracy_dir', type=str, default=ACCURACY_RESULTS_DIR,
                       help='Directory containing accuracy JSON files')
    parser.add_argument('--seed', type=int, default=None,
                       help='Seed number for vanilla accuracy lookup (appends _seed{N} to model name)')
    parser.add_argument('--vanilla_run_dir', type=str, default=None,
                       help='Explicit vanilla run directory (bypasses log_summary lookup)')
    
    args = parser.parse_args()
    
    run_path = Path(args.run_dir)
    if not run_path.is_absolute():
        run_path = Path(args.wandb_base_dir) / args.run_dir
    
    logger.info(f"\n{'='*60}")
    logger.info(f"WEIGHTED SPECTRAL ENERGY ALL MODES - REPHRASED")
    logger.info(f"Dataset: {args.dataset}, Model: {args.model_name}")
    logger.info(f"Similarity: {args.similarity}")
    if args.similarity == "deberta":
        logger.info(f"Score mode: {args.score_mode}, Combine: {args.combine}")
    logger.info(f"Weighting scheme: {args.weighting_scheme}, Divergence: {args.divergence_measure}")
    logger.info(f"Accuracy metric: {args.metric}")
    logger.info(f"{'='*60}\n")
    
    # Load accuracy based on metric
    if args.metric == 'squad':
        # For seed-based runs, the log_summary model column has _seed{N} suffix
        vanilla_model_query = args.model_name
        if args.seed is not None:
            vanilla_model_query = f"{args.model_name}_seed{args.seed}"
        accuracy_dict, overall_acc = load_vanilla_accuracy_dict(
            args.dataset, vanilla_model_query,
            vanilla_run_dir=args.vanilla_run_dir,
            wandb_base_dir=args.wandb_base_dir
        )
        logger.info(f"Loaded SQUAD (vanilla) accuracy for {len(accuracy_dict)} questions (mean: {overall_acc:.4f})")
    elif args.metric == 'rougel':
        # For translation, accuracy is stored within the generations data
        accuracy_dict = {}
        overall_acc = 0.0
        logger.info("Metric is rougel, will extract accuracy from generations data.")
    else:
        accuracy_dict, overall_acc = load_accuracy_from_json(
            args.dataset, args.model_name, args.metric,
            accuracy_dir=args.accuracy_dir
        )
    
    # Load rephrased generations
    generations = load_rephrased_generations(run_path)
    
    # Compute weighted spectral energy based on similarity measure
    if args.similarity == "deberta":
        # Require precomputed entailments for deberta
        if not args.entailments_file:
            raise ValueError("--entailments_file is required for deberta similarity")
        
        entailments_file = Path(args.entailments_file)
        if not entailments_file.exists():
            raise FileNotFoundError(f"Precomputed entailments not found: {entailments_file}")
        
        logger.info(f"Loading precomputed entailments: {entailments_file}")
        precomputed_entailments = load_results_npz(entailments_file)
        logger.info(f"Loaded {len(precomputed_entailments)} precomputed questions")
        
        graph_results = compute_weighted_spectral_energy_with_precomputed(
            generations, precomputed_entailments, args.output_dir,
            score_mode=args.score_mode, combine=args.combine,
            weighting_scheme=args.weighting_scheme,
            divergence_measure=args.divergence_measure,
            sim_threshold=args.sim_threshold,
            k_low_t=args.k_low_t, subsample_high_t=args.subsample_high_t, subsample_seed=args.subsample_seed,
            accuracy_dict=accuracy_dict,
            self_similarity=args.self_similarity or args.non_rephrased,
            same_input_probes=args.same_input_probes
        )
    else:
        # Use text-based similarity (rouge or jaccard)
        graph_results = compute_weighted_spectral_energy_with_similarity(
            generations, args.output_dir, args.similarity,
            weighting_scheme=args.weighting_scheme,
            divergence_measure=args.divergence_measure,
            sim_threshold=args.sim_threshold,
            k_low_t=args.k_low_t, 
            subsample_high_t=args.subsample_high_t if args.subsample_high_t else 50,
            subsample_seed=args.subsample_seed,
            accuracy_dict=accuracy_dict,
            non_rephrased=args.non_rephrased,
            same_input_probes=args.same_input_probes
        )
    
    if not graph_results:
        logger.error("No results computed")
        return 1
    
    # Compute AUROC, AUARC, AUCPR
    auroc, auarc_val, aucpr_val = compute_metrics(graph_results, args.output_dir, args.score_mode, args.combine, 
                                                   args.weighting_scheme, args.similarity, args.dataset)
    
    # Save summary
    output_path = Path(args.output_dir)
    summary = {
        'dataset': args.dataset, 
        'model_name': args.model_name,
        'similarity': args.similarity,
        'score_mode': args.score_mode if args.similarity == 'deberta' else 'N/A',
        'combine': args.combine if args.similarity == 'deberta' else 'N/A',
        'weighting_scheme': args.weighting_scheme,
        'divergence_measure': args.divergence_measure,
        'metric': args.metric,
        'auroc': auroc,
        'auarc': auarc_val,
        'prr': aucpr_val,
        'n_questions': len(graph_results)
    }
    
    if args.similarity == "deberta":
        csv_name = f"summary_{args.score_mode}_{args.combine}_{args.weighting_scheme}.csv"
    else:
        csv_name = f"summary_{args.similarity}_{args.weighting_scheme}.csv"
    
    pd.DataFrame([summary]).to_csv(output_path / csv_name, index=False)
    
    logger.info(f"Results saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
