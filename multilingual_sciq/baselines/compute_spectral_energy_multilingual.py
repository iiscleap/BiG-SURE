#!/usr/bin/env python3
"""
WEIGHTED Spectral Energy with ALL scoring modes for MULTILINGUAL rephrased runs.

This script computes WEIGHTED spectral energy (SVD-based confidence) using various
similarity measures for the bipartite weight matrix W, with paraphrase-aware
column weighting.

Key features:
- Paraphrase-aware column weighting based on divergence from global distribution
- W_weighted = W @ D^0.5 where D is diagonal matrix of weights
- Multiple weighting schemes (entropy_confidence, exp_consistency, etc.)
- Support for both PREM and Gemini accuracy metrics
- Clustering-based distribution computation for divergence measurement

Usage:
    python compute_spectral_energy_multilingual.py \
        --vanilla_file data/triviaqa/vanilla/generate.json \
        --sampling_file data/triviaqa/sampling/generate.json \
        --entailments_file entailments/triviaqa_sampling.npz \
        --output_dir results/triviaqa \
        --score_mode baseline_relaxed \
        --combine min \
        --weighting_scheme entropy_confidence \
        --metric prem
"""

import argparse
import json
import logging
import numpy as np
import math
import pandas as pd
import re
import random
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multilingual_utils import LANGUAGES, calculate_prem_accuracy, load_json_data

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

def load_results_npz(filepath):
    """Load precomputed entailments."""
    data = np.load(filepath, allow_pickle=True)
    question_ids = data['question_ids'].tolist()
    languages = data['languages'].tolist()
    
    results = {}
    for qid, lang in zip(question_ids, languages):
        prefix = f"q{qid}_lang{lang}"
        results[(qid, lang)] = {
            'probs_fwd': data[f"{prefix}_probs_fwd"],
            'probs_bwd': data[f"{prefix}_probs_bwd"],
            'low_texts': data[f"{prefix}_low_texts"].tolist(),
            'high_texts': data[f"{prefix}_high_texts"].tolist()
        }
    
    return results


# ==================== WEIGHT MATRIX COMPUTATION ====================

def probs_to_weights_matrix(probs_fwd, probs_bwd, score_mode, combine):
    """
    Convert entailment probabilities to similarity weights.
    EXACT replica of SNNE implementation.
    
    Args:
        probs_fwd: (m, n, 3) - P(class | low[i] -> high[j])
        probs_bwd: (m, n, 3) or (n, m, 3) - P(class | high[j] -> low[i])
        score_mode: scoring method
        combine: aggregation for directional scores (min/geom_mean/mean)
    
    Returns:
        W: (m, n) weight matrix
    """
    eps = 1e-10
    m, n = probs_fwd.shape[:2]
    
    def probs_to_weight(probs, mode):
        """Convert probability matrix to weight matrix for a single direction."""
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
    
    # Handle old format: probs_bwd might be (n, m, 3) instead of (m, n, 3)
    W_backward_raw = probs_to_weight(probs_bwd, score_mode)
    if W_backward_raw.shape != (m, n):
        # Old format detected, transpose
        W_backward = W_backward_raw.T
    else:
        W_backward = W_backward_raw
    
    if combine == "min":
        W = np.minimum(W_forward, W_backward)
    elif combine == "geom_mean":
        W = np.sqrt(W_forward * W_backward + eps)
    elif combine == "mean":
        W = 0.5 * (W_forward + W_backward)
    else:
        raise ValueError(f"Unknown combine mode: {combine}")
    
    return W.astype(np.float32)


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


# ==================== WEIGHT MATRIX COMPUTATION ====================
    """
    Convert entailment probabilities to similarity weights.
    EXACT replica of SNNE implementation.
    
    Args:
        probs_fwd: (m, n, 3) - P(class | low[i] -> high[j])
        probs_bwd: (m, n, 3) or (n, m, 3) - P(class | high[j] -> low[i])
        score_mode: scoring method
        combine: aggregation for directional scores (min/geom_mean/mean)
    
    Returns:
        W: (m, n) weight matrix
    """
    eps = 1e-10
    m, n = probs_fwd.shape[:2]
    
    def probs_to_weight(probs, mode):
        """Convert probability matrix to weight matrix for a single direction."""
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
    
    # Handle old format: probs_bwd might be (n, m, 3) instead of (m, n, 3)
    W_backward_raw = probs_to_weight(probs_bwd, score_mode)
    if W_backward_raw.shape != (m, n):
        # Old format detected, transpose
        W_backward = W_backward_raw.T
    else:
        W_backward = W_backward_raw
    
    if combine == "min":
        W = np.minimum(W_forward, W_backward)
    elif combine == "geom_mean":
        W = np.sqrt(W_forward * W_backward + eps)
    elif combine == "mean":
        W = 0.5 * (W_forward + W_backward)
    else:
        raise ValueError(f"Unknown combine mode: {combine}")
    
    return W.astype(np.float32)


def compute_weighted_spectral_energy_from_W(W, weights):
    """
    Compute weighted spectral energy from weight matrix W and column weights.
    
    Args:
        W: (m, n) weight matrix
        weights: (n,) column weights
    
    Returns:
        confidence: Normalized confidence score
        uncertainty: 1 - confidence
        m: number of rows
        n: number of columns
    """
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


def compute_spectral_energy(W):
    """
    Compute spectral energy from weight matrix (unweighted version).
    EXACT replica of SNNE implementation.
    
    Args:
        W: (m, n) weight matrix
    
    Returns:
        confidence: Normalized confidence score
        uncertainty: 1 - confidence
        m: number of rows
        n: number of columns
    """
    m, n = W.shape
    
    if m == 0 or n == 0:
        return None
    
    try:
        _, S, _ = np.linalg.svd(W, full_matrices=False)
    except np.linalg.LinAlgError:
        S = np.zeros(min(m, n))
    
    energy = np.linalg.norm(S)
    max_energy = math.sqrt(m * n)
    eps = 1e-10
    
    conf = energy / (max_energy + eps)
    unc = 1.0 - conf
    
    return float(conf), float(unc), int(m), int(n)


def quantile_power_normalize(x, gamma=0.5, clip=(1, 99)):
    """
    Quantile power normalize with inversion.
    EXACT replica of SNNE implementation.
    
    This inverts the values first (1/x) so that higher uncertainty maps to lower rank,
    then applies quantile normalization with power transform.
    
    Args:
        x: Input array of uncertainty values
        gamma: Power transform exponent (default: 0.5)
        clip: Percentile range for clipping (default: (1, 99))
    
    Returns:
        Normalized values in [0, 1] range
    """
    x = np.asarray(x, dtype=float)
    epsilon = 1e-9
    
    # Inversion: convert uncertainty to confidence-like score for ranking
    x = 1.0 / (x + epsilon)
    
    if clip is not None and len(x) > 0:
        lo, hi = np.percentile(x, clip)
        x = np.clip(x, lo, hi)
    
    # Get ranks (ties broken arbitrarily)
    ranks = np.argsort(np.argsort(x)) + 1
    
    # Convert to uniform [0, 1] with power transform
    u = ranks / (len(x) + 1.0)
    return u ** gamma


def compute_multilingual_spectral_energy(vanilla_data, sampling_data, precomputed_entailments,
                                        score_mode='baseline_relaxed', combine='min',
                                        weighting_scheme='entropy_confidence', 
                                        divergence_measure='js', sim_threshold=0.7,
                                        subsample_high_t=None, subsample_seed=2000,
                                        metric='prem'):
    """
    Compute WEIGHTED spectral energy for all (question_id, language) pairs.
    Uses paraphrase-aware column weighting based on divergence from global distribution.
    """
    results = {}
    
    # Build vanilla lookup for accuracy
    vanilla_lookup = {}
    for item in vanilla_data:
        qid = item['question_id']
        vanilla_lookup[str(qid)] = item
    
    # Build sampling lookup to get paraphrase structure
    sampling_lookup = defaultdict(lambda: defaultdict(list))
    for item in sampling_data:
        qid = item['question_id']
        for lang in LANGUAGES:
            if lang in item.get('output', {}):
                outputs = item['output'][lang]
                sampling_lookup[qid][lang] = outputs
    
    logger.info(f"Computing WEIGHTED spectral energy for {len(precomputed_entailments)} (question, language) pairs")
    logger.info(f"Loaded {len(vanilla_lookup)} vanilla questions for accuracy computation")
    logger.info(f"Sample vanilla question IDs (first 5): {list(vanilla_lookup.keys())[:5]}")
    logger.info(f"Sample entailment question IDs (first 5): {[str(qid) for qid, _ in list(precomputed_entailments.keys())[:5]]}")
    logger.info(f"Weighting scheme: {weighting_scheme}, Divergence: {divergence_measure}, Sim threshold: {sim_threshold}")
    if subsample_high_t:
        logger.info(f"Subsampling to {subsample_high_t} high-T columns with seed {subsample_seed}")
    
    # Debug: Count accuracy sources
    accuracy_found = 0
    accuracy_missing = 0
    
    for (qid, lang), entail_data in tqdm(precomputed_entailments.items(), desc="Computing weighted spectral energy"):
        probs_fwd = entail_data['probs_fwd']
        probs_bwd = entail_data['probs_bwd']
        high_texts = entail_data.get('high_texts', [])
        
        # Extract base question ID (remove rephrasing suffix like _r0, _r1, etc.)
        base_qid = str(qid).split('_r')[0] if '_r' in str(qid) else str(qid)
        
        # Get paraphrase structure from sampling data
        # For multilingual: we have 5 paraphrases × 10 samples each = 50 high-temp answers
        # Paraphrase indices: [0,0,...,0 (10 times), 1,1,...,1 (10 times), ...]
        num_paraphrases = 5
        
        # Determine samples per paraphrase based on matrix dimensions
        n_high = probs_fwd.shape[1]  # Number of high-T columns in entailments
        if n_high >= num_paraphrases:
            samples_per_paraphrase = n_high // num_paraphrases
        else:
            samples_per_paraphrase = 1
            num_paraphrases = n_high
        
        paraphrase_indices = []
        for para_idx in range(num_paraphrases):
            paraphrase_indices.extend([para_idx] * samples_per_paraphrase)
        # Handle remainder
        if len(paraphrase_indices) < n_high:
            remainder = n_high - len(paraphrase_indices)
            paraphrase_indices.extend([num_paraphrases - 1] * remainder)
        paraphrase_indices = paraphrase_indices[:n_high]
        
        # Subsample if needed (sample uniformly from all columns, preserve paraphrase tracking)
        if subsample_high_t and probs_fwd.shape[1] > subsample_high_t:
            rng = np.random.RandomState(subsample_seed)
            indices = sorted(rng.choice(probs_fwd.shape[1], subsample_high_t, replace=False))
            probs_fwd = probs_fwd[:, indices, :]
            probs_bwd = probs_bwd[:, indices, :]
            high_texts = [high_texts[i] for i in indices] if len(high_texts) == n_high else high_texts[:subsample_high_t]
            paraphrase_indices = [paraphrase_indices[i] for i in indices]
        else:
            high_texts = high_texts[:n_high] if len(high_texts) > n_high else high_texts
        
        # Compute weight matrix
        W = probs_to_weights_matrix(probs_fwd, probs_bwd, score_mode, combine)
        
        # Compute paraphrase weights
        if len(high_texts) > 0 and len(high_texts) == W.shape[1]:
            weights = compute_paraphrase_weights(
                high_texts, paraphrase_indices,
                weighting_scheme, divergence_measure, sim_threshold
            )
        else:
            # Fallback to uniform weights if texts don't match
            weights = np.ones(W.shape[1])
        
        # Ensure weights match W columns
        if len(weights) != W.shape[1]:
            weights = np.ones(W.shape[1])
        
        # Compute weighted spectral energy
        result = compute_weighted_spectral_energy_from_W(W, weights)
        if result is None:
            continue
        
        conf, unc, m, n = result
        
        # Get accuracy
        accuracy = None
        if metric in ('claude', 'gemini'):
            # Use precomputed accuracy labels from vanilla_data
            if base_qid in vanilla_lookup:
                vanilla_item = vanilla_lookup[base_qid]
                if 'accuracy' in vanilla_item and lang in vanilla_item['accuracy']:
                    accuracy = vanilla_item['accuracy'][lang]
        elif base_qid in vanilla_lookup:
            # Use PREM accuracy
            vanilla_item = vanilla_lookup[base_qid]
            outputs = vanilla_item.get('output', {}).get(lang, [])
            if outputs and len(outputs) > 0:
                # Extract greedy answer (first output is temp=0)
                greedy_raw = outputs[0]
                # Handle different output formats
                if isinstance(greedy_raw, str):
                    greedy_answer = greedy_raw
                elif isinstance(greedy_raw, dict):
                    greedy_answer = greedy_raw.get('response', greedy_raw.get('answer', greedy_raw.get('text', str(greedy_raw))))
                elif isinstance(greedy_raw, (list, tuple)) and len(greedy_raw) > 0:
                    greedy_answer = str(greedy_raw[0])
                else:
                    greedy_answer = str(greedy_raw)
                
                # answer is a dict with language keys, extract the one for current language
                answer_dict = vanilla_item.get('answer', {})
                if isinstance(answer_dict, dict):
                    ground_truth = answer_dict.get(lang, '')
                else:
                    ground_truth = str(answer_dict) if answer_dict else ''
                
                if ground_truth and greedy_answer:
                    accuracy = calculate_prem_accuracy(greedy_answer, ground_truth)
                elif accuracy_found == 0 and accuracy_missing < 5:
                    # Log details for first few failures
                    logger.warning(f"Failed to compute accuracy for qid={qid}, base_qid={base_qid}, lang={lang}")
                    logger.warning(f"  greedy_raw type: {type(greedy_raw)}, greedy_answer: '{greedy_answer[:50] if greedy_answer else None}'")
                    logger.warning(f"  ground_truth: '{ground_truth[:50] if ground_truth else None}'")
        
        if accuracy is not None:
            accuracy_found += 1
        else:
            accuracy_missing += 1
        
        results[(qid, lang)] = {
            'confidence': conf,
            'uncertainty': unc,
            'm': m,
            'n': n,
            'accuracy': accuracy,
            'W_mean': float(W.mean()),
            'weights_mean': float(weights.mean())
        }
    
    logger.info(f"Accuracy found: {accuracy_found}/{len(results)} ({100*accuracy_found/len(results):.1f}%)")
    if accuracy_missing > 0:
        logger.warning(f"Accuracy missing for {accuracy_missing} samples")
    
    return results


def compute_auroc(results):
    """
    Compute AUROC for each language separately.
    Aggregates across rephrasings by base question ID (like SNNE).
    """
    from sklearn.metrics import roc_auc_score
    
    # First, aggregate by (base_qid, language) across all rephrasings
    # Group results by base question ID and language
    by_base_qid_lang = defaultdict(lambda: {'uncertainties': [], 'confidence': None, 'accuracy': None})
    
    for (qid, lang), data in results.items():
        # Extract base question ID
        base_qid = str(qid).split('_r')[0] if '_r' in str(qid) else str(qid)
        key = (base_qid, lang)
        
        if data['accuracy'] is not None:
            by_base_qid_lang[key]['uncertainties'].append(data['uncertainty'])
            by_base_qid_lang[key]['accuracy'] = data['accuracy']  # Same for all rephrasings
    
    # Aggregate uncertainties across rephrasings (use mean)
    aggregated_results = {}
    for (base_qid, lang), data in by_base_qid_lang.items():
        if data['uncertainties'] and data['accuracy'] is not None:
            aggregated_results[(base_qid, lang)] = {
                'uncertainty': np.mean(data['uncertainties']),
                'accuracy': data['accuracy']
            }
    
    # Group by language for AUROC computation
    by_lang = defaultdict(lambda: {'uncertainties': [], 'accuracies': []})
    
    for (base_qid, lang), data in aggregated_results.items():
        by_lang[lang]['uncertainties'].append(data['uncertainty'])
        by_lang[lang]['accuracies'].append(data['accuracy'])
    
    logger.info(f"\nAccuracy statistics by language (aggregated by base question):")
    for lang in LANGUAGES:
        if lang in by_lang:
            n_samples = len(by_lang[lang]['accuracies'])
            n_correct = sum(by_lang[lang]['accuracies'])
            logger.info(f"  {lang}: {n_samples} base questions, {n_correct} correct ({100*n_correct/n_samples:.1f}%)")
        else:
            logger.info(f"  {lang}: 0 samples")
    
    aurocs = {}
    eps = 1e-10
    
    for lang in LANGUAGES:
        if lang not in by_lang or len(by_lang[lang]['uncertainties']) == 0:
            logger.warning(f"  {lang}: No samples with accuracy, skipping AUROC")
            continue
        
        uncertainties = np.array(by_lang[lang]['uncertainties'])
        accuracies = np.array(by_lang[lang]['accuracies'])
        
        if len(set(accuracies)) < 2:
            logger.warning(f"  {lang}: Only one class present (all {accuracies[0]}), AUROC = -1.0")
            aurocs[lang] = -1.0
            continue
        
        # Apply quantile power normalization (with inversion built-in)
        uncertainties_normalized = quantile_power_normalize(uncertainties)
        
        # Compute AUROC
        aurocs[lang] = roc_auc_score(accuracies, uncertainties_normalized)
        logger.info(f"  {lang}: AUROC = {aurocs[lang]:.4f}")
    
    return aurocs


def save_results(results, aurocs, output_dir, dataset, score_mode, combine, 
                weighting_scheme, divergence_measure):
    """Save results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Count base questions per language for summary
    base_questions_per_lang = defaultdict(set)
    for (qid, lang), data in results.items():
        base_qid = str(qid).split('_r')[0] if '_r' in str(qid) else str(qid)
        base_questions_per_lang[lang].add(base_qid)
    
    n_base_questions = {lang: len(base_questions_per_lang[lang]) for lang in base_questions_per_lang}
    
    # Create detailed CSV
    rows = []
    for (qid, lang), data in results.items():
        rows.append({
            'question_id': qid,
            'language': lang,
            'confidence': data['confidence'],
            'uncertainty': data['uncertainty'],
            'm': data['m'],
            'n': data['n'],
            'accuracy': data['accuracy'],
            'W_mean': data.get('W_mean', 0.0),
            'weights_mean': data.get('weights_mean', 1.0)
        })
    
    df = pd.DataFrame(rows)
    csv_path = f"{output_dir}/results_{dataset}_{score_mode}_{combine}_{weighting_scheme}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved detailed results to {csv_path}")
    
    # Create summary
    summary = {
        'dataset': dataset,
        'score_mode': score_mode,
        'combine': combine,
        'weighting_scheme': weighting_scheme,
        'divergence_measure': divergence_measure,
        'total_rephrasing_language_pairs': len(results),
        'base_questions_per_language': n_base_questions,
        'auroc_by_language': aurocs,
        'mean_auroc': np.mean([v for v in aurocs.values() if v >= 0]) if aurocs else 0
    }
    
    summary_path = f"{output_dir}/summary_{dataset}_{score_mode}_{combine}_{weighting_scheme}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Results Summary - {dataset}")
    logger.info(f"{'='*60}")
    logger.info(f"Score mode: {score_mode}")
    logger.info(f"Combine: {combine}")
    logger.info(f"Weighting scheme: {weighting_scheme}")
    logger.info(f"Divergence: {divergence_measure}")
    logger.info(f"\nBase questions per language (for AUROC):")
    for lang in LANGUAGES:
        if lang in n_base_questions:
            logger.info(f"  {lang}: {n_base_questions[lang]} base questions")
    logger.info(f"\nAUROC by language:")
    for lang in LANGUAGES:
        if lang in aurocs:
            logger.info(f"  {lang}: {aurocs[lang]:.4f}")
    logger.info(f"\nMean AUROC: {summary['mean_auroc']:.4f}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Compute WEIGHTED multilingual spectral energy")
    parser.add_argument('--vanilla_file', type=str, required=True, help='Vanilla JSON file')
    parser.add_argument('--sampling_file', type=str, required=True, help='Sampling JSON file')
    parser.add_argument('--entailments_file', type=str, required=True, help='Precomputed entailments .npz')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--dataset', type=str, default='dataset', help='Dataset name for output files')
    
    # Scoring configuration
    parser.add_argument('--score_mode', type=str, default='baseline_relaxed',
                       choices=['baseline_relaxed', 'baseline_strict', 'entail_prob',
                               'entail_over_noncontrad', 'noncontrad_prob'])
    parser.add_argument('--combine', type=str, default='min',
                       choices=['min', 'geom_mean', 'mean'])
    
    # Weighting configuration
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
    parser.add_argument('--subsample_high_t', type=int, default=10,
                       help='Subsample to this many high-T columns (default: 10)')
    parser.add_argument('--subsample_seed', type=int, default=2000,
                       help='Random seed for subsampling (default: 2000)')
    
    # Accuracy metric
    parser.add_argument('--metric', type=str, default='prem',
                       choices=['prem', 'claude', 'gemini'],
                       help='Accuracy metric to use (prem, claude, or gemini)')
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"WEIGHTED MULTILINGUAL SPECTRAL ENERGY")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Score mode: {args.score_mode}, Combine: {args.combine}")
    logger.info(f"Weighting scheme: {args.weighting_scheme}")
    logger.info(f"Divergence: {args.divergence_measure}, Sim threshold: {args.sim_threshold}")
    logger.info(f"Accuracy metric: {args.metric}")
    logger.info(f"{'='*60}\n")
    
    # Load data
    logger.info(f"Loading vanilla data from {args.vanilla_file}")
    vanilla_data = load_json_data(args.vanilla_file)
    
    logger.info(f"Loading sampling data from {args.sampling_file}")
    sampling_data = load_json_data(args.sampling_file)
    
    logger.info(f"Loading precomputed entailments from {args.entailments_file}")
    precomputed_entailments = load_results_npz(args.entailments_file)
    
    # Compute weighted spectral energy
    results = compute_multilingual_spectral_energy(
        vanilla_data, sampling_data, precomputed_entailments,
        score_mode=args.score_mode,
        combine=args.combine,
        weighting_scheme=args.weighting_scheme,
        divergence_measure=args.divergence_measure,
        sim_threshold=args.sim_threshold,
        subsample_high_t=args.subsample_high_t,
        subsample_seed=args.subsample_seed,
        metric=args.metric
    )
    
    # Compute AUROC
    aurocs = compute_auroc(results)
    
    # Save results
    save_results(results, aurocs, args.output_dir, args.dataset, 
                args.score_mode, args.combine, args.weighting_scheme,
                args.divergence_measure)


if __name__ == "__main__":
    main()
