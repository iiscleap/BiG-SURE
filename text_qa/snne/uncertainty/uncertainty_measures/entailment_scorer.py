"""
Soft-scoring entailment utilities for graph-based uncertainty.

This module provides batched entailment scoring using softmax probabilities
instead of argmax decisions, enabling graded edge weights in bipartite graphs.
"""

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EntailmentScorer:
    """
    Batched entailment scorer using DeBERTa-MNLI.
    
    Provides soft probability-based scoring instead of hard argmax decisions.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v2-xlarge-mnli",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
        fp16: bool = False,
        cache_size: int = 0
    ):
        """
        Initialize the entailment scorer.
        
        Args:
            model_name: HuggingFace model name for NLI
            device: Device to use (None for auto-detect)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            fp16: Whether to use FP16 inference
            cache_size: LRU cache size (0 to disable)
        """
        self.device = device or DEVICE
        self.batch_size = batch_size
        self.max_length = max_length
        self.fp16 = fp16
        self.cache_size = cache_size
        
        logger.info(f"Loading entailment model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup caching if requested
        if cache_size > 0:
            self._cached_single_prob = lru_cache(maxsize=cache_size)(self._single_prob_uncached)
        else:
            self._cached_single_prob = self._single_prob_uncached
        
        logger.info(f"Entailment scorer ready on {self.device}")
    
    def _single_prob_uncached(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        """Compute probabilities for a single pair (uncached version)."""
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            if self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
            
            probs = F.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        
        return tuple(probs.astype(np.float32))
    
    def batch_probs(
        self,
        premises: List[str],
        hypotheses: List[str]
    ) -> np.ndarray:
        """
        Compute entailment probabilities for batches of premise-hypothesis pairs.
        
        Args:
            premises: List of premise strings
            hypotheses: List of hypothesis strings (same length as premises)
        
        Returns:
            np.ndarray of shape (N, 3) with probabilities [p_contradiction, p_neutral, p_entail]
        """
        assert len(premises) == len(hypotheses), "Premises and hypotheses must have same length"
        
        if len(premises) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        
        all_probs = []
        
        for i in range(0, len(premises), self.batch_size):
            batch_premises = premises[i:i + self.batch_size]
            batch_hypotheses = hypotheses[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            with torch.no_grad():
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
                
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
            
            all_probs.append(probs)
        
        return np.vstack(all_probs).astype(np.float32)
    
    def batch_weight(
        self,
        premises: List[str],
        hypotheses: List[str],
        score_mode: str = "entail_prob"
    ) -> np.ndarray:
        """
        Compute scalar weights from entailment probabilities.
        
        Args:
            premises: List of premise strings
            hypotheses: List of hypothesis strings
            score_mode: One of:
                - "entail_prob": weight = p_entail
                - "entail_over_noncontrad": weight = p_entail / (p_entail + p_neutral + eps)
                - "noncontrad_prob": weight = p_entail + p_neutral
        
        Returns:
            np.ndarray of shape (N,) with weights in [0, 1]
        """
        probs = self.batch_probs(premises, hypotheses)
        
        if len(probs) == 0:
            return np.zeros(0, dtype=np.float32)
        
        p_contradiction = probs[:, 0]
        p_neutral = probs[:, 1]
        p_entail = probs[:, 2]
        
        eps = 1e-10
        
        if score_mode == "entail_prob":
            weights = p_entail
        elif score_mode == "entail_over_noncontrad":
            weights = p_entail / (p_entail + p_neutral + eps)
        elif score_mode == "noncontrad_prob":
            weights = p_entail + p_neutral
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}. "
                           f"Must be one of: entail_prob, entail_over_noncontrad, noncontrad_prob")
        
        return weights.astype(np.float32)
    
    def pairwise_matrix(
        self,
        left_texts: List[str],
        right_texts: List[str],
        score_mode: str = "entail_prob",
        direction: str = "left_to_right"
    ) -> np.ndarray:
        """
        Compute pairwise weight matrix for all combinations.
        
        Args:
            left_texts: List of m texts
            right_texts: List of n texts
            score_mode: Scoring mode (see batch_weight)
            direction: One of:
                - "left_to_right": premise=left, hypothesis=right
                - "right_to_left": premise=right, hypothesis=left
        
        Returns:
            np.ndarray of shape (m, n) where M[i,j] is the weight for pair (i, j)
        """
        m = len(left_texts)
        n = len(right_texts)
        
        if m == 0 or n == 0:
            return np.zeros((m, n), dtype=np.float32)
        
        # Flatten all pairs
        premises = []
        hypotheses = []
        
        for i in range(m):
            for j in range(n):
                if direction == "left_to_right":
                    premises.append(left_texts[i])
                    hypotheses.append(right_texts[j])
                elif direction == "right_to_left":
                    premises.append(right_texts[j])
                    hypotheses.append(left_texts[i])
                else:
                    raise ValueError(f"Unknown direction: {direction}")
        
        # Compute weights
        weights = self.batch_weight(premises, hypotheses, score_mode)
        
        # Reshape to matrix
        return weights.reshape(m, n)
    
    def bipartite_weights(
        self,
        low_texts: List[str],
        high_texts: List[str],
        score_mode: str = "entail_prob",
        combine: str = "min"
    ) -> np.ndarray:
        """
        Compute bidirectional bipartite weight matrix.
        
        Computes forward and backward entailment weights and combines them.
        
        Args:
            low_texts: List of m low-T texts
            high_texts: List of n high-T texts
            score_mode: Scoring mode (see batch_weight)
            combine: Combination method:
                - "min": W = min(W_forward, W_backward)
                - "geom_mean": W = sqrt(W_forward * W_backward + eps)
                - "mean": W = 0.5 * (W_forward + W_backward)
        
        Returns:
            np.ndarray of shape (m, n) with combined weights in [0, 1]
        """
        # Forward: low -> high
        W_forward = self.pairwise_matrix(low_texts, high_texts, score_mode, "left_to_right")
        
        # Backward: high -> low (but keep shape m x n)
        W_backward = self.pairwise_matrix(low_texts, high_texts, score_mode, "right_to_left")
        
        eps = 1e-10
        
        if combine == "min":
            W = np.minimum(W_forward, W_backward)
        elif combine == "geom_mean":
            W = np.sqrt(W_forward * W_backward + eps)
        elif combine == "mean":
            W = 0.5 * (W_forward + W_backward)
        else:
            raise ValueError(f"Unknown combine mode: {combine}. "
                           f"Must be one of: min, geom_mean, mean")
        
        return W.astype(np.float32)
    
    def pairwise_probs_matrix(
        self,
        left_texts: List[str],
        right_texts: List[str],
        direction: str = "left_to_right"
    ) -> np.ndarray:
        """
        Compute pairwise probability matrix for all combinations.
        
        Returns:
            np.ndarray of shape (m, n, 3) with [p_contra, p_neutral, p_entail] per pair
        """
        m = len(left_texts)
        n = len(right_texts)
        
        if m == 0 or n == 0:
            return np.zeros((m, n, 3), dtype=np.float32)
        
        premises = []
        hypotheses = []
        
        for i in range(m):
            for j in range(n):
                if direction == "left_to_right":
                    premises.append(left_texts[i])
                    hypotheses.append(right_texts[j])
                else:
                    premises.append(right_texts[j])
                    hypotheses.append(left_texts[i])
        
        probs = self.batch_probs(premises, hypotheses)
        return probs.reshape(m, n, 3)
    
    def bipartite_weights_baseline(
        self,
        low_texts: List[str],
        high_texts: List[str],
        strict: bool = False
    ) -> np.ndarray:
        """
        Compute bidirectional bipartite weights using the EXACT SAME LOGIC
        as the binary baseline, but with soft probabilities.
        
        Binary baseline formula:
            is_equiv = (0 not in implications) AND (implications != [1, 1])
            
            Where 0=contradiction, 1=neutral, 2=entailment
            
            This means: NOT contradiction in either direction AND 
                        NOT (both neutral)
        
        Soft equivalent:
            weight = (1 - p_contra_fwd) × (1 - p_contra_bwd) × 
                     (1 - p_neutral_fwd × p_neutral_bwd)
        
        For strict_entailment mode (both must be entailment):
            weight = p_entail_fwd × p_entail_bwd
        
        Args:
            low_texts: List of m low-T texts
            high_texts: List of n high-T texts
            strict: If True, require bidirectional entailment (strict mode)
        
        Returns:
            np.ndarray of shape (m, n) with weights in [0, 1]
        """
        # Get full probability matrices
        probs_fwd = self.pairwise_probs_matrix(low_texts, high_texts, "left_to_right")
        probs_bwd = self.pairwise_probs_matrix(low_texts, high_texts, "right_to_left")
        
        # Extract individual probabilities [p_contra, p_neutral, p_entail]
        p_contra_fwd = probs_fwd[:, :, 0]
        p_neutral_fwd = probs_fwd[:, :, 1]
        p_entail_fwd = probs_fwd[:, :, 2]
        
        p_contra_bwd = probs_bwd[:, :, 0]
        p_neutral_bwd = probs_bwd[:, :, 1]
        p_entail_bwd = probs_bwd[:, :, 2]
        
        if strict:
            # Strict: both must be entailment
            # Binary: (impl_1 == 2) AND (impl_2 == 2)
            # Soft: p_entail_fwd × p_entail_bwd
            W = p_entail_fwd * p_entail_bwd
        else:
            # Relaxed: (0 not in implications) AND (implications != [1, 1])
            # = NOT contra_fwd AND NOT contra_bwd AND NOT (neutral_fwd AND neutral_bwd)
            # Soft: (1 - p_contra_fwd) × (1 - p_contra_bwd) × (1 - p_neutral_fwd × p_neutral_bwd)
            not_contra_fwd = 1.0 - p_contra_fwd
            not_contra_bwd = 1.0 - p_contra_bwd
            not_both_neutral = 1.0 - p_neutral_fwd * p_neutral_bwd
            
            W = not_contra_fwd * not_contra_bwd * not_both_neutral
        
        return np.clip(W, 0.0, 1.0).astype(np.float32)


# ============================================================================
# Graph Helper Functions
# ============================================================================

def support_per_high(W: np.ndarray, pool: str = "max", tau: float = 0.05) -> np.ndarray:
    """
    Compute support score for each high-T node from low-T anchors.
    
    Args:
        W: Weight matrix of shape (m, n) where m = low-T, n = high-T
        pool: Pooling method:
            - "max": s[j] = max_i W[i,j]
            - "softmax": s[j] = tau * log(sum_i exp(W[i,j] / tau)) (numerically stable)
        tau: Temperature for softmax pooling (default: 0.05)
    
    Returns:
        np.ndarray of shape (n,) with support scores per high-T node
    """
    m, n = W.shape
    
    if pool == "max":
        s = W.max(axis=0)
    elif pool == "softmax":
        # Numerically stable softmax pooling (log-sum-exp scaled by tau)
        # s[j] = tau * log(sum_i exp(W[i,j] / tau))
        W_scaled = W / tau
        max_W = W_scaled.max(axis=0, keepdims=True)
        s = tau * (max_W.squeeze() + np.log(np.sum(np.exp(W_scaled - max_W), axis=0)))
    else:
        raise ValueError(f"Unknown pool mode: {pool}. Must be one of: max, softmax")
    
    # Clip to [0, 1]
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def uncertainty_from_support(s: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convert support scores to per-sample uncertainty.
    
    Args:
        s: Support scores of shape (n,)
        eps: Small constant for numerical stability
    
    Returns:
        np.ndarray of shape (n,) with uncertainty values u_j = -log(s_j + eps)
    """
    return (-np.log(s + eps)).astype(np.float32)


def sparsify_topk(W: np.ndarray, topk_per_high: int) -> np.ndarray:
    """
    Sparsify weight matrix by keeping only top-K edges per high-T node.
    
    Args:
        W: Weight matrix of shape (m, n)
        topk_per_high: Number of edges to keep per high-T node (column)
    
    Returns:
        Sparsified weight matrix with same shape
    """
    if topk_per_high <= 0:
        return W
    
    W_sparse = W.copy()
    m, n = W.shape
    
    for j in range(n):
        col = W[:, j]
        if m <= topk_per_high:
            continue
        # Find threshold (k-th largest value)
        threshold = np.partition(col, -topk_per_high)[-topk_per_high]
        # Zero out values below threshold
        W_sparse[:, j] = np.where(col >= threshold, col, 0.0)
    
    return W_sparse.astype(np.float32)


# ============================================================================
# Rephrase Spread Weighting Utilities
# ============================================================================

def char_ngram_set(text: str, n: int = 3) -> set:
    """
    Extract character n-gram set from text.
    
    Args:
        text: Input text string
        n: N-gram size (default: 3 for trigrams)
    
    Returns:
        Set of character n-grams
    """
    text = text.lower().strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
    
    Returns:
        Jaccard similarity in [0, 1]
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_rephrase_spread_weights(
    rephrase_questions: List[str],
    original_question: Optional[str] = None,
    tau: float = 0.1,
    floor: float = 0.02,
    drift_beta: float = 1.0,
    drift_threshold: Optional[float] = None,
    ngram_size: int = 3,
    spread_mode: str = "mean"
) -> Tuple[np.ndarray, dict]:
    """
    Compute spread-based weights for rephrasings based on question text only.
    
    Higher weights for rephrasings that are more lexically different from others.
    Optionally downweight rephrasings that drift too far from original.
    
    Args:
        rephrase_questions: List of K rephrased question strings
        original_question: Original question (for drift guardrail)
        tau: Temperature for softmax weight conversion (lower = peakier)
        floor: Minimum weight floor before normalization
        drift_beta: Exponent for drift penalty (g_i ^ beta)
        drift_threshold: If set, filter out rephrasings with sim < threshold
        ngram_size: Character n-gram size for similarity
        spread_mode: "mean" (average distance to others) or "min" (min distance)
    
    Returns:
        weights: np.ndarray of shape (K,) with normalized weights
        info: Dict with intermediate values for debugging
    """
    K = len(rephrase_questions)
    
    if K == 0:
        return np.array([]), {}
    
    if K == 1:
        return np.array([1.0]), {'spreads': [0.0], 'drifts': [1.0]}
    
    # Step 1: Compute char n-gram sets for each rephrase
    ngram_sets = [char_ngram_set(q, ngram_size) for q in rephrase_questions]
    
    # Step 2: Compute pairwise Jaccard similarities and distances
    sim_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = jaccard_similarity(ngram_sets[i], ngram_sets[j])
    
    dist_matrix = 1.0 - sim_matrix
    
    # Step 3: Compute spread score per rephrase
    if spread_mode == "mean":
        # Mean distance to others
        spreads = np.array([
            np.sum(dist_matrix[i]) / (K - 1) for i in range(K)
        ])
    elif spread_mode == "min":
        # Min distance to closest neighbor
        spreads = np.array([
            np.min([dist_matrix[i, j] for j in range(K) if j != i])
            for i in range(K)
        ])
    else:
        raise ValueError(f"Unknown spread_mode: {spread_mode}")
    
    # Step 4: Convert spreads to weights via softmax
    spreads_scaled = spreads / tau
    spreads_scaled = spreads_scaled - spreads_scaled.max()  # numerical stability
    weights = np.exp(spreads_scaled)
    weights = weights / weights.sum()
    
    # Step 5: Add floor and renormalize
    weights = weights + floor
    weights = weights / weights.sum()
    
    # Step 6: Drift guardrail (similarity to original)
    drifts = np.ones(K)
    if original_question is not None:
        orig_ngrams = char_ngram_set(original_question, ngram_size)
        drifts = np.array([
            jaccard_similarity(ngram_sets[i], orig_ngrams) for i in range(K)
        ])
        
        if drift_threshold is not None:
            # Filter mode: zero out weights for drifted rephrasings
            mask = drifts >= drift_threshold
            weights = weights * mask
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # All filtered out, fall back to uniform
                weights = np.ones(K) / K
        elif drift_beta > 0:
            # Downweight mode: w_i *= g_i^beta
            weights = weights * (drifts ** drift_beta)
            weights = weights / weights.sum()
    
    info = {
        'spreads': spreads.tolist(),
        'drifts': drifts.tolist(),
        'sim_matrix': sim_matrix,
        'dist_matrix': dist_matrix,
        'raw_weights_before_drift': (np.exp(spreads_scaled) / np.exp(spreads_scaled).sum()).tolist()
    }
    
    return weights.astype(np.float32), info


def stratified_sample_by_weights(
    items_per_group: List[List],
    weights: np.ndarray,
    k_total: int,
    seed: Optional[int] = None
) -> Tuple[List, List[int]]:
    """
    Stratified sampling: allocate k_total samples across groups by weights.
    
    Uses largest-remainder method for deterministic allocation.
    
    Args:
        items_per_group: List of K groups, each group is a list of items
        weights: np.ndarray of shape (K,) with normalized weights (sum=1)
        k_total: Total number of items to sample
        seed: Random seed for within-group sampling
    
    Returns:
        sampled_items: Flat list of sampled items
        group_ids: List of group indices for each sampled item
    """
    import random
    
    K = len(items_per_group)
    
    if K == 0:
        return [], []
    
    # Compute desired counts (fractional)
    desired = weights * k_total
    
    # Integer part
    counts = np.floor(desired).astype(int)
    
    # Remainder for largest-remainder allocation
    remainders = desired - counts
    leftover = k_total - counts.sum()
    
    # Allocate leftover to groups with largest remainders
    if leftover > 0:
        indices = np.argsort(-remainders)
        for i in range(int(leftover)):
            counts[indices[i]] += 1
    
    # Cap counts by available items in each group
    for i in range(K):
        counts[i] = min(counts[i], len(items_per_group[i]))
    
    # If we're short due to capping, redistribute
    while counts.sum() < k_total:
        for i in range(K):
            if counts[i] < len(items_per_group[i]):
                counts[i] += 1
                if counts.sum() >= k_total:
                    break
        else:
            break  # No more items available
    
    # Sample from each group
    rnd = random.Random(seed) if seed is not None else random
    
    sampled_items = []
    group_ids = []
    
    for group_idx, (group, count) in enumerate(zip(items_per_group, counts)):
        if count > 0:
            if count >= len(group):
                selected = group
            else:
                selected = rnd.sample(group, count) if seed else group[:count]
            sampled_items.extend(selected)
            group_ids.extend([group_idx] * len(selected))
    
    return sampled_items, group_ids


def weighted_uncertainty_aggregate(
    per_sample_uncertainty: np.ndarray,
    sample_group_ids: np.ndarray,
    group_weights: np.ndarray
) -> float:
    """
    Compute weighted uncertainty using group (rephrase) weights.
    
    U = sum_j(a_j * u_j) / sum_j(a_j)
    
    where a_j = w_{r(j)} (weight of sample j's rephrase group)
    
    Args:
        per_sample_uncertainty: np.ndarray of shape (n,) with -log(s_j + eps)
        sample_group_ids: np.ndarray of shape (n,) with group index for each sample
        group_weights: np.ndarray of shape (K,) with rephrase weights
    
    Returns:
        Weighted uncertainty scalar
    """
    # Assign answer weight from its rephrase's weight
    answer_weights = group_weights[sample_group_ids.astype(int)]
    
    # Weighted average
    weighted_sum = np.sum(answer_weights * per_sample_uncertainty)
    weight_total = np.sum(answer_weights)
    
    return weighted_sum / (weight_total + 1e-10)

