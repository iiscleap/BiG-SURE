#!/usr/bin/env python3
"""
Soft Edge Fraction for VQA dataset.

This script computes graph-based uncertainty using precomputed entailments
from precompute_entailments_vqa.py. It supports all scoring modes:
- baseline_relaxed: (1-p_contra_fwd)×(1-p_contra_bwd)×(1-p_neutral_fwd×p_neutral_bwd)
- baseline_strict: p_entail_fwd × p_entail_bwd
- entail_prob: p_entail with combine (min/geom_mean/mean)
- entail_over_noncontrad: p_entail/(p_entail+p_neutral) with combine
- noncontrad_prob: (p_entail+p_neutral) with combine

Input:
- Precomputed entailments .npz file (from precompute_entailments_vqa.py)
- Accuracy JSON file (from vqa_accuracy_results/) - fallback if not in npz

Output:
- Graph results pickle with confidence scores
- AUROC metrics
- Summary CSV

Usage:
    python compute_soft_edge_all_modes_vqa.py \
        --entailments_file vqa_entailments/qwen_rephrased.npz \
        --accuracy_file vqa_accuracy_results/qwen2.5-vl-7b_vqa_accuracy.json \
        --output_dir vqa_soft_edge_results/qwen \
        --score_mode baseline_relaxed \
        --combine min
"""

import os
import sys
import pickle
import json
import argparse
import logging
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from snne.uncertainty.utils.normalization_utils import quantile_power_normalize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        if f"{prefix}accuracy" in data.files:
            results[qid]['accuracy'] = float(data[f"{prefix}accuracy"][0])
    
    return results


def probs_to_weights_matrix(probs_fwd, probs_bwd, score_mode, combine):
    """Convert precomputed probability matrices to edge weights.
    
    Args:
        probs_fwd: (m, n, 3) array - P(low_t -> high_t) [contradiction, neutral, entailment]
        probs_bwd: (m, n, 3) array - P(high_t -> low_t)
        score_mode: Scoring method to use
        combine: How to combine forward and backward scores
        
    Returns:
        W: (m, n) weight matrix
    """
    eps = 1e-10
    
    def probs_to_weight(probs, mode):
        """Convert 3-class probabilities to edge weights."""
        p_contradiction = probs[:, :, 0]
        p_neutral = probs[:, :, 1]
        p_entail = probs[:, :, 2]
        
        if mode == "entail_prob":
            # Just use entailment probability
            return p_entail
        elif mode == "entail_over_noncontrad":
            # Entailment normalized by non-contradiction
            return p_entail / (p_entail + p_neutral + eps)
        elif mode == "noncontrad_prob":
            # Total non-contradiction probability
            return p_entail + p_neutral
        elif mode == "baseline_relaxed":
            # Relaxed: consider edge if entailment > 0.5
            return (p_entail > 0.5).astype(np.float32)
        elif mode == "baseline_strict":
            # Strict: consider edge only if entailment is argmax
            return (np.argmax(probs, axis=2) == 2).astype(np.float32)
        else:
            raise ValueError(f"Unknown score_mode: {mode}")
    
    W_forward = probs_to_weight(probs_fwd, score_mode)
    W_backward = probs_to_weight(probs_bwd, score_mode)
    
    # Combine forward and backward scores
    if combine == "min":
        W = np.minimum(W_forward, W_backward)
    elif combine == "geom_mean":
        W = np.sqrt(W_forward * W_backward + eps)
    elif combine == "mean":
        W = 0.5 * (W_forward + W_backward)
    else:
        raise ValueError(f"Unknown combine mode: {combine}")
    
    return W.astype(np.float32)


def compute_normalized_cut(W, alpha=0.5):
    """
    Compute normalized cut score for bipartite graph.
    
    The normalized cut measures how well low-T and high-T clusters are separated.
    Lower values indicate tighter semantic clusters (lower uncertainty).
    
    Args:
        W: (m, n) weight matrix between low-T (rows) and high-T (columns)
        alpha: Weight for balancing cut and association terms
        
    Returns:
        ncut: Normalized cut score (0 to 1, lower = more certain)
    """
    m, n = W.shape
    if m == 0 or n == 0:
        return 1.0  # Maximum uncertainty if no edges
    
    # Total weight in the graph
    total_weight = W.sum()
    if total_weight == 0:
        return 1.0  # No edges = maximum uncertainty
    
    # Compute cut value (edges between low-T and high-T that are weak)
    # For bipartite, all edges are "cuts" - we measure their average strength
    mean_edge_strength = total_weight / (m * n)
    
    # Uncertainty = 1 - mean_edge_strength (weak edges = high uncertainty)
    return 1.0 - mean_edge_strength


def compute_graph_cut_metrics(W):
    """
    Compute various graph-based uncertainty metrics from weight matrix.
    
    Args:
        W: (m, n) weight matrix
        
    Returns:
        dict with various metrics
    """
    m, n = W.shape
    
    # Basic statistics
    mean_weight = float(W.mean()) if W.size > 0 else 0.0
    min_weight = float(W.min()) if W.size > 0 else 0.0
    max_weight = float(W.max()) if W.size > 0 else 0.0
    
    # Fraction of strong edges (> 0.5 threshold)
    strong_edge_frac = float((W > 0.5).mean()) if W.size > 0 else 0.0
    
    # Row-wise min (worst high-T match for each low-T)
    row_mins = W.min(axis=1) if n > 0 else np.zeros(m)
    mean_row_min = float(row_mins.mean()) if m > 0 else 0.0
    
    # Column-wise max (best low-T match for each high-T)
    col_maxs = W.max(axis=0) if m > 0 else np.zeros(n)
    mean_col_max = float(col_maxs.mean()) if n > 0 else 0.0
    
    return {
        'mean_weight': mean_weight,
        'min_weight': min_weight,
        'max_weight': max_weight,
        'strong_edge_frac': strong_edge_frac,
        'mean_row_min': mean_row_min,
        'mean_col_max': mean_col_max,
    }


def compute_soft_edge_with_precomputed(
    precomputed_entailments,
    output_dir,
    score_mode="baseline_relaxed",
    combine="min",
    accuracy_dict=None
):
    """
    Compute soft edge fraction using precomputed entailment probabilities.
    
    Args:
        precomputed_entailments: Dict of precomputed entailments from load_results_npz
        output_dir: Directory to save results
        score_mode: Scoring method
        combine: Combination method for bidirectional scores
        accuracy_dict: Dict mapping question ID to accuracy (0.0 to 1.0)
        
    Returns:
        graph_results: Dict with per-question confidence and metrics
    """
    logger.info(f"Processing {len(precomputed_entailments)} questions")
    logger.info(f"Score mode: {score_mode}, Combine: {combine}")
    
    graph_results = {}
    
    for qid in tqdm(sorted(precomputed_entailments.keys()), desc="Computing graph metrics"):
        precomp = precomputed_entailments[qid]
        
        # Convert precomputed probabilities to weight matrix
        W = probs_to_weights_matrix(
            precomp['probs_fwd'],
            precomp['probs_bwd'],
            score_mode,
            combine
        )
        
        # Compute confidence (mean edge weight)
        confidence = float(W.mean()) if W.size > 0 else 0.0
        
        # Compute additional graph metrics
        graph_metrics = compute_graph_cut_metrics(W)
        
        # Get accuracy (prefer external dict, fallback to precomputed)
        if accuracy_dict is not None and str(qid) in accuracy_dict:
            accuracy = accuracy_dict[str(qid)]
        else:
            accuracy = precomp.get('accuracy')
        
        graph_results[str(qid)] = {
            'accuracy': accuracy,
            'confidence': confidence,
            'm': precomp['m'],
            'n': precomp['n'],
            **graph_metrics
        }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_pickle(graph_results, output_path / f"results_{score_mode}_{combine}.pkl")
    
    return graph_results


def compute_auroc(graph_results, output_dir, score_mode, combine):
    """
    Compute AUROC using uncertainty to predict correctness.
    
    For VQA, accuracy can be continuous (0.0 to 1.0) based on the
    10-choose-9 averaging method. We binarize at 0.5 threshold.
    
    Args:
        graph_results: Dict with per-question results
        output_dir: Directory for output files
        score_mode: Scoring mode used
        combine: Combination method used
        
    Returns:
        auroc: Area under ROC curve (or None if cannot compute)
    """
    question_ids = sorted(graph_results.keys())
    
    # Extract confidences and accuracies
    confidences = np.array([graph_results[qid]['confidence'] for qid in question_ids])
    accuracies = [graph_results[qid].get('accuracy') for qid in question_ids]
    
    # Filter out None accuracies
    valid_mask = [a is not None for a in accuracies]
    valid_confidences = confidences[valid_mask]
    valid_accuracies = np.array([a for a in accuracies if a is not None])
    
    if len(valid_accuracies) == 0:
        logger.warning("No valid accuracy labels found")
        return None
    
    # For VQA, binarize accuracy at 0.5 threshold
    # (accuracy >= 0.5 means at least 5/10 annotators agree)
    binary_accuracies = (valid_accuracies >= 0.5).astype(float)
    
    # Convert confidence to uncertainty (lower confidence = higher uncertainty)
    eps = 1e-10
    uncertainties_raw = -np.log(valid_confidences + eps)
    
    # quantile_power_normalize does 1/x inversion internally, 
    # so output is a CONFIDENCE score (higher = more confident = less uncertain)
    confidence_normalized = quantile_power_normalize(uncertainties_raw)
    
    # Compute AUROC
    # AUROC expects: (y_true, y_score) where higher y_score predicts y_true=1
    # y_true: binary accuracy (1=Correct, 0=Incorrect)  
    # y_score: confidence_normalized (higher = more confident)
    # Higher confidence should predict correctness
    auroc = None
    if len(binary_accuracies) > 0 and len(set(binary_accuracies)) > 1:
        auroc = roc_auc_score(binary_accuracies, confidence_normalized)
    
    auroc_str = f"{auroc:.4f}" if auroc else "N/A"
    logger.info(f"Score: {score_mode}, Combine: {combine} -> AUROC: {auroc_str}")
    
    # Also compute accuracy
    overall_accuracy = float(valid_accuracies.mean())
    binary_accuracy = float(binary_accuracies.mean())
    logger.info(f"Overall accuracy (mean): {overall_accuracy:.4f}")
    logger.info(f"Binary accuracy (>=0.5): {binary_accuracy:.4f} ({len(valid_accuracies)} questions)")
    
    return auroc


def main():
    parser = argparse.ArgumentParser(description='Soft Edge Fraction for VQA')
    
    # Input files
    parser.add_argument('--entailments_file', type=str, required=True,
                       help='Path to precomputed entailments .npz file')
    parser.add_argument('--accuracy_file', type=str, default=None,
                       help='Path to accuracy JSON file (optional, uses npz if not provided)')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    
    # Scoring parameters
    parser.add_argument('--score_mode', type=str, default='baseline_relaxed',
                       choices=['baseline_relaxed', 'baseline_strict',
                                'entail_prob', 'entail_over_noncontrad', 'noncontrad_prob'],
                       help='Scoring method for edge weights')
    parser.add_argument('--combine', type=str, default='min',
                       choices=['min', 'geom_mean', 'mean'],
                       help='Method to combine forward and backward scores')
    
    # Metadata
    parser.add_argument('--model_name', type=str, default='',
                       help='Model name for logging')
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SOFT EDGE FRACTION - VQA")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Entailments: {args.entailments_file}")
    logger.info(f"Score mode: {args.score_mode}, Combine: {args.combine}")
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
    
    # Compute graph metrics
    graph_results = compute_soft_edge_with_precomputed(
        precomputed_entailments,
        args.output_dir,
        score_mode=args.score_mode,
        combine=args.combine,
        accuracy_dict=accuracy_dict
    )
    
    if not graph_results:
        logger.error("No results computed")
        return 1
    
    # Compute AUROC
    auroc = compute_auroc(graph_results, args.output_dir, args.score_mode, args.combine)
    
    # Save summary
    output_path = Path(args.output_dir)
    summary = {
        'model_name': args.model_name,
        'score_mode': args.score_mode,
        'combine': args.combine,
        'auroc': auroc,
        'num_questions': len(graph_results)
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path / f"summary_{args.score_mode}_{args.combine}.csv", index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"AUROC: {auroc:.4f}" if auroc else "AUROC: N/A")
    logger.info(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
