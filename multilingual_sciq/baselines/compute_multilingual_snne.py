#!/usr/bin/env python3
"""
Compute Multilingual Soft Nearest Neighbor Entropy (SNNE) for uncertainty quantification.

This script computes SNNE from sampling outputs for each language.
SNNE uses a similarity matrix (lexical or NLI-based) to compute
soft nearest neighbor loss as an uncertainty measure.

Usage:
    python compute_multilingual_snne.py \
        --vanilla_json /path/to/vanilla/generate.json \
        --sampling_json /path/to/sampling/generate.json \
        --output_dir ./results/snne \
        --languages en zh ja fr th
"""
import os
import argparse
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rouge_score import tokenizers
import evaluate

from multilingual_utils import (
    load_json_data,
    get_languages,
    extract_vanilla_data,
    extract_sampling_data,
    calculate_prem_accuracy,
    MultilingualEntailmentDeberta,
    get_semantic_ids_using_entailment,
    auroc,
    auarc,
    aucpr,
    validate_generation_pair,
    LANGUAGES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_original_ids_from_rephrased(rephrased_json_path: str) -> set:
    """
    Load the original question IDs from a rephrased JSON file.
    Original questions are those without an 'original_id' field.
    
    Args:
        rephrased_json_path: Path to the rephrased JSON file
        
    Returns:
        Set of original question IDs (strings)
    """
    import json
    with open(rephrased_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Original questions don't have 'original_id' field
    original_ids = {str(item['question_id']) for item in data if 'original_id' not in item}
    logger.info(f"Loaded {len(original_ids)} original question IDs from {rephrased_json_path}")
    return original_ids


def filter_data_by_ids(data: List[Dict], valid_ids: set) -> List[Dict]:
    """
    Filter data to only include samples with question_id in valid_ids.
    
    Args:
        data: List of data dictionaries
        valid_ids: Set of valid question IDs
        
    Returns:
        Filtered list of data dictionaries
    """
    filtered = [item for item in data if str(item.get('question_id', '')) in valid_ids]
    logger.info(f"Filtered data from {len(data)} to {len(filtered)} samples")
    return filtered


def compute_lexical_similarity_matrix(responses: List[str]) -> np.ndarray:
    """
    Compute lexical similarity matrix using Rouge-L.
    
    Returns:
        NxN similarity matrix where N is the number of responses
    """
    n = len(responses)
    if n == 0:
        return np.array([])
    
    rouge = evaluate.load('rouge', keep_in_memory=True)
    tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize
    
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif j < i:
                similarity_matrix[i, j] = similarity_matrix[j, i]
            else:
                try:
                    score = rouge.compute(
                        predictions=[responses[i]],
                        references=[responses[j]],
                        rouge_types=['rougeL'],
                        tokenizer=tokenizer
                    )['rougeL']
                    similarity_matrix[i, j] = score
                except Exception:
                    similarity_matrix[i, j] = 0.0
    
    return similarity_matrix


def compute_entailment_similarity_matrix(
    responses: List[str],
    nli_model: MultilingualEntailmentDeberta,
    strict_entailment: bool = True
) -> np.ndarray:
    """
    Compute similarity matrix using NLI entailment scores.
    
    Returns:
        NxN similarity matrix
    """
    n = len(responses)
    if n == 0:
        return np.array([])
    
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif j < i:
                similarity_matrix[i, j] = similarity_matrix[j, i]
            else:
                score = nli_model.get_similarity_score(
                    responses[i], 
                    responses[j], 
                    strict_entailment=strict_entailment
                )
                similarity_matrix[i, j] = score
    
    return similarity_matrix


def soft_nearest_neighbor_entropy(
    similarity_matrix: np.ndarray,
    semantic_ids: List[int],
    temperature: float = 1.0,
    exclude_diagonal: bool = True,
    weights: np.ndarray = None
) -> float:
    """
    Compute Soft Nearest Neighbor Entropy.
    
    Args:
        similarity_matrix: NxN similarity matrix
        semantic_ids: Cluster assignments for each response
        temperature: Temperature for softmax scaling
        exclude_diagonal: Whether to exclude self-similarity
        weights: Optional weights for responses
    
    Returns:
        SNNE score (higher = more uncertain)
    """
    n = len(semantic_ids)
    if n <= 1:
        return 0.0
    
    similarity_matrix = np.array(similarity_matrix)
    semantic_ids = np.array(semantic_ids)
    
    # Scale by temperature
    scaled_sim = similarity_matrix / temperature
    
    if exclude_diagonal:
        np.fill_diagonal(scaled_sim, -np.inf)
    
    # Compute softmax weights
    exp_sim = np.exp(scaled_sim - np.max(scaled_sim, axis=1, keepdims=True))
    if exclude_diagonal:
        np.fill_diagonal(exp_sim, 0)
    
    softmax_weights = exp_sim / (np.sum(exp_sim, axis=1, keepdims=True) + 1e-10)
    
    # Compute probability of picking same cluster
    same_cluster = (semantic_ids[:, None] == semantic_ids[None, :]).astype(float)
    if exclude_diagonal:
        np.fill_diagonal(same_cluster, 0)
    
    # P(same cluster) for each point
    p_same = np.sum(softmax_weights * same_cluster, axis=1)
    
    # Apply response weights if provided
    if weights is not None:
        weights = np.array(weights) / np.sum(weights)
        snne = -np.sum(weights * np.log(p_same + 1e-10))
    else:
        snne = -np.mean(np.log(p_same + 1e-10))
    
    return snne


def compute_snne_for_language(
    vanilla_data: List[Dict],
    sampling_data: List[Dict],
    language: str,
    nli_model: MultilingualEntailmentDeberta,
    similarity_type: str = 'lexical',
    temperature: float = 1.0,
    strict_entailment: bool = False,
    metric: str = 'prem'
) -> Dict:
    """
    Compute SNNE for a single language.
    
    Returns:
        Dictionary with per-question uncertainties and overall metrics
    """
    # Extract language-specific data
    v_questions, v_answers, v_outputs, v_probs = extract_vanilla_data(vanilla_data, language)
    s_questions, s_answers, s_outputs, s_probs = extract_sampling_data(sampling_data, language)
    
    # Verify alignment
    assert len(v_questions) == len(s_questions), "Vanilla and sampling data mismatch"
    
    # Compute measures for each question
    uncertainties = []
    correctness = []
    question_ids = []
    
    for idx in tqdm(range(len(v_questions)), desc=f"Processing {language}"):
        outputs = s_outputs[idx]
        probs = s_probs[idx]
        # v_outputs[idx] is a list of k low-T responses; use first as greedy answer
        vanilla_output_list = v_outputs[idx]
        if isinstance(vanilla_output_list, list) and len(vanilla_output_list) > 0:
            vanilla_output = vanilla_output_list[0]
        else:
            vanilla_output = str(vanilla_output_list) if vanilla_output_list else ""
        ground_truth = v_answers[idx]
        
        # Correctness label
        if metric in ('claude', 'gemini'):
            # Use precomputed accuracy labels from vanilla_data
            is_correct = vanilla_data[idx].get('accuracy', {}).get(language, 0.0)
        else:
            is_correct = calculate_prem_accuracy(vanilla_output, ground_truth)
        correctness.append(is_correct)
        question_ids.append(idx)
        
        # Skip if insufficient outputs
        if not outputs or len(outputs) < 2:
            uncertainties.append(np.nan)
            continue
        
        # Get semantic IDs
        semantic_ids = get_semantic_ids_using_entailment(
            outputs, 
            nli_model, 
            strict_entailment=strict_entailment
        )
        
        # Compute similarity matrix
        if similarity_type == 'lexical':
            similarity_matrix = compute_lexical_similarity_matrix(outputs)
        else:  # entailment
            similarity_matrix = compute_entailment_similarity_matrix(
                outputs, nli_model, strict_entailment
            )
        
        # Compute SNNE
        snne = soft_nearest_neighbor_entropy(
            similarity_matrix,
            semantic_ids,
            temperature=temperature,
            exclude_diagonal=True
        )
        uncertainties.append(snne)
    
    # Compute overall metrics
    results = {
        'language': language,
        'num_samples': len(correctness),
        'accuracy': np.mean(correctness),
        'mean_snne': np.nanmean(uncertainties),
        'uncertainties': uncertainties,
        'correctness': correctness,
        'question_ids': question_ids
    }
    
    # Compute AUROC
    is_incorrect = [1 - c for c in correctness]
    
    valid_mask = ~np.isnan(uncertainties)
    if np.sum(valid_mask) >= 10:
        valid_unc = np.array(uncertainties)[valid_mask]
        valid_incorrect = np.array(is_incorrect)[valid_mask]
        valid_correct = np.array(correctness)[valid_mask]
        
        try:
            results['snne_auroc'] = auroc(valid_incorrect, valid_unc)
        except Exception as e:
            logger.warning(f"Failed to compute AUROC: {e}")
            results['snne_auroc'] = -1
        
        try:
            results['snne_auarc'] = auarc(valid_unc, valid_correct)
        except Exception as e:
            results['snne_auarc'] = -1
        
        try:
            results['snne_aucpr'] = aucpr(valid_unc, valid_correct)
        except Exception as e:
            results['snne_aucpr'] = -1
    else:
        results['snne_auroc'] = -1
        results['snne_auarc'] = -1
        results['snne_aucpr'] = -1
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute multilingual SNNE for uncertainty quantification'
    )
    parser.add_argument('--vanilla_json', type=str, required=True,
                       help='Path to vanilla inference JSON')
    parser.add_argument('--sampling_json', type=str, required=True,
                       help='Path to sampling inference JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--languages', type=str, nargs='+', default=None,
                       help='Languages to process (default: all available)')
    parser.add_argument('--similarity', type=str, default='lexical',
                       choices=['lexical', 'entailment'],
                       help='Similarity type for SNNE')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for SNNE softmax')
    parser.add_argument('--strict_entailment', action='store_true',
                       help='Require bidirectional entailment for equivalence')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to process (for debugging)')
    parser.add_argument('--metric', choices=['prem', 'claude', 'gemini'], default='prem',
                        help='Accuracy metric: prem (default), claude, or gemini (reads from JSON)')
    parser.add_argument('--filter_ids_from', type=str, default=None,
                        help='Path to rephrased JSON to extract original question IDs for filtering')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("MULTILINGUAL SNNE")
    logger.info("=" * 60)
    logger.info(f"Vanilla JSON: {args.vanilla_json}")
    logger.info(f"Sampling JSON: {args.sampling_json}")
    logger.info(f"Similarity: {args.similarity}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Metric: {args.metric}")
    
    # Load data
    vanilla_data = load_json_data(args.vanilla_json)
    sampling_data = load_json_data(args.sampling_json)
    
    # Filter by question IDs from rephrased JSON if specified
    if args.filter_ids_from:
        valid_ids = load_original_ids_from_rephrased(args.filter_ids_from)
        vanilla_data = filter_data_by_ids(vanilla_data, valid_ids)
        sampling_data = filter_data_by_ids(sampling_data, valid_ids)
    
    if args.max_samples:
        vanilla_data = vanilla_data[:args.max_samples]
        sampling_data = sampling_data[:args.max_samples]
    
    # Determine languages
    available_languages = get_languages(vanilla_data)
    languages = args.languages if args.languages else available_languages
    logger.info(f"Languages to process: {languages}")
    validate_generation_pair(
        vanilla_data, sampling_data, languages,
        require_accuracy=args.metric in ('claude', 'gemini')
    )
    
    # Load NLI model
    nli_model = MultilingualEntailmentDeberta()
    
    # Process each language
    all_results = []
    detailed_results = {}
    
    for lang in languages:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing language: {lang}")
        logger.info(f"{'='*40}")
        
        results = compute_snne_for_language(
            vanilla_data,
            sampling_data,
            lang,
            nli_model,
            similarity_type=args.similarity,
            temperature=args.temperature,
            strict_entailment=args.strict_entailment,
            metric=args.metric
        )
        
        detailed_results[lang] = results
        
        # Summary row
        summary = {
            'language': lang,
            'num_samples': results['num_samples'],
            'accuracy': results['accuracy'],
            'mean_snne': results['mean_snne'],
            'snne_auroc': results['snne_auroc'],
            'snne_auarc': results['snne_auarc'],
            'snne_aucpr': results['snne_aucpr']
        }
        all_results.append(summary)
        
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Mean SNNE: {results['mean_snne']:.4f}")
        logger.info(f"SNNE AUROC: {results['snne_auroc']:.4f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Summary CSV
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(args.output_dir, f'snne_{args.similarity}_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSaved summary to {summary_path}")
    
    # Detailed per-question results
    for lang, results in detailed_results.items():
        detailed_df = pd.DataFrame({
            'question_id': results['question_ids'],
            'correctness': results['correctness'],
            'snne': results['uncertainties']
        })
        detailed_path = os.path.join(args.output_dir, f'snne_{args.similarity}_{lang}_detailed.csv')
        detailed_df.to_csv(detailed_path, index=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    
    print("\n" + summary_df.to_string(index=False))
    
    return 0


if __name__ == "__main__":
    exit(main())
