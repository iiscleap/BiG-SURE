#!/usr/bin/env python3
"""
Compute Multilingual Semantic Entropy for uncertainty quantification.

This script computes semantic entropy from sampling outputs for each language.
Semantic entropy groups model outputs by semantic equivalence (using NLI)
and computes entropy over these semantic clusters.

Usage:
    python compute_multilingual_semantic_entropy.py \
        --vanilla_json /path/to/vanilla/generate.json \
        --sampling_json /path/to/sampling/generate.json \
        --output_dir ./results/semantic_entropy \
        --languages en zh ja fr th
"""
import os
import argparse
import logging
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from multilingual_utils import (
    load_json_data,
    get_languages,
    extract_vanilla_data,
    extract_sampling_data,
    calculate_prem_accuracy,
    MultilingualEntailmentDeberta,
    get_semantic_ids_using_entailment,
    predictive_entropy,
    cluster_assignment_entropy,
    semantic_entropy,
    auroc,
    auarc,
    aucpr,
    save_results_csv,
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


def compute_semantic_entropy_for_language(
    vanilla_data: List[Dict],
    sampling_data: List[Dict],
    language: str,
    nli_model: MultilingualEntailmentDeberta,
    strict_entailment: bool = False,
    condition_on_question: bool = True,
    claude_accuracy_dict: dict = None
) -> Dict:
    """
    Compute semantic entropy and related measures for a single language.
    
    Returns:
        Dictionary with per-question uncertainties and overall metrics
    """
    # Extract language-specific data
    v_questions, v_answers, v_outputs, v_probs = extract_vanilla_data(vanilla_data, language)
    s_questions, s_answers, s_outputs, s_probs = extract_sampling_data(sampling_data, language)
    
    # Verify alignment
    assert len(v_questions) == len(s_questions), "Vanilla and sampling data mismatch"
    
    # Compute measures for each question
    uncertainties = defaultdict(list)
    correctness = []
    question_ids = []
    
    for idx in tqdm(range(len(v_questions)), desc=f"Processing {language}"):
        question = s_questions[idx]
        outputs = s_outputs[idx]
        probs = s_probs[idx]
        
        # Get vanilla answer and ground truth for accuracy
        # v_outputs[idx] is a list of k low-T responses; use first as greedy answer
        vanilla_output_list = v_outputs[idx]
        if isinstance(vanilla_output_list, list) and len(vanilla_output_list) > 0:
            vanilla_output = vanilla_output_list[0]
        else:
            vanilla_output = str(vanilla_output_list) if vanilla_output_list else ""
        ground_truth = v_answers[idx]
        
        # Calculate accuracy using PREM or Claude (match SNNE logic)
        if claude_accuracy_dict is not None:
            # SNNE expects accuracy in vanilla_data[idx]['accuracy'][language]
            is_correct = vanilla_data[idx].get('accuracy', {}).get(language, 0.0)
        else:
            is_correct = calculate_prem_accuracy(vanilla_output, ground_truth)
        correctness.append(is_correct)
        
        # Skip if no sampling outputs
        if not outputs or len(outputs) == 0:
            for key in ['semantic_entropy', 'predictive_entropy', 'cluster_entropy', 'num_clusters']:
                uncertainties[key].append(np.nan)
            question_ids.append(idx)
            continue
        
        # Convert probs to log probs
        probs = np.array(probs)
        probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
        log_probs = np.log(probs)
        
        # Get semantic IDs using NLI clustering
        context = question if condition_on_question else None
        semantic_ids = get_semantic_ids_using_entailment(
            outputs, 
            nli_model, 
            strict_entailment=strict_entailment,
            question=context
        )
        
        # Compute entropy measures
        pred_entropy = predictive_entropy(log_probs)
        cluster_entropy = cluster_assignment_entropy(semantic_ids)
        sem_entropy = semantic_entropy(semantic_ids, log_probs)
        num_clusters = len(set(semantic_ids))
        
        uncertainties['semantic_entropy'].append(sem_entropy)
        uncertainties['predictive_entropy'].append(pred_entropy)
        uncertainties['cluster_entropy'].append(cluster_entropy)
        uncertainties['num_clusters'].append(num_clusters)
        question_ids.append(idx)
    
    # Compute overall metrics
    results = {
        'language': language,
        'num_samples': len(correctness),
        'accuracy': np.mean(correctness),
        'uncertainties': dict(uncertainties),
        'correctness': correctness,
        'question_ids': question_ids
    }
    
    # Compute AUROC for each uncertainty measure
    # Note: For uncertainty, higher = more uncertain = more likely wrong
    # So we want to predict (1 - correctness) from uncertainty
    is_incorrect = [1 - c for c in correctness]
    
    for measure_name, values in uncertainties.items():
        if measure_name == 'num_clusters':
            continue
        
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < 10:
            results[f'{measure_name}_auroc'] = -1
            results[f'{measure_name}_auarc'] = -1
            results[f'{measure_name}_aucpr'] = -1
            continue
        
        valid_values = np.array(values)[valid_mask]
        valid_incorrect = np.array(is_incorrect)[valid_mask]
        valid_correct = np.array(correctness)[valid_mask]
        
        try:
            results[f'{measure_name}_auroc'] = auroc(valid_incorrect, valid_values)
        except Exception as e:
            logger.warning(f"Failed to compute AUROC for {measure_name}: {e}")
            results[f'{measure_name}_auroc'] = -1
        
        try:
            results[f'{measure_name}_auarc'] = auarc(valid_values, valid_correct)
        except Exception as e:
            logger.warning(f"Failed to compute AUARC for {measure_name}: {e}")
            results[f'{measure_name}_auarc'] = -1
        
        try:
            results[f'{measure_name}_aucpr'] = aucpr(valid_values, valid_correct)
        except Exception as e:
            logger.warning(f"Failed to compute AUCPR for {measure_name}: {e}")
            results[f'{measure_name}_aucpr'] = -1
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute multilingual semantic entropy for uncertainty quantification'
    )
    parser.add_argument('--vanilla_json', type=str, required=True,
                       help='Path to vanilla inference JSON')
    parser.add_argument('--sampling_json', type=str, required=True,
                       help='Path to sampling inference JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--languages', type=str, nargs='+', default=None,
                       help='Languages to process (default: all available)')
    parser.add_argument('--strict_entailment', action='store_true',
                       help='Require bidirectional entailment for equivalence')
    parser.add_argument('--condition_on_question', action='store_true', default=True,
                       help='Prepend question to answers for NLI')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to process (for debugging)')
    parser.add_argument('--metric', type=str, default='prem',
                        choices=['prem', 'claude', 'gemini'],
                        help='Accuracy metric: prem (default), claude, or gemini')
    # Remove gemini_accuracy_file argument, not needed
    parser.add_argument('--filter_ids_from', type=str, default=None,
                        help='Path to rephrased JSON to extract original question IDs for filtering')
    
    args = parser.parse_args()
    
    # Validation
    # No need to check for gemini_accuracy_file
    
    logger.info("=" * 60)
    logger.info("MULTILINGUAL SEMANTIC ENTROPY")
    logger.info("=" * 60)
    logger.info(f"Vanilla JSON: {args.vanilla_json}")
    logger.info(f"Sampling JSON: {args.sampling_json}")
    logger.info(f"Output directory: {args.output_dir}")
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
        
        # For claude/gemini metric, pass a dummy dict to trigger JSON accuracy logic
        claude_accuracy_dict = None
        if args.metric in ('claude', 'gemini'):
            claude_accuracy_dict = {}  # triggers vanilla_data[idx]['accuracy'][language] usage
        
        results = compute_semantic_entropy_for_language(
            vanilla_data,
            sampling_data,
            lang,
            nli_model,
            strict_entailment=args.strict_entailment,
            condition_on_question=args.condition_on_question,
            claude_accuracy_dict=claude_accuracy_dict
        )
        
        detailed_results[lang] = results
        
        # Summary row
        summary = {
            'language': lang,
            'num_samples': results['num_samples'],
            'accuracy': results['accuracy'],
        }
        
        for measure in ['semantic_entropy', 'predictive_entropy', 'cluster_entropy']:
            summary[f'{measure}_auroc'] = results.get(f'{measure}_auroc', -1)
            summary[f'{measure}_auarc'] = results.get(f'{measure}_auarc', -1)
            summary[f'{measure}_aucpr'] = results.get(f'{measure}_aucpr', -1)
        
        all_results.append(summary)
        
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        for measure in ['semantic_entropy', 'predictive_entropy', 'cluster_entropy']:
            logger.info(f"{measure} AUROC: {results.get(f'{measure}_auroc', -1):.4f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Summary CSV
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(args.output_dir, 'semantic_entropy_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSaved summary to {summary_path}")
    
    # Detailed per-question results per language
    for lang, results in detailed_results.items():
        detailed_df = pd.DataFrame({
            'question_id': results['question_ids'],
            'correctness': results['correctness'],
            **{k: v for k, v in results['uncertainties'].items()}
        })
        detailed_path = os.path.join(args.output_dir, f'semantic_entropy_{lang}_detailed.csv')
        detailed_df.to_csv(detailed_path, index=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    
    # Print summary table
    print("\n" + summary_df.to_string(index=False))
    
    return 0


if __name__ == "__main__":
    exit(main())
