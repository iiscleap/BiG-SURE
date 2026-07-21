#!/usr/bin/env python3
"""
Precompute entailment probabilities for VQA-like datasets (VQA, OKVQA, MathVista).

This script works with pickle files from wandb runs and optionally JSON accuracy files.

Input formats:
  - Accuracy file (OPTIONAL): JSON file with per_example accuracy from *_accuracy_results/
    If not provided, accuracy is extracted from vanilla pkl's most_likely_answer['accuracy']
  - Vanilla pkl: validation_generations.pkl from vanilla generation
    - low_temp_responses list as low-T samples (T=0.1)
    - most_likely_answer['accuracy'] used as accuracy if no JSON provided
    - Separate low-temp pkl (OPTIONAL): validation_generations.pkl from low-T input-sampled run
        - can contain IDs like {original_id}_rephrased{1-5}
        - low-T is aggregated across these variants (typically 5 x 10 = 50 samples)
        - sample source prefers low_temp_responses, falls back to responses
  - Rephrased pkl: validation_generations.pkl from rephrased generation
    - responses list as high-T samples (T=1.0)
    - IDs are in format: {original_id}_rephrased{1-5}
  - Perturbed pkl: validation_generations.pkl from perturbed generation
    - responses list as high-T samples (T=1.0)
    - IDs are in format: {original_id}_rephrased{1-5}_{perturbation_name}
    - Example: 262148000_rephrased1_contrast1

Mode:
  REPHRASED mode (--mode rephrased):
    - Uses low_temp_responses from vanilla pkl as low-T (3 samples at T=0.1)
    - Uses responses from rephrased pkl as high-T (5 rephrased × 10 samples = 50 total)
    - Accuracy from JSON or vanilla pkl

  PERTURBED mode (--mode perturbed):
    - Uses low_temp_responses from vanilla pkl as low-T (3 samples at T=0.1)
    - Uses responses from perturbed pkl as high-T (5 rephrased × 7 perturbations × 10 samples)
    - Each original question has ~350 high-T samples (text + image perturbations)
    - Accuracy from JSON or vanilla pkl

Output:
  - Saves as .npz (compressed numpy archive) with entailment probabilities

Usage:
    # With accuracy JSON
    python precompute_entailments_vqa.py \
        --accuracy_file vqa_accuracy_results/qwen2.5-vl-7b_vqa_accuracy.json \
        --vanilla_pkl wandb/run-YYY/files/validation_generations.pkl \
        --rephrased_pkl wandb/run-XXX/files/validation_generations.pkl \
        --output_file vqa_entailments/qwen_rephrased.npz \
        --mode rephrased

    # Without accuracy JSON (accuracy from wandb pkl)
    python precompute_entailments_vqa.py \
        --vanilla_pkl wandb/run-YYY/files/validation_generations.pkl \
        --rephrased_pkl wandb/run-ZZZ/files/validation_generations.pkl \
        --output_file okvqa_entailments/llava_perturbed.npz \
        --mode perturbed

    # With separate low-temp pkl (low-T from new runs, accuracy from vanilla pkl)
    python precompute_entailments_vqa.py \
        --vanilla_pkl wandb/run-YYY/files/validation_generations.pkl \
        --low_temp_pkl wandb/run-NEW/files/validation_generations.pkl \
        --rephrased_pkl wandb/run-ZZZ/files/validation_generations.pkl \
        --output_file okvqa_entailments/llava_perturbed.npz \
        --mode perturbed
"""

import os
import re
import sys
import argparse
import pickle
import json
import logging
import random
import hashlib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import numpy as np

from snne.uncertainty.uncertainty_measures.entailment_scorer import EntailmentScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_accuracy_json(accuracy_path):
    """
    Load accuracy file from *_accuracy_results JSON.
    
    Returns dict mapping question ID -> accuracy (float, 0.0 to 1.0)
    """
    with open(accuracy_path, 'r') as f:
        data = json.load(f)
    
    accuracy_dict = {}
    per_example = data.get('per_example', {})
    
    for qid, example_data in per_example.items():
        if isinstance(example_data, dict):
            accuracy_dict[str(qid)] = float(example_data.get('accuracy', 0.0))
        else:
            accuracy_dict[str(qid)] = float(example_data)
    
    logger.info(f"Loaded {len(accuracy_dict)} accuracy entries from {accuracy_path}")
    return accuracy_dict


def extract_accuracy_from_pkl(vanilla_data):
    """
    Extract accuracy from vanilla pkl's most_likely_answer['accuracy'] field.
    
    This is the default accuracy source when no external JSON is provided.
    The wandb runs store accuracy in the pkl files directly.
    
    Returns dict mapping question ID -> accuracy (float)
    """
    accuracy_dict = {}
    for qid, example in vanilla_data.items():
        mla = example.get('most_likely_answer', {})
        if isinstance(mla, dict) and 'accuracy' in mla:
            accuracy_dict[str(qid)] = float(mla['accuracy'])
    
    logger.info(f"Extracted {len(accuracy_dict)} accuracy entries from vanilla pkl")
    return accuracy_dict


def load_pkl_file(pkl_path):
    """Load pickle file containing generations."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f'{pkl_path} must contain a dictionary keyed by question ID')
    # Pickles created from JSON-backed datasets sometimes use integer IDs while
    # CSV-backed runs use strings. All cross-file joins use canonical strings.
    data = {str(qid): example for qid, example in data.items()}
    logger.info(f"Loaded {len(data)} entries from {pkl_path}")
    return data


def extract_original_id(rephrased_id, mode='rephrased'):
    """
    Extract original question ID from rephrased or perturbed ID.
    
    Examples:
        For mode='rephrased':
            '262148000_rephrased1' -> '262148000'
            '131089000_rephrased5' -> '131089000'
            '262148000' -> '262148000'
        
        For mode='perturbed':
            '262148000_rephrased1_contrast1' -> '262148000'
            '262148000_rephrased3_blur1' -> '262148000'
            '131089000_rephrased5_noise1' -> '131089000'
    """
    if mode == 'perturbed':
        # List of known perturbations used in these datasets
        perturbations = ['contrast', 'blur', 'rotate', 'shift', 'noise', 'masking', 'bw']
        
        # 1. Try stripping from _rephrased onwards
        base = re.split(r'_rephrased\d+', str(rephrased_id))[0]
        
        # 2. Check if the remaining base contains a perturbation to strip
        if '_' in base:
            parts = base.split('_')
            last_segment = parts[-1]
            for p in perturbations:
                if re.match('^' + p + r'\d*$', last_segment):
                    # Strip the perturbation segment and return the base ID
                    return '_'.join(parts[:-1])
        
        return base


    else:
        # Match pattern: {original_id}_rephrased{N}
        match = re.match(r'^(.+)_rephrased\d+$', str(rephrased_id))
        if match:
            return match.group(1)
    
    return str(rephrased_id)


def build_rephrased_mapping(rephrased_data, mode='rephrased'):
    """
    Build a mapping from original question ID to all rephrased/perturbed entries.
    
    Args:
        rephrased_data: Dict with keys like '262148000_rephrased1' or '262148000_rephrased1_contrast1'
        mode: 'rephrased' or 'perturbed'
        
    Returns:
        Dict mapping original_id -> list of (rephrased_id, data) tuples
    """
    mapping = defaultdict(list)
    
    for rephrased_id, data in rephrased_data.items():
        original_id = extract_original_id(rephrased_id, mode=mode)
        mapping[original_id].append((rephrased_id, data))
    
    mode_str = "perturbed" if mode == 'perturbed' else "rephrased"
    logger.info(f"Built {mode_str} mapping: {len(mapping)} original IDs, "
                f"{len(rephrased_data)} total {mode_str} entries")
    
    # Log sample mapping
    if mapping:
        sample_id = list(mapping.keys())[0]
        sample_count = len(mapping[sample_id])
        sample_ids = [rid for rid, _ in mapping[sample_id][:3]]
        logger.info(f"Sample: {sample_id} has {sample_count} {mode_str} variants")
        logger.info(f"  Sample IDs: {sample_ids}")
    
    for entries in mapping.values():
        entries.sort(key=lambda pair: str(pair[0]))
    return dict(mapping)


def validate_okvqa_inputs(vanilla_data, accuracy_dict, rephrased_mapping, mode,
                           k_low_t, subsample_high_t, high_t_per_variant=10):
    expected_ids = set(vanilla_data)
    if expected_ids != set(accuracy_dict) or expected_ids != set(rephrased_mapping):
        raise ValueError(
            'Vanilla, accuracy, and perturbed generation artifacts must contain '
            'exactly the same original question IDs.'
        )

    variants_per_question = 35 if mode == 'perturbed' else 5
    for qid in sorted(expected_ids):
        low_responses = vanilla_data[qid].get('low_temp_responses', [])
        if len(low_responses) != k_low_t:
            raise ValueError(
                f'Vanilla question {qid} has {len(low_responses)} low-temperature '
                f'responses; expected exactly {k_low_t}.'
            )
        for index, response in enumerate(low_responses):
            if (not isinstance(response, (list, tuple)) or len(response) < 2
                    or not isinstance(response[0], str) or not response[0].strip()):
                raise ValueError(f'Vanilla question {qid} has invalid low-T response {index}.')

        variants = rephrased_mapping[qid]
        if len(variants) != variants_per_question:
            raise ValueError(
                f'Question {qid} has {len(variants)} perturbed/rephrased variants; '
                f'expected {variants_per_question}.'
            )
        if mode == 'perturbed':
            rephrase_counts = defaultdict(int)
            for variant_id, _ in variants:
                rephrase_counts[extract_rephrasing_number(variant_id)] += 1
            if rephrase_counts != {index: 7 for index in range(1, 6)}:
                raise ValueError(
                    f'Question {qid} must have seven image perturbations for each of '
                    f'five rephrasings; found {dict(rephrase_counts)}.'
                )
        for variant_id, variant in variants:
            responses = variant.get('responses', [])
            if len(responses) != high_t_per_variant:
                raise ValueError(
                    f'Variant {variant_id} has {len(responses)} high-T responses; '
                    f'expected {high_t_per_variant}.'
                )
            for index, response in enumerate(responses):
                if (not isinstance(response, (list, tuple)) or len(response) < 2
                        or not isinstance(response[0], str) or not response[0].strip()):
                    raise ValueError(f'Variant {variant_id} has invalid high-T response {index}.')

        available_high_t = variants_per_question * high_t_per_variant
        if not 0 < subsample_high_t <= available_high_t:
            raise ValueError(
                f'--subsample_high_t must be in [1, {available_high_t}], '
                f'got {subsample_high_t}.'
            )


def extract_low_t_samples_from_vanilla(vanilla_data, qid):
    """
    Extract low-T samples (T=0.1) from vanilla pkl's low_temp_responses.
    
    These are the 3 low-temperature samples generated at T=0.1.
    
    Returns:
        list of response strings
    """
    low_t_samples = []
    
    if qid in vanilla_data:
        vanilla_gen = vanilla_data[qid]
        
        # Get low_temp_responses from vanilla (T=0.1 samples)
        low_temp_responses = vanilla_gen.get('low_temp_responses', [])
        for resp_tuple in low_temp_responses:
            if resp_tuple and len(resp_tuple) >= 1:
                resp = resp_tuple[0]  # First element is the response text
                if resp and resp.strip():
                    low_t_samples.append(resp)
    
    return low_t_samples


def extract_low_t_samples_from_mapping(low_temp_mapping, original_id, mode='rephrased'):
    """
    Extract low-T samples for an original question ID from a mapped low-temp run.

    This supports separate low-temp runs where IDs are rephrased/perturbed
    variants (for example: {qid}_rephrased1, {qid}_rephrased2, ...).

    Sampling source preference per entry:
      1) low_temp_responses (if present)
      2) responses

    Returns:
        list of response strings
    """
    low_t_samples = []

    if original_id not in low_temp_mapping:
        return low_t_samples

    for rephrased_id, gen in low_temp_mapping[original_id]:
        sample_field = 'low_temp_responses' if 'low_temp_responses' in gen else 'responses'
        samples = gen.get(sample_field, [])

        for resp_tuple in samples:
            if resp_tuple and len(resp_tuple) >= 1:
                resp = resp_tuple[0]
                if resp and resp.strip():
                    low_t_samples.append(resp)

    return low_t_samples


def extract_rephrasing_number(rephrased_id):
    """
    Extract the rephrasing number from a rephrased or perturbed ID.
    
    Examples:
        '262148000_rephrased1' -> 1
        '262148000_rephrased3_contrast1' -> 3
        '131089000_rephrased5_blur1' -> 5
    
    Returns:
        int: The rephrasing number (1-5), or 0 if not found
    """
    match = re.search(r'_rephrased(\d+)', str(rephrased_id))
    if match:
        return int(match.group(1))
    return 0


def extract_high_t_samples_from_rephrased_mapping(rephrased_mapping, original_id, mode='rephrased'):
    """
    Extract all high-T samples (T=1.0) for an original question ID from rephrased/perturbed pkl.
    
    Collects responses from all rephrased variants (e.g., _rephrased1 through _rephrased5).
    
    For 'rephrased' mode:
        - Each rephrased variant gets a sequential para_idx (0, 1, 2, ...)
        - 5 rephrased × 10 samples = 50 total samples
        
    For 'perturbed' mode:
        - Groups by rephrasing NUMBER, not by full perturbed ID
        - All image perturbations of the same text rephrasing get the SAME para_idx
        - e.g., rephrased1_contrast1, rephrased1_blur1, ... all get para_idx=0
        - 5 rephrasings × 7 perturbations × 10 samples = 350 total samples, 5 unique para_idx
    
    Returns:
        tuple (list of response strings, list of paraphrase indices)
    """
    high_t_samples = []
    paraphrase_indices = []
    
    if original_id in rephrased_mapping:
        if mode == 'perturbed':
            # Group by rephrasing number (1-5) for perturbed mode
            for rephrased_id, gen in rephrased_mapping[original_id]:
                # Extract rephrasing number and use as para_idx (0-indexed)
                rephrase_num = extract_rephrasing_number(rephrased_id)
                para_idx = rephrase_num - 1 if rephrase_num > 0 else 0
                
                responses = gen.get('responses', [])
                for resp_tuple in responses:
                    if resp_tuple and len(resp_tuple) >= 1:
                        resp = resp_tuple[0]  # First element is the response text
                        if resp and resp.strip():
                            high_t_samples.append(resp)
                            paraphrase_indices.append(para_idx)
        else:
            # Original behavior for rephrased mode - sequential indexing
            for para_idx, (rephrased_id, gen) in enumerate(rephrased_mapping[original_id]):
                responses = gen.get('responses', [])
                for resp_tuple in responses:
                    if resp_tuple and len(resp_tuple) >= 1:
                        resp = resp_tuple[0]  # First element is the response text
                        if resp and resp.strip():
                            high_t_samples.append(resp)
                            paraphrase_indices.append(para_idx)
    
    return high_t_samples, paraphrase_indices


def save_results_npz(results, filepath):
    """Save results as compressed numpy archive."""
    save_dict = {}
    question_ids = sorted(results.keys())
    
    for qid in question_ids:
        data = results[qid]
        prefix = f"q_{qid}_"
        save_dict[f"{prefix}probs_fwd"] = data['probs_fwd']
        save_dict[f"{prefix}probs_bwd"] = data['probs_bwd']
        save_dict[f"{prefix}low_texts"] = np.array(data['low_texts'], dtype=object)
        save_dict[f"{prefix}high_texts"] = np.array(data['high_texts'], dtype=object)
        save_dict[f"{prefix}paraphrase_indices"] = np.array(data.get('paraphrase_indices', []), dtype=np.int32)
        save_dict[f"{prefix}m"] = np.array([data['m']])
        save_dict[f"{prefix}n"] = np.array([data['n']])
        if data.get('accuracy') is not None:
            save_dict[f"{prefix}accuracy"] = np.array([data['accuracy']])
    
    save_dict['question_ids'] = np.array(question_ids, dtype=object)
    save_dict['schema_version'] = np.array([2], dtype=np.int16)
    np.savez_compressed(filepath, **save_dict)


def load_results_npz(filepath):
    """Load results from compressed numpy archive."""
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


def compute_entailment_probs_matrix(scorer, texts_a, texts_b):
    """
    Compute m x n x 3 probability matrix for entailment between two lists of texts.
    
    Uses EntailmentScorer.pairwise_probs_matrix for efficient batched inference.
    
    Returns:
        probs_fwd: (m, n, 3) array - P(a -> b) for each pair
        probs_bwd: (m, n, 3) array - P(b -> a) for each pair
    
    Dimensions: [0] = contradiction, [1] = neutral, [2] = entailment
    """
    # Forward: P(a entails b) - premise=a, hypothesis=b
    probs_fwd = scorer.pairwise_probs_matrix(texts_a, texts_b, direction="left_to_right")
    
    # Backward: P(b entails a) - premise=b, hypothesis=a
    probs_bwd = scorer.pairwise_probs_matrix(texts_a, texts_b, direction="right_to_left")
    
    return probs_fwd, probs_bwd


def precompute_rephrased_entailments(
    accuracy_dict,
    vanilla_data,
    rephrased_mapping,
    scorer,
    k_low_t=3,
    subsample_high_t=50,
    subsample_seed=0,
    mode='rephrased',
    low_temp_data=None,
    low_temp_mapping=None
):
    """
    Precompute entailment probabilities for rephrased VQA runs.
    
    Uses low_temp_responses (T=0.1) from vanilla pkl as low-T
    and responses (T=1.0) from rephrased pkl as high-T.
    
    Args:
        accuracy_dict: Dict mapping question ID to accuracy (0.0 to 1.0)
        vanilla_data: Dict from vanilla pkl file (for low-T samples)
        rephrased_mapping: Dict mapping original_id -> list of rephrased data
        scorer: EntailmentScorer instance
        k_low_t: Number of low-T samples to use (default 3)
        subsample_high_t: Number of high-T samples to use (default 50)
        subsample_seed: Random seed for subsampling
        low_temp_data: Optional dict from a separate low-temp pkl (if provided,
            low-T samples are extracted from this instead of vanilla_data)
        low_temp_mapping: Optional mapping original_id -> list[(variant_id, data)]
            built from low_temp_data when that pkl is keyed by rephrased/perturbed IDs
    """
    results = {}
    
    # Source for low-T IDs: mapped low-temp IDs when available, else raw low-temp/vanilla keys
    if low_temp_mapping is not None:
        low_t_ids = set(low_temp_mapping.keys())
    else:
        low_t_source = low_temp_data if low_temp_data is not None else vanilla_data
        low_t_ids = set(low_t_source.keys())

    # Find common IDs across accuracy, low-T source, and rephrased (using original IDs)
    common_ids = set(accuracy_dict.keys()) & low_t_ids & set(rephrased_mapping.keys())
    logger.info(f"Found {len(common_ids)} common question IDs across accuracy, low-T source, and rephrased data")
    
    # Debug: print sample IDs from each source
    if len(common_ids) == 0:
        logger.warning("DEBUG: Sample IDs from each source:")
        logger.warning(f"  Accuracy dict sample: {list(accuracy_dict.keys())[:5]}")
        if low_temp_mapping is not None:
            logger.warning(f"  Low-T mapped IDs sample: {list(low_t_ids)[:5]}")
        else:
            logger.warning(f"  Low-T source sample: {list(low_t_ids)[:5]}")
        logger.warning(f"  Rephrased mapping sample: {list(rephrased_mapping.keys())[:5]}")
    
    skipped_no_low_t = 0
    skipped_no_high_t = 0
    
    for qid in tqdm(sorted(common_ids), desc="Precomputing rephrased entailments"):
        # Seed for reproducibility
        try:
            base = int(qid.replace('-', '')[:8], 16) if '-' in qid else int(
                hashlib.sha256(qid.encode('utf-8')).hexdigest()[:8], 16
            )
        except (TypeError, ValueError):
            base = int(hashlib.sha256(str(qid).encode('utf-8')).hexdigest()[:8], 16)
        rnd = random.Random(subsample_seed + base)
        
        # Get low-T texts from mapped low-temp variants or vanilla low_temp_responses
        if low_temp_mapping is not None:
            low_t_all = extract_low_t_samples_from_mapping(low_temp_mapping, qid, mode=mode)
        else:
            low_t_source = low_temp_data if low_temp_data is not None else vanilla_data
            low_t_all = extract_low_t_samples_from_vanilla(low_t_source, qid)
        
        if len(low_t_all) == 0:
            raise ValueError(f'Question {qid} has no usable low-temperature responses.')
        
        # Subsample low-T
        if len(low_t_all) > k_low_t:
            low_texts = rnd.sample(low_t_all, k_low_t)
        else:
            low_texts = low_t_all[:k_low_t]
        
        # Get high-T texts from all rephrased variants (T=1.0)
        high_t_all, para_indices_all = extract_high_t_samples_from_rephrased_mapping(rephrased_mapping, qid, mode=mode)
        
        if len(high_t_all) == 0:
            raise ValueError(f'Question {qid} has no usable perturbed responses.')
        
        # Subsample high-T (with their paraphrase indices)
        if len(high_t_all) > subsample_high_t:
            # Sample indices to keep paraphrase mapping
            all_indices = list(range(len(high_t_all)))
            sampled_indices = rnd.sample(all_indices, subsample_high_t)
            high_texts = [high_t_all[i] for i in sampled_indices]
            paraphrase_indices = [para_indices_all[i] for i in sampled_indices]
        else:
            high_texts = high_t_all
            paraphrase_indices = para_indices_all
        
        # Filter empty strings (keeping paraphrase indices in sync)
        low_texts = [t for t in low_texts if t.strip()]
        filtered_high = [(t, idx) for t, idx in zip(high_texts, paraphrase_indices) if t.strip()]
        if filtered_high:
            high_texts, paraphrase_indices = zip(*filtered_high)
            high_texts = list(high_texts)
            paraphrase_indices = list(paraphrase_indices)
        else:
            high_texts = []
            paraphrase_indices = []
        
        if len(low_texts) != k_low_t or len(high_texts) != subsample_high_t:
            raise ValueError(f'Validated response counts changed while processing question {qid}.')
        
        # Compute entailment probabilities
        probs_fwd, probs_bwd = compute_entailment_probs_matrix(
            scorer, low_texts, high_texts
        )
        
        results[qid] = {
            'probs_fwd': probs_fwd,
            'probs_bwd': probs_bwd,
            'low_texts': low_texts,
            'high_texts': high_texts,
            'paraphrase_indices': paraphrase_indices,
            'accuracy': accuracy_dict.get(qid),
            'm': len(low_texts),
            'n': len(high_texts)
        }
    
    if low_temp_mapping is not None:
        logger.info(f"Skipped {skipped_no_low_t} questions with no low-T samples (mapped low-temp responses)")
    else:
        logger.info(f"Skipped {skipped_no_low_t} questions with no low-T samples (vanilla low_temp_responses)")
    logger.info(f"Skipped {skipped_no_high_t} questions with no high-T samples (rephrased responses)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Precompute entailment probabilities for VQA')
    
    parser.add_argument('--accuracy_file', type=str, default=None,
                       help='Path to accuracy JSON file (optional). If not provided, accuracy is extracted from vanilla pkl.')
    parser.add_argument('--vanilla_pkl', type=str, required=True,
                       help='Path to vanilla validation_generations.pkl (for accuracy labels)')
    parser.add_argument('--low_temp_pkl', type=str, default=None,
                       help='Path to a separate low-temp validation_generations.pkl (for low-T samples). '
                           'If provided, low-T samples are taken from this pkl instead of vanilla_pkl. '
                           'Supports input-sampled runs keyed by rephrased/perturbed IDs and aggregates '
                           'samples per original question.')
    parser.add_argument('--rephrased_pkl', type=str, required=True,
                       help='Path to rephrased validation_generations.pkl (for high-T samples)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file path (.npz)')
    parser.add_argument('--mode', type=str, default='rephrased', choices=['rephrased', 'perturbed'],
                       help='Mode: rephrased (vanilla low-T + rephrased high-T) or perturbed (vanilla low-T + perturbed high-T)')
    
    parser.add_argument('--k_low_t', type=int, default=3,
                       help='Number of low-T answers to use (default 3)')
    parser.add_argument('--subsample_high_t', type=int, default=50,
                       help='Number of high-T answers to randomly sample. For rephrased: 5×10=50. For perturbed: 5×7×10=350 available.')
    parser.add_argument('--subsample_seed', type=int, default=0,
                       help='Random seed for subsampling')
    
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for entailment scorer')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 inference')
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PRECOMPUTE ENTAILMENTS ({args.mode.upper()})")
    logger.info(f"Accuracy file: {args.accuracy_file or '(from vanilla pkl)'}")
    logger.info(f"Vanilla pkl (accuracy): {args.vanilla_pkl}")
    if args.low_temp_pkl:
        logger.info(f"Low-temp pkl (low-T samples): {args.low_temp_pkl}")
    logger.info(f"Rephrased pkl (high-T): {args.rephrased_pkl}")
    logger.info(f"Output: {args.output_file}")
    logger.info(f"k_low_t={args.k_low_t}, subsample_high_t={args.subsample_high_t}")
    logger.info(f"{'='*60}\n")
    
    # Load accuracy data
    if args.accuracy_file:
        logger.info("Loading accuracy from JSON file...")
        accuracy_dict = load_accuracy_json(args.accuracy_file)
    else:
        logger.info("No accuracy file provided. Will extract from vanilla pkl after loading.")
        accuracy_dict = None
    
    # Load vanilla pkl (for low-T samples)
    logger.info("Loading vanilla pkl (for accuracy labels)...")
    vanilla_data = load_pkl_file(args.vanilla_pkl)
    
    # If no accuracy JSON, extract from vanilla pkl
    if accuracy_dict is None:
        accuracy_dict = extract_accuracy_from_pkl(vanilla_data)
    
    # Load low-temp pkl if provided (for low-T samples)
    low_temp_data = None
    low_temp_mapping = None
    if args.low_temp_pkl:
        logger.info("Loading low-temp pkl (for low-T samples)...")
        low_temp_data = load_pkl_file(args.low_temp_pkl)

        # Support low-temp runs keyed by rephrased/perturbed IDs by building
        # original_id -> variant mapping.
        logger.info(f"Building low-temp mapping using mode={args.mode}...")
        low_temp_mapping = build_rephrased_mapping(low_temp_data, mode=args.mode)

    # Load rephrased pkl (for high-T samples)
    logger.info("Loading rephrased pkl (for high-T samples)...")
    rephrased_data = load_pkl_file(args.rephrased_pkl)
    
    # Build mapping from original ID to rephrased/perturbed entries
    logger.info(f"Building {args.mode} mapping (original_id -> entries)...")
    rephrased_mapping = build_rephrased_mapping(rephrased_data, mode=args.mode)

    if low_temp_mapping is None:
        validate_okvqa_inputs(
            vanilla_data,
            accuracy_dict,
            rephrased_mapping,
            args.mode,
            args.k_low_t,
            args.subsample_high_t,
        )
    
    # Initialize entailment scorer (batched inference)
    logger.info("Loading EntailmentScorer...")
    scorer = EntailmentScorer(batch_size=args.batch_size, fp16=args.fp16)
    
    # Precompute based on mode
    if args.mode in ('rephrased', 'perturbed'):
        # Both modes use the same function: vanilla low-T + rephrased/perturbed high-T
        # The difference is how paraphrase indices are assigned (perturbed groups by rephrasing number)
        results = precompute_rephrased_entailments(
            accuracy_dict,
            vanilla_data,
            rephrased_mapping,
            scorer,
            k_low_t=args.k_low_t,
            subsample_high_t=args.subsample_high_t,
            subsample_seed=args.subsample_seed,
            mode=args.mode,
            low_temp_data=low_temp_data,
            low_temp_mapping=low_temp_mapping
        )
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_results_npz(results, output_path)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Saved precomputed entailments: {output_path}")
    logger.info(f"Total questions: {len(results)}")
    logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Print sample statistics
    if results:
        sample_id = list(results.keys())[0]
        sample = results[sample_id]
        logger.info(f"Sample shape: probs_fwd={sample['probs_fwd'].shape}, probs_bwd={sample['probs_bwd'].shape}")
        logger.info(f"Sample m={sample['m']}, n={sample['n']}")
    
    logger.info(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
