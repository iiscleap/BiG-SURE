#!/usr/bin/env python3
"""
Precompute and save entailment probabilities for multilingual vanilla and sampling runs.

This script adapts the SNNE entailment precomputation for multilingual JSON data,
using mDeBERTa-v3-base for cross-lingual NLI.

Data format:
- vanilla: k low-temp responses per question per language
- sampling: n high-temp responses per question per language

Output format:
- Saved as compressed numpy archive (.npz) for efficient storage
- Contains probability arrays and metadata for each (question_id, language) pair

Usage:
    # Vanilla
    python precompute_entailments_multilingual.py \
        --vanilla_file vanilla/generate.json \
        --output_file vanilla_entailments.npz \
        --mode vanilla
    
    # Sampling (using vanilla for low-T)
    python precompute_entailments_multilingual.py \
        --sampling_file sampling/generate.json \
        --vanilla_file vanilla/generate.json \
        --output_file sampling_entailments.npz \
        --mode sampling
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multilingual_utils import MultilingualEntailmentDeberta, LANGUAGES, load_json_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json(filepath):
    """Load either JSON-array or JSONL generation data."""
    return load_json_data(filepath)


def save_results_npz(results, filepath):
    """Save results as compressed numpy archive."""
    save_dict = {}
    keys = sorted(results.keys())
    
    for key in keys:
        qid, lang = key
        prefix = f"q{qid}_lang{lang}"
        save_dict[f"{prefix}_probs_fwd"] = results[key]['probs_fwd']
        save_dict[f"{prefix}_probs_bwd"] = results[key]['probs_bwd']
        save_dict[f"{prefix}_low_texts"] = np.array(results[key]['low_texts'], dtype=object)
        save_dict[f"{prefix}_high_texts"] = np.array(results[key]['high_texts'], dtype=object)
        save_dict[f"{prefix}_paraphrase_indices"] = np.array(
            results[key].get('paraphrase_indices', []), dtype=np.int32
        )
    
    # Save metadata
    save_dict['question_ids'] = np.array([k[0] for k in keys], dtype=object)
    save_dict['languages'] = np.array([k[1] for k in keys], dtype=object)
    save_dict['schema_version'] = np.array([2], dtype=np.int32)
    
    np.savez_compressed(filepath, **save_dict)
    logger.info(f"Saved precomputed entailments to {filepath}")


def extract_text(item):
    """
    Extract text string from various output formats.
    
    Handles:
    - Plain strings
    - Dicts with 'response', 'answer', 'text', or 'output' keys
    - Tuples/lists (takes first element)
    """
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ['response', 'answer', 'text', 'output']:
            if key in item:
                return extract_text(item[key])  # Recursive in case of nested
        return None
    if isinstance(item, (list, tuple)):
        if len(item) > 0:
            return extract_text(item[0])  # Take first element
        return None
    return str(item)  # Fallback: convert to string


def ensure_string_list(texts, name="texts"):
    """
    Ensure all items in a list are strings. 
    Applies extract_text to each item and filters out None values.
    """
    result = []
    for i, t in enumerate(texts):
        text = extract_text(t)
        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            logger.warning(f"{name}[{i}] is not a valid string: {type(t).__name__}, skipping")
            continue
        result.append(text)
    return result


def validate_sampling_inputs(vanilla_data, sampling_data, k_low_t, high_t_per_rephrase):
    """Validate the 4-language, five-rephrase BiG-SURE generation contract."""
    vanilla_lookup = {str(item.get('question_id')): item for item in vanilla_data}
    if len(vanilla_lookup) != len(vanilla_data) or not vanilla_lookup:
        raise ValueError('Vanilla data must contain unique, non-empty question IDs.')

    grouped = {}
    rephrased_ids = []
    for item in sampling_data:
        if item.get('question_id') is None or item.get('original_id') is None:
            raise ValueError('Every rephrased record must contain question_id and original_id.')
        qid = str(item.get('question_id'))
        rephrased_ids.append(qid)
        base_qid = str(item['original_id'])
        grouped.setdefault(base_qid, []).append(item)
    if len(set(rephrased_ids)) != len(rephrased_ids):
        raise ValueError('Rephrased sampling contains duplicate question IDs.')

    if set(vanilla_lookup) != set(grouped):
        missing_groups = sorted(set(vanilla_lookup) - set(grouped))
        extra_groups = sorted(set(grouped) - set(vanilla_lookup))
        raise ValueError(
            'Vanilla and rephrased sampling question IDs do not match. '
            f'Missing: {missing_groups[:3]}; extra: {extra_groups[:3]}.'
        )

    for qid, vanilla_item in vanilla_lookup.items():
        variants = grouped[qid]
        if len(variants) != 5:
            raise ValueError(
                f'Question {qid} has {len(variants)} rephrased records; expected 5.'
            )
        for language in LANGUAGES:
            low_outputs = vanilla_item.get('output', {}).get(language, [])
            if len(low_outputs) != k_low_t + 1:
                raise ValueError(
                    f'Question {qid}/{language} has {len(low_outputs)} vanilla outputs; '
                    f'expected one greedy plus {k_low_t} low-temperature outputs.'
                )
            low_probs = vanilla_item.get('probs', {}).get(language, [])
            if len(low_probs) != k_low_t + 1:
                raise ValueError(f'Question {qid}/{language} has mismatched vanilla probabilities.')
            for variant in variants:
                outputs = variant.get('output', {}).get(language, [])
                if len(outputs) != high_t_per_rephrase + 1:
                    raise ValueError(
                        f'Rephrased question {variant.get("question_id")}/{language} has '
                        f'{len(outputs)} outputs; expected one greedy plus '
                        f'{high_t_per_rephrase} stochastic outputs.'
                    )
                probs = variant.get('probs', {}).get(language, [])
                if len(probs) != high_t_per_rephrase + 1:
                    raise ValueError(
                        f'Rephrased question {variant.get("question_id")}/{language} '
                        'has mismatched probabilities.'
                    )

    return grouped


def compute_entailment_probs_batch(model, texts_a, texts_b, batch_size=32):
    """
    Compute entailment probabilities for all pairs (texts_a[i], texts_b[j]).
    
    IMPORTANT: Both matrices have shape (m, n, 3) to match the original SNNE convention:
    - probs_fwd[i, j] = P(class | texts_a[i] is premise, texts_b[j] is hypothesis)
    - probs_bwd[i, j] = P(class | texts_b[j] is premise, texts_a[i] is hypothesis)
    
    This allows element-wise combination of forward and backward scores.
    
    Returns:
        probs_fwd: (m, n, 3) array - P(class | low[i] -> high[j])
        probs_bwd: (m, n, 3) array - P(class | high[j] -> low[i])
    """
    import torch
    import torch.nn.functional as F
    
    # Ensure all texts are strings
    texts_a = ensure_string_list(texts_a, "texts_a")
    texts_b = ensure_string_list(texts_b, "texts_b")
    
    m = len(texts_a)
    n = len(texts_b)
    
    if m == 0 or n == 0:
        logger.warning(f"Empty text lists after validation: texts_a={m}, texts_b={n}")
        return np.zeros((max(m, 1), max(n, 1), 3), dtype=np.float32), np.zeros((max(m, 1), max(n, 1), 3), dtype=np.float32)
    
    # Forward: texts_a[i] is premise, texts_b[j] is hypothesis
    # probs_fwd[i, j] = P(class | a[i] -> b[j])
    probs_fwd = np.zeros((m, n, 3), dtype=np.float32)
    
    for i in tqdm(range(m), desc="Computing forward entailments", leave=False):
        premise = texts_a[i]
        for j in range(0, n, batch_size):
            batch_hypotheses = texts_b[j:min(j+batch_size, n)]
            
            # Tokenize batch
            inputs = model.tokenizer(
                [premise] * len(batch_hypotheses),
                batch_hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(model.model.device)
            
            with torch.no_grad():
                outputs = model.model(**inputs)
                logits = outputs.logits
                batch_probs = F.softmax(logits, dim=1).cpu().numpy()
            
            # Remap from model's output to our convention
            # Model: 0=entail, 1=neutral, 2=contradict
            # Ours: 0=contradict, 1=neutral, 2=entail
            for k, probs in enumerate(batch_probs):
                idx = j + k
                probs_fwd[i, idx] = [probs[2], probs[1], probs[0]]  # contradict, neutral, entail
    
    # Backward: texts_b[j] is premise, texts_a[i] is hypothesis
    # SNNE convention: probs_bwd is still (m, n, 3) with swapped premise/hypothesis
    # probs_bwd[i, j] = P(class | b[j] -> a[i])
    probs_bwd = np.zeros((m, n, 3), dtype=np.float32)
    
    for i in tqdm(range(m), desc="Computing backward entailments", leave=False):
        hypothesis = texts_a[i]
        for j in range(0, n, batch_size):
            batch_premises = texts_b[j:min(j+batch_size, n)]
            
            inputs = model.tokenizer(
                batch_premises,
                [hypothesis] * len(batch_premises),
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(model.model.device)
            
            with torch.no_grad():
                outputs = model.model(**inputs)
                logits = outputs.logits
                batch_probs = F.softmax(logits, dim=1).cpu().numpy()
            
            # Remap from model's output to our convention
            for k, probs in enumerate(batch_probs):
                idx = j + k
                # Store at [i, idx] so probs_bwd has shape (m, n, 3) - MATCHES SNNE
                probs_bwd[i, idx] = [probs[2], probs[1], probs[0]]
    
    return probs_fwd, probs_bwd


def precompute_vanilla_entailments(data, model, k_low_t=3, batch_size=32):
    """
    Precompute entailments for vanilla (low-temp only) runs.
    
    For each (question_id, language), compute self-entailment among k low-T responses.
    
    Note: Vanilla output structure per language:
      - output[0] = greedy answer (temp=0) - used for accuracy, NOT for entailments
      - output[1:k_low_t+1] = low-T samples (temp=0.1) - used for entailments
    """
    results = {}
    
    logger.info(f"Precomputing vanilla entailments for {len(data)} questions across {len(LANGUAGES)} languages")
    
    for item in tqdm(data, desc="Vanilla precompute"):
        qid = item['question_id']
        
        for lang in LANGUAGES:
            # Extract low-T responses
            # output[0] = greedy (temp=0), output[1:4] = low-T samples (temp=0.1)
            outputs = item['output'].get(lang, [])
            if not outputs or len(outputs) < 2:
                continue
            
            # Skip index 0 (greedy) and take the low-T samples (indices 1 to k_low_t+1)
            low_texts = outputs[1:k_low_t+1]
            
            if len(low_texts) < 2:
                logger.warning(f"Question {qid}, lang {lang}: insufficient low-T responses ({len(low_texts)})")
                continue
            
            # Compute self-entailment
            probs_fwd, probs_bwd = compute_entailment_probs_batch(
                model, low_texts, low_texts, batch_size
            )
            
            results[(qid, lang)] = {
                'probs_fwd': probs_fwd,
                'probs_bwd': probs_bwd,
                'low_texts': low_texts,
                'high_texts': low_texts  # Same as low for vanilla
            }
    
    logger.info(f"Completed {len(results)} vanilla entailment computations")
    return results


def precompute_sampling_entailments(sampling_data, vanilla_data, model, 
                                    k_low_t=3, subsample_high_t=10, 
                                    subsample_seed=0, batch_size=32):
    """
    Precompute entailments for sampling runs.
    
    Uses low-T from vanilla (indices 1:k_low_t+1), high-T from sampling (all samples).
    
    Data structure:
      - Vanilla: output[0] = greedy (temp=0), output[1:4] = low-T samples (temp=0.1)
      - Sampling: output[0:k] = high-T samples (temp=1.0)
    """
    results = {}
    
    vanilla_lookup = {str(item['question_id']): item for item in vanilla_data}

    # Rephrased sampling has five records per original question. Aggregate those
    # records before entailment so each matrix is 3 low-T x 50 high-T and carries
    # the true paraphrase assignment for every high-T column.
    sampling_groups = {}
    for item in sampling_data:
        qid = str(item['question_id'])
        base_qid = str(item.get('original_id', qid.split('_r')[0]))
        sampling_groups.setdefault(base_qid, []).append(item)

    for items in sampling_groups.values():
        items.sort(key=lambda item: (item.get('rephrase_idx', 0), str(item['question_id'])))
    
    logger.info(
        "Precomputing sampling entailments for %d original questions from %d records",
        len(sampling_groups), len(sampling_data)
    )
    
    rng = np.random.RandomState(subsample_seed)
    
    for base_qid, items in tqdm(sampling_groups.items(), desc="Sampling precompute"):
        if base_qid not in vanilla_lookup:
            logger.warning(f"Question {base_qid} not found in vanilla data, skipping")
            continue
        
        vanilla_item = vanilla_lookup[base_qid]
        
        for lang in LANGUAGES:
            # Get low-T from vanilla (skip index 0 = greedy, use indices 1 to k_low_t+1)
            vanilla_outputs = vanilla_item['output'].get(lang, [])
            if not vanilla_outputs or len(vanilla_outputs) < 2:
                continue
            low_texts = vanilla_outputs[1:k_low_t+1]  # indices 1,2,3 for k_low_t=3
            
            high_t_all = []
            paraphrase_indices_all = []
            for para_idx, item in enumerate(items):
                sampling_outputs = item.get('output', {}).get(lang, [])
                # Rephrased generation stores one greedy answer followed by k
                # stochastic answers. Standard sampling stores only stochastic answers.
                if 'original_id' in item:
                    sampling_outputs = sampling_outputs[1:]
                valid_outputs = ensure_string_list(
                    sampling_outputs, f'{base_qid}/{lang}/paraphrase{para_idx}'
                )
                high_t_all.extend(valid_outputs)
                paraphrase_indices_all.extend([para_idx] * len(valid_outputs))

            if not high_t_all:
                continue
            
            # Subsample high-T if needed
            if len(high_t_all) > subsample_high_t:
                indices = sorted(rng.choice(len(high_t_all), subsample_high_t, replace=False))
                high_texts = [high_t_all[i] for i in indices]
                paraphrase_indices = [paraphrase_indices_all[i] for i in indices]
            else:
                high_texts = high_t_all
                paraphrase_indices = paraphrase_indices_all
            
            if len(low_texts) < 1 or len(high_texts) < 1:
                continue
            
            # Compute bipartite entailment
            probs_fwd, probs_bwd = compute_entailment_probs_batch(
                model, low_texts, high_texts, batch_size
            )
            
            results[(base_qid, lang)] = {
                'probs_fwd': probs_fwd,
                'probs_bwd': probs_bwd,
                'low_texts': low_texts,
                'high_texts': high_texts,
                'paraphrase_indices': paraphrase_indices,
            }
    
    logger.info(f"Completed {len(results)} sampling entailment computations")
    return results


def main():
    parser = argparse.ArgumentParser(description="Precompute multilingual entailments")
    parser.add_argument('--check_file', type=str, default=None,
                       help='Exit successfully only if this is a current entailment archive')
    parser.add_argument('--vanilla_file', type=str, help='Path to vanilla JSON file')
    parser.add_argument('--sampling_file', type=str, help='Path to sampling JSON file')
    parser.add_argument('--output_file', type=str, help='Output .npz file')
    parser.add_argument('--mode', type=str, choices=['vanilla', 'sampling'])
    parser.add_argument('--k_low_t', type=int, default=3, help='Number of low-T responses')
    parser.add_argument('--subsample_high_t', type=int, default=10, help='Number of high-T responses')
    parser.add_argument('--samples_per_rephrase', type=int, default=10,
                       help='Expected stochastic outputs after each rephrased greedy output')
    parser.add_argument('--subsample_seed', type=int, default=0, help='Random seed for subsampling')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for entailment computation')
    parser.add_argument('--model_name', type=str, 
                       default='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
                       help='Multilingual NLI model')
    
    args = parser.parse_args()

    if args.check_file:
        try:
            archive = np.load(args.check_file, allow_pickle=True)
            version = int(archive['schema_version'][0]) if 'schema_version' in archive.files else 1
        except Exception as exc:
            logger.error('Invalid entailment archive %s: %s', args.check_file, exc)
            raise SystemExit(1) from exc
        raise SystemExit(0 if version >= 2 else 1)
    
    # Validation
    if not args.mode or not args.output_file:
        parser.error("--mode and --output_file are required for precomputation")
    if args.mode == 'vanilla' and not args.vanilla_file:
        parser.error("--vanilla_file required for vanilla mode")
    if args.mode == 'sampling' and (not args.vanilla_file or not args.sampling_file):
        parser.error("Both --vanilla_file and --sampling_file required for sampling mode")
    
    # Load data
    if args.mode == 'vanilla':
        logger.info(f"Loading vanilla data from {args.vanilla_file}")
        vanilla_data = load_json(args.vanilla_file)
        logger.info("Loading multilingual NLI model...")
        model = MultilingualEntailmentDeberta(args.model_name)
        results = precompute_vanilla_entailments(
            vanilla_data, model,
            k_low_t=args.k_low_t,
            batch_size=args.batch_size
        )
    
    elif args.mode == 'sampling':
        logger.info(f"Loading vanilla data from {args.vanilla_file}")
        vanilla_data = load_json(args.vanilla_file)
        logger.info(f"Loading sampling data from {args.sampling_file}")
        sampling_data = load_json(args.sampling_file)
        validate_sampling_inputs(
            vanilla_data,
            sampling_data,
            k_low_t=args.k_low_t,
            high_t_per_rephrase=args.samples_per_rephrase,
        )
        logger.info("Loading multilingual NLI model...")
        model = MultilingualEntailmentDeberta(args.model_name)
        results = precompute_sampling_entailments(
            sampling_data, vanilla_data, model,
            k_low_t=args.k_low_t,
            subsample_high_t=args.subsample_high_t,
            subsample_seed=args.subsample_seed,
            batch_size=args.batch_size
        )
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    save_results_npz(results, args.output_file)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
