#!/usr/bin/env python3
"""
Compute DeBERTa entailment between vanilla samples and rephrased high-T samples.

For each of 400 original questions:
- 1 vanilla answer (low-T) from generate_answers
- 50 rephrased answers (5 rephrasings × 10 high-T samples each) from generate_rephrased_answers

Performs bidirectional entailment between vanilla and each of the 50 rephrased samples.
Then computes PMFs, KL divergence, uncertainty, and AUROC.

Uncertainty Measure:
- Observed PMF: [P(not_equiv), P(equiv)] from entailment results
- Ideal PMF: [0, 1] - expects all rephrased to be equivalent to vanilla
- Uncertainty = KL(ideal || observed)
- Higher uncertainty indicates more disagreement between vanilla and rephrased answers

Usage:
    python compute_vanilla_rephrased_entailment.py \
        --vanilla_run_dir run-20251115_020439-0cppii9o \
        --rephrased_run_dir run-20251117_033534-w9j9ukin \
        --dataset bioasq \
        --model_name Meta-Llama-3.1-8B-Instruct \
        --output_dir ./entailment_results
"""

import os
import sys
import re
import pickle
import argparse
import logging
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import random
import hashlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Use shared entailment implementation (provides `check_implication`, `get_similarity_score`)
from snne.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta


def load_pickle(filepath):
    """Load pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, filepath):
    """Save data to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_vanilla_generations(wandb_run_dir):
    """
    Load vanilla generations from generate_answers run.
    
    Returns:
        dict: {question_id: {'most_likely_answer': {...}, 'question': str, 'context': str, 'reference': list}}
    """
    validation_pkl = Path(wandb_run_dir) / "files" / "validation_generations.pkl"
    
    if not validation_pkl.exists():
        raise FileNotFoundError(f"Vanilla generations not found: {validation_pkl}")
    
    logger.info(f"Loading vanilla generations from: {validation_pkl}")
    generations = load_pickle(validation_pkl)
    logger.info(f"Loaded {len(generations)} vanilla samples")
    
    return generations


def load_rephrased_generations(wandb_run_dir):
    """
    Load rephrased generations from generate_rephrased_answers run.
    
    Returns:
        dict: {rephrased_id: {'original_id': str, 'rephrase_idx': int, 
                              'most_likely_answer': {...}, 'responses': list, 
                              'question': str, 'original_question': str}}
    """
    p = Path(wandb_run_dir)

    # Support either: (a) direct path to a .pkl file, or (b) a wandb run directory
    if p.is_file() and p.suffix == '.pkl':
        validation_pkl = p
    else:
        validation_pkl = p / "files" / "validation_generations.pkl"

    if not validation_pkl.exists():
        raise FileNotFoundError(f"Rephrased generations not found: {validation_pkl}")

    logger.info(f"Loading rephrased generations from: {validation_pkl}")
    generations = load_pickle(validation_pkl)
    logger.info(f"Loaded {len(generations)} rephrased samples")

    return generations


def organize_rephrased_by_original(rephrased_generations):
    """
    Organize rephrased samples by original question ID.
    
    Returns:
        dict: {original_id: [list of rephrased sample dicts]}
    """
    organized = defaultdict(list)

    for rephrased_id, data in rephrased_generations.items():
        # Prefer explicit original_id stored in the rephrased entry
        orig_id = None
        if isinstance(data, dict) and 'original_id' in data and data['original_id'] is not None:
            orig_id = str(data['original_id'])
        else:
            # Fallback: parse original id from the rephrased_id string (format: "<origid>_...")
            if isinstance(rephrased_id, str) and '_' in rephrased_id:
                orig_id = rephrased_id.split('_', 1)[0]
            else:
                # last-resort: extract leading digits
                m = re.match(r"^(\d+)", str(rephrased_id))
                orig_id = m.group(1) if m else str(rephrased_id)

        organized[orig_id].append({
            'rephrased_id': rephrased_id,
            'rephrase_idx': data.get('rephrase_idx') if isinstance(data, dict) else None,
            'question': data.get('question') if isinstance(data, dict) else None,
            'responses': data.get('responses') if isinstance(data, dict) else None
        })

    return organized


def perform_entailment_analysis(vanilla_gens, rephrased_gens, output_dir, gemini_labels=None, strict_entailment=False, subsample=50, subsample_seed=0):
    """
    Perform entailment between vanilla and rephrased samples.
    
    For each original question:
    - Compare vanilla answer with all high-T rephrased answers
    - Store entailment results
    """
    # Initialize DeBERTa model
    entailment_model = EntailmentDeberta()
    
    # Organize rephrased samples by original ID
    rephrased_by_original = organize_rephrased_by_original(rephrased_gens)
    
    # Results storage
    entailment_results = {}
    # Store both aggregated totals (for backward compatibility) and per-vanilla counts
    equivalence_counts = defaultdict(lambda: {
        'equivalent': 0,
        'not_equivalent': 0,
        'per_vanilla_counts': []  # list of {'equivalent': int, 'not_equivalent': int}
    })
    
    # Find common IDs
    vanilla_ids = set(vanilla_gens.keys())
    rephrased_original_ids = set(rephrased_by_original.keys())
    common_ids = vanilla_ids & rephrased_original_ids
    
    logger.info(f"Found {len(common_ids)} common question IDs")
    logger.info(f"Vanilla only: {len(vanilla_ids - common_ids)}")
    logger.info(f"Rephrased only: {len(rephrased_original_ids - common_ids)}")
    
    if len(common_ids) == 0:
        logger.error("No common IDs found! Check that original_id in rephrased matches vanilla IDs.")
        return None, None
    
    # Process each original question
    for original_id in tqdm(sorted(common_ids), desc="Processing questions"):
        # Robust vanilla lookup: keys may be strings or ints; try several options
        vanilla_data = None
        if original_id in vanilla_gens:
            vanilla_data = vanilla_gens[original_id]
        else:
            # try int key
            try:
                int_key = int(original_id)
                if int_key in vanilla_gens:
                    vanilla_data = vanilla_gens[int_key]
            except Exception:
                vanilla_data = None

        if vanilla_data is None:
            # fallback: try stringified keys
            for k in vanilla_gens.keys():
                if str(k) == str(original_id):
                    vanilla_data = vanilla_gens[k]
                    break

        if vanilla_data is None:
            logger.warning(f"Vanilla data missing for id {original_id}; skipping")
            continue

        # Extract up to 3 vanilla answers (if only one exists, it will be repeated)
        def get_top_k_vanilla_answers(vdata, k=3):
            answers = []
            # Primary most_likely_answer
            mla = vdata.get('most_likely_answer') if isinstance(vdata, dict) else None
            if mla and isinstance(mla, dict) and mla.get('response') is not None:
                answers.append({'response': mla.get('response'), 'accuracy': mla.get('accuracy')})

            # If there is a 'responses' list like rephrased entries, extract their first element
            resp_list = vdata.get('responses') if isinstance(vdata, dict) else None
            if resp_list and isinstance(resp_list, (list, tuple)):
                for tup in resp_list:
                    try:
                        # response tuple shape may be (answer, ...)
                        ans = tup[0]
                    except Exception:
                        ans = None
                    if ans is not None and all(a['response'] != ans for a in answers):
                        answers.append({'response': ans, 'accuracy': None})
                    if len(answers) >= k:
                        break

            # Fallback: sometimes vanilla data stores multiple candidates under 'candidates' or 'generations'
            for key in ('candidates', 'generations', 'other_answers'):
                if len(answers) >= k:
                    break
                cand = vdata.get(key) if isinstance(vdata, dict) else None
                if cand and isinstance(cand, (list, tuple)):
                    for item in cand:
                        if isinstance(item, dict) and item.get('response') is not None:
                            if all(a['response'] != item['response'] for a in answers):
                                answers.append({'response': item['response'], 'accuracy': item.get('accuracy')})
                        else:
                            try:
                                candidate_ans = item[0]
                            except Exception:
                                candidate_ans = None
                            if candidate_ans is not None and all(a['response'] != candidate_ans for a in answers):
                                answers.append({'response': candidate_ans, 'accuracy': None})
                        if len(answers) >= k:
                            break

            # If still not enough, duplicate the last available answer to have k entries
            if len(answers) == 0:
                answers = [{'response': None, 'accuracy': None} for _ in range(k)]
            while len(answers) < k:
                answers.append(answers[-1].copy())

            return answers[:k]

        vanilla_answers = get_top_k_vanilla_answers(vanilla_data, k=3)

        # Decide vanilla_accuracy (use the primary most_likely_answer accuracy if available)
        if gemini_labels is not None:
            vanilla_accuracy = gemini_labels.get(str(original_id))
        else:
            vanilla_accuracy = vanilla_answers[0].get('accuracy')
        
        # Get all rephrased versions of this question
        rephrased_list = rephrased_by_original[original_id]

        # Flatten all rephrased samples (each response tuple) so we can subsample across the full set
        all_samples = []
        for rephrased_item in rephrased_list:
            ridx = rephrased_item.get('rephrase_idx')
            for sample_idx, response_tuple in enumerate(rephrased_item.get('responses', [])):
                all_samples.append({
                    'rephrase_idx': ridx,
                    'sample_idx': sample_idx,
                    'response_tuple': response_tuple,
                    'rephrased_question': rephrased_item.get('question')
                })

        total_available = len(all_samples)
        use_k = subsample if subsample is not None else total_available
        if use_k > total_available:
            logger.warning(f"Requested subsample {use_k} > available {total_available}; using all available samples")
            use_k = total_available

        # Deterministic per-question sampling: combine the provided seed with a stable hash of original_id
        if use_k < total_available:
            try:
                base = int(str(original_id))
            except Exception:
                # stable integer from md5 of original_id
                base = int(hashlib.md5(str(original_id).encode()).hexdigest()[:8], 16)
            seed = int(subsample_seed) + int(base)
            rnd = random.Random(seed)
            selected_samples = rnd.sample(all_samples, k=use_k)
        else:
            selected_samples = all_samples
        
        # Store detailed results for this question. Keep per-vanilla answers and per-comparison flags.
        question_results = {
            'vanilla_answers': vanilla_answers,
            'vanilla_accuracy': vanilla_accuracy,
            'question': vanilla_data.get('question'),
            'reference': vanilla_data.get('reference', []),
            'rephrased_comparisons': []  # each entry will include per-vanilla equivalence list
        }

        # Initialize per-vanilla counts
        per_v_counts = [{'equivalent': 0, 'not_equivalent': 0} for _ in vanilla_answers]

        # Compare vanilla answers with each selected rephrased high-T sample (subsampled if requested)
        for sample in selected_samples:
            rephrase_idx = sample.get('rephrase_idx')
            sample_idx = sample.get('sample_idx')
            response_tuple = sample.get('response_tuple')
            rephrased_question = sample.get('rephrased_question')

            rephrased_answer = None
            try:
                rephrased_answer = response_tuple[0]
            except Exception:
                rephrased_answer = None

            per_v_equiv = []
            # Compare against each vanilla sample
            for v_idx, v_entry in enumerate(vanilla_answers):
                v_ans = v_entry.get('response')
                if v_ans is None or rephrased_answer is None:
                    is_equiv = False
                else:
                    # Use canonical entailment checks
                    implication_1 = entailment_model.check_implication(v_ans, rephrased_answer)
                    implication_2 = entailment_model.check_implication(rephrased_answer, v_ans)
                    if strict_entailment:
                        is_equiv = (implication_1 == 2) and (implication_2 == 2)
                    else:
                        implications = [implication_1, implication_2]
                        # Not contradiction and not both neutral
                        is_equiv = (0 not in implications) and (implications != [1, 1])

                per_v_equiv.append(bool(is_equiv))

                if is_equiv:
                    per_v_counts[v_idx]['equivalent'] += 1
                else:
                    per_v_counts[v_idx]['not_equivalent'] += 1

            # Store result with per-vanilla equivalence booleans
            question_results['rephrased_comparisons'].append({
                'rephrase_idx': rephrase_idx,
                'sample_idx': sample_idx,
                'rephrased_answer': rephrased_answer,
                'per_vanilla_equivalent': per_v_equiv,
                'rephrased_question': rephrased_question
            })

        # Store per-vanilla counts and also aggregated totals for backward compatibility
        equivalence_counts[original_id]['per_vanilla_counts'] = per_v_counts
        total_equiv = sum(c['equivalent'] for c in per_v_counts)
        total_not_equiv = sum(c['not_equivalent'] for c in per_v_counts)
        equivalence_counts[original_id]['equivalent'] = total_equiv
        equivalence_counts[original_id]['not_equivalent'] = total_not_equiv

        # Compute averaged PMF across the vanilla samples and store it
        pmfs = [compute_pmf_from_counts(c['equivalent'], c['not_equivalent']) for c in per_v_counts]
        # elementwise average
        avg_pmf = [float(np.mean([p[i] for p in pmfs])) for i in range(len(pmfs[0]))]
        question_results['pmf_per_vanilla'] = pmfs
        question_results['pmf_avg'] = avg_pmf

        entailment_results[original_id] = question_results
        
        entailment_results[original_id] = question_results
        
        # Log progress every 10 questions
        if len(entailment_results) % 10 == 0:
            # compute average equivalence rate using aggregated totals (backward-compatible)
            rates = []
            for counts in equivalence_counts.values():
                denom = (counts.get('equivalent', 0) + counts.get('not_equivalent', 0))
                if denom > 0:
                    rates.append(counts.get('equivalent', 0) / denom)
            avg_equiv = np.mean(rates) if len(rates) > 0 else 0.0
            logger.info(f"Processed {len(entailment_results)} questions. Avg equivalence rate: {avg_equiv:.3f}")
    
    # Save entailment results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "entailment_results.pkl"
    save_pickle(entailment_results, results_file)
    logger.info(f"Saved entailment results to: {results_file}")
    
    counts_file = output_path / "equivalence_counts.pkl"
    save_pickle(dict(equivalence_counts), counts_file)
    logger.info(f"Saved equivalence counts to: {counts_file}")
    
    return entailment_results, equivalence_counts


def compute_pmf_from_counts(equiv_count, not_equiv_count):
    """
    Compute probability mass function from equivalence counts.
    
    Returns:
        list: [P(not_equivalent), P(equivalent)]
    """
    total = equiv_count + not_equiv_count
    if total == 0:
        return [0.5, 0.5]  # Uniform if no data
    
    return [not_equiv_count / total, equiv_count / total]


def kl_divergence(p, p_star, eps=1e-10):
    """
    Compute KL divergence: KL(p_star || p)
    
    Args:
        p: observed distribution
        p_star: target/ideal distribution
        eps: small value for numerical stability
    """
    p = np.asarray(p, dtype=float)
    p_star = np.asarray(p_star, dtype=float)
    
    return float(np.sum(p_star * np.log((p_star + eps) / (p + eps))))


def quantile_power_normalize(x, gamma=0.5, clip=(1, 99)):
    """
    Normalize values using quantile-based power transformation.
    Higher raw values -> higher normalized scores.
    """
    x = np.asarray(x, dtype=float)
    epsilon = 1e-9
    
    # Invert so higher uncertainty = higher value
    x = 1.0 / (x + epsilon)
    
    if clip is not None and len(x) > 0:
        lo, hi = np.percentile(x, clip)
        x = np.clip(x, lo, hi)
    
    # Convert to quantile ranks
    ranks = np.argsort(np.argsort(x)) + 1
    u = ranks / (len(x) + 1.0)
    
    return u ** gamma


def compute_uncertainty_and_auroc(equivalence_counts, entailment_results, output_dir):
    """
    Compute uncertainty scores and AUROC.
    
    Uncertainty is based on KL divergence from ideal distribution:
    - Ideal distribution: [0, 1] (all rephrased should be equivalent to vanilla)
    - Observed distribution: [P(not_equiv), P(equiv)]
    
    Higher KL = higher uncertainty (more disagreement between vanilla and rephrased)
    """
    question_ids = sorted(equivalence_counts.keys())
    
    uncertainties = []
    accuracies = []
    
    for qid in question_ids:
        counts = equivalence_counts[qid]

        # Get vanilla accuracy
        vanilla_acc = None
        if qid in entailment_results:
            vanilla_acc = entailment_results[qid].get('vanilla_accuracy')
        else:
            logger.warning(f"No entailment results for question {qid}, skipping")
            continue

        if vanilla_acc is None:
            logger.warning(f"No accuracy for question {qid}, skipping")
            continue

        # Prefer averaged PMF stored in entailment_results (pmf_avg). Fall back to aggregated counts.
        if 'pmf_avg' in entailment_results[qid] and entailment_results[qid]['pmf_avg'] is not None:
            pmf_observed = entailment_results[qid]['pmf_avg']
        else:
            equiv = counts.get('equivalent', 0)
            not_equiv = counts.get('not_equivalent', 0)
            pmf_observed = compute_pmf_from_counts(equiv, not_equiv)

        # Define ideal PMF: we expect all rephrased answers to be equivalent to vanilla
        # Ideal = [0, 1] meaning P(not_equiv)=0, P(equiv)=1
        pmf_ideal = [0.0, 1.0]

        # Compute KL divergence as uncertainty: KL(ideal || observed)
        kl = kl_divergence(pmf_observed, pmf_ideal)

        uncertainties.append(kl)
        accuracies.append(vanilla_acc)
    
    uncertainties = np.array(uncertainties)
    accuracies = np.array(accuracies)
    
    logger.info(f"\n{'='*60}")
    logger.info("UNCERTAINTY STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Number of samples: {len(uncertainties)}")
    logger.info(f"Mean KL uncertainty: {np.mean(uncertainties):.4f}")
    logger.info(f"Std KL uncertainty: {np.std(uncertainties):.4f}")
    logger.info(f"Min KL uncertainty: {np.min(uncertainties):.4f}")
    logger.info(f"Max KL uncertainty: {np.max(uncertainties):.4f}")
    logger.info(f"\nVanilla accuracy: {np.mean(accuracies):.4f}")
    
    # Normalize uncertainties for AUROC
    uncertainties_norm = quantile_power_normalize(uncertainties)
    
    # Compute AUROC (higher uncertainty should predict lower accuracy)
    # So we use accuracies as positive class (1 = correct, 0 = incorrect)
    if len(set(accuracies)) > 1:
        # AUROC expects: higher score -> positive class
        # We have: higher uncertainty -> incorrect (negative class)
        # So we use uncertainty directly (already inverted in normalization)
        auroc = roc_auc_score(accuracies, uncertainties_norm)
        
        logger.info(f"\n{'='*60}")
        logger.info("AUROC RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"AUROC: {auroc:.4f}")
        logger.info(f"(Higher uncertainty should correlate with incorrect answers)")
        
        # Additional analysis
        correct_idx = accuracies == 1
        incorrect_idx = accuracies == 0
        
        if np.sum(correct_idx) > 0 and np.sum(incorrect_idx) > 0:
            logger.info(f"\nMean uncertainty (correct answers): {np.mean(uncertainties[correct_idx]):.4f}")
            logger.info(f"Mean uncertainty (incorrect answers): {np.mean(uncertainties[incorrect_idx]):.4f}")
    else:
        logger.warning("Only one class present in accuracies, cannot compute AUROC")
        auroc = None
    
    # Save results
    output_path = Path(output_dir)
    results_dict = {
        'question_ids': question_ids,
        'uncertainties': uncertainties.tolist(),
        'uncertainties_normalized': uncertainties_norm.tolist(),
        'accuracies': accuracies.tolist(),
        'auroc': auroc,
        'mean_uncertainty': float(np.mean(uncertainties)),
        'std_uncertainty': float(np.std(uncertainties)),
        'vanilla_accuracy': float(np.mean(accuracies))
    }
    
    results_file = output_path / "uncertainty_auroc_results.pkl"
    save_pickle(results_dict, results_file)
    logger.info(f"\nSaved uncertainty and AUROC results to: {results_file}")
    
    return results_dict


def main():
    parser = argparse.ArgumentParser(
        description='Compute entailment between vanilla and rephrased samples, then calculate uncertainty and AUROC'
    )
    
    # Input arguments
    parser.add_argument('--vanilla_run_dir', type=str, required=True,
                       help='Path to wandb run directory with vanilla generations (e.g., run-20251115_020439-0cppii9o)')
    parser.add_argument('--rephrased_run_dir', type=str, required=True,
                       help='Path to wandb run directory with rephrased generations (e.g., run-20251117_033534-w9j9ukin)')
    parser.add_argument('--wandb_base_dir', type=str,
                       default='./malaymilindp/uncertainty/wandb',
                       help='Base directory for wandb runs')
    parser.add_argument('--vanilla_gemini_json', type=str, default=None,
                       help='Optional path to JSON with per-example Gemini correctness to use for AUROC')
    parser.add_argument('--strict_entailment', action='store_true',
                       help='If set, equivalence requires strict bidirectional entailment (both directions entail).')
    parser.add_argument('--subsample', type=int, default=50,
                       help='Number of rephrased samples to use per question (<= available, default 50)')
    parser.add_argument('--subsample_seed', type=int, default=0,
                       help='Random seed for subsampling (deterministic)')
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save entailment and uncertainty results')
    
    # Optional metadata
    parser.add_argument('--dataset', type=str, default='bioasq',
                       help='Dataset name (for logging)')
    parser.add_argument('--model_name', type=str, default='Meta-Llama-3.1-8B-Instruct',
                       help='Model name (for logging)')
    
    args = parser.parse_args()
    
    # Construct full paths
    vanilla_full_path = Path(args.wandb_base_dir) / args.vanilla_run_dir

    # `--rephrased_run_dir` may be either:
    #  - a wandb run dir name (relative to --wandb_base_dir), or
    #  - an absolute path to a `.pkl` file produced by the old rephrased generator.
    rephrased_arg = Path(args.rephrased_run_dir)
    if rephrased_arg.is_file() and rephrased_arg.suffix == '.pkl':
        rephrased_full_path = rephrased_arg
    else:
        rephrased_full_path = Path(args.wandb_base_dir) / args.rephrased_run_dir
    
    logger.info(f"\n{'='*60}")
    logger.info("ENTAILMENT AND UNCERTAINTY COMPUTATION")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Vanilla run: {args.vanilla_run_dir}")
    logger.info(f"Rephrased run: {args.rephrased_run_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"{'='*60}\n")
    
    # Load generations
    vanilla_gens = load_vanilla_generations(vanilla_full_path)
    rephrased_gens = load_rephrased_generations(rephrased_full_path)
    gemini_labels = None
    if args.vanilla_gemini_json and os.path.exists(args.vanilla_gemini_json):
        with open(args.vanilla_gemini_json, 'r') as f:
            payload = json.load(f)
        if isinstance(payload, dict) and 'per_example' in payload and isinstance(payload['per_example'], dict):
            raw_map = payload['per_example']
        else:
            raw_map = payload if isinstance(payload, dict) else {}
        gemini_labels = {str(k): int(v) for k, v in raw_map.items()}
    
    # Perform entailment analysis
    logger.info("\nStarting entailment analysis...")
    entailment_results, equivalence_counts = perform_entailment_analysis(
        vanilla_gens,
        rephrased_gens,
        args.output_dir,
        gemini_labels=gemini_labels,
        strict_entailment=args.strict_entailment,
        subsample=args.subsample,
        subsample_seed=args.subsample_seed
    )
    
    if entailment_results is None:
        logger.error("Entailment analysis failed. Exiting.")
        return 1
    
    # Compute uncertainty and AUROC
    logger.info("\nComputing uncertainty and AUROC...")
    uncertainty_results = compute_uncertainty_and_auroc(
        equivalence_counts, entailment_results, args.output_dir
    )

    # Write a one-row CSV summarizing hyperparameters and AUROC results
    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_row = {
            'dataset': args.dataset,
            'model_name': args.model_name,
            'vanilla_run_dir': str(args.vanilla_run_dir),
            'rephrased_run_dir': str(args.rephrased_run_dir),
            'subsample': args.subsample,
            'subsample_seed': args.subsample_seed,
            'strict_entailment': bool(args.strict_entailment),
            'vanilla_gemini_json': args.vanilla_gemini_json,
            'output_dir': str(args.output_dir),
            'auroc': uncertainty_results.get('auroc') if uncertainty_results else None,
            'mean_uncertainty': uncertainty_results.get('mean_uncertainty') if uncertainty_results else None,
            'std_uncertainty': uncertainty_results.get('std_uncertainty') if uncertainty_results else None,
            'vanilla_accuracy': uncertainty_results.get('vanilla_accuracy') if uncertainty_results else None
        }
        df_summary = pd.DataFrame([summary_row])
        csv_name = f"{args.dataset}_{args.model_name}_{args.vanilla_run_dir}_{args.rephrased_run_dir}_uncertainty_auroc_summary.csv"
        csv_path = out_dir / csv_name
        df_summary.to_csv(csv_path, index=False)
        logger.info(f"Saved AUROC summary CSV to: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to write AUROC summary CSV: {e}")

    # If VQA dataset, also write summary CSVs to central results directory
    if args.dataset == 'vqa':
        try:
            central_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'entailment_analysis'
            central_dir.mkdir(parents=True, exist_ok=True)

            # Equivalence counts CSV
            rows = []
            for qid, counts in equivalence_counts.items():
                vanilla_acc = None
                # attempt to get vanilla accuracy
                vdata = vanilla_gens.get(qid) or vanilla_gens.get(int(qid)) if isinstance(qid, str) and qid.isdigit() else vanilla_gens.get(qid)
                if vdata:
                    vanilla_acc = vdata.get('most_likely_answer', {}).get('accuracy')
                rows.append({'question_id': str(qid), 'equivalent': counts['equivalent'], 'not_equivalent': counts['not_equivalent'], 'vanilla_accuracy': vanilla_acc})
            df_counts = pd.DataFrame(rows)
            counts_csv = central_dir / f"{args.vanilla_run_dir}_{args.rephrased_run_dir}_equivalence_counts.csv"
            df_counts.to_csv(counts_csv, index=False)
            logger.info(f"Saved equivalence counts CSV to: {counts_csv}")

            # Uncertainty/AUROC CSV
            if uncertainty_results is not None:
                ur = uncertainty_results
                df_unc = pd.DataFrame({
                    'question_id': ur.get('question_ids', []),
                    'uncertainty': ur.get('uncertainties', []),
                    'uncertainty_normalized': ur.get('uncertainties_normalized', []),
                    'accuracy': ur.get('accuracies', [])
                })
                unc_csv = central_dir / f"{args.vanilla_run_dir}_{args.rephrased_run_dir}_uncertainty_results.csv"
                df_unc.to_csv(unc_csv, index=False)
                logger.info(f"Saved uncertainty results CSV to: {unc_csv}")
        except Exception as e:
            logger.warning(f"Failed to write central CSVs for VQA: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
