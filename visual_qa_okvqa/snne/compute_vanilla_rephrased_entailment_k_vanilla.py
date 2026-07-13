#!/usr/bin/env python3
"""
Compute DeBERTa entailment between three vanilla samples and rephrased high-T samples.

This is a variant of `compute_vanilla_rephrased_entailment.py` that uses three
vanilla samples per original question instead of one. For each question:
- collect (or synthesize) up to 3 vanilla answers
- perform bidirectional entailment between each vanilla sample and each rephrased sample
- compute a PMF [P(not_equiv), P(equiv)] for each vanilla sample
- average the three PMFs element-wise to produce a single PMF per question
- compute KL uncertainty (ideal=[0,1]) and AUROC as before

If fewer than 3 vanilla answers are available we will duplicate the available
answer(s) to reach 3 samples (with a warning).

Usage is the same as the original script but points to this script instead.
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
import pandas as pd
from sklearn.metrics import roc_auc_score
from snne.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
import random
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_vanilla_generations(wandb_run_dir):
    validation_pkl = Path(wandb_run_dir) / "files" / "validation_generations.pkl"
    if not validation_pkl.exists():
        raise FileNotFoundError(f"Vanilla generations not found: {validation_pkl}")
    logger.info(f"Loading vanilla generations from: {validation_pkl}")
    generations = load_pickle(validation_pkl)
    logger.info(f"Loaded {len(generations)} vanilla samples")
    return generations


def load_rephrased_generations(wandb_run_dir):
    p = Path(wandb_run_dir)
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
    organized = defaultdict(list)

    for rephrased_id, data in rephrased_generations.items():
        orig_id = None
        if isinstance(data, dict) and 'original_id' in data and data['original_id'] is not None:
            orig_id = str(data['original_id'])
        else:
            if isinstance(rephrased_id, str) and '_' in rephrased_id:
                orig_id = rephrased_id.split('_', 1)[0]
            else:
                m = re.match(r"^(\d+)", str(rephrased_id))
                orig_id = m.group(1) if m else str(rephrased_id)

        organized[orig_id].append({
            'rephrased_id': rephrased_id,
            'rephrase_idx': data.get('rephrase_idx') if isinstance(data, dict) else None,
            'question': data.get('question') if isinstance(data, dict) else None,
            'responses': data.get('responses') if isinstance(data, dict) else None
        })

    return organized


def _extract_response_text(item):
    """Helper to extract a text response from different possible container formats."""
    # Possible formats seen in repo: tuple like (answer, ...), dict with 'response', or raw string
    if item is None:
        return None
    if isinstance(item, tuple) or isinstance(item, list):
        if len(item) > 0:
            return item[0]
        return None
    if isinstance(item, dict):
        # common key used in vanilla entries
        if 'response' in item:
            return item.get('response')
        # alternative key
        if 'answer' in item:
            return item.get('answer')
        # if item itself is a text field
        return None
    if isinstance(item, str):
        return item
    return None


def get_k_vanilla_answers(vanilla_data, k=3):
    """Return a list of k vanilla answer strings for a given vanilla_data dict.

    Strategy mirrors the previous helper but is parameterized by k. If fewer
    than k answers are present, answers are duplicated to reach k (with a
    warning).
    """
    answers = []

    # 1) responses list
    if isinstance(vanilla_data.get('responses'), (list, tuple)) and len(vanilla_data.get('responses')) > 0:
        for t in vanilla_data.get('responses')[:k]:
            txt = _extract_response_text(t)
            if txt is not None:
                answers.append(txt)

    # 2) most_likely_answers (plural)
    if len(answers) < k and isinstance(vanilla_data.get('most_likely_answers'), (list, tuple)):
        for x in vanilla_data.get('most_likely_answers')[:k]:
            if isinstance(x, dict) and 'response' in x:
                answers.append(x['response'])
            else:
                txt = _extract_response_text(x)
                if txt is not None:
                    answers.append(txt)

    # 3) most_likely_answer + alternatives
    if len(answers) < k:
        mla = vanilla_data.get('most_likely_answer')
        if isinstance(mla, dict) and 'response' in mla:
            answers.append(mla['response'])
        else:
            txt = _extract_response_text(mla)
            if txt is not None:
                answers.append(txt)

        # try alternative containers
        for key in ['alternatives', 'other_answers', 'topk', 'alt_answers']:
            if len(answers) >= k:
                break
            arr = vanilla_data.get(key)
            if isinstance(arr, (list, tuple)):
                for elem in arr:
                    if len(answers) >= k:
                        break
                    txt = _extract_response_text(elem)
                    if txt is not None:
                        answers.append(txt)

    # If still fewer than k, duplicate last available answer as fallback
    if len(answers) == 0:
        logger.warning("No vanilla answers found in entry; using empty strings")
        answers = [""] * k
    elif len(answers) < k:
        logger.warning(f"Only {len(answers)} vanilla answer(s) found; duplicating to create {k} samples")
        # duplicate last element until reach k
        while len(answers) < k:
            answers.append(answers[-1])
    else:
        answers = answers[:k]

    return answers


def compute_pmf_from_counts(equiv_count, not_equiv_count):
    total = equiv_count + not_equiv_count
    if total == 0:
        return [0.5, 0.5]
    return [not_equiv_count / total, equiv_count / total]


def kl_divergence(p, p_star, eps=1e-10):
    p = np.asarray(p, dtype=float)
    p_star = np.asarray(p_star, dtype=float)
    return float(np.sum(p_star * np.log((p_star + eps) / (p + eps))))


def quantile_power_normalize(x, gamma=0.5, clip=(1, 99)):
    x = np.asarray(x, dtype=float)
    epsilon = 1e-9
    x = 1.0 / (x + epsilon)
    if clip is not None and len(x) > 0:
        lo, hi = np.percentile(x, clip)
        x = np.clip(x, lo, hi)
    ranks = np.argsort(np.argsort(x)) + 1
    u = ranks / (len(x) + 1.0)
    return u ** gamma


def perform_entailment_analysis_kvanilla(vanilla_gens, rephrased_gens, output_dir, k=3, gemini_labels=None, strict_entailment=False, subsample=50, subsample_seed=0):
    entailment_model = EntailmentDeberta()
    rephrased_by_original = organize_rephrased_by_original(rephrased_gens)

    entailment_results = {}
    equivalence_pmfs = {}

    vanilla_ids = set(vanilla_gens.keys()) if isinstance(vanilla_gens, dict) else set()
    rephrased_original_ids = set(rephrased_by_original.keys())

    # If vanilla_gens is empty, we'll extract vanilla samples from the rephrased run.
    if len(vanilla_ids) == 0:
        common_ids = rephrased_original_ids
    else:
        common_ids = vanilla_ids & rephrased_original_ids

    logger.info(f"Found {len(common_ids)} question IDs to process")
    logger.info(f"Vanilla only: {len(vanilla_ids - common_ids)}")
    logger.info(f"Rephrased only: {len(rephrased_original_ids - common_ids)}")

    if len(common_ids) == 0:
        logger.error("No question IDs found to process. Check inputs.")
        return None, None

    for original_id in tqdm(sorted(common_ids), desc="Processing questions"):
        # Robust vanilla lookup if provided
        vanilla_data = None
        if vanilla_ids:
            if original_id in vanilla_gens:
                vanilla_data = vanilla_gens[original_id]
            else:
                try:
                    int_key = int(original_id)
                    if int_key in vanilla_gens:
                        vanilla_data = vanilla_gens[int_key]
                except Exception:
                    vanilla_data = None

            if vanilla_data is None:
                for kkey in vanilla_gens.keys():
                    if str(kkey) == str(original_id):
                        vanilla_data = vanilla_gens[kkey]
                        break

        # If no separate vanilla gens, aggregate most-likely answers from rephrased run
        if vanilla_data is None:
            gathered_ml = []
            for r in rephrased_by_original.get(str(original_id), []):
                rid = r.get('rephrased_id')
                raw_entry = rephrased_gens.get(rid, r)

                mls = raw_entry.get('most_likely_answers') if isinstance(raw_entry, dict) else None
                if isinstance(mls, (list, tuple)) and len(mls) > 0:
                    for x in mls:
                        txt = _extract_response_text(x)
                        acc = None
                        if isinstance(x, dict):
                            acc = x.get('accuracy')
                        if txt is not None:
                            gathered_ml.append({'response': txt, 'accuracy': acc})
                else:
                    mla = raw_entry.get('most_likely_answer') if isinstance(raw_entry, dict) else None
                    txt = _extract_response_text(mla)
                    acc = None
                    if isinstance(mla, dict):
                        acc = mla.get('accuracy')
                    if txt is not None:
                        gathered_ml.append({'response': txt, 'accuracy': acc})

            if len(gathered_ml) == 0:
                logger.warning(f"No most-likely vanilla answers found for original id {original_id}; skipping")
                continue

            # Keep the original dict shape (response + optional accuracy) so downstream
            # code can read `most_likely_answer['accuracy']` when available.
            vanilla_data = {
                'most_likely_answers': gathered_ml,
                'most_likely_answer': gathered_ml[0],
                'question': rephrased_by_original.get(str(original_id), [{}])[0].get('question')
            }

        # Now we have vanilla_data (from separate run or synthesized). Extract k vanilla answers.
        vanilla_answers = get_k_vanilla_answers(vanilla_data, k=k)

        if gemini_labels is not None:
            vanilla_accuracy = gemini_labels.get(str(original_id))
        else:
            vanilla_accuracy = vanilla_data.get('most_likely_answer', {}).get('accuracy', None)

        rephrased_list = rephrased_by_original.get(str(original_id), [])

        # Flatten all rephrased samples for deterministic subsampling
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

        if use_k < total_available:
            try:
                base = int(str(original_id))
            except Exception:
                base = int(hashlib.md5(str(original_id).encode()).hexdigest()[:8], 16)
            seed = int(subsample_seed) + int(base)
            rnd = random.Random(seed)
            selected_samples = rnd.sample(all_samples, k=use_k)
        else:
            selected_samples = all_samples

        question_results = {
            'vanilla_answers': vanilla_answers,
            'vanilla_accuracy': vanilla_accuracy,
            'question': vanilla_data.get('question'),
            'reference': vanilla_data.get('reference', []),
            'rephrased_comparisons': []
        }

        # For each vanilla sample, keep counts
        per_v_equiv_counts = [0] * k
        per_v_not_equiv_counts = [0] * k
        # For per-sample KL aggregation, collect KL per rephrased sample
        per_sample_kls = []

        # Compare each vanilla to the selected rephrased samples (subsampled)
        for sample in selected_samples:
            rephrase_idx = sample.get('rephrase_idx')
            sample_idx = sample.get('sample_idx')
            response_tuple = sample.get('response_tuple')
            rephrased_question = sample.get('rephrased_question')

            rephrased_answer = _extract_response_text(response_tuple)

            equivs = []
            for i, v_ans in enumerate(vanilla_answers):
                if v_ans is None or rephrased_answer is None:
                    is_equiv = False
                else:
                    implication_1 = entailment_model.check_implication(v_ans, rephrased_answer)
                    implication_2 = entailment_model.check_implication(rephrased_answer, v_ans)
                    if strict_entailment:
                        is_equiv = (implication_1 == 2) and (implication_2 == 2)
                    else:
                        implications = [implication_1, implication_2]
                        is_equiv = (0 not in implications) and (implications != [1, 1])

                equivs.append(bool(is_equiv))
                if is_equiv:
                    per_v_equiv_counts[i] += 1
                else:
                    per_v_not_equiv_counts[i] += 1

            question_results['rephrased_comparisons'].append({
                'rephrase_idx': rephrase_idx,
                'sample_idx': sample_idx,
                'rephrased_answer': rephrased_answer,
                'equivalent_by_vanilla': equivs,
                'rephrased_question': rephrased_question
            })

            # Per-sample KL: each sample induces a PMF over [not_equiv, equiv]
            # r = fraction of vanilla answers equivalent to this sample
            try:
                r = float(sum(equivs)) / float(k) if k > 0 else 0.0
            except Exception:
                r = 0.0
            sample_pmf = [1.0 - r, r]
            pmf_ideal = [0.0, 1.0]
            per_sample_kls.append(kl_divergence(sample_pmf, pmf_ideal))

        # compute pmf per vanilla
        pmfs = [compute_pmf_from_counts(eq, neq) for eq, neq in zip(per_v_equiv_counts, per_v_not_equiv_counts)]

        # average pmfs element-wise (strategy A)
        pmfs_arr = np.array(pmfs)
        avg_pmf = list(np.mean(pmfs_arr, axis=0))

        # compute per-vanilla KL uncertainties (strategy B)
        pmf_ideal = [0.0, 1.0]
        per_v_kls = [kl_divergence(p, pmf_ideal) for p in pmfs]

        equivalence_pmfs[str(original_id)] = {
            'per_vanilla_pmfs': pmfs,
            'avg_pmf': avg_pmf,
            'per_v_equiv_counts': per_v_equiv_counts,
            'per_v_not_equiv_counts': per_v_not_equiv_counts,
            'per_v_kl_uncertainties': per_v_kls,
            'mean_kl_across_vanillas': float(np.mean(per_v_kls)) if len(per_v_kls) > 0 else None,
            'per_sample_kl_values': per_sample_kls,
            'per_sample_kl_mean': float(np.mean(per_sample_kls)) if len(per_sample_kls) > 0 else None
        }

        entailment_results[str(original_id)] = question_results

        # Log progress every 10 questions
        if len(entailment_results) % 10 == 0:
            avg_equiv = np.mean([
                (c['equivalent'] / (c['equivalent'] + c['not_equivalent']))
                if (c['equivalent'] + c['not_equivalent']) > 0 else 0.0
                for c in [{'equivalent': sum(per_v_equiv_counts), 'not_equivalent': sum(per_v_not_equiv_counts)}]
            ])
            logger.info(f"Processed {len(entailment_results)} questions.")

    # Save entailment results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"entailment_results_k{k}.pkl"
    save_pickle(entailment_results, results_file)
    logger.info(f"Saved entailment results to: {results_file}")

    pmf_file = output_path / f"equivalence_pmfs_k{k}.pkl"
    save_pickle(equivalence_pmfs, pmf_file)
    logger.info(f"Saved equivalence PMFs to: {pmf_file}")

    return entailment_results, equivalence_pmfs

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"entailment_results_k{k}.pkl"
    save_pickle(entailment_results, results_file)
    logger.info(f"Saved entailment results to: {results_file}")

    pmf_file = output_path / f"equivalence_pmfs_k{k}.pkl"
    save_pickle(equivalence_pmfs, pmf_file)
    logger.info(f"Saved equivalence PMFs to: {pmf_file}")

    return entailment_results, equivalence_pmfs


def compute_uncertainty_and_auroc_from_pmfs(equivalence_pmfs, entailment_results, output_dir):
    """Compute two uncertainty variants:

    A) KL from the averaged PMF per question (avg-PMF -> KL) — this mirrors
       the original averaging-then-KL strategy.
    B) Mean of per-vanilla KLs (KL per vanilla -> average KL) — alternative
       aggregation.

    Both are evaluated with AUROC against vanilla accuracy.
    """
    question_ids = sorted(equivalence_pmfs.keys())

    # Strategy A: KL from averaged PMF
    uncertainties_A = []
    # Strategy B: mean of per-vanilla KLs
    uncertainties_B = []
    # Strategy C: mean of per-sample KLs
    uncertainties_C = []
    accuracies = []

    for qid in question_ids:
        entry = equivalence_pmfs[qid]
        avg_pmf = entry.get('avg_pmf')
        per_v_kls = entry.get('per_v_kl_uncertainties')
        per_sample_mean = entry.get('per_sample_kl_mean')

        if avg_pmf is None:
            logger.warning(f"No avg PMF for question {qid}, skipping")
            continue

        pmf_ideal = [0.0, 1.0]
        kl_A = kl_divergence(avg_pmf, pmf_ideal)
        uncertainties_A.append(kl_A)

        if per_v_kls is None:
            # If per-vanilla KLs missing, fall back to KL of avg_pmf
            uncertainties_B.append(kl_A)
        else:
            uncertainties_B.append(float(np.mean(per_v_kls)))

        # Strategy C: mean per-sample KLs (if missing, fall back to KL_A)
        if per_sample_mean is None:
            uncertainties_C.append(kl_A)
        else:
            uncertainties_C.append(float(per_sample_mean))

        vanilla_acc = entailment_results[qid].get('vanilla_accuracy')
        if vanilla_acc is None:
            logger.warning(f"No accuracy for question {qid}, skipping for AUROC")
            continue
        accuracies.append(vanilla_acc)

    uncertainties_A = np.array(uncertainties_A)
    uncertainties_B = np.array(uncertainties_B)
    uncertainties_C = np.array(uncertainties_C)
    accuracies = np.array(accuracies)

    def _compute_and_log(name, uncertainties_arr):
        logger.info(f"\n{'='*60}")
        logger.info(f"UNCERTAINTY STATISTICS ({name})")
        logger.info(f"{'='*60}")
        logger.info(f"Number of samples: {len(uncertainties_arr)}")
        if len(uncertainties_arr) > 0:
            logger.info(f"Mean KL uncertainty: {np.mean(uncertainties_arr):.4f}")
            logger.info(f"Std KL uncertainty: {np.std(uncertainties_arr):.4f}")
            logger.info(f"Min KL uncertainty: {np.min(uncertainties_arr):.4f}")
            logger.info(f"Max KL uncertainty: {np.max(uncertainties_arr):.4f}")

        if len(accuracies) > 0:
            logger.info(f"\nVanilla accuracy: {np.mean(accuracies):.4f}")

        uncertainties_norm = quantile_power_normalize(uncertainties_arr)

        if len(set(accuracies)) > 1:
            auroc = roc_auc_score(accuracies, uncertainties_norm)
            logger.info(f"\n{'='*60}")
            logger.info(f"AUROC RESULTS ({name})")
            logger.info(f"{'='*60}")
            logger.info(f"AUROC: {auroc:.4f}")
            return {
                'uncertainties': uncertainties_arr.tolist(),
                'uncertainties_normalized': uncertainties_norm.tolist(),
                'auroc': auroc,
                'mean_uncertainty': float(np.mean(uncertainties_arr)),
                'std_uncertainty': float(np.std(uncertainties_arr))
            }
        else:
            logger.warning("Only one class present in accuracies, cannot compute AUROC")
            return {
                'uncertainties': uncertainties_arr.tolist(),
                'uncertainties_normalized': quantile_power_normalize(uncertainties_arr).tolist(),
                'auroc': None,
                'mean_uncertainty': float(np.mean(uncertainties_arr)) if len(uncertainties_arr) > 0 else None,
                'std_uncertainty': float(np.std(uncertainties_arr)) if len(uncertainties_arr) > 0 else None
            }

    results_A = _compute_and_log('avg-PMF-then-KL', uncertainties_A)
    results_B = _compute_and_log('mean-KLs-across-vanillas', uncertainties_B)
    results_C = _compute_and_log('mean-per-sample-KLs', uncertainties_C)

    # Save combined results
    output_path = Path(output_dir)
    results_dict = {
        'question_ids': question_ids,
        'accuracies': accuracies.tolist(),
        'avg_pmf_then_kl': results_A,
        'mean_kl_across_vanillas': results_B,
        'per_sample_kl': results_C
    }

    results_file = output_path / "uncertainty_auroc_results_kvanilla.pkl"
    save_pickle(results_dict, results_file)
    logger.info(f"\nSaved uncertainty and AUROC results to: {results_file}")

    return results_dict


def main():
    parser = argparse.ArgumentParser(
        description='Compute entailment between 3 vanilla samples and rephrased samples, then calculate uncertainty and AUROC'
    )

    parser.add_argument('--vanilla_run_dir', type=str, required=False, default=None,
                       help='(Optional) Path to wandb run directory with vanilla generations. If omitted, vanilla samples will be extracted from the rephrased run dir.')
    parser.add_argument('--rephrased_run_dir', type=str, required=True,
                       help='Path to wandb run directory with rephrased generations')
    parser.add_argument('--k_most_likely', type=int, default=3,
                       help='Number of most-likely (low-temp) vanilla samples to use per question')
    parser.add_argument('--subsample', type=int, default=50,
                       help='Number of rephrased samples to use per question (<= available, default 50)')
    parser.add_argument('--subsample_seed', type=int, default=0,
                       help='Random seed for subsampling (deterministic)')
    parser.add_argument('--wandb_base_dir', type=str,
                       default='./malaymilindp/uncertainty/wandb',
                       help='Base directory for wandb runs')
    parser.add_argument('--vanilla_gemini_json', type=str, default=None,
                       help='Optional path to JSON with per-example Gemini correctness to use for AUROC')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save entailment and uncertainty results')
    parser.add_argument('--strict_entailment', action='store_true',
                       help='If set, equivalence requires strict bidirectional entailment (both directions entail).')
    parser.add_argument('--dataset', type=str, default='bioasq',
                       help='Dataset name (for logging)')
    parser.add_argument('--model_name', type=str, default='Meta-Llama-3.1-8B-Instruct',
                       help='Model name (for logging)')

    args = parser.parse_args()

    # Determine run paths. Vanilla run is optional; if not provided we will extract
    # most-likely answers from the rephrased run directory.
    vanilla_full_path = None
    if args.vanilla_run_dir:
        vanilla_full_path = Path(args.wandb_base_dir) / args.vanilla_run_dir
    rephrased_arg = Path(args.rephrased_run_dir)
    if rephrased_arg.is_file() and rephrased_arg.suffix == '.pkl':
        rephrased_full_path = rephrased_arg
    else:
        rephrased_full_path = Path(args.wandb_base_dir) / args.rephrased_run_dir

    logger.info(f"\n{'='*60}")
    logger.info("ENTAILMENT AND UNCERTAINTY COMPUTATION (3 VANILLA)")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Vanilla run: {args.vanilla_run_dir}")
    logger.info(f"Rephrased run: {args.rephrased_run_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"{'='*60}\n")

    # Load rephrased gens (this always exists)
    rephrased_gens = load_rephrased_generations(rephrased_full_path)

    # Load vanilla gens only if a separate vanilla run dir was specified; otherwise
    # we'll extract most-likely answers from the rephrased gens per-original.
    vanilla_gens = {}
    if vanilla_full_path is not None:
        try:
            vanilla_gens = load_vanilla_generations(vanilla_full_path)
        except Exception as e:
            logger.warning(f"Failed to load separate vanilla gens from {vanilla_full_path}: {e}. Falling back to extracting from rephrased run.")
    gemini_labels = None
    if args.vanilla_gemini_json and os.path.exists(args.vanilla_gemini_json):
        with open(args.vanilla_gemini_json, 'r') as f:
            payload = json.load(f)
        if isinstance(payload, dict) and 'per_example' in payload and isinstance(payload['per_example'], dict):
            raw_map = payload['per_example']
        else:
            raw_map = payload if isinstance(payload, dict) else {}
        gemini_labels = {str(k): int(v) for k, v in raw_map.items()}

    logger.info(f"\nStarting entailment analysis ({args.k_most_likely} vanilla answers per question)...")
    entailment_results, equivalence_pmfs = perform_entailment_analysis_kvanilla(
        vanilla_gens,
        rephrased_gens,
        args.output_dir,
        k=args.k_most_likely,
        gemini_labels=gemini_labels,
        strict_entailment=args.strict_entailment,
        subsample=args.subsample,
        subsample_seed=args.subsample_seed
    )

    if entailment_results is None:
        logger.error("Entailment analysis failed. Exiting.")
        return 1

    logger.info("\nComputing uncertainty and AUROC from PMFs (two aggregation strategies)...")
    uncertainty_results = compute_uncertainty_and_auroc_from_pmfs(
        equivalence_pmfs, entailment_results, args.output_dir
    )

    # Write a CSV with two rows—one for each aggregation strategy
    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        rows = []
        if uncertainty_results is not None:
            # Extract results for both strategies
            res_A = uncertainty_results.get('avg_pmf_then_kl', {})
            res_B = uncertainty_results.get('mean_kl_across_vanillas', {})
            res_C = uncertainty_results.get('per_sample_kl', {})
            
            # Common hyperparameters for both rows
            base_row = {
                'dataset': args.dataset,
                'model_name': args.model_name,
                'vanilla_run_dir': str(args.vanilla_run_dir) if args.vanilla_run_dir is not None else None,
                'rephrased_run_dir': str(args.rephrased_run_dir),
                'k_most_likely': args.k_most_likely,
                'subsample': args.subsample,
                'subsample_seed': args.subsample_seed,
                'strict_entailment': bool(args.strict_entailment),
                'vanilla_gemini_json': args.vanilla_gemini_json,
                'output_dir': str(args.output_dir),
            }
            
            # Row 1: avg-PMF-then-KL strategy
            row_A = base_row.copy()
            row_A['aggregation_strategy'] = 'avg_pmf_then_kl'
            row_A['auroc'] = res_A.get('auroc')
            row_A['mean_uncertainty'] = res_A.get('mean_uncertainty')
            row_A['std_uncertainty'] = res_A.get('std_uncertainty')
            rows.append(row_A)
            
            # Row 2: mean-KL-across-vanillas strategy
            row_B = base_row.copy()
            row_B['aggregation_strategy'] = 'mean_kl_across_vanillas'
            row_B['auroc'] = res_B.get('auroc')
            row_B['mean_uncertainty'] = res_B.get('mean_uncertainty')
            row_B['std_uncertainty'] = res_B.get('std_uncertainty')
            rows.append(row_B)

            # Row 3: mean per-sample KL strategy
            row_C = base_row.copy()
            row_C['aggregation_strategy'] = 'per_sample_kl'
            row_C['auroc'] = res_C.get('auroc')
            row_C['mean_uncertainty'] = res_C.get('mean_uncertainty')
            row_C['std_uncertainty'] = res_C.get('std_uncertainty')
            rows.append(row_C)
        
        df_summary = pd.DataFrame(rows)
        vname = args.vanilla_run_dir or 'from_rephrased'
        csv_name = f"{args.dataset}_{args.model_name}_{vname}_{args.rephrased_run_dir}_auroc_summary_k{args.k_most_likely}.csv"
        csv_path = out_dir / csv_name
        df_summary.to_csv(csv_path, index=False)
        logger.info(f"Saved k-vanilla AUROC summary CSV (2 rows for 2 strategies) to: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to write k-vanilla AUROC summary CSV: {e}")

    # Optionally write CSVs for vqa dataset similar to original script
    if args.dataset == 'vqa':
        try:
            central_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'entailment_analysis'
            central_dir.mkdir(parents=True, exist_ok=True)

            rows = []
            for qid, entry in equivalence_pmfs.items():
                vanilla_acc = None
                vdata = vanilla_gens.get(qid) or (vanilla_gens.get(int(qid)) if isinstance(qid, str) and qid.isdigit() else vanilla_gens.get(qid))
                if vdata:
                    vanilla_acc = vdata.get('most_likely_answer', {}).get('accuracy')
                per_counts = entry.get('per_v_equiv_counts', [])
                rows.append({'question_id': str(qid), 'avg_p_not_equiv': entry.get('avg_pmf', [None, None])[0],
                             'avg_p_equiv': entry.get('avg_pmf', [None, None])[1],
                             'vanilla_accuracy': vanilla_acc,
                             'per_v_equiv_counts': per_counts})
            df_counts = pd.DataFrame(rows)
            # include dataset, model and k in filename for clarity
            vname = args.vanilla_run_dir or 'from_rephrased'
            counts_csv = central_dir / f"{args.dataset}_{args.model_name}_{vname}_{args.rephrased_run_dir}_equivalence_counts_k{args.k_most_likely}.csv"
            df_counts.to_csv(counts_csv, index=False)
            logger.info(f"Saved equivalence counts CSV to: {counts_csv}")

            if uncertainty_results is not None:
                ur = uncertainty_results
                # uncertainty_results contains nested structures for the two strategies
                # flatten a simple CSV with question_id, uncertainty_A, uncertainty_B, accuracy (where available)
                qids = ur.get('question_ids', [])
                accs = ur.get('accuracies', [])
                a_unc = ur.get('avg_pmf_then_kl', {}).get('uncertainties', [])
                b_unc = ur.get('mean_kl_across_vanillas', {}).get('uncertainties', [])
                df_unc = pd.DataFrame({
                    'question_id': qids,
                    'uncertainty_avg_pmf_then_kl': a_unc,
                    'uncertainty_mean_kls': b_unc,
                    'accuracy': accs
                })
                vname = args.vanilla_run_dir or 'from_rephrased'
                unc_csv = central_dir / f"{args.dataset}_{args.model_name}_{vname}_{args.rephrased_run_dir}_uncertainty_results_k{args.k_most_likely}.csv"
                df_unc.to_csv(unc_csv, index=False)
                logger.info(f"Saved uncertainty results CSV to: {unc_csv}")
        except Exception as e:
            logger.warning(f"Failed to write central CSVs for VQA: {e}")

    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING COMPLETE (3 VANILLA)")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
