#!/usr/bin/env python3
"""
Compute DeBERTa entailment between the low-temperature vanilla answer and its
corresponding high-temperature samples (per vanilla generation entry).

This script is a variant of the vanilla-vs-rephrased entailment script but
expects a single `vanilla_run_dir` which contains for each rephrased_id:
- `most_likely_answer` (the low-temp / k=1 answer)
- `responses` -- a list of high-temperature generations (tuples like
  (answer, token_log_likelihoods, embedding, acc))

For each generation entry this script:
- extracts the low-t vanilla answer
- compares it (bidirectional entailment via the project's EntailmentDeberta)
  to each of the high-T samples in `responses`
- computes a PMF [P(not_equiv), P(equiv)] across the high-T samples
- computes KL uncertainty vs ideal=[0,1] and evaluates AUROC against
  the vanilla accuracy (or an optional gemini JSON provided via CLI)

This keeps the rest of the logging / saving conventions consistent with the
other entailment scripts in the repo.
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
    p = Path(wandb_run_dir)
    if p.is_file() and p.suffix == '.pkl':
        validation_pkl = p
    else:
        validation_pkl = p / "files" / "validation_generations.pkl"

    if not validation_pkl.exists():
        raise FileNotFoundError(f"Vanilla generations not found: {validation_pkl}")
    logger.info(f"Loading vanilla generations from: {validation_pkl}")
    generations = load_pickle(validation_pkl)
    logger.info(f"Loaded {len(generations)} vanilla samples")
    return generations


def _extract_response_text(item):
    if item is None:
        return None
    if isinstance(item, (tuple, list)):
        if len(item) > 0:
            return item[0]
        return None
    if isinstance(item, dict):
        if 'response' in item:
            return item.get('response')
        if 'answer' in item:
            return item.get('answer')
        return None
    if isinstance(item, str):
        return item
    return None


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


def perform_entailment_analysis_vanilla_hight(vanilla_gens, output_dir, gemini_labels=None, strict_entailment=False):
    """For each vanilla generation entry, entail the low-t answer with its high-T responses."""
    entailment_model = EntailmentDeberta()

    entailment_results = {}
    equivalence_pmfs = {}

    all_ids = list(vanilla_gens.keys()) if isinstance(vanilla_gens, dict) else []
    logger.info(f"Found {len(all_ids)} vanilla generation IDs to process")

    for vid in tqdm(sorted(all_ids), desc="Processing vanilla entries"):
        entry = vanilla_gens[vid]
        # Extract low-t vanilla
        mla = None
        if isinstance(entry, dict):
            mla = entry.get('most_likely_answer') or entry.get('most_likely_answers', [None])[0]
        else:
            mla = None

        lowt_text = _extract_response_text(mla)
        # responses: high-T list stored as list of tuples (answer, logliks, emb, acc)
        responses = entry.get('responses') if isinstance(entry, dict) else None
        if responses is None or len(responses) == 0:
            logger.warning(f"No high-T responses found for id {vid}; skipping")
            continue

        # Determine vanilla accuracy (prefer gemini_labels if provided)
        if gemini_labels is not None:
            vanilla_accuracy = gemini_labels.get(str(vid))
        else:
            vanilla_accuracy = None
            if isinstance(mla, dict):
                vanilla_accuracy = mla.get('accuracy')

        # Compare low-t to each high-T sample
        equiv_count = 0
        not_equiv_count = 0
        comparisons = []
        for sample_idx, resp in enumerate(responses):
            high_text = _extract_response_text(resp)
            if lowt_text is None or high_text is None:
                is_equiv = False
            else:
                imp_fwd = entailment_model.check_implication(lowt_text, high_text, example=entry.get('question'))
                imp_bwd = entailment_model.check_implication(high_text, lowt_text, example=entry.get('question'))
                if strict_entailment:
                    is_equiv = (imp_fwd == 2) and (imp_bwd == 2)
                else:
                    implications = [imp_fwd, imp_bwd]
                    is_equiv = (0 not in implications) and (implications != [1, 1])

            if is_equiv:
                equiv_count += 1
            else:
                not_equiv_count += 1

            comparisons.append({
                'sample_idx': sample_idx,
                'high_text': high_text,
                'is_equiv': bool(is_equiv)
            })

        pmf = compute_pmf_from_counts(equiv_count, not_equiv_count)
        pmf_ideal = [0.0, 1.0]
        kl_unc = kl_divergence(pmf, pmf_ideal)

        entailment_results[str(vid)] = {
            'lowt_answer': lowt_text,
            'vanilla_accuracy': vanilla_accuracy,
            'comparisons': comparisons,
            'equiv_count': equiv_count,
            'not_equiv_count': not_equiv_count,
            'question': entry.get('question') if isinstance(entry, dict) else None
        }

        equivalence_pmfs[str(vid)] = {
            'pmf': pmf,
            'equiv_count': equiv_count,
            'not_equiv_count': not_equiv_count,
            'kl_uncertainty': kl_unc
        }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "entailment_results_vanilla.pkl"
    save_pickle(entailment_results, results_file)
    logger.info(f"Saved entailment results to: {results_file}")

    pmf_file = output_path / "equivalence_pmfs_vanilla.pkl"
    save_pickle(equivalence_pmfs, pmf_file)
    logger.info(f"Saved equivalence PMFs to: {pmf_file}")

    return entailment_results, equivalence_pmfs


def compute_uncertainty_and_auroc_from_pmfs_vanilla(equivalence_pmfs, entailment_results, output_dir):
    question_ids = sorted(equivalence_pmfs.keys())

    uncertainties = []
    accuracies = []

    for qid in question_ids:
        entry = equivalence_pmfs[qid]
        pmf = entry.get('pmf')
        if pmf is None:
            logger.warning(f"No PMF for {qid}; skipping")
            continue
        kl = kl_divergence(pmf, [0.0, 1.0])
        uncertainties.append(kl)

        vanilla_acc = entailment_results[qid].get('vanilla_accuracy')
        if vanilla_acc is None:
            logger.warning(f"No accuracy for question {qid}; skipping for AUROC")
            continue
        accuracies.append(int(vanilla_acc))

    uncertainties = np.array(uncertainties)
    accuracies = np.array(accuracies)

    logger.info(f"Number of samples (for AUROC): {len(accuracies)}")
    if len(accuracies) > 0:
        logger.info(f"Mean uncertainty: {np.mean(uncertainties):.4f}")

    uncertainties_norm = quantile_power_normalize(uncertainties)

    results = None
    if len(set(accuracies)) > 1:
        auroc = roc_auc_score(accuracies, uncertainties_norm)
        logger.info(f"AUROC: {auroc:.4f}")
        results = {
            'question_ids': question_ids,
            'accuracies': accuracies.tolist(),
            'uncertainties': uncertainties.tolist(),
            'uncertainties_normalized': uncertainties_norm.tolist(),
            'auroc': auroc
        }
    else:
        logger.warning("Only one class present in accuracies; cannot compute AUROC")
        results = {
            'question_ids': question_ids,
            'accuracies': accuracies.tolist(),
            'uncertainties': uncertainties.tolist(),
            'uncertainties_normalized': uncertainties_norm.tolist(),
            'auroc': None
        }

    output_path = Path(output_dir)
    results_file = output_path / "uncertainty_auroc_results_vanilla.pkl"
    save_pickle(results, results_file)
    logger.info(f"Saved uncertainty & AUROC results to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute entailment between vanilla low-t and its high-T samples, then calculate uncertainty and AUROC'
    )

    parser.add_argument('--vanilla_run_dir', type=str, required=True,
                        help='Path to wandb run directory (or .pkl) with vanilla generations')
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

    # Resolve path
    re = Path(args.vanilla_run_dir)
    if re.is_file() and re.suffix == '.pkl':
        vanilla_full_path = re
    else:
        vanilla_full_path = Path(args.wandb_base_dir) / args.vanilla_run_dir

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Vanilla run: {args.vanilla_run_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    vanilla_gens = load_vanilla_generations(vanilla_full_path)

    gemini_labels = None
    if args.vanilla_gemini_json and os.path.exists(args.vanilla_gemini_json):
        with open(args.vanilla_gemini_json, 'r') as f:
            payload = json.load(f)
        if isinstance(payload, dict) and 'per_example' in payload and isinstance(payload['per_example'], dict):
            raw_map = payload['per_example']
        else:
            raw_map = payload if isinstance(payload, dict) else {}
        gemini_labels = {str(k): int(v) for k, v in raw_map.items()}

    entailment_results, equivalence_pmfs = perform_entailment_analysis_vanilla_hight(
        vanilla_gens, args.output_dir, gemini_labels=gemini_labels, strict_entailment=args.strict_entailment
    )

    if entailment_results is None:
        logger.error("Entailment analysis failed. Exiting.")
        return 1

    uncertainty_results = compute_uncertainty_and_auroc_from_pmfs_vanilla(
        equivalence_pmfs, entailment_results, args.output_dir
    )

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
                    mla = vdata.get('most_likely_answer') if isinstance(vdata, dict) else None
                    if isinstance(mla, dict):
                        vanilla_acc = mla.get('accuracy')
                pmf = entry.get('pmf')
                rows.append({'question_id': str(qid), 'p_not_equiv': pmf[0] if pmf else None,
                             'p_equiv': pmf[1] if pmf else None,
                             'vanilla_accuracy': vanilla_acc,
                             'equiv_count': entry.get('equiv_count')})
            df_counts = pd.DataFrame(rows)
            vname = args.vanilla_run_dir
            counts_csv = central_dir / f"{args.dataset}_{args.model_name}_{vname}_vanilla_equivalence_counts.csv"
            df_counts.to_csv(counts_csv, index=False)
            logger.info(f"Saved equivalence counts CSV to: {counts_csv}")

            if uncertainty_results is not None:
                ur = uncertainty_results
                qids = ur.get('question_ids', [])
                accs = ur.get('accuracies', [])
                uncs = ur.get('uncertainties', [])
                df_unc = pd.DataFrame({'question_id': qids, 'uncertainty': uncs, 'accuracy': accs})
                unc_csv = central_dir / f"{args.dataset}_{args.model_name}_{vname}_vanilla_uncertainty_results.csv"
                df_unc.to_csv(unc_csv, index=False)
                logger.info(f"Saved uncertainty results CSV to: {unc_csv}")
        except Exception as e:
            logger.warning(f"Failed to write central CSVs for VQA: {e}")

    logger.info("Processing complete. Results saved to: %s", args.output_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
