#!/usr/bin/env python3
"""
Compute KL divergence-based uncertainty between greedy (k vanilla) and sampled outputs
by constructing PMFs over answer supports.

Behavior:
- For each question (original id) we obtain up to `k` greedy answers (from a vanilla run
  or synthesized from rephrased run). The greedy PMF support is the set of unique greedy
  answers and probabilities are by frequency.
- For sampled outputs we construct two variants:
  * Aggregated: flatten all sampled answers across rephrased samples and create a PMF
    over the sampled support (answers frequency).
  * Per-sample: treat each sampled generation as a sample PMF (delta-like, smoothed)
    and compute KL against greedy PMF, then average KLs across samples.

Outputs:
- Pickle files with per-question results and PMFs
- A CSV summary with two rows: one for aggregated-KL strategy and one for per-sample-KL

Usage mirrors the original k-vanilla entailment script (uses wandb run dirs or pkl paths).
"""

import os
import sys
import re
import pickle
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import random
import json

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score

# Optional: Entailment model for semantics-aware binning
try:
    from snne.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
except Exception:
    EntailmentDeberta = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_rephrased_generations(wandb_run_dir):
    p = Path(wandb_run_dir)
    if p.is_file() and p.suffix == '.pkl':
        validation_pkl = p
    else:
        validation_pkl = p / 'files' / 'validation_generations.pkl'

    if not validation_pkl.exists():
        raise FileNotFoundError(f'Rephrased generations not found: {validation_pkl}')

    logger.info(f'Loading rephrased generations from: {validation_pkl}')
    generations = load_pickle(validation_pkl)
    logger.info(f'Loaded {len(generations)} rephrased samples')
    return generations


def load_vanilla_generations(wandb_run_dir):
    if wandb_run_dir is None:
        return {}
    validation_pkl = Path(wandb_run_dir) / 'files' / 'validation_generations.pkl'
    if not validation_pkl.exists():
        raise FileNotFoundError(f'Vanilla generations not found: {validation_pkl}')
    logger.info(f'Loading vanilla generations from: {validation_pkl}')
    generations = load_pickle(validation_pkl)
    logger.info(f'Loaded {len(generations)} vanilla samples')
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
    """Extract text from various container formats used across the repo."""
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


def get_k_vanilla_answers(vanilla_data, k=3):
    """Return k vanilla answer strings; duplicate last if fewer than k."""
    answers = []

    if isinstance(vanilla_data.get('responses'), (list, tuple)) and len(vanilla_data.get('responses')) > 0:
        for t in vanilla_data.get('responses')[:k]:
            txt = _extract_response_text(t)
            if txt is not None:
                answers.append(txt)

    if len(answers) < k and isinstance(vanilla_data.get('most_likely_answers'), (list, tuple)):
        for x in vanilla_data.get('most_likely_answers')[:k]:
            if isinstance(x, dict) and 'response' in x:
                answers.append(x['response'])
            else:
                txt = _extract_response_text(x)
                if txt is not None:
                    answers.append(txt)

    if len(answers) < k:
        mla = vanilla_data.get('most_likely_answer')
        if isinstance(mla, dict) and 'response' in mla:
            answers.append(mla['response'])
        else:
            txt = _extract_response_text(mla)
            if txt is not None:
                answers.append(txt)

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

    if len(answers) == 0:
        logger.warning('No vanilla answers found in entry; using empty strings')
        answers = [''] * k
    elif len(answers) < k:
        logger.warning(f'Only {len(answers)} vanilla answer(s) found; duplicating to create {k} samples')
        while len(answers) < k:
            answers.append(answers[-1])
    else:
        answers = answers[:k]

    return answers


def create_pmf(answers, smoothing=1e-10):
    if not answers:
        return {}
    counts = Counter(answers)
    total = sum(counts.values())
    vocab_size = len(counts)
    pmf = {}
    for answer, count in counts.items():
        pmf[answer] = (count + smoothing) / (total + smoothing * vocab_size)
    return pmf


def normalize_pmf(pmf):
    total = sum(pmf.values())
    if total > 0:
        return {k: v / total for k, v in pmf.items()}
    return pmf


def kl_divergence(p_pmf, q_pmf, smoothing=1e-10):
    # union support
    all_answers = set(p_pmf.keys()) | set(q_pmf.keys())
    if len(all_answers) == 0:
        return 0.0
    p_probs = []
    q_probs = []
    for a in all_answers:
        p_probs.append(p_pmf.get(a, smoothing))
        q_probs.append(q_pmf.get(a, smoothing))
    p = np.array(p_probs, dtype=float)
    q = np.array(q_probs, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    # use stable rel_entr from scipy if available else manual
    try:
        from scipy.special import rel_entr
        kl = float(np.sum(rel_entr(p, q)))
    except Exception:
        # fallback: sum p * log(p/q)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(q <= 0, 1.0, p / q)
            logs = np.log(ratio + 1e-20)
            kl = float(np.nansum(p * logs))
    if np.isnan(kl) or np.isinf(kl):
        return 0.0
    return kl


def quantile_power_normalize(x, gamma=0.5, clip=(1, 99)):
    x = np.asarray(x, dtype=float)
    epsilon = 1e-9

    # Reciprocal: higher KL -> lower raw score; invert so higher -> higher confidence after transform
    x = 1.0 / (x + epsilon)

    # Percentile clipping to reduce outlier effects
    if clip is not None and len(x) > 0:
        lo, hi = np.percentile(x, clip)
        x = np.clip(x, lo, hi)

    # Rank-based empirical CDF
    ranks = np.argsort(np.argsort(x)) + 1
    u = ranks / (len(x) + 1.0)

    # Power transformation
    confidence = u ** gamma

    # Min-max normalization to [0, 1]
    confidence = (confidence - np.min(confidence)) / (np.max(confidence) - np.min(confidence) + epsilon)

    return confidence


def _extract_numeric(s):
    try:
        if s is None:
            return None
        # strip non-number surroundings, simple heuristic
        txt = str(s).strip()
        # handle comma separators
        txt = txt.replace(',', '')
        # extract first float-like token
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
        if not m:
            return None
        return float(m.group(0))
    except Exception:
        return None


def perform_pmf_analysis_kvanilla(vanilla_gens, rephrased_gens, output_dir, k=3, subsample=50, subsample_seed=0,
                                  use_entailment_mapping=False, strict_entailment=False):
    rephrased_by_original = organize_rephrased_by_original(rephrased_gens)

    results = {}
    pmf_summaries = {}

    vanilla_ids = set(vanilla_gens.keys()) if isinstance(vanilla_gens, dict) else set()
    rephrased_original_ids = set(rephrased_by_original.keys())

    if len(vanilla_ids) == 0:
        common_ids = rephrased_original_ids
    else:
        common_ids = vanilla_ids & rephrased_original_ids

    logger.info(f'Found {len(common_ids)} question IDs to process')

    # Lazy-init entailment model if requested
    entail_model = None
    if use_entailment_mapping and EntailmentDeberta is not None:
        entail_model = EntailmentDeberta()

    for original_id in tqdm(sorted(common_ids), desc='Processing questions'):
        # load or synthesize vanilla_data
        vanilla_data = None
        if vanilla_ids:
            vanilla_data = vanilla_gens.get(original_id) or vanilla_gens.get(int(original_id)) if isinstance(original_id, str) and original_id.isdigit() else None
            if vanilla_data is None:
                for kkey in vanilla_gens.keys():
                    if str(kkey) == str(original_id):
                        vanilla_data = vanilla_gens[kkey]
                        break

        if vanilla_data is None:
            # synthesize from rephrased run: collect most_likely_answers
            gathered_ml = []
            for r in rephrased_by_original.get(str(original_id), []):
                rid = r.get('rephrased_id')
                raw_entry = rephrased_gens.get(rid, r)
                mls = raw_entry.get('most_likely_answers') if isinstance(raw_entry, dict) else None
                if isinstance(mls, (list, tuple)) and len(mls) > 0:
                    for x in mls:
                        txt = _extract_response_text(x)
                        acc = x.get('accuracy') if isinstance(x, dict) else None
                        if txt is not None:
                            gathered_ml.append({'response': txt, 'accuracy': acc})
                else:
                    mla = raw_entry.get('most_likely_answer') if isinstance(raw_entry, dict) else None
                    txt = _extract_response_text(mla)
                    acc = mla.get('accuracy') if isinstance(mla, dict) else None
                    if txt is not None:
                        gathered_ml.append({'response': txt, 'accuracy': acc})

            if len(gathered_ml) == 0:
                logger.warning(f'No most-likely vanilla answers found for original id {original_id}; skipping')
                continue

            vanilla_data = {
                'most_likely_answers': gathered_ml,
                'most_likely_answer': gathered_ml[0],
                'question': rephrased_by_original.get(str(original_id), [{}])[0].get('question')
            }

        vanilla_answers = get_k_vanilla_answers(vanilla_data, k=k)

        # Build greedy PMF (support = unique greedy answers)
        greedy_pmf = create_pmf(vanilla_answers)

        # Collect all sampled answers across rephrased items for this original
        rephrased_list = rephrased_by_original.get(str(original_id), [])
        all_samples = []
        per_item_samples = []
        for r in rephrased_list:
            for resp in r.get('responses', []) or []:
                ans = _extract_response_text(resp)
                all_samples.append(ans)
                per_item_samples.append(ans)

        total_available = len(all_samples)
        use_k = subsample if subsample is not None else total_available
        if use_k > total_available:
            logger.warning(f'Requested subsample {use_k} > available {total_available}; using all available samples')
            use_k = total_available

        if use_k < total_available:
            try:
                base = int(str(original_id))
            except Exception:
                base = int(hash(str(original_id)) & 0xffffffff)
            rnd = random.Random(subsample_seed + base)
            chosen_idx = rnd.sample(range(total_available), use_k)
            sampled_answers_chosen = [all_samples[i] for i in chosen_idx]
        else:
            sampled_answers_chosen = list(all_samples)

        # Build sampled answers as list-of-samples (each sample may itself be a list)
        # Our rephrased entries are individual responses, so represent each chosen answer as a single-sample list
        sampled_answers_list = [[ans] for ans in sampled_answers_chosen]
        # Sampled PMF
        sampled_pmf = {}
        # Method 1: Aggregated KL (group out-of-support answers into __OTHER__)
        def compute_aggregated_kl_local(greedy_answers, sampled_answers_list, smoothing=1e-10):
            greedy_pmf_raw = create_pmf(greedy_answers, smoothing)
            greedy_support = set(greedy_pmf_raw.keys())
            greedy_list = list(greedy_support)

            # Flatten sampled answers
            all_sampled = [ans for sample in sampled_answers_list for ans in sample]

            sampled_counts = Counter()
            other_count = 0
            for answer in all_sampled:
                bin_key = None
                # 1) exact match
                if answer in greedy_support:
                    bin_key = answer
                else:
                    # 2) numeric-equality match
                    a_num = _extract_numeric(answer)
                    if a_num is not None:
                        for g in greedy_list:
                            g_num = _extract_numeric(g)
                            if g_num is not None and abs(a_num - g_num) <= 1e-8:
                                bin_key = g
                                break
                if bin_key is None and use_entailment_mapping and entail_model is not None:
                    # 3) semantics-aware binning via entailment
                    for g in greedy_list:
                        if answer is None or g is None:
                            continue
                        fwd = entail_model.check_implication(answer, g)
                        bwd = entail_model.check_implication(g, answer)
                        if strict_entailment:
                            is_equiv = (fwd == 2) and (bwd == 2)
                        else:
                            implications = [fwd, bwd]
                            is_equiv = (0 not in implications) and (implications != [1, 1])
                        if is_equiv:
                            bin_key = g
                            break
                if bin_key is not None:
                    sampled_counts[bin_key] += 1
                else:
                    other_count += 1

            if other_count > 0:
                sampled_counts['__OTHER__'] = other_count

            extended_support = list(greedy_support) + (['__OTHER__'] if other_count > 0 else [])

            # Greedy PMF: include tiny mass for OTHER
            greedy_pmf = {}
            total_greedy = len(greedy_answers)
            for answer in greedy_support:
                greedy_pmf[answer] = greedy_pmf_raw[answer]
            if other_count > 0:
                greedy_pmf['__OTHER__'] = smoothing / (total_greedy + smoothing * len(extended_support))

            # Sampled PMF
            total_sampled = len(all_sampled)
            if total_sampled == 0:
                return 0.0

            vocab_size = len(extended_support)
            for answer in extended_support:
                count = sampled_counts.get(answer, 0)
                sampled_pmf[answer] = (count + smoothing) / (total_sampled + smoothing * vocab_size)

            kl = kl_divergence(greedy_pmf, sampled_pmf, smoothing)
            return kl

        # Method 2: Per-sample KL (compare each sample to greedy, grouping OTHER)
        def compute_per_sample_kl_local(greedy_answers, sampled_answers_list, smoothing=1e-10):
            greedy_pmf_raw = create_pmf(greedy_answers, smoothing)
            greedy_support = set(greedy_pmf_raw.keys())
            greedy_list = list(greedy_support)

            kl_values = []
            for sample_answers in sampled_answers_list:
                sample_counts = Counter()
                other_count = 0
                for answer in sample_answers:
                    bin_key = None
                    if answer in greedy_support:
                        bin_key = answer
                    else:
                        a_num = _extract_numeric(answer)
                        if a_num is not None:
                            for g in greedy_list:
                                g_num = _extract_numeric(g)
                                if g_num is not None and abs(a_num - g_num) <= 1e-8:
                                    bin_key = g
                                    break
                    if bin_key is None and use_entailment_mapping and entail_model is not None:
                        for g in greedy_list:
                            if answer is None or g is None:
                                continue
                            fwd = entail_model.check_implication(answer, g)
                            bwd = entail_model.check_implication(g, answer)
                            if strict_entailment:
                                is_equiv = (fwd == 2) and (bwd == 2)
                            else:
                                implications = [fwd, bwd]
                                is_equiv = (0 not in implications) and (implications != [1, 1])
                            if is_equiv:
                                bin_key = g
                                break
                    if bin_key is not None:
                        sample_counts[bin_key] += 1
                    else:
                        other_count += 1

                if other_count > 0:
                    sample_counts['__OTHER__'] = other_count

                extended_support = list(greedy_support) + (['__OTHER__'] if other_count > 0 else [])

                greedy_pmf = {}
                total_greedy = len(greedy_answers)
                for answer in greedy_support:
                    greedy_pmf[answer] = greedy_pmf_raw[answer]
                if other_count > 0:
                    greedy_pmf['__OTHER__'] = smoothing / (total_greedy + smoothing * len(extended_support))

                total_sample = len(sample_answers)
                if total_sample == 0:
                    continue

                sample_pmf = {}
                vocab_size = len(extended_support)
                for answer in extended_support:
                    count = sample_counts.get(answer, 0)
                    sample_pmf[answer] = (count + smoothing) / (total_sample + smoothing * vocab_size)

                kl = kl_divergence(greedy_pmf, sample_pmf, smoothing)
                kl_values.append(kl)

            if not kl_values:
                return 0.0, 0.0, []
            mean_kl = float(np.mean(kl_values))
            std_kl = float(np.std(kl_values))
            return mean_kl, std_kl, kl_values

        kl_agg = compute_aggregated_kl_local(vanilla_answers, sampled_answers_list)
        mean_kl, std_kl, kl_values = compute_per_sample_kl_local(vanilla_answers, sampled_answers_list)

        results[str(original_id)] = {
            'question': vanilla_data.get('question'),
            'vanilla_answers': vanilla_answers,
            'sampled_answers': sampled_answers_chosen,
            'kl_aggregated': kl_agg,
            'kl_per_sample_mean': mean_kl,
            'kl_per_sample_std': std_kl,
            'kl_per_sample_values': kl_values,
            'vanilla_accuracy': vanilla_data.get('most_likely_answer', {}).get('accuracy')
        }

        pmf_summaries[str(original_id)] = {
            'greedy_pmf': greedy_pmf,
            'sampled_pmf': sampled_pmf
        }

    # Save outputs
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    res_file = outdir / 'pmf_kl_results_k{}.pkl'.format(k)
    save_pickle(results, res_file)
    logger.info(f'Saved PMF-KL results to: {res_file}')

    pmf_file = outdir / 'pmf_summaries_k{}.pkl'.format(k)
    save_pickle(pmf_summaries, pmf_file)
    logger.info(f'Saved PMF summaries to: {pmf_file}')

    return results


def compute_auroc_from_results(results, output_dir):
    # Build arrays
    qids = sorted(results.keys())
    kl_agg = []
    kl_per = []
    accs = []
    for q in qids:
        entry = results[q]
        if entry.get('kl_aggregated') is None:
            continue
        kl_agg.append(entry['kl_aggregated'])
        kl_per.append(entry['kl_per_sample_mean'])
        accs.append(entry.get('vanilla_accuracy'))

    kl_agg = np.array(kl_agg)
    kl_per = np.array(kl_per)
    accs = np.array([a for a in accs if a is not None]) if len(accs) > 0 else np.array([])

    def safe_quantile_norm(arr):
        if len(arr) == 0:
            return arr
        return quantile_power_normalize(arr)

    # Prepare result structure
    out = {}

    # Compute AUROC only if accuracies available and not all same
    if len(accs) > 1 and len(set(accs.tolist())) > 1:
        # align sizes: accs may be shorter due to missing values; reconstruct per-qid lists
        y_true = []
        X_agg = []
        X_per = []
        for q in qids:
            e = results[q]
            if e.get('vanilla_accuracy') is None or e.get('kl_aggregated') is None:
                continue
            y_true.append(e.get('vanilla_accuracy'))
            X_agg.append(e.get('kl_aggregated'))
            X_per.append(e.get('kl_per_sample_mean'))

        y_true = np.array(y_true)
        X_agg = np.array(X_agg)
        X_per = np.array(X_per)

        # convert to binary label using threshold 0.5 (as in reference)
        y_binary = (y_true > 0.5).astype(int)

        conf_agg = safe_quantile_norm(X_agg)
        conf_per = safe_quantile_norm(X_per)

        auroc_agg = roc_auc_score(y_binary, conf_agg) if len(np.unique(y_binary)) > 1 else None
        auroc_per = roc_auc_score(y_binary, conf_per) if len(np.unique(y_binary)) > 1 else None

        out['aggregated_kl'] = {
            'auroc': float(auroc_agg) if auroc_agg is not None else None,
            'mean_uncertainty': float(np.mean(X_agg)) if len(X_agg) > 0 else None,
            'std_uncertainty': float(np.std(X_agg)) if len(X_agg) > 0 else None,
        }
        out['per_sample_kl'] = {
            'auroc': float(auroc_per) if auroc_per is not None else None,
            'mean_uncertainty': float(np.mean(X_per)) if len(X_per) > 0 else None,
            'std_uncertainty': float(np.std(X_per)) if len(X_per) > 0 else None,
        }
    else:
        out['aggregated_kl'] = {'auroc': None}
        out['per_sample_kl'] = {'auroc': None}

    # Save summary CSV with two rows
    try:
        rows = []
        base_row = {
            'script': 'pmf_kl_kvanilla',
        }
        rowA = base_row.copy()
        rowA['aggregation_strategy'] = 'aggregated_kl'
        rowA.update(out.get('aggregated_kl', {}))
        rows.append(rowA)

        rowB = base_row.copy()
        rowB['aggregation_strategy'] = 'per_sample_kl'
        rowB.update(out.get('per_sample_kl', {}))
        rows.append(rowB)

        df = pd.DataFrame(rows)
        out_path = Path(output_dir) / 'pmf_kl_auroc_summary_k.csv'
        df.to_csv(out_path, index=False)
        logger.info(f'Saved AUROC summary CSV to: {out_path}')
    except Exception as e:
        logger.warning(f'Failed to write AUROC CSV: {e}')

    # Save json/pkl summary
    save_pickle(out, Path(output_dir) / 'pmf_kl_auroc_results.pkl')
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Compute PMF-based KL between k vanilla and rephrased samples, then calculate uncertainty and AUROC'
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
    parser.add_argument('--dataset', type=str, default='bioasq',
                       help='Dataset name (for logging)')
    parser.add_argument('--model_name', type=str, default='Meta-Llama-3.1-8B-Instruct',
                       help='Model name (for logging)')
    parser.add_argument('--use_entailment_mapping', action='store_true',
                       help='Map sampled answers to greedy bins using NLI entailment when no exact/numeric match')
    parser.add_argument('--strict_entailment', action='store_true',
                       help='If set, equivalence requires strict bidirectional entailment (both directions entail).')

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
    logger.info("PMF-KL UNCERTAINTY (k VANILLA)")
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
            logger.warning(f'Failed to load separate vanilla gens from {vanilla_full_path}: {e}. Falling back to extracting from rephrased run.')

    gemini_labels = None
    if args.vanilla_gemini_json and os.path.exists(args.vanilla_gemini_json):
        with open(args.vanilla_gemini_json, 'r') as f:
            payload = json.load(f)
        if isinstance(payload, dict) and 'per_example' in payload and isinstance(payload['per_example'], dict):
            raw_map = payload['per_example']
        else:
            raw_map = payload if isinstance(payload, dict) else {}
        gemini_labels = {str(k): int(v) for k, v in raw_map.items()}

    logger.info(f"\nStarting PMF-based KL analysis (k={args.k_most_likely})...")

    results = perform_pmf_analysis_kvanilla(
        vanilla_gens,
        rephrased_gens,
        args.output_dir,
        k=args.k_most_likely,
        subsample=args.subsample,
        subsample_seed=args.subsample_seed,
        use_entailment_mapping=bool(args.use_entailment_mapping),
        strict_entailment=bool(args.strict_entailment),
    )

    # If gemini labels provided, override vanilla_accuracy in results
    if gemini_labels is not None:
        for qid, entry in results.items():
            if str(qid) in gemini_labels:
                entry['vanilla_accuracy'] = gemini_labels.get(str(qid))

    auroc = compute_auroc_from_results(results, args.output_dir)

    # Write a CSV with two rows—one for each aggregation strategy (match previous format)
    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        # Common hyperparameters for both rows
        base_row = {
            'dataset': args.dataset,
            'model_name': args.model_name,
            'vanilla_run_dir': str(args.vanilla_run_dir) if args.vanilla_run_dir is not None else None,
            'rephrased_run_dir': str(args.rephrased_run_dir),
            'k_most_likely': args.k_most_likely,
            'subsample': args.subsample,
            'subsample_seed': args.subsample_seed,
            'output_dir': str(args.output_dir),
        }

        res_A = auroc.get('aggregated_kl', {}) if isinstance(auroc, dict) else {}
        res_B = auroc.get('per_sample_kl', {}) if isinstance(auroc, dict) else {}

        row_A = base_row.copy()
        row_A['aggregation_strategy'] = 'aggregated_kl'
        row_A['auroc'] = res_A.get('auroc')
        row_A['mean_uncertainty'] = res_A.get('mean_uncertainty')
        row_A['std_uncertainty'] = res_A.get('std_uncertainty')
        rows.append(row_A)

        row_B = base_row.copy()
        row_B['aggregation_strategy'] = 'per_sample_kl'
        row_B['auroc'] = res_B.get('auroc')
        row_B['mean_uncertainty'] = res_B.get('mean_uncertainty')
        row_B['std_uncertainty'] = res_B.get('std_uncertainty')
        rows.append(row_B)

        df_summary = pd.DataFrame(rows)
        vname = args.vanilla_run_dir or 'from_rephrased'
        csv_name = f"{args.dataset}_{args.model_name}_{vname}_{args.rephrased_run_dir}_auroc_summary_k{args.k_most_likely}.csv"
        csv_path = out_dir / csv_name
        df_summary.to_csv(csv_path, index=False)
        logger.info(f"Saved k-vanilla AUROC summary CSV (2 rows for 2 strategies) to: {csv_path}")
    except Exception as e:
        logger.warning(f'Failed to write k-vanilla AUROC summary CSV: {e}')

    logger.info('Done.')


if __name__ == '__main__':
    main()
