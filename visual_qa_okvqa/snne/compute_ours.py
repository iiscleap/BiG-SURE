#!/usr/bin/env python3
"""
KL-based uncertainty and AUROC for Falcon-7B entailment outputs using DeBERTa comparison results.

Mirrors semantic_entropy/semantic_uncertainty/entailment_basic/baseline_entropy_py/lalm_entail_kl_auc_deberta.py

Usage:
python entail_kl_auc_falcon7b.py --model falcon7b

Expected input files (with default paths):
- entail_falcon7b_equiv_orig.tsv (original/rephrased perturbations)
- entail_falcon7b_equiv_complement.tsv (complementary perturbations)
- falcon7b_triviaqa_deberta_evaluation.tsv (DeBERTa ground truth)
"""
import os
import re
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def kl_divergence(p, p_star):
    """Compute KL(p_star || p) with small epsilon for stability.

    Args:
        p: iterable of probabilities (actual)
        p_star: iterable of probabilities (ideal)
    """
    p = np.asarray(p, dtype=float)
    p_star = np.asarray(p_star, dtype=float)
    eps = 1e-10
    return float(np.sum(p_star * np.log((p_star + eps) / (p + eps))))


def quantile_power_normalize(x, gamma=0.5, clip=(1, 99)):
    """Normalize values using quantile-based power transformation (higher -> higher)."""
    x = np.asarray(x, dtype=float)
    epsilon = 1e-9
    x = 1.0 / (x + epsilon)
    if clip is not None and len(x) > 0:
        lo, hi = np.percentile(x, clip)
        x = np.clip(x, lo, hi)
    ranks = np.argsort(np.argsort(x)) + 1
    u = ranks / (len(x) + 1.0)
    return u ** gamma


def extract_base_id_from_deberta_id(deberta_id: str) -> str:
    """Extract base_id like '150_721624' from DeBERTa IDs like 'dpql_6276--150/150_721624.txt#0_0'."""
    m = re.search(r"\b(\d+_\d+)\b", deberta_id)
    if m:
        return m.group(1)
    return deberta_id


def load_pruned_ids(csv_path: Path) -> set:
    """Load the set of valid IDs from the pruned CSV file.
    
    Returns set of base_ids (e.g., '150_721624')
    """
    pruned_ids = set()
    if not csv_path.exists():
        print(f"Warning: Pruned CSV file not found: {csv_path}")
        return pruned_ids
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_id = row.get('id', '').strip()
                if full_id:
                    # Extract base_id from full CSV id
                    base_id = extract_base_id_from_deberta_id(full_id)
                    pruned_ids.add(base_id)
        
        print(f"Loaded {len(pruned_ids)} pruned IDs from {csv_path}")
    except Exception as e:
        print(f"Error loading pruned CSV: {e}")
    
    return pruned_ids


def load_gemini_labels(gemini_tsv_path: Path) -> dict:
    """Load Gemini comparison results from TSV file.
    
    Expected format:
    - Column 0: id (e.g., 'dpql_6276--150/150_721624.txt#0_0')
    - Column 4: equivalent ('YES' or 'NO')
    
    Returns dict mapping base_id -> 1 (correct/YES) or 0 (incorrect/NO)
    """
    labels: dict = {}
    if not gemini_tsv_path.exists():
        print(f"Warning: Gemini TSV file not found: {gemini_tsv_path}")
        return labels

    try:
        with open(gemini_tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)  # Skip header
            
            for row in reader:
                if len(row) >= 5:  # Need at least 5 columns (id at 0, equivalent at 4)
                    deberta_id = row[0].strip()  # id column
                    equivalent = row[4].strip().upper()  # equivalent column
                    
                    # Extract base_id from DeBERTa id format
                    base_id = extract_base_id_from_deberta_id(deberta_id)
                    
                    # Convert YES/NO to 1/0
                    if equivalent == 'YES':
                        labels[base_id] = 1
                    elif equivalent == 'NO':
                        labels[base_id] = 0

        print(f"Loaded {len(labels)} Gemini labels")
        label_counts = {'correct': sum(labels.values()), 'incorrect': len(labels) - sum(labels.values())}
        print(f"Label distribution: {label_counts}")
    except Exception as e:
        print(f"Error loading Gemini TSV file: {e}")
    return labels


def read_entail_pmf(entail_tsv: Path) -> dict:
    """Read entail TSV and count [not_equiv(0), equiv(1)] per base_id."""
    counts: dict = defaultdict(lambda: [0, 0])
    if not entail_tsv.exists():
        print(f"Warning: entail file not found: {entail_tsv}")
        return {}
    with entail_tsv.open('r', encoding='utf-8') as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 7:
                continue
            base_id = parts[2]
            try:
                entails = int(parts[6])
            except ValueError:
                continue
            if entails == 1:
                counts[base_id][1] += 1
            else:
                counts[base_id][0] += 1
    
    # Convert to tuples
    return {k: (v[0], v[1]) for k, v in counts.items()}


def pmf_from_counts(cnt0: int, cnt1: int) -> list:
    total = cnt0 + cnt1
    if total <= 0:
        return [0.5, 0.5]
    return [cnt0 / total, cnt1 / total]


def main():
    task_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='KL-based uncertainty and AUROC for Falcon-7B entailment outputs using DeBERTa comparison results')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., falcon-7b, falcon7b)')
    
    parser.add_argument('--entail_dir', type=Path, 
                       default=task_root / 'outputs' / 'entailment',
                       help='Directory with Falcon-7B entail TSV outputs')
    parser.add_argument('--gt_dir', type=Path,
                       default=task_root / 'outputs' / 'gemini_eval',
                       help='Directory with Gemini comparison TSVs')
    parser.add_argument('--gt_file', type=Path, default=None,
                       help='Optional explicit path to Gemini TSV file')
    parser.add_argument('--entail_orig', type=Path, default=None,
                       help='Optional explicit path to orig entailment TSV')
    parser.add_argument('--entail_complement', type=Path, default=None,
                       help='Optional explicit path to complement entailment TSV')
    parser.add_argument('--pruned_csv', type=Path,
                       default=task_root.parent / 'data' / 'text_qa' / 'triviaqa' / 'llama_validation_trivia_qa.csv',
                       help='Path to pruned CSV file with valid sample IDs')
    args = parser.parse_args()

    model = args.model

    # Expect two files: orig and complement
    # Default naming: entail_falcon7b_equiv_orig.tsv and entail_falcon7b_equiv_complement.tsv
    if args.entail_orig:
        entail_orig = args.entail_orig
    else:
        entail_orig = args.entail_dir / f"entail_{model}_equiv_orig.tsv"
    
    if args.entail_complement:
        entail_neg = args.entail_complement
    else:
        entail_neg = args.entail_dir / f"entail_{model}_equiv_complement.tsv"

    print(f"Processing {model} using Gemini comparison results")
    
    # Load pruned IDs
    pruned_ids = load_pruned_ids(args.pruned_csv)
    if not pruned_ids:
        print("Warning: No pruned IDs loaded, continuing with all samples")
    
    print(f"Reading entailment (orig): {entail_orig}")
    orig_counts = read_entail_pmf(entail_orig)
    print(f"Reading entailment (neg):  {entail_neg}")
    neg_counts = read_entail_pmf(entail_neg)

    # Build the union of base_ids present in either set
    all_base_ids = sorted(set(orig_counts.keys()) | set(neg_counts.keys()))
    print(f"Total base_ids found in entailment files: {len(all_base_ids)}")
    
    # Filter to only pruned IDs if available
    if pruned_ids:
        base_ids = [bid for bid in all_base_ids if bid in pruned_ids]
        print(f"After pruning: {len(base_ids)} base_ids remain")
    else:
        base_ids = all_base_ids

    if len(base_ids) == 0:
        print("ERROR: No base_ids found in entailment files!")
        return

    # Compute KL divergences per base_id
    orig_kls = {}
    neg_kls = {}
    for bid in base_ids:
        oc0, oc1 = orig_counts.get(bid, (0, 0))
        nc0, nc1 = neg_counts.get(bid, (0, 0))
        p_orig = pmf_from_counts(oc0, oc1)
        p_neg = pmf_from_counts(nc0, nc1)
        kl_orig = kl_divergence(p_orig, [0.0, 1.0])
        kl_neg = kl_divergence(p_neg, [1.0, 0.0])
        orig_kls[bid] = kl_orig
        neg_kls[bid] = kl_neg

    # Combine to a single uncertainty score
    combined_scores = []
    combined_ids = []
    for bid in base_ids:
        k1 = orig_kls.get(bid, 0.0)
        k2 = neg_kls.get(bid, 0.0)
        combined_scores.append((k1 + k2) / 2.0)
        combined_ids.append(bid)

    y_prob = quantile_power_normalize(np.array(combined_scores))

    # Load DeBERTa comparison results
    if args.gt_file:
        gemini_tsv_file = args.gt_file
    else:
        gemini_tsv_file = args.gt_dir / f"{model}_triviaqa_gemini_evaluation.tsv"

    print(f"Reading Gemini results from: {gemini_tsv_file}")
    gemini_labels = load_gemini_labels(gemini_tsv_file)

    if len(gemini_labels) == 0:
        print(f"ERROR: No Gemini labels found in {gemini_tsv_file}")
        return

    # Calculate accuracy for matched samples (DeBERTa)
    matched_gemini = []
    for bid in combined_ids:
        if bid in gemini_labels:
            matched_gemini.append(gemini_labels[bid])
    
    if matched_gemini:
        gemini_accuracy = np.mean(matched_gemini)
        print(f"Accuracy for {len(matched_gemini)} samples with perturbations (Gemini): {gemini_accuracy:.4f}")

    # Match ids to DeBERTa labels for AUROC
    matched_probs = []
    matched_labels = []
    for bid, up in zip(combined_ids, y_prob):
        if bid in gemini_labels:
            matched_probs.append(up)
            matched_labels.append(gemini_labels[bid])
    print(f"Matched {len(matched_probs)} samples with both KL uncertainty and Gemini labels")

    if len(matched_probs) == 0:
        print("ERROR: No samples matched between entailment outputs and DeBERTa labels.")
        return

    y_true = np.array(matched_labels)
    
    # Show some statistics
    print(f"Mean KL uncertainty: {np.mean(combined_scores):.4f}")
    print(f"First 5 uncertainty scores: {y_prob[:5]}")
    print(f"First 5 ground truth labels: {y_true[:5]}")

    # AUROC: higher uncertainty should correlate with incorrect (label=0)
    if len(set(y_true)) > 1:
        auroc = roc_auc_score(y_true, matched_probs)
        print(f"\nAUROC for {model} (Gemini): {auroc:.4f}")
        
        # Compute separate AUROCs for orig and neg KL
        print("\n--- Component-wise AUROC ---")
        
        # Match orig KL scores
        matched_orig_kls = []
        matched_neg_kls = []
        matched_labels_comp = []
        for bid in combined_ids:
            if bid in gemini_labels:
                matched_orig_kls.append(orig_kls.get(bid, 0.0))
                matched_neg_kls.append(neg_kls.get(bid, 0.0))
                matched_labels_comp.append(gemini_labels[bid])
        
        y_true_comp = np.array(matched_labels_comp)
        
        # Normalize orig KL
        if len(matched_orig_kls) > 0 and len(set(y_true_comp)) > 1:
            y_orig_norm = quantile_power_normalize(np.array(matched_orig_kls))
            auroc_orig = roc_auc_score(y_true_comp, y_orig_norm)
            print(f"AUROC (Orig KL only):       {auroc_orig:.4f}")
        
        # Normalize neg KL
        if len(matched_neg_kls) > 0 and len(set(y_true_comp)) > 1:
            y_neg_norm = quantile_power_normalize(np.array(matched_neg_kls))
            auroc_neg = roc_auc_score(y_true_comp, y_neg_norm)
            print(f"AUROC (Complement KL only): {auroc_neg:.4f}")
        
        print(f"AUROC (Combined):           {auroc:.4f}")
    else:
        print("Cannot compute AUROC: only one class present in labels")

if __name__ == "__main__":
    main()
    
