#!/usr/bin/env python3
"""
Consolidate VQA uncertainty results (SNNE + Spectral Energy) across
vqa, okvqa, advqa, and vqarad datasets.

Supported OKVQA models (configure MODELS_MAP):
  - llava-v1.6-mistral-7b
  - pixtral-12b
  - qwen3-vl-8b

Creates pivot tables similar to the multilingual consolidation script.
Includes per-model accuracy computed from spectral energy pkl results.

Accuracy metrics:
  - vqa, okvqa, advqa: vqa_acc (continuous accuracy binarized at 0.5)
  - vqarad: vqarad_exact (exact match)

Outputs:
  - consolidated_vqa_pivot.csv

Usage:
    python consolidate_vqa_results.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# DATASETS = ['vqa', 'okvqa', 'advqa', 'vqarad']
DATASETS = ['okvqa']

# Internal model name (result CSV prefix) -> display name (pivot table + spectral summaries)
MODELS_MAP = {
    'llava-v1.6-mistral-7b-hf': 'llava-v1.6-mistral-7b',
    'Pixtral-12B-2409': 'pixtral-12b',
    'Qwen3-VL-8B-Instruct': 'qwen3-vl-8b',
}

# Alternate filename prefixes when results were saved under older model_name values
MODEL_FILE_PREFIXES = {
    'llava-v1.6-mistral-7b-hf': ['llava-v1.6-mistral-7b-hf', 'llava-hf_llava-v1.6-mistral-7b-hf'],
    'Pixtral-12B-2409': ['Pixtral-12B-2409', 'mistralai_Pixtral-12B-2409'],
    'Qwen3-VL-8B-Instruct': ['Qwen3-VL-8B-Instruct', 'Qwen_Qwen3-VL-8B-Instruct'],
}

# Display name -> model_name column in spectral energy master_summaries.csv
SPECTRAL_MODEL_MAP = {
    'llava-v1.6-mistral-7b': 'llava-v1.6-mistral-7b',
    'qwen3-vl-8b': 'qwen3-vl-8b',
    'pixtral-12b': 'pixtral-12b',
}

# Accuracy metric per dataset
ACCURACY_METRICS = {
    'vqa': 'vqa_acc',
    'okvqa': 'vqa_acc',
    'advqa': 'vqa_acc',
    'vqarad': 'vqarad_exact',
}

# Spectral energy master summary files (10 samples)
SPECTRAL_SUMMARY_TEMPLATE = 'outputs/consolidated/{dataset}_spectral_energy_weighted_results_10_master_summaries.csv'

# SNNE results directory
SNNE_RESULTS_DIR = 'snne_results'

# KLE results directory
KLE_RESULTS_DIR = 'kle_results'

# Graph baseline results directory
GRAPH_BASELINE_DIR = 'graph_baseline_results'

# Blackbox semantic entropy results directory
BLACKBOX_SE_DIR = 'blackbox_se_results'

# Accuracy sources: VQA accuracy JSONs (for vqa dataset)
VQA_ACCURACY_DIR = 'vqa_accuracy_results'
VQA_ACCURACY_MAP = {
    'llava-v1.6-mistral-7b': 'llava-v1.6-mistral-7b_vqa_accuracy.json',
    'qwen3-vl-8b': 'qwen3-vl-8b_vqa_accuracy.json',
    'pixtral-12b': 'pixtral-12b_vqa_accuracy.json',
}

# For okvqa/advqa/vqarad: accuracy comes from spectral energy pkl files
# Pattern: {dataset}_spectral_energy_weighted_results_10_deberta_entropy_confidence/
#           {dataset}_{short_model}_baseline_relaxed/results_baseline_relaxed_min_entropy_confidence.pkl
SPECTRAL_PKL_DIR_TEMPLATE = 'outputs/spectral_energy'
SHORT_MODEL_MAP = {
    'llava-v1.6-mistral-7b': 'llava7b',
    'pixtral-12b': 'pixtral12b',
    'qwen3-vl-8b': 'qwen8b',
}

OUTPUT_FILE = 'outputs/consolidated/consolidated_vqa_pivot.csv'

# Subsample seed used in SNNE / graph / KLE result filenames (reset_seed_temp1.0_seed10)
SUBSAMPLE_SEED = 10


def get_model_file_prefixes(model_internal):
    return MODEL_FILE_PREFIXES.get(model_internal, [model_internal])


def is_subsample_seed10_metrics_file(filename):
    """Match subsample seed 10 metrics CSVs (exclude per_example and other seeds)."""
    if '_per_example' in filename:
        return False
    if filename.endswith('_seed10.csv'):
        return True
    if re.search(rf'reset_seed_temp1\.0_seed{SUBSAMPLE_SEED}(?:_seed10)?\.csv$', filename):
        return True
    return False


def find_result_file(results_dir, dataset, model_internal, metric=None, extra_contains=None,
                     must_endwith=None):
    """Find best-matching metrics CSV, trying MODEL_FILE_PREFIXES in order."""
    if not os.path.isdir(results_dir):
        return None

    extra_contains = extra_contains or []
    candidates = []
    for prefix in get_model_file_prefixes(model_internal):
        for fname in os.listdir(results_dir):
            if not fname.startswith(f'{dataset}_{prefix}_'):
                continue
            if not fname.endswith('.csv'):
                continue
            if metric and metric not in fname:
                continue
            if not all(token in fname for token in extra_contains):
                continue
            if must_endwith and not fname.endswith(must_endwith):
                continue
            if not is_subsample_seed10_metrics_file(fname) and must_endwith is None:
                if 'reset_seed_temp' in fname and not is_subsample_seed10_metrics_file(fname):
                    continue
            candidates.append(fname)

    if not candidates:
        return None

    def rank(fname):
        score = 0
        if fname.startswith(f'{dataset}_{model_internal}_'):
            score += 20
        if 'google_' in fname or 'Qwen_Qwen' in fname:
            score += 10
        if is_subsample_seed10_metrics_file(fname):
            score += 5
        return score

    candidates.sort(key=rank, reverse=True)
    return os.path.join(results_dir, candidates[0])


def get_accuracy_from_json(dataset, model_display):
    """Get accuracy from VQA accuracy JSON for vqa dataset."""
    import json
    filename = VQA_ACCURACY_MAP.get(model_display)
    if not filename:
        return None
    filepath = os.path.join(VQA_ACCURACY_DIR, filename)
    if not os.path.exists(filepath):
        logger.warning(f"VQA accuracy JSON not found: {filepath}")
        return None
    try:
        with open(filepath) as f:
            data = json.load(f)
        return data.get('accuracy')
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None


def get_accuracy_from_pkl(dataset, model_display):
    """Get mean accuracy from spectral energy pkl results."""
    short_model = SHORT_MODEL_MAP.get(model_display)
    if not short_model:
        return None
    
    pkl_dir = SPECTRAL_PKL_DIR_TEMPLATE.format(dataset=dataset)
    # Final OKVQA BiG-SURE run uses entail_prob + min aggregation.
    candidates = [
        os.path.join(
            pkl_dir,
            f'{dataset}_{short_model}_seed10_ss123_entail_prob_min',
            'results_entail_prob_min_entropy_confidence.pkl',
        ),
        os.path.join(
            pkl_dir,
            f'{dataset}_{short_model}_baseline_relaxed',
            'results_baseline_relaxed_min_entropy_confidence.pkl',
        ),
    ]
    pkl_file = next((path for path in candidates if os.path.exists(path)), None)
    
    if pkl_file is None:
        logger.warning(f"Spectral energy pkl not found in candidates: {candidates}")
        return None
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        accs = [v['accuracy'] for v in data.values() if 'accuracy' in v]
        if accs:
            metric = ACCURACY_METRICS[dataset]
            if metric in ('vqa_acc',):
                # vqa_acc: continuous accuracy binarized at 0.5
                binary_accs = [1 if a >= 0.5 else 0 for a in accs]
                return sum(binary_accs) / len(binary_accs)
            elif metric == 'vqarad_exact':
                # vqarad_exact: exact match (already 0/1)
                return sum(accs) / len(accs)
            else:
                return sum(accs) / len(accs)
        return None
    except Exception as e:
        logger.error(f"Error reading {pkl_file}: {e}")
        return None


def get_accuracy(dataset, model_display):
    """Get accuracy for a dataset-model pair using the appropriate source."""
    if dataset == 'vqa':
        return get_accuracy_from_json(dataset, model_display)
    else:
        return get_accuracy_from_pkl(dataset, model_display)


def load_snne_results(dataset, model_internal, model_display):
    """Load SNNE entailment and lexical auroc values for a dataset-model pair."""
    snne_ent_auroc = None
    snne_lex_auroc = None

    metric = ACCURACY_METRICS[dataset]
    filepath = find_result_file(
        SNNE_RESULTS_DIR,
        dataset,
        model_internal,
        metric=metric,
        extra_contains=['reset_seed_temp1.0'],
    )

    if not filepath:
        logger.warning(f"SNNE result file not found for {dataset} {model_internal}")
        return snne_ent_auroc, snne_lex_auroc
    try:
        df = pd.read_csv(filepath)
        subset_entail = df[
            (df['method'] == 'snne') &
            (df['temperature'] == 1.0) &
            (df['similarity'] == 'entailment_sim')
        ]
        subset_lexical = df[
            (df['method'] == 'snne') &
            (df['temperature'] == 1.0) &
            (df['similarity'] == 'lexical_sim')
        ]
        
        if not subset_entail.empty:
            snne_ent_auroc = subset_entail.iloc[0]['auroc']
        if not subset_lexical.empty:
            snne_lex_auroc = subset_lexical.iloc[0]['auroc']
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
    
    return snne_ent_auroc, snne_lex_auroc


def load_spectral_energy_results(dataset):
    """Load spectral energy master summary for a dataset."""
    summary_file = SPECTRAL_SUMMARY_TEMPLATE.format(dataset=dataset)
    if not os.path.exists(summary_file):
        logger.warning(f"Spectral energy summary not found: {summary_file}")
        return None
    try:
        return pd.read_csv(summary_file)
    except Exception as e:
        logger.error(f"Error reading {summary_file}: {e}")
        return None


def load_graph_baseline_results(dataset, model_internal):
    """Load graph baseline results for a dataset-model pair.

    Returns a dict mapping graph_<method> -> auroc for all 5 baseline methods.
    """
    metric = ACCURACY_METRICS[dataset]
    filepath = find_result_file(
        GRAPH_BASELINE_DIR,
        dataset,
        model_internal,
        metric=metric,
        extra_contains=['reset_seed_temp1.0'],
    )

    if not filepath:
        logger.warning(f"Graph baseline file not found for {dataset} {model_internal}")
        return {}
    try:
        df = pd.read_csv(filepath)
        result = {}
        for _, row in df.iterrows():
            method = row['method']
            # Normalise eccentricity column name (strip threshold suffix)
            if method.startswith('eccentricity'):
                col = 'graph_eccentricity'
            else:
                col = f"graph_{method}"
            result[col] = row['auroc']
        return result
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return {}


def load_kle_results(dataset, model_internal):
    """Load KLE results and return heat average and best matern_kapp entries.

    Returns a dict with:
        kle_heat_avg_auroc     – average AUROC among all heat-kernel methods
        kle_matern_kapp_auroc  – best AUROC among full_klu_matern[n]_kappa=… rows
        kle_matern_kapp_method – method name with the winning hyperparams
    """
    metric = ACCURACY_METRICS[dataset]
    filepath = find_result_file(
        KLE_RESULTS_DIR,
        dataset,
        model_internal,
        metric=metric,
    )
    if not filepath:
        # Legacy KLE naming: deberta_concatTrue_vqa_acc_seed10.csv
        filepath = find_result_file(
            KLE_RESULTS_DIR,
            dataset,
            model_internal,
            metric=metric,
            must_endwith='_seed10.csv',
        )

    if not filepath:
        logger.warning(f"KLE file not found for {dataset} {model_internal}")
        return {}
    try:
        df = pd.read_csv(filepath)
        result = {}

        # Average over all heat-kernel methods
        heat_df = df[df['method'].str.contains('heat') & df['method'].str.contains('kernel')]
        if not heat_df.empty:
            result['kle_heat_avg_auroc'] = heat_df['auroc'].mean()

        # # Best matern_kapp: rows starting with 'full_klu_matern' (catches both matern_ and maternn_)
        # matern_df = df[df['method'].str.match(r'^full_klu_matern')]
        # if not matern_df.empty:
        #     best_matern = matern_df.loc[matern_df['auroc'].idxmax()]
        #     result['kle_matern_kapp_auroc'] = best_matern['auroc']
        #     result['kle_matern_kapp_method'] = best_matern['method']

        return result
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return {}


def load_blackbox_se_results(dataset, model_internal):
    """Load blackbox semantic entropy AUROC for a dataset-model pair.

    Uses method == 'blackbox_se' explicitly (not num_clusters).
    Returns dict with key 'DSE'.
    """
    metric = ACCURACY_METRICS[dataset]
    filepath = find_result_file(
        BLACKBOX_SE_DIR,
        dataset,
        model_internal,
        metric=metric,
    )

    if not filepath:
        logger.warning(f"Blackbox SE file not found for {dataset} {model_internal}")
        return {'DSE': None}
    try:
        df = pd.read_csv(filepath)
        row = df[df['method'] == 'blackbox_se']
        if row.empty:
            logger.warning(f"blackbox_se method not found in {filepath}")
            return {'DSE': None}
        return {'DSE': row.iloc[0]['auroc']}
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return {'DSE': None}


def consolidate_all():
    """Consolidate all results into a single pivot table."""
    results = {}
    
    for dataset in DATASETS:
        logger.info(f"Processing dataset: {dataset}")
        
        # Load spectral energy summary
        df_spec = load_spectral_energy_results(dataset)
        
        for model_internal, model_display in MODELS_MAP.items():
            key = (dataset, model_display)
            
            if key not in results:
                # Get SNNE results
                snne_ent, snne_lex = load_snne_results(dataset, model_internal, model_display)
                dse_results = load_blackbox_se_results(dataset, model_internal)
                
                # Get accuracy
                accuracy = get_accuracy(dataset, model_display)
                
                results[key] = {
                    'dataset': dataset,
                    'model': model_display,
                    'accuracy_metric': ACCURACY_METRICS[dataset],
                    'accuracy': accuracy,
                    'snne_entailment': snne_ent,
                    'snne_lexical': snne_lex,
                    'DSE': dse_results.get('DSE'),
                }

                # Add graph baseline results
                graph_results = load_graph_baseline_results(dataset, model_internal)
                results[key].update(graph_results)

                # Add KLE results (heat average and best matern_kapp)
                kle_results = load_kle_results(dataset, model_internal)
                results[key].update(kle_results)

            # Add spectral energy columns
            if df_spec is not None:
                spectral_model = SPECTRAL_MODEL_MAP.get(model_display, model_display)
                model_spec = df_spec[df_spec['model_name'] == spectral_model]
                if model_spec.empty and spectral_model != model_display:
                    logger.warning(
                        f"No spectral rows for {model_display} (looked up as {spectral_model})"
                    )
                for _, row in model_spec.iterrows():
                    sim = row['similarity']
                    mode = row.get('score_mode')
                    if pd.isna(mode) or mode == 'nan' or mode == 'N/A' or not mode:
                        col_name = f"spec_{sim}"
                    else:
                        col_name = f"spec_{sim}_{mode}"
                    results[key][col_name] = row['auroc']

    return results


def create_pivot_with_averages(df):
    """Sort by dataset and append per-dataset averages across models."""
    # Sort by dataset, then model
    df = df.sort_values(['dataset', 'model']).reset_index(drop=True)

    # Identify numeric vs string extra columns
    fixed_cols = ['dataset', 'model', 'accuracy_metric', 'accuracy']
    extra_cols = [c for c in df.columns if c not in fixed_cols]
    numeric_cols = [c for c in extra_cols if pd.api.types.is_numeric_dtype(df[c])]
    string_cols = [c for c in extra_cols if c not in numeric_cols]

    def make_avg_row(dataset_label, model_label, accuracy_metric_label, subset):
        row = {
            'dataset': dataset_label,
            'model': model_label,
            'accuracy_metric': accuracy_metric_label,
            'accuracy': subset['accuracy'].dropna().mean() if subset['accuracy'].notna().any() else np.nan,
        }
        for col in numeric_cols:
            valid_vals = subset[col].dropna()
            valid_vals = valid_vals[valid_vals >= 0]  # Exclude -1
            row[col] = valid_vals.mean() if len(valid_vals) > 0 else np.nan
        for col in string_cols:
            row[col] = np.nan
        return row

    # Compute per-dataset averages across models
    avg_rows = []
    for dataset in sorted(df['dataset'].unique()):
        dataset_data = df[df['dataset'] == dataset]
        accuracy_metric = dataset_data['accuracy_metric'].iloc[0]
        avg_rows.append(make_avg_row(dataset, 'AVG', accuracy_metric, dataset_data))

    # Overall average across all datasets and models
    avg_rows.append(make_avg_row('AVG', 'AVG', 'avg', df))

    df_avg = pd.DataFrame(avg_rows)
    df = pd.concat([df, df_avg], ignore_index=True)

    return df


def main():
    logger.info("Consolidating VQA results across vqa, okvqa, advqa, vqarad")
    logger.info(f"Models: {list(MODELS_MAP.values())}")
    logger.info(f"Datasets: {DATASETS}")
    
    # Consolidate
    results = consolidate_all()
    
    if not results:
        logger.error("No results found!")
        return 1
    
    # Build dataframe
    rows = list(results.values())
    df = pd.DataFrame(rows)
    
    # Reorder columns
    fixed_cols = ['dataset', 'model', 'accuracy_metric', 'accuracy', 'snne_entailment', 'snne_lexical', 'DSE']
    graph_cols = sorted([c for c in df.columns if c.startswith('graph_')])
    kle_cols = sorted([c for c in df.columns if c.startswith('kle_')])
    spec_cols = sorted([c for c in df.columns if c not in fixed_cols + graph_cols + kle_cols])
    df = df[fixed_cols + graph_cols + kle_cols + spec_cols]
    
    # Add averages
    df = create_pivot_with_averages(df)
    
    # Print
    print(f"\n{'='*100}")
    print("CONSOLIDATED VQA RESULTS — SNNE + Spectral Energy")
    print(f"{'='*100}")
    print(f"Datasets: {DATASETS}")
    print(f"Models: {list(MODELS_MAP.values())}")
    print(f"Accuracy metrics: vqa/okvqa/advqa=vqa_acc, vqarad=vqarad_exact")
    print(f"\n{'-'*100}")
    print(df.to_string(index=False))
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
    
    return 0


if __name__ == "__main__":
    exit(main())
