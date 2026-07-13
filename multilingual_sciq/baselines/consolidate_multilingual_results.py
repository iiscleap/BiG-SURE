#!/usr/bin/env python3
"""
Consolidate multilingual uncertainty results (KLE, Semantic Entropy, SNNE,
Graph baselines, Spectral Energy) into CSVs split by metric.

Fixes for Spectral Energy:
- Parses spectral-energy method names directly from the folder suffix, e.g.
  sciq_apertus_gemini_seed10_baseline_relaxed -> spec_baseline_relaxed
  sciq_apertus_gemini_seed10_entail_prob_min  -> spec_entail_prob_min
- Keeps seed as a grouping key until seed-stat aggregation.
- Writes both long seed statistics and a wide pivot with mean/std/var columns.
- Logs seed coverage so missing/collapsed spectral runs are visible.

Expected spectral folder formats:
- {dataset}_{model}_{metric}_seed{N}_{spectral_method_suffix}
- {dataset}_{model}_{metric}_{spectral_method_suffix}  # legacy, no seed

Examples:
- sciq_aya_gemini_seed10_baseline_relaxed
- sciq_aya_gemini_seed10_entail_prob_mean
- triviaqa_hindi_krutrim2_prem_seed20_entail_over_noncontrad_geom_mean

Usage:
    python consolidate_multilingual_results.py \
        --results_dir ../results \
        --output_dir ../results/consolidated
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the scope of models and datasets to include.
ALLOWED_CONFIGS = [
    ('triviaqa', 'apertus'),
    ('triviaqa', 'aya'),
    ('sciq', 'apertus'),
    ('sciq', 'aya'),
    ('triviaqa_hindi', 'krutrim2'),
]

METRIC_NAMES = {'gemini', 'prem'}
SEED_RE = re.compile(r'^seed\d+$')

METHOD_ORDER = [
    'snne',
    'kle_heat_best', 'kle_matern_best', 'kle_heat_avg',
    'graph_degree', 'graph_eccen', 'graph_eigen', 'graph_lexsim', 'graph_numset',
    'se_semantic', 'se_predictive', 'se_cluster',
    # 'spec_baseline_relaxed', 'spec_baseline_strict',
    'spec_entail_prob', 'spec_entail_prob_min',
    # 'spec_entail_over_noncontrad', 'spec_entail_over_noncontrad_geom_mean', 'spec_entail_over_noncontrad_mean',
    # 'spec_noncontrad_prob', 'spec_noncontrad_prob_geom_mean', 'spec_noncontrad_prob_mean',
]


def is_allowed_config(dataset, model):
    """Check if a (dataset, model) pair is in the allowed scope."""
    return (dataset, model) in ALLOWED_CONFIGS


def _seed_index(parts):
    return next((i for i, part in enumerate(parts) if SEED_RE.match(part)), None)


def _metric_index(parts):
    """Return the first known metric index, if present."""
    return next((i for i, part in enumerate(parts) if part in METRIC_NAMES), None)


def parse_folder_name(folder_name):
    """
    Parse non-spectral experiment folders.

    Supported examples:
    - sciq_apertus_seed10_gemini
    - sciq_apertus_gemini_seed10
    - triviaqa_hindi_krutrim2_prem_seed20
    - sciq_apertus_gemini
    """
    parts = folder_name.split('_')
    if len(parts) < 3:
        return None

    seed_idx = _seed_index(parts)
    metric_idx = _metric_index(parts)

    try:
        if seed_idx is not None and metric_idx is not None:
            seed = parts[seed_idx]
            metric = parts[metric_idx]

            if metric_idx == seed_idx + 1:
                # {dataset}_{model}_seed10_gemini
                model = parts[seed_idx - 1]
                dataset = '_'.join(parts[:seed_idx - 1])
            elif metric_idx == seed_idx - 1:
                # {dataset}_{model}_gemini_seed10
                model = parts[metric_idx - 1]
                dataset = '_'.join(parts[:metric_idx - 1])
            else:
                # Fallback: metric marks the dataset/model boundary.
                model = parts[metric_idx - 1]
                dataset = '_'.join(parts[:metric_idx - 1])

        elif metric_idx is not None:
            # {dataset}_{model}_{metric}
            seed = None
            metric = parts[metric_idx]
            model = parts[metric_idx - 1]
            dataset = '_'.join(parts[:metric_idx - 1])

        else:
            # Backward-compatible fallback for unknown metric names.
            seed = None
            if parts[-1].startswith('seed'):
                seed = parts[-1]
                metric = parts[-2]
                model = parts[-3]
                dataset = '_'.join(parts[:-3])
            elif len(parts) >= 4 and parts[-2].startswith('seed'):
                seed = parts[-2]
                metric = parts[-1]
                model = parts[-3]
                dataset = '_'.join(parts[:-3])
            else:
                metric = parts[-1]
                model = parts[-2]
                dataset = '_'.join(parts[:-2])

        if not dataset or not model or not metric:
            return None
        return {'dataset': dataset, 'model': model, 'metric': metric, 'seed': seed}
    except IndexError:
        return None


def parse_spectral_energy_folder_name(folder_name):
    """
    Parse spectral-energy folder names and derive the method from the folder suffix.

    This is the important fix: do not depend on JSON fields like score_mode/combine
    for method naming, because those fields are often absent or inconsistent.

    Examples:
    - sciq_apertus_gemini_seed10_baseline_relaxed
      -> dataset=sciq, model=apertus, metric=gemini, seed=seed10,
         method=spec_baseline_relaxed
    - sciq_aya_gemini_seed10_entail_prob_min
      -> method=spec_entail_prob_min
    - triviaqa_aya_prem_baseline_relaxed  # legacy, no seed
      -> method=spec_baseline_relaxed
    """
    parts = folder_name.split('_')
    if len(parts) < 4:
        return None

    seed_idx = _seed_index(parts)

    try:
        if seed_idx is not None:
            if seed_idx < 3 or seed_idx == len(parts) - 1:
                return None

            metric = parts[seed_idx - 1]
            model = parts[seed_idx - 2]
            dataset = '_'.join(parts[:seed_idx - 2])
            seed = parts[seed_idx]
            method_suffix = '_'.join(parts[seed_idx + 1:])

        else:
            metric_idx = _metric_index(parts)
            if metric_idx is None or metric_idx < 2 or metric_idx == len(parts) - 1:
                return None

            metric = parts[metric_idx]
            model = parts[metric_idx - 1]
            dataset = '_'.join(parts[:metric_idx - 1])
            seed = None
            method_suffix = '_'.join(parts[metric_idx + 1:])

        if metric not in METRIC_NAMES:
            logger.warning("Unknown metric '%s' in spectral folder: %s", metric, folder_name)
        if not dataset or not model or not method_suffix:
            return None

        return {
            'dataset': dataset,
            'model': model,
            'metric': metric,
            'seed': seed,
            'method': f'spec_{method_suffix}',
            'spectral_method_suffix': method_suffix,
        }
    except IndexError:
        return None


def _coerce_float(value):
    """Return float(value), or np.nan if it is not numeric."""
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _valid_auroc(series):
    """Return AUROC values that are valid for aggregation."""
    values = pd.to_numeric(series, errors='coerce')
    return values.where(values >= 0)


def load_snne_summary(filepath):
    """Load SNNE summary CSV and extract AUROC values."""
    df = pd.read_csv(filepath)
    records = []

    for _, row in df.iterrows():
        records.append({
            'language': row['language'],
            'num_samples': row['num_samples'],
            'accuracy': row['accuracy'],
            'method': 'snne',
            'auroc': row['snne_auroc'],
            'auarc': row.get('snne_auarc', None),
            'aucpr': row.get('snne_aucpr', None),
        })

    return records


def load_kle_from_folder(folder_path):
    """
    Load KLE results from per-language CSV files in a folder.

    For heat-kernel variants we compute the average AUROC across all heat_t=*
    methods rather than selecting the single best.
    """
    records = []
    csv_files = sorted(folder_path.glob('*_seed*.csv'))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.empty or 'method' not in df.columns:
            continue

        lang = df['language'].iloc[0] if 'language' in df.columns else 'unknown'
        accuracy = df['accuracy'].iloc[0] if 'accuracy' in df.columns else None
        num_samples = len(df)

        heat_rows = df[df['method'].str.startswith('heat_t=', na=False) & ~df['method'].str.startswith('heatn_', na=False)]
        if not heat_rows.empty:
            mean_auroc = float(pd.to_numeric(heat_rows['auroc'], errors='coerce').mean())
            mean_auarc = None
            if 'auarc' in heat_rows.columns and heat_rows['auarc'].notna().any():
                mean_auarc = float(pd.to_numeric(heat_rows['auarc'], errors='coerce').mean())
            records.append({
                'language': lang,
                'num_samples': num_samples,
                'accuracy': accuracy,
                'method': 'kle_heat_avg',
                'auroc': mean_auroc,
                'auarc': mean_auarc,
                'aucpr': None,
            })

    return records


def load_graph_baseline_from_folder(folder_path):
    """Load graph baseline results from per-language CSV files in a folder."""
    records = []
    csv_files = sorted(folder_path.glob('*_seed*.csv'))

    method_map = {
        'degree_mat': 'graph_degree',
        'eccentricity_thr0.9': 'graph_eccen',
        'sum_eigv': 'graph_eigen',
        'lexical_sim': 'graph_lexsim',
        'num_set': 'graph_numset',
    }

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.empty or 'method' not in df.columns:
            continue

        lang = df['language'].iloc[0] if 'language' in df.columns else 'unknown'
        accuracy = df['accuracy'].iloc[0] if 'accuracy' in df.columns else None
        num_samples = len(df)

        for csv_method, our_method in method_map.items():
            method_row = df[df['method'] == csv_method]
            if not method_row.empty:
                row = method_row.iloc[0]
                records.append({
                    'language': lang,
                    'num_samples': num_samples,
                    'accuracy': accuracy,
                    'method': our_method,
                    'auroc': row['auroc'],
                    'auarc': row.get('auarc', None),
                    'aucpr': None,
                })

    return records


def load_semantic_entropy_summary(filepath):
    """Load Semantic Entropy summary CSV and extract AUROC values."""
    df = pd.read_csv(filepath)
    records = []

    for _, row in df.iterrows():
        lang = row['language']
        num_samples = row['num_samples']
        accuracy = row['accuracy']

        entropy_variants = [
            ('se_semantic', 'semantic_entropy'),
            ('se_predictive', 'predictive_entropy'),
            ('se_cluster', 'cluster_entropy'),
        ]

        for method_name, col_prefix in entropy_variants:
            auroc_col = f'{col_prefix}_auroc'
            auarc_col = f'{col_prefix}_auarc'
            aucpr_col = f'{col_prefix}_aucpr'

            if auroc_col in row:
                records.append({
                    'language': lang,
                    'num_samples': num_samples,
                    'accuracy': accuracy,
                    'method': method_name,
                    'auroc': row[auroc_col],
                    'auarc': row.get(auarc_col, None),
                    'aucpr': row.get(aucpr_col, None),
                })

    return records


def _extract_per_language_auroc(data):
    """
    Return dict: language -> AUROC from common spectral summary layouts.
    """
    auroc_by_language = (
        data.get('auroc_by_language')
        or data.get('auroc_per_language')
        or data.get('per_language_auroc')
        or {}
    )

    if isinstance(auroc_by_language, dict):
        return auroc_by_language

    # Defensive fallback for list-of-records layouts.
    if isinstance(auroc_by_language, list):
        out = {}
        for row in auroc_by_language:
            if not isinstance(row, dict):
                continue
            lang = row.get('language') or row.get('lang')
            if not lang:
                continue
            out[lang] = row
        return out

    return {}


def _extract_auroc_value(value):
    """Handle either plain float AUROC values or nested dict rows."""
    if isinstance(value, dict):
        for key in ('auroc', 'spectral_energy_auroc', 'score_auroc'):
            if key in value:
                return _coerce_float(value[key])
        return np.nan
    return _coerce_float(value)


def _extract_num_samples(total_samples, lang, value):
    """Handle scalar total_samples, per-language dict, or nested per-language rows."""
    if isinstance(value, dict):
        for key in ('num_samples', 'n_samples', 'total_samples'):
            if key in value:
                return value[key]
    if isinstance(total_samples, dict):
        return total_samples.get(lang)
    return total_samples


def _extract_accuracy(accuracy_by_language, lang, value):
    """Handle optional spectral accuracy if present."""
    if isinstance(value, dict):
        for key in ('accuracy', 'acc'):
            if key in value:
                return value[key]
    if isinstance(accuracy_by_language, dict):
        return accuracy_by_language.get(lang)
    return None


def load_spectral_energy_summary(filepath, method_name):
    """
    Load Spectral Energy summary JSON and extract AUROC values.

    method_name is supplied by parse_spectral_energy_folder_name(), so method
    identity is stable even when the JSON lacks score_mode/combine.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    auroc_by_language = _extract_per_language_auroc(data)
    total_samples = data.get('total_samples', data.get('num_samples', None))
    accuracy_by_language = data.get('accuracy_by_language', data.get('accuracy_per_language', None))

    if not auroc_by_language:
        logger.warning("No per-language AUROC found in spectral summary: %s", filepath)
        return records

    for lang, raw_value in auroc_by_language.items():
        auroc = _extract_auroc_value(raw_value)
        if pd.isna(auroc) or auroc < 0:
            continue

        records.append({
            'language': lang,
            'num_samples': _extract_num_samples(total_samples, lang, raw_value),
            'accuracy': _extract_accuracy(accuracy_by_language, lang, raw_value),
            'method': method_name,
            'auroc': auroc,
            'auarc': None,
            'aucpr': None,
        })

    return records


def consolidate_results(results_dir):
    """
    Consolidate all summary CSVs/JSONs into a single dataframe,
    filtered to only include allowed (dataset, model) configs.
    """
    results_path = Path(results_dir)
    all_records = []

    # Process SNNE with summary files.
    snne_path = results_path / 'snne'
    if snne_path.exists():
        for exp_folder in sorted(snne_path.iterdir()):
            if not exp_folder.is_dir():
                continue

            summary_path = exp_folder / 'snne_entailment_summary.csv'
            if not summary_path.exists():
                continue

            parsed = parse_folder_name(exp_folder.name)
            if parsed is None or not is_allowed_config(parsed['dataset'], parsed['model']):
                continue

            try:
                records = load_snne_summary(summary_path)
                for rec in records:
                    rec.update({
                        'dataset': parsed['dataset'],
                        'model': parsed['model'],
                        'metric': parsed['metric'],
                        'seed': parsed['seed'],
                        'source_file': str(summary_path),
                    })
                all_records.extend(records)
                logger.info("Loaded %d SNNE records from %s", len(records), summary_path)
            except Exception as e:
                logger.error("Error loading SNNE %s: %s", summary_path, e)

    # Process KLE with per-language CSV files.
    kle_path = results_path / 'kle'
    if kle_path.exists():
        for exp_folder in sorted(kle_path.iterdir()):
            if not exp_folder.is_dir():
                continue

            parsed = parse_folder_name(exp_folder.name)
            if parsed is None or not is_allowed_config(parsed['dataset'], parsed['model']):
                continue

            try:
                records = load_kle_from_folder(exp_folder)
                for rec in records:
                    rec.update({
                        'dataset': parsed['dataset'],
                        'model': parsed['model'],
                        'metric': parsed['metric'],
                        'seed': parsed['seed'],
                        'source_file': str(exp_folder),
                    })
                all_records.extend(records)
                logger.info("Loaded %d KLE records from %s", len(records), exp_folder)
            except Exception as e:
                logger.error("Error loading KLE %s: %s", exp_folder, e)

    # Process Semantic Entropy summary files.
    se_path = results_path / 'semantic_entropy'
    if se_path.exists():
        for exp_folder in sorted(se_path.iterdir()):
            if not exp_folder.is_dir():
                continue

            summary_path = exp_folder / 'semantic_entropy_summary.csv'
            if not summary_path.exists():
                continue

            parsed = parse_folder_name(exp_folder.name)
            if parsed is None or not is_allowed_config(parsed['dataset'], parsed['model']):
                continue

            try:
                records = load_semantic_entropy_summary(summary_path)
                for rec in records:
                    rec.update({
                        'dataset': parsed['dataset'],
                        'model': parsed['model'],
                        'metric': parsed['metric'],
                        'seed': parsed['seed'],
                        'source_file': str(summary_path),
                    })
                all_records.extend(records)
                logger.info("Loaded %d Semantic Entropy records from %s", len(records), summary_path)
            except Exception as e:
                logger.error("Error loading Semantic Entropy %s: %s", summary_path, e)

    # Process graph baseline with per-language CSV files.
    graph_path = results_path / 'graph_baselines'
    if graph_path.exists():
        for exp_folder in sorted(graph_path.iterdir()):
            if not exp_folder.is_dir():
                continue

            parsed = parse_folder_name(exp_folder.name)
            if parsed is None or not is_allowed_config(parsed['dataset'], parsed['model']):
                continue

            try:
                records = load_graph_baseline_from_folder(exp_folder)
                for rec in records:
                    rec.update({
                        'dataset': parsed['dataset'],
                        'model': parsed['model'],
                        'metric': parsed['metric'],
                        'seed': parsed['seed'],
                        'source_file': str(exp_folder),
                    })
                all_records.extend(records)
                logger.info("Loaded %d graph baseline records from %s", len(records), exp_folder)
            except Exception as e:
                logger.error("Error loading graph baseline %s: %s", exp_folder, e)

    # Process Spectral Energy JSON summaries.
    spectral_path = results_path / 'spectral_energy'
    if spectral_path.exists():
        for exp_folder in sorted(spectral_path.iterdir()):
            if not exp_folder.is_dir():
                continue

            json_files = sorted(exp_folder.glob('summary_*.json'))
            if not json_files:
                logger.debug("No JSON summary found in: %s", exp_folder)
                continue

            parsed = parse_spectral_energy_folder_name(exp_folder.name)
            if parsed is None:
                logger.warning("Could not parse spectral energy folder name: %s", exp_folder.name)
                continue

            if not is_allowed_config(parsed['dataset'], parsed['model']):
                logger.debug("Skipping spectral %s - not in allowed configs", exp_folder.name)
                continue

            for summary_path in json_files:
                try:
                    records = load_spectral_energy_summary(summary_path, parsed['method'])
                    for rec in records:
                        rec.update({
                            'dataset': parsed['dataset'],
                            'model': parsed['model'],
                            'metric': parsed['metric'],
                            'seed': parsed['seed'],
                            'source_file': str(summary_path),
                        })
                    all_records.extend(records)
                    logger.info(
                        "Loaded %d spectral energy records as %s from %s",
                        len(records), parsed['method'], summary_path,
                    )
                except Exception as e:
                    logger.error("Error loading spectral energy %s: %s", summary_path, e)
    else:
        logger.warning("Spectral energy directory not found: %s", spectral_path)

    if not all_records:
        logger.error("No records found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    cols = [
        'dataset', 'model', 'metric', 'seed', 'language', 'method',
        'num_samples', 'accuracy', 'auroc', 'auarc', 'aucpr', 'source_file',
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def compute_seed_statistics(df):
    """
    Compute mean, variance, and std of AUROC across seeds for each
    (dataset, model, metric, language, method) group.

    Notes:
    - Rows with invalid AUROC values are ignored.
    - Duplicate rows for the same seed/method/language are collapsed before
      aggregation. This protects against multiple summary_*.json files in one
      spectral folder.
    - Variance/std are sample variance/std (ddof=1). For one seed, they are 0.0.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df['auroc'] = _valid_auroc(df['auroc'])

    if 'seed' in df.columns:
        seeded = df[df['seed'].notna()].copy()
        unseeded = df[df['seed'].isna()].copy()

        if not unseeded.empty:
            logger.warning(
                "Dropping %d unseeded records before seed-stat aggregation. "
                "These cannot contribute to var/std across seeds.",
                len(unseeded),
            )
        df = seeded

        dedupe_cols = ['dataset', 'model', 'metric', 'language', 'method', 'seed']
        before = len(df)
        df = df.drop_duplicates(subset=dedupe_cols, keep='last')
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d duplicate seed records before aggregation", dropped)

    group_cols = ['dataset', 'model', 'metric', 'language', 'method']

    def _var(x):
        values = x.dropna()
        return values.var(ddof=1) if len(values) > 1 else 0.0

    def _std(x):
        values = x.dropna()
        return values.std(ddof=1) if len(values) > 1 else 0.0

    stats = (
        df.groupby(group_cols, dropna=False)['auroc']
        .agg(mean='mean', var=_var, std=_std, n_seeds=lambda x: x.notna().sum())
        .reset_index()
    )

    return stats.sort_values(group_cols).reset_index(drop=True)


def _ordered_method_columns(methods, suffix=''):
    """Return methods in METHOD_ORDER, then any extras."""
    suffix = f'_{suffix}' if suffix else ''
    methods = list(methods)
    ordered = [f'{m}{suffix}' for m in METHOD_ORDER if f'{m}{suffix}' in methods]
    extras = sorted(m for m in methods if m not in ordered)
    return ordered + extras


def create_accuracy_df(raw_df):
    """Return one optional accuracy column per dataset/model/metric/language."""
    if raw_df is None or raw_df.empty or 'accuracy' not in raw_df.columns:
        return None

    tmp = raw_df[['dataset', 'model', 'metric', 'language', 'accuracy']].copy()
    tmp['accuracy'] = pd.to_numeric(tmp['accuracy'], errors='coerce')
    tmp = tmp.dropna(subset=['accuracy'])
    if tmp.empty:
        return None

    return (
        tmp.groupby(['dataset', 'model', 'metric', 'language'], dropna=False)['accuracy']
        .mean()
        .reset_index()
    )


def create_mean_pivot_table(stats_df, accuracy_df=None):
    """Pivot seed-averaged mean AUROC with optional accuracy column."""
    if stats_df.empty:
        return stats_df

    index_cols = ['dataset', 'model', 'metric', 'language']
    pivot = stats_df.pivot_table(index=index_cols, columns='method', values='mean').reset_index()

    method_cols = _ordered_method_columns([c for c in pivot.columns if c not in index_cols])
    if accuracy_df is not None and not accuracy_df.empty:
        pivot = pivot.merge(accuracy_df, on=index_cols, how='left')
        cols = index_cols + ['accuracy'] + method_cols
    else:
        cols = index_cols + method_cols

    pivot = pivot[[c for c in cols if c in pivot.columns]]
    return pivot.sort_values(['dataset', 'model', 'language']).reset_index(drop=True)


def create_wide_seed_stats_table(stats_df, accuracy_df=None):
    """
    Wide table with {method}_mean, {method}_std, {method}_var per model-language.
    """
    if stats_df.empty:
        return stats_df

    index_cols = ['dataset', 'model', 'metric', 'language']
    methods = sorted(stats_df['method'].unique())

    wide = stats_df[index_cols + ['n_seeds']].drop_duplicates()
    wide = wide.groupby(index_cols, as_index=False)['n_seeds'].max()

    for method in methods:
        method_stats = stats_df[stats_df['method'] == method][index_cols + ['mean', 'std', 'var']]
        method_stats = method_stats.rename(columns={
            'mean': f'{method}_mean',
            'std': f'{method}_std',
            'var': f'{method}_var',
        })
        wide = wide.merge(method_stats, on=index_cols, how='left')

    method_cols = []
    stat_suffixes = ['_mean', '_std', '_var']
    for method in METHOD_ORDER:
        if method in methods:
            method_cols.extend([f'{method}{s}' for s in stat_suffixes if f'{method}{s}' in wide.columns])
    for method in methods:
        if method not in METHOD_ORDER:
            method_cols.extend([f'{method}{s}' for s in stat_suffixes if f'{method}{s}' in wide.columns])

    if accuracy_df is not None and not accuracy_df.empty:
        wide = wide.merge(accuracy_df, on=index_cols, how='left')
        cols = index_cols + ['accuracy', 'n_seeds'] + method_cols
    else:
        cols = index_cols + ['n_seeds'] + method_cols

    wide = wide[[c for c in cols if c in wide.columns]]
    return wide.sort_values(['dataset', 'model', 'language']).reset_index(drop=True)


def append_dataset_averages(pivot, numeric_cols):
    """Append AVG rows averaging across datasets for each language."""
    if pivot.empty:
        return pivot

    pivot_valid = pivot.copy()
    for col in numeric_cols:
        pivot_valid[col] = pd.to_numeric(pivot_valid[col], errors='coerce')
        pivot_valid[col] = pivot_valid[col].where(pivot_valid[col] >= 0)

    avg_rows = []
    for lang in sorted(pivot_valid['language'].dropna().unique()):
        lang_data = pivot_valid[pivot_valid['language'] == lang]
        avg_row = {
            'dataset': 'AVG',
            'model': 'all',
            'metric': pivot['metric'].iloc[0] if not pivot.empty else '',
            'language': lang,
        }
        for col in numeric_cols:
            valid_vals = lang_data[col].dropna()
            avg_row[col] = valid_vals.mean() if len(valid_vals) > 0 else np.nan
        avg_rows.append(avg_row)

    if avg_rows:
        return pd.concat([pivot, pd.DataFrame(avg_rows)], ignore_index=True)
    return pivot


def create_pivot_table(stats_df, raw_df=None):
    """
    Create a pivot table with mean AUROC averaged across seeds as columns.
    This file intentionally contains means only. Use *_pivot_seed_stats.csv for
    mean/std/var columns.
    """
    if stats_df.empty:
        return stats_df

    accuracy_df = create_accuracy_df(raw_df)
    pivot = create_mean_pivot_table(stats_df, accuracy_df)
    numeric_cols = [c for c in pivot.columns if c not in ['dataset', 'model', 'metric', 'language']]
    return append_dataset_averages(pivot, numeric_cols)


def log_seed_coverage(df, label='all'):
    """Log seed coverage by dataset/model/metric/method for debugging."""
    if df.empty or 'seed' not in df.columns:
        return

    coverage = (
        df.dropna(subset=['seed'])
        .drop_duplicates(['dataset', 'model', 'metric', 'method', 'seed'])
        .groupby(['dataset', 'model', 'metric', 'method'], dropna=False)['seed']
        .agg(lambda s: ','.join(sorted(s)))
        .reset_index(name='seeds')
    )
    if coverage.empty:
        logger.warning("No seeded records found for %s", label)
        return

    coverage['n_seeds'] = coverage['seeds'].apply(lambda s: len(s.split(',')) if s else 0)
    low = coverage[coverage['n_seeds'] < 2]
    if not low.empty:
        logger.warning(
            "Some %s groups have <2 seeds, so std/var will be 0 or NaN-equivalent:\n%s",
            label,
            low.to_string(index=False),
        )

    logger.info("Seed coverage for %s:\n%s", label, coverage.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Consolidate multilingual uncertainty results")
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing method subdirectories')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save consolidated CSVs')
    parser.add_argument('--save_raw', action='store_true',
                        help='Also save the raw long-format consolidated records')
    args = parser.parse_args()

    logger.info("Consolidating results from %s", args.results_dir)
    logger.info("Scope: apertus & aya on triviaqa/sciq + krutrim2 on triviaqa_hindi")

    df = consolidate_results(args.results_dir)
    if df.empty:
        logger.error("No results to consolidate")
        return 1

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.save_raw:
        raw_path = output_path / 'consolidated_aurocs_raw_records.csv'
        df.to_csv(raw_path, index=False)
        logger.info("Saved raw records to %s", raw_path)

    spectral_raw = df[df['method'].astype(str).str.startswith('spec_')].copy()
    log_seed_coverage(spectral_raw, label='spectral-energy')

    for metric_name in ['gemini', 'prem']:
        df_metric = df[df['metric'] == metric_name].copy()

        if df_metric.empty:
            logger.warning("No results found for metric: %s", metric_name)
            continue

        df_metric = df_metric.sort_values(['dataset', 'model', 'language', 'method', 'seed']).reset_index(drop=True)

        stats = compute_seed_statistics(df_metric)
        stats_path = output_path / f'consolidated_aurocs_{metric_name}_seed_stats.csv'
        stats.to_csv(stats_path, index=False)
        logger.info("Saved %s seed statistics to %s", metric_name, stats_path)

        pivot = create_pivot_table(stats, raw_df=df_metric)
        pivot_path = output_path / f'consolidated_aurocs_{metric_name}_pivot.csv'
        pivot.to_csv(pivot_path, index=False)
        logger.info("Saved %s mean-only pivot table to %s", metric_name, pivot_path)

        accuracy_df = create_accuracy_df(df_metric)
        wide_stats = create_wide_seed_stats_table(stats, accuracy_df)
        wide_stats_path = output_path / f'consolidated_aurocs_{metric_name}_pivot_seed_stats.csv'
        wide_stats.to_csv(wide_stats_path, index=False)
        logger.info("Saved %s wide seed stats to %s", metric_name, wide_stats_path)

        print(f"\n{'=' * 80}")
        print(f"PIVOT TABLE (mean AUROC across seeds) — {metric_name.upper()}")
        print(f"{'=' * 80}")
        print(f"Total seed-stat rows: {len(stats)} | Methods: {stats['method'].unique().tolist()}")
        print(f"{'-' * 80}")
        print(pivot.to_string(index=False))
        print(f"\n{'-' * 80}")
        print(f"Mean-only pivot:      {pivot_path}")
        print(f"Seed stats (long):   {stats_path}")
        print(f"Seed stats (wide):   {wide_stats_path}")
        print("Use the *_pivot_seed_stats.csv file for mean/std/var columns.")

    print(f"\n{'=' * 80}")
    print("CONSOLIDATION COMPLETE")
    print(f"{'=' * 80}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
