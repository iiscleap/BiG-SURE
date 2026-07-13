#!/usr/bin/env python3
"""Compute squad and gemini accuracy for VQA runs and save JSON files.

This script expects a `validation_generations.pkl` file in the `--vanilla_run_dir`
(or a custom `--predictions` path) which maps example ids to generation dicts
(created by `snne/generate_vqa_answers.py`). It uses the metadata CSV to get
reference answers (column `answers_joined`) and picks the most-common answer as
reference for each question.

Outputs:
  - squad_accuracy.json: Computed using SQuAD metric
  - gemini_accuracy.json: Extracted from existing pkl (most_likely_answer['accuracy'])

Example:
  python scripts/compute_vqa_accuracies.py \
    --vanilla_run_dir run-20251204_015024-oaqyie5m \
    --metadata snne/vqav2/metadata.csv \
    --dataset vqa
"""

import argparse
import os
import pickle
import json
from collections import Counter
from typing import Dict, Tuple

import pandas as pd

from snne.uncertainty.utils.metric_utils import get_metric


def load_metadata_most_common(metadata_csv: str) -> Dict[str, str]:
    df = pd.read_csv(metadata_csv)
    per = {}
    if 'question_id' in df.columns:
        for _, row in df.iterrows():
            qid = str(row['question_id'])
            answers_joined = row.get('answers_joined')
            if pd.isna(answers_joined) or not isinstance(answers_joined, str):
                continue
            parts = [a.strip() for a in answers_joined.split(';') if a.strip()]
            if not parts:
                continue
            # choose most common answer
            most_common = Counter(parts).most_common(1)[0][0]
            per[qid] = most_common
    else:
        raise ValueError('metadata CSV does not contain question_id column')
    return per


def load_predictions_from_pickle(pkl_path: str) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Load predictions and gemini accuracies from pickle.
    
    Returns:
        Tuple of (predictions dict, gemini_accuracies dict)
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    preds = {}
    gemini_accs = {}
    # data is expected to be dict keyed by example id
    for k, v in data.items():
        # prefer low-temp most_likely_answer.response
        pred = None
        gemini_acc = None
        if isinstance(v, dict):
            ml = v.get('most_likely_answer')
            if ml and isinstance(ml, dict):
                pred = ml.get('response')
                # extract gemini accuracy if present
                gemini_acc = ml.get('accuracy')
            # fallback: responses list
            if pred is None:
                responses = v.get('responses')
                if responses and isinstance(responses, list) and len(responses) > 0:
                    first = responses[0]
                    if isinstance(first, (list, tuple)) and len(first) >= 1:
                        pred = first[0]
                    elif isinstance(first, dict) and 'response' in first:
                        pred = first['response']
        if pred is None:
            # try if v itself is a string
            if isinstance(v, str):
                pred = v
        if pred is not None:
            preds[str(k)] = str(pred)
        if gemini_acc is not None:
            gemini_accs[str(k)] = float(gemini_acc)
    return preds, gemini_accs


def find_pkl(run_dir):
    """Find validation_generations.pkl in various possible locations.
    
    Searches in order:
    1. run_dir/validation_generations.pkl (direct)
    2. malaymilindp/uncertainty/wandb/run_dir/files/validation_generations.pkl
    3. Any .pkl file in run_dir or wandb structure
    """
    from glob import glob
    
    # Try direct path
    if os.path.isfile(run_dir) and run_dir.endswith('.pkl'):
        return run_dir
    
    # Try run_dir/validation_generations.pkl
    candidate = os.path.join(run_dir, 'validation_generations.pkl')
    if os.path.exists(candidate):
        return candidate
    
    # Try wandb structure: malaymilindp/uncertainty/wandb/<run_dir>/files/validation_generations.pkl
    user = os.environ.get('USER', 'malaymilindp')
    wandb_candidate = os.path.join(user, 'uncertainty', 'wandb', run_dir, 'files', 'validation_generations.pkl')
    if os.path.exists(wandb_candidate):
        return wandb_candidate
    
    # Try this task's portable W&B directory if run_dir is a run ID.
    if run_dir.startswith('run-'):
        task_root = Path(__file__).resolve().parents[1]
        local_wandb = task_root / 'outputs' / 'wandb' / 'wandb' / run_dir / 'files' / 'validation_generations.pkl'
        if local_wandb.exists():
            return str(local_wandb)
    
    # Fallback: look for any .pkl in run_dir
    pkl_files = glob(os.path.join(run_dir, '*.pkl'))
    if pkl_files:
        return pkl_files[0]
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Compute VQA squad accuracy JSON')
    parser.add_argument('--vanilla_run_dir', type=str, required=True, help='Run directory or run ID (e.g., run-20251204_015024-oaqyie5m)')
    parser.add_argument('--predictions', type=str, default=None, help='Path to predictions pickle (overrides auto detection)')
    parser.add_argument('--metadata', type=str, default='snne/vqav2/metadata.csv', help='Path to VQA metadata CSV')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--dataset', type=str, default='vqa', help='Dataset name to save in JSON')
    parser.add_argument('--skip_gemini', action='store_true', help='Skip extracting gemini accuracy from pickle')
    parser.add_argument('--gemini_output', type=str, default=None, help='Output JSON path for Gemini accuracy')
    args = parser.parse_args()

    # detect predictions file
    pred_path = args.predictions
    if pred_path is None:
        pred_path = find_pkl(args.vanilla_run_dir)
        if not pred_path:
            raise FileNotFoundError(
                f'No predictions file found for {args.vanilla_run_dir}. '
                f'Searched in run dir, wandb structure, and .pkl files. '
                f'Pass --predictions explicitly.'
            )

    # load metadata
    if not os.path.exists(args.metadata):
        raise FileNotFoundError(f'Metadata CSV not found: {args.metadata}')

    print(f'Loading metadata from {args.metadata}...')
    refs = load_metadata_most_common(args.metadata)
    print(f'Loaded {len(refs)} ground-truth entries (most-common answers).')

    # load predictions and gemini accuracies
    print(f'Loading predictions from {pred_path}...')
    preds, gemini_accs = load_predictions_from_pickle(pred_path)
    print(f'Loaded {len(preds)} predictions and {len(gemini_accs)} gemini accuracies.')

    # build examples and compute squad metric
    metric = get_metric('squad')

    per_example = {}
    total = 0
    correct = 0

    for qid, pred in preds.items():
        # only evaluate when we have a reference entry for this qid
        if qid not in refs:
            continue
        ref = refs[qid]
        # build example in the shape expected by metric_utils.get_metric
        example = {
            'id': qid,
            'question': '',
            'answers': {'text': [ref], 'answer_start': []}
        }
        score = metric(pred, example)
        val = 1 if float(score) >= 0.5 else 0
        per_example[qid] = val
        total += 1
        correct += val

    accuracy = (correct / total) if total > 0 else 0.0

    out = {
        'total': total,
        'correct': correct,
        'accuracy': round(accuracy, 8),
        'dataset': args.dataset,
        'per_example': per_example,
    }

    # Default output path: use the directory containing the predictions pickle if no output specified
    if args.output:
        out_path = args.output
    else:
        pkl_dir = os.path.dirname(pred_path)
        out_path = os.path.join(pkl_dir, 'squad_accuracy.json')
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, separators=(',', ':'), ensure_ascii=False)

    print(f'Wrote squad accuracy to {out_path} (total={total}, correct={correct}, accuracy={accuracy:.4f})')

    # Extract Gemini-based VQA accuracy from pickle (unless skipped)
    if not args.skip_gemini:
        print('\nExtracting Gemini accuracy from pickle...')
        per_example_gemini = {}
        total_g = 0
        correct_g = 0

        for qid in gemini_accs.keys():
            acc_val = gemini_accs[qid]
            # convert to binary (1 or 0)
            val = 1 if float(acc_val) >= 0.5 else 0
            per_example_gemini[qid] = val
            total_g += 1
            correct_g += val

        accuracy_g = (correct_g / total_g) if total_g > 0 else 0.0
        gem_out = {
            'total': total_g,
            'correct': correct_g,
            'accuracy': round(accuracy_g, 8),
            'dataset': args.dataset,
            'per_example': per_example_gemini,
        }

        # Default gemini output path: same directory as squad accuracy
        if args.gemini_output:
            gem_out_path = args.gemini_output
        else:
            pkl_dir = os.path.dirname(pred_path)
            gem_out_path = os.path.join(pkl_dir, 'gemini_accuracy.json')
        
        os.makedirs(os.path.dirname(gem_out_path), exist_ok=True)
        with open(gem_out_path, 'w', encoding='utf-8') as f:
            json.dump(gem_out, f, separators=(',', ':'), ensure_ascii=False)

        print(f'Wrote gemini accuracy to {gem_out_path} (total={total_g}, correct={correct_g}, accuracy={accuracy_g:.4f})')


if __name__ == '__main__':
    main()
