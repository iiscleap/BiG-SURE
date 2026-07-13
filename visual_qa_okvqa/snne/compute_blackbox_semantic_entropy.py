"""Compute blackbox semantic entropy (cluster_assignment_entropy) from precomputed cluster assignments.

Blackbox SE = entropy over the empirical cluster distribution:
    H = -sum_k (n_k / N) * log(n_k / N)

No log-likelihoods are used -- cluster membership counts alone determine the
uncertainty estimate.  This mirrors the 'cluster_assignment_entropy' function
from semantic_entropy.py.

A 'num_clusters' baseline (plain count of distinct semantic clusters) is also
reported for free.

Usage:
    python snne/compute_blackbox_semantic_entropy.py \\
        --dataset okvqa \\
        --num_generations 10 \\
        --model_name llava-v1.6-mistral-7b-hf \\
        --data_path sriramg/uncertainty/wandb/run-xxx/files \\
        --metric vqa_acc \\
        --metric_threshold 0.5

The script expects the wandb run directory to already contain:
  - validation_generations.pkl  (model outputs, with most_likely_answer['accuracy'])
  - uncertainty_measures.pkl    (must contain 'semantic_ids' key; if missing they
                                  are recomputed with DeBERTa entailment)
"""
import os
import logging

import pandas as pd
import wandb
from tqdm import tqdm

from snne.uncertainty.utils.eval_utils import auroc, auarc, aucpr, is_binary_list
from snne.uncertainty.utils import utils
from snne.uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from snne.uncertainty.utils.compute_utils import (
    get_parser,
    setup_wandb,
    load_precomputed_results,
    load_gemini_labels,
    load_vqa_labels,
    build_example_metadata,
    per_example_output_path,
    save_per_example_uncertainties,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
utils.setup_logger()

args = get_parser()
logging.info("Args: %s", args)
utils.set_all_seeds(args.random_seed)

setup_wandb(args, prefix='compute_blackbox_se')

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
precomputed_results = load_precomputed_results(args)
validation_generations = precomputed_results['validation_generations']
# list_semantic_ids[i] is the cluster assignment list for example i,
# already sliced to args.num_generations by load_precomputed_results.
list_semantic_ids = precomputed_results['list_semantic_ids']

# ---------------------------------------------------------------------------
# Gather accuracy labels (stored in validation_generations by the generator)
# ---------------------------------------------------------------------------
validation_is_true = []
for tid in tqdm(validation_generations, desc='Reading accuracy labels'):
    example = validation_generations[tid]
    acc = example.get('most_likely_answer', {}).get('accuracy', 0.0)
    validation_is_true.append(float(acc if acc is not None else 0.0))

# Optional label overrides (Gemini / external VQA JSON)
if getattr(args, 'gemini_json', None):
    new_labels = load_gemini_labels(args, validation_generations)
    if new_labels is not None and len(new_labels) == len(validation_is_true):
        validation_is_true = new_labels
        print(f"Overwrote validation_is_true with Gemini labels ({len(new_labels)}).")
    else:
        print("No Gemini labels applied (missing file or length mismatch).")

if getattr(args, 'vqa_json', None):
    new_labels = load_vqa_labels(args, validation_generations)
    if new_labels is not None and len(new_labels) == len(validation_is_true):
        validation_is_true = new_labels
        print(f"Overwrote validation_is_true with VQA labels ({len(new_labels)}).")
    else:
        print("No VQA labels applied (missing file or length mismatch).")

# Binarize continuous accuracy if needed (VQA / OKVQA)
if args.metric == 'vqa_acc':
    validation_is_true_binary = [
        1.0 if acc >= args.metric_threshold else 0.0
        for acc in validation_is_true
    ]
    print(
        f"VQA accuracy binarized at {args.metric_threshold}: "
        f"{sum(validation_is_true_binary):.0f} correct, "
        f"{len(validation_is_true_binary) - sum(validation_is_true_binary):.0f} incorrect"
    )
    validation_is_false = [1.0 - v for v in validation_is_true_binary]
elif args.metric == 'vqarad_exact':
    print(
        f"VQA-RAD exact (binary): {sum(validation_is_true):.0f} correct, "
        f"{len(validation_is_true) - sum(validation_is_true):.0f} incorrect"
    )
    validation_is_false = [1.0 - v for v in validation_is_true]
else:
    validation_is_false = [1.0 - v for v in validation_is_true]

is_binary = is_binary_list(validation_is_false)

# ---------------------------------------------------------------------------
# Compute blackbox semantic entropy (no log-likelihoods)
# ---------------------------------------------------------------------------
list_blackbox_se = []
for semantic_ids in tqdm(list_semantic_ids, desc='Computing blackbox SE'):
    list_blackbox_se.append(cluster_assignment_entropy(semantic_ids))

mean_se = sum(list_blackbox_se) / max(1, len(list_blackbox_se))
logging.info("Computed blackbox SE for %d examples. Mean SE = %.4f", len(list_blackbox_se), mean_se)

# ---------------------------------------------------------------------------
# Num-clusters baseline (a simpler count-based signal)
# ---------------------------------------------------------------------------
list_num_clusters = [float(len(set(sem_ids))) for sem_ids in list_semantic_ids]

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
list_method_name, list_auroc_vals, list_auarc_vals, list_aucpr_vals = [], [], [], []

for name, scores in [('blackbox_se', list_blackbox_se), ('num_clusters', list_num_clusters)]:
    _auroc = auroc(validation_is_false, scores) if is_binary else -1.0
    _auarc = auarc(scores, validation_is_true)
    _aucpr = aucpr(scores, validation_is_true)
    list_method_name.append(name)
    list_auroc_vals.append(_auroc)
    list_auarc_vals.append(_auarc)
    list_aucpr_vals.append(_aucpr)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
df_metrics = pd.DataFrame({
    'method': list_method_name,
    'auroc':  list_auroc_vals,
    'auarc':  list_auarc_vals,
    'prr':    list_aucpr_vals,
})
logging.info("\n%s", df_metrics.to_string(index=False))

os.makedirs('blackbox_se_results', exist_ok=True)
out_csv = (
    f'blackbox_se_results/{args.dataset}_{args.model_name}_'
    f'{args.num_generations}generations{args.suffix}_seed{args.random_seed}.csv'
)
df_metrics.to_csv(out_csv, index=False)
logging.info("Results saved → %s", out_csv)

method_uncertainties = {
    'blackbox_se': list_blackbox_se,
    'num_clusters': list_num_clusters,
}
example_metadata = build_example_metadata(validation_generations, validation_is_true, args)
save_per_example_uncertainties(
    per_example_output_path(out_csv),
    example_metadata,
    method_uncertainties,
)

wandb.log({'mean_blackbox_se': mean_se})
wandb.log({f'{m}_auroc': r for m, r in zip(list_method_name, list_auroc_vals)})
wandb.log({f'{m}_auarc': r for m, r in zip(list_method_name, list_auarc_vals)})
wandb.log({f'{m}_aucpr': r for m, r in zip(list_method_name, list_aucpr_vals)})

wandb.finish()
