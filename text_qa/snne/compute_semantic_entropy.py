import os
import json
import logging
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import wandb

from snne.uncertainty.utils.eval_utils import auroc, auarc, aucpr, is_binary_list
from snne.uncertainty.utils import utils
from snne.uncertainty.utils.compute_utils import get_parser, setup_wandb, load_precomputed_results, load_gemini_labels
from snne.uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from snne.uncertainty.utils.metric_utils import get_metric

# Set up log
utils.setup_logger()

# Parse arguments
args = get_parser()
logging.info("Args: %s", args)
utils.set_all_seeds(args.random_seed)

# Set up wandb
setup_wandb(args, prefix='compute_semantic_entropy')

# Load pre-computed results
precomputed_results = load_precomputed_results(args)
validation_generations = precomputed_results['validation_generations']
list_semantic_ids = precomputed_results['list_semantic_ids']

if list_semantic_ids is None:
    logging.info("Semantic IDs not found in precomputed results. Computing them now...")
    from snne.uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids_using_entailment, EntailmentDeberta
    
    entailment_model_for_ids = EntailmentDeberta()
    list_semantic_ids = []
    for tid in tqdm(validation_generations, desc="Computing Semantic IDs"):
        example = validation_generations[tid]
        full_responses = example["responses"]
        responses = [r[0] for r in full_responses]
        responses = responses[:args.num_generations]
        
        ids = get_semantic_ids_using_entailment(
            responses, 
            entailment_model_for_ids,
            strict_entailment=getattr(args, 'strict_entailment', True),
            cluster_method=getattr(args, 'cluster_method', 'greedy'),
            example=example
        )
        list_semantic_ids.append(ids)
    
    logging.info(f"Computed semantic IDs for {len(list_semantic_ids)} examples.")
    
    if args.subsample is None:
        uncertainty_pkl_path = os.path.join(args.data_path, 'uncertainty_measures.pkl')
        try:
            logging.info(f"Saving computed semantic IDs to {uncertainty_pkl_path}")
            full_results = {}
            if os.path.isfile(uncertainty_pkl_path):
                with open(uncertainty_pkl_path, 'rb') as infile:
                    full_results = pickle.load(infile)
            full_results['semantic_ids'] = list_semantic_ids
            full_results['schema_version'] = 1
            full_results['question_ids'] = [str(tid) for tid in validation_generations]
            with open(uncertainty_pkl_path, 'wb') as outfile:
                pickle.dump(full_results, outfile)
            logging.info("Successfully saved semantic IDs to pickle.")
        except Exception as e:
            logging.error(f"Failed to save semantic IDs to {uncertainty_pkl_path}: {e}")

    del entailment_model_for_ids
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# Subsample if requested
if args.subsample is not None and args.subsample < args.num_generations:
    logging.info(f"Subsampling {args.subsample} generations from {args.num_generations}")
    new_list_semantic_ids = []
    
    for idx, tid in enumerate(validation_generations):
        example = validation_generations[tid]
        responses = example['responses']
        
        current_len = len(responses)
        available_len = min(current_len, args.num_generations)
        
        semantic_ids = list_semantic_ids[idx]
        
        # Subsample indices
        if available_len <= args.subsample:
            indices = np.arange(available_len)
        else:
            indices = np.sort(np.random.choice(available_len, args.subsample, replace=False))
            
        # Update responses
        new_responses = [responses[i] for i in indices]
        example['responses'] = new_responses
        
        # Update semantic_ids
        new_semantic_ids_entry = [semantic_ids[i] for i in indices]
        new_list_semantic_ids.append(new_semantic_ids_entry)
        
    list_semantic_ids = new_list_semantic_ids
    args.num_generations = args.subsample

# Recompute accuracy if requested
if args.recompute_accuracy:
    logging.warning('Recompute accuracy enabled.')
    metric = get_metric(args.metric)
    metric_model = utils.init_model_from_name(args.metric_model) if args.metric_model else None
    
    validation_is_true = []
    for tid in tqdm(validation_generations, desc="Recomputing accuracy"):
        example = validation_generations[tid]
        most_likely_answer = example['most_likely_answer']
        
        if utils.is_answerable(example):
            try:
                acc = metric(most_likely_answer['response'], example, metric_model)
            except Exception as e:
                logging.error(f"Unable to calculate metric due to error: {e}")
                acc = most_likely_answer['accuracy']
        else:
            acc = 0.0
        
        validation_generations[tid]['most_likely_answer']["accuracy"] = acc
        validation_is_true.append(acc)
else:
    # Extract accuracy from precomputed results
    validation_is_true = []
    for tid in validation_generations:
        example = validation_generations[tid]
        validation_is_true.append(example['most_likely_answer']['accuracy'])

# If Gemini labels provided, overwrite accuracy
if getattr(args, 'gemini_json', None):
    new_labels = load_gemini_labels(args, validation_generations)
    if new_labels is not None and len(new_labels) == len(validation_is_true):
        validation_is_true = new_labels
        logging.info(f"Overwrote validation_is_true with Gemini labels; {len(new_labels)} labels loaded.")

# Compute cluster assignment entropy (blackbox: only depends on semantic IDs)
logging.info("Computing cluster assignment entropy...")

list_cluster_entropy = []

for idx in tqdm(range(len(validation_is_true)), desc="Computing cluster entropy"):
    semantic_ids = list_semantic_ids[idx]
    
    # Compute cluster assignment entropy (no log likelihoods needed)
    cae = cluster_assignment_entropy(semantic_ids)
    list_cluster_entropy.append(cae)

# Compute metrics
logging.info("Computing AUROC/AUARC/AUCPR...")

validation_is_false = [1.0 - is_t for is_t in validation_is_true]
is_binary = is_binary_list(validation_is_false)

results = []

measures = [
    ('cluster_assignment_entropy', list_cluster_entropy),
]

for measure_name, measure_values in measures:
    if is_binary:
        measure_auroc = auroc(validation_is_false, measure_values)
    else:
        measure_auroc = -1
    
    measure_auarc = auarc(measure_values, validation_is_true)
    measure_aucpr = aucpr(measure_values, validation_is_true)
    
    results.append({
        'method': measure_name,
        'auroc': measure_auroc,
        'auarc': measure_auarc,
        'aucpr': measure_aucpr
    })
    
    logging.info(f"{measure_name}: AUROC={measure_auroc:.4f}, AUARC={measure_auarc:.4f}, AUCPR={measure_aucpr:.4f}")

# Save results to CSV
df_results = pd.DataFrame(results)
os.makedirs('semantic_entropy_results', exist_ok=True)
csv_path = f'semantic_entropy_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}.csv'
df_results.to_csv(csv_path, index=False)
logging.info(f"Saved results to: {csv_path}")

# Save per-sample JSON
sample_ids = list(validation_generations.keys())
per_sample_results = {
    "metadata": {
        "dataset": args.dataset,
        "model": args.model_name,
        "num_generations": args.num_generations,
        "suffix": args.suffix,
        "seed": args.random_seed
    },
    "samples": {}
}

for idx, tid in enumerate(sample_ids):
    example = validation_generations[tid]
    
    question = example.get('question', '')
    most_likely_answer = example.get('most_likely_answer', {})
    greedy_answer = most_likely_answer.get('response', '')
    
    ground_truth = example.get('answers', [])
    if not ground_truth:
        ground_truth = example.get('ground_truth', [])
    if not ground_truth and 'answer' in example:
        ground_truth = [example['answer']]
    
    per_sample_results["samples"][tid] = {
        "sample_id": tid,
        "question": question,
        "greedy_answer": greedy_answer,
        "ground_truth": ground_truth,
        "accuracy": validation_is_true[idx],
        "cluster_assignment_entropy": list_cluster_entropy[idx]
    }

json_path = f'semantic_entropy_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}_per_sample.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(per_sample_results, f, indent=2, ensure_ascii=False, default=str)
logging.info(f"Saved per-sample results to: {json_path}")

# Print summary
print("\n" + "="*70)
print("SEMANTIC ENTROPY COMPUTATION SUMMARY")
print("="*70)
print(df_results.to_string(index=False))
print("="*70)

wandb.finish()
