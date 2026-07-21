import os
import logging
import json

import wandb
from tqdm import tqdm
import pandas as pd
import evaluate
from rouge_score import tokenizers

import pickle
import torch
import numpy as np

from snne.uncertainty.utils.eval_utils import auroc, auarc, aucpr, is_binary_list
from snne.uncertainty.utils import utils
from snne.uncertainty.utils.entropy_utils import ( 
    compute_lexical_similarity, 
    get_spectral_eigv, 
    get_degreeuq, 
    get_eccentricity
)
from snne.uncertainty.utils.metric_utils import get_metric
from snne.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from snne.uncertainty.utils.compute_utils import get_parser, setup_wandb, load_precomputed_results, collect_info


# Set up log
utils.setup_logger()

# Parse arguments
args = get_parser()
logging.info("Args: %s", args)
utils.set_all_seeds(args.random_seed)

# Set up wandb
setup_wandb(args, prefix='compute_graph_baselines')

# Load pre-computed results
precomputed_results = load_precomputed_results(args)
validation_generations = precomputed_results['validation_generations']
save_embedding_path = precomputed_results['save_embedding_path']
save_dict = precomputed_results['save_dict']
lexsim_exist = precomputed_results['lexsim_exist']
list_semantic_ids = precomputed_results['list_semantic_ids']

if getattr(args, 'subsample', None) is not None and args.subsample < args.num_generations:
    logging.info(f"Subsampling {args.subsample} generations from {args.num_generations}")
    keys = list(validation_generations.keys()) if isinstance(validation_generations, dict) else range(len(validation_generations))
    for tid in keys:
        example = validation_generations[tid]
        responses = example['responses']
        available_len = min(len(responses), args.num_generations)
        if available_len <= args.subsample:
            indices = np.arange(available_len)
        else:
            indices = np.sort(np.random.choice(available_len, args.subsample, replace=False))
        example['responses'] = [responses[i] for i in indices]
    args.num_generations = args.subsample
    # Force recalculation for the subsampled set
    list_semantic_ids = None

if list_semantic_ids is None:
    logging.info("Semantic IDs not found in precomputed results. Computing them now...")
    from snne.uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids_using_entailment
    entailment_model_for_ids = EntailmentDeberta()
    list_semantic_ids = []
    
    iterator = validation_generations if isinstance(validation_generations, list) else validation_generations.keys()
    for tid in tqdm(iterator, desc="Computing Semantic IDs"):
        example = validation_generations[tid]
        responses = [r[0] for r in example["responses"][:args.num_generations]]
        ids = get_semantic_ids_using_entailment(
            responses, entailment_model_for_ids,
            strict_entailment=getattr(args, 'strict_entailment', True),
            cluster_method=getattr(args, 'cluster_method', 'greedy'),
            example=example
        )
        list_semantic_ids.append(ids)
    
    # Only save back if NOT subsampling
    if args.subsample is None:
        uncertainty_pkl_path = os.path.join(args.data_path, 'uncertainty_measures.pkl')
        try:
            full_results = {}
            if os.path.isfile(uncertainty_pkl_path):
                with open(uncertainty_pkl_path, 'rb') as infile:
                    full_results = pickle.load(infile)
            full_results['semantic_ids'] = list_semantic_ids
            full_results['schema_version'] = 1
            full_results['question_ids'] = [str(tid) for tid in validation_generations]
            with open(uncertainty_pkl_path, 'wb') as outfile:
                pickle.dump(full_results, outfile)
            logging.info(f'Saved semantic IDs to {uncertainty_pkl_path}')
        except Exception as e:
            logging.error(f"Failed to save semantic IDs: {e}")
    del entailment_model_for_ids
    import gc
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        logging.warning(f"torch.cuda.empty_cache() failed: {e}")
        


# Load models
save_list = ['entail']
load_list = []

if lexsim_exist:
    load_list.append('lexsim')
else:
    save_list.append('lexsim')
if 'entail' in save_list:
    entailment_model = EntailmentDeberta()
else:
    entailment_model = None
tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize
rouge = evaluate.load('rouge', keep_in_memory=True)

if args.recompute_accuracy:
    # This is usually not enabled.
    logging.warning('Recompute accuracy enabled.')
    metric = get_metric(args.metric)
else:
    metric = None

# Collect info
result_dict = collect_info(
    args, 
    validation_generations, 
    metric, 
    entailment_model, 
    None, 
    rouge,
    tokenizer,
    list_semantic_ids,
    save_dict, 
    save_embedding_path, 
    save_list, 
    load_list
)

validation_is_true = result_dict['validation_is_true']
list_generation = result_dict['list_generation_with_question']
list_generation_entailment_similarity = result_dict['list_generation_with_question_entailment_similarity']
list_generation_log_likelihoods = result_dict['list_generation_log_likelihoods']
list_num_sets = result_dict['list_num_sets']
list_generation_lexcial_sim = result_dict['list_generation_with_question_lexical_sim']

# Calculate baselines
list_method_name = []
list_auroc = []
list_auarc = []
list_aucpr = []

validation_is_false = [1.0 - is_t for is_t in validation_is_true]
is_binary = is_binary_list(validation_is_false)

per_sample_scores = {}

## Num sets
if is_binary:
    num_set_auroc = auroc(validation_is_false, list_num_sets)
else:
    num_set_auroc = -1
num_set_auarc = auarc(list_num_sets, validation_is_true)
num_set_aucpr = aucpr(list_num_sets, validation_is_true)

list_method_name.append('num_set')
list_auroc.append(num_set_auroc)
list_auarc.append(num_set_auarc)
list_aucpr.append(num_set_aucpr)

per_sample_scores['num_set'] = list_num_sets

## Lexical similarity
list_similarity_matrix = [list_generation_lexcial_sim]
similarity_name_choice = ['lexical_sim']

for similarity_matrix, similarity_name in zip(list_similarity_matrix, similarity_name_choice):
    list_lex_sim = []
    
    for idx in tqdm(range(len(validation_is_true))):
        try:
            # Uncertainty = - Lexical similarity
            list_lex_sim.append(-compute_lexical_similarity(similarity_matrix[idx]))
        except (IndexError, ValueError) as e:
            logging.warning(f"Error computing lexical similarity for idx {idx}: {e}")
            list_lex_sim.append(0.0)  # Default uncertainty value
    
    if is_binary:
        lexical_similarity_auroc = auroc(validation_is_false, list_lex_sim)
    else:
        lexical_similarity_auroc = -1
    lexical_similarity_auarc = auarc(list_lex_sim, validation_is_true)
    lexical_similarity_aucpr = aucpr(list_lex_sim, validation_is_true)
    
    list_method_name.append(similarity_name)
    list_auroc.append(lexical_similarity_auroc)
    list_auarc.append(lexical_similarity_auarc)
    list_aucpr.append(lexical_similarity_aucpr)
    
    per_sample_scores[similarity_name] = list_lex_sim

## Sum of eigenvalues
list_similarity_matrix = [list_generation_entailment_similarity]
similarity_name_choice = ['sum_eigv']

for similarity_matrix, similarity_name in zip(list_similarity_matrix, similarity_name_choice):
    list_sum_eigv = []
    
    for idx in tqdm(range(len(validation_is_true))):
        try:
            # Uncertainty = Sum eigen values
            list_sum_eigv.append(get_spectral_eigv(similarity_matrix[idx].numpy()))
        except (IndexError, ValueError, AttributeError) as e:
            logging.warning(f"Error computing spectral eigenvalues for idx {idx}: {e}")
            list_sum_eigv.append(0.0)  # Default uncertainty value
    
    if is_binary:
        sum_eigv_auroc = auroc(validation_is_false, list_sum_eigv)
    else:
        sum_eigv_auroc = -1
    sum_eigv_auarc = auarc(list_sum_eigv, validation_is_true)
    sum_eigv_aucpr = aucpr(list_sum_eigv, validation_is_true)
    
    list_method_name.append(similarity_name)
    list_auroc.append(sum_eigv_auroc)
    list_auarc.append(sum_eigv_auarc)
    list_aucpr.append(sum_eigv_aucpr)
    
    per_sample_scores[similarity_name] = list_sum_eigv

## Degree matrix
similarity_name_choice = ['degree_mat']

for similarity_matrix, similarity_name in zip(list_similarity_matrix, similarity_name_choice):
    list_degree_mat = []
    
    for idx in tqdm(range(len(validation_is_true))):
        try:
            # Uncertainty = Degree uncertainty
            degree_result = get_degreeuq(similarity_matrix[idx].numpy())
            list_degree_mat.append(degree_result[0])
        except (IndexError, ValueError, AttributeError) as e:
            logging.warning(f"Error computing degree matrix for idx {idx}: {e}")
            list_degree_mat.append(0.0)  # Default uncertainty value
    
    if is_binary:
        degree_mat_auroc = auroc(validation_is_false, list_degree_mat)
    else:
        degree_mat_auroc = -1
    degree_mat_auarc = auarc(list_degree_mat, validation_is_true)
    degree_mat_aucpr = aucpr(list_degree_mat, validation_is_true)
    
    list_method_name.append(similarity_name)
    list_auroc.append(degree_mat_auroc)
    list_auarc.append(degree_mat_auarc)
    list_aucpr.append(degree_mat_aucpr)

    per_sample_scores[similarity_name] = list_degree_mat

## Eccentricity
similarity_name_choice = ['eccentricity_thr{}']
eigv_threshold = 0.9

for similarity_matrix, similarity_name in zip(list_similarity_matrix, similarity_name_choice):
    list_eccentricity = []
    
    for idx in tqdm(range(len(validation_is_true))):
        try:
            # Uncertainty = Eccentricitiy
            eccentricity_result = get_eccentricity(similarity_matrix[idx].numpy())
            list_eccentricity.append(eccentricity_result[0])
        except (IndexError, ValueError, AttributeError) as e:
            logging.warning(f"Error computing eccentricity for idx {idx}: {e}")
            list_eccentricity.append(0.0)  # Default uncertainty value
    
    if is_binary:
        eccentricity_auroc = auroc(validation_is_false, list_eccentricity)
    else:
        eccentricity_auroc = -1
    eccentricity_auarc = auarc(list_eccentricity, validation_is_true)
    eccentricity_aucpr = aucpr(list_eccentricity, validation_is_true)
    
    method_n = similarity_name.format(eigv_threshold)
    list_method_name.append(method_n)
    list_auroc.append(eccentricity_auroc)
    list_auarc.append(eccentricity_auarc)
    list_aucpr.append(eccentricity_aucpr)

    per_sample_scores[method_n] = list_eccentricity
                
# Output to CSV
data_metrics = {
    'method': list_method_name,
    'auroc': list_auroc,
    'auarc': list_auarc,
    'prr': list_aucpr
}

df_metrics = pd.DataFrame(data_metrics)
logging.info(df_metrics.head())
os.makedirs('graph_baseline_results', exist_ok=True)
df_metrics.to_csv(f'graph_baseline_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Save per-sample JSON with uncertainty values
# ─────────────────────────────────────────────────────────────────────────────
per_sample_results = {
    "metadata": {
        "dataset": args.dataset,
        "model": args.model_name,
        "num_generations": args.num_generations,
    },
    "samples": {}
}

# Get sample IDs
if isinstance(validation_generations, dict):
    sample_ids = list(validation_generations.keys())
else:
    sample_ids = list(validation_generations.keys())

for idx, tid in enumerate(sample_ids):
    example = validation_generations[tid]
    
    # Extract sample info
    question = example.get('question', '')
    most_likely_answer = example.get('most_likely_answer', {})
    greedy_answer = most_likely_answer.get('response', '')
    
    # Get ground truth
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
        "accuracy": validation_is_true[idx]
    }
    
    # Add baseline scores
    for method_name, scores_list in per_sample_scores.items():
        if idx < len(scores_list):
            per_sample_results["samples"][tid][method_name] = scores_list[idx]

# Save per-sample JSON
per_sample_json_path = f'graph_baseline_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}_per_sample.json'
with open(per_sample_json_path, 'w', encoding='utf-8') as f:
    json.dump(per_sample_results, f, indent=2, ensure_ascii=False, default=str)
logging.info(f"Saved per-sample results to: {per_sample_json_path}")

wandb.finish()
