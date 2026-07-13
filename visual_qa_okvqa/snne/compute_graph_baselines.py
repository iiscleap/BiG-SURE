import os
import logging

import wandb
from tqdm import tqdm
import pandas as pd
import evaluate
from rouge_score import tokenizers

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
from snne.uncertainty.utils.compute_utils import (
    get_parser,
    setup_wandb,
    load_precomputed_results,
    collect_info,
    load_gemini_labels,
    load_vqa_labels,
    build_example_metadata,
    get_auroc_label_arrays,
    per_example_output_path,
    save_per_example_uncertainties,
)


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

# Load models
save_list = []
load_list = []

# Always recompute both (like SNNE) to avoid save_dict overwrite bugs
save_list.append('lexsim')
save_list.append('entail')
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
# If a Gemini per-example JSON is provided, try to overwrite validation_is_true
if getattr(args, 'gemini_json', None):
    new_labels = load_gemini_labels(args, validation_generations)
    if new_labels is not None and len(new_labels) == len(validation_is_true):
        validation_is_true = new_labels
        print(f"Overwrote validation_is_true with Gemini labels; {len(new_labels)} labels loaded.")
    else:
        print("No Gemini labels applied (missing file or length mismatch).")

# If a VQA per-example JSON is provided, try to overwrite validation_is_true
if getattr(args, 'vqa_json', None):
    new_labels = load_vqa_labels(args, validation_generations)
    if new_labels is not None and len(new_labels) == len(validation_is_true):
        validation_is_true = new_labels
        print(f"Overwrote validation_is_true with VQA labels; {len(new_labels)} labels loaded.")
    else:
        print("No VQA labels applied (missing file or length mismatch).")

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

if args.metric == 'vqa_acc':
    validation_is_true_binary = [1.0 if acc >= args.metric_threshold else 0.0 for acc in validation_is_true]
    print(f"VQA accuracy binarized at threshold {args.metric_threshold}:")
    print(f"  {sum(validation_is_true_binary)} correct, {len(validation_is_true_binary) - sum(validation_is_true_binary)} incorrect")
elif args.metric == 'vqarad_exact':
    print(f"VQA-RAD exact match (binary):")
    print(f"  {sum(validation_is_true)} correct, {len(validation_is_true) - sum(validation_is_true)} incorrect")
validation_is_false, is_binary = get_auroc_label_arrays(args, validation_is_true)
method_uncertainties = {}

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
method_uncertainties['num_set'] = list_num_sets

## Lexical similarity
list_similarity_matrix = [list_generation_lexcial_sim]
similarity_name_choice = ['lexical_sim']

for similarity_matrix, similarity_name in zip(list_similarity_matrix, similarity_name_choice):
    list_lex_sim = []
    
    for idx in tqdm(range(len(validation_is_true))):
        # Uncertainty = - Lexical similarity
        list_lex_sim.append(-compute_lexical_similarity(similarity_matrix[idx]))
    
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
    method_uncertainties[similarity_name] = list_lex_sim
    
## Sum of eigenvalues
list_similarity_matrix = [list_generation_entailment_similarity]
similarity_name_choice = ['sum_eigv']

for similarity_matrix, similarity_name in zip(list_similarity_matrix, similarity_name_choice):
    list_sum_eigv = []
    
    for idx in tqdm(range(len(validation_is_true))):
        # Uncertainty = Sum eigen values
        list_sum_eigv.append(get_spectral_eigv(similarity_matrix[idx].numpy()))
    
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
    method_uncertainties[similarity_name] = list_sum_eigv
    
## Degree matrix
similarity_name_choice = ['degree_mat']

for similarity_matrix, similarity_name in zip(list_similarity_matrix, similarity_name_choice):
    list_degree_mat = []
    
    for idx in tqdm(range(len(validation_is_true))):
        # Uncertainty = Degree uncertainty
        list_degree_mat.append(get_degreeuq(similarity_matrix[idx].numpy())[0])
    
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
    method_uncertainties[similarity_name] = list_degree_mat

## Eccentricity
similarity_name_choice = ['eccentricity_thr{}']
eigv_threshold = 0.9

for similarity_matrix, similarity_name in zip(list_similarity_matrix, similarity_name_choice):
    list_eccentricity = []
    
    for idx in tqdm(range(len(validation_is_true))):
        # Uncertainty = Eccentricitiy
        list_eccentricity.append(get_eccentricity(similarity_matrix[idx].numpy())[0])
    
    if is_binary:
        eccentricity_auroc = auroc(validation_is_false, list_eccentricity)
    else:
        eccentricity_auroc = -1
    eccentricity_auarc = auarc(list_eccentricity, validation_is_true)
    eccentricity_aucpr = aucpr(list_eccentricity, validation_is_true)
    
    method_key = similarity_name.format(eigv_threshold)
    list_method_name.append(method_key)
    list_auroc.append(eccentricity_auroc)
    list_auarc.append(eccentricity_auarc)
    list_aucpr.append(eccentricity_aucpr)
    method_uncertainties[method_key] = list_eccentricity
                
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
metrics_csv = f'graph_baseline_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}.csv'
df_metrics.to_csv(metrics_csv, index=False)

example_metadata = build_example_metadata(validation_generations, validation_is_true, args)
save_per_example_uncertainties(
    per_example_output_path(metrics_csv),
    example_metadata,
    method_uncertainties,
)

wandb.finish()