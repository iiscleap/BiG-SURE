import os
import logging

import wandb
import pandas as pd
import evaluate
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer

from snne.uncertainty.utils.eval_utils import auroc, auarc, aucpr, is_binary_list
from snne.uncertainty.utils import utils
from snne.uncertainty.utils.entropy_utils import get_sar
from snne.uncertainty.utils.metric_utils import get_metric
from snne.uncertainty.utils.compute_utils import get_parser, setup_wandb, load_precomputed_results, collect_info


HF_MAPPING = {
    "Meta-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct"
}


# Set up log
utils.setup_logger()

# Parse arguments
args = get_parser()
logging.info("Args: %s", args)
utils.set_all_seeds(args.random_seed)

# Set up wandb
setup_wandb(args, prefix='compute_sar')

# Load pre-computed results
precomputed_results = load_precomputed_results(args)
validation_generations = precomputed_results['validation_generations']
save_embedding_path = precomputed_results['save_embedding_path']
save_dict = precomputed_results['save_dict']
sar_exist = precomputed_results['sar_exist']
list_semantic_ids = precomputed_results['list_semantic_ids']

# Load models
save_list = []
load_list = []

if not sar_exist:
    print("Load measurement model")
    entailment_model = CrossEncoder(model_name=args.entailment_model, num_labels=1)
    save_list = ['sar']
else:
    entailment_model = None
    load_list = ['sar']

tokenizer = AutoTokenizer.from_pretrained(HF_MAPPING[args.model_name], use_fast=False)
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
    save_list=save_list,
    load_list=load_list
)

validation_is_true = result_dict['validation_is_true']
list_sar_token_importance = result_dict['list_sar_token_importance']
list_sar_token_log_likelihoods = result_dict['list_sar_token_log_likelihoods']
list_sar_sentence_similarity_matrix = result_dict['list_sar_sentence_similarity_matrix']

# Calculate LUQ-pair
list_method_name = []
list_auroc = []
list_auarc = []
list_aucpr = []

validation_is_false = [1.0 - is_t for is_t in validation_is_true]
is_binary = is_binary_list(validation_is_false)

similarity_name_choice = ['sar']

for similarity_name in similarity_name_choice:
    list_sar = get_sar(
        list_sar_token_importance,
        list_sar_sentence_similarity_matrix,
        list_sar_token_log_likelihoods
    )
    
    if is_binary:
        mat_auroc = auroc(validation_is_false, list_sar)
    else:
        mat_auroc = -1
    mat_auarc = auarc(list_sar, validation_is_true)
    mat_aucpr = aucpr(list_sar, validation_is_true)
    
    list_method_name.append(similarity_name)
    list_auroc.append(mat_auroc)
    list_auarc.append(mat_auarc)
    list_aucpr.append(mat_aucpr)
                
# Output to CSV
data_metrics = {
    'method': list_method_name,
    'auroc': list_auroc,
    'auarc': list_auarc,
    'prr': list_aucpr
}

df_metrics = pd.DataFrame(data_metrics)
logging.info(df_metrics.head())
os.makedirs('sar_results', exist_ok=True)
df_metrics.to_csv(f'sar_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}.csv', index=False)

wandb.finish()