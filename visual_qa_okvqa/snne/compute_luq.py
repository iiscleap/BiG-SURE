import os
import logging

import wandb
from tqdm import tqdm
import pandas as pd
import evaluate
from rouge_score import tokenizers

from snne.uncertainty.utils.eval_utils import auroc, auarc, aucpr, is_binary_list
from snne.uncertainty.utils import utils
from snne.uncertainty.utils.entropy_utils import get_luq_pair
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
setup_wandb(args, prefix='compute_luq')

# Load pre-computed results
precomputed_results = load_precomputed_results(args)
validation_generations = precomputed_results['validation_generations']
save_embedding_path = precomputed_results['save_embedding_path']
save_dict = precomputed_results['save_dict']
luq_sim_exist = precomputed_results['luq_sim_exist']
list_semantic_ids = precomputed_results['list_semantic_ids']

# Load models
save_list = []
load_list = []

if not luq_sim_exist:
    entailment_model = EntailmentDeberta()
    save_list = ['luq']
else:
    entailment_model = None
    load_list = ['luq']
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
    save_list=save_list,
    load_list=load_list
)

validation_is_true = result_dict['validation_is_true']
list_generation = result_dict['list_generation']
list_generation_luq_similarity = result_dict['list_generation_luq_similarity']

# Calculate LUQ-pair
list_method_name = []
list_auroc = []
list_auarc = []
list_aucpr = []

validation_is_false = [1.0 - is_t for is_t in validation_is_true]
is_binary = is_binary_list(validation_is_false)
list_responses = [list_generation]

list_similarity_matrix = [list_generation_luq_similarity]
similarity_name_choice = ['luq']

for similarity_matrix, similarity_name in zip(list_similarity_matrix, similarity_name_choice):
    list_luq = []
    
    for idx in tqdm(range(len(validation_is_true))):
        # Uncertainty      
        list_luq.append(get_luq_pair(similarity_matrix[idx].numpy())[0])
    
    if is_binary:
        mat_auroc = auroc(validation_is_false, list_luq)
    else:
        mat_auroc = -1
    mat_auarc = auarc(list_luq, validation_is_true)
    mat_aucpr = aucpr(list_luq, validation_is_true)
    
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
os.makedirs('luq_results', exist_ok=True)
df_metrics.to_csv(f'luq_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}.csv', index=False)

wandb.finish()