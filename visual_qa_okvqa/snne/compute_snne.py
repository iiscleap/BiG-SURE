import os
import logging

import wandb
from tqdm import tqdm
import torch
import pandas as pd
import evaluate
from rouge_score import tokenizers
from sentence_transformers import SentenceTransformer

from snne.uncertainty.utils.eval_utils import auroc, auarc, aucpr, is_binary_list
from snne.uncertainty.utils import utils
from snne.uncertainty.utils.metric_utils import get_metric
from snne.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta, soft_nearest_neighbor_loss
from snne.uncertainty.utils.compute_utils import (
    get_parser,
    setup_wandb,
    load_precomputed_results,
    collect_info,
    print_best_scores,
    load_gemini_labels,
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
setup_wandb(args, prefix='compute_snne')

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

# if lexsim_exist:
#     load_list.append('lexsim')
# else:
#     save_list.append('lexsim')
save_list.append('lexsim')
save_list.append('entail')
if 'entail' in save_list:
    entailment_model = EntailmentDeberta()
else:
    entailment_model = None
if 'embedding' in save_list:
    if args.embedding_model == 'qwen':
        embedding_model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
        embedding_model.max_seq_length = 8192
    else:
        embedding_model = SentenceTransformer("Salesforce/SFR-Embedding-2_R")
else:
    embedding_model = None
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
    embedding_model, 
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

list_generation = result_dict['list_generation']
list_generation_log_likelihoods = result_dict['list_generation_log_likelihoods']
list_generation_lexcial_sim = result_dict['list_generation_lexcial_sim']
list_generation_entailment = result_dict['list_generation_entailment_similarity']

# Sanity checks: make sure all per-example lists match the number of validation examples.
expected_n = len(result_dict['validation_is_true'])
def _len(x):
    try:
        return len(x)
    except Exception:
        return None

checks = {
    'validation_is_true': _len(result_dict.get('validation_is_true')),
    'list_generation': _len(list_generation),
    'list_generation_log_likelihoods': _len(list_generation_log_likelihoods),
    'list_generation_lexcial_sim': _len(list_generation_lexcial_sim),
    'list_generation_entailment': _len(list_generation_entailment),
    'list_semantic_ids': _len(list_semantic_ids)
}

bad = [k for k,v in checks.items() if v is None or v != expected_n]
if bad:
    logging.error('Per-example list length mismatch detected for run. Expected %d items per list.', expected_n)
    for k,v in checks.items():
        logging.error('  %-40s : %s', k, str(v))
    logging.error('This likely means the precomputed `embedding_and_similarity.pkl` or the `validation_generations.pkl` are out-of-sync for this run. Suggest deleting the precomputed file in the run folder and re-running the computation, or ensuring the validation PKL and saved similarity file were produced from the same run.')
    raise SystemExit(2)

# Calculate SNN score
list_method_name = []
list_auroc = []
list_auarc = []
list_aucpr = []
list_temperature = []
list_variant = []
list_selfsim = []
list_similarity_name = []

print(len([is_t for is_t in validation_is_true if is_t==1]), "positive examples.")
print(len([is_t for is_t in validation_is_true if is_t==0]), "negative examples.")
print(validation_is_true[:10])

if args.metric == 'vqa_acc':
    validation_is_true_binary = [1.0 if acc >= args.metric_threshold else 0.0 for acc in validation_is_true]
    print(f"VQA accuracy binarized at threshold {args.metric_threshold}:")
    print(f"  {sum(validation_is_true_binary)} correct, {len(validation_is_true_binary) - sum(validation_is_true_binary)} incorrect")
elif args.metric == 'vqarad_exact':
    print(f"VQA-RAD exact match (binary):")
    print(f"  {sum(validation_is_true)} correct, {len(validation_is_true) - sum(validation_is_true)} incorrect")
validation_is_false, is_binary = get_auroc_label_arrays(args, validation_is_true)
method_uncertainties = {}
method_extra_cols = {}
temperature_choice = [0.1, 1, 10, 100]
variant_choice = ['only_denom']
selfsim_choice = [True]
list_similarity_matrix = [
    list_generation_lexcial_sim, list_generation_entailment
]
similarity_name_choice = [
    'lexical_sim', 'entailment_sim'
]

# Repeat the responses list so the original zip-based loop iterates over
# each similarity matrix while using the same response set for each.
list_responses = [list_generation for _ in list_similarity_matrix]

for variant in variant_choice:
    for selfsim in selfsim_choice:
        for temperature in temperature_choice:
            for response, similarity_matrix, similarity_name in zip(list_responses, list_similarity_matrix, similarity_name_choice):
                method_name_postfix = f'{variant}_temp{temperature}_selfsim{selfsim}_{similarity_name}-similarity'
                logging.info(method_name_postfix.center(100, '-'))
                list_snne = []
                list_wsnne = []
                
                for idx in tqdm(range(len(validation_is_true))):
                    # Compute SNN
                    snne = soft_nearest_neighbor_loss(
                        response[idx],
                        entailment_model, 
                        embedding_model, 
                        list_semantic_ids[idx],
                        similarity_matrix=similarity_matrix[idx],
                        variant=variant, 
                        temperature=temperature, 
                        exclude_diagonal=not selfsim).item()
                    list_snne.append(snne)
                    
                    # Compute WSNN
                    if args.compute_wsnn:
                        weight_pe = torch.exp(torch.tensor(list_generation_log_likelihoods[idx]))
                        weight_pe = weight_pe / weight_pe.mean()
                        wsnne = soft_nearest_neighbor_loss(
                            response[idx],
                            entailment_model, 
                            embedding_model, 
                            list_semantic_ids[idx],
                            similarity_matrix=similarity_matrix[idx],
                            variant=variant, 
                            temperature=temperature, 
                            exclude_diagonal=not selfsim,
                            weight=weight_pe).item()

                        list_wsnne.append(wsnne)

                # Collect AUROC score
                snne_choice = [list_snne, list_wsnne]
                list_snne_name = ['snne', 'wsnne']
                if not args.compute_wsnn:
                    snne_choice = snne_choice[:-1]
                    list_snne_name = list_snne_name[:-1]
                
                for snn, snne_name in zip(snne_choice, list_snne_name):
                    if is_binary:
                        snne_auroc = auroc(validation_is_false, snn)
                    else:
                        snne_auroc = -1
                    snne_auarc = auarc(snn, validation_is_true)
                    snne_aucpr = aucpr(snn, validation_is_true)
                    list_variant.append(variant)
                    list_selfsim.append(selfsim)
                    list_temperature.append(temperature)
                    list_similarity_name.append(similarity_name)
                    list_method_name.append(snne_name)
                    list_auroc.append(snne_auroc)
                    list_auarc.append(snne_auarc)
                    list_aucpr.append(snne_aucpr)

                    method_key = (
                        f'{snne_name}_{variant}_temp{temperature}_selfsim{selfsim}_{similarity_name}'
                    )
                    method_uncertainties[method_key] = snn
                    method_extra_cols[method_key] = {
                        'variant': variant,
                        'temperature': temperature,
                        'selfsim': selfsim,
                        'similarity': similarity_name,
                        'snne_method': snne_name,
                    }
                
# Output to CSV
data_metrics = {
    'method': list_method_name,
    'variant': list_variant,
    'selfsim': list_selfsim,
    'temperature': list_temperature,
    'similarity': list_similarity_name,
    'auroc': list_auroc,
    'auarc': list_auarc,
    'prr': list_aucpr
}

df_metrics = pd.DataFrame(data_metrics)
# logging.info(df_metrics.head())

# Save results
os.makedirs('snne_results', exist_ok=True)
metrics_csv = f'snne_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}.csv'
df_metrics.to_csv(metrics_csv, index=False)

example_metadata = build_example_metadata(validation_generations, validation_is_true, args)
save_per_example_uncertainties(
    per_example_output_path(metrics_csv),
    example_metadata,
    method_uncertainties,
    method_extra_cols=method_extra_cols,
)

# Print the best scores
print_best_scores(df_metrics, keyword='', list_scores=['auroc', 'auarc', 'prr'])

wandb.finish()