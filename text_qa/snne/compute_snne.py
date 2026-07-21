import os
import json
import logging

import wandb
import pickle
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
from snne.uncertainty.utils.compute_utils import get_parser, setup_wandb, load_precomputed_results, collect_info, print_best_scores, load_gemini_labels
##compute acc on low temp samples

import numpy as np
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

if args.subsample is not None and args.subsample < args.num_generations:
    logging.info(f"Subsampling {args.subsample} generations from {args.num_generations}")
    for tid in validation_generations:
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
    
    # Initialize entailment model if needed for ID computation
    # We'll use EntailmentDeberta by default if IDs are missing
    entailment_model_for_ids = EntailmentDeberta()
    
    list_semantic_ids = []
    # Use keys in same order as collect_info will use
    # validation_generations is usually a dict
    for tid in tqdm(validation_generations, desc="Computing Semantic IDs"):
        example = validation_generations[tid]
        # Get responses. Note: compute_uncertainty_measures uses specific logic.
        # We need to extract just the text responses.
        # validation_generations entries structure: 
        # 'responses': list of [text, [logprobs], embeddings] (or similar, depending on generation script)
        # Based on collect_info: gen_info[0] is text.
        
        # We assume we want IDs for ALL generations present, then collect_info or subsample will slice/filter.
        full_responses = example["responses"]
        responses = [r[0] for r in full_responses]
        
        # Limit to num_generations if specified here? 
        # compute_utils.slice_1d tried to slice to args.num_generations.
        # We should compute for as many as we have or up to num_generations.
        # But we haven't subsampled yet.
        # Let's compute for all valid responses in the file to be safe, or just args.num_generations.
        responses = responses[:args.num_generations]
        
        # Compute IDs
        # Default args matching compute_uncertainty_measures: strict_entailment=True, cluster_method='greedy'
        # We should check if args has these, but args from get_parser in compute_utils might usually have defaults.
        ids = get_semantic_ids_using_entailment(
            responses, 
            entailment_model_for_ids,
            strict_entailment=getattr(args, 'strict_entailment', True),
            cluster_method=getattr(args, 'cluster_method', 'greedy'),
            example=example
        )
        list_semantic_ids.append(ids)
    
    logging.info(f"Computed semantic IDs for {len(list_semantic_ids)} examples.")
    
    # Only save back to the original pickle file if we are NOT subsampling.
    # This prevents overwriting the full-set IDs with a subsampled set.
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

    # Clean up model if we don't need it later? 
    # But we might need it for 'entail' similarity later.
    # To avoid re-initialization, we can assign it to a variable we check later, 
    # OR just rely on the existing code initializing it again (it's somewhat expensive but safe).
    # Since we are in a rush to fix, let's just let it double-init or we can optimize later.
    del entailment_model_for_ids
    import gc
    gc.collect()
    torch.cuda.empty_cache()



# Load models
save_list = []
load_list = []

measures = args.similarity_measures
if 'all' in measures:
    save_list.append('lexsim')
    save_list.append('entail')
    save_list.append('jaccardlex')
else:
    if 'lexsim' in measures:
        save_list.append('lexsim')
    if 'entail' in measures:
        save_list.append('entail')
    if 'jaccardlex' in measures:
        save_list.append('jaccardlex')

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
list_generation_jaccardlex = result_dict['list_generation_jaccardlex_sim']

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
    'list_semantic_ids': _len(list_semantic_ids)
}

if 'all' in measures or 'lexsim' in measures:
    checks['list_generation_lexcial_sim'] = _len(list_generation_lexcial_sim)
if 'all' in measures or 'entail' in measures:
    checks['list_generation_entailment'] = _len(list_generation_entailment)
if 'all' in measures or 'jaccardlex' in measures:
    checks['list_generation_jaccardlex'] = _len(list_generation_jaccardlex)

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
validation_is_false = [1.0 - is_t for is_t in validation_is_true]
is_binary = is_binary_list(validation_is_false)
temperature_choice = [0.1, 1, 10, 100]
variant_choice = ['only_denom']
selfsim_choice = [True]

list_similarity_matrix = []
similarity_name_choice = []

# Dynamically add available similarity matrices
if list_generation_lexcial_sim and len(list_generation_lexcial_sim) > 0:
    list_similarity_matrix.append(list_generation_lexcial_sim)
    similarity_name_choice.append('lexical_sim')

if list_generation_entailment and len(list_generation_entailment) > 0:
    list_similarity_matrix.append(list_generation_entailment)
    similarity_name_choice.append('entailment_sim')

if list_generation_jaccardlex and len(list_generation_jaccardlex) > 0:
    list_similarity_matrix.append(list_generation_jaccardlex)
    similarity_name_choice.append('jaccardlex_sim')

# Repeat the responses list so the original zip-based loop iterates over
# each similarity matrix while using the same response set for each.
list_responses = [list_generation for _ in list_similarity_matrix]

# Store SNNE scores for per-sample export (default config)
default_temperature = 1
default_variant = 'only_denom'
default_selfsim = True
saved_snne_scores = {}  # {similarity_name: [scores]}

for variant in variant_choice:
    for selfsim in selfsim_choice:
        for temperature in temperature_choice:
            for response, similarity_matrix, similarity_name in zip(list_responses, list_similarity_matrix, similarity_name_choice):
                method_name_postfix = f'{variant}_temp{temperature}_selfsim{selfsim}_{similarity_name}-similarity'
                logging.info(method_name_postfix.center(100, '-'))
                list_snne = []
                list_wsnne = []
                
                # Check if this is the default config we want to save
                is_default_config = (variant == default_variant and 
                                   temperature == default_temperature and 
                                   selfsim == default_selfsim)
                
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
                
                # Save scores for default config
                if is_default_config:
                    saved_snne_scores[similarity_name] = list_snne

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
df_metrics.to_csv(f'snne_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Save per-sample JSON with uncertainty values (reuse computed scores)
# ─────────────────────────────────────────────────────────────────────────────
# Store per-sample results for each similarity type
per_sample_results = {
    "metadata": {
        "dataset": args.dataset,
        "model": args.model_name,
        "num_generations": args.num_generations,
        "temperature": default_temperature,
        "variant": default_variant,
        "selfsim": default_selfsim
    },
    "samples": {}
}

# Get sample IDs from validation_generations (in iteration order)
sample_ids = list(validation_generations.keys())

# Build per-sample results using saved SNNE scores
for idx, tid in enumerate(sample_ids):
    example = validation_generations[tid]
    
    # Extract sample info
    question = example.get('question', '')
    most_likely_answer = example.get('most_likely_answer', {})
    greedy_answer = most_likely_answer.get('response', '')
    
    # Get ground truth from example
    ground_truth = example.get('answers', [])
    if not ground_truth:
        ground_truth = example.get('ground_truth', [])
    if not ground_truth and 'answer' in example:
        ground_truth = [example['answer']]
    
    # Create sample entry
    per_sample_results["samples"][tid] = {
        "sample_id": tid,
        "question": question,
        "greedy_answer": greedy_answer,
        "ground_truth": ground_truth,
        "accuracy": validation_is_true[idx]
    }
    
    # Add saved SNNE scores for each similarity type
    for similarity_name in saved_snne_scores:
        per_sample_results["samples"][tid][f"snne_{similarity_name}"] = saved_snne_scores[similarity_name][idx]

# Save per-sample JSON
per_sample_json_path = f'snne_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}_per_sample.json'
with open(per_sample_json_path, 'w', encoding='utf-8') as f:
    json.dump(per_sample_results, f, indent=2, ensure_ascii=False, default=str)
logging.info(f"Saved per-sample results to: {per_sample_json_path}")

# Print the best scores
print_best_scores(df_metrics, keyword='', list_scores=['auroc', 'auarc', 'prr'])

wandb.finish()
