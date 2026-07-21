import os
import argparse
import pickle
from collections import Counter

import wandb
from tqdm import tqdm
import numpy as np
import json

from snne.uncertainty.utils.entropy_utils import (
    entailment_similarity_matrix, 
    lexical_similarity_matrix,
    jaccardlex_similarity_matrix, 
    get_tokenwise_importance,
    get_sentence_similarites
)


def get_parser():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', 
                        type=int, default=10)
    parser.add_argument("--num_generations", 
                        type=int, default=10, help="Number of generations to use")
    parser.add_argument("--model_name", 
                        type=str, default="Llama-2-7b-chat", help="Model name")
    parser.add_argument("--dataset", 
                        type=str, default="trivia_qa", 
                        choices=['trivia_qa', 'squad', 'bioasq', 'nq', 'svamp', 'gsm8k', 'math', 'xsum', 'aeslc', 'de-en', 'fr-en', 'vqa'],
                        help="Dataset to use")
    parser.add_argument("--embedding_model", type=str, default="qwen",
                        choices=['qwen', 'sfr'],
                        help="Pretrained embedding model.")
    parser.add_argument("--data_path", 
                        type=str, default=None, help="Old wandb dir",)
    parser.add_argument("--suffix", 
                        type=str, default='', help="Additional name",)
    parser.add_argument('--recompute_accuracy',
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--metric", 
                        type=str, default="squad",
                        choices=['squad', 'llm', 'llm_gpt-3.5', 'llm_gpt-4', 'gsm8k', 'math', 'squad_raw', 'entail', 'rougel', 'bertscore', 'gemini'],
                        help="Metric to assign accuracy to generations.")
    parser.add_argument("--metric_threshold", 
                        type=float, default=0.5,
                        help="Threshold to assign accuracy to generations.")
    parser.add_argument("--entailment_model", default='deberta', type=str)
    parser.add_argument(
        "--entailment_cache_id", default=None, type=str,
        help='Restore entailment predictions from previous run for GPT-4/LLaMa-Entailment.')
    parser.add_argument('--entailment_cache_only', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--strict_entailment',
                        default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--compute_wsnn',
                        default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--truncate_length', 
                        type=int, default=1024,
                        help="The maximum generation length")
    parser.add_argument('--condition_on_question',
                        default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--gemini_json', type=str, default=None, help='Path to Gemini per-example correctness JSON (id->0/1)')
    parser.add_argument('--similarity_measures', type=str, nargs='+', default=['all'], 
                        choices=['all', 'lexsim', 'entail', 'jaccardlex'],
                        help='Similarity measures to compute.')
    parser.add_argument("--subsample", 
                        type=int, default=None, 
                        help="Number of generations to subsample")
    
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    
    return args


def setup_wandb(args, prefix='compute'):
    user = os.environ.get('USER', 'user')
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_dir = os.getenv('WANDB_DIR', f'{scratch_dir}/{user}/uncertainty')
    os.makedirs(wandb_dir, exist_ok=True)
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    project = os.getenv('WANDB_PROJECT', 'bigsure-text-qa')
    entity = os.getenv('WANDB_ENTITY') or os.getenv('WANDB_SEM_UNC_ENTITY')
    host_name = os.uname()[1]
    run_name = f"{prefix}_{args.model_name}_{args.dataset}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}_{host_name}"
    config = {
        'seed': args.random_seed,
        'num_generations': args.num_generations,
        'data_path': args.data_path,
        'model': args.model_name,
        'dataset': args.dataset
    }

    wandb.init(
        entity=entity,
        project=project,
        dir=wandb_dir,
        name=run_name,
        config=config,
        notes=slurm_jobid
    )
    
    
def load_precomputed_results(args):
    with open(f"{args.data_path}/validation_generations.pkl", 'rb') as infile:
        validation_generations = pickle.load(infile)

    if not isinstance(validation_generations, dict) or not validation_generations:
        raise ValueError(
            'validation_generations.pkl must be a non-empty dictionary keyed by question ID.'
        )
    question_ids = [str(qid) for qid in validation_generations]
    for qid, example in validation_generations.items():
        if not isinstance(example, dict):
            raise ValueError(f'Generation entry {qid!r} must be a dictionary.')
        missing = {'question', 'most_likely_answer', 'responses'} - set(example)
        if missing:
            raise ValueError(f'Generation entry {qid!r} is missing keys: {sorted(missing)}')
        greedy = example['most_likely_answer']
        if not isinstance(greedy, dict) or not isinstance(greedy.get('response'), str):
            raise ValueError(f'Generation entry {qid!r} has an invalid greedy answer.')
        if greedy.get('accuracy') is None:
            raise ValueError(f'Generation entry {qid!r} has no greedy accuracy label.')
        if not greedy.get('token_log_likelihoods'):
            raise ValueError(f'Generation entry {qid!r} has no greedy token log-likelihoods.')
        if len(example['responses']) != args.num_generations:
            raise ValueError(
                f'Generation entry {qid!r} has {len(example["responses"])} stochastic '
                f'responses; exactly {args.num_generations} are required.'
            )
        low_responses = example.get('low_temp_answers')
        if not isinstance(low_responses, list) or len(low_responses) != 3:
            raise ValueError(f'Generation entry {qid!r} must contain exactly 3 low-T responses.')
        for response_idx, response in enumerate(low_responses):
            if (not isinstance(response, (tuple, list)) or len(response) < 2
                    or not isinstance(response[0], str) or not response[0].strip()
                    or response[1] is None or len(response[1]) == 0):
                raise ValueError(f'Generation entry {qid!r} has invalid low-T response {response_idx}.')
        for response_idx, response in enumerate(example['responses'][:args.num_generations]):
            if not isinstance(response, (tuple, list)) or len(response) < 2:
                raise ValueError(
                    f'Generation entry {qid!r} response {response_idx} must contain '
                    '(text, token_log_likelihoods, ...).'
                )
            if not isinstance(response[0], str) or not response[0].strip():
                raise ValueError(f'Generation entry {qid!r} response {response_idx} is empty.')
            if response[1] is None or len(response[1]) == 0:
                raise ValueError(
                    f'Generation entry {qid!r} response {response_idx} has no token '
                    'log-likelihoods.'
                )

    results_old = {}
    uncertainty_pkl_path = f"{args.data_path}/uncertainty_measures.pkl"
    if os.path.isfile(uncertainty_pkl_path):
        with open(uncertainty_pkl_path, 'rb') as infile:
            results_old = pickle.load(infile)
        if not isinstance(results_old, dict):
            results_old = {}

    save_embedding_path = f'{args.data_path}/embedding_and_similarity.pkl'
    save_dict = None

    if os.path.isfile(save_embedding_path):
        with open(save_embedding_path, 'rb') as infile:
            save_dict = pickle.load(infile)
        if not isinstance(save_dict, dict) or [
                str(qid) for qid in save_dict.get('question_ids', [])
        ] != question_ids:
            print('Cached similarities do not match validation_generations.pkl; recomputing.')
            save_dict = None
            
    if save_dict is None:
        # save_dict["entail"] = {}
        save_dict_exist = False
    else:
        save_dict_exist = True

    if save_dict_exist and (args.embedding_model in save_dict.keys()):
        embedding_exist = True
    else:
        embedding_exist = False
    
    if save_dict_exist and ('entail' in save_dict) and ('list_generation_luq_similarity' in save_dict['entail']):
        luq_sim_exist = True
    else:
        luq_sim_exist = False
        
    if save_dict_exist and ('lexsim' in save_dict.keys() or 'rougel' in save_dict.keys()):
        lexsim_exist = True
    else:
        lexsim_exist = False
        
    if save_dict_exist and ('jaccardlex' in save_dict.keys()):
        jaccardlex_exist = True
    else:
        jaccardlex_exist = False
    
    if save_dict_exist and ('sar' in save_dict.keys()):
        sar_exist = True
    else:
        sar_exist = False
    
    if save_dict_exist and ('eigenscore' in save_dict.keys()):
        eigenscore_exist = True
    else:
        eigenscore_exist = False

    print(f"Save dict exist is {save_dict_exist}. Embedding exist is {embedding_exist}. Lexsim exist is {lexsim_exist}. Jaccardlex exist is {jaccardlex_exist}. LUQ sim exist is {luq_sim_exist}. SAR exist is {sar_exist}. Eigenscore exist is {eigenscore_exist}")
        
    semantic_ids_aligned = (
        [str(qid) for qid in results_old.get('question_ids', [])] == question_ids
        and isinstance(results_old.get('semantic_ids'), list)
        and len(results_old['semantic_ids']) == len(question_ids)
        and all(len(ids) >= args.num_generations for ids in results_old['semantic_ids'])
    )
    if semantic_ids_aligned:
        # Check if list_semantic_ids needs slicing
        list_semantic_ids = results_old['semantic_ids']
        if list_semantic_ids is not None and isinstance(list_semantic_ids, list) and len(list_semantic_ids) > 0:
            if isinstance(list_semantic_ids[0], list):
                 list_semantic_ids = slice_1d(list_semantic_ids, args.num_generations)
    else:
        list_semantic_ids = None

    if save_dict is None:
        save_dict = {}
    save_dict['schema_version'] = 1
    save_dict['question_ids'] = question_ids
    
    precomputed_results = {
        'validation_generations': validation_generations,
        'save_embedding_path': save_embedding_path,
        'save_dict': save_dict,
        'save_dict_exist': save_dict_exist,
        'embedding_exist': embedding_exist,
        'lexsim_exist': lexsim_exist,
        'jaccardlex_exist': jaccardlex_exist,
        'luq_sim_exist': luq_sim_exist,
        'sar_exist': sar_exist,
        'list_semantic_ids': list_semantic_ids,
        'eigenscore_exist': eigenscore_exist
    }
    
    return precomputed_results


def collect_info(args, validation_generations, metric, entailment_model, embedding_model, rouge, tokenizer, list_semantic_ids, save_dict, save_embedding_path, save_list, load_list):
    print(f"Compute and save {save_list}")
    print(f"Load precomputed {load_list}")
    validation_is_true = []
    list_generation = []
    list_generation_log_likelihoods = []
    list_sample_embeddings = []
    list_most_likely_answer_embeddings, list_generation_embeddings = [], []
    list_generation_with_question, list_generation_with_question_embeddings = [], []
    list_generation_entailment_similarity, list_generation_with_question_entailment_similarity = [], []
    list_generation_embedding_similarity, list_generation_with_question_embedding_similarity = [], []
    list_generation_luq_similarity, list_generation_with_question_luq_similarity = [], []
    list_num_sets = []
    list_generation_lexcial_sim, list_generation_with_question_lexical_sim = [], []
    list_generation_jaccardlex_sim, list_generation_with_question_jaccardlex_sim = [], []
    list_sar_token_importance, list_sar_sentence_similarity_matrix = [], []
    list_sar_token_log_likelihoods = []

    for idx, tid in tqdm(enumerate(validation_generations)):
        example = validation_generations[tid]
        question = example['question']
        full_responses = example["responses"][:args.num_generations]
        example_generation = []
        example_generation_log_likelihoods = []
        example_generation_with_question = []
        token_log_likelihoods = []
        sample_embeddings = []
        
        for gen_info in full_responses:
            truncated_response = gen_info[0][-args.truncate_length:]
            example_generation.append(truncated_response)
            token_log_likelihoods.append(gen_info[1])
            # Length normalization of generation probability
            example_generation_log_likelihoods.append(np.mean(gen_info[1]))
            if args.condition_on_question:
                example_generation_with_question.append(f'{question} {truncated_response}')
            else:
                example_generation_with_question.append(truncated_response)
            sample_embeddings.append(gen_info[2].squeeze().float().numpy())
        
        most_likely_answer = example['most_likely_answer']
        if args.recompute_accuracy:
            is_true = False
            if args.metric == 'entail':
                is_true = metric(most_likely_answer['response'], example, entailment_model, strict_entailment=args.strict_entailment)
            elif args.metric == 'squad_raw':
                is_true = (metric(most_likely_answer['response'], example, None) >= args.metric_threshold)
            else:
                is_true = metric(most_likely_answer['response'], example, None)
            is_true = is_true * 1.0
        else:
            is_true = most_likely_answer['accuracy']
        
        validation_is_true.append(is_true)
        list_generation.append(example_generation)
        list_generation_log_likelihoods.append(example_generation_log_likelihoods)
        list_generation_with_question.append(example_generation_with_question)
        
        # Calculate similarity matrix based on entailment
        if 'entail' in save_list:
            generation_entailment_similarity = entailment_similarity_matrix(entailment_model, example_generation)
            generation_with_question_entailment_similarity = entailment_similarity_matrix(entailment_model, example_generation_with_question)
            list_generation_entailment_similarity.append(generation_entailment_similarity)
            list_generation_with_question_entailment_similarity.append(generation_with_question_entailment_similarity)
        
        # Calculate embeddings
        if 'embedding' in save_list:
            example_embeddings = embedding_model.encode([most_likely_answer['response']] + example_generation + example_generation_with_question, normalize_embeddings=True)
            generation_embeddings = example_embeddings[1:args.num_generations+1]
            generation_with_question_embeddings = example_embeddings[args.num_generations+1:]
            generation_embedding_similarity = embedding_model.similarity(generation_embeddings, generation_embeddings)
            generation_with_question_embedding_similarity = embedding_model.similarity(generation_with_question_embeddings, generation_with_question_embeddings)
            
            # Add to lists
            list_most_likely_answer_embeddings.append(example_embeddings[0])
            list_generation_embeddings.append(generation_embeddings)
            list_generation_with_question_embeddings.append(generation_with_question_embeddings)
            list_generation_embedding_similarity.append(generation_embedding_similarity)
            list_generation_with_question_embedding_similarity.append(generation_with_question_embedding_similarity)
        
        # Get lexical similarity matrix
        if 'lexsim' in save_list:
            generation_lexical_sim = lexical_similarity_matrix(rouge, example_generation, tokenizer=tokenizer)
            generation_with_question_lexical_sim = lexical_similarity_matrix(rouge, example_generation_with_question, tokenizer=tokenizer)
            
            list_generation_lexcial_sim.append(generation_lexical_sim)
            list_generation_with_question_lexical_sim.append(generation_with_question_lexical_sim)
            
        # Get jaccardlex similarity matrix
        if 'jaccardlex' in save_list:
            generation_jaccardlex_sim = jaccardlex_similarity_matrix(example_generation)
            generation_with_question_jaccardlex_sim = jaccardlex_similarity_matrix(example_generation_with_question)
            
            list_generation_jaccardlex_sim.append(generation_jaccardlex_sim)
            list_generation_with_question_jaccardlex_sim.append(generation_with_question_jaccardlex_sim)
        
        # Get LUQ similarity matrix
        if 'luq' in save_list:
            generation_luq_similarity = entailment_similarity_matrix(entailment_model, example_generation, strict_entailment=False, exclude_neutral=True, bidirectional=False)
            generation_with_question_luq_similarity = entailment_similarity_matrix(entailment_model, example_generation_with_question, strict_entailment=False, exclude_neutral=True, bidirectional=False)
            list_generation_luq_similarity.append(generation_luq_similarity)
            list_generation_with_question_luq_similarity.append(generation_with_question_luq_similarity)
        
        # Calculate tokenwise importance and sentence similarity
        if 'sar' in save_list:
            token_importance_list = get_tokenwise_importance(
                entailment_model, tokenizer, example_generation, question
            )
            sentence_similarity_matrix = get_sentence_similarites(
                entailment_model, example_generation_with_question
            )
            list_sar_token_importance.append(token_importance_list)
            list_sar_sentence_similarity_matrix.append(sentence_similarity_matrix)
            list_sar_token_log_likelihoods.append(token_log_likelihoods)
        
        if 'eigenscore' in save_list:
            list_sample_embeddings.append(sample_embeddings)
        
        # Get num sets
        semantic_ids = list_semantic_ids[idx]
        num_sets = max(semantic_ids) + 1
        list_num_sets.append(num_sets)
        
    print(Counter(validation_is_true))
    # If a Gemini JSON with per-example correctness is provided, overwrite
    # the computed `validation_is_true` ordering to match the pkl ordering.
    if getattr(args, 'gemini_json', None):
        try:
            gemini_labels = None
            with open(args.gemini_json, 'r') as f:
                gemini_json = json.load(f)
            gemini_labels = gemini_json.get('per_example', gemini_json)
            # Build list in same order as `validation_generations` iteration
            new_validation_is_true = []
            if isinstance(validation_generations, dict):
                for tid in validation_generations:
                    ex_id = str(tid)
                    new_validation_is_true.append(float(gemini_labels.get(ex_id, 0)))
            else:
                for example in validation_generations:
                    ex_id = str(example.get('id')) if 'id' in example else str(example.get('question', ''))
                    new_validation_is_true.append(float(gemini_labels.get(ex_id, 0)))

            if len(new_validation_is_true) == len(validation_is_true):
                validation_is_true = new_validation_is_true
                print(f"Using Gemini labels to overwrite validation_is_true. Loaded {len(new_validation_is_true)} labels.")
            else:
                print("Gemini labels length mismatch with validation set; skipping overwrite.")
        except Exception as e:
            print(f"Failed to load/parse gemini_json {args.gemini_json}: {e}")
    
    result_dict = {
        # Generation info
        'validation_is_true': validation_is_true,
        'list_generation': list_generation,
        'list_generation_with_question': list_generation_with_question,
        'list_generation_log_likelihoods': list_generation_log_likelihoods,
        # Embedding
        'list_most_likely_answer_embeddings': list_most_likely_answer_embeddings,
        'list_generation_embeddings': list_generation_embeddings,
        'list_generation_with_question_embeddings': list_generation_with_question_embeddings,
        'list_generation_embedding_similarity': list_generation_embedding_similarity,
        'list_generation_with_question_embedding_similarity': list_generation_with_question_embedding_similarity,
        # Entailment
        'list_generation_entailment_similarity': list_generation_entailment_similarity,
        'list_generation_with_question_entailment_similarity': list_generation_with_question_entailment_similarity,
        # LUQ
        'list_generation_luq_similarity': list_generation_luq_similarity,
        'list_generation_with_question_luq_similarity': list_generation_with_question_luq_similarity,
        # BB methods
        'list_num_sets': list_num_sets, 
        'list_generation_lexcial_sim': list_generation_lexcial_sim,
        'list_generation_with_question_lexical_sim': list_generation_with_question_lexical_sim,
        # Jaccardlex
        'list_generation_jaccardlex_sim': list_generation_jaccardlex_sim,
        'list_generation_with_question_jaccardlex_sim': list_generation_with_question_jaccardlex_sim,
        # SAR
        'list_sar_token_importance': list_sar_token_importance,
        'list_sar_sentence_similarity_matrix': list_sar_sentence_similarity_matrix,
        'list_sar_token_log_likelihoods': list_sar_token_log_likelihoods,
        'list_sample_embeddings': list_sample_embeddings
    }
    
    result_dict = save_or_load_results(
        args, 
        result_dict, 
        save_dict, 
        save_embedding_path, 
        save_list,
        load_list
    )

    return result_dict


def slice_1d(arr, num):
    return [x[:num] for x in arr]


def slice_2d(arr, num):
    return [x[:num,:num] for x in arr]


def save_or_load_results(args, result_dict, save_dict, save_embedding_path, save_list, load_list):
    if save_dict is None:
        save_dict = {}

    if 'entail' in save_list:
        print("Save entailment.")
        save_dict["entail"] = {
            'list_generation_entailment_similarity': result_dict['list_generation_entailment_similarity'],
            'list_generation_with_question_entailment_similarity': result_dict['list_generation_with_question_entailment_similarity']
        }
    elif 'entail' in load_list:
        print("Load precomputed entailment.")
        result_dict['list_generation_entailment_similarity'] = slice_2d(
            save_dict['entail']['list_generation_entailment_similarity'],
            args.num_generations) 
        result_dict['list_generation_with_question_entailment_similarity'] = slice_2d(
            save_dict['entail']['list_generation_with_question_entailment_similarity'], args.num_generations)
        
    if 'embedding' in save_list:
        print("Save new embedding.")
        save_dict[args.embedding_model] = {
            'list_most_likely_answer_embeddings': result_dict['list_most_likely_answer_embeddings'],
            'list_generation_embeddings': result_dict['list_generation_embeddings'],
            'list_generation_with_question_embeddings': result_dict['list_generation_with_question_embeddings'],
            'list_generation_embedding_similarity': result_dict['list_generation_embedding_similarity'],
            'list_generation_with_question_embedding_similarity': result_dict['list_generation_with_question_embedding_similarity']
        }
    elif 'embedding' in load_list:
        print("Load precomputed embedding.")
        result_dict['list_most_likely_answer_embeddings'] = slice_1d(
            save_dict[args.embedding_model]['list_most_likely_answer_embeddings'],
            args.num_generations
        )
        result_dict['list_generation_embeddings'] = slice_1d(
            save_dict[args.embedding_model]['list_generation_embeddings'], 
            args.num_generations)
        result_dict['list_generation_with_question_embeddings'] = slice_1d(
            save_dict[args.embedding_model]['list_generation_with_question_embeddings'],
            args.num_generations)
        result_dict['list_generation_embedding_similarity'] = slice_2d(
            save_dict[args.embedding_model]['list_generation_embedding_similarity'],
            args.num_generations)
        result_dict['list_generation_with_question_embedding_similarity'] = slice_2d(
            save_dict[args.embedding_model]['list_generation_with_question_embedding_similarity'],
            args.num_generations)
        
    if 'lexsim' in save_list:
        print("Save lexical similarity.")
        save_dict['lexsim'] = {
            'list_generation_lexcial_sim': result_dict['list_generation_lexcial_sim'],
            'list_generation_with_question_lexical_sim': result_dict['list_generation_with_question_lexical_sim']
        }
    elif 'lexsim' in load_list:
        load_lexsim = True
        if 'lexsim' in save_dict.keys():
            lexsim_key = 'lexsim'
            embed1_key = 'list_generation_lexcial_sim'
            embed2_key = 'list_generation_with_question_lexical_sim'
        elif 'rougel' in save_dict.keys():
            lexsim_key = 'rougel'
            embed1_key = 'list_generation_rougel_similarity'
            embed2_key = 'list_generation_with_question_rougel_similarity'
        else:
            load_lexsim = False
        
        if load_lexsim:
            print("Load precomputed lexical similarity.")
            result_dict['list_generation_lexcial_sim'] = slice_2d(
                save_dict[lexsim_key][embed1_key],
                args.num_generations)
            result_dict['list_generation_with_question_lexical_sim'] = slice_2d(
                save_dict[lexsim_key][embed2_key],
                args.num_generations)
        else:
            print("Don't load precomputed lexical similarity.")
            
    if 'jaccardlex' in save_list:
        print("Save jaccardlex similarity.")
        save_dict['jaccardlex'] = {
            'list_generation_jaccardlex_sim': result_dict['list_generation_jaccardlex_sim'],
            'list_generation_with_question_jaccardlex_sim': result_dict['list_generation_with_question_jaccardlex_sim']
        }
    elif 'jaccardlex' in load_list:
        if 'jaccardlex' in save_dict.keys():
            print("Load precomputed jaccardlex similarity.")
            result_dict['list_generation_jaccardlex_sim'] = slice_2d(
                save_dict['jaccardlex']['list_generation_jaccardlex_sim'],
                args.num_generations)
            result_dict['list_generation_with_question_jaccardlex_sim'] = slice_2d(
                save_dict['jaccardlex']['list_generation_with_question_jaccardlex_sim'],
                args.num_generations)
        else:
            print("Precomputed jaccardlex missing.")
    
    if 'luq' in save_list:
        print("Save LUQ similarity matrix.")
        save_dict.setdefault('entail', {})
        save_dict['entail']['list_generation_luq_similarity'] = result_dict['list_generation_luq_similarity']
        save_dict['entail']['list_generation_with_question_luq_similarity'] = result_dict['list_generation_with_question_luq_similarity']
    elif 'luq' in load_list:
        print("Load precomputed LUQ similarity matrix.")
        result_dict['list_generation_luq_similarity'] = slice_2d(
            save_dict['entail']['list_generation_luq_similarity'],
            args.num_generations) 
        result_dict['list_generation_with_question_luq_similarity'] = slice_2d(
            save_dict['entail']['list_generation_with_question_luq_similarity'], args.num_generations)
    
    if 'sar' in save_list:
        print("Save SAR token importance and sentence similarity matrix.")
        save_dict['sar'] = {
            'list_sar_token_importance': result_dict['list_sar_token_importance'],
            'list_sar_sentence_similarity_matrix': result_dict['list_sar_sentence_similarity_matrix'],
            'list_sar_token_log_likelihoods': result_dict['list_sar_token_log_likelihoods']
        }
    elif 'sar' in load_list:
        print("Load precomputed SAR token importance and sentence similarity matrix.")
        result_dict['list_sar_token_importance'] = slice_1d(
            save_dict['sar']['list_sar_token_importance'],
            args.num_generations) 
        result_dict['list_sar_sentence_similarity_matrix'] = slice_2d(
            save_dict['sar']['list_sar_sentence_similarity_matrix'], args.num_generations)
        result_dict['list_sar_token_log_likelihoods'] = slice_1d(
            save_dict['sar']['list_sar_token_log_likelihoods'],
            args.num_generations
        )
    
    if 'eigenscore' in save_list:
        save_dict['eigenscore'] = {
            'list_sample_embeddings': result_dict['list_sample_embeddings']
        }
    elif 'eigenscore' in load_list:
        result_dict['list_sample_embeddings'] = slice_1d(
            save_dict['eigenscore']['list_sample_embeddings'], 
            args.num_generations
        )
    
    with open(save_embedding_path, 'wb') as f:
        pickle.dump(save_dict, f)
        
    return result_dict


def print_best_scores(df, keyword='', list_scores=['auroc', 'auarc', 'prr']):
    similarity_with_keyword = []
    similarity_wo_keyword = []

    for sim in df.similarity.unique():
        if keyword in sim:
            similarity_with_keyword.append(sim)
        else:
            similarity_wo_keyword.append(sim)
            
    print(similarity_with_keyword, similarity_wo_keyword)
    
    for method in df.method.unique():
        print(f"Method {method}")
        df_method = df[df.method == method]
        # print(df_method.head())
        df_method_sim = df_method[df_method.similarity.isin(similarity_with_keyword)]
        # print(df_method_sim.head())
        for score in list_scores:
            try:
                print(f"{score}: {df_method_sim[score].max()}")
            except KeyError:
                print(f"{score} is not calculated.")


def load_gemini_labels(args, validation_generations):
    """Load per-example Gemini correctness JSON and return labels list matching
    iteration order over `validation_generations` (dict keys or list order).
    Returns None on failure.
    """
    if not getattr(args, 'gemini_json', None):
        return None
    try:
        with open(args.gemini_json, 'r') as f:
            gemini_json = json.load(f)
        gemini_labels = gemini_json.get('per_example', gemini_json)
        expected_ids = (
            [str(tid) for tid in validation_generations]
            if isinstance(validation_generations, dict)
            else [str(example.get('id', example.get('question_id', '')))
                  for example in validation_generations]
        )
        missing_ids = [ex_id for ex_id in expected_ids if ex_id not in gemini_labels]
        if missing_ids:
            raise ValueError(
                f'{len(missing_ids)} generated question IDs are absent from the Gemini JSON; '
                f'first missing ID: {missing_ids[0]}'
            )
        new_validation_is_true = []
        if isinstance(validation_generations, dict):
            for tid in validation_generations:
                ex_id = str(tid)
                new_validation_is_true.append(float(gemini_labels[ex_id]))
        else:
            for example in validation_generations:
                ex_id = str(example.get('id', example.get('question_id', '')))
                new_validation_is_true.append(float(gemini_labels[ex_id]))
        return new_validation_is_true
    except Exception as e:
        raise ValueError(f"Invalid gemini_json {args.gemini_json}: {e}") from e
