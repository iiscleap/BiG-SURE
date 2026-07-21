"""Sample answers from LLMs on QA task with 3-tier sampling.

For each datapoint, this script generates:
- 1 greedy answer (temp=0) - the deterministic "most likely" answer
- 3 low-temperature samples (temp=0.1) - near-greedy but with slight variation
- N high-temperature samples (default 10, temp=1.0) - for uncertainty estimation

Output fields:
- `greedy_answer`: The single temp=0 response
- `low_temp_answers`: List of 3 temp=0.1 responses
- `most_likely_answer`: Alias for greedy_answer (backward compatibility)
- `most_likely_answers`: Combined list [greedy + low_temp] (backward compatibility)
- `responses`: List of high-temp samples
- `ground_truth`: List of correct answers
"""
import gc
import os
import logging
import random
from tqdm import tqdm

import argparse
import numpy as np
import pandas as pd
import torch
import wandb
import json
from snne.uncertainty.utils.data_utils import load_ds
from snne.uncertainty.utils import utils
from snne.uncertainty.utils.metric_utils import get_metric, get_reference
from snne.utils.run_manifest import record_wandb_run


utils.setup_logger()


def generate_tier_batched(model, prompts, temperature, batch_size=8, num_samples_per_prompt=1, return_logits=False):
    """
    Generate answers for all prompts at a given temperature.
    
    NOTE: Currently uses sequential generation (predict) instead of predict_batch
    because batch mode has issues with padding affecting answer extraction.
    
    Args:
        model: The HuggingFace model
        prompts: List of full prompts (few-shot + question)
        temperature: Temperature for this tier
        batch_size: Unused (kept for API compatibility)
        num_samples_per_prompt: How many samples to generate per prompt
        return_logits: Whether to return full logits
    
    Returns:
        List of results, where each result is a list of tuples
    """
    from tqdm import tqdm
    
    all_results = [[] for _ in range(len(prompts))]
    
    for sample_idx in range(num_samples_per_prompt):
        logging.info(f'Generating sample {sample_idx + 1}/{num_samples_per_prompt} at temp={temperature}')
        
        # Use sequential generation (proven to work correctly)
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"Sample {sample_idx+1}")):
            result = model.predict(prompt, temperature, return_logits=return_logits)
            all_results[prompt_idx].append(result)
    
    return all_results


def main(args):

    # Setup run.
    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            args.use_context = True
        args.compute_std = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True

    experiment_details = {'args': args}
    utils.set_all_seeds(args.random_seed)
    
    # Initialize a dedicated random state for data-related sampling to ensure
    # consistency across different model seeds if data_seed is fixed.
    data_random = random.Random(args.data_seed)
    user = os.environ.get('USER', 'user')
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")
    args.run_name = utils.get_run_name("generate_answers", args)

    wandb_dir = args.wandb_dir if args.wandb_dir else f"{scratch_dir}/{user}/uncertainty"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    wandb.init(
        entity=args.entity,
        project=os.getenv('WANDB_PROJECT', 'bigsure-text-qa') if not args.debug else "bigsure-text-qa-debug",
        name=args.run_name,
        dir=wandb_dir,
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')
    # Get accuracy metric.
    metric = get_metric(args.metric)

    # Load dataset.
    train_num_samples = args.num_samples + args.num_few_shot + args.p_true_num_fewshot
    val_num_samples = args.num_samples
    train_dataset, validation_dataset = load_ds(
        args.dataset, 
        add_options=args.use_mc_options, 
        seed=args.data_seed,
        train_num_samples=train_num_samples,
        val_num_samples=val_num_samples
    )
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    # Force TriviaQA to use the bundled canonical subset so all stages align.
    validation_ids = None
    if args.dataset == 'trivia_qa':
        if not args.subset_csv or not os.path.exists(args.subset_csv):
            raise FileNotFoundError(
                'TriviaQA requires --subset_csv pointing to the canonical subset CSV.')
        logging.info('Forcing TriviaQA validation subset from %s', args.subset_csv)
        df_val = pd.read_csv(args.subset_csv)
        validation_ids = set(df_val['id'].astype(str))

        original_answerable_count = len(answerable_indices)
        answerable_indices = [
            i for i in answerable_indices
            if str(train_dataset[i]['id']) not in validation_ids
        ]
        logging.info(
            'Excluded %d validation samples from few-shot pool.',
            original_answerable_count - len(answerable_indices))

    if args.answerable_only:
        unanswerable_indices = []
        # (TriviaQA case) We already have validation_dataset = train_dataset if we followed my load_ds fix
        # But split_dataset only works on the passed dataset.
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        # NOTE: If we use the full dataset, we should be careful here.
        # But TriviaQA-in-SQuAD-format is mostly answerable.
        # validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = data_random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, 
        prompt_indices, 
        BRIEF, 
        args.brief_always if args.enable_brief else True, 
        make_prompt
    )
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info(f'Prompt is: {prompt}')

    # Initialize model.
    model = utils.init_model(args)

    # Initialize prompt for p_true baseline.
    # if args.compute_p_true:
    #     logging.info(80*'#')
    #     logging.info('Constructing few-shot prompt for p_true.')

    #     p_true_indices = data_random.sample(answerable_indices, args.p_true_num_fewshot)
    #     remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
    #     p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
    #         model=model, dataset=train_dataset, indices=p_true_indices,
    #         prompt=prompt, brief=BRIEF,
    #         brief_always=args.brief_always and args.enable_brief,
    #         make_prompt=make_prompt, num_generations=args.num_generations,
    #         metric=metric)
    #     wandb.config.update(
    #         {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
    #     wandb.log(dict(len_p_true=len_p_true))
    #     experiment_details['p_true_indices'] = p_true_indices
    #     experiment_details['p_true_responses'] = p_true_responses
    #     experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
    #     logging.info('Finished constructing few-shot prompt for p_true.')
    #     logging.info(80*'#')
    #     logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
    #     logging.info(80*'#')

    # Generation parameters
    n_greedy = 1
    n_low_temp = args.n_low_temp  # default 3
    n_high_temp = args.num_generations  # default 10
    low_temp = args.low_temp  # default 0.1
    high_temp = args.temperature  # default 1.0
    batch_size = getattr(args, 'batch_size', 8)  # default 8

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers with batched inference:')
    logging.info(80 * '=')
    
    for dataset_split in ['validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))
        else:
            dataset = validation_dataset
            
            # Use fixed TriviaQA IDs if available
            if args.dataset == 'trivia_qa' and validation_ids is not None:
                logging.info(f'Filtering TriviaQA validation set using CSV-loaded IDs.')
                indices = []
                for idx, item in enumerate(dataset):
                    if str(item['id']) in validation_ids:
                        indices.append(idx)
                
                logging.info(f"Found {len(indices)} matching samples in dataset for TriviaQA CSV.")
                experiment_details[dataset_split] = {'indices': indices}
            # Otherwise use specific IDs if provided via arg
            elif args.sample_ids_path:
                logging.info(f'Filtering {dataset_split} dataset using IDs from {args.sample_ids_path}.')
                with open(args.sample_ids_path, 'r') as f:
                    target_ids = set(json.load(f))
                
                indices = []
                for idx, item in enumerate(dataset):
                    if str(item['id']) in target_ids:
                        indices.append(idx)
                
                if not indices:
                    logging.warning(f"No matching IDs found in {dataset_split} dataset for {args.sample_ids_path}. Fallback to random.")
                    possible_indices = range(0, len(dataset))
                    indices = data_random.sample(possible_indices, min(args.num_samples, len(dataset)))
                else:
                    logging.info(f"Found {len(indices)} matching samples in dataset.")
                    experiment_details[dataset_split] = {'indices': indices}
            else:
                possible_indices = range(0, len(dataset))
                indices = data_random.sample(possible_indices, min(args.num_samples, len(dataset)))
                experiment_details[dataset_split] = {'indices': indices}

        if args.num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

        # Collect all examples and prompts
        examples = [dataset[idx] for idx in indices]
        prompts = []
        for example in examples:
            question, context = example["question"], example['context']
            current_input = make_prompt(
                context, question, None, BRIEF, args.brief_always and args.enable_brief)
            local_prompt = prompt + current_input
            prompts.append(local_prompt)

        logging.info(f'Collected {len(prompts)} prompts for batched generation')

        # Determine if we skip high-temp for training
        skip_high_temp = (dataset_split == 'train' and args.get_training_set_generations_most_likely_only)

        # Reset seed if needed
        if args.reset_seed:
            torch.manual_seed(args.random_seed)

        # TIER 1: Greedy generation (temp=0.0)
        logging.info(f'Generating Tier 1 (greedy, temp=0.0) for {len(prompts)} questions...')
        greedy_results = generate_tier_batched(
            model, prompts, temperature=0.0, 
            batch_size=batch_size, num_samples_per_prompt=n_greedy,
            return_logits=args.save_logits
        )
        gc.collect(); torch.cuda.empty_cache()

        # TIER 2: Low-temp generation (temp=0.1)
        logging.info(f'Generating Tier 2 (low-temp, temp={low_temp}) for {len(prompts)} questions, {n_low_temp} samples each...')
        low_temp_results = generate_tier_batched(
            model, prompts, temperature=low_temp,
            batch_size=batch_size, num_samples_per_prompt=n_low_temp
        )
        gc.collect(); torch.cuda.empty_cache()

        # TIER 3: High-temp generation (temp=1.0)
        if not skip_high_temp:
            logging.info(f'Generating Tier 3 (high-temp, temp={high_temp}) for {len(prompts)} questions, {n_high_temp} samples each...')
            high_temp_results = generate_tier_batched(
                model, prompts, temperature=high_temp,
                batch_size=batch_size, num_samples_per_prompt=n_high_temp
            )
            gc.collect(); torch.cuda.empty_cache()
        else:
            high_temp_results = [[] for _ in range(len(prompts))]

        # Process results and compute accuracy
        logging.info('Processing results and computing accuracy...')
        for i, example in enumerate(tqdm(examples, desc="Processing results")):
            question, context = example["question"], example['context']
            correct_answer = example['answers']['text']
            
            # Initialize generation entry with ground_truth
            generations[example['id']] = {
                'question': question, 
                'context': context,
                'ground_truth': correct_answer,  # Added ground_truth field
            }

            # Process greedy result
            greedy_answer_tuple = greedy_results[i][0]  # (answer, log_likelihoods, embedding, [logits])
            greedy_response = greedy_answer_tuple[0]
            greedy_log_likelihoods = greedy_answer_tuple[1]
            greedy_embedding = greedy_answer_tuple[2]
            greedy_logits = greedy_answer_tuple[3] if len(greedy_answer_tuple) > 3 else None
            
            greedy_acc = metric(greedy_response, example, model) if correct_answer else 0.0
            accuracies.append(greedy_acc)
            
            greedy_answer = {
                'response': greedy_response,
                'token_log_likelihoods': greedy_log_likelihoods,
                'embedding': greedy_embedding,
                'accuracy': greedy_acc,
                'temperature': 0.0
            }
            if greedy_logits is not None:
                greedy_answer['logits'] = greedy_logits
            
            # Process low-temp results
            low_temp_answers = []
            most_likely_answers = [greedy_answer]  # Start with greedy
            for lt_tuple in low_temp_results[i]:
                lt_response = lt_tuple[0]
                lt_log_likelihoods = lt_tuple[1]
                lt_embedding = lt_tuple[2]
                
                lt_acc = metric(lt_response, example, model) if correct_answer else 0.0
                accuracies.append(lt_acc)
                
                lt_answer = {
                    'response': lt_response,
                    'token_log_likelihoods': lt_log_likelihoods,
                    'embedding': lt_embedding,
                    'accuracy': lt_acc,
                    'temperature': low_temp
                }
                low_temp_answers.append(lt_answer)
                most_likely_answers.append(lt_answer)
            
            # Process high-temp results
            full_responses = []
            for ht_tuple in high_temp_results[i]:
                ht_response = ht_tuple[0]
                ht_log_likelihoods = ht_tuple[1]
                ht_embedding = ht_tuple[2]
                
                if args.compute_accuracy_at_all_temps:
                    ht_acc = metric(ht_response, example, model) if correct_answer else 0.0
                else:
                    ht_acc = 0.0
                
                full_responses.append((ht_response, ht_log_likelihoods, ht_embedding, ht_acc))
            
            # Log sample info
            if i < 5:  # Log first 5 examples
                logging.info(f'Example {i}: question="{question[:50]}...", greedy="{greedy_response}", acc={greedy_acc}')
            
            # Store in generations
            generations[example['id']].update({
                'greedy_answer': greedy_answer,
                'low_temp_answers': low_temp_answers,
                'most_likely_answers': most_likely_answers,
                'most_likely_answer': greedy_answer,
                'reference': get_reference(example)
            })
            generations[example['id']]['responses'] = full_responses

            # if args.compute_p_true and dataset_split == 'validation':
            #     p_true = p_true_utils.calculate_p_true(
            #         model, question, greedy_response,
            #         [r[0] for r in full_responses], p_true_few_shot_prompt,
            #         hint=args.p_true_hint)
            #     p_trues.append(p_true)

        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})

        utils.save(results_dict, 'uncertainty_measures.pkl')

    utils.save(experiment_details, 'experiment_details.pkl')
    record_wandb_run(
        os.getenv('BIGSURE_RUN_MANIFEST'), args.dataset, args.model_name,
        args.random_seed, 'vanilla', wandb.run.dir)
    logging.info('Run complete.')
    del model


if __name__ == '__main__':

    parser = utils.get_parser()
    parser.add_argument('--n_low_temp', type=int, default=3,
                        help='Number of low-temperature samples (temp=0.1) per question. Default: 3')
    parser.add_argument('--low_temp', type=float, default=0.1,
                        help='Temperature for low-temp samples. Default: 0.1')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for batched inference. Default: 8')
    parser.add_argument('--save_logits', action=argparse.BooleanOptionalAction, default=False,
                        help='Save all logits for the greedy answer.')
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help='Custom directory for wandb logging.')
    parser.add_argument('--subset_csv', type=str, default=None,
                        help='Canonical subset CSV; required for TriviaQA.')
    parser.add_argument('--sample_ids_path', type=str, default=None,
                        help='Path to a JSON file containing a list of sample IDs to process.')
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default.
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        # logging.info(50 * '#X')
        # logging.info('STARTING `compute_uncertainty_measures`!')
        # main_compute(args)
        # logging.info('FINISHED `compute_uncertainty_measures`!')
