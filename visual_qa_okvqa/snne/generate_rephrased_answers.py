"""Generate answers for rephrased questions using actual dataset for context and ground truth."""
import gc
import os
import logging
import random
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import wandb

from snne.uncertainty.utils.data_utils import load_ds
from snne.uncertainty.utils import utils
from snne.uncertainty.utils.metric_utils import get_metric, get_reference
from snne.uncertainty.uncertainty_measures import p_true as p_true_utils
from snne.compute_uncertainty_measures import main as main_compute


utils.setup_logger()


def main(args):
    """
    Generate answers for rephrased questions.
    
    Key differences from generate_answers.py:
    1. No training split generation (only validation)
    2. Loads rephrased questions from CSV (but still loads actual dataset for context/answers)
    3. Maps CSV original_id to dataset examples
    4. Uses rephrased question + original context + original ground truth
    """

    # Setup run (identical to generate_answers.py)
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
    user = os.environ['USER']
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")
    args.run_name = utils.get_run_name("generate_rephrased_answers", args)

    wandb.init(
        entity=args.entity,
        project="snne" if not args.debug else "snne_debug",
        name=args.run_name,
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')

    # Get accuracy metric
    metric = get_metric(args.metric)

    # Load dataset (need actual dataset for context and ground truth)
    # Even though we're not generating training set, we need it for few-shot prompts
    train_num_samples = args.num_samples + args.num_few_shot + args.p_true_num_fewshot
    val_num_samples = args.num_samples
    train_dataset, validation_dataset = load_ds(
        args.dataset, 
        add_options=args.use_mc_options, 
        seed=args.random_seed,
        train_num_samples=train_num_samples,
        val_num_samples=val_num_samples
    )
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices of answerable and unanswerable questions (for few-shot prompt)
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt (identical to generate_answers.py)
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

    # Initialize model
    model = utils.init_model(args)

    # Initialize prompt for p_true baseline (if needed)
    if args.compute_p_true:
        logging.info(80*'#')
        logging.info('Constructing few-shot prompt for p_true.')

        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=model, dataset=train_dataset, indices=p_true_indices,
            prompt=prompt, brief=BRIEF,
            brief_always=args.brief_always and args.enable_brief,
            make_prompt=make_prompt, num_generations=args.num_generations,
            metric=metric)
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))
        experiment_details['p_true_indices'] = p_true_indices
        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
        logging.info('Finished constructing few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    # Load CSV with rephrased questions
    logging.info('Loading rephrased questions from CSV: %s', args.input_csv)
    df = pd.read_csv(args.input_csv)
    
    # Validate CSV structure
    required_cols = ['id', 'original_id', 'question']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'CSV must have columns: {required_cols}. Missing: {missing_cols}')
    
    logging.info('Loaded %d rephrased questions from CSV.', len(df))
    
    # Create mapping: original_id -> list of {id, question, rephrase_idx}
    rephrase_map = {}
    for _, row in df.iterrows():
        orig_id = str(row['original_id'])  # Ensure string comparison
        rephrased_q = row['question']
        rephrased_id = row['id']
        rephrase_idx = row.get('rephrase_idx', None)
        
        if orig_id not in rephrase_map:
            rephrase_map[orig_id] = []
        rephrase_map[orig_id].append({
            'id': rephrased_id,
            'question': rephrased_q,
            'rephrase_idx': rephrase_idx
        })
    
    logging.info('Created rephrase map for %d unique original IDs.', len(rephrase_map))

    # Create dataset ID to example mapping for fast lookup
    dataset_map = {str(example['id']): example for example in validation_dataset}
    logging.info('Created dataset map with %d validation examples.', len(dataset_map))

    # Start answer generation
    logging.info(80 * '=')
    logging.info('Generating answers for rephrased questions (validation split only):')
    logging.info(80 * '=')

    # This will store all input data and model predictions (same structure as generate_answers.py)
    accuracies, generations, results_dict, p_trues = [], {}, {}, []

    # Sample indices for validation (only those with rephrased questions)
    available_ids = list(set(dataset_map.keys()) & set(rephrase_map.keys()))
    
    if not available_ids:
        raise ValueError('No matching IDs found between dataset and CSV! '
                        'Check that original_id in CSV matches dataset IDs.')
    
    logging.info('Found %d validation examples with rephrased questions.', len(available_ids))
    
    # Sample subset if needed
    num_to_sample = min(args.num_samples, len(available_ids))
    sampled_ids = random.sample(available_ids, num_to_sample)
    experiment_details['validation'] = {'sampled_original_ids': sampled_ids}
    
    if args.num_samples > len(available_ids):
        logging.warning('Not enough samples with rephrased questions. Using all %d samples.', 
                       len(available_ids))

    # Iterate over sampled original IDs
    it = 0
    total_rephrased = sum(len(rephrase_map[orig_id]) for orig_id in sampled_ids)
    logging.info('Will generate answers for %d rephrased questions from %d original questions.',
                total_rephrased, len(sampled_ids))
    
    pbar = tqdm(total=total_rephrased, desc="Generating answers")
    
    for original_id in sampled_ids:
        # Get original example from dataset
        example = dataset_map[original_id]
        context = example['context']
        correct_answer = example['answers']['text']
        original_question = example['question']
        
        # Process each rephrased version of this question
        for rephrase_info in rephrase_map[original_id]:
            if (it + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            rephrased_question = rephrase_info['question']
            rephrased_id = rephrase_info['id']
            rephrase_idx = rephrase_info['rephrase_idx']
            
            # Initialize generation entry (same structure as generate_answers.py)
            generations[rephrased_id] = {
                'question': rephrased_question,
                'context': context,
                'original_id': original_id,
                'original_question': original_question,
                'rephrase_idx': rephrase_idx
            }

            # Create prompt with rephrased question but original context
            current_input = make_prompt(
                context, rephrased_question, None, BRIEF, 
                args.brief_always and args.enable_brief)
            local_prompt = prompt + current_input

            logging.info('Current input: '.ljust(15) + current_input)

            full_responses = []

            # Reset seed if needed
            if args.reset_seed:
                torch.manual_seed(args.random_seed)

            num_generations = args.num_generations + 1
            
            for i in range(num_generations):
                # Temperature for first generation is always `0.1`
                temperature = 0.1 if i == 0 else args.temperature
                min_p = 0.0 if i == 0 else args.min_p

                predicted_answer, token_log_likelihoods, embedding = model.predict(
                    local_prompt, temperature, min_p=min_p)
                embedding = embedding.cpu() if embedding is not None else None

                # Compute accuracy using ground truth from original dataset
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0

                if i == 0:
                    logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                    logging.info('original_id: '.ljust(20) + str(original_id))
                    logging.info('rephrase_idx: '.ljust(20) + str(rephrase_idx))
                    if args.use_context:
                        logging.info('context: '.ljust(20) + str(context))
                    logging.info('original question: '.ljust(20) + original_question)
                    logging.info('rephrased question: '.ljust(20) + rephrased_question)
                    logging.info('low-t prediction: '.ljust(20) + predicted_answer)
                    logging.info('correct answer: '.ljust(20) + str(correct_answer))
                    logging.info('accuracy: '.ljust(20) + str(acc))

                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': embedding,
                        'accuracy': acc
                    }
                    generations[rephrased_id].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': get_reference(example)
                    })

                else:
                    logging.info('high-t prediction '.ljust(20) + str(i) + ' : ' + predicted_answer)
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding, acc))

            # Append all predictions for this example to `generations`
            generations[rephrased_id]['responses'] = full_responses

            if args.compute_p_true:
                # Compute p_true here to avoid cost in compute_uncertainty script
                p_true = p_true_utils.calculate_p_true(
                    model, rephrased_question, most_likely_answer_dict['response'],
                    [r[0] for r in full_responses], p_true_few_shot_prompt,
                    hint=args.p_true_hint)
                p_trues.append(p_true)
                logging.info('p_true: %s', p_true)
            
            pbar.update(1)
    
    pbar.close()

    # Save generations (use utils.save which handles wandb.run.dir automatically)
    logging.info('Saving validation_generations.pkl...')
    utils.save(generations, 'validation_generations.pkl')
    logging.info('Saved %d rephrased question generations.', len(generations))

    # Log overall accuracy
    accuracy = np.mean(accuracies)
    print(f"Overall validation split accuracy: {accuracy}")
    wandb.log({"validation_accuracy": accuracy})

    # Save uncertainty measures (if p_true computed)
    if args.compute_p_true:
        results_dict['uncertainty_measures'] = {
            'p_false':  [1 - p for p in p_trues],
            'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
        }
    utils.save(results_dict, 'uncertainty_measures.pkl')

    # Save experiment details
    utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')
    del model


if __name__ == '__main__':

    parser = utils.get_parser()
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to rephrased questions CSV with columns: id, original_id, question.')
    
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unknown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # First sample generations from LLM
    logging.info('STARTING `generate_rephrased_answers`!')
    main(args)
    logging.info('FINISHED `generate_rephrased_answers`!')

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')
