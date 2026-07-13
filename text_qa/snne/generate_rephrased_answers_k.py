"""Generate rephrased answers but produce k most-likely low-temp samples per question.

This is a copy of `generate_rephrased_answers.py` with an added feature:
- `--n_low_temp`: generate n low-temperature samples (stored in `low_temp_answers`).
- `--low_temp`: temperature to use for the k samples (default 1.0 as requested).

The script keeps backward-compatible fields: `most_likely_answer` will be the
first of the `most_likely_answers` list to avoid downstream breakage.
"""
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
from snne.utils.run_manifest import record_wandb_run


utils.setup_logger()


def main(args):
    """Main entry — same behavior as original but with k most-likely samples."""

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
    user = os.environ.get('USER', 'user')
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")
    args.run_name = utils.get_run_name("generate_rephrased_answers_k", args)

    wandb_dir = args.wandb_dir or f"{scratch_dir}/{user}/uncertainty"
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(
        entity=args.entity,
        project=os.getenv('WANDB_PROJECT', 'bigsure-text-qa') if not args.debug else "bigsure-text-qa-debug",
        name=args.run_name,
        dir=wandb_dir,
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')
    # Get accuracy metric
    metric = get_metric(args.metric)

    # Load dataset (need actual dataset for context and ground truth)
    train_num_samples = args.num_samples + args.num_few_shot + args.p_true_num_fewshot
    # Load ALL validation samples to ensure all rephrased questions can be matched
    val_num_samples = args.num_samples  # Large number to load full validation split
    train_dataset, validation_dataset = load_ds(
        args.dataset,
        add_options=args.use_mc_options,
        seed=args.data_seed,
        train_num_samples=train_num_samples,
        val_num_samples=val_num_samples
    )
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices and construct prompt
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt
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
    # if args.compute_p_true:
    #     logging.info(80*'#')
    #     logging.info('Constructing few-shot prompt for p_true.')

    #     p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
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

    # Load CSV with rephrased questions
    logging.info('Loading rephrased questions from CSV: %s', args.input_csv)
    df = pd.read_csv(args.input_csv)

    required_cols = ['id', 'original_id', 'question']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'CSV must have columns: {required_cols}. Missing: {missing_cols}')

    logging.info('Loaded %d rephrased questions from CSV.', len(df))

    # Create dataset map for lookup
    dataset_map = {str(example['id']): example for example in validation_dataset}
    logging.info('Created dataset map with %d validation examples.', len(dataset_map))
    
    # Check which original IDs are available in the loaded dataset
    csv_original_ids = set(str(row['original_id']) for _, row in df.iterrows())
    available_original_ids = csv_original_ids & set(dataset_map.keys())
    missing_original_ids = csv_original_ids - set(dataset_map.keys())
    
    logging.info(f'CSV contains {len(csv_original_ids)} unique original IDs')
    logging.info(f'Available in validation dataset: {len(available_original_ids)}')
    logging.info(f'Missing from validation dataset: {len(missing_original_ids)}')
    
    if len(missing_original_ids) > 0:
        logging.warning(f'First 5 missing IDs: {list(missing_original_ids)[:5]}')

    logging.info(80 * '=')
    logging.info('Generating answers for rephrased questions from CSV:')
    logging.info(80 * '=')

    accuracies, generations, results_dict, p_trues = [], {}, {}, []

    it = 0
    processed_count = 0
    skipped_count = 0
    processed_original_ids = set()
    
    pbar = tqdm(total=len(df), desc="Processing CSV rows")

    # Iterate directly through CSV rows
    for idx, row in df.iterrows():
        if (it + 1) % 10 == 0:
            gc.collect(); torch.cuda.empty_cache()
        it += 1

        original_id = str(row['original_id'])
        rephrased_question = row['question']
        rephrased_id = row['id']
        rephrase_idx = row.get('rephrase_idx', None)
        rephrased_context = row.get('context', None)

        # Skip original samples (rephrase_idx == 0)
        if rephrase_idx == 0:
            skipped_count += 1
            pbar.update(1)
            continue

        # Check if original_id exists in dataset
        if original_id not in dataset_map:
            skipped_count += 1
            pbar.update(1)
            continue

        # Get ground truth and original data from dataset
        example = dataset_map[original_id]
        original_context = example['context']
        correct_answer = example['answers']['text']
        original_question = example['question']

        # Use rephrased context if available, otherwise use original
        use_context = rephrased_context if rephrased_context is not None else original_context
        
        # Track processed original IDs
        processed_original_ids.add(original_id)

        generations[rephrased_id] = {
            'question': rephrased_question,
            'context': use_context,
            'original_id': original_id,
            'original_question': original_question,
            'original_context': original_context,
            'rephrase_idx': rephrase_idx,
            'ground_truth': correct_answer,
        }

        current_input = make_prompt(
            use_context, rephrased_question, None, BRIEF,
            args.brief_always and args.enable_brief)
        local_prompt = prompt + current_input

        logging.info('Current input: '.ljust(15) + current_input)

        full_responses = []
        if args.reset_seed:
            torch.manual_seed(args.random_seed)

        n_low_temp = args.n_low_temp  # number of low-temp samples (default 3)
        low_temp = args.low_temp  # temperature for low-temp (default 0.1)
        n_high_temp = args.num_generations  # number of high-temp samples
        
        most_likely_answers = []
        low_temp_answers = []

        # TIER 1: Greedy answer (temp=0)
        greedy_answer, greedy_log_likelihoods, greedy_embedding = model.predict(
            local_prompt, temperature=0.0, min_p=0.0)
        greedy_embedding = greedy_embedding.cpu() if greedy_embedding is not None else None
        
        greedy_acc = metric(greedy_answer, example, model) if correct_answer else 0.0
        accuracies.append(greedy_acc)
        
        greedy_dict = {
            'response': greedy_answer,
            'token_log_likelihoods': greedy_log_likelihoods,
            'embedding': greedy_embedding,
            'accuracy': greedy_acc,
            'temperature': 0.0
        }
        most_likely_answers.append(greedy_dict)
        
        processed_count += 1
        
        logging.info('Iteration ' + str(processed_count) + ':  ' + 80*'#')
        logging.info('original_id: '.ljust(20) + str(original_id))
        logging.info('rephrase_idx: '.ljust(20) + str(rephrase_idx))
        if args.use_context:
            logging.info('original context: '.ljust(20) + str(original_context))
            if rephrased_context is not None:
                logging.info('rephrased context: '.ljust(20) + str(rephrased_context))
        logging.info('original question: '.ljust(20) + original_question)
        logging.info('rephrased question: '.ljust(20) + rephrased_question)
        logging.info('greedy prediction (temp=0): '.ljust(20) + greedy_answer)
        logging.info('correct answer: '.ljust(20) + str(correct_answer))
        logging.info('accuracy: '.ljust(20) + str(greedy_acc))

        # TIER 2: Low-temp answers (temp=0.1)
        for i in range(n_low_temp):
            lt_answer, lt_log_likelihoods, lt_embedding = model.predict(
                local_prompt, temperature=low_temp, min_p=0.0)
            lt_embedding = lt_embedding.cpu() if lt_embedding is not None else None
            
            lt_acc = metric(lt_answer, example, model) if correct_answer else 0.0
            accuracies.append(lt_acc)
            
            lt_dict = {
                'response': lt_answer,
                'token_log_likelihoods': lt_log_likelihoods,
                'embedding': lt_embedding,
                'accuracy': lt_acc,
                'temperature': low_temp
            }
            low_temp_answers.append(lt_dict)
            most_likely_answers.append(lt_dict)
            logging.info(f'low-t prediction {i+1}/{n_low_temp}: '.ljust(20) + lt_answer)

        # TIER 3: High-temp answers (temp=1.0)
        for i in range(n_high_temp):
            ht_answer, ht_log_likelihoods, ht_embedding = model.predict(
                local_prompt, temperature=args.temperature, min_p=args.min_p)
            ht_embedding = ht_embedding.cpu() if ht_embedding is not None else None
            
            if args.compute_accuracy_at_all_temps:
                ht_acc = metric(ht_answer, example, model) if correct_answer else 0.0
            else:
                ht_acc = 0.0
            
            full_responses.append((ht_answer, ht_log_likelihoods, ht_embedding, ht_acc))
            logging.info(f'high-t prediction {i+1}/{n_high_temp}: '.ljust(20) + ht_answer)

        generations[rephrased_id].update({
            'greedy_answer': greedy_dict,
            'low_temp_answers': low_temp_answers,
            'most_likely_answers': most_likely_answers,
            'most_likely_answer': greedy_dict,  # First most_likely is greedy
            'reference': get_reference(example),
            'responses': full_responses,
        })

        # if args.compute_p_true:
        #     first_ml = None
        #     if generations[rephrased_id].get('most_likely_answer'):
        #         first_ml = generations[rephrased_id]['most_likely_answer'].get('response')
        #     elif generations[rephrased_id].get('most_likely_answers'):
        #         first_ml = generations[rephrased_id]['most_likely_answers'][0].get('response')

        #     p_true = p_true_utils.calculate_p_true(
        #         model, rephrased_question, first_ml,
        #         [r[0] for r in full_responses], p_true_few_shot_prompt,
        #         hint=args.p_true_hint)
        #     p_trues.append(p_true)
        #     logging.info('p_true: %s', p_true)

        pbar.update(1)

    pbar.close()

    logging.info('='*80)
    logging.info(f'Processed {processed_count} rephrased questions')
    logging.info(f'Unique original IDs processed: {len(processed_original_ids)}')
    logging.info(f'Skipped {skipped_count} rows (rephrase_idx==0 or missing original_id)')
    logging.info('='*80)
    
    experiment_details['validation'] = {
        'processed_original_ids': list(processed_original_ids),
        'num_processed': processed_count,
        'num_skipped': skipped_count
    }

    logging.info('Saving validation_generations.pkl...')
    utils.save(generations, 'validation_generations.pkl')
    logging.info('Saved %d rephrased question generations.', len(generations))

    accuracy = np.mean(accuracies)
    print(f"Overall validation split accuracy: {accuracy}")
    wandb.log({"validation_accuracy": accuracy})

    # if args.compute_p_true:
        # results_dict['uncertainty_measures'] = {
        #     'p_false':  [1 - p for p in p_trues],
        #     'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
        # }
    utils.save(results_dict, 'uncertainty_measures.pkl')

    utils.save(experiment_details, 'experiment_details.pkl')
    record_wandb_run(
        os.getenv('BIGSURE_RUN_MANIFEST'), args.dataset, args.model_name,
        args.random_seed, 'rephrased', wandb.run.dir)
    logging.info('Run complete.')
    del model


if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to rephrased questions CSV with columns: id, original_id, question.')
    parser.add_argument('--n_low_temp', type=int, default=3,
                        help='Number of low-temperature samples (temp=0.1) per question. Default: 3')
    parser.add_argument('--low_temp', type=float, default=0.1,
                        help='Temperature for low-temp samples. Default: 0.1')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (unused, kept for compatibility). Default: 256')
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help='Directory in which W&B stores run files.')

    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unknown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    logging.info('STARTING `generate_rephrased_answers_k`!')
    main(args)
    logging.info('FINISHED `generate_rephrased_answers_k`!')

    # if args.compute_uncertainties:
    #     args.assign_new_wandb_id = False
    #     gc.collect(); torch.cuda.empty_cache()
        # logging.info('STARTING `compute_uncertainty_measures`!')
        # main_compute(args)
        # logging.info('FINISHED `compute_uncertainty_measures`!')
