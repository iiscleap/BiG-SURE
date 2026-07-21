"""Generate high-temperature VQA answers for perturbed rephrased questions CSV.

Produces a `validation_generations.pkl` containing high-temperature
generations for image-perturbed samples.

Features:
- Periodic checkpointing (every 500 samples by default)
- Resume from cancelled runs using --resume_wandb_id
- Saves to same wandb run when resuming

Usage examples:
  # Fresh run
  python snne/generate_vqa_rephrased_perturbed.py --input_csv snne/vqav2/rephrased_perturbations.csv \
      --model_name Qwen-2.5-7B-Instruct --num_generations 10

  # Resume from cancelled run
  python snne/generate_vqa_rephrased_perturbed.py --input_csv snne/vqav2/rephrased_perturbations.csv \
      --model_name Qwen-2.5-7B-Instruct --num_generations 10 \
      --resume_wandb_id 0uji5elk
"""
import os
import logging
import random
import pickle
from tqdm import tqdm
import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
import pandas as pd

from snne.uncertainty.utils import utils
from snne.generate_okvqa_answers import init_vl_model, get_image_path as _get_image_path_fallback

utils.setup_logger()


def load_vqa_perturbed_from_csv(csv_path, image_dir=None):
    """Load VQA examples from a CSV and preserve all columns as metadata.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        list: List of dicts containing all row data + standardized keys.
    """
    df = pd.read_csv(csv_path)
    dataset = []
    for _, row in df.iterrows():
        # Standardize keys expected by generation loop
        qid = str(row.get('id', row.get('question_id')))
        question = str(row.get('question'))
        
        # Ensure image_path is handled (it should be in the CSV for perturbed data)
        image_path = row.get('image_path')
        if pd.isna(image_path):
            image_path = None
        elif image_dir:
            image_path = str(Path(image_dir) / Path(str(image_path)).name)
            
        example = row.to_dict()
        # Ensure critical keys exist
        example['id'] = qid
        example['question'] = question
        example['image_path'] = image_path
        example['context'] = None
        example['image_id'] = row.get('image_id', None)
        
        dataset.append(example)
    
    logging.info(f"Loaded {len(dataset)} perturbed examples from CSV {csv_path}")
    return dataset


def load_checkpoint(wandb_run_dir, checkpoint_name='checkpoint_generations.pkl'):
    """Load existing generations from a checkpoint file."""
    checkpoint_path = Path(wandb_run_dir) / 'files' / checkpoint_name
    if checkpoint_path.exists():
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return {}


def save_checkpoint(generations, checkpoint_name='checkpoint_generations.pkl'):
    """Save current generations as a checkpoint."""
    logging.info(f"Saving checkpoint with {len(generations)} examples")
    utils.save(generations, checkpoint_name)


def save_local_output(generations, output_path):
    if not output_path:
        return
    logging.info('Writing local generations to %s', output_path)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as destination:
        pickle.dump(generations, destination)


def main(args):
    utils.set_all_seeds(args.random_seed)

    # Initialize wandb run
    user = os.environ.get('USER', 'unknown')
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_base_dir = args.wandb_dir or f"{scratch_dir}/{user}/uncertainty"
    if not os.path.exists(wandb_base_dir):
        os.makedirs(wandb_base_dir, exist_ok=True)
    
    # Check if we're resuming from a previous run
    generations = {}
    if args.resume_wandb_id:
        # Resume from existing wandb run
        logging.info(f"Resuming from wandb run: {args.resume_wandb_id}")
        
        # Find the run directory
        wandb_run_dir = f"{wandb_base_dir}/wandb/run-*-{args.resume_wandb_id}"
        import glob
        run_dirs = glob.glob(wandb_run_dir)
        
        if run_dirs:
            run_dir = run_dirs[0]
            logging.info(f"Found run directory: {run_dir}")
            
            # Try to load checkpoint first, then validation_generations.pkl
            generations = load_checkpoint(run_dir, 'checkpoint_generations.pkl')
            if not generations:
                val_gen_path = Path(run_dir) / 'files' / 'validation_generations.pkl'
                if val_gen_path.exists():
                    logging.info(f"Loading from validation_generations.pkl")
                    with open(val_gen_path, 'rb') as f:
                        generations = pickle.load(f)
            
            logging.info(f"Loaded {len(generations)} existing generations")
        else:
            logging.warning(f"Could not find run directory for {args.resume_wandb_id}")
        
        # Resume the wandb run
        wandb.init(
            entity=args.entity,
            project=os.getenv('WANDB_PROJECT', 'bigsure-okvqa') if not args.debug else "bigsure-okvqa-debug",
            id=args.resume_wandb_id,
            resume="must",
            dir=wandb_base_dir,
        )
    else:
        # Fresh run
        args.run_name = utils.get_run_name("gen_vqa_perturbed", args)
        wandb.init(
            entity=args.entity,
            project=os.getenv('WANDB_PROJECT', 'bigsure-okvqa') if not args.debug else "bigsure-okvqa-debug",
            name=args.run_name,
            dir=wandb_base_dir,
            config=args,
            notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
        )
    
    logging.info('Finished wandb init for perturbed generator.')

    logging.info('Loading perturbed CSV: %s', args.input_csv)
    dataset = load_vqa_perturbed_from_csv(args.input_csv, image_dir=args.vqa_image_dir)
    logging.info('Loaded %d perturbed examples', len(dataset))

    # Build index of already completed IDs
    completed_ids = set(generations.keys())
    logging.info(f"Already completed: {len(completed_ids)} examples")

    # All indices to process
    all_indices = list(range(len(dataset)))
    
    # Filter to only incomplete examples
    indices_to_process = [idx for idx in all_indices if str(dataset[idx]['id']) not in completed_ids]
    logging.info(f"Remaining to generate: {len(indices_to_process)} examples")

    if len(indices_to_process) == 0:
        logging.info("All examples already completed!")
        # Still save final output
        logging.info('Saving final generations to wandb run files')
        utils.save(generations, 'validation_generations.pkl')
        save_local_output(generations, args.output)
        return

    # Initialize vision-language model
    model = init_vl_model(args)

    # Checkpoint interval
    checkpoint_interval = args.checkpoint_interval
    samples_since_checkpoint = 0

    # For each example, generate 1 greedy + num_generations high-temperature samples
    for idx in tqdm(indices_to_process, desc='Generating'):
        example = dataset[idx]
        qid = str(example['id'])
        question = example['question']
        image_path = example.get('image_path')
        
        # Skip if already completed (double-check)
        if qid in generations:
            continue
        
        # Fallback if image path is missing (should not happen for perturbed CSV)
        if not image_path:
            image_id = example.get('image_id')
            image_path = _get_image_path_fallback(int(image_id), args.vqa_image_dir) if image_id is not None else None

        # Store all example metadata in the output dictionary
        generations[qid] = {
            'question': question,
            'context': None,
            'image_id': example.get('image_id', None),
            'image_path': image_path,
            'perturbation_type': example.get('perturbation_type'),
            'perturbation_intensity': example.get('perturbation_intensity'),
            'perturbation_name': example.get('perturbation_name'),
            'original_id': example.get('original_id'),
            'original_question': example.get('original_question'),
        }

        responses = []
        
        num_high_temp = args.num_generations
        total_generations = 1 + num_high_temp # 1 greedy + N high-temp

        if args.reset_seed:
            torch.manual_seed(args.random_seed)

        for i in range(total_generations):
            if i == 0:
                temperature = 0.0
                min_p = 0.0
            else:
                temperature = args.temperature
                min_p = args.min_p

            local_prompt = f"Question: {question}\nAnswer:"

            try:
                pred, token_log_liks, embedding = model.predict(local_prompt, temperature, min_p=min_p, image_path=image_path)
                embedding = embedding.cpu() if hasattr(embedding, 'cpu') else embedding
            except Exception as e:
                logging.error(f"Error generating for {qid}: {e}")
                pred = ""
                token_log_liks = []
                embedding = None

            if i == 0:
                # Greedy sample
                logging.info('Question %s, greedy prediction (T=0): %s', qid, pred)
                generations[qid]['most_likely_answer'] = {
                    'response': pred,
                    'token_log_likelihoods': token_log_liks,
                    'embedding': embedding
                }
            else:
                # High-temp samples
                logging.info('Question %s, high-t prediction %d/%d (T=%.1f): %s', 
                            qid, i, num_high_temp, temperature, pred)
                # Structure: (text, log_probs, embedding, ?)
                responses.append((pred, token_log_liks, embedding, None))

        generations[qid]['responses'] = responses
        samples_since_checkpoint += 1

        # Periodic checkpoint
        if samples_since_checkpoint >= checkpoint_interval:
            logging.info(f"Checkpoint: saving {len(generations)} generations...")
            save_checkpoint(generations, 'checkpoint_generations.pkl')
            samples_since_checkpoint = 0

        if idx % 50 == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Save final generations
    logging.info('Saving final generations to wandb run files as `validation_generations.pkl`')
    utils.save(generations, 'validation_generations.pkl')
    
    # Also save a final checkpoint
    save_checkpoint(generations, 'checkpoint_generations.pkl')
    
    save_local_output(generations, args.output)

    logging.info('Done. Generated %d total examples with %d generations each.', len(generations), args.num_generations)


if __name__ == '__main__':
    parser = utils.get_parser()

    parser.add_argument('--input_csv', type=str, required=True, help='Path to perturbed rephrased CSV')
    parser.add_argument('--vqa_image_dir', type=str, default='vqav2_images/val2014', help='Fallback directory (usually unused here)')
    parser.add_argument('--output', type=str, default=None, help='Optional local output pkl path')
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help='Directory in which W&B stores run files.')
    parser.add_argument('--resume_wandb_id', type=str, default=None, 
                       help='Wandb run ID to resume from (e.g., 0uji5elk). Will load existing generations and continue.')
    parser.add_argument('--checkpoint_interval', type=int, default=500,
                       help='Save checkpoint every N samples (default: 500)')

    args, unknown = parser.parse_known_args()
    logging.info('Args: %s', args)

    if unknown:
        raise ValueError(f'Unknown args: {unknown}')

    main(args)
