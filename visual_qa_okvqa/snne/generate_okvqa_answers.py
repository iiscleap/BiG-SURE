"""Sample answers from Vision-Language Models on VQAv2 task."""
import multiprocessing as mp
import os

# # DO THIS FIRST - before any other imports
if __name__ == '__main__':
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
# import os
# # Force the threading layer BEFORE any other imports
# os.environ["MKL_THREADING_LAYER"] = "GNU"
# os.environ["VLLM_USE_V1"] = "0"
import numpy as np

import gc
import logging
import random
import json
import pickle
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import wandb

from snne.uncertainty.utils import utils
from snne.uncertainty.utils.metric_utils import get_metric, get_reference

def load_vqa_data(questions_json, answers_json=None):
    """Load VQA questions and optionally answers.
    
    Args:
        questions_json: Path to VQA questions JSON file
        answers_json: Optional path to VQA answers JSON file
        
    Returns:
        list: List of dicts with keys: id, question, image_id, answers (if available)
    """
    with open(questions_json, 'r') as f:
        questions_data = json.load(f)
    
    questions = questions_data['questions']
    
    # Convert to dict keyed by question_id
    data_dict = {}
    for q in questions:
        data_dict[q['question_id']] = {
            'id': str(q['question_id']),
            'question': q['question'],
            'image_id': q['image_id'],
            'context': None,  # VQA has no text context
            'answers': {'text': []},  # Will be filled if answers available
        }
    
    # Load answers if available
    if answers_json and os.path.exists(answers_json):
        with open(answers_json, 'r') as f:
            answers_data = json.load(f)
        
        for ans_entry in answers_data['annotations']:
            qid = ans_entry['question_id']
            if qid in data_dict:
                # Get all answer texts
                answer_texts = [a['answer'] for a in ans_entry['answers']]
                data_dict[qid]['answers']['text'] = answer_texts
    
    # Convert to list
    dataset = list(data_dict.values())
    logging.info(f"Loaded {len(dataset)} questions from {questions_json}")
    
    return dataset


def load_vqa_from_csv(csv_path, image_dir=None):
    """Load VQA examples from a CSV with columns: question_id, question, image_path, answers_joined.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        list: List of dicts with keys: id, question, image_path, context, answers
    """
    df = pd.read_csv(csv_path)
    dataset = []
    for _, row in df.iterrows():
        qid = str(row.get('question_id') if 'question_id' in row else row.get('id'))
        question = str(row.get('question') if 'question' in row else row.get('edited_question'))
        image_path = row.get('image_path') if 'image_path' in row else row.get('metadata_image_path')
        if pd.isna(image_path):
            image_path = None
        elif image_dir:
            image_path = str(Path(image_dir) / Path(str(image_path)).name)
        
        # Parse answers from answers_joined column (semicolon-separated)
        answers_list = []
        if 'answers_joined' in row and not pd.isna(row['answers_joined']):
            answers_str = str(row['answers_joined'])
            # Split by semicolon and strip whitespace
            answers_list = [ans.strip() for ans in answers_str.split(';') if ans.strip()]
        elif 'answer' in row and not pd.isna(row['answer']):
            # Single answer column (e.g. MathVista)
            answers_list = [str(row['answer']).strip()]
        
        example = {
            'id': qid,
            'question': question,
            'image_id': row.get('image_id', None),
            'image_path': image_path,
            'context': None,
            'answers': {'text': answers_list},
        }
        dataset.append(example)
    logging.info(f"Loaded {len(dataset)} examples from CSV {csv_path}")
    return dataset


def get_image_path(image_id, image_dir='data/vqav2/val2014'):
    """Get path to image file given image_id.
    
    Args:
        image_id: COCO image ID
        image_dir: Directory containing images
        
    Returns:
        str: Path to image file
    """
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    return os.path.join(image_dir, filename)


def init_vl_model(args):
    """Initialize vision-language model.
    
    Args:
        args: Arguments containing model_name
        
    Returns:
        Model instance (LlavaVLModel, QwenVLModel, Gemma3VLModel, Phi4VLModel, or PixtralVLModel)
    """
    from snne.uncertainty.models.huggingface_models import LlavaVLModel, QwenVLModel, Qwen3VLModel, Gemma3VLModel, Phi4VLModel, PixtralVLModel, GeminiModel
    
    model_name = args.model_name
    
    if 'gemini' in model_name.lower():
        model = GeminiModel(
            model_name=model_name,
            stop_sequences='default',
            max_new_tokens=32,
        )
    elif 'llava' in model_name.lower():
        model = LlavaVLModel(
            model_name=model_name,
            stop_sequences='default',
            max_new_tokens=32,
            # token_limit=args.token_limit
        )
    elif 'qwen3' in model_name.lower():
        model = Qwen3VLModel(
            model_name=model_name,
            stop_sequences='default',
            max_new_tokens=32,
            # token_limit=args.token_limit
        )
    elif 'qwen' in model_name.lower():
        model = QwenVLModel(
            model_name=model_name,
            stop_sequences='default',
            max_new_tokens=32,
            # token_limit=args.token_limit
        )
    elif 'gemma-3' in model_name.lower() or 'gemma3' in model_name.lower():
        model = Gemma3VLModel(
            model_name=model_name,
            stop_sequences='default',
            max_new_tokens=32,
            # token_limit=args.token_limit
        )
    elif 'phi-4' in model_name.lower() or 'phi4' in model_name.lower():
        model = Phi4VLModel(
            model_name=model_name,
            stop_sequences='default',
            max_new_tokens=32,
            # token_limit=args.token_limit
        )
    elif 'pixtral' in model_name.lower():
        model = PixtralVLModel(
            model_name=model_name,
            stop_sequences='default',
            max_new_tokens=32,
            # token_limit=args.token_limit
        )
    else:
        raise ValueError(f'Unsupported VL model: {model_name}')
    
    return model


def main(args):
    utils.setup_logger()
    # Setup run
    experiment_details = {'args': args}
    utils.set_all_seeds(args.random_seed)
    user = os.environ.get('USER', 'user')
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")
    args.run_name = utils.get_run_name("generate_vqa_answers", args)

    wandb_dir = args.wandb_dir or f"{scratch_dir}/{user}/uncertainty"
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(
        entity=args.entity,
        project=os.getenv('WANDB_PROJECT', 'bigsure-okvqa') if not args.debug else "bigsure-okvqa-debug",
        name=args.run_name,
        dir=wandb_dir,
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')

    # Get accuracy metric
    metric = get_metric(args.metric)

    # Load VQA dataset (from CSV if provided, otherwise from JSON)
    image_dir = getattr(args, 'vqa_image_dir', 'vqav2_images/val2014')
    
    if getattr(args, 'input_csv', None):
        logging.info(f"Loading VQA data from CSV {args.input_csv}")
        validation_dataset = load_vqa_from_csv(args.input_csv, image_dir=image_dir)
    else:
        questions_json = args.vqa_questions_json
        answers_json = getattr(args, 'vqa_answers_json', None)
        logging.info(f"Loading VQA data from JSON {questions_json}")
        validation_dataset = load_vqa_data(questions_json, answers_json)

    # Initialize VL model
    model = init_vl_model(args)

    # Start answer generation
    logging.info(80 * '=')
    logging.info('Generating VQA answers: ')
    logging.info(80 * '=')
    
    # This will store all input data and model predictions
    accuracies, generations = [], {}

    # Sample indices
    possible_indices = range(0, len(validation_dataset))
    indices = random.sample(possible_indices, min(args.num_samples, len(validation_dataset)))
    experiment_details['validation'] = {'indices': indices}

    if args.num_samples > len(validation_dataset):
        logging.warning('Not enough samples in dataset. Using all %d samples.', len(validation_dataset))

    it = 0
    for index in tqdm(indices):
        if (it + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        it += 1

        # Grab example at index
        example = validation_dataset[index]
        question = example["question"]
        
        # Prefer image_path if provided by CSV loader, otherwise construct from image_id
        image_path = example.get('image_path', None)
        if not image_path:
            image_id = example['image_id']
            image_path = get_image_path(image_id, image_dir)
        
        generations[example['id']] = {
            'question': question,
            'context': None,  # No text context for VQA
            'image_id': example.get('image_id', None),
            'image_path': image_path
        }
        
        correct_answer = example['answers']['text']

        # For VQA, just pass the question - VL models have built-in few-shot prompts
        local_prompt = f"Question: {question}\nAnswer:"

        logging.info('Question: '.ljust(15) + question)
        logging.info('Image path: '.ljust(15) + image_path)

        full_responses = []
        low_temp_responses = []

        # Generation structure:
        # - 1 greedy sample (temp=0) for accuracy calculation
        # - num_low_temp_generations low-temp samples (temp=low_temp, default 0.1)
        # - num_generations high-temp samples (temp=temperature, default 1.0)
        num_low_temp = getattr(args, 'num_low_temp_generations', 3)
        num_high_temp = args.num_generations  # default 10
        total_generations = 1 + num_low_temp + num_high_temp
        
        # Reset seed if needed
        if args.reset_seed:
            torch.manual_seed(args.random_seed)

        for i in range(total_generations):
            # Determine temperature based on generation index:
            # i=0: greedy (temp=0)
            # i=1 to num_low_temp: low-temp (temp=low_temp, e.g. 0.1)
            # i>num_low_temp: high-temp (temp=temperature, e.g. 1.0)
            if i == 0:
                temperature = 0.0  # Greedy
                min_p = 0.0
            elif i <= num_low_temp:
                temperature = getattr(args, 'low_temp', 0.1)  # Low-temp
                min_p = 0.0
            else:
                temperature = args.temperature  # High-temp (default 1.0)
                min_p = args.min_p

            predicted_answer, token_log_likelihoods, embedding = model.predict(
                local_prompt, temperature, min_p=min_p, image_path=image_path)
            embedding = embedding.cpu() if embedding is not None else None

            if i == 0:
                # Greedy sample - compute accuracy
                if correct_answer:
                    acc = metric(predicted_answer, example)
                else:
                    acc = 0.0
                logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                logging.info('question: '.ljust(15) + question)
                logging.info('greedy prediction (T=0): '.ljust(15) + predicted_answer)
                logging.info('correct answer: '.ljust(15) + str(correct_answer))
                logging.info('accuracy: '.ljust(15) + str(acc))

                accuracies.append(acc)
                most_likely_answer_dict = {
                    'response': predicted_answer,
                    'token_log_likelihoods': token_log_likelihoods,
                    'embedding': embedding,
                    'accuracy': acc}
                generations[example['id']].update({
                    'most_likely_answer': most_likely_answer_dict,
                    'reference': get_reference(example)})
            elif i <= num_low_temp:
                # Low-temp samples (temp=0.1)
                logging.info('low-t prediction (T=%.1f) %d/%d: %s', temperature, i, num_low_temp, predicted_answer)
                low_temp_responses.append(
                    (predicted_answer, token_log_likelihoods, embedding, None))
            else:
                # High-temp samples (temp=1.0)
                logging.info('high-t prediction (T=%.1f) %d/%d: %s', temperature, i - num_low_temp, num_high_temp, predicted_answer)
                full_responses.append(
                    (predicted_answer, token_log_likelihoods, embedding, None))

        # Append all predictions for this example to `generations`
        generations[example['id']]['low_temp_responses'] = low_temp_responses
        generations[example['id']]['responses'] = full_responses

    # Save generations
    utils.save(generations, 'validation_generations.pkl')

    # Log overall accuracy
    accuracy = np.mean(accuracies)
    print(f"Overall validation split accuracy: {accuracy}")
    wandb.log({"validation_accuracy": accuracy})

    results_dict = {
        'schema_version': 1,
        'question_ids': [str(qid) for qid in generations],
    }
    utils.save(results_dict, 'uncertainty_measures.pkl')
    utils.save(experiment_details, 'experiment_details.pkl')
    if args.output:
        local_dir = os.path.dirname(args.output) or '.'
        os.makedirs(local_dir, exist_ok=True)
        with open(args.output, 'wb') as destination:
            pickle.dump(generations, destination)
        with open(os.path.join(local_dir, 'uncertainty_measures.pkl'), 'wb') as destination:
            pickle.dump(results_dict, destination)
        with open(os.path.join(local_dir, 'experiment_details.pkl'), 'wb') as destination:
            pickle.dump(experiment_details, destination)
        logging.info('Saved local generations to %s', args.output)
    logging.info('Run complete.')
    del model


if __name__ == '__main__':
    parser = utils.get_parser()
    
    # Add VQA-specific arguments
    parser.add_argument(
        "--input_csv", type=str, default=None,
        help="Path to CSV file with VQA questions (id, edited_question, metadata_image_path)")
    parser.add_argument(
        "--vqa_questions_json", type=str, 
        default="vqav2_questions/v2_OpenEnded_mscoco_val2014_questions.json",
        help="Path to VQA questions JSON file (used if --input_csv not provided)")
    parser.add_argument(
        "--vqa_answers_json", type=str, default=None,
        help="Path to VQA answers JSON file (optional)")
    parser.add_argument(
        "--vqa_image_dir", type=str, default="vqav2_images/val2014",
        help="Directory containing VQA images")
    parser.add_argument('--low_temp', type=float, default=0.1,
                        help='Temperature for low-temp generations (default 0.1)')
    parser.add_argument('--num_low_temp_generations', type=int, default=3,
                        help='Number of low-temp (T=low_temp) samples to generate (default 3)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional local output pkl used by downstream stages.')
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help='Directory in which W&B stores run files.')
    
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unknown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # First sample generations from VL model
    logging.info('STARTING `generate_vqa_answers`!')
    main(args)
    logging.info('FINISHED `generate_vqa_answers`!')

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        # logging.info(50 * '#X')
        # logging.info('STARTING `compute_uncertainty_measures`!')
        # main_compute(args)
        # logging.info('FINISHED `compute_uncertainty_measures`!')
