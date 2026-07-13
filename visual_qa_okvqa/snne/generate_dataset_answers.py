"""Sample answers from Vision-Language Models for VQA-style datasets.

Supports datasets: vqa, okvqa, advqa, vqarad, gqa.
Per example generation includes:
- 1 greedy sample at T=0 for accuracy (most_likely_answer)
- Temperature sweep samples from low_t to high_t (inclusive) with temp_step
  and n_samples_per_temp generations at each temperature.
"""
import gc
import json
import logging
import multiprocessing as mp
import os
import random
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

from snne.compute_uncertainty_measures import main as main_compute
from snne.uncertainty.utils import utils
from snne.uncertainty.utils.metric_utils import get_metric, get_reference


DATASET_CONFIG = {
    "vqa": {
        "input_csv": "snne/vqav2/metadata.csv",
        "image_dir": "snne/vqav2/images",
        "metric": "vqa_acc",
        "answers_delimiter": ";",
    },
    "okvqa": {
        "input_csv": "snne/okvqa/metadata.csv",
        "image_dir": "snne/okvqa/images",
        "metric": "vqa_acc",
        "answers_delimiter": ";",
    },
    "advqa": {
        "input_csv": "snne/advqa/metadata.csv",
        "image_dir": "snne/advqa/images",
        "metric": "vqa_acc",
        "answers_delimiter": "|",
    },
    "vqarad": {
        "input_csv": "snne/vqarad/metadata.csv",
        "image_dir": "snne/vqarad/images",
        "metric": "vqarad_exact",
        "answers_delimiter": ";",
    },
    "gqa": {
        "input_csv": "snne/gqa/metadata.csv",
        "image_dir": "snne/gqa/images",
        "metric": "vqarad_exact",
        "answers_delimiter": ";",
    },
}


def parse_answers(row, delimiter):
    """Parse answer list from a metadata row."""
    answers_list = []
    if "answers_joined" in row and not pd.isna(row["answers_joined"]):
        answers_raw = str(row["answers_joined"])
        answers_list = [a.strip() for a in answers_raw.split(delimiter) if a.strip()]
    elif "answer" in row and not pd.isna(row["answer"]):
        answers_list = [str(row["answer"]).strip()]
    elif "multiple_choice_answer" in row and not pd.isna(row["multiple_choice_answer"]):
        answers_list = [str(row["multiple_choice_answer"]).strip()]
    return answers_list


def resolve_image_path(raw_image_path, image_id, image_dir, csv_path):
    """Resolve image path from CSV path, image id fallback, and image dir."""
    image_path = raw_image_path
    if pd.isna(image_path):
        image_path = None

    if image_path:
        image_path = str(image_path)
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        if os.path.exists(image_path):
            return image_path
        csv_relative = os.path.join(os.path.dirname(csv_path), image_path)
        if os.path.exists(csv_relative):
            return csv_relative
        if not os.path.isabs(image_path):
            image_basename = os.path.basename(image_path)
            image_dir_candidate = os.path.join(image_dir, image_basename)
            if os.path.exists(image_dir_candidate):
                return image_dir_candidate

    if image_id is not None and not pd.isna(image_id):
        try:
            image_id_int = int(image_id)
            candidates = [
                os.path.join(image_dir, f"COCO_val2014_{image_id_int:012d}.jpg"),
                os.path.join(image_dir, f"OKVQA_val2014_{image_id_int:012d}.jpg"),
                os.path.join(image_dir, f"VQAv2_validation_{image_id_int:012d}_{image_id_int}.jpg"),
                os.path.join(image_dir, f"VQARAD_train_{image_id_int:012d}_{image_id_int}.jpg"),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
        except (ValueError, TypeError):
            pass

    return image_path


def load_dataset_from_csv(csv_path, image_dir, answers_delimiter):
    """Load VQA-style examples from metadata CSV."""
    df = pd.read_csv(csv_path)
    dataset = []

    for _, row in df.iterrows():
        qid_val = row.get("question_id") if "question_id" in row else row.get("id")
        qid = str(qid_val)

        question_val = row.get("question") if "question" in row else row.get("edited_question")
        question = str(question_val)

        raw_image_path = row.get("image_path") if "image_path" in row else row.get("metadata_image_path")
        image_id = row.get("image_id", None)
        image_path = resolve_image_path(raw_image_path, image_id, image_dir, csv_path)

        answers_list = parse_answers(row, answers_delimiter)

        example = {
            "id": qid,
            "question": question,
            "image_id": image_id if not pd.isna(image_id) else None,
            "image_path": image_path,
            "context": None,
            "answers": {"text": answers_list},
        }
        dataset.append(example)

    logging.info("Loaded %d examples from CSV %s", len(dataset), csv_path)
    return dataset


def init_vl_model(args):
    """Initialize vision-language model."""
    from snne.uncertainty.models.huggingface_models import (
        GeminiModel,
        Gemma3VLModel,
        LlavaVLModel,
        Phi4VLModel,
        PixtralVLModel,
        Qwen3VLModel,
        QwenVLModel,
    )

    model_name = args.model_name
    model_name_l = model_name.lower()

    if "gemini" in model_name_l:
        model = GeminiModel(model_name=model_name, stop_sequences="default", max_new_tokens=32)
    elif "llava" in model_name_l:
        model = LlavaVLModel(model_name=model_name, stop_sequences="default", max_new_tokens=32)
    elif "qwen3" in model_name_l:
        model = Qwen3VLModel(model_name=model_name, stop_sequences="default", max_new_tokens=32)
    elif "qwen" in model_name_l:
        model = QwenVLModel(model_name=model_name, stop_sequences="default", max_new_tokens=32)
    elif "gemma-3" in model_name_l or "gemma3" in model_name_l:
        model = Gemma3VLModel(model_name=model_name, stop_sequences="default", max_new_tokens=32)
    elif "phi-4" in model_name_l or "phi4" in model_name_l:
        model = Phi4VLModel(model_name=model_name, stop_sequences="default", max_new_tokens=32)
    elif "pixtral" in model_name_l:
        model = PixtralVLModel(model_name=model_name, stop_sequences="default", max_new_tokens=32)
    else:
        raise ValueError(f"Unsupported VL model: {model_name}")

    return model


def build_temperature_schedule(low_t, high_t, temp_step):
    """Create an inclusive temperature list [low_t, ..., high_t]."""
    low_d = Decimal(str(low_t))
    high_d = Decimal(str(high_t))
    step_d = Decimal(str(temp_step))

    if step_d <= 0:
        raise ValueError("--temp_step must be > 0")
    if high_d < low_d:
        raise ValueError("--high_t must be >= --low_t")

    temps = []
    cur = low_d
    precision = Decimal("0.0001")

    while cur <= high_d + precision:
        temps.append(float(cur.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)))
        cur += step_d

    if len(temps) == 0:
        temps = [float(low_d)]

    if temps[-1] > float(high_d) + 1e-9:
        temps[-1] = float(high_d)

    if abs(temps[-1] - float(high_d)) > 1e-9:
        temps.append(float(high_d))

    deduped = []
    for t in temps:
        if len(deduped) == 0 or abs(deduped[-1] - t) > 1e-9:
            deduped.append(t)
    return deduped


def main(args):
    utils.setup_logger()
    utils.set_all_seeds(args.random_seed)

    dataset_key = args.dataset_name.lower()
    if dataset_key not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset '{args.dataset_name}'. Choices: {sorted(DATASET_CONFIG.keys())}")

    cfg = DATASET_CONFIG[dataset_key]

    if args.input_csv is None:
        args.input_csv = cfg["input_csv"]
    if args.vqa_image_dir is None:
        args.vqa_image_dir = cfg["image_dir"]
    if args.metric is None:
        args.metric = cfg["metric"]

    temperatures = build_temperature_schedule(args.low_t, args.high_t, args.temp_step)

    experiment_details = {
        "args": args,
        "temperature_schedule": temperatures,
        "samples_per_temperature": args.n_samples_per_temp,
    }

    user = os.environ["USER"]
    slurm_jobid = os.getenv("SLURM_JOB_ID", None)
    scratch_dir = os.getenv("SCRATCH_DIR", ".")
    run_dir = f"{scratch_dir}/{user}/uncertainty"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    args.run_name = utils.get_run_name("generate_dataset_answers", args)

    wandb.init(
        entity=args.entity,
        project="snne" if not args.debug else "snne_debug",
        name=args.run_name,
        dir=run_dir,
        config=args,
        notes=f"slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}",
        tags=[f"dataset={dataset_key}"],
    )
    logging.info("Finished wandb init.")

    metric = get_metric(args.metric)

    validation_dataset = load_dataset_from_csv(
        csv_path=args.input_csv,
        image_dir=args.vqa_image_dir,
        answers_delimiter=cfg["answers_delimiter"],
    )

    model = init_vl_model(args)

    accuracies, generations, results_dict = [], {}, {}

    possible_indices = range(0, len(validation_dataset))
    indices = random.sample(possible_indices, min(args.num_samples, len(validation_dataset)))
    experiment_details["validation"] = {"indices": indices}

    if args.num_samples > len(validation_dataset):
        logging.warning("Not enough samples in dataset. Using all %d samples.", len(validation_dataset))

    logging.info("=" * 80)
    logging.info("Generating answers for dataset=%s", dataset_key)
    logging.info("Temperature schedule: %s", temperatures)
    logging.info("Samples per temperature: %d", args.n_samples_per_temp)
    logging.info("=" * 80)

    it = 0
    for index in tqdm(indices):
        if (it + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        it += 1

        example = validation_dataset[index]
        question = example["question"]
        image_path = example.get("image_path", None)
        correct_answer = example["answers"]["text"]

        local_prompt = f"Question: {question}\nAnswer:"

        generations[example["id"]] = {
            "question": question,
            "context": None,
            "image_id": example.get("image_id", None),
            "image_path": image_path,
        }

        if args.reset_seed:
            torch.manual_seed(args.random_seed)

        greedy_pred, greedy_log_liks, greedy_embedding = model.predict(
            local_prompt, 0.0, min_p=0.0, image_path=image_path
        )
        greedy_embedding = (
            greedy_embedding.cpu() if hasattr(greedy_embedding, "cpu") and greedy_embedding is not None else greedy_embedding
        )

        if correct_answer:
            acc = metric(greedy_pred, example)
        else:
            acc = 0.0

        accuracies.append(acc)

        generations[example["id"]].update(
            {
                "most_likely_answer": {
                    "response": greedy_pred,
                    "token_log_likelihoods": greedy_log_liks,
                    "embedding": greedy_embedding,
                    "accuracy": acc,
                },
                "reference": get_reference(example),
            }
        )

        logging.info("Iteration %d: %s", it, "#" * 80)
        logging.info("question: %s", question)
        logging.info("greedy prediction (T=0): %s", greedy_pred)
        logging.info("correct answer: %s", str(correct_answer))
        logging.info("accuracy: %.4f", acc)
        logging.info("image path: %s", str(image_path))

        temperature_responses = {}

        for temp in temperatures:
            sample_bucket = []
            for sample_idx in range(args.n_samples_per_temp):
                if args.reset_seed:
                    torch.manual_seed(args.random_seed + sample_idx)

                pred, token_log_likelihoods, embedding = model.predict(
                    local_prompt,
                    temp,
                    min_p=args.min_p,
                    image_path=image_path,
                )
                embedding = embedding.cpu() if hasattr(embedding, "cpu") and embedding is not None else embedding
                sample_tuple = (pred, token_log_likelihoods, embedding, None)
                sample_bucket.append(sample_tuple)

                logging.info(
                    "T=%.4f sample %d/%d: %s",
                    temp,
                    sample_idx + 1,
                    args.n_samples_per_temp,
                    pred,
                )

            temperature_responses[f"{temp:.4f}"] = sample_bucket

        low_key = f"{temperatures[0]:.4f}"
        high_key = f"{temperatures[-1]:.4f}"

        low_temp_responses = temperature_responses.get(low_key, [])
        high_temp_responses = temperature_responses.get(high_key, [])

        all_sweep_responses = []
        for _, bucket in temperature_responses.items():
            all_sweep_responses.extend(bucket)

        generations[example["id"]]["low_temp_responses"] = low_temp_responses
        generations[example["id"]]["high_temp_responses"] = high_temp_responses
        generations[example["id"]]["responses"] = all_sweep_responses
        generations[example["id"]]["temperature_responses"] = temperature_responses

    utils.save(generations, "validation_generations.pkl")

    accuracy = float(np.mean(accuracies)) if len(accuracies) > 0 else 0.0
    print(f"Overall validation split accuracy: {accuracy}")
    wandb.log({"validation_accuracy": accuracy})

    utils.save(results_dict, "uncertainty_measures.pkl")
    utils.save(experiment_details, "experiment_details.pkl")
    logging.info("Run complete.")

    del model


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = utils.get_parser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=sorted(DATASET_CONFIG.keys()),
        help="Dataset key: vqa, okvqa, advqa, vqarad, gqa",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Path to metadata CSV; defaults per dataset_name",
    )
    parser.add_argument(
        "--vqa_image_dir",
        type=str,
        default=None,
        help="Directory containing images; defaults per dataset_name",
    )
    parser.add_argument(
        "--low_t",
        type=float,
        default=0.1,
        help="Lowest temperature in sweep (inclusive)",
    )
    parser.add_argument(
        "--high_t",
        type=float,
        default=1.0,
        help="Highest temperature in sweep (inclusive)",
    )
    parser.add_argument(
        "--temp_step",
        type=float,
        default=0.1,
        help="Temperature step size",
    )
    parser.add_argument(
        "--n_samples_per_temp",
        type=int,
        default=10,
        help="Number of samples to generate per temperature",
    )

    args, unknown = parser.parse_known_args()
    logging.info("Starting run with args: %s", args)

    if unknown:
        raise ValueError(f"Unknown args: {unknown}")

    if args.n_samples_per_temp <= 0:
        raise ValueError("--n_samples_per_temp must be > 0")

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    logging.info("STARTING `generate_dataset_answers`!")
    main(args)
    logging.info("FINISHED `generate_dataset_answers`!")

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * "#X")
        # logging.info("STARTING `compute_uncertainty_measures`!")
        # main_compute(args)
        # logging.info("FINISHED `compute_uncertainty_measures`!")
