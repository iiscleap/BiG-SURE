r"""
Author: XUE Boyang      Filename: inference.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Generate answers on QA val set in few-shot.
"""
import os
# Disable torch dynamo/compile to avoid RecompileLimitExceeded with variable input lengths
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import time
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import operator
from functools import reduce
from pathlib import Path

from tqdm import tqdm
import json
import random
from ipdb import set_trace

import numpy as np

import torch
import transformers
import wandb
from transformers import set_seed, GenerationConfig, AutoModelForCausalLM, AutoTokenizer, Mistral3ForConditionalGeneration, FineGrainedFP8Config
from transformers.tokenization_mistral_common import (
    MistralCommonTokenizer as MistralCommonBackend
)

from utils import *


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

MULTILINGUAL_ROOT = Path(__file__).resolve().parents[1]
BIGSURE_ROOT = MULTILINGUAL_ROOT.parent
DATA_ROOT = BIGSURE_ROOT / "data" / "multilingual_sciq"
PROMPT_ROOT = MULTILINGUAL_ROOT / "prompt"
OUTPUT_ROOT = MULTILINGUAL_ROOT / "outputs"

language_list = ["en", "zh", "ja", "fr"]

# Model path dictionary mapping model names to HuggingFace model paths
model_path_dict = {
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "gpt2": "gpt2",
    "aya": "CohereLabs/aya-expanse-8b",
    "gemma3": "google/gemma-3-12b-it",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "apertus": "swiss-ai/Apertus-8B-Instruct-2509",
    "ministral": "mistralai/Ministral-3-8B-Instruct-2512",
}

@dataclass
class ModelArguments:
    model_name: str = field(default="gpt-3.5-turbo", metadata={"help": "Model name.", "choices": ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "llama3", "vicuna", "gpt2", "aya", "gemma3", "qwen", "apertus", "ministral"]})
    model_max_length: int = field(default=4096, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})


@dataclass
class DataArguments:
    data_dir: str = field(default=str(DATA_ROOT / "{}"), metadata={"help": "Directory containing mling_<dataset>.json."})
    rephrased_data_dir: str = field(default=str(DATA_ROOT / "sciq"), metadata={"help": "Directory containing <dataset>_rephrased_300.json."})
    dataset: str = field(default="triviaqa", metadata={"help": "Dataset name.", "choices": ["triviaqa", "gsm8k", "common", "sciq"]})
    data_suffix: str = field(default="2k_1s", metadata={"help": "Data file suffix."})
    prompt_dir: str = field(default=str(PROMPT_ROOT), metadata={"help": "Path to prompt templates."})
    continue_generate: bool = field(default=True, metadata={"help": "Continue from the previous generations."})
    filter_ids_from: str = field(default=None, metadata={"help": "Path to rephrased JSON to filter to only original 300 question IDs."})


@dataclass
class InferenceArguments:
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling or not."})
    output_dir: str = field(default=str(OUTPUT_ROOT / "{}" / "inference"), metadata={"help": "Directory to save results."})
    suffix: str = field(default="infer", metadata={"help": "File name to save the results."})
    num_sampling: int = field(default=5, metadata={"help": "Number of samples."})
    temperature: float = field(default=0.8, metadata={"help": "Temperature for sampling."})
    top_p: float = field(default=1.0, metadata={"help": "Top p for sampling."})
    top_k: int = field(default=40, metadata={"help": "Top k for sampling."})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for sampling."})
    max_length: int = field(default=16, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    repetition_penalty: float = field(default=1.1, metadata={"help": "Repetition penalty."})
    num_examples: int = field(default=8, metadata={"help": "Number of examples to generate."})
    inference_mode: str = field(default="vanilla", metadata={"help": "Inference mode: 'vanilla', 'sampling', or 'rephrased_sampling'.", "choices": ["vanilla", "sampling", "rephrased_sampling"]})
    k: int = field(default=10, metadata={"help": "Number of high-temp samples to generate in sampling/rephrased_sampling mode."})
    wandb_project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "bigsure-multilingual-sciq"), metadata={"help": "W&B project name."})
    wandb_entity: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_ENTITY"), metadata={"help": "Optional W&B entity."})
    wandb_mode: str = field(default_factory=lambda: os.getenv("WANDB_MODE", "online"), metadata={"help": "W&B mode: online, offline, or disabled."})
    wandb_dir: str = field(default=str(OUTPUT_ROOT / "wandb"), metadata={"help": "Local W&B storage directory."})


@dataclass
class DeviceArguments:
    device: str = field(default="cuda", metadata={"help": "Device to use."})
    seed: int = field(default=3407, metadata={"help": "Random seed."})
    gpu_num: int = field(default=1, metadata={"help": "Number of GPUs."})
    local_rank: int = field(default=0, metadata={"help": "Local rank."})
    global_rank: int = field(default=0, metadata={"help": "Global rank."})
    world_size: int = field(default=0, metadata={"help": "World size."})


# Parse arguments.
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, InferenceArguments, DeviceArguments))
model_args, data_args, infer_args, device_args = parser.parse_args_into_dataclasses()


# Resize tokenizer and embedding.
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# Format the few-shot examplar of list to string.
def format_examplar(few_shot_examples, examplar_split):
    few_shot_examplar_list = {}
    # import pdb; pdb.set_trace()
    for language in language_list:
        few_shot_examplar_list[language] = []

    for few_shot_example in few_shot_examples:
        for language in language_list:
            # if data_args.dataset in ["triviaqa", "common", "sciq"]:
            few_shot_examplar_list[language].append("*** {} ***: {}\n*** {} ***: {}".format(examplar_split[language][0], few_shot_example["question"][language], 
                                                        examplar_split[language][1], few_shot_example["answer"][language]))
            # elif data_args.dataset == "gsm8k":
            #     few_shot_examplar_list[language].append("{}: {}\n{}: {}".format(examplar_split[language][0], few_shot_example["question"][language], 
            #                                                 examplar_split[language][1], few_shot_example["answer"][language]))
    for language in language_list:
        few_shot_examplar_list[language] = "\n\n".join(few_shot_examplar_list[language])

    # import pdb; pdb.set_trace()
    return few_shot_examplar_list


def load_original_ids_from_rephrased(rephrased_json_path: str) -> set:
    """
    Load the original question IDs from a rephrased JSON file.
    Original questions are those without an 'original_id' field.
    
    Args:
        rephrased_json_path: Path to the rephrased JSON file
        
    Returns:
        Set of original question IDs (strings)
    """
    with open(rephrased_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Original questions don't have 'original_id' field
    original_ids = {str(item['question_id']) for item in data if 'original_id' not in item}
    logging.info(f"Loaded {len(original_ids)} original question IDs from {rephrased_json_path}")
    return original_ids


# Split the generation to get the answer part.
def output_split(output, tokenizer, split_len, prompt_split):
    logits = output.scores
    probs = [torch.softmax(log, dim=-1) for log in logits]

    generated_ids = output.sequences
    if data_args.dataset == "gsm8k":
        response = tokenizer.decode(generated_ids[0][split_len:], 
                                skip_special_tokens=True).split(prompt_split)[0].lstrip()
    else:
        response = tokenizer.decode(generated_ids[0][split_len:], 
                                skip_special_tokens=True).split(prompt_split)[0].replace("\n", "").lstrip()
    token_ids, token_probs = [], []
    # print(response)
    for i, token_id in enumerate(generated_ids[0][split_len:]):
        token_prob = probs[i][0, token_id].item()
        token = tokenizer.decode(token_id)
        # print(f"Token ID: {token_id}, Probability: {token_prob}, Token: {token}")
        if token == prompt_split:
            break
        token_ids.append(token_id)
        token_probs.append(token_prob)
        
    # import pdb; pdb.set_trace()
    norm_prob = float(np.array(pow(reduce(operator.mul, token_probs), 1/len(token_probs))))
    return response, norm_prob


def dataset_loader():
    data_path = os.path.join(data_args.data_dir.format(data_args.dataset),
                             "mling_{}.json".format(data_args.dataset))
    logging.info(f"Loading data from {data_path} ...")
    dataset = json.load(open(data_path))

    # Load prompt and select the prompt type.
    prompt_template = json.load(open(os.path.join(data_args.prompt_dir, "infer_temp.json")))
    instruction = prompt_template["instruction"]
    prompt_split = prompt_template["output_split"]
    few_shot_split = prompt_template["few_shot_split"]
    prompt_input = prompt_template["standard_prompt"]

    few_shot_examplar_list = format_examplar(dataset[:infer_args.num_examples], few_shot_split)
    dataset = dataset[infer_args.num_examples:]
    
    # Filter to only original 300 IDs if filter_ids_from is specified
    valid_ids = None
    if data_args.filter_ids_from and os.path.exists(data_args.filter_ids_from):
        valid_ids = load_original_ids_from_rephrased(data_args.filter_ids_from)
        logging.info(f"Filtering dataset to {len(valid_ids)} original question IDs")

    samples = []
    for data in dataset:
        # Skip if filtering is enabled and this ID is not in valid_ids
        if valid_ids is not None and str(data["question_id"]) not in valid_ids:
            continue
            
        sample = {
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": data["answer"],
            "input": {}
        }
        for language in language_list:
            sample["input"][language] = prompt_input[language]. \
                format(instruction=instruction[language], 
                        examples=few_shot_examplar_list[language], 
                        question=sample["question"][language])

        samples.append(sample)
    
    if valid_ids is not None:
        logging.info(f"After filtering: {len(samples)} samples")

    # import pdb; pdb.set_trace()
    return samples, prompt_split


def rephrased_dataset_loader():
    """Load rephrased dataset from our_data folder for rephrased_sampling mode."""
    data_path = os.path.join(data_args.rephrased_data_dir, 
                              f"{data_args.dataset}_rephrased_300.json")
    logging.info(f"Loading rephrased data from {data_path} ...")
    dataset = json.load(open(data_path))

    # Load prompt and select the prompt type.
    prompt_template = json.load(open(os.path.join(data_args.prompt_dir, "infer_temp.json")))
    instruction = prompt_template["instruction"]
    prompt_split = prompt_template["output_split"]
    few_shot_split = prompt_template["few_shot_split"]
    prompt_input = prompt_template["standard_prompt"]

    # For rephrased data, we need few-shot examples from the original dataset
    original_data_path = os.path.join(data_args.data_dir.format(data_args.dataset),
                                      "mling_{}.json".format(data_args.dataset))
    original_dataset = json.load(open(original_data_path))
    few_shot_examplar_list = format_examplar(original_dataset[:infer_args.num_examples], few_shot_split)

    samples = []
    for data in dataset:
        # Only include rephrased samples (those with original_id field)
        # Skip original questions that don't have the original_id field
        if "original_id" not in data:
            continue
            
        sample = {
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": data["answer"],
            "input": {}
        }
        # Preserve original_id and rephrase_idx
        sample["original_id"] = data["original_id"]
        if "rephrase_idx" in data:
            sample["rephrase_idx"] = data["rephrase_idx"]
            
        for language in language_list:
            sample["input"][language] = prompt_input[language]. \
                format(instruction=instruction[language], 
                        examples=few_shot_examplar_list[language], 
                        question=sample["question"][language])

        samples.append(sample)

    logging.info(f"Loaded {len(samples)} rephrased samples (excluding original questions)")
    return samples, prompt_split


def llama_generate(dataset, prompt_split):
    # import pdb; pdb.set_trace()
    # Info: Device settings: random seed, using cuda or not, distributed setting.
    set_seed(device_args.seed)

    device_args.num_gpu = torch.cuda.device_count()
    device_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name_or_path = model_path_dict[model_args.model_name]
    logging.info(f"Loading model and tokenizer from {model_name_or_path} ...")
    # Use bfloat16 and eager attention for Gemma3 to avoid numerical issues
    if model_args.model_name == "gemma3":
        # Disable torch.compile/dynamo to avoid RecompileLimitExceeded error
        # This happens because variable input lengths cause too many recompilations
        # Note: Use torch._dynamo directly from the global torch import to avoid
        # creating a local 'torch' variable that shadows the global import
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()
        
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            attn_implementation="eager"
        )
    elif model_args.model_name == "apertus":
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            dtype=torch.bfloat16,            
            device_map="balanced",
            attn_implementation="eager",      
        ).eval()
    elif model_args.model_name == "ministral":
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            device_map="auto",
            quantization_config=FineGrainedFP8Config(dequantize=True)
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        
    if model_args.model_name == "ministral":
        tokenizer = MistralCommonBackend.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            model_max_length=model_args.model_max_length,
            use_fast=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            model_max_length=model_args.model_max_length,
            padding_side="right",
            use_fast=True,
    )

    # Resize tokenizer and embedding (skip for Gemma3 as it has proper token config)
    if model_args.model_name != "gemma3":
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token == "":
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token == "":
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token == "":
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

    # Sample the data.
    logging.info("Start generating ...")
    print(f"Max length: {infer_args.max_length}")
    with tqdm(total=data_len) as t:
        for idx, batch in enumerate(dataset):            
            # import pdb; pdb.set_trace()
            # time.sleep(1)
            generations, norm_probs = {}, {}
            for lang in language_list:
                input_ids = tokenizer(batch["input"][lang], return_tensors="pt")["input_ids"].to(device_args.device)
                with torch.no_grad():
                    # Configure generation based on inference mode
                    if infer_args.inference_mode == "sampling":
                        # Sampling mode: temp=1, top_p=0.95, do_sample=True
                        # Generate k samples
                        generation_config = GenerationConfig(
                                                do_sample=True,
                                                temperature=1.0,
                                                top_p=0.95,
                                                top_k=50,
                                                num_beams=1,
                                                repetition_penalty=infer_args.repetition_penalty)
                        
                        lang_generations = []
                        lang_probs = []
                        for sample_idx in range(infer_args.k):
                            # For Gemma3 and Apertus, don't override token IDs as they have specific defaults
                            if model_args.model_name in ["gemma3", "apertus"]:
                                generation = model.generate(input_ids,
                                                        generation_config=generation_config,
                                                        return_dict_in_generate=True,
                                                        output_scores=True,
                                                        max_new_tokens=infer_args.max_length)
                            else:
                                generation = model.generate(input_ids,
                                                        generation_config=generation_config,
                                                        return_dict_in_generate=True,
                                                        output_scores=True,
                                                        max_new_tokens=infer_args.max_length,
                                                        pad_token_id=tokenizer.pad_token_id,
                                                        eos_token_id=tokenizer.eos_token_id, 
                                                        bos_token_id=tokenizer.bos_token_id)
                            gen_text, norm_prob = output_split(generation, tokenizer, len(input_ids[0]), prompt_split)
                            lang_generations.append(gen_text)
                            lang_probs.append(norm_prob)
                        
                        generations[lang] = lang_generations
                        norm_probs[lang] = lang_probs
                    elif infer_args.inference_mode == "rephrased_sampling":
                        # Rephrased sampling mode: 1 greedy (t=0) + k high-temp (t=1) samples
                        # This is for rephrased dataset experiments
                        lang_generations = []
                        lang_probs = []
                        
                        # First, generate 1 greedy sample (temperature=0)
                        if model_args.model_name == "gemma3":
                            greedy_config = GenerationConfig(
                                                    do_sample=False,
                                                    num_beams=infer_args.num_beams,
                                                    repetition_penalty=infer_args.repetition_penalty,
                                                    temperature=None,
                                                    top_p=None,
                                                    top_k=None)
                        else:
                            greedy_config = GenerationConfig(
                                                    do_sample=False,
                                                    num_beams=infer_args.num_beams,
                                                    repetition_penalty=infer_args.repetition_penalty)

                        if model_args.model_name in ["gemma3", "apertus"]:
                            generation = model.generate(input_ids,
                                                    generation_config=greedy_config,
                                                    return_dict_in_generate=True,
                                                    output_scores=True,
                                                    max_new_tokens=infer_args.max_length,
                                                    do_sample=False)
                        else:
                            generation = model.generate(input_ids,
                                                    generation_config=greedy_config,
                                                    return_dict_in_generate=True,
                                                    output_scores=True,
                                                    max_new_tokens=infer_args.max_length,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    eos_token_id=tokenizer.eos_token_id, 
                                                    bos_token_id=tokenizer.bos_token_id)
                        
                        gen_text, norm_prob = output_split(generation, tokenizer, len(input_ids[0]), prompt_split)
                        lang_generations.append(gen_text)
                        lang_probs.append(norm_prob)
                        
                        # Now generate k high-temperature samples (temperature=1.0)
                        high_t_generation_config = GenerationConfig(
                                                do_sample=True,
                                                temperature=1.0,
                                                top_p=0.95,
                                                top_k=50,
                                                num_beams=1,
                                                repetition_penalty=infer_args.repetition_penalty)
                        
                        for sample_idx in range(infer_args.k):
                            if model_args.model_name in ["gemma3", "apertus"]:
                                generation = model.generate(input_ids,
                                                        generation_config=high_t_generation_config,
                                                        return_dict_in_generate=True,
                                                        output_scores=True,
                                                        max_new_tokens=infer_args.max_length)
                            else:
                                generation = model.generate(input_ids,
                                                        generation_config=high_t_generation_config,
                                                        return_dict_in_generate=True,
                                                        output_scores=True,
                                                        max_new_tokens=infer_args.max_length,
                                                        pad_token_id=tokenizer.pad_token_id,
                                                        eos_token_id=tokenizer.eos_token_id, 
                                                        bos_token_id=tokenizer.bos_token_id)
                            gen_text, norm_prob = output_split(generation, tokenizer, len(input_ids[0]), prompt_split)
                            lang_generations.append(gen_text)
                            lang_probs.append(norm_prob)

                        generations[lang] = lang_generations
                        norm_probs[lang] = lang_probs
                    else:
                        # Vanilla mode: greedy decoding + 3 low-T samples (temperature=0.1)
                        lang_generations = []
                        lang_probs = []
                        
                        # First, generate the greedy sample (temperature=0)
                        # For Gemma3 and Apertus, explicitly set do_sample=False to override model defaults
                        if model_args.model_name in ["gemma3", "apertus"]:
                            generation_config = GenerationConfig(
                                                    do_sample=False,
                                                    num_beams=infer_args.num_beams,
                                                    repetition_penalty=infer_args.repetition_penalty,
                                                    temperature=None,
                                                    top_p=None,
                                                    top_k=None)
                        else:
                            generation_config = GenerationConfig(
                                                    do_sample=False,
                                                    num_beams=infer_args.num_beams,
                                                    repetition_penalty=infer_args.repetition_penalty)

                        # For Gemma3 and Apertus, don't override token IDs as they have specific defaults
                        if model_args.model_name in ["gemma3", "apertus"]:
                            generation = model.generate(input_ids,
                                                    generation_config=generation_config,
                                                    return_dict_in_generate=True,
                                                    output_scores=True,
                                                    max_new_tokens=infer_args.max_length,
                                                    do_sample=False)
                        else:
                            generation = model.generate(input_ids,
                                                    generation_config=generation_config,
                                                    return_dict_in_generate=True,
                                                    output_scores=True,
                                                    max_new_tokens=infer_args.max_length,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    eos_token_id=tokenizer.eos_token_id, 
                                                    bos_token_id=tokenizer.bos_token_id)
                        
                        gen_text, norm_prob = output_split(generation, tokenizer, len(input_ids[0]), prompt_split)
                        lang_generations.append(gen_text)
                        lang_probs.append(norm_prob)
                        
                        # Now generate 3 low-temperature samples (temperature=0.1)
                        low_t_generation_config = GenerationConfig(
                                                do_sample=True,
                                                temperature=0.1,
                                                top_p=0.95,
                                                top_k=50,
                                                num_beams=1,
                                                repetition_penalty=infer_args.repetition_penalty)
                        
                        for sample_idx in range(3):
                            if model_args.model_name in ["gemma3", "apertus"]:
                                generation = model.generate(input_ids,
                                                        generation_config=low_t_generation_config,
                                                        return_dict_in_generate=True,
                                                        output_scores=True,
                                                        max_new_tokens=infer_args.max_length)
                            else:
                                generation = model.generate(input_ids,
                                                        generation_config=low_t_generation_config,
                                                        return_dict_in_generate=True,
                                                        output_scores=True,
                                                        max_new_tokens=infer_args.max_length,
                                                        pad_token_id=tokenizer.pad_token_id,
                                                        eos_token_id=tokenizer.eos_token_id, 
                                                        bos_token_id=tokenizer.bos_token_id)
                            gen_text, norm_prob = output_split(generation, tokenizer, len(input_ids[0]), prompt_split)
                            lang_generations.append(gen_text)
                            lang_probs.append(norm_prob)

                        generations[lang] = lang_generations
                        norm_probs[lang] = lang_probs

            # print(data_point)
            instance = {
                "question_id": batch["question_id"] if "question_id" in batch.keys() else f"id_{idx+1}",
                "question": batch["question"],
                "answer": batch["answer"],
                "output": generations,
                "probs": norm_probs
            }
            # Preserve rephrasing metadata if present
            if "original_id" in batch:
                instance["original_id"] = batch["original_id"]
            if "rephrase_idx" in batch:
                instance["rephrase_idx"] = batch["rephrase_idx"]
            # print(instance)

            # Real-time saving the results.
            with open(infer_args.save_path, "a+") as fw: 
                instance_write = json.dumps(obj=instance, ensure_ascii=False)
                fw.write(instance_write + '\n')

            t.set_postfix()
            t.update(1)


def gpt_generate(dataset):
    # Generate answer on input QA val set on multiple languages.
    logging.info("Start generating ...")
    with tqdm(total=len(dataset)) as t:
        # for data in data_pool[:3]:
        for idx, batch in enumerate(dataset):
            input_context = batch["input"]
            # print(input_context)
            # set_trace()
            generations, norm_probs = {}, {}
            for lang in language_list:
                generated_text, token_probs = get_chatgpt_info(model_args.model_name, input_context[lang], infer_args.temperature, max_tokens=infer_args.max_length, logprobs=False)
                # print(token_probs)
                norm_prob = float(np.array(pow(reduce(operator.mul, token_probs), 1/len(token_probs))))
                generations[lang] = generated_text
                norm_probs[lang] = norm_prob
            
            instance = {
                "question_id": batch["question_id"] if "question_id" in batch.keys() else f"id_{idx+1}",
                "question": batch["question"],
                "answer": batch["answer"],
                "output": generations,
                "probs": norm_probs
            }

            with open(infer_args.save_path, mode="a+") as fw: 
                data_rec = json.dumps(obj=instance, ensure_ascii=False)
                fw.write(data_rec + '\n')

            t.set_postfix()
            t.update(1)


if __name__=="__main__":
    # Choose the appropriate dataset loader based on inference mode
    if infer_args.inference_mode == "rephrased_sampling":
        dataset, prompt_split = rephrased_dataset_loader()
    else:
        dataset, prompt_split = dataset_loader()

    # Set up logging.
    infer_args.output_dir = os.path.join(infer_args.output_dir.format(data_args.dataset), 
                                         f"{model_args.model_name}_{data_args.dataset}_{infer_args.suffix}")
    if not os.path.exists(infer_args.output_dir):
        os.makedirs(infer_args.output_dir)

    log_path = os.path.join(infer_args.output_dir, f"generate.log")
    # print(f"Log path: {infer_args.log_path}")
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    # Format the output file.
    # import pdb; pdb.set_trace()
    infer_args.save_path = os.path.join(infer_args.output_dir, "generate.json")
    if data_args.continue_generate and os.path.exists(infer_args.save_path):
        exist_num = len(read_jsonl(infer_args.save_path))
        # Split the dataset if needed.
        dataset = dataset[exist_num::]
    else:
        # dataset = dataset[:3]
        open(infer_args.save_path, "w").close()

    data_len = len(dataset)
    logging.info(f"The number of dataset: {data_len}")
    logging.info(f"Arguments:\nModel Arguments: {model_args}\nData Arguments: {data_args}\nInference Arguments: {infer_args}")

    Path(infer_args.wandb_dir).mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=infer_args.wandb_project,
        entity=infer_args.wandb_entity,
        mode=infer_args.wandb_mode,
        dir=infer_args.wandb_dir,
        name=f"{model_args.model_name}_{data_args.dataset}_{infer_args.suffix}",
        config={
            "model": model_args.model_name,
            "dataset": data_args.dataset,
            "inference_mode": infer_args.inference_mode,
            "seed": device_args.seed,
            "num_examples": data_len,
            "k": infer_args.k,
            "temperature": infer_args.temperature,
        },
    )

    start_time = time.time()
    if model_args.model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
        gpt_generate(dataset)
    elif model_args.model_name in ["llama3", "vicuna", "gpt2", "aya", "gemma3", "qwen", "apertus", "ministral"]:
        llama_generate(dataset, prompt_split)

    # import pdb; pdb.set_trace()
    elapsed_time = format_seconds(time.time() - start_time)
    logging.info(f"Total elapsed time: {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s")

    # Convert jsonl to json format.
    logging.info("Generating is done.")
    jsonl2json(infer_args.save_path, infer_args.save_path)
    logging.info(f"Save to {infer_args.save_path}")
    wandb.log({"generated_examples": data_len, "elapsed_seconds": time.time() - start_time})
    artifact = wandb.Artifact(run.name, type="generations")
    artifact.add_file(infer_args.save_path)
    run.log_artifact(artifact)
    wandb.finish()
