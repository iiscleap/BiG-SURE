#!/usr/bin/env python3
"""
Utility to load vanilla accuracy for consistent AUROC evaluation.

All uncertainty methods should use accuracy from the VANILLA run's greedy answer
(most_likely_answer['accuracy'] or greedy_answer['accuracy']) for fair comparison.

This matches how compute_snne.py evaluates accuracy.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

TASK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WANDB_BASE = str(TASK_ROOT / "outputs" / "wandb")
DEFAULT_LOG_SUMMARY = str(TASK_ROOT / "outputs" / "log_summary.csv")

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_vanilla_generations(vanilla_run_dir: str, wandb_base_dir: str = DEFAULT_WANDB_BASE) -> Dict[str, Any]:
    """
    Load vanilla generations from a wandb run directory.
    
    Args:
        vanilla_run_dir: Wandb run ID or path to run directory
        wandb_base_dir: Base directory for wandb runs
    
    Returns:
        Dictionary of generations keyed by question ID
    """
    p = Path(vanilla_run_dir)
    if not p.is_absolute():
        p = Path(wandb_base_dir) / vanilla_run_dir
    
    validation_pkl = p if p.is_file() and p.suffix == '.pkl' else p / "files" / "validation_generations.pkl"
    
    if not validation_pkl.exists():
        raise FileNotFoundError(f"Vanilla generations not found: {validation_pkl}")
    
    generations = load_pickle(validation_pkl)
    logger.info(f"Loaded {len(generations)} vanilla samples from {validation_pkl}")
    return generations


def extract_vanilla_accuracy(vanilla_generations: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract accuracy from vanilla generations.
    
    Uses greedy_answer['accuracy'] or most_likely_answer['accuracy'] (same as SNNE).
    
    Args:
        vanilla_generations: Dictionary from load_vanilla_generations
    
    Returns:
        Dictionary mapping question_id -> accuracy (0 or 1)
    """
    accuracy_dict = {}
    
    for qid, data in vanilla_generations.items():
        accuracy = None
        
        # Try greedy_answer first (newer format)
        if isinstance(data.get('greedy_answer'), dict):
            accuracy = data['greedy_answer'].get('accuracy')
        
        # Fallback to most_likely_answer (older format)
        if accuracy is None and isinstance(data.get('most_likely_answer'), dict):
            accuracy = data['most_likely_answer'].get('accuracy')
        
        if accuracy is not None:
            accuracy_dict[str(qid)] = float(accuracy)
    
    logger.info(f"Extracted accuracy for {len(accuracy_dict)} questions "
                f"(mean: {sum(accuracy_dict.values())/len(accuracy_dict):.4f})")
    return accuracy_dict


def get_vanilla_run_id(dataset: str, model_name: str, log_summary_path: str = DEFAULT_LOG_SUMMARY) -> Optional[str]:
    """
    Look up vanilla run ID from log_summary.csv.
    
    Args:
        dataset: Dataset name (e.g., 'bioasq', 'svamp', 'trivia_qa')
        model_name: Model name (e.g., 'Meta-Llama-3.1-8B-Instruct')
        log_summary_path: Path to log_summary.csv
    
    Returns:
        Vanilla run ID (e.g., 'run-20260110_120246-jpt8z51v') or None if not found
    """
    try:
        df = pd.read_csv(log_summary_path)
        
        # Handle trivia_qa naming convention:
        # In log_summary.csv: dataset='trivia', model='qa_<model_name>'
        dataset_query = dataset
        model_query = model_name
        
        if dataset == 'trivia_qa':
            dataset_query = 'trivia'
            # Add 'qa_' prefix if not already present
            if not model_name.startswith('qa_'):
                model_query = 'qa_' + model_name
        
        # Filter for vanilla (rephrased=False) runs matching dataset and model
        mask = (
            (df['dataset'] == dataset_query) &
            (df['model'] == model_query) &
            (df['rephrased'] == False)
        )
        
        matches = df[mask]
        if len(matches) == 0:
            logger.warning(f"No vanilla run found for {dataset}/{model_name} "
                          f"(queried as {dataset_query}/{model_query})")
            return None
        
        wandb_folder_id = matches.iloc[0]['wandb_folder_id']
        vanilla_run = f"run-{wandb_folder_id}"
        logger.info(f"Found vanilla run for {dataset}/{model_name}: {vanilla_run}")
        return vanilla_run
        
    except Exception as e:
        logger.warning(f"Failed to look up vanilla run: {e}")
        return None


def load_vanilla_accuracy_dict(
    dataset: str,
    model_name: str,
    vanilla_run_dir: Optional[str] = None,
    wandb_base_dir: str = DEFAULT_WANDB_BASE,
    log_summary_path: str = DEFAULT_LOG_SUMMARY
) -> Tuple[Dict[str, float], float]:
    """
    Load vanilla accuracy dictionary for a dataset/model combination.
    
    This is the main entry point for getting vanilla accuracy.
    
    Args:
        dataset: Dataset name
        model_name: Model name
        vanilla_run_dir: Optional explicit vanilla run directory (if not provided, looks up from log_summary.csv)
        wandb_base_dir: Base directory for wandb runs
        log_summary_path: Path to log_summary.csv
    
    Returns:
        Tuple of (accuracy_dict, overall_accuracy)
        - accuracy_dict: Dict mapping question_id -> accuracy (0 or 1)
        - overall_accuracy: Mean accuracy across all questions
    """
    # Get vanilla run directory
    if vanilla_run_dir is None:
        vanilla_run_dir = get_vanilla_run_id(dataset, model_name, log_summary_path)
        if vanilla_run_dir is None:
            raise ValueError(f"Could not find vanilla run for {dataset}/{model_name}")
    
    # Load generations and extract accuracy
    vanilla_generations = load_vanilla_generations(vanilla_run_dir, wandb_base_dir)
    accuracy_dict = extract_vanilla_accuracy(vanilla_generations)
    
    overall_accuracy = sum(accuracy_dict.values()) / len(accuracy_dict) if accuracy_dict else 0.0
    
    return accuracy_dict, overall_accuracy
