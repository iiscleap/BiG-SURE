#!/usr/bin/env python3
"""
Shared utilities for multilingual uncertainty measures.

This module provides:
- JSON data loading for vanilla and sampling inference results
- Multilingual NLI model wrapper (mDeBERTa)
- PREM accuracy calculation (from MlingConf paper)
- Semantic clustering via entailment
- Evaluation metrics (AUROC, AUARC, AUCPR)
"""
import os
import re
import json
import string
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported languages in MlingConf data
LANGUAGES = ['en', 'zh', 'ja', 'fr']
# LANGUAGES = ['hi']


# =============================================================================
# Data Loading
# =============================================================================

def load_json_data(path: str) -> List[Dict]:
    """Load JSON data from file (standard JSON array or JSONL)."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        logger.warning(f"Empty file: {path}")
        return []
    if content[0] == '[':
        data = json.loads(content)
    else:
        data = []
        for line_no, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"{e.msg} (line {line_no})", e.doc, e.pos
                ) from e
    logger.info(f"Loaded {len(data)} samples from {path}")
    return data


def get_languages(data: List[Dict]) -> List[str]:
    """Extract available languages from data."""
    if len(data) == 0:
        return []
    # Check first sample's output field for available languages
    sample = data[0]
    if 'output' in sample and isinstance(sample['output'], dict):
        return list(sample['output'].keys())
    return LANGUAGES


def validate_generation_pair(
    vanilla_data: List[Dict],
    sampling_data: List[Dict],
    languages: List[str],
    num_vanilla_outputs: int = 4,
    num_sampling_outputs: int = 10,
    require_accuracy: bool = False,
) -> None:
    """Validate the aligned vanilla/sampling contract used by all baselines."""
    if not vanilla_data or not sampling_data:
        raise ValueError('Vanilla and sampling generation files must both be non-empty.')

    if any(item.get('question_id') is None for item in vanilla_data + sampling_data):
        raise ValueError('Every generation record must contain a question_id.')
    vanilla_ids = [str(item['question_id']) for item in vanilla_data]
    sampling_ids = [str(item['question_id']) for item in sampling_data]
    if len(set(vanilla_ids)) != len(vanilla_ids):
        raise ValueError('Vanilla generation file contains duplicate question IDs.')
    if len(set(sampling_ids)) != len(sampling_ids):
        raise ValueError('Sampling generation file contains duplicate question IDs.')
    if vanilla_ids != sampling_ids:
        missing_sampling = sorted(set(vanilla_ids) - set(sampling_ids))
        missing_vanilla = sorted(set(sampling_ids) - set(vanilla_ids))
        raise ValueError(
            'Vanilla and sampling records are not aligned in question-ID order. '
            f'Missing from sampling: {missing_sampling[:3]}; '
            f'missing from vanilla: {missing_vanilla[:3]}.'
        )

    empty_outputs = 0
    for mode, records, required_outputs in (
        ('vanilla', vanilla_data, num_vanilla_outputs),
        ('sampling', sampling_data, num_sampling_outputs),
    ):
        for item in records:
            qid = str(item['question_id'])
            for field in ('question', 'answer', 'output', 'probs'):
                if not isinstance(item.get(field), dict):
                    raise ValueError(f'{mode} question {qid} has invalid or missing {field}.')
            for language in languages:
                outputs = item['output'].get(language)
                probs = item['probs'].get(language)
                if not isinstance(outputs, list) or len(outputs) != required_outputs:
                    raise ValueError(
                        f'{mode} question {qid}/{language} has '
                        f'{len(outputs) if isinstance(outputs, list) else 0} outputs; '
                        f'expected {required_outputs}.'
                    )
                if not all(isinstance(output, str) for output in outputs):
                    raise ValueError(f'{mode} question {qid}/{language} contains a non-string output.')
                empty_outputs += sum(not output.strip() for output in outputs)
                if not isinstance(probs, list) or len(probs) != required_outputs:
                    raise ValueError(
                        f'{mode} question {qid}/{language} probability count does not '
                        f'match its {required_outputs} outputs.'
                    )
                try:
                    numeric_probs = np.asarray(probs, dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f'{mode} question {qid}/{language} contains a non-numeric probability.'
                    ) from exc
                if (not np.isfinite(numeric_probs).all()
                        or (numeric_probs < 0).any() or (numeric_probs > 1).any()):
                    raise ValueError(
                        f'{mode} question {qid}/{language} probabilities must be finite '
                        'and in [0, 1].'
                    )
            if mode == 'vanilla' and require_accuracy:
                accuracy = item.get('accuracy')
                missing_languages = [lang for lang in languages if lang not in (accuracy or {})]
                if missing_languages:
                    raise ValueError(
                        f'Vanilla question {qid} is missing accuracy labels for '
                        f'{missing_languages}.'
                    )
                try:
                    accuracy_values = np.asarray(
                        [accuracy[language] for language in languages], dtype=float
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f'Vanilla question {qid} contains a non-numeric accuracy label.'
                    ) from exc
                if (not np.isfinite(accuracy_values).all()
                        or (accuracy_values < 0).any() or (accuracy_values > 1).any()):
                    raise ValueError(
                        f'Vanilla question {qid} accuracy labels must be in [0, 1].'
                    )
    if empty_outputs:
        logger.warning(
            'Generation pair contains %d empty model outputs; preserving them as valid '
            'model behavior.', empty_outputs
        )


def extract_vanilla_data(data: List[Dict], language: str) -> Tuple[List[str], List[str], List[str], List[float]]:
    """
    Extract data for a single language from vanilla JSON.
    
    Only the first (greedy) sample is extracted when output/probs are lists.
    
    Returns:
        questions: List of questions
        answers: List of ground truth answers
        outputs: List of model outputs (greedy only)
        probs: List of probabilities (greedy only)
    """
    questions = []
    answers = []
    outputs = []
    probs = []
    
    for item in data:
        questions.append(item['question'].get(language, ''))
        answers.append(item['answer'].get(language, ''))
        
        out = item['output'].get(language, '')
        if isinstance(out, list):
            out = out[0] if len(out) > 0 else ''
        outputs.append(out)
        
        pr = item['probs'].get(language, 0.0)
        if isinstance(pr, list):
            pr = pr[0] if len(pr) > 0 else 0.0
        probs.append(pr)
    
    return questions, answers, outputs, probs


def extract_sampling_data(data: List[Dict], language: str) -> Tuple[List[str], List[str], List[List[str]], List[List[float]]]:
    """
    Extract data for a single language from sampling JSON.
    
    Returns:
        questions: List of questions
        answers: List of ground truth answers
        outputs: List of lists of model outputs (multiple samples per question)
        probs: List of lists of probabilities
    """
    questions = []
    answers = []
    outputs = []
    probs = []
    
    for item in data:
        questions.append(item['question'].get(language, ''))
        answers.append(item['answer'].get(language, ''))
        outputs.append(item['output'].get(language, []))
        probs.append(item['probs'].get(language, []))
    
    return questions, answers, outputs, probs


# =============================================================================
# PREM Accuracy (from MlingConf paper)
# =============================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_prem_accuracy(prediction: str, ground_truth: str) -> int:
    """
    Calculate PREM (Positive-Recall Exact Matching) accuracy.
    
    Returns 1 if:
    - normalized(ground_truth) is contained in normalized(prediction), OR
    - normalized(prediction) is contained in normalized(ground_truth)
    
    Otherwise returns 0.
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    
    if not norm_pred or not norm_gt:
        return 0
    
    return int(norm_gt in norm_pred or norm_pred in norm_gt)


def normalize_text(s: str) -> str:
    """Normalize text for similarity comparison (alias for normalize_answer)."""
    return normalize_answer(s)


def char_ngrams(text: str, n: int = 3) -> set:
    """Generate character n-grams from text."""
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def jaccard_set(s1: set, s2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))


def calculate_batch_accuracy(predictions: List[str], ground_truths: List[str]) -> List[int]:
    """Calculate PREM accuracy for a batch of predictions."""
    return [calculate_prem_accuracy(pred, gt) for pred, gt in zip(predictions, ground_truths)]


# =============================================================================
# Multilingual NLI Model
# =============================================================================

class MultilingualEntailmentDeberta:
    """
    Multilingual NLI model using mDeBERTa-v3-base-xnli-multilingual-nli-2mil7.
    
    This model supports 27 languages including: en, zh, ja, fr, th.
    """
    
    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"):
        logger.info(f"Loading multilingual NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        
        # This checkpoint uses raw labels 0=entailment, 1=neutral, 2=contradiction.
        # check_implication remaps them to the SNNE convention.
        self.id2label = self.model.config.id2label
        logger.info(f"Model labels: {self.id2label}")
    
    def check_implication(
        self,
        text1,
        text2,
        *args,
        batch_size: int = 256,
        **kwargs,
    ):
        """
        Check if text1 implies text2.

        Scalar inputs return ``(int, float)``. Sequence inputs return parallel
        ``(List[int], List[float])`` values, matching the batched KLE graph API.
        Implications use 0=contradiction, 1=neutral, 2=entailment; confidence is
        the predicted class probability, matching KLE's EntailmentDeberta.

        Note: This model's raw output is {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        so we remap the class index to match the SNNE convention.
        """
        # Remap: model's raw 0->entailment becomes 2, raw 2->contradiction becomes 0
        label_remap = {0: 2, 1: 1, 2: 0}  # entail->2, neutral->1, contradict->0
        text1_is_batch = isinstance(text1, (list, tuple))
        text2_is_batch = isinstance(text2, (list, tuple))

        if text1_is_batch != text2_is_batch:
            raise TypeError("text1 and text2 must both be strings or both be sequences")

        if not text1_is_batch:
            inputs = self.tokenizer(
                text1, text2, return_tensors="pt", truncation=True, max_length=512
            ).to(DEVICE)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            raw_prediction = torch.argmax(logits, dim=1).item()
            confidence = F.softmax(logits, dim=1).max(dim=1).values.item()
            return label_remap[raw_prediction], confidence

        if len(text1) != len(text2):
            raise ValueError("Batched text1 and text2 must have the same length")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        implications = []
        confidences = []
        for start in range(0, len(text1), batch_size):
            batch_text1 = list(text1[start:start + batch_size])
            batch_text2 = list(text2[start:start + batch_size])
            inputs = self.tokenizer(
                batch_text1,
                batch_text2,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(DEVICE)
            with torch.no_grad():
                logits = self.model(**inputs).logits

            raw_predictions = torch.argmax(logits, dim=1).cpu().tolist()
            prediction_confidences = F.softmax(logits, dim=1).max(dim=1).values.cpu().tolist()
            implications.extend(label_remap[prediction] for prediction in raw_predictions)
            confidences.extend(float(probability) for probability in prediction_confidences)

        return implications, confidences
    
    def get_similarity_score(self, text1: str, text2: str, strict_entailment: bool = True) -> float:
        """
        Get entailment similarity score between two texts.
        
        Returns the softmax probability of the entailment class.
        """
        inputs = self.tokenizer(text1, text2, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
        
        # Model's index 0 = entailment for this model
        return probs[0, 0].item()
    
    def check_equivalence(self, text1: str, text2: str, strict: bool = False) -> bool:
        """
        Check if two texts are semantically equivalent.
        
        Args:
            text1: First text
            text2: Second text
            strict: If True, requires bidirectional entailment. 
                   If False, requires no contradiction and not both neutral.
        """
        impl_1, _ = self.check_implication(text1, text2)
        impl_2, _ = self.check_implication(text2, text1)

        if strict:
            return impl_1 == 2 and impl_2 == 2
        else:
            # No contradiction and not both neutral
            implications = [impl_1, impl_2]
            return (0 not in implications) and (implications != [1, 1])


# =============================================================================
# Semantic Clustering
# =============================================================================

def get_semantic_ids_using_entailment(
    strings_list: List[str], 
    model: MultilingualEntailmentDeberta,
    strict_entailment: bool = False,
    question: str = None
) -> List[int]:
    """
    Group predictions by semantic equivalence using NLI model.
    
    Uses greedy clustering: iterate through strings, assign to existing cluster
    if equivalent, otherwise create new cluster.
    
    Returns:
        List of semantic IDs (cluster assignments)
    """
    if len(strings_list) == 0:
        return []
    
    semantic_ids = [-1] * len(strings_list)
    next_id = 0
    
    # Optionally prepend question for context
    if question:
        strings_list = [f"{question} {s}" for s in strings_list]
    
    for i in range(len(strings_list)):
        if semantic_ids[i] >= 0:
            continue
        
        # Start a new cluster
        semantic_ids[i] = next_id
        
        # Find all equivalent strings
        for j in range(i + 1, len(strings_list)):
            if semantic_ids[j] >= 0:
                continue
            
            if model.check_equivalence(strings_list[i], strings_list[j], strict=strict_entailment):
                semantic_ids[j] = next_id
        
        next_id += 1
    
    return semantic_ids


# =============================================================================
# Evaluation Metrics
# =============================================================================

def auroc(y_true: List[float], y_score: List[float]) -> float:
    """
    Compute Area Under ROC Curve.
    
    Args:
        y_true: Binary labels (0 or 1)
        y_score: Uncertainty scores (higher = more uncertain = more likely wrong)
    
    Returns:
        AUROC score
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Skip NaNs
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]
    
    if len(set(y_true)) < 2:
        logger.warning("Only one class present in y_true, cannot compute AUROC")
        return -1.0
    
    return roc_auc_score(y_true, y_score)


def auarc(y_score: List[float], y_true: List[float]) -> float:
    """
    Compute Area Under Accuracy-Rejection Curve.
    
    Sorts by uncertainty (ascending), computes cumulative accuracy.
    """
    import pandas as pd
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Skip NaNs
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(y_true) < 2:
        return float('nan')
    
    df = pd.DataFrame({"u": y_score, 'a': y_true}).sort_values('u', ascending=True)
    df['amean'] = df['a'].expanding().mean()
    
    from sklearn.metrics import auc
    return auc(np.linspace(0, 1, len(df)), df['amean'])


def aucpr(y_score: List[float], y_true: List[float]) -> float:
    """
    Compute Area Under Prediction-Rejection curve (PRR).
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Skip NaNs
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(y_true) == 0:
        return float('nan')
    
    # Normalize y_true
    min_t, max_t = np.min(y_true), np.max(y_true)
    if np.isclose(min_t, max_t):
        min_t -= 1
        max_t += 1
    y_true = (y_true - min_t) / (max_t - min_t)
    
    ue = np.array(y_score)
    num_obs = len(ue)
    num_rej = int(num_obs)
    
    # Sort in ascending order: the least uncertain come first
    ue_argsort = np.argsort(ue)
    sorted_metrics = y_true[ue_argsort]
    
    cumsum = np.cumsum(sorted_metrics)[-num_rej:]
    scores = (cumsum / np.arange((num_obs - num_rej) + 1, num_obs + 1))[::-1]
    prr_score = np.sum(scores) / num_rej
    
    return prr_score


# =============================================================================
# Entropy Calculations
# =============================================================================

def predictive_entropy(log_probs: List[float]) -> float:
    """
    Compute MC estimate of predictive entropy.
    
    E[-log p(x)] ~= -1/N sum_i log p(x_i)
    """
    log_probs = np.array(log_probs)
    return -np.mean(log_probs)


def cluster_assignment_entropy(semantic_ids: List[int]) -> float:
    """
    Compute entropy over cluster assignment distribution.
    
    Estimates categorical distribution over clusters and computes entropy.
    """
    if len(semantic_ids) == 0:
        return 0.0
    
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    
    # Avoid log(0)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log(probabilities))
    
    return entropy


def logsumexp_by_id(
    semantic_ids: List[int], 
    log_likelihoods: List[float], 
    agg: str = 'sum_normalized'
) -> List[float]:
    """
    Aggregate log likelihoods by semantic ID using logsumexp.
    
    Returns log likelihood per semantic class.
    """
    if len(semantic_ids) != len(log_likelihoods):
        raise ValueError(
            'semantic_ids and log_likelihoods must contain the same number of values'
        )
    if not semantic_ids:
        return []

    def stable_logsumexp(values):
        values = np.asarray(values, dtype=float)
        maximum = np.max(values)
        if np.isneginf(maximum):
            return float('-inf')
        return float(maximum + np.log(np.exp(values - maximum).sum()))

    unique_ids = sorted(set(semantic_ids))
    log_likelihood_per_semantic_id = []
    normalization = stable_logsumexp(log_likelihoods)
    
    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        
        if agg == 'sum_normalized':
            logsumexp_value = stable_logsumexp(id_log_likelihoods) - normalization
        else:
            logsumexp_value = stable_logsumexp(id_log_likelihoods)
        
        log_likelihood_per_semantic_id.append(logsumexp_value)
    
    return log_likelihood_per_semantic_id


def semantic_entropy(semantic_ids: List[int], log_probs: List[float]) -> float:
    """
    Compute semantic entropy.
    
    Aggregates probabilities by semantic cluster and computes entropy.
    """
    if len(semantic_ids) == 0:
        return 0.0
    
    log_probs_per_cluster = logsumexp_by_id(semantic_ids, log_probs)
    probs_per_cluster = np.exp(log_probs_per_cluster)
    
    # Normalize
    probs_per_cluster = probs_per_cluster / np.sum(probs_per_cluster)
    
    # Compute entropy
    probs_per_cluster = probs_per_cluster[probs_per_cluster > 0]
    entropy = -np.sum(probs_per_cluster * np.log(probs_per_cluster))
    
    return entropy


# =============================================================================
# KL Divergence
# =============================================================================

def kl_divergence(p: List[float], q: List[float], eps: float = 1e-10) -> float:
    """
    Compute KL divergence: KL(q || p)
    
    Args:
        p: observed distribution
        q: target/ideal distribution
        eps: small value for numerical stability
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    return float(np.sum(q * np.log((q + eps) / (p + eps))))


def compute_pmf(equiv_count: int, total_count: int) -> List[float]:
    """Compute PMF from equivalence counts: [P(not_equiv), P(equiv)]"""
    if total_count == 0:
        return [0.5, 0.5]
    return [(total_count - equiv_count) / total_count, equiv_count / total_count]


# =============================================================================
# Output Utilities
# =============================================================================

def save_results_csv(results: Dict[str, Any], output_path: str):
    """Save results dictionary to CSV."""
    import pandas as pd
    
    df = pd.DataFrame(results)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")


def save_results_json(results: Dict[str, Any], output_path: str):
    """Save results dictionary to JSON."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to {output_path}")
