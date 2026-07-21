"""Evaluate model generations using Gemini API (batched across all languages).

This script loads vanilla generate.json files from Apertus/Aya models and evaluates
the correctness of the FIRST (greedy) prediction using the Gemini API.
All 5 languages are batched into a single API call per example.

Supports: SciQ, GSM8K, TriviaQA datasets
Models: Apertus, Aya

Usage:
    python gemini_evaluate_generations.py \
        --input_json exp/triviaqa/inference/apertus_triviaqa_infer_vanilla/generate.json \
        --output_json exp/triviaqa/inference/apertus_triviaqa_infer_vanilla/generate_with_accuracy.json \
        --dataset triviaqa \
        --model apertus
"""

import os
import json
import re
import argparse
import logging
import time
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LANGUAGES = ['en', 'zh', 'ja', 'fr']


def load_json_records(path: str) -> list:
    """Load either a JSON array or newline-delimited JSON records."""
    content = Path(path).read_text(encoding='utf-8').strip()
    if not content:
        return []
    if content[0] == '[':
        data = json.loads(content)
    else:
        data = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f'{path} must contain a JSON array or JSONL records')
    return data


def get_system_prompt(dataset: str) -> str:
    """Get dataset-specific system prompt for Gemini evaluation."""
    dataset = dataset.lower()

    if 'triviaqa' in dataset or 'trivia' in dataset:
        return (
            "You are evaluating trivia question answering across multiple languages. "
            "Determine if each Predicted answer is semantically equivalent to the Ground truth. "
            "Be flexible with formatting, extra words, and phrasing as long as the core answer is correct."
        )
    elif 'sciq' in dataset or 'science' in dataset:
        return (
            "You are evaluating science questions across multiple languages. "
            "Determine if each Predicted answer matches the Ground truth. "
            "Be flexible with extra explanation but the core answer must be correct."
        )
    elif 'gsm8k' in dataset or 'math' in dataset:
        return (
            "You are evaluating grade school math word problems across multiple languages. "
            "Determine if each Predicted answer contains the correct numerical answer "
            "that matches the Ground truth. Ignore extra explanation or units as long as the number is correct."
        )
    else:
        return (
            "You are evaluating question answering across multiple languages. "
            "Determine if each Predicted answer is semantically equivalent to the Ground truth."
        )


def build_batched_prompt(example: dict, system_prompt: str, languages: list) -> str:
    """Build a single prompt that asks Gemini to evaluate all languages at once.

    Only uses the first (greedy) prediction per language.
    """
    lines = [system_prompt, ""]
    lines.append(
        "For each language below, decide if the predicted answer is correct (1) or incorrect (0)."
    )
    lines.append(
        'Return ONLY a JSON object mapping each language code to 1 or 0.'
    )
    lines.append('Example response: {"en": 1, "zh": 0, "ja": 1, "fr": 0}')
    lines.append("")

    for lang in languages:
        question = example.get('question', {}).get(lang, '')
        ground_truth = example.get('answer', {}).get(lang, '')
        predictions = example.get('output', {}).get(lang, [])

        if not question or not ground_truth or not predictions:
            continue

        # Only the first (greedy) prediction
        pred = predictions[0] if predictions else ''

        lines.append(f"=== {lang} ===")
        lines.append(f"Question: {question}")
        lines.append(f"Ground truth: {ground_truth}")
        lines.append(f"Predicted: {pred}")
        lines.append("")

    lines.append("Return ONLY the JSON object. No extra text.")
    return "\n".join(lines)


def parse_batched_response(response_text: str, languages: list) -> dict:
    """Parse the JSON response from Gemini into accuracy dict.

    Returns:
        Dict mapping language code -> float (0.0 or 1.0)
    """
    text = response_text.strip()
    # Remove markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse Gemini JSON response: {text[:200]}")
        return None

    accuracies = {}
    for lang in languages:
        if lang in result:
            accuracies[lang] = 1.0 if result[lang] == 1 else 0.0
        else:
            accuracies[lang] = 0.0
    return accuracies


def evaluate_batched(
    example: dict,
    gemini_model,
    system_prompt: str,
    languages: list,
    max_retries: int = 3
) -> dict:
    """Evaluate the greedy prediction for all languages in a single Gemini call.

    Returns:
        Dict mapping language code -> float (0.0 or 1.0)
    """
    prompt = build_batched_prompt(example, system_prompt, languages)

    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(prompt)
            result = parse_batched_response(response.text, languages)

            if result is not None:
                return result
            else:
                logger.warning(f"Parse failed, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue

        except Exception as e:
            logger.error(
                f"Gemini API error (attempt {attempt + 1}/{max_retries}): "
                f"{type(e).__name__}: {str(e)}"
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

    # All retries exhausted
    return {lang: 0.0 for lang in languages}


def load_original_ids_from_rephrased(rephrased_json_path: str) -> set:
    """Load original question IDs from a rephrased JSON file."""
    with open(rephrased_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    original_ids = {str(item['question_id']) for item in data if 'original_id' not in item}
    logger.info(f"Loaded {len(original_ids)} original question IDs from {rephrased_json_path}")
    return original_ids


def process_json(
    input_path: str,
    output_path: str,
    dataset: str,
    languages: list = None,
    api_key: str = None,
    filter_ids_from: str = None
):
    """Process generate.json and add accuracy labels for all languages.

    Only evaluates the first (greedy) prediction per language.
    All languages are batched into a single Gemini call per example.
    """
    if languages is None:
        languages = LANGUAGES

    # Load input JSON
    logger.info(f"Loading input file: {input_path}")
    data = load_json_records(input_path)

    logger.info(f"Loaded {len(data)} examples")

    # Filter to 300 original IDs if specified
    if filter_ids_from and os.path.exists(filter_ids_from):
        valid_ids = load_original_ids_from_rephrased(filter_ids_from)
        data = [item for item in data if str(item.get('question_id', '')) in valid_ids]
        logger.info(f"Filtered to {len(data)} original samples")

    # Initialize Gemini
    if api_key is None:
        api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable must be set or passed as argument")

    genai.configure(api_key=api_key)

    system_prompt = get_system_prompt(dataset)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

    logger.info(f"Initialized Gemini 2.5 Flash for dataset: {dataset}")
    logger.info(f"Evaluating languages: {languages}")
    logger.info(f"Using only first (greedy) prediction per language")
    logger.info(f"Batching all languages into single API call per example")

    # Per-language stats
    lang_correct = {lang: 0 for lang in languages}
    lang_total = {lang: 0 for lang in languages}

    for example in tqdm(data, desc="Evaluating"):
        question_id = example.get('question_id', 'unknown')

        # Find which languages have valid data
        active_langs = []
        for lang in languages:
            q = example.get('question', {}).get(lang, '')
            a = example.get('answer', {}).get(lang, '')
            preds = example.get('output', {}).get(lang, [])
            if q and a and preds:
                active_langs.append(lang)

        if not active_langs:
            logger.warning(f"No valid language data for ID {question_id}, skipping")
            continue

        # Single batched API call for all languages
        accuracies = evaluate_batched(
            example=example,
            gemini_model=gemini_model,
            system_prompt=system_prompt,
            languages=active_langs
        )

        # Store results (single value per language, not a list)
        if 'accuracy' not in example:
            example['accuracy'] = {}
        for lang in active_langs:
            example['accuracy'][lang] = accuracies.get(lang, 0.0)
            lang_correct[lang] += accuracies.get(lang, 0.0)
            lang_total[lang] += 1

        # Rate limiting
        time.sleep(0.5)

    # Log per-language accuracy
    logger.info("")
    overall_correct = 0
    overall_total = 0
    for lang in languages:
        if lang_total[lang] > 0:
            acc = lang_correct[lang] / lang_total[lang]
            logger.info(f"Accuracy ({lang}): {acc:.4f} ({int(lang_correct[lang])}/{lang_total[lang]})")
            overall_correct += lang_correct[lang]
            overall_total += lang_total[lang]

    if overall_total > 0:
        overall_accuracy = overall_correct / overall_total
        logger.info(f"Overall Accuracy: {overall_accuracy:.4f} ({int(overall_correct)}/{overall_total})")
    else:
        overall_accuracy = 0.0
        logger.warning("No predictions were evaluated!")

    # Save output
    logger.info(f"Saving results to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Done!")
    return overall_accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model generations using Gemini API (batched across languages)'
    )
    parser.add_argument(
        '--input_json', type=str, required=True,
        help='Path to input generate.json file'
    )
    parser.add_argument(
        '--output_json', type=str, required=True,
        help='Path to save output JSON with accuracy labels'
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['triviaqa', 'sciq', 'gsm8k', 'triviaqa_hindi'],
        help='Dataset name'
    )
    parser.add_argument(
        '--model', type=str, default='apertus',
        choices=['apertus', 'aya', 'krutrim2'],
        help='Model type (for logging purposes)'
    )
    parser.add_argument(
        '--languages', type=str, nargs='+',
        default=['en', 'zh', 'ja', 'fr'],
        help='Language codes to evaluate (default: all 5)'
    )
    parser.add_argument(
        '--api_key', type=str, default=None,
        help='Google API key (defaults to GOOGLE_API_KEY env var)'
    )
    parser.add_argument(
        '--filter_ids_from', type=str, default=None,
        help='Path to rephrased JSON to extract original 300 IDs for filtering'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Gemini Evaluation Script (Batched, Greedy Only)")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Languages: {args.languages}")
    logger.info(f"Input: {args.input_json}")
    logger.info(f"Output: {args.output_json}")
    logger.info("=" * 80)

    accuracy = process_json(
        input_path=args.input_json,
        output_path=args.output_json,
        dataset=args.dataset,
        languages=args.languages,
        api_key=args.api_key,
        filter_ids_from=args.filter_ids_from
    )

    logger.info("=" * 80)
    logger.info(f"Final Overall Accuracy: {accuracy:.4f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
