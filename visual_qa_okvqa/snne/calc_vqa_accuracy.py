#!/usr/bin/env python3
"""Compute VQA accuracy using the official VQAv2 evaluation metric.

This script implements the official VQA evaluation metric which is robust to 
inter-human variability. Machine accuracies are averaged over all 10-choose-9 
sets of human annotators.

Based on the official VQA evaluation code by Aishwarya Agrawal:
https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py

Usage:
  python snne/calc_vqa_accuracy.py --input PATH --output OUT.json [--dataset DATASET]
"""
import argparse
import os
import pickle
import logging
import json
import re
import sys
from glob import glob


class VQAAccuracyEvaluator:
    """VQA accuracy evaluator based on official VQAv2 evaluation code."""
    
    def __init__(self, n=2):
        self.n = n
        self.contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
            "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
            "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
            "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
            "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
            "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
            "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
            "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
            "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
            "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
            "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
            "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
            "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
            "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
            "somebodys": "somebody's", "someoned": "someone'd",
            "someoned've": "someone'd've", "someone'dve": "someone'd've",
            "someonell": "someone'll", "someones": "someone's",
            "somethingd": "something'd", "somethingd've": "something'd've",
            "something'dve": "something'd've", "somethingll": "something'll",
            "thats": "that's", "thered": "there'd", "thered've": "there'd've",
            "there'dve": "there'd've", "therere": "there're", "theres": "there's",
            "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
            "theyll": "they'll", "theyre": "they're", "theyve": "they've",
            "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
            "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
            "whatll": "what'll", "whatre": "what're", "whats": "what's",
            "whatve": "what've", "whens": "when's", "whered": "where'd",
            "wheres": "where's", "whereve": "where've", "whod": "who'd",
            "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
            "whos": "who's", "whove": "who've", "whyll": "why'll",
            "whyre": "why're", "whys": "why's", "wont": "won't",
            "wouldve": "would've", "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
            "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
            "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
            "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
            "youve": "you've"
        }
        
        self.manualMap = {
            'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
            'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
            'nine': '9', 'ten': '10'
        }
        
        self.articles = ['a', 'an', 'the']
        
        # Regex patterns from official code
        self.periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile(r"(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']
    
    def processPunctuation(self, inText):
        """Process punctuation following official VQA evaluation."""
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) is not None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText
    
    def processDigitArticle(self, inText):
        """Process digits and articles following official VQA evaluation."""
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.get(word, word)
            if word not in self.articles:
                outText.append(word)
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText
    
    def normalize_answer(self, answer):
        """Normalize an answer string using VQA preprocessing."""
        if not answer:
            return ''
        answer = answer.replace('\n', ' ')
        answer = answer.replace('\t', ' ')
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        return answer
    
    def compute_accuracy(self, predicted, ground_truths):
        """Compute VQA accuracy for a single prediction.
        
        VQA accuracy is computed as: min(# humans that said that answer / 3, 1)
        Averaged over all 10-choose-9 sets of annotators.
        
        Args:
            predicted: The model's predicted answer (string)
            ground_truths: List of ground truth answers from annotators
            
        Returns:
            float: Accuracy score between 0 and 1
        """
        if not ground_truths:
            return 0.0
        
        # Clean up ground truth answers
        gtAnswers = []
        for ans in ground_truths:
            if isinstance(ans, dict):
                ans = ans.get('answer', str(ans))
            ans = str(ans).replace('\n', ' ').replace('\t', ' ').strip()
            gtAnswers.append(ans)
        
        # Clean up predicted answer
        resAns = str(predicted).replace('\n', ' ').replace('\t', ' ').strip()
        
        # Only apply normalization if there's variation in GT answers
        if len(set(gtAnswers)) > 1:
            gtAnswers = [self.normalize_answer(ans) for ans in gtAnswers]
            resAns = self.normalize_answer(resAns)
        else:
            # Still lowercase for comparison
            gtAnswers = [ans.lower() for ans in gtAnswers]
            resAns = resAns.lower()
        
        # Compute accuracy using 10-choose-9 averaging
        gtAcc = []
        for i, gtAns in enumerate(gtAnswers):
            # Get other annotators' answers (excluding this one)
            otherGTAns = gtAnswers[:i] + gtAnswers[i+1:]
            # Count how many other annotators gave the same answer as prediction
            matchingAns = [ans for ans in otherGTAns if ans == resAns]
            acc = min(1.0, float(len(matchingAns)) / 3.0)
            gtAcc.append(acc)
        
        # Average over all annotators
        avgGTAcc = float(sum(gtAcc)) / len(gtAcc) if gtAcc else 0.0
        return avgGTAcc


def find_pkl(path):
    """Find pickle file from path or directory."""
    if os.path.isfile(path) and path.endswith('.pkl'):
        return path
    gen_pkl = os.path.join(path, 'validation_generations.pkl')
    if os.path.exists(gen_pkl):
        return gen_pkl
    # Try files subdirectory (wandb structure)
    files_pkl = os.path.join(path, 'files', 'validation_generations.pkl')
    if os.path.exists(files_pkl):
        return files_pkl
    candidates = glob(os.path.join(path, '*.pkl'))
    return candidates[0] if candidates else None


def load_generations(pkl_path):
    """Load generations from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def extract_prediction_gt_and_question(entry):
    """Extract the greedy prediction (temp=0), ground truth, and question from an entry.
    
    Returns: (prediction, ground_truths, question) or (None, None, None) if not available
    """
    # Get greedy/most-likely prediction (temp=0)
    pred = None
    if 'most_likely_answer' in entry and entry['most_likely_answer']:
        mla = entry['most_likely_answer']
        if isinstance(mla, dict):
            pred = mla.get('response', '')
        elif isinstance(mla, str):
            pred = mla
    elif 'most_likely_answers' in entry and len(entry.get('most_likely_answers', [])) > 0:
        first = entry['most_likely_answers'][0]
        if isinstance(first, dict):
            pred = first.get('response', '')
        elif isinstance(first, str):
            pred = first
    
    if pred is None:
        return None, None, None
    
    # Get ground truth references
    gt = []
    if 'reference' in entry and entry['reference']:
        reference = entry['reference']
        if isinstance(reference, dict):
            gt = reference.get('answers', {}).get('text', [])
        elif isinstance(reference, (list, tuple)):
            gt = list(reference)
    
    # Fallback: check if GT is stored elsewhere
    if not gt and 'answers' in entry:
        answers = entry['answers']
        if isinstance(answers, dict) and 'text' in answers:
            gt = answers['text']
        elif isinstance(answers, (list, tuple)):
            gt = list(answers)
    
    # Get question
    question = entry.get('question', '')
    if not question and 'reference' in entry and isinstance(entry['reference'], dict):
        question = entry['reference'].get('question', '')
    
    return pred, gt, question


def main():
    parser = argparse.ArgumentParser(
        description='Compute VQA accuracy using official VQAv2 evaluation metric'
    )
    parser.add_argument('--input', required=True, 
                        help='Path to pkl file or wandb run directory')
    parser.add_argument('--dataset', default='vqa', 
                        help='Dataset name for metadata (default: vqa)')
    parser.add_argument('--max-examples', type=int, default=None, 
                        help='Limit number of examples for quick runs')
    parser.add_argument('--output', required=True, 
                        help='Path to write JSON with per-example accuracy and summary stats')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-example results')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    pkl_path = find_pkl(args.input)
    if not pkl_path:
        raise FileNotFoundError(f'No pickle file found at {args.input}')
    
    logging.info(f'Loading generations from {pkl_path}')
    gens = load_generations(pkl_path)
    logging.info(f'Loaded {len(gens)} examples')

    # Initialize evaluator
    evaluator = VQAAccuracyEvaluator()
    
    total = 0
    total_accuracy = 0.0
    per_example = {}
    skipped = 0
    
    for i, exid in enumerate(sorted(gens.keys())):
        if args.max_examples and i >= args.max_examples:
            break
        
        entry = gens[exid]
        pred, gt, question = extract_prediction_gt_and_question(entry)
        
        if pred is None:
            logging.warning(f'No prediction for example {exid}, skipping.')
            skipped += 1
            continue
        
        if not gt:
            logging.warning(f'No ground truth for example {exid}, skipping.')
            skipped += 1
            continue
        
        # Compute VQA accuracy using official metric
        acc = evaluator.compute_accuracy(pred, gt)
        
        total += 1
        total_accuracy += acc
        per_example[exid] = {
            'id': exid,
            'accuracy': acc,
            'question': question,
            'prediction': pred,
            'prediction_normalized': evaluator.normalize_answer(pred),
            'ground_truths': gt,  # Full ground truth set
        }
        
        if args.verbose:
            print(f'[{exid}] Q: "{question}"')
            print(f'       Pred: "{pred}" -> Norm: "{evaluator.normalize_answer(pred)}"')
            print(f'       GT: {gt}')
            print(f'       Accuracy: {acc:.4f}')

    avg_accuracy = total_accuracy / total if total > 0 else 0.0
    
    print(f'\n{"="*60}')
    print(f'VQA Accuracy Results (Official VQAv2 Metric)')
    print(f'{"="*60}')
    print(f'Input: {pkl_path}')
    print(f'Total examples evaluated: {total}')
    print(f'Skipped examples: {skipped}')
    print(f'Average VQA Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)')
    print(f'{"="*60}')

    # Write JSON output
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    payload = {
        'total': total,
        'skipped': skipped,
        'accuracy': avg_accuracy,
        'accuracy_percent': round(avg_accuracy * 100, 2),
        'dataset': args.dataset,
        'input_path': pkl_path,
        'per_example': per_example,  # Now includes full details: id, accuracy, question, prediction, ground_truths
    }
    
    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)
    
    print(f'Wrote VQA accuracy JSON to {args.output}')
    
    return avg_accuracy


if __name__ == '__main__':
    main()
