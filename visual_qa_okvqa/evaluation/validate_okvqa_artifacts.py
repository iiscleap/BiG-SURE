#!/usr/bin/env python3
"""Validate OKVQA generation and evaluation artifacts without loading models."""

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path


def load_pickle(path):
    with Path(path).open('rb') as infile:
        data = pickle.load(infile)
    if not isinstance(data, dict) or not data:
        raise ValueError(f'{path} must contain a non-empty question-ID dictionary')
    ids = [str(qid) for qid in data]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{path} contains duplicate question IDs after string normalization')
    return data, ids


def validate_responses(qid, responses, expected, label):
    if not isinstance(responses, list) or len(responses) != expected:
        raise ValueError(
            f'{qid} has {len(responses) if isinstance(responses, list) else 0} '
            f'{label} responses; expected {expected}'
        )
    for index, response in enumerate(responses):
        if not isinstance(response, (list, tuple)) or len(response) < 2:
            raise ValueError(f'{qid} {label} response {index} has an invalid tuple schema')
        if not isinstance(response[0], str) or not response[0].strip():
            raise ValueError(f'{qid} {label} response {index} has empty text')
        if response[1] is None or len(response[1]) == 0:
            raise ValueError(f'{qid} {label} response {index} has no token log-likelihoods')


def validate_vanilla(path, expected_examples, high_t, low_t):
    data, ids = load_pickle(path)
    if len(data) != expected_examples:
        raise ValueError(f'{path} has {len(data)} examples; expected {expected_examples}')
    for qid, example in data.items():
        if not isinstance(example, dict) or not isinstance(example.get('most_likely_answer'), dict):
            raise ValueError(f'{qid} is missing most_likely_answer')
        if not example['most_likely_answer'].get('response', '').strip():
            raise ValueError(f'{qid} has an empty greedy answer')
        if 'reference' not in example:
            raise ValueError(f'{qid} is missing its answer reference')
        validate_responses(qid, example.get('low_temp_responses'), low_t, 'low-temperature')
        validate_responses(qid, example.get('responses'), high_t, 'stochastic')
    return ids


def validate_perturbed(path, expected_examples, high_t):
    data, ids = load_pickle(path)
    if len(data) != expected_examples:
        raise ValueError(f'{path} has {len(data)} examples; expected {expected_examples}')
    for qid, example in data.items():
        if not isinstance(example, dict) or not example.get('original_id'):
            raise ValueError(f'{qid} is missing perturbation metadata')
        greedy = example.get('most_likely_answer')
        if (not isinstance(greedy, dict)
                or not isinstance(greedy.get('response'), str)
                or not greedy['response'].strip()):
            raise ValueError(f'{qid} is missing its greedy perturbed answer')
        validate_responses(qid, example.get('responses'), high_t, 'stochastic')

    pattern = re.compile(
        r'^(?P<original>.+)_(?P<perturb>contrast|blur|rotate|shift|noise|masking|bw)\d+_rephrased(?P<rephrase>[1-5])$'
    )
    groups = Counter()
    combinations = {}
    for qid in ids:
        match = pattern.match(qid)
        if not match:
            raise ValueError(f'Perturbed ID {qid!r} does not match the expected schema')
        original = match.group('original')
        groups[original] += 1
        combinations.setdefault(original, Counter())[
            (match.group('perturb'), int(match.group('rephrase')))
        ] += 1
    expected_originals = expected_examples // 35
    if expected_examples % 35 or len(groups) != expected_originals or set(groups.values()) != {35}:
        raise ValueError(
            f'{path} must contain 35 variants for each of {expected_originals} original questions'
        )
    expected_combinations = {
        (perturbation, rephrase)
        for perturbation in ('contrast', 'blur', 'rotate', 'shift', 'noise', 'masking', 'bw')
        for rephrase in range(1, 6)
    }
    for original, counts in combinations.items():
        if set(counts) != expected_combinations or set(counts.values()) != {1}:
            raise ValueError(f'Question {original} does not have all 7 x 5 perturbation/rephrase variants')
    return ids


def validate_accuracy(path, expected_ids):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    labels = payload.get('per_example') if isinstance(payload, dict) else None
    if not isinstance(labels, dict):
        raise ValueError(f'{path} is missing the per_example accuracy dictionary')
    label_ids = set(map(str, labels))
    expected = set(expected_ids)
    if label_ids != expected:
        raise ValueError(
            f'{path} question IDs do not match generations: '
            f'missing={sorted(expected - label_ids)[:3]}, extra={sorted(label_ids - expected)[:3]}'
        )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--vanilla')
    group.add_argument('--perturbed')
    parser.add_argument('--accuracy')
    parser.add_argument('--expected-examples', type=int, required=True)
    parser.add_argument('--high-temp', type=int, default=10)
    parser.add_argument('--low-temp', type=int, default=3)
    args = parser.parse_args()

    if args.vanilla:
        ids = validate_vanilla(
            args.vanilla, args.expected_examples, args.high_temp, args.low_temp
        )
    else:
        ids = validate_perturbed(args.perturbed, args.expected_examples, args.high_temp)
    if args.accuracy:
        validate_accuracy(args.accuracy, ids)
    print(f'Validated {len(ids)} OKVQA records.')


if __name__ == '__main__':
    main()
