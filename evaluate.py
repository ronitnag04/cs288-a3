""" 
Based on the official evaluation script for v1.1 of the SQuAD dataset. 
https://github.com/jojonki/qa-squad/blob/master/evaluate-v1.1.py
Modified for Berkeley CS 288 Spring 2026 Assignment 3.
"""
from collections import Counter
import string
import re
import argparse
import json
import os
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metrics(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def evaluate(questions_file, predictions_file):
    with open(questions_file, 'r') as f:
        questions = []
        for line in f:
            question = json.loads(line.strip())
            questions.append(question)
    with open(predictions_file, 'r') as f:
        predictions = [line.strip() for line in f]

    sum_f1 = 0
    sum_precision = 0
    sum_recall = 0
    total = len(questions)

    for question, prediction in zip(questions, predictions):
        ground_truth = question['answer']
        f1, precision, recall = metrics(prediction, ground_truth)
        sum_f1 += f1
        sum_precision += precision
        sum_recall += recall

    average_f1 = sum_f1 / total
    average_precision = sum_precision / total
    average_recall = sum_recall / total

    return {'average_f1': average_f1, 'average_precision': average_precision, 'average_recall': average_recall}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_file", type=str, default='questions/test_questions.txt')
    parser.add_argument("--predictions_file", type=str, default='predictions/test_predictions.txt')
    args = parser.parse_args()

    # Evaluate the predictions
    results = evaluate(args.questions_file, args.predictions_file)
    print(f"Average F1: {results['average_f1']}")
    print(f"Average Precision: {results['average_precision']}")
    print(f"Average Recall: {results['average_recall']}")