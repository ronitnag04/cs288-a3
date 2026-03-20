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

    prediction_metrics = []

    for question, prediction in zip(questions, predictions):
        ground_truth = question['answer']
        question_text = question['question']
        answer_url = question['url']
        f1, precision, recall = metrics(prediction, ground_truth)
        d = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'question': question_text,
            'answer_url': answer_url
        }
        prediction_metrics.append(d)

    return prediction_metrics


def get_average_metrics(prediction_metrics):
    average_f1 = sum(metric['f1'] for metric in prediction_metrics) / len(prediction_metrics)
    average_precision = sum(metric['precision'] for metric in prediction_metrics) / len(prediction_metrics)
    average_recall = sum(metric['recall'] for metric in prediction_metrics) / len(prediction_metrics)
    return average_f1, average_precision, average_recall


def log_mistakes(prediction_metrics, log_path):
    with open(log_path, 'w') as f:
        for i, metric in enumerate(prediction_metrics):
            if metric['f1'] < 1.0:
                question_idx = i + 1
                f1, precision, recall = metric['f1'], metric['precision'], metric['recall']
                ground_truth = metric['ground_truth']
                prediction = metric['prediction']
                question = metric['question']
                answer_url = metric['answer_url']

                f.write(f"=== Question {question_idx} ===\n")
                f.write(f"Question: {question}\n")
                f.write(f"Answer URL: {answer_url}\n")
                f.write(f"Ground Truth: {ground_truth}\n")
                f.write(f"Prediction: {prediction}\n")
                f.write(f"F1: {f1} Precision: {precision} Recall: {recall}\n")
                f.write("\n")
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--questions", type=str, default='questions/test_questions.txt')
    parser.add_argument("-p", "--predictions", type=str, default='predictions/test_predictions.txt')
    parser.add_argument("-l", "--log-mistakes", action='store_true', help='Log mistakes')
    args = parser.parse_args()

    # Evaluate the predictions
    prediction_metrics = evaluate(args.questions, args.predictions)
    average_f1, average_precision, average_recall = get_average_metrics(prediction_metrics)
    print(f"Average F1: {average_f1}")
    print(f"Average Precision: {average_precision}")
    print(f"Average Recall: {average_recall}")
    if args.log_mistakes:
        log_path = os.path.splitext(args.predictions)[0] + '_mistakes.log'
        log_mistakes(prediction_metrics, log_path)
        print(f"Logged mistakes to {log_path}")