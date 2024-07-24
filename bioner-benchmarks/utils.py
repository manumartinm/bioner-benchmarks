import os
from typing import List
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np
from sklearn.metrics import log_loss
import pandas as pd

def create_directories(dirs: List[str]) -> None:
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def save_to_file(filepath: str, content: List[str]) -> None:
    with open(filepath, 'w') as f:
        f.write('\n'.join(content))

def read_pubtator_file(file_path: str):
    with open(file_path, 'r') as f:
        docs = f.read().split('\n\n')
        data = []
        key_chars = ['t', 'a']

        for doc in docs:
            doc_lines = doc.split('\n')
            for line in doc_lines:
                if any([f'|{key_char}|'in line for key_char in key_chars]):
                    continue

                line_data = line.split('\t')
                if len(line_data) == 1:
                    continue

                sentence_id, entity_start, entity_end, entity_text, entity_type = line_data

                data.append({
                    'sentence_id': sentence_id,
                    'words': entity_text,
                    'labels': entity_type
                })

    return pd.DataFrame(data)

def calculate_metrics(true_df, pred_df):
    true_labels = true_df.groupby('sentence_id')['labels'].apply(list).tolist()
    pred_labels = pred_df.groupby('sentence_id')['labels'].apply(list).tolist()

    report = classification_report(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)

    true_labels_flat = [label for sublist in true_labels for label in sublist]
    pred_labels_flat = [label for sublist in pred_labels for label in sublist]

    true_labels_binary = [1 if label != "O" else 0 for label in true_labels_flat]
    pred_labels_binary = [1 if label != "O" else 0 for label in pred_labels_flat]

    pred_probabilities = np.array(pred_labels_binary) * 0.9 + 0.1

    loss = log_loss(true_labels_binary, pred_probabilities)

    return {
        "classification_report": report,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "loss": loss
    }