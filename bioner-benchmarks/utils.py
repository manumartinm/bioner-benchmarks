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

def read_pubtator_content(pubtator_content: str, chunk_df):
    docs = pubtator_content.split('\n\n')

    data = []

    for id, doc in enumerate(docs):
        doc_id = doc.split('|')[0]
        doc_tokens = chunk_df[chunk_df['sentence_id'] == str(doc_id)]['tokens']

        if doc_tokens.empty:
            continue

        doc_tokens = doc_tokens.tolist()[0]

        doc_labels = []
        entities = []

        doc_lines = doc.split('\n')

        for line in doc_lines:
            if '|t|' in line or '|a|' in line:
                continue

            if not line.strip():
                continue

            line_data = line.split('\t')
            start, end, entity_text, entity_type = int(line_data[1]), int(line_data[2]), line_data[3], line_data[4]
            entities.append((start, end, entity_text, entity_type))

        doc_labels = [0] * len(doc_tokens)

        current_char_pos = 0
        token_positions = []

        for token in doc_tokens:
            token_start = current_char_pos
            token_end = current_char_pos + len(token)
            token_positions.append((token_start, token_end))
            current_char_pos = token_end + 1

        for start, end, entity_text, entity_type in entities:
            for idx, (token_start, token_end) in enumerate(token_positions):
                if token_start >= start and token_end <= end:
                    if token_start == start:
                        doc_labels[idx] = 1
                    else:
                        doc_labels[idx] = 2

        data.append({
            'sentence_id': str(doc_id),
            'tokens': doc_tokens,
            'labels': doc_labels
        })

    return pd.DataFrame(data)

def parse_labels(labels, entity_map):
    if labels is None or not isinstance(labels, (list, np.ndarray)):
        print(labels)
        return []

    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    return [entity_map[label] for label in labels]

def calculate_metrics(true_df, pred_df, entity_map, i):
    true_df['sentence_id'] = true_df['sentence_id'].astype(int)
    pred_df['sentence_id'] = pred_df['sentence_id'].astype(int)

    merged_df = pd.merge(true_df, pred_df, how='outer', on='sentence_id', suffixes=('_true', '_pred'))

    merged_df['parsed_labels_true'] = merged_df['labels_true'].apply(lambda x: parse_labels(x, entity_map))
    merged_df['parsed_labels_pred'] = merged_df['labels_pred'].apply(lambda x: parse_labels(x, entity_map))

    true_labels = merged_df['parsed_labels_true'].tolist()
    pred_labels = merged_df['parsed_labels_pred'].tolist()

    if len(true_labels) == 0 or len(pred_labels) == 0 or list(map(len, true_labels)) != list(map(len, pred_labels)):
        return {
            "classification_report": 0,
            "f1_score": 0,
            "precision": 0,
            "recall": 0,
            "loss": 0
        }

    report = classification_report(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')

    true_labels_flat = [label for sublist in true_labels for label in sublist]
    pred_labels_flat = [label for sublist in pred_labels for label in sublist]

    true_labels_binary = [1 if label != 0 else 0 for label in true_labels_flat]
    pred_labels_binary = [1 if label != 0 else 0 for label in pred_labels_flat]

    pred_probabilities = np.array(pred_labels_binary) * 0.9 + 0.1

    loss = log_loss(true_labels_binary, pred_probabilities, labels=[0, 1])

    return {
        "classification_report": report,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "loss": loss
    }