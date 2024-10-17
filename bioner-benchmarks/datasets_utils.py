import pandas as pd
from typing import Dict, List, Optional
from transformers import AutoTokenizer

def from_hf_tokens_to_valid(df: pd.DataFrame, entity_map: Optional[Dict[int, str]]) -> pd.DataFrame:
  if entity_map is None:
     raise ValueError('Entity map is required for this dataset')

  data = []

  for index, row in df.iterrows():
      tokens = row['tokens']
      ner_tags = row['ner_tags']
      sentence_id = row['id']

      for i, token in enumerate(tokens):
          new_row = {
              'words': token,
              'labels': entity_map[ner_tags[i]],
              'sentence_id': sentence_id
          }
          data.append(new_row)

  format_df = pd.DataFrame(data)

  return format_df

def parse_bcdr_to_format(df: pd.DataFrame, entity_map: Optional[Dict[int, str]], model: Optional[str]) -> pd.DataFrame:
  new_data = []
  checkpoint_path = model if model else "NeuML/pubmedbert-base-embeddings"

  tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True, do_lower_case=True)

  for index, row in df.iterrows():
    for passage in row['passages']:
      text = passage['text']
      entities = passage['entities']
      tokens = tokenizer.tokenize(text)
      token_offsets = tokenizer(text, return_offsets_mapping=True)['offset_mapping']

      labels = ['O'] * len(token_offsets)
      for entity in entities:
          entity_offsets = entity['offsets'][0]
          start_offset, end_offset = entity_offsets

          for i, (start, end) in enumerate(token_offsets):
              if start >= start_offset and end <= end_offset:
                  if labels[i] == 'O' and ((i - 1) < 0 or labels[i - 1] == 'B'):
                      labels[i] = 'B-' + entity['type']
                  else:
                      labels[i] = 'I-' + entity['type']

      for i, token in enumerate(tokens):
          new_data.append({
            'words': token,
            'labels': labels[i],
            'sentence_id': passage['document_id']
          })

  format_df = pd.DataFrame(new_data)

  return format_df


def from_hf_to_pubtator(df: pd.DataFrame, entity_map: Optional[Dict[int, str]], add_entities: Optional[bool] = True) -> List[str]:
    data = []
    for index, row in df.iterrows():
        tokens = row['tokens']
        ner_tags = row['ner_tags']
        sentence_id = row['id']
        text = ' '.join(tokens)
        line = f'{sentence_id}|t|\n'
        line += f'{sentence_id}|a|{text}\n'

        if not add_entities:
            data.append(line)
            continue

        entity = ""
        start = None
        for i, (ner_tag, token) in enumerate(zip(ner_tags, tokens)):
            entity_label = entity_map[ner_tag]
            if entity_label.startswith("B-"):
                if entity:
                    line += f'{sentence_id}\t{start}\t{i}\t{entity}\t{entity_label[2:]}\n'
                entity = token
                start = i
            elif entity_label.startswith("I-") and entity:
                entity += " " + token
            else:
                if entity:
                    line += f'{sentence_id}\t{start}\t{i}\t{entity}\t{entity_label[2:]}\n'
                    entity = ""
                    start = None

        if entity:
            line += f'{sentence_id}\t{start}\t{i+1}\t{entity}\t{entity_label[2:]}\n'

        data.append(line)

    return data

def parse_bcdr_to_pubtator(df: pd.DataFrame, entity_map: Optional[Dict[int, str]] = None, add_entities: Optional[bool] = True) -> List[str]:
    new_data = []

    for index, row in df.iterrows():
        document_id = None
        document_lines = []
        document_entities = []

        for passage in row['passages']:
            document_id = passage['document_id']
            text = passage['text']
            type = passage['type']
            document_lines.append(f'{document_id}|{type[0].lower()}|{text}\n')

            if not add_entities:
                continue

            for entity in passage.get('entities', []):
                entity_offsets = entity['offsets'][0]
                start_offset = entity_offsets[0]
                end_offset = entity_offsets[1]
                entity_text = entity['text'][0]
                entity_type = entity['type']

                document_entities.append(f'{document_id}\t{start_offset}\t{end_offset}\t{entity_text}\t{entity_type}\n')

        new_data.append(''.join(document_lines + document_entities))

    return new_data

entity_map = {
    0: 'O',
    1: 'I-Entity',
    2: 'B-Entity'
}

jnlpba_mapping = {
    0: 'O',
    1: 'B-DNA',
    2: 'I-DNA',
    3: 'B-RNA',
    4: 'I-RNA',
    5: 'B-cell_line',
    6: 'I-cell_line',
    7: 'B-cell_type',
    8: 'I-cell_type',
    9: 'B-protein',
    10: 'I-protein'
}

datasets_map = {
    "ncbi": {
        "dataset_name": "ncbi_disease",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator,
        "entity_map": entity_map,
    },
    "bc2gm": {
        "dataset_name": "spyysalo/bc2gm_corpus",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator,
        "entity_map": entity_map,
    },
    "linnaeus": {
        "dataset_name": "cambridgeltl/linnaeus",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator,
        "entity_map": entity_map,
    },
    "bc5dr": {
        "dataset_name": "bigbio/bc5cdr",
        "parse_function": parse_bcdr_to_format,
        "to_pubtator": parse_bcdr_to_pubtator,
    },
    "jnlpba": {
        "dataset_name": "jnlpba/jnlpba",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator,
        "entity_map": jnlpba_mapping
    }
}