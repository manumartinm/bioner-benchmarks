from datasets import load_dataset
from typing import List
import pandas as pd

class NERBenchmarkDataWrangler:
    def __init__(self, dataset_name: str, datasets: dict):
        self.dataset_name = datasets[dataset_name]['dataset_name']
        self.parse_function = datasets[dataset_name]['parse_function']
        self.to_pubtator_fn = datasets[dataset_name]['to_pubtator']
        self.entity_map = datasets[dataset_name].get('entity_map', None)
        self.dataset = load_dataset(self.dataset_name, trust_remote_code=True)

        if 'train' not in self.dataset:
            raise ValueError('Dataset does not have a train split')

        self._split_dataset()

    def _split_dataset(self):
        if 'test' not in self.dataset and 'validation' in self.dataset:
            self._split_validation_only()
        elif 'test' not in self.dataset and 'validation' not in self.dataset:
            self._split_train_only()
        elif 'test' in self.dataset and 'validation' not in self.dataset:
            self._split_train_and_test()
        else:
            self._load_all_splits()

    def _split_validation_only(self):
        splited_data = self.dataset['validation'].train_test_split(test_size=0.3)
        self.train_df = self.dataset['train'].to_pandas()
        self.test_df = splited_data['test'].to_pandas()
        self.valid_df = splited_data['train'].to_pandas()

    def _split_train_only(self):
        splited_data = self.dataset['train'].train_test_split(test_size=0.3)
        val_splited_data = splited_data['test'].train_test_split(test_size=0.5)
        self.train_df = splited_data['train'].to_pandas()
        self.test_df = val_splited_data['train'].to_pandas()
        self.valid_df = val_splited_data['test'].to_pandas()

    def _split_train_and_test(self):
        splited_data = self.dataset['train'].train_test_split(test_size=0.3)
        self.train_df = self.dataset['train'].to_pandas()
        self.test_df = self.dataset['test'].to_pandas()
        self.valid_df = splited_data['test'].to_pandas()

    def _load_all_splits(self):
        self.train_df = self.dataset['train'].to_pandas()
        self.test_df = self.dataset['test'].to_pandas()
        self.valid_df = self.dataset['validation'].to_pandas()

    def get_format_data(self, type: str) -> pd.DataFrame:
        df = self._get_dataframe_by_type(type)
        return self.parse_function(df, self.entity_map)

    def to_pubtator(self, type: str) -> List[str]:
        df = self._get_dataframe_by_type(type)
        return self.to_pubtator_fn(df, self.entity_map)

    def get_labels(self) -> List[str]:
        parsed_data = self.get_format_data('train')
        return parsed_data['labels'].unique().tolist()

    def _get_dataframe_by_type(self, type: str) -> pd.DataFrame:
        if type == 'train':
            return self.train_df
        elif type == 'test':
            return self.test_df
        elif type == 'valid':
            return self.valid_df
        else:
            raise ValueError(f"Invalid data type: {type}")