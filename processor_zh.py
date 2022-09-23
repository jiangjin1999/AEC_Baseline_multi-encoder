"""Base Data Processors"""
from __future__ import annotations
import os
from abc import ABC
from typing import List
import json
from boto import config
from torch.utils.data import Dataset, Subset, DataLoader
import pandas as pd
from dataclasses import dataclass

from wandb import Config


@dataclass
class TextInputExample:
    """
    Input Example for a single example
    """
    # utt: str = ""
    rec: str = ""
    lab: str = ""


class DataProcessor(ABC):
    """Abstract Data Processor Class which handle the different corpus"""

    def get_train_dataset(self) -> Dataset:
        """get_train_dataset
        """
        raise NotImplementedError

    def get_test_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_dev_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_labels(self) -> List[str]:
        pass

    def get_train_labels(self) -> List[str]:
        return self.get_labels()

    def get_test_labels(self) -> List[str]:
        return self.get_labels()

    def get_dev_labels(self) -> List[str]:
        return self.get_labels()


class TextDataProcessor(DataProcessor):
    def __init__(self, data_dir, config) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.dataset = config.current_dataset
    
    def _read(self, file: str) -> List[TextInputExample]:
        with open(file, 'r', encoding='utf-8') as f:
            data = f.readlines()
            example = [TextInputExample(item.strip().split(' ')[0], item.strip().split(' ')[1]) for item in data]
            # return example
            return example[0:100]
    
    def _load_dataset(self, mode: str = 'train.txt') -> Dataset:
        file = os.path.join(self.data_dir, mode)
        examples = self._read(file)
        indices = [i for i in range(len(examples))] 
        return Subset(examples, indices) 

    def get_train_dataset(self) -> Dataset:
        return self._load_dataset(self.dataset+'/'+self.dataset+'_train.txt')
        
    def get_dev_dataset(self) -> Dataset:
        return self._load_dataset(self.dataset+'/'+self.dataset+'_dev.txt')
        
    def get_test_dataset(self) -> Dataset:
        return self._load_dataset(self.dataset+'/'+self.dataset+'_test.txt')



    