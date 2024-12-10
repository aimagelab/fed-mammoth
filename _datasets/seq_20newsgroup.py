import os
import sys
import requests
from torch.utils.data import Dataset
import pandas as pd
import json
from transformers import T5Tokenizer, AutoTokenizer
import numpy as np

from _datasets import register_dataset
from _datasets._utils import BaseDataset
from utils.global_consts import DATASET_PATH
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.datasets import fetch_20newsgroups


class MyNewsGroup(Dataset):
    def __init__(self, root: str, train: bool = True, tokenizer: T5Tokenizer = None, download: bool = True) -> None:
        self.root = root
        self.train = train
        self.tokenizer = tokenizer

        #if not os.path.exists(self.root + "/20NewsGroup") and download:
        #    print("Downloading 20NewsGroup...", file=sys.stderr)
        #    data = fetch_20newsgroups(subset="train" if train else "test", data_home=self.root + "/20NewsGroup", remove=('headers', 'footers', 'quotes'))
        #    print("Done", file=sys.stderr)

        data = fetch_20newsgroups(subset="train" if train else "test", data_home=self.root + "/20NewsGroup", remove=('headers', 'footers', 'quotes'))
        texts = data['data']  # converte i dati in una lista di stringhe da passare al tokenizer
        labels = data['target']
        label_mapping = {
            label: idx for idx, label in enumerate(sorted(set(labels)))
        }  # converte le label testuali in label numeriche necessarie per la testa di classificazione di T5

        self.data = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        self.targets = np.array([label_mapping[label] for label in labels], dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        input_ids = self.data["input_ids"][index]
        attamat = self.data["attention_mask"][index]
        query = {"input_ids": input_ids, "attention_mask": attamat}
        query, target = BatchEncoding(query), self.targets[index]

        return query, target
    

@register_dataset("seq-20ng")
class Sequential20NewsGroup(BaseDataset):
    N_CLASSES_PER_TASK = 2
    N_TASKS = 10
    IS_TEXT = True

    # tokenizer = AutoTokenizer.from_pretrained("textattack/t5-small-imdb")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    INPUT_SHAPE = 512

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
        train_transform = None, #I put them anyway to avoid possible errors, but they won't be used
        test_transform = None,
        partition_mode: str = "distribution",
        distribution_alpha: float = 0.5,
        class_quantity: int = 1,
    ):
        super().__init__(num_clients, batch_size, partition_mode, distribution_alpha, class_quantity)

        for split in ["train", "test"]:
            dataset = MyNewsGroup(
                DATASET_PATH,
                train=True if split == "train" else False,
                tokenizer=getattr(self, "tokenizer"),
                download=True,
            )
            setattr(self, f"{split}_dataset", dataset)

        self._split_fcil_OOS(num_clients, partition_mode, distribution_alpha, class_quantity)

        for split in ["train", "test"]:
            getattr(self, f"{split}_dataset").data = None
            getattr(self, f"{split}_dataset").targets = None

    def train_transform(self, x):
        return x
    
    def test_transform(self, x):
        return x


@register_dataset("seq-20ng_4")
class Sequential20NewsGroup_4(Sequential20NewsGroup):
    N_CLASSES_PER_TASK = 5
    N_TASKS = 4

@register_dataset("joint-20ng")
class JointOOS(Sequential20NewsGroup):
    N_CLASSES_PER_TASK = 20
    N_TASKS = 1