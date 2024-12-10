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



class MyOOS(Dataset):
    def __init__(self, root: str, train: bool = True, tokenizer: T5Tokenizer = None, download: bool = True) -> None:
        self.root = root
        self.train = train
        self.tokenizer = tokenizer

        if not os.path.exists(self.root + "/OOS") and download:
            print("Downloading OOS...", file=sys.stderr)
            r = requests.get("https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json")
            print(r.status_code)
            if r.status_code == 200:
                os.makedirs(self.root + "/OOS", exist_ok=True)
                file_path = os.path.join(self.root, "OOS/oos.json")
                with open(file_path, "w", encoding="utf-8") as json_file:
                    json.dump(r.json(), json_file, ensure_ascii=False, indent=4)

            print("Done", file=sys.stderr)

        self.data_split = pd.DataFrame(
            json.load(open(self.root + "/OOS/oos.json", "r"))["train" if self.train == True else "test"]
        )  # TODO non capisco come mai il modulo json rogna. Intanto per risolvere altre cose uso un altro path
        # self.data_split = pd.DataFrame(
        # json.load(open("C:\Riccardo\Dottorato\DL practice\lettura dataset\dataset\oos\oos.json", "r"))[
        # "train" if self.train == True else "test"
        # ]
        # )

        texts = self.data_split.iloc[:, 0].tolist()  # converte i dati in una lista di stringhe da passare al tokenizer
        labels = self.data_split.iloc[:, 1].tolist()
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


@register_dataset("seq-oos")
class SequentialOOS(BaseDataset):
    N_CLASSES_PER_TASK = 15
    N_TASKS = 10
    IS_TEXT = True

    # tokenizer = AutoTokenizer.from_pretrained("textattack/t5-small-imdb")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    INPUT_SHAPE = 49

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
        train_transform: str = None, #I put them anyway to avoid possible errors, but they won't be used
        test_transform: str = None,
        partition_mode: str = "distribution",
        distribution_alpha: float = 0.5,
        class_quantity: int = 1,
    ):
        super().__init__(num_clients, batch_size, partition_mode, distribution_alpha, class_quantity)

        for split in ["train", "test"]:
            dataset = MyOOS(
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


@register_dataset("seq-oos_5")
class SequentialOOS5(SequentialOOS):
    N_CLASSES_PER_TASK = 30
    N_TASKS = 5

@register_dataset("seq-oos_3")
class SequentialOOS5(SequentialOOS):
    N_CLASSES_PER_TASK = 50
    N_TASKS = 3


@register_dataset("joint-oos")
class JointOOS(SequentialOOS):
    N_CLASSES_PER_TASK = 150
    N_TASKS = 1
