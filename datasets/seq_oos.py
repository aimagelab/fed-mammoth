import os
import sys
import requests
import io
from torch.utils.data import Dataset
import pandas as pd
import json
from transformers import T5Tokenizer

from datasets import register_dataset
from datasets.utils import BaseDataset
from utils.global_consts import DATASET_PATH


class MyOOS(Dataset):
    def __init__(self, root: str, train: bool = True, tokenizer: T5Tokenizer = None, download: bool = True) -> None:
        self.root = root
        self.train = train
        self.tokenizer = tokenizer

        if not os.path.exists(self.root + "/OOS") and download:
            print("Downloading OOS...", file=sys.stderr)
            r = requests.get("https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json")
            os.makedirs(self.root + "/OOS", exist_ok=True)
            file_path = os.path.join(self.root, "OOS/oos.json")
            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(r, json_file, ensure_ascii=False, indent=4)

            print("Done", file=sys.stderr)

        self.data_split = pd.DataFrame(
            json.load(open(self.root + "/OOS/oos.json", "r"))["train" if self.train == True else "test"]
        )

        self.data = self.data_split[0].values
        self.targets = self.data_split[1].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        query, target = self.data[index], self.targets[index]

        return query, target


@register_dataset("seq-oos")
class SequentialOOS(BaseDataset):
    N_CLASSES_PER_TASK = 15
    N_TASKS = 10

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
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

        self._split_fcil(num_clients, partition_mode, distribution_alpha, class_quantity)

        for split in ["train", "test"]:
            getattr(self, f"{split}_dataset").data = None
            getattr(self, f"{split}_dataset").targets = None
