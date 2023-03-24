from torch.utils.data import Dataset
import torch
from typing import Dict
import collections
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, df, mode, cfg):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.cfg = cfg

        self.df[self.cfg.dataset.label_columns] = LabelEncoder().fit_transform(
            self.df[self.cfg.dataset.label_columns]
        )

        if self.cfg.training.loss_function == "ArcFaceLossAdaptiveMargin":
            tmp = self.df[self.cfg.dataset.label_columns].value_counts().sort_index().values
            tt = 1/np.log1p(tmp)
            tt = (tt - tt.min()) / (tt.max() - tt.min())
            self.cfg.dataset._margins = tt * self.cfg.training.arcface_margin + 0.05

        self.labels = self.df[self.cfg.dataset.label_columns] 

        self.cfg.dataset._num_labels = self.df[self.cfg.dataset.label_columns].nunique(dropna=False)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.architecture.backbone)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.sep_token is None:
            self.tokenizer.sep_token = " "

        self.text = self.get_texts(self.df, self.cfg, self.tokenizer.sep_token)
        self.cfg._tokenizer_sep_token = self.tokenizer.sep_token

        self.cfg._tokenizer_cls_token_id = self.tokenizer.cls_token_id
        self.cfg._tokenizer_sep_token_id = self.tokenizer.sep_token_id

        for j in range(2):
            print(f"Example of text tokens: {self.tokenizer.decode(self.tokenizer(self.text[j], padding='max_length', truncation=True, max_length=self.cfg.tokenizer.max_length)['input_ids'])}")

    def batch_to_device(batch, device):

        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, collections.abc.Mapping):
            return {
                key: CustomDataset.batch_to_device(value, device)
                for key, value in batch.items()
            }
        elif isinstance(batch, collections.abc.Sequence):
            return [CustomDataset.batch_to_device(value, device) for value in batch]
        else:
            raise ValueError(f"Can not move {type(batch)} to device.")

    def get_texts(cls, df, cfg, separator, processed=False):
        columns = cfg.dataset.text_column
        columns = [x for x in columns if x != "None"]

        for c in columns:

            if "tude" in c:
                if not processed:
                    df[c] = df[c].astype(float).astype(str)

                    df[c] = df[c].apply(lambda x: f" ".join([xx if jj == 0 else " ".join(["".join(x) for x in list(zip(*[iter(xx[:cfg.dataset.coordinate_digits])]*3))]) for jj,xx in enumerate(x.split("."))]))

            df[c] = df[c].fillna(" ").astype(str).fillna(" ")

            if hasattr(cfg, "tokenizer") and cfg.tokenizer.lowercase:
                df[c] = df[c].str.lower()

        if "All [CLS]" in cfg.architecture.pool:
            join_str = f" {separator} "
        elif "xlm-roberta" in cfg.architecture.backbone:
            join_str = f" {separator}{separator} "
        else:
            join_str = f" {separator} "
        texts = df[columns].apply(lambda x: join_str.join(x), axis=1).values

        return texts

    def _read_data(self, idx: int, sample):

        text = self.text[idx]

        sample["target"] = self.labels[idx]

        sample.update(self.encode(text))
        return sample
    
    def encode(self, text: str):
        sample = dict()
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.tokenizer.max_length,
        )
        sample["input_ids"] = encodings["input_ids"][0]
        sample["attention_mask"] = encodings["attention_mask"][0]
        return sample

    def __getitem__(self, idx):
        sample: Dict = dict()

        sample = self._read_data(idx=idx, sample=sample)

        return sample

    def __len__(self):
        return self.df.shape[0]
