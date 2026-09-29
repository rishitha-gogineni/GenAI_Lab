import json

import numpy as np
import torch
from torch.utils.data import Dataset

from text_utils import tokenize


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_indices(path):
    data = np.load(path)
    return (
        data["train_indices"].astype(np.int64),
        data["val_indices"].astype(np.int64),
    )


class YelpDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.pad_id = vocab.get("<PAD>", 0)
        self.unk_id = vocab.get("<UNK>", 1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        tokens = tokenize(self.texts[index])
        ids = [
            self.vocab.get(token, self.unk_id)
            for token in tokens[:self.max_length]
        ]

        if len(ids) < self.max_length:
            ids.extend(
                [self.pad_id] * (self.max_length - len(ids))
            )

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(
                self.labels[index],
                dtype=torch.float32,
            ),
        )


def subset_from_indices(dataset_split, indices):
    texts = []
    labels = []

    for index in indices:
        row = dataset_split[int(index)]
        texts.append(row["text"])
        labels.append(row["label"])

    return texts, labels


def remove_cleaned_empty(texts, labels):
    kept_texts = []
    kept_labels = []
    removed = 0

    for text, label in zip(texts, labels):
        if tokenize(text):
            kept_texts.append(text)
            kept_labels.append(label)
        else:
            removed += 1

    return kept_texts, kept_labels, removed
