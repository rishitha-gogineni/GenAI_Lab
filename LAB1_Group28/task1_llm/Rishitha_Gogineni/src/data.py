import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CharTokenizer:
    def __init__(self, chars):
        self.chars = sorted(chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, ids):
        return "".join(self.idx_to_char[int(i)] for i in ids)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "char_to_idx.json", "w", encoding="utf-8") as f:
            json.dump(self.char_to_idx, f, ensure_ascii=False, indent=2)
        with open(folder / "idx_to_char.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.idx_to_char.items()}, f, ensure_ascii=False, indent=2)


class CharSequenceDataset(Dataset):
    def __init__(self, encoded, context_length, stride):
        self.data = torch.tensor(encoded, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride
        usable = len(self.data) - context_length
        self.count = max(0, usable // stride)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start:start + self.context_length]
        y = self.data[start + 1:start + self.context_length + 1]
        return x, y


def load_tinystories(config, root, smoke_test=False):
    from datasets import load_dataset

    name = config["data"]["dataset_name"]
    ds = load_dataset(name)
    train_count = 200 if smoke_test else config["data"]["train_stories"]
    val_count = 50 if smoke_test else config["data"]["val_stories"]
    seed = config["seed"]

    train_all = ds["train"].shuffle(seed=seed)
    train_split = train_all.select(range(train_count))
    train_indices = list(range(train_count))

    if "validation" in ds:
        val_all = ds["validation"].shuffle(seed=seed)
        val_split = val_all.select(range(val_count))
        val_indices = list(range(val_count))
        val_source = "validation"
    else:
        val_split = train_all.select(range(train_count, train_count + val_count))
        val_indices = list(range(train_count, train_count + val_count))
        val_source = "train"

    split_info = {
        "seed": seed,
        "train_source": "train",
        "validation_source": val_source,
        "train_count": train_count,
        "validation_count": val_count,
        "train_selected_positions": train_indices,
        "validation_selected_positions": val_indices,
    }
    out = root / "data_processed"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    train_text = "\n".join(train_split["text"])
    val_text = "\n".join(val_split["text"])
    return train_text, val_text


def build_data(config, root, smoke_test=False):
    train_text, val_text = load_tinystories(config, root, smoke_test)
    chars = set(train_text) | set(val_text)
    tokenizer = CharTokenizer(chars)
    tokenizer.save(root / "data_processed")

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    context = config["data"]["context_length"]
    stride = config["data"].get("sequence_stride", context)
    train_set = CharSequenceDataset(train_ids, context, stride)
    val_set = CharSequenceDataset(val_ids, context, stride)
    return tokenizer, train_set, val_set
