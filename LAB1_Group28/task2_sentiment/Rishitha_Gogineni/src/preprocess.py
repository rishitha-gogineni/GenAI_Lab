import argparse
import json
from collections import Counter

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from text_utils import tokenize
from utils import load_config, project_root


def build_vocab(counter, vocab_size):
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token, _ in counter.most_common(vocab_size - 2):
        vocab[token] = len(vocab)

    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/data_config.json",
    )
    args = parser.parse_args()

    root = project_root()
    config = load_config(root / args.config)

    dataset = load_dataset(config["dataset_name"])
    train_data = dataset["train"]
    test_data = dataset["test"]

    vocab_limit = config["preprocessing"]["vocab_size"]
    max_length = config["preprocessing"]["max_length"]
    val_fraction = config["preprocessing"]["validation_fraction"]
    seed = config["seed"]

    usable_indices = []
    usable_labels = []
    cleaned_empty = 0

    for index, (text, label) in enumerate(
        zip(train_data["text"], train_data["label"])
    ):
        if tokenize(text):
            usable_indices.append(index)
            usable_labels.append(label)
        else:
            cleaned_empty += 1

    train_idx, val_idx = train_test_split(
        usable_indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=usable_labels,
    )

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)

    counter = Counter()
    total_tokens = 0
    reviews_over_max = 0
    tokens_removed = 0

    for index in train_idx:
        tokens = tokenize(train_data[int(index)]["text"])

        counter.update(tokens)
        total_tokens += len(tokens)

        if len(tokens) > max_length:
            reviews_over_max += 1
            tokens_removed += len(tokens) - max_length

    vocab = build_vocab(counter, vocab_limit)

    covered_tokens = sum(
        count
        for token, count in counter.items()
        if token in vocab
    )

    summary = {
        "original_train_reviews": len(train_data),
        "test_reviews": len(test_data),
        "cleaned_empty_reviews": cleaned_empty,
        "usable_reviews": len(usable_indices),
        "train_reviews": len(train_idx),
        "validation_reviews": len(val_idx),
        "validation_fraction": val_fraction,
        "validation_seed": seed,
        "vocabulary_source": "training_subset_only",
        "unique_training_tokens": len(counter),
        "requested_vocab_size": vocab_limit,
        "final_vocab_size": len(vocab),
        "vocab_coverage_percent": (
            covered_tokens / max(total_tokens, 1) * 100
        ),
        "max_length": max_length,
        "train_reviews_over_max_length": reviews_over_max,
        "train_reviews_over_max_length_percent": (
            reviews_over_max / max(len(train_idx), 1) * 100
        ),
        "train_tokens_removed_by_truncation_percent": (
            tokens_removed / max(total_tokens, 1) * 100
        ),
        "stopword_removal": False,
        "stemming": False,
        "lemmatization": False,
        "negation_preserved": True,
    }

    out = root / "data_processed"
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "split_indices.npz",
        train_indices=train_idx,
        val_indices=val_idx,
    )

    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    with open(
        out / "preprocessing_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    preprocessing_config = {
        "vocab_size": len(vocab),
        "max_length": max_length,
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "lowercase": True,
        "remove_urls": True,
        "normalize_whitespace": True,
        "expand_contractions": True,
        "preserve_negation": True,
        "stopword_removal": False,
        "stemming": False,
        "lemmatization": False,
        "vocabulary_source": "training_subset_only",
        "split_file": "split_indices.npz",
    }

    with open(
        out / "preprocessing_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(preprocessing_config, f, indent=2)

    print("Yelp Polarity preprocessing summary")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print()
    print(f"Saved: {out / 'vocab.json'}")
    print(f"Saved: {out / 'split_indices.npz'}")
    print(f"Saved: {out / 'preprocessing_summary.json'}")
    print(f"Saved: {out / 'preprocessing_config.json'}")


if __name__ == "__main__":
    main()
