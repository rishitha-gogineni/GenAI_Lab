import argparse
import csv
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from data import check_split, get_label_column, get_text_column, load_yelp
from utils import load_config, make_dirs, project_root, set_seed


def word_lengths(texts):
    return np.fromiter((len(text.split()) for text in texts), dtype=np.int32)


def length_stats(lengths):
    return {
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "std": float(lengths.std()),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }


def class_stats(labels):
    counts = Counter(labels)
    total = len(labels)
    return {
        "negative_count": counts.get(0, 0),
        "positive_count": counts.get(1, 0),
        "negative_percent": 100 * counts.get(0, 0) / max(total, 1),
        "positive_percent": 100 * counts.get(1, 0) / max(total, 1),
    }


def save_class_plot(train_labels, path):
    counts = Counter(train_labels)
    names = ["Negative", "Positive"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    plt.figure(figsize=(6, 4))
    plt.bar(names, values)
    plt.title("Yelp Polarity Class Distribution")
    plt.ylabel("Reviews")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_length_plot(lengths, max_words, path):
    shown = lengths[lengths <= max_words]
    plt.figure(figsize=(8, 5))
    plt.hist(shown, bins=60)
    plt.title("Training Review Length Distribution")
    plt.xlabel("Words per review")
    plt.ylabel("Reviews")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_summary_csv(summary, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "metric", "value"])
        for section, values in summary.items():
            if isinstance(values, dict):
                for metric, value in values.items():
                    writer.writerow([section, metric, value])
            else:
                writer.writerow(["dataset", section, values])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data_config.json")
    args = parser.parse_args()

    root = project_root()
    config = load_config(root / args.config)
    set_seed(config["seed"])
    make_dirs(root)

    dataset = load_yelp(config)
    train = dataset["train"]
    test = dataset["test"]

    text_col = get_text_column(train)
    label_col = get_label_column(train)

    train_checks = check_split(train, text_col, label_col)
    test_checks = check_split(test, text_col, label_col)

    train_labels = train[label_col]
    test_labels = test[label_col]
    train_lengths = word_lengths(train[text_col])
    test_lengths = word_lengths(test[text_col])

    summary = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_classes": class_stats(train_labels),
        "test_classes": class_stats(test_labels),
        "train_review_length_words": length_stats(train_lengths),
        "test_review_length_words": length_stats(test_lengths),
        "train_data_checks": train_checks,
        "test_data_checks": test_checks,
    }

    output_json = root / "outputs" / "eda_summary.json"
    output_csv = root / "outputs" / "eda_summary.csv"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_summary_csv(summary, output_csv)

    save_class_plot(
        train_labels,
        root / "outputs" / "plots" / "class_distribution.png",
    )
    save_length_plot(
        train_lengths,
        config["eda"]["histogram_max_words"],
        root / "outputs" / "plots" / "review_length_distribution.png",
    )

    print("Yelp Polarity EDA")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print()
    print("Training class distribution")
    for key, value in summary["train_classes"].items():
        print(f"{key}: {value}")
    print()
    print("Training review length in words")
    for key, value in summary["train_review_length_words"].items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    print()
    print("Training data checks")
    for key, value in train_checks.items():
        print(f"{key}: {value}")
    print()
    print(f"Saved: {output_json.relative_to(root)}")
    print(f"Saved: {output_csv.relative_to(root)}")
    print("Saved: outputs/plots/class_distribution.png")
    print("Saved: outputs/plots/review_length_distribution.png")


if __name__ == "__main__":
    main()
