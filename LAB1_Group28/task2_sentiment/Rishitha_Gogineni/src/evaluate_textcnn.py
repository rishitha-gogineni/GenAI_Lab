import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from dataset import YelpDataset, load_vocab, remove_cleaned_empty
from model_textcnn import TextCNN
from text_utils import tokenize
from utils import load_config, make_dirs, project_root, set_seed


def expected_calibration_error(y_true, probabilities, bins=15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for i in range(bins):
        left = edges[i]
        right = edges[i + 1]

        if i == bins - 1:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)

        if not np.any(mask):
            continue

        confidence = probabilities[mask].mean()
        accuracy = y_true[mask].mean()
        ece += mask.mean() * abs(accuracy - confidence)

    return float(ece)


def bootstrap_intervals(y_true, y_pred, samples, seed):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    accuracy_values = []
    macro_f1_values = []
    mcc_values = []

    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        yt = y_true[indices]
        yp = y_pred[indices]

        accuracy_values.append(accuracy_score(yt, yp))
        macro_f1_values.append(
            f1_score(yt, yp, average="macro", zero_division=0)
        )
        mcc_values.append(matthews_corrcoef(yt, yp))

    def interval(values):
        low, high = np.percentile(values, [2.5, 97.5])
        return float(low), float(high)

    return {
        "accuracy": interval(accuracy_values),
        "macro_f1": interval(macro_f1_values),
        "mcc": interval(mcc_values),
    }


def classification_metrics(y_true, y_pred, probabilities):
    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )
    precision_micro, recall_micro, f1_micro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, probabilities),
        "ece": expected_calibration_error(
            y_true,
            probabilities,
            bins=15,
        ),
    }


def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)

    ax.set_title("TextCNN Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Negative", "Positive"])
    ax.set_yticks([0, 1], labels=["Negative", "Positive"])

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true, probabilities, path):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc_value = roc_auc_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("TextCNN ROC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_pr_curve(y_true, probabilities, path):
    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )
    auc_value = average_precision_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {auc_value:.4f}")
    ax.set_title("TextCNN Precision Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_slice_table(texts, y_true, y_pred):
    rows = []

    lengths = np.array([len(tokenize(text)) for text in texts])
    negation_terms = {"not", "no", "never", "cannot"}

    has_negation = np.array([
        any(token in negation_terms for token in tokenize(text))
        for text in texts
    ])

    slices = {
        "length_short_0_50": lengths <= 50,
        "length_medium_51_200": (lengths > 50) & (lengths <= 200),
        "length_long_201_plus": lengths > 200,
        "negation_present": has_negation,
        "negation_absent": ~has_negation,
    }

    for name, mask in slices.items():
        count = int(mask.sum())

        if count == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        rows.append({
            "slice": name,
            "count": count,
            "macro_f1": f1_score(
                yt,
                yp,
                average="macro",
                zero_division=0,
            ),
            "error_rate": 1.0 - accuracy_score(yt, yp),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/textcnn_config.json",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    root = project_root()
    config = load_config(root / args.config)

    make_dirs(root)
    set_seed(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = root / args.checkpoint

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    run_id = state["run_id"]

    dataset = load_dataset(config["data"]["dataset_name"])
    test_data = dataset["test"]

    texts = list(test_data["text"])
    labels = list(test_data["label"])

    texts, labels, removed = remove_cleaned_empty(
        texts,
        labels,
    )

    vocab = load_vocab(
        root / "data_processed" / "vocab.json"
    )

    model_config = config["model"]

    test_set = YelpDataset(
        texts,
        labels,
        vocab,
        config["data"]["max_length"],
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=model_config["embedding_dim"],
        num_filters=model_config["num_filters"],
        kernel_sizes=model_config["kernel_sizes"],
        dropout=model_config["dropout"],
        pad_id=vocab.get("<PAD>", 0),
    ).to(device)

    model.load_state_dict(state["model"])
    model.eval()

    probabilities = []
    true_labels = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)

            logits = model(x)
            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )
            true_labels.extend(
                y.numpy().tolist()
            )

    evaluation_time = time.time() - start

    y_true = np.asarray(
        true_labels,
        dtype=np.int64,
    )
    probs = np.asarray(
        probabilities,
        dtype=np.float64,
    )
    y_pred = (probs >= 0.5).astype(np.int64)

    metrics = classification_metrics(
        y_true,
        y_pred,
        probs,
    )

    intervals = bootstrap_intervals(
        y_true,
        y_pred,
        samples=args.bootstrap_samples,
        seed=config["seed"],
    )

    metrics["accuracy_ci95_low"] = intervals["accuracy"][0]
    metrics["accuracy_ci95_high"] = intervals["accuracy"][1]
    metrics["macro_f1_ci95_low"] = intervals["macro_f1"][0]
    metrics["macro_f1_ci95_high"] = intervals["macro_f1"][1]
    metrics["mcc_ci95_low"] = intervals["mcc"][0]
    metrics["mcc_ci95_high"] = intervals["mcc"][1]

    metrics["parameter_count"] = sum(
        p.numel()
        for p in model.parameters()
    )
    metrics["test_examples"] = len(test_set)
    metrics["cleaned_empty_test_removed"] = removed
    metrics["evaluation_time_seconds"] = evaluation_time
    metrics["test_examples_per_sec"] = (
        len(test_set) / max(evaluation_time, 1e-8)
    )
    metrics["peak_memory_gb"] = (
        torch.cuda.max_memory_allocated() / 1024**3
        if device.type == "cuda"
        else 0.0
    )

    metrics_path = (
        root
        / "outputs"
        / f"textcnn_{run_id}_test_metrics.csv"
    )

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["run_id", run_id])
        writer.writerow(["model", "textcnn"])
        writer.writerow(
            ["checkpoint_epoch", state["epoch"]]
        )
        writer.writerow(
            [
                "checkpoint_val_loss",
                f"{state['val_loss']:.6f}",
            ]
        )
        writer.writerow(
            [
                "checkpoint_val_accuracy",
                f"{state['val_accuracy']:.6f}",
            ]
        )

        for key, value in metrics.items():
            if isinstance(value, float):
                writer.writerow([key, f"{value:.6f}"])
            else:
                writer.writerow([key, value])

    predictions_path = (
        root
        / "outputs"
        / "predictions"
        / f"textcnn_{run_id}_predictions.csv"
    )

    prediction_frame = pd.DataFrame({
        "index": np.arange(len(texts)),
        "true_label": y_true,
        "predicted_label": y_pred,
        "positive_probability": probs,
        "confidence": np.maximum(
            probs,
            1.0 - probs,
        ),
    })

    prediction_frame.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
    )

    slice_frame = build_slice_table(
        texts,
        y_true,
        y_pred,
    )

    slice_path = (
        root
        / "outputs"
        / f"textcnn_{run_id}_slice_metrics.csv"
    )

    slice_frame.to_csv(
        slice_path,
        index=False,
    )

    plot_root = root / "outputs" / "plots"

    confusion_path = (
        plot_root
        / f"textcnn_{run_id}_confusion_matrix.png"
    )
    roc_path = (
        plot_root
        / f"textcnn_{run_id}_roc_curve.png"
    )
    pr_path = (
        plot_root
        / f"textcnn_{run_id}_pr_curve.png"
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        confusion_path,
    )
    save_roc_curve(
        y_true,
        probs,
        roc_path,
    )
    save_pr_curve(
        y_true,
        probs,
        pr_path,
    )

    print(f"Run ID: {run_id}")
    print(f"Checkpoint epoch: {state['epoch']}")
    print(f"Test examples: {len(test_set)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
    print()
    print(f"Saved: {metrics_path.relative_to(root)}")
    print(f"Saved: {predictions_path.relative_to(root)}")
    print(f"Saved: {slice_path.relative_to(root)}")
    print(f"Saved: {confusion_path.relative_to(root)}")
    print(f"Saved: {roc_path.relative_to(root)}")
    print(f"Saved: {pr_path.relative_to(root)}")


if __name__ == "__main__":
    main()
