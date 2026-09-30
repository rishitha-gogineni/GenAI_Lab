import argparse
import csv
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import (
    YelpDataset,
    load_split_indices,
    load_vocab,
    subset_from_indices,
)
from model_textcnn import TextCNN
from utils import (
    append_log,
    load_config,
    make_dirs,
    new_run_id,
    project_root,
    set_seed,
    write_environment,
)


def evaluate(model, loader, device, amp_enabled, max_batches=None):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    batches = 0

    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)

            predictions = (torch.sigmoid(logits) >= 0.5).float()

            total_loss += loss.item()
            total_correct += (predictions == y).sum().item()
            total_examples += y.numel()
            batches += 1

    return (
        total_loss / max(batches, 1),
        total_correct / max(total_examples, 1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/textcnn_config.json",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    root = project_root()
    config = load_config(root / args.config)

    make_dirs(root)
    set_seed(config["seed"])
    write_environment(root)

    run_id = new_run_id()
    log_path = (
        root
        / "logs"
        / "raw"
        / f"textcnn_{run_id}.log"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = load_dataset(config["data"]["dataset_name"])
    train_data = dataset["train"]

    train_idx, val_idx = load_split_indices(
        root / "data_processed" / "split_indices.npz"
    )

    if args.smoke_test:
        train_idx = train_idx[:1800]
        val_idx = val_idx[:200]

    train_texts, train_labels = subset_from_indices(
        train_data,
        train_idx,
    )
    val_texts, val_labels = subset_from_indices(
        train_data,
        val_idx,
    )

    vocab = load_vocab(
        root / "data_processed" / "vocab.json"
    )

    train_set = YelpDataset(
        train_texts,
        train_labels,
        vocab,
        config["data"]["max_length"],
    )
    val_set = YelpDataset(
        val_texts,
        val_labels,
        vocab,
        config["data"]["max_length"],
    )

    batch_size = (
        16
        if args.smoke_test
        else config["training"]["batch_size"]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model_config = config["model"]

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=model_config["embedding_dim"],
        num_filters=model_config["num_filters"],
        kernel_sizes=model_config["kernel_sizes"],
        dropout=model_config["dropout"],
        pad_id=vocab.get("<PAD>", 0),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    amp_enabled = (
        config["training"]["use_amp"]
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    epochs = (
        1
        if args.smoke_test
        else config["training"]["epochs"]
    )

    max_steps = 3 if args.smoke_test else None
    best_val_loss = float("inf")

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    append_log(log_path, f"run_id={run_id}")
    append_log(log_path, "model=textcnn")
    append_log(log_path, f"device={device}")
    append_log(log_path, f"parameters={parameter_count}")
    append_log(
        log_path,
        "vocabulary_source=training_subset_only",
    )
    append_log(
        log_path,
        f"train_examples={len(train_set)}",
    )
    append_log(
        log_path,
        f"val_examples={len(val_set)}",
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    global_step = 0

    for epoch in range(epochs):
        model.train()

        epoch_start = time.time()
        total_loss = 0.0
        total_correct = 0
        examples_seen = 0
        steps = 0

        for x, y in train_loader:
            if max_steps is not None and steps >= max_steps:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
            ):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            total_loss += loss.item()
            total_correct += (
                predictions == y
            ).sum().item()
            examples_seen += y.numel()
            steps += 1
            global_step += 1

            if (
                global_step
                % config["training"]["log_every"]
                == 0
                or args.smoke_test
            ):
                append_log(
                    log_path,
                    (
                        f"step={global_step} "
                        f"loss={loss.item():.4f}"
                    ),
                )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device,
            amp_enabled,
            3 if args.smoke_test else None,
        )

        train_loss = (
            total_loss / max(steps, 1)
        )
        train_accuracy = (
            total_correct / max(examples_seen, 1)
        )

        epoch_time = time.time() - epoch_start
        examples_per_sec = (
            examples_seen
            / max(epoch_time, 1e-8)
        )

        append_log(
            log_path,
            (
                f"epoch={epoch + 1} "
                f"train_loss={train_loss:.4f} "
                f"train_accuracy={train_accuracy:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_accuracy={val_accuracy:.4f} "
                f"epoch_time_sec={epoch_time:.2f} "
                f"examples_per_sec={examples_per_sec:.2f}"
            ),
        )

        state = {
            "run_id": run_id,
            "model_name": "textcnn",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "vocab_size": len(vocab),
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        last_path = (
            root
            / "checkpoints"
            / f"textcnn_{run_id}_last.pt"
        )
        torch.save(state, last_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_path = (
                root
                / "checkpoints"
                / f"textcnn_{run_id}_best.pt"
            )
            torch.save(state, best_path)

    total_time = time.time() - start_time

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**3
        if device.type == "cuda"
        else 0.0
    )

    metrics_path = (
        root
        / "outputs"
        / f"textcnn_{run_id}_training.csv"
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
            ["train_loss", f"{train_loss:.6f}"]
        )
        writer.writerow(
            [
                "train_accuracy",
                f"{train_accuracy:.6f}",
            ]
        )
        writer.writerow(
            [
                "validation_loss",
                f"{val_loss:.6f}",
            ]
        )
        writer.writerow(
            [
                "validation_accuracy",
                f"{val_accuracy:.6f}",
            ]
        )
        writer.writerow(
            ["parameter_count", parameter_count]
        )
        writer.writerow(
            ["peak_memory_gb", f"{peak_memory:.3f}"]
        )
        writer.writerow(
            [
                "training_time_seconds",
                f"{total_time:.2f}",
            ]
        )

    print(f"Run ID: {run_id}")
    print(
        f"Finished. Raw log: "
        f"{log_path.relative_to(root)}"
    )
    print(
        "Best checkpoint:",
        f"checkpoints\\textcnn_{run_id}_best.pt",
    )


if __name__ == "__main__":
    main()
