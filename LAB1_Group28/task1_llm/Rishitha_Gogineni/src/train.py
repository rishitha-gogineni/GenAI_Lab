import argparse
import csv
import math
import time

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import build_data
from model import GPT
from utils import (
    append_checkpoint_manifest,
    append_log,
    load_config,
    make_dirs,
    new_log_path,
    new_run_id,
    project_root,
    set_seed,
    write_environment,
)


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total = 0.0
    count = 0
    correct = 0
    tokens = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, loss = model(x, y)
            total += loss.item()
            count += 1
            correct += (logits.argmax(-1) == y).sum().item()
            tokens += y.numel()
    return total / max(count, 1), correct / max(tokens, 1)


def save_loss_plot(root, train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "outputs" / "plots" / "loss_curve.png", dpi=150)
    plt.close()


def save_grad_plot(root, grad_values):
    if not grad_values:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(grad_values) + 1), grad_values)
    plt.xlabel("Training step")
    plt.ylabel("Gradient norm")
    plt.tight_layout()
    plt.savefig(root / "outputs" / "plots" / "gradient_norm.png", dpi=150)
    plt.close()


def write_metrics(root, rows):
    with open(root / "metrics_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/gpt_config.json")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    root = project_root()
    config = load_config(root / args.config)
    make_dirs(root)
    set_seed(config["seed"])
    write_environment(root)
    run_id = new_run_id()
    log_path = new_log_path(root, run_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tokenizer, train_set, val_set = build_data(config, root, args.smoke_test)
    batch_size = 2 if args.smoke_test else config["training"]["batch_size"]
    loader_args = {
        "batch_size": batch_size,
        "num_workers": config["data"]["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = GPT(tokenizer.vocab_size, config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs = 1 if args.smoke_test else config["training"]["epochs"]
    max_steps = 3 if args.smoke_test else None
    steps_per_epoch = min(len(train_loader), max_steps) if max_steps else len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * config["training"]["warmup_ratio"]))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp_enabled = config["training"]["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val = float("inf")
    global_step = 0
    nan_count = 0
    loss_spikes = 0
    previous_loss = None
    train_losses = []
    val_losses = []
    grad_values = []
    total_train_tokens = 0
    training_start = time.time()

    append_log(log_path, f"run_id={run_id}")
    append_log(log_path, f"device={device}")
    append_log(log_path, f"parameters={parameter_count}")
    append_log(log_path, f"train_sequences={len(train_set)} val_sequences={len(val_set)}")

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running = 0.0
        steps = 0
        epoch_tokens = 0

        for x, y in train_loader:
            if max_steps is not None and steps >= max_steps:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                _, loss = model(x, y)

            if not torch.isfinite(loss):
                nan_count += 1
                append_log(log_path, f"step={global_step + 1} non_finite_loss={loss.item()}")
                optimizer.zero_grad(set_to_none=True)
                continue

            current_loss = loss.item()
            if previous_loss is not None and current_loss > previous_loss * 2.0:
                loss_spikes += 1
            previous_loss = current_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"]["grad_clip"]
            )
            grad_value = float(grad_norm)
            grad_values.append(grad_value)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += current_loss
            steps += 1
            global_step += 1
            batch_tokens = y.numel()
            epoch_tokens += batch_tokens
            total_train_tokens += batch_tokens

            if global_step % config["training"]["log_every"] == 0 or args.smoke_test:
                append_log(
                    log_path,
                    f"step={global_step} loss={current_loss:.4f} "
                    f"grad_norm={grad_value:.4f} lr={scheduler.get_last_lr()[0]:.8f}",
                )

        val_loss, val_acc = evaluate(
            model, val_loader, device, 3 if args.smoke_test else None
        )
        train_loss = running / max(steps, 1)
        epoch_time = time.time() - epoch_start
        tokens_per_sec = epoch_tokens / max(epoch_time, 1e-9)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        append_log(
            log_path,
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_top1={val_acc:.4f} "
            f"epoch_time_sec={epoch_time:.2f} tokens_per_sec={tokens_per_sec:.2f}",
        )

        state = {
            "run_id": run_id,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "vocab_size": tokenizer.vocab_size,
            "epoch": epoch + 1,
            "val_loss": val_loss,
        }
        last_path = root / "checkpoints" / f"{run_id}_last.pt"
        torch.save(state, last_path)
        if val_loss < best_val:
            best_val = val_loss
            best_path = root / "checkpoints" / f"{run_id}_best.pt"
            torch.save(state, best_path)

    elapsed = time.time() - training_start
    train_tokens_per_sec = total_train_tokens / max(elapsed, 1e-9)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    avg_grad = sum(grad_values) / max(len(grad_values), 1)

    save_loss_plot(root, train_losses, val_losses)
    save_grad_plot(root, grad_values)

    rows = [
        ["run_id", run_id],
        ["train_cross_entropy_loss", f"{train_loss:.6f}"],
        ["validation_cross_entropy_loss", f"{val_loss:.6f}"],
        ["perplexity", f"{math.exp(val_loss):.6f}"],
        ["bits_per_character", f"{val_loss / math.log(2):.6f}"],
        ["generalization_gap", f"{val_loss - train_loss:.6f}"],
        ["top1_next_character_accuracy", f"{val_acc:.6f}"],
        ["average_gradient_norm", f"{avg_grad:.6f}"],
        ["loss_spike_count", loss_spikes],
        ["nan_or_non_finite_loss_count", nan_count],
        ["parameter_count", parameter_count],
        ["training_tokens_per_sec", f"{train_tokens_per_sec:.2f}"],
        ["peak_memory_gb", f"{peak_gb:.3f}"],
        ["total_training_time_seconds", f"{elapsed:.2f}"],
    ]
    write_metrics(root, rows)

    source_log = str(log_path.relative_to(root))
    append_checkpoint_manifest(root, [
        [run_id, str(best_path.relative_to(root)), "lowest validation loss", f"{best_val:.6f}", source_log],
        [run_id, str(last_path.relative_to(root)), "final epoch", f"{val_loss:.6f}", source_log],
    ])

    print(f"Run ID: {run_id}")
    print(f"Finished. Raw log: {source_log}")
    print(f"Best checkpoint: {best_path.relative_to(root)}")


if __name__ == "__main__":
    main()
