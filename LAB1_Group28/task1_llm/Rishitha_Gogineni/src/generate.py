import argparse
import csv
import json
import time
from pathlib import Path

import torch

from data import CharTokenizer
from model import GPT
from utils import load_config, project_root


def load_tokenizer(folder):
    with open(folder / "char_to_idx.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    return CharTokenizer(mapping.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    root = project_root()
    config = load_config(root / "config" / "gpt_config.json")

    tokenizer = load_tokenizer(root / "data_processed")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = root / args.checkpoint
    state = torch.load(checkpoint_path, map_location=device)

    run_id = state.get(
        "run_id",
        checkpoint_path.stem.replace("_best", "").replace("_last", "")
    )

    model = GPT(tokenizer.vocab_size, config).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    prompt = args.prompt or config["generation"]["prompt"]

    ids = tokenizer.encode(prompt)
    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    output = model.generate(
        x,
        config["generation"]["max_new_chars"],
        config["generation"]["temperature"]
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start

    text = tokenizer.decode(output[0].tolist())

    generated_chars = output.shape[1] - x.shape[1]
    generation_speed = generated_chars / max(elapsed, 1e-8)

    output_folder = root / "outputs" / "generations"
    output_folder.mkdir(parents=True, exist_ok=True)

    sample_path = output_folder / f"sample_{run_id}.txt"
    sample_path.write_text(text, encoding="utf-8")

    metrics_path = output_folder / f"generation_metrics_{run_id}.csv"

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["run_id", run_id])
        writer.writerow(["checkpoint", args.checkpoint])
        writer.writerow(["generated_characters", generated_chars])
        writer.writerow(["prompt", prompt])
        writer.writerow(["prompt_characters", len(prompt)])
        writer.writerow(["generation_time_seconds", f"{elapsed:.4f}"])
        writer.writerow(
            ["generation_tokens_per_sec", f"{generation_speed:.2f}"]
        )

    print(f"Run ID: {run_id}")
    print(f"Sample saved: {sample_path.relative_to(root)}")
    print(f"Metrics saved: {metrics_path.relative_to(root)}")
    print(f"Generation speed: {generation_speed:.2f} chars/sec")
    print()
    print(text)


if __name__ == "__main__":
    main()