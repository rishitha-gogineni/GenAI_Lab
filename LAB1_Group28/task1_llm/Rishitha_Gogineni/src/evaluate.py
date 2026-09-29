import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def distinct_n(text, n):
    tokens = list(text)
    grams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]
    return len(set(grams)) / max(len(grams), 1)


def repeated_ngram_rate(text, n=4):
    tokens = list(text)
    grams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]

    counts = Counter(grams)
    repeated = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated / max(len(grams), 1)


def read_metric_csv(path):
    rows = {}
    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                rows[row[0]] = row[1]
    return rows


def write_metric_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for name, value in rows.items():
            writer.writerow([name, value])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    sample_path = (
        root
        / "outputs"
        / "generations"
        / f"sample_{args.run_id}.txt"
    )

    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_path}")

    full_text = sample_path.read_text(encoding="utf-8")

    generation_metrics_path = (
        root
        / "outputs"
        / "generations"
        / f"generation_metrics_{args.run_id}.csv"
    )
    generation_rows = read_metric_csv(generation_metrics_path)

    prompt = generation_rows.get("prompt")
    if prompt is None:
        config_path = root / "config" / "gpt_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        prompt = config["generation"]["prompt"]

    if full_text.startswith(prompt):
        generated_text = full_text[len(prompt):]
    else:
        generated_text = full_text
        print(
            "Warning: saved sample does not start with the recorded prompt; "
            "metrics were computed over the full sample."
        )

    metrics = {
        "distinct_1": distinct_n(generated_text, 1),
        "distinct_2": distinct_n(generated_text, 2),
        "distinct_3": distinct_n(generated_text, 3),
        "repeated_4gram_rate": repeated_ngram_rate(generated_text, 4),
    }

    generation_rows["prompt"] = prompt
    generation_rows["prompt_characters"] = str(len(prompt))
    generation_rows["evaluated_generated_characters"] = str(len(generated_text))
    for name, value in metrics.items():
        generation_rows[name] = f"{value:.6f}"
    write_metric_csv(generation_metrics_path, generation_rows)

    report_path = root / "metrics_report.csv"
    report_rows = read_metric_csv(report_path)
    for name, value in metrics.items():
        report_rows[name] = f"{value:.6f}"
    write_metric_csv(report_path, report_rows)

    print(f"Prompt excluded from diversity metrics: {prompt!r}")
    print(f"Generated characters evaluated: {len(generated_text)}")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    print(f"Updated: {generation_metrics_path.relative_to(root)}")
    print(f"Updated: {report_path.relative_to(root)}")


if __name__ == "__main__":
    main()
