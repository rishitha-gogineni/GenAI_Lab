import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def mcnemar_exact(base, other, name):
    if not np.array_equal(base["true_label"].to_numpy(), other["true_label"].to_numpy()):
        raise ValueError("Prediction files do not use the same test order.")

    base_correct = base["predicted_label"].to_numpy() == base["true_label"].to_numpy()
    other_correct = other["predicted_label"].to_numpy() == other["true_label"].to_numpy()

    base_only = int(np.sum(base_correct & ~other_correct))
    other_only = int(np.sum(~base_correct & other_correct))
    discordant = base_only + other_only

    p_value = 1.0
    if discordant:
        p_value = binomtest(
            min(base_only, other_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "comparison": name,
        "baseline_only_correct": base_only,
        "experimental_only_correct": other_only,
        "discordant_pairs": discordant,
        "exact_mcnemar_p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root)
    pred = root / "outputs" / "predictions"

    baseline = pd.read_csv(pred / "baseline_20260926_130146_predictions.csv")
    dpcnn = pd.read_csv(pred / "dpcnn_20260926_152502_predictions.csv")
    textcnn = pd.read_csv(pred / "textcnn_20260926_133806_predictions.csv")

    rows = [
        mcnemar_exact(baseline, dpcnn, "Baseline vs DPCNN"),
        mcnemar_exact(baseline, textcnn, "Baseline vs TextCNN"),
    ]

    result = pd.DataFrame(rows)
    output = root / "outputs" / "mcnemar_tests.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
