#!/usr/bin/env python3
import os
import csv
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt


def load_summary(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find: {csv_path}")

    # structure: data[metric][attack][model] = value
    data = {
        "acc": defaultdict(dict),
        "prec": defaultdict(dict),
        "rec": defaultdict(dict),
        "f1": defaultdict(dict),
    }

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attack = row["attack"]
            model = row["model"]
            for metric in ["acc", "prec", "rec", "f1"]:
                val = row.get(metric, "")
                if val is None or val == "":
                    continue
                try:
                    data[metric][attack][model] = float(val)
                except ValueError:
                    # Skip malformed values
                    continue

    return data


def plot_metric_bar(data_metric, metric_name, out_prefix):
    """
    data_metric: dict[attack][model] = value
    metric_name: 'acc', 'prec', 'rec', or 'f1'
    out_prefix: e.g., 'artifacts/adaptive_attacks_acc'
    """
    if not data_metric:
        print(f"[WARN] No data for metric '{metric_name}', skipping.")
        return

    attacks = sorted(data_metric.keys())
    # collect all models that appear in the data
    models = sorted({m for attack in attacks for m in data_metric[attack].keys()})

    x = range(len(attacks))
    width = 0.8 / max(len(models), 1)

    plt.figure(figsize=(7, 4))

    for i, model in enumerate(models):
        vals = [data_metric[atk].get(model, 0.0) for atk in attacks]
        offsets = [xi + (i - (len(models)-1)/2) * width for xi in x]
        plt.bar(offsets, vals, width, label=model)

    metric_label = {
        "acc": "Accuracy",
        "prec": "Precision",
        "rec": "Recall",
        "f1": "F1-score",
    }.get(metric_name, metric_name)

    plt.xticks(list(x), [atk.upper() for atk in attacks])
    plt.ylabel(metric_label)
    plt.ylim(0, 1.1)
    plt.title(f"Adaptive attacks: {metric_label} by model and attack")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        out_path = f"{out_prefix}.{ext}"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Wrote {out_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot robustness of TIPSO-GAN and baselines under "
            "adaptive attacks using adaptive_attacks_summary.csv"
        )
    )
    parser.add_argument(
        "--csv",
        default="artifacts/adaptive_attacks_summary.csv",
        help="Path to adaptive_attacks_summary.csv "
             "(default: artifacts/adaptive_attacks_summary.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts",
        help="Directory to save figures (default: artifacts)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_summary(args.csv)

    # Generate one figure per metric
    for metric in ["acc", "prec", "rec", "f1"]:
        out_prefix = os.path.join(args.out_dir, f"adaptive_attacks_{metric}")
        plot_metric_bar(data[metric], metric, out_prefix)


if __name__ == "__main__":
    main()
