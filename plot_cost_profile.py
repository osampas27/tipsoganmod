#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import json

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot TIPSO-DeepRed cost profile (params & latency) from cost_metrics_*.json."
    )
    p.add_argument(
        "--files", "-f",
        nargs="+",
        default=[
            os.path.join("artifacts", "cost_metrics_cicids2018.json"),
            os.path.join("artifacts", "cost_metrics_cicddos2019.json"),
            os.path.join("artifacts", "cost_metrics_cicaptiiot.json"),
        ],
        help=(
            "One or more cost_metrics_*.json files to include in the plots.\n"
            "Each file must have fields: dee_params, avg_infer_time_per_batch_s, batch_size.\n"
            "Example:\n"
            "  -f artifacts/cost_metrics_cicids2018.json\n"
            "  -f artifacts/cost_metrics_cicids2018.json artifacts/cost_metrics_cicddos2019.json"
        ),
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Also display the plots interactively (in addition to saving PNGs).",
    )
    return p.parse_args()


def load_cost_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Try to infer a nice label
    dataset_label = data.get("dataset")
    if dataset_label:
        dataset_label = os.path.splitext(os.path.basename(dataset_label))[0]
    else:
        # fallback to file basename
        dataset_label = os.path.splitext(os.path.basename(path))[0]
        if dataset_label.startswith("cost_metrics_"):
            dataset_label = dataset_label[len("cost_metrics_"):]
    params = int(data.get("dee_params", 0))
    avg_s = data.get("avg_infer_time_per_batch_s", None)
    batch = int(data.get("batch_size", 0))
    return dataset_label, params, avg_s, batch


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    labels = []
    params_list = []
    times_list = []
    batches_list = []

    print("[INFO] Using cost metric files:")
    for fp in args.files:
        print("  -", fp)

    for path in args.files:
        if not os.path.isfile(path):
            print(f"[WARN] Skipping {path}: file not found.")
            continue

        label, params, avg_s, batch = load_cost_json(path)
        labels.append(label)
        params_list.append(params)
        times_list.append(avg_s)
        batches_list.append(batch)

    if not labels:
        print("[ERROR] No valid cost metric files loaded; nothing to plot.")
        return

    # ---- Plot 1: parameter counts ----
    x = range(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x, params_list)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Number of parameters")
    plt.title("TIPSO-DeepRed model size across datasets")
    plt.tight_layout()

    out_params = os.path.join("artifacts", "cost_profile_params.png")
    plt.savefig(out_params, dpi=300)
    print(f"[OK] Saved parameter plot -> {out_params}")

    if args.show:
        plt.show()
    else:
        plt.close()

    # ---- Plot 2: inference time per batch ----
    # convert seconds to milliseconds for readability
    times_ms = [
        (t * 1000.0) if (t is not None) else None
        for t in times_list
    ]

    # Filter any None values (if a dataset had empty Xte)
    x2 = []
    labels2 = []
    times_ms2 = []
    batches2 = []

    for i, (lbl, t_ms, b) in enumerate(zip(labels, times_ms, batches_list)):
        if t_ms is None:
            print(f"[WARN] No timing available for {lbl}; skipping in latency plot.")
            continue
        x2.append(len(x2))
        labels2.append(lbl)
        times_ms2.append(t_ms)
        batches2.append(b)

    if times_ms2:
        plt.figure(figsize=(6, 4))
        plt.bar(x2, times_ms2)
        plt.xticks(x2, labels2, rotation=30, ha="right")
        plt.ylabel("Avg inference time per batch (ms)")
        plt.title("TIPSO-DeepRed inference latency across datasets")
        plt.tight_layout()

        out_latency = os.path.join("artifacts", "cost_profile_latency.png")
        plt.savefig(out_latency, dpi=300)
        print(f"[OK] Saved latency plot ->", out_latency)

        if args.show:
            plt.show()
        else:
            plt.close()
    else:
        print("[WARN] No valid timing data; latency plot was not generated.")

    print("\n[DONE] Finished cost profile plotting.")


if __name__ == "__main__":
    main()
