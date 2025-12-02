#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import json
import time

from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure TIPSO-DeepRed parameter count and inference cost for one or more datasets."
    )
    p.add_argument(
        "--data", "-d",
        nargs="+",
        default=["cicids2018.csv", "cicddos2019.csv", "cicaptiiot.csv"],
        help=(
            "One or more CIC-style CSV files, e.g.\n"
            "  -d cicids2018.csv cicddos2019.csv cicaptiiot.csv\n"
            "Each dataset is trained separately and gets its own cost_metrics_<base>.json."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size to use for timing inference (default: 512).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated forward passes to average inference time (default: 3).",
    )
    return p.parse_args()


def count_params(model):
    return int(model.count_params())


def timed_inference(model, X, repeats=3):
    # Warm-up run (to avoid first-call overhead)
    model.predict(X[:min(128, len(X))], verbose=0)
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict(X, verbose=0)
    return (time.perf_counter() - t0) / repeats


def run_cost_for_dataset(data_file: str, batch_size: int, repeats: int):
    base = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\n[INFO] === Cost profile for dataset: {data_file} (base={base}) ===")

    os.makedirs("artifacts", exist_ok=True)

    # Load data
    Xtr, ytr, Xv, yv, Xte, yte, feats = load_cicids_csv_preset([data_file])
    print(f"[INFO] {base}: Xtr={Xtr.shape}, ytr={ytr.shape}, Xte={Xte.shape}")

    # Robust X_normal selection (same logic as other scripts)
    mask_n = (ytr.flatten() == 0)
    num_norm = int(mask_n.sum())
    print(f"[INFO] {base}: train samples = {len(ytr)}, normals(label=0) = {num_norm}")

    if num_norm >= 16:
        Xn = Xtr[mask_n]
    else:
        Xn = Xtr[:max(16, len(Xtr) // 4)]
        print(
            f"[WARN] {base}: few/no label-0 samples in train. "
            "Using subset of Xtr as pseudo-normal for GAN training."
        )

    print(f"[INFO] {base}: using {Xn.shape[0]} samples as X_normal")

    # Train TIPSO-GAN classifier to get a representative cost profile
    t = TIPSOTrainer(input_dim=Xtr.shape[1])
    t.pretrain_psogan(Xn, epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t.train_tipso(
        Xn,
        Xtr, ytr,
        Xv, yv,
        epochs=cfg.epochs_tipso,
        batch_size=cfg.batch_size,
        balance_strategy="class_weight",
    )

    # Parameter count
    params = count_params(t.dee)

    # Inference timing
    if len(Xte) == 0:
        print(f"[WARN] {base}: Xte is empty; cannot time inference. Skipping timing.")
        avg_s = None
        eff_batch = 0
    else:
        eff_batch = min(batch_size, len(Xte))
        batch = Xte[:eff_batch]
        avg_s = timed_inference(t.dee, batch, repeats=repeats)

    # Save cost metrics
    out_path = os.path.join("artifacts", f"cost_metrics_{base}.json")
    payload = {
        "dataset": data_file,
        "dee_params": params,
        "avg_infer_time_per_batch_s": float(avg_s) if avg_s is not None else None,
        "batch_size": eff_batch,
        "repeats": repeats,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[OK] {base}: wrote {out_path}")

    # Backwards compatibility: keep original name for cicids2018
    if base.lower() == "cicids2018":
        legacy_path = os.path.join("artifacts", "cost_metrics.json")
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] {base}: also wrote {legacy_path} (legacy name)")


def main():
    args = parse_args()
    print("[INFO] Datasets for cost profiling:", args.data)

    for data_file in args.data:
        run_cost_for_dataset(data_file, batch_size=args.batch_size, repeats=args.repeats)

    print("\n[DONE] Finished cost profiling for all datasets.")


if __name__ == "__main__":
    main()
