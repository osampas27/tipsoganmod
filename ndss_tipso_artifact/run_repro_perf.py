#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import csv
import argparse
import numpy as np

from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics, save_json, Timer


def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce TIPSO-GAN performance separately for each dataset."
    )
    p.add_argument(
        "--data", "-d",
        nargs="+",
        default=["cicids2018.csv", "cicddos2019.csv", "cicaptiiot.csv"],
        help=(
            "One or more CIC-style CSV files, e.g.\n"
            "  -d cicids2018.csv cicddos2019.csv cicapt_iiot_2024.csv\n"
            "Each dataset is processed separately and gets its own metrics files."
        ),
    )
    return p.parse_args()


def run_single_dataset(data_file: str):
    """
    Train + evaluate TIPSO-GAN on a single dataset (data_file),
    and write per-dataset artifacts.
    """
    base = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\n[INFO] === Processing dataset: {data_file} (base={base}) ===")

    # Load data for this dataset only
    Xtr, ytr, Xv, yv, Xte, yte, feats = load_cicids_csv_preset([data_file])

    print("[INFO]", base, "shapes:",
          "Xtr =", Xtr.shape, "ytr =", ytr.shape,
          "Xv =", Xv.shape, "yv =", yv.shape,
          "Xte =", Xte.shape, "yte =", yte.shape)

    # ----- Robust construction of X_normal -----
    mask_n = (ytr.flatten() == 0)
    num_normal = int(mask_n.sum())
    print(f"[INFO] {base}: total train samples = {len(ytr)}")
    print(f"[INFO] {base}: normal (label 0) in train = {num_normal}")

    if num_normal >= 16:
        X_normal = Xtr[mask_n]
    else:
        # Fallback: use a subset of Xtr as pseudo-normal so GAN training is well-defined
        X_normal = Xtr[:max(16, len(Xtr) // 4)]
        print(f"[WARN] {base}: few/no label-0 samples in train. "
              "Using a subset of Xtr as pseudo-normal for GAN training.")

    print(f"[INFO] {base}: using {X_normal.shape[0]} samples as X_normal")

    # Fresh trainer per dataset
    t = TIPSOTrainer(input_dim=Xtr.shape[1])

    # ----- Train TIPSO-GAN + classifier -----
    with Timer() as t_train:
        t.pretrain_psogan(
            X_normal,
            epochs=cfg.epochs_pretrain,
            batch_size=cfg.batch_size
        )
        t.train_tipso(
            X_normal,
            Xtr, ytr, Xv, yv,
            epochs=cfg.epochs_tipso,
            batch_size=cfg.batch_size,
            balance_strategy="class_weight",
            collect_loss=True
        )

    # ----- Evaluate -----
    with Timer() as t_test:
        preds = t.dee.predict(Xte, verbose=0)

    y_pred = preds.argmax(axis=1)
    summary, cm = compute_metrics(yte, y_pred)

    summary["train_time_s"] = round(float(t_train.elapsed), 3)
    summary["test_time_s"]  = round(float(t_test.elapsed), 3)
    summary["dataset"]      = data_file

    # ----- Save per-dataset outputs -----
    perf_path = f"artifacts/perf_summary_{base}.json"
    cm_path   = f"artifacts/confusion_matrix_{base}.json"
    loss_path = f"artifacts/loss_history_{base}.csv"

    save_json(perf_path, summary)
    save_json(cm_path, {"labels": [0, 1], "matrix": cm})

    with open(loss_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "gen_loss", "disc_loss"])
        for i, (g, d) in enumerate(
                zip(t.history["gen_loss"], t.history["disc_loss"]),
                start=1):
            w.writerow([i, g, d])

    print(f"[OK] {base}: wrote {perf_path}, {cm_path}, {loss_path}")


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    print("[INFO] Datasets to process:", args.data)

    for data_file in args.data:
        run_single_dataset(data_file)

    print("\n[DONE] Finished processing all datasets.")


if __name__ == "__main__":
    main()
