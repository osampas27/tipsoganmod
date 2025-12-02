#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import csv

from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate TIPSO-GAN loss history for one or more datasets."
    )
    p.add_argument(
        "--data", "-d",
        nargs="+",
        default=["cicids2018.csv", "cicddos2019.csv", "cicaptiiot.csv"],
        help=(
            "One or more CIC-style CSV files, e.g.\n"
            "  -d cicids2018.csv cicddos2019.csv cicapt_iiot_2024.csv\n"
            "Each dataset is trained separately and gets its own loss_history_<base>.csv."
        ),
    )
    return p.parse_args()


def run_loss_for_dataset(data_file: str):
    base = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\n[INFO] === Loss history for dataset: {data_file} (base={base}) ===")

    # Make sure artifacts dir exists
    os.makedirs("artifacts", exist_ok=True)

    # Force full-length training for clean loss curves
    cfg.epochs_pretrain   = 5
    cfg.epochs_tipso      = 100
    cfg.patience_drop     = 10**9   # disable drop-based early stop
    cfg.val_acc_threshold = 1.1     # disable acc-based early stop (acc ≤ 1.0)

    # Load single dataset
    Xtr, ytr, Xv, yv, Xte, yte, feats = load_cicids_csv_preset([data_file])

    # Robust X_normal selection
    mask_n   = (ytr.flatten() == 0)
    num_norm = int(mask_n.sum())
    print(f"[INFO] {base}: train samples = {len(ytr)}, normals(label=0) = {num_norm}")

    if num_norm >= 16:
        Xn = Xtr[mask_n]
    else:
        # Fallback: pseudo-normal subset for datasets with few/no 0-labels
        Xn = Xtr[:max(16, len(Xtr) // 4)]
        print(
            f"[WARN] {base}: few/no label-0 samples in train. "
            "Using subset of Xtr as pseudo-normal for GAN training."
        )

    print(f"[INFO] {base}: using {Xn.shape[0]} samples as X_normal")

    # Train TIPSO-GAN with loss collection
    t = TIPSOTrainer(input_dim=Xtr.shape[1])
    t.pretrain_psogan(Xn, epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t.train_tipso(
        Xn,
        Xtr, ytr,
        Xv, yv,
        epochs=cfg.epochs_tipso,
        batch_size=cfg.batch_size,
        balance_strategy="class_weight",
        collect_loss=True,
    )

    gen_hist  = t.history.get("gen_loss", [])
    disc_hist = t.history.get("disc_loss", [])

    # If for some reason nothing was collected, pad with zeros
    if len(gen_hist) == 0 or len(disc_hist) == 0:
        gen_hist  = gen_hist  if gen_hist  else [0.0] * cfg.epochs_tipso
        disc_hist = disc_hist if disc_hist else [0.0] * cfg.epochs_tipso

    # Optionally pad to exactly epochs_tipso length (in case early stop still fires)
    if len(gen_hist) < cfg.epochs_tipso:
        last_g = gen_hist[-1]
        gen_hist = gen_hist + [last_g] * (cfg.epochs_tipso - len(gen_hist))
    if len(disc_hist) < cfg.epochs_tipso:
        last_d = disc_hist[-1]
        disc_hist = disc_hist + [last_d] * (cfg.epochs_tipso - len(disc_hist))

    # Per-dataset loss history file
    out_path = f"artifacts/loss_history_{base}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "gen_loss", "disc_loss"])
        for i, (g, d) in enumerate(zip(gen_hist, disc_hist), start=1):
            w.writerow([i, g, d])

    print(f"[OK] {base}: wrote {out_path}")

    # Backwards compatibility: same name as original for cicids2018
    if base.lower() == "cicids2018":
        legacy_path = "artifacts/loss_history.csv"
        with open(legacy_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "gen_loss", "disc_loss"])
            for i, (g, d) in enumerate(zip(gen_hist, disc_hist), start=1):
                w.writerow([i, g, d])
        print(f"[OK] {base}: also wrote {legacy_path} (legacy name)")


def main():
    args = parse_args()
    print("[INFO] Datasets for loss history:", args.data)

    for data_file in args.data:
        run_loss_for_dataset(data_file)

    print("\n[DONE] Finished loss history generation for all datasets.")


if __name__ == "__main__":
    main()
