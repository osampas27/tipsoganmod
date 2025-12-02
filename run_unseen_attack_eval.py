#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_with_attack_names
from tipso_gan.metrics import compute_metrics, save_json


def parse_args():
    p = argparse.ArgumentParser(
        description="Unseen attack evaluation for TIPSO-GAN on one or more datasets."
    )
    p.add_argument(
        "--data", "-d",
        nargs="+",
        default=["cicids2018.csv"],
        help=(
            "One or more CIC-style CSV files with attack names, e.g.\n"
            "  -d cicids2018.csv cicddos2019.csv cicapt_iiot_2024.csv\n"
            "Each dataset is evaluated separately with its own held-out attack."
        ),
    )
    return p.parse_args()


def run_unseen_for_dataset(data_file: str):
    base = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\n[INFO] === Unseen-attack eval for dataset: {data_file} (base={base}) ===")

    # Load full dataset with attack names
    X, y, names, feats = load_with_attack_names([data_file])
    y = y.reshape(-1)
    names = np.array(names)

    print(f"[INFO] {base}: X shape = {X.shape}, y shape = {y.shape}")

    uniq = np.unique(names)
    attacks = [n for n in uniq if str(n).upper() != "BENIGN"]

    if len(attacks) == 0:
        print(f"[WARN] {base}: No attack families found; cannot run unseen eval. Skipping.")
        return

    # Choose held-out attack family as the most frequent one
    counts = {a: int(np.sum(names == a)) for a in attacks}
    held_out = max(counts.items(), key=lambda kv: kv[1])[0]

    print(f"[INFO] {base}: Held-out attack family = {held_out} "
          f"(count = {counts[held_out]})")

    # Split indices for held-out attack vs the rest
    mask_train = (names != held_out)
    mask_held  = (names == held_out)

    X_all_train, y_all_train = X[mask_train], y[mask_train]
    X_attack,    y_attack    = X[mask_held],  y[mask_held]

    benign_idx = np.where(y == 0)[0]

    if len(X_attack) == 0 or len(benign_idx) == 0:
        print(f"[WARN] {base}: Insufficient held-out attack or benign; skipping unseen eval.")
        return

    # Build balanced test set: held-out attack vs benign
    k = min(len(X_attack), len(benign_idx))
    Xte = np.vstack([X_attack[:k], X[benign_idx[:k]]])
    yte = np.hstack([y_attack[:k], y[benign_idx[:k]]])

    if len(X_all_train) < 20 or len(np.unique(y_all_train)) < 2:
        print(f"[WARN] {base}: Training set too small or only one class after hold-out; skipping.")
        return

    # Train/val split on remaining data
    Xtr, Xv, ytr, yv = train_test_split(
        X_all_train,
        y_all_train,
        test_size=0.2,
        random_state=42,
        stratify=y_all_train
    )

    # Robust X_normal selection
    mask_normal = (ytr == 0)
    num_normal = int(mask_normal.sum())
    print(f"[INFO] {base}: Train samples = {len(ytr)}, normal(label=0) = {num_normal}")

    if num_normal >= 16:
        Xn = Xtr[mask_normal]
    else:
        Xn = Xtr[:max(16, len(Xtr) // 4)]
        print(f"[WARN] {base}: Few/no normals after hold-out. "
              "Using a subset of Xtr as pseudo-normal for GAN training.")

    print(f"[INFO] {base}: Using {Xn.shape[0]} samples as X_normal")

    # Train TIPSO-GAN
    t = TIPSOTrainer(input_dim=X.shape[1])
    t.pretrain_psogan(Xn, epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t.train_tipso(
        Xn,
        Xtr, ytr.reshape(-1, 1),
        Xv,  yv.reshape(-1, 1),
        epochs=cfg.epochs_tipso,
        batch_size=cfg.batch_size,
        balance_strategy="class_weight"
    )

    # Evaluate on held-out attack vs benign
    preds = t.dee.predict(Xte, verbose=0).argmax(axis=1)
    m, _ = compute_metrics(yte, preds)

    out_path = f"artifacts/unseen_report_{base}.json"
    save_json(
        out_path,
        {
            "dataset": data_file,
            "held_out_attack": str(held_out),
            "held_out_count": int(len(X_attack)),
            "metrics": m,
        },
    )
    print(f"[OK] {base}: held-out {held_out} -> {out_path}")


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    print("[INFO] Datasets to evaluate (unseen attacks):", args.data)
    for data_file in args.data:
        run_unseen_for_dataset(data_file)

    print("\n[DONE] Finished unseen-attack evaluation for all datasets.")


if __name__ == "__main__":
    main()
