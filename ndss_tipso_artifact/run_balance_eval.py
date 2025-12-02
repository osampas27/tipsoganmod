#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import csv
import argparse
import numpy as np

from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate class balancing strategies across one or more datasets."
    )
    p.add_argument(
        "--data", "-d",
        nargs="+",
        default=["cicids2018.csv", "cicddos2019.csv", "cicaptiiot.csv"],
        help="One or more CIC-style CSV files, e.g. cicids2018.csv cicddos2019.csv cicaptiiot.csv",
    )
    return p.parse_args()


def robust_select_normals(Xtr, ytr):
    """
    Select normal (label=0) samples for GAN training.
    If there are too few or none, fall back to a pseudo-normal subset of Xtr.
    """
    mask_norm = (ytr.flatten() == 0)
    Xn = Xtr[mask_norm]
    num_norm = int(mask_norm.sum())

    if num_norm >= 16:
        print(f"[INFO] Using {num_norm} true normal samples for GAN training.")
        return Xn

    print(f"[WARN] Only {num_norm} normal samples; using pseudo-normal subset of Xtr.")
    k = max(16, len(Xtr) // 4)
    Xpseudo = Xtr[:k]
    factor = int(np.ceil(cfg.batch_size / max(1, Xpseudo.shape[0])))
    Xpseudo = np.tile(Xpseudo, (factor, 1))
    print(f"[INFO] Pseudo-normal shape for GAN: {Xpseudo.shape}")
    return Xpseudo


def run_balance_for_dataset(data_file: str):
    base = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\n[INFO] === Class balancing for dataset: {data_file} (base={base}) ===")

    # Per-dataset artifact folder
    outdir = os.path.join("artifacts", base)
    os.makedirs(outdir, exist_ok=True)

    # Load dataset
    Xtr, ytr, Xv, yv, Xte, yte, feats = load_cicids_csv_preset([data_file])
    print(f"[INFO] {base}: Xtr={Xtr.shape}, Xv={Xv.shape}, Xte={Xte.shape}")

    # Save y_test for ROC use
    y_test_path = os.path.join(outdir, "y_test.npy")
    np.save(y_test_path, yte)
    print(f"[OK] {base}: saved test labels → {y_test_path}")

    strategies = ["none", "undersample", "oversample", "class_weight"]
    rows = []

    for s in strategies:
        print(f"[INFO] {base}: strategy = {s}")

        # Robust normal selection for GAN
        Xn = robust_select_normals(Xtr, ytr)

        # Train TIPSO-GAN
        t = TIPSOTrainer(input_dim=Xtr.shape[1])
        t.pretrain_psogan(Xn, epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)

        t.train_tipso(
            Xn,
            Xtr, ytr,
            Xv, yv,
            epochs=cfg.epochs_tipso,
            batch_size=cfg.batch_size,
            balance_strategy=s,
        )

        # Predict probabilities for ROC, not just argmax
        probs = t.dee.predict(Xte, verbose=0)
        preds = probs.argmax(axis=1)

        # Save per-strategy probabilities for ROC plotting
        preds_path = os.path.join(outdir, f"preds_{s}.npy")
        np.save(preds_path, probs)
        print(f"[OK] {base}: saved probs for {s} → {preds_path}")

        # Metrics
        m, _ = compute_metrics(yte, preds)
        m["strategy"] = s
        m["dataset"] = base
        rows.append(m)

    # Per-dataset CSV
    csv_path = os.path.join(outdir, "balance_grid.csv")
    cols = ["dataset", "strategy", "accuracy", "precision", "recall",
            "f1", "fp", "fn", "tp", "tn"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    print(f"[OK] {base}: wrote {csv_path}")

    # Backwards compatibility for CICIDS-2018 (original artifact layout)
    if base.lower() == "cicids2018":
        legacy_csv = "artifacts/balance_grid.csv"
        legacy_y   = "artifacts/y_test.npy"

        # Copy CSV content to legacy name
        with open(legacy_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})
        print(f"[OK] {base}: also wrote legacy CSV → {legacy_csv}")

        # Copy y_test to legacy path
        np.save(legacy_y, yte)
        print(f"[OK] {base}: also wrote legacy y_test → {legacy_y}")

        # Copy preds_*.npy to top-level for the existing ROC script
        for s in strategies:
            src = os.path.join(outdir, f"preds_{s}.npy")
            if os.path.exists(src):
                dst = os.path.join("artifacts", f"preds_{s}.npy")
                np.save(dst, np.load(src))
        print(f"[OK] {base}: also copied preds_*.npy to artifacts/ for legacy ROC plotting")


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    print("[INFO] Running class balancing across datasets:", args.data)

    for df in args.data:
        run_balance_for_dataset(df)

    print("\n[DONE] Finished class balancing evaluation for all datasets.")


if __name__ == "__main__":
    main()
