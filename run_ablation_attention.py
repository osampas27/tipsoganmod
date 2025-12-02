#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import os
import csv
import argparse
import numpy as np

from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Ablation: TIPSO-GAN with vs without attention, across datasets."
    )
    p.add_argument(
        "--data", "-d",
        nargs="+",
        default=["cicids2018.csv", "cicddos2019.csv", "cicaptiiot.csv"],
        help="One or more CIC-style CSV files.",
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


def train_eval_for_dataset(data_file: str, use_attention: bool):
    base = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\n[INFO] === Dataset: {base} | use_attention={use_attention} ===")

    Xtr, ytr, Xv, yv, Xte, yte, feats = load_cicids_csv_preset([data_file])

    # Robust normal selection
    Xn = robust_select_normals(Xtr, ytr)

    # Build trainer with or without attention
    t = TIPSOTrainer(input_dim=Xtr.shape[1], use_attention=use_attention)

    # GAN pretrain
    t.pretrain_psogan(Xn, epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)

    # Full TIPSO training
    t.train_tipso(
        Xn,
        Xtr, ytr,
        Xv, yv,
        epochs=cfg.epochs_tipso,
        batch_size=cfg.batch_size,
        balance_strategy="class_weight",
    )

    # Evaluate
    preds = t.dee.predict(Xte, verbose=0).argmax(axis=1)
    m, _ = compute_metrics(yte, preds)
    m["use_attention"] = bool(use_attention)
    m["dataset"] = base
    return m


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    print("[INFO] Datasets for attention ablation:", args.data)

    for data_file in args.data:
        base = os.path.splitext(os.path.basename(data_file))[0]
        rows = []

        for flag in [True, False]:
            m = train_eval_for_dataset(data_file, use_attention=flag)
            rows.append(m)

        # Per-dataset CSV
        out_path = f"artifacts/attention_ablation_{base}.csv"
        cols = ["dataset", "use_attention",
                "accuracy", "precision", "recall", "f1",
                "fp", "fn", "tp", "tn"]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})

        print(f"[OK] Wrote {out_path}")

        # Backwards compatibility: original filename for cicids2018
        if base.lower() == "cicids2018":
            legacy_path = "artifacts/attention_ablation.csv"
            with open(legacy_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in cols})
            print(f"[OK] Also wrote {legacy_path} (legacy name)")


if __name__ == "__main__":
    main()
