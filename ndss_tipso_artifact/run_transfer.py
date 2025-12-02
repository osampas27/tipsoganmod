#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse

from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics, save_json


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate transfer learning for TIPSO-DeepRed on one or more datasets."
    )
    p.add_argument(
        "--data", "-d",
        nargs="+",
        default=["cicids2018.csv","cicddos2019.csv","cicaptiiot.csv"],
        help=(
            "One or more CIC-style CSV files, e.g.\n"
            "  -d cicids2018.csv cicddos2019.csv cicapt_iiot_2024.csv\n"
            "Each dataset is processed separately and gets its own dee_transfer_report_<base>.json."
        ),
    )
    return p.parse_args()


def split_domain(X, y, fraction=0.5):
    n = X.shape[0]
    k = int(n * fraction)
    return (X[:k], y[:k]), (X[k:], y[k:])


def run_transfer_for_dataset(data_file: str):
    base = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\n[INFO] === Transfer eval for dataset: {data_file} (base={base}) ===")

    # Load one dataset
    Xtr, ytr, Xv, yv, Xte, yte, feats = load_cicids_csv_preset([data_file])
    print(f"[INFO] {base}: Xtr={Xtr.shape}, ytr={ytr.shape}, Xv={Xv.shape}, Xte={Xte.shape}")

    # Domain split of training data
    (Xa, ya), (Xb, yb) = split_domain(Xtr, ytr, 0.5)

    # ---------- Baseline (no transfer) ----------
    t_base = TIPSOTrainer(input_dim=Xtr.shape[1])

    # Robust X_normal for domain B
    mask_b_norm = (yb.flatten() == 0)
    num_b_norm = int(mask_b_norm.sum())
    print(f"[INFO] {base}: domain B samples = {len(yb)}, normals in B = {num_b_norm}")

    if num_b_norm >= 16:
        Xb_normal = Xb[mask_b_norm]
    else:
        Xb_normal = Xb[:max(16, len(Xb) // 4)]
        print(f"[WARN] {base}: few/no label-0 in domain B. "
              "Using subset of Xb as pseudo-normal for GAN training.")

    # Pretrain + train baseline model
    t_base.pretrain_psogan(
        Xb_normal,
        epochs=cfg.epochs_pretrain,
        batch_size=cfg.batch_size
    )
    t_base.train_tipso(
        Xb_normal,
        Xb, yb,
        Xv, yv,
        epochs=cfg.epochs_tipso,
        batch_size=cfg.batch_size,
        balance_strategy="class_weight"
    )
    preds_base = t_base.dee.predict(Xte, verbose=0).argmax(axis=1)
    base_metrics, _ = compute_metrics(yte, preds_base)

    # ---------- Transfer model ----------
    t_tr = TIPSOTrainer(input_dim=Xtr.shape[1])

    # First, pretrain classifier on domain A
    t_tr.dee.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["acc"]
    )
    t_tr.dee.fit(
        Xa, ya,
        epochs=5,
        batch_size=cfg.batch_size,
        verbose=0
    )

    # Then, same GAN training on domain B (with robust Xb_normal)
    t_tr.pretrain_psogan(
        Xb_normal,
        epochs=cfg.epochs_pretrain,
        batch_size=cfg.batch_size
    )
    t_tr.train_tipso(
        Xb_normal,
        Xb, yb,
        Xv, yv,
        epochs=cfg.epochs_tipso,
        batch_size=cfg.batch_size,
        balance_strategy="class_weight"
    )
    preds_tr = t_tr.dee.predict(Xte, verbose=0).argmax(axis=1)
    tr_metrics, _ = compute_metrics(yte, preds_tr)

    # ---------- Save per-dataset report ----------
    report = {
        "dataset": data_file,
        "baseline": base_metrics,
        "transfer": tr_metrics,
    }

    out_path = f"artifacts/dee_transfer_report_{base}.json"
    save_json(out_path, report)
    print(f"[OK] {base}: wrote {out_path}")

    # Backwards compatibility: if this is cicids2018 only, also write the old name
    if base.lower() == "cicids2018":
        legacy_path = "artifacts/dee_transfer_report.json"
        save_json(legacy_path, report)
        print(f"[OK] {base}: also wrote {legacy_path} (legacy name)")


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    print("[INFO] Datasets for transfer eval:", args.data)
    for data_file in args.data:
        run_transfer_for_dataset(data_file)

    print("\n[DONE] Finished transfer evaluation for all datasets.")


if __name__ == "__main__":
    main()
