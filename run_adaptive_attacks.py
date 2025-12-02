#!/usr/bin/env python3
import os, sys, json, csv, argparse
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics, save_json
from tipso_gan.attacks import feature_bounds_from_data, fgsm, bim, pgd_linf


# ===========================================================
#   PARSE ARGUMENTS
# ===========================================================
def parse_args():
    p = argparse.ArgumentParser("Multi-dataset adaptive attack evaluation for TIPSO-GAN.")
    p.add_argument(
        "--data", "-d",
        nargs="+",
        default=["cicids2018.csv", "cicddos2019.csv", "cicaptiiot.csv"],
        help="One or more CIC-style CSV files."
    )
    p.add_argument("--eps", type=float, default=0.05, help="L∞ epsilon.")
    p.add_argument("--iters", type=int, default=10, help="Iterations for BIM/PGD.")
    p.add_argument("--alpha", type=float, default=0.01, help="Step size for BIM/PGD.")
    return p.parse_args()


# ===========================================================
#   SAFELY TRAIN BASELINES
# ===========================================================
def train_baselines(Xtr, ytr):
    y_unique = np.unique(ytr)

    # If dataset contains only one class → skip baselines
    if len(y_unique) < 2:
        print("[WARN] Baselines skipped: dataset contains only ONE class.")
        return None, None

    lr = LogisticRegression(max_iter=500)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)

    lr.fit(Xtr, ytr.ravel())
    rf.fit(Xtr, ytr.ravel())

    return lr, rf


# ===========================================================
#   SAFE EVALUATION
# ===========================================================
def safe_eval(model, X, y, is_keras):
    """Return None-metrics if baseline is missing."""
    if model is None:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}, None
    return compute_metrics(y, model.predict(X) if not is_keras else model.predict(X, verbose=0).argmax(axis=1))


# ===========================================================
#   RUN ADAPTIVE ATTACKS FOR ONE DATASET
# ===========================================================
def run_for_dataset(data_file, eps, alpha, iters):
    base = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\n[INFO] === Running Adaptive Attacks for: {base} ===")

    # Dedicated folder per dataset
    outdir = os.path.join("artifacts", base)
    os.makedirs(outdir, exist_ok=True)

    # Load dataset
    Xtr, ytr, Xv, yv, Xte, yte, feats = load_cicids_csv_preset([data_file])

    # Compute bounds
    xmin, xmax, width = feature_bounds_from_data(Xtr)
    xmin_tf = tf.constant(xmin, dtype=tf.float32)
    xmax_tf = tf.constant(xmax, dtype=tf.float32)

    # =======================================================
    #   TRAIN TIPSO-GAN
    # =======================================================
    print("[INFO] Training TIPSO-GAN...")
    t = TIPSOTrainer(input_dim=Xtr.shape[1])
    normals = Xtr[ytr.flatten() == 0]

    if normals.shape[0] < 16:
        print("[WARN] Few/no normal samples → using fallback tiling.")
        normals = np.tile(Xtr[:max(16, len(Xtr)//4)], (2,1))

    t.pretrain_psogan(normals, epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t.train_tipso(
        normals,
        Xtr, ytr,
        Xv, yv,
        epochs=cfg.epochs_tipso,
        batch_size=cfg.batch_size,
        balance_strategy="class_weight"
    )

    # =======================================================
    #   TRAIN BASELINES (LR/RF)
    # =======================================================
    lr, rf = train_baselines(Xtr, ytr)

    # =======================================================
    #   CLEAN PERFORMANCE
    # =======================================================
    clean_tipso, _ = compute_metrics(
        yte, t.dee.predict(Xte, verbose=0).argmax(axis=1)
    )

    clean_lr, _ = safe_eval(lr, Xte, yte, is_keras=False)
    clean_rf, _ = safe_eval(rf, Xte, yte, is_keras=False)

    # =======================================================
    #   CRAFT FGSM/BIM/PGD
    # =======================================================
    print("[INFO] Generating adversarial samples...")
    Xte_fgsm = fgsm(t.dee, Xte, yte, eps, xmin_tf, xmax_tf)
    Xte_bim  = bim(t.dee, Xte, yte, eps, alpha, iters, xmin_tf, xmax_tf)
    Xte_pgd  = pgd_linf(t.dee, Xte, yte, eps, alpha, iters, xmin_tf, xmax_tf, random_start=True)

    # =======================================================
    #   EVALUATE ATTACKS
    # =======================================================
    fgsm_tipso, _ = compute_metrics(yte, t.dee.predict(Xte_fgsm, verbose=0).argmax(axis=1))
    bim_tipso,  _ = compute_metrics(yte, t.dee.predict(Xte_bim,  verbose=0).argmax(axis=1))
    pgd_tipso,  _ = compute_metrics(yte, t.dee.predict(Xte_pgd,  verbose=0).argmax(axis=1))

    fgsm_lr, _ = safe_eval(lr, Xte_fgsm, yte, is_keras=False)
    bim_lr,  _ = safe_eval(lr, Xte_bim,  yte, is_keras=False)
    pgd_lr,  _ = safe_eval(lr, Xte_pgd,  yte, is_keras=False)

    fgsm_rf, _ = safe_eval(rf, Xte_fgsm, yte, is_keras=False)
    bim_rf,  _ = safe_eval(rf, Xte_bim,  yte, is_keras=False)
    pgd_rf,  _ = safe_eval(rf, Xte_pgd,  yte, is_keras=False)

    # =======================================================
    #   SAVE JSON REPORT
    # =======================================================
    report = {
        "dataset": base,
        "params": {"eps": float(eps), "alpha": float(alpha), "iters": iters},
        "clean": {"TIPSO": clean_tipso, "LR": clean_lr, "RF": clean_rf},
        "fgsm":  {"TIPSO": fgsm_tipso, "LR": fgsm_lr, "RF": fgsm_rf},
        "bim":   {"TIPSO": bim_tipso,  "LR": bim_lr,  "RF": bim_rf},
        "pgd":   {"TIPSO": pgd_tipso,  "LR": pgd_lr,  "RF": pgd_rf}
    }

    save_json(os.path.join(outdir, "adaptive_attacks_report.json"), report)
    print(f"[OK] Saved → {outdir}/adaptive_attacks_report.json")

    # =======================================================
    #   SAVE CSV SUMMARY
    # =======================================================
    rows = []

    def row(name, model, metrics):
        rows.append([name, model, metrics.get("accuracy"), metrics.get("precision"),
                     metrics.get("recall"), metrics.get("f1")])

    row("clean", "TIPSO", clean_tipso)
    row("clean", "LR", clean_lr)
    row("clean", "RF", clean_rf)

    row("fgsm", "TIPSO", fgsm_tipso)
    row("fgsm", "LR", fgsm_lr)
    row("fgsm", "RF", fgsm_rf)

    row("bim", "TIPSO", bim_tipso)
    row("bim", "LR", bim_lr)
    row("bim", "RF", bim_rf)

    row("pgd", "TIPSO", pgd_tipso)
    row("pgd", "LR", pgd_lr)
    row("pgd", "RF", pgd_rf)

    with open(os.path.join(outdir, "adaptive_attacks_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attack", "model", "acc", "prec", "rec", "f1"])
        w.writerows(rows)

    print(f"[OK] Saved → {outdir}/adaptive_attacks_summary.csv")


# ===========================================================
#   MAIN
# ===========================================================
def main():
    args = parse_args()
    print("[INFO] Running Adaptive Attacks on datasets:", args.data)

    for df in args.data:
        run_for_dataset(df, args.eps, args.alpha, args.iters)

    print("\n[DONE] All datasets processed successfully.\n")


if __name__ == "__main__":
    main()
