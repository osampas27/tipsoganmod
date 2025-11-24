# ndss_tipso_artifact_v3_2/run_adaptive_attacks.py
import os, sys, json, csv, argparse
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from tipso_gan.train import TIPSOTrainer, cfg
from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics, save_json
from tipso_gan.attacks import feature_bounds_from_data, fgsm, bim, pgd_linf

def parse_args():
    p = argparse.ArgumentParser("Adaptive attacks against TIPSO-GAN and baselines (NDSS artifact).")
    p.add_argument("--data", "-d", nargs="+", default=None,
                   help="One or more CICIDS-style CSV files. Defaults to cicids2018.csv.")
    p.add_argument("--eps", type=float, default=0.05, help="L_inf epsilon (per-feature scale).")
    p.add_argument("--iters", type=int, default=10, help="Iterations for BIM/PGD.")
    p.add_argument("--alpha", type=float, default=0.01, help="Step size for BIM/PGD.")
    return p.parse_args()

def resolve_data_files(cli_list):
    if cli_list: return cli_list
    env_val = os.environ.get("TIPSO_DATA", "").strip()
    if env_val:
        parts = [s for s in env_val.replace(";", ",").split(",") if s.strip()]
        if parts: return parts
    return ["cicids2018.csv"]

def train_baselines(Xtr, ytr):
    lr = LogisticRegression(max_iter=500, n_jobs=None)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    lr.fit(Xtr, ytr.ravel())
    rf.fit(Xtr, ytr.ravel())
    return lr, rf

def eval_model(model, X, y, is_keras=True):
    if is_keras:
        probs = model.predict(X, verbose=0)
        yp = probs.argmax(axis=1)
    else:
        yp = model.predict(X)
    return compute_metrics(y, yp)

def main():
    args = parse_args()
    data_files = resolve_data_files(args.data)
    os.makedirs("artifacts", exist_ok=True)

    print(f"[INFO] Using data files: {data_files}")
    Xtr, ytr, Xv, yv, Xte, yte, feats = load_cicids_csv_preset(data_files)

    # Bounds for clipping adversarial examples
    xmin, xmax, width = feature_bounds_from_data(Xtr)
    # broadcast shapes for TF ops
    import tensorflow as tf
    xmin_tf = tf.convert_to_tensor(xmin, dtype=tf.float32)
    xmax_tf = tf.convert_to_tensor(xmax, dtype=tf.float32)

    # Train TIPSO-GAN
    t = TIPSOTrainer(input_dim=Xtr.shape[1])
    
    t.pretrain_psogan(Xtr[ytr.flatten()==0], epochs=cfg.epochs_pretrain, batch_size=cfg.batch_size)
    t.train_tipso(Xtr[ytr.flatten()==0], Xtr, ytr, Xv, yv,
                  epochs=cfg.epochs_tipso, batch_size=cfg.batch_size,
                  balance_strategy='class_weight', collect_loss=False)

    # Train simple baselines
    lr, rf = train_baselines(Xtr, ytr)

    # Clean metrics
    clean_tipso, _ = eval_model(t.dee, Xte, yte, is_keras=True)
    clean_lr,   _  = eval_model(lr,   Xte, yte, is_keras=False)
    clean_rf,   _  = eval_model(rf,   Xte, yte, is_keras=False)

    # Craft adversarial examples (white-box for TIPSO; transfer to baselines)
    eps   = np.float32(args.eps)
    alpha = np.float32(args.alpha)
    iters = int(args.iters)

    Xte_fgsm = fgsm(t.dee, Xte, yte, eps, xmin_tf, xmax_tf)
    Xte_bim  = bim(t.dee,  Xte, yte, eps, alpha, iters, xmin_tf, xmax_tf)
    Xte_pgd  = pgd_linf(t.dee, Xte, yte, eps, alpha, iters, xmin_tf, xmax_tf, random_start=True)

    # Evaluate under attack (white-box for TIPSO; transfer attacks for LR/RF)
    fgsm_tipso, _ = eval_model(t.dee, Xte_fgsm, yte, is_keras=True)
    bim_tipso,  _ = eval_model(t.dee, Xte_bim,  yte, is_keras=True)
    pgd_tipso,  _ = eval_model(t.dee, Xte_pgd,  yte, is_keras=True)

    fgsm_lr, _ = eval_model(lr, Xte_fgsm, yte, is_keras=False)
    bim_lr,  _ = eval_model(lr, Xte_bim,  yte, is_keras=False)
    pgd_lr,  _ = eval_model(lr, Xte_pgd,  yte, is_keras=False)

    fgsm_rf, _ = eval_model(rf, Xte_fgsm, yte, is_keras=False)
    bim_rf,  _ = eval_model(rf, Xte_bim,  yte, is_keras=False)
    pgd_rf,  _ = eval_model(rf, Xte_pgd,  yte, is_keras=False)

    # Save JSON report
    report = {
        "params": {"eps": float(eps), "alpha": float(alpha), "iters": iters},
        "clean": {
            "TIPSO": clean_tipso,
            "LR": clean_lr,
            "RF": clean_rf
        },
        "fgsm": {
            "TIPSO": fgsm_tipso,
            "LR": fgsm_lr,
            "RF": fgsm_rf
        },
        "bim": {
            "TIPSO": bim_tipso,
            "LR": bim_lr,
            "RF": bim_rf
        },
        "pgd": {
            "TIPSO": pgd_tipso,
            "LR": pgd_lr,
            "RF": pgd_rf
        }
    }
    save_json("artifacts/adaptive_attacks_report.json", report)
    print("Wrote artifacts/adaptive_attacks_report.json")

    # Also a CSV summary (accuracy, recall, f1)
    rows = []
    def pick(m):  # compact summary
        return {
            "acc": m.get("accuracy"),
            "prec": m.get("precision"),
            "rec": m.get("recall"),
            "f1": m.get("f1")
        }
    rows.append(["clean","TIPSO", *pick(clean_tipso).values()])
    rows.append(["clean","LR",    *pick(clean_lr).values()])
    rows.append(["clean","RF",    *pick(clean_rf).values()])

    rows.append(["fgsm","TIPSO", *pick(fgsm_tipso).values()])
    rows.append(["fgsm","LR",    *pick(fgsm_lr).values()])
    rows.append(["fgsm","RF",    *pick(fgsm_rf).values()])

    rows.append(["bim","TIPSO", *pick(bim_tipso).values()])
    rows.append(["bim","LR",    *pick(bim_lr).values()])
    rows.append(["bim","RF",    *pick(bim_rf).values()])

    rows.append(["pgd","TIPSO", *pick(pgd_tipso).values()])
    rows.append(["pgd","LR",    *pick(pgd_lr).values()])
    rows.append(["pgd","RF",    *pick(pgd_rf).values()])

    with open("artifacts/adaptive_attacks_summary.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["attack","model","acc","prec","rec","f1"]); w.writerows(rows)

    print("Wrote artifacts/adaptive_attacks_summary.csv")

if __name__ == "__main__":
    main()
