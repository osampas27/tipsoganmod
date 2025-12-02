#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from tipso_gan.cicids_loader import load_cicids_csv_preset
from tipso_gan.metrics import compute_metrics, save_json
from tipso_gan.train import TIPSOTrainer, cfg


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare TIPSO-GAN with baseline MLP and Tiny Transformer on one or more datasets."
    )
    p.add_argument(
        "--data", "-d",
        nargs="+",
        default=["cicids2018.csv", "cicddos2019.csv", "cicaptiiot.csv"],
        help=(
            "One or more CIC-style CSV files, e.g.\n"
            "  -d cicids2018.csv cicddos2019.csv cicaptiiot.csv\n"
            "Each dataset is processed separately and gets its own baselines_perf_<base>.json."
        ),
    )
    return p.parse_args()


def build_baseline_mlp(input_dim: int) -> tf.keras.Model:
    i = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(i)
    x = layers.Dense(128, activation="relu")(x)
    o = layers.Dense(2, activation="softmax")(x)
    return models.Model(i, o, name="BaselineMLP")


def build_tiny_transformer(input_dim: int) -> tf.keras.Model:
    # Not a real transformer, but a lightweight attention-style MLP baseline
    i = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(i)
    q = layers.Dense(64, activation="relu")(x)
    k = layers.Dense(64, activation="relu")(x)
    v = layers.Dense(64, activation="relu")(x)
    # Simple fusion of original features + "value" path
    x = layers.Concatenate()([x, v])
    x = layers.Dense(128, activation="relu")(x)
    o = layers.Dense(2, activation="softmax")(x)
    return models.Model(i, o, name="TinyTransformer")


def train_and_eval(model: tf.keras.Model,
                   Xtr, ytr, Xv, yv, Xte, yte):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    model.fit(
        Xtr, ytr,
        validation_data=(Xv, yv),
        epochs=10,
        batch_size=cfg.batch_size,
        verbose=0,
    )
    ypred = model.predict(Xte, verbose=0).argmax(axis=1)
    metrics, _ = compute_metrics(yte, ypred)
    return metrics


def run_baselines_for_dataset(data_file: str):
    base = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\n[INFO] === Baseline comparison for dataset: {data_file} (base={base}) ===")

    os.makedirs("artifacts", exist_ok=True)

    # Load dataset
    Xtr, ytr, Xv, yv, Xte, yte, feats = load_cicids_csv_preset([data_file])
    print(f"[INFO] {base}: Xtr={Xtr.shape}, ytr={ytr.shape}, Xv={Xv.shape}, Xte={Xte.shape}")

    # ----- Robust X_normal selection for TIPSO-GAN -----
    mask_n   = (ytr.flatten() == 0)
    num_norm = int(mask_n.sum())
    print(f"[INFO] {base}: train samples = {len(ytr)}, normals(label=0) = {num_norm}")

    if num_norm >= 16:
        Xn = Xtr[mask_n]
    else:
        # Fallback to a subset if there are few/no true normals
        Xn = Xtr[:max(16, len(Xtr) // 4)]
        print(
            f"[WARN] {base}: few/no label-0 samples in train. "
            "Using subset of Xtr as pseudo-normal for GAN training."
        )

    print(f"[INFO] {base}: using {Xn.shape[0]} samples as X_normal")

    # ----- Train TIPSO-GAN -----
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

    tipso_preds = t.dee.predict(Xte, verbose=0).argmax(axis=1)
    tipso_metrics, _ = compute_metrics(yte, tipso_preds)

    # ----- Baseline MLP -----
    mlp_metrics = train_and_eval(
        build_baseline_mlp(Xtr.shape[1]),
        Xtr, ytr, Xv, yv, Xte, yte,
    )

    # ----- Tiny Transformer-like baseline -----
    tiny_metrics = train_and_eval(
        build_tiny_transformer(Xtr.shape[1]),
        Xtr, ytr, Xv, yv, Xte, yte,
    )

    # ----- Save per-dataset metrics -----
    out_path = os.path.join("artifacts", f"baselines_perf_{base}.json")
    payload = {
        "dataset": data_file,
        "TIPSO_GAN": tipso_metrics,
        "BaselineMLP": mlp_metrics,
        "TinyTransformer": tiny_metrics,
    }
    save_json(out_path, payload)
    print(f"[OK] {base}: wrote {out_path}")

    # Backwards compatibility: keep original name for cicids2018
    if base.lower() == "cicids2018":
        legacy_path = os.path.join("artifacts", "baselines_perf.json")
        save_json(legacy_path, payload)
        print(f"[OK] {base}: also wrote {legacy_path} (legacy name)")


def main():
    args = parse_args()
    print("[INFO] Datasets for baseline comparison:", args.data)

    for data_file in args.data:
        run_baselines_for_dataset(data_file)

    print("\n[DONE] Finished baseline comparison for all datasets.")


if __name__ == "__main__":
    main()
