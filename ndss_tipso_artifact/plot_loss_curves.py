#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import csv

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot TIPSO-GAN generator/discriminator loss curves from loss_history_*.csv."
    )
    p.add_argument(
        "--files", "-f",
        nargs="+",
        default=[
            os.path.join("artifacts", "loss_history_cicids2018.csv"),
            os.path.join("artifacts", "loss_history_cicddos2019.csv"),
            os.path.join("artifacts", "loss_history_cicaptiiot.csv"),
        ],
        help=(
            "One or more loss_history_*.csv files to plot.\n"
            "Each file must have columns: epoch,gen_loss,disc_loss.\n"
            "Example:\n"
            "  -f artifacts/loss_history_cicids2018.csv\n"
            "  -f artifacts/loss_history_cicids2018.csv artifacts/loss_history_cicddos2019.csv"
        ),
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Also display the plots interactively (in addition to saving PNGs).",
    )
    return p.parse_args()


def load_loss_csv(path):
    epochs = []
    gen_loss = []
    disc_loss = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                e = int(row["epoch"])
                g = float(row["gen_loss"])
                d = float(row["disc_loss"])
            except (KeyError, ValueError):
                continue
            epochs.append(e)
            gen_loss.append(g)
            disc_loss.append(d)

    return epochs, gen_loss, disc_loss


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    print("[INFO] Plotting loss curves for files:")
    for fp in args.files:
        print("  -", fp)

    for csv_path in args.files:
        if not os.path.isfile(csv_path):
            print(f"[WARN] Skipping {csv_path}: file not found.")
            continue

        base_csv = os.path.basename(csv_path)
        # e.g., loss_history_cicids2018.csv -> cicids2018
        base = os.path.splitext(base_csv)[0]
        if base.startswith("loss_history_"):
            dataset_name = base[len("loss_history_"):]
        else:
            dataset_name = base

        print(f"\n[INFO] Processing {csv_path} (dataset={dataset_name})")

        epochs, gen_loss, disc_loss = load_loss_csv(csv_path)
        if len(epochs) == 0:
            print(f"[WARN] No valid rows in {csv_path}; skipping.")
            continue

        # Plot
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, gen_loss, label="Generator loss")
        plt.plot(epochs, disc_loss, label="Discriminator loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"TIPSO-GAN Loss Curves ({dataset_name})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_path = os.path.join("artifacts", f"loss_curves_{dataset_name}.png")
        plt.savefig(out_path, dpi=300)
        print(f"[OK] Saved plot -> {out_path}")

        if args.show:
            plt.show()
        else:
            plt.close()

    print("\n[DONE] Finished plotting loss curves.")


if __name__ == "__main__":
    main()
