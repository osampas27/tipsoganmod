#!/usr/bin/env python3
import os
import json
import argparse
import matplotlib.pyplot as plt


def load_metrics(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "baseline" not in data or "transfer" not in data:
        raise ValueError(f"{json_path} must contain 'baseline' and 'transfer' keys.")

    return data["baseline"], data["transfer"]


def plot_transfer(baseline, transfer, title, out_prefix):
    metrics = ["accuracy", "precision", "recall", "f1"]

    bl = [baseline.get(m, None) for m in metrics]
    tr = [transfer.get(m, None) for m in metrics]

    # If any metric is None, skip plotting (keeps things honest)
    if any(v is None for v in bl + tr):
        print(f"[WARN] Missing metric(s) for {title}; skipping plot.")
        return

    x = range(len(metrics))

    plt.figure(figsize=(6, 4))
    plt.bar([i - 0.2 for i in x], bl, width=0.4, label="Baseline")
    plt.bar([i + 0.2 for i in x], tr, width=0.4, label="Transfer Learning")

    plt.xticks(list(x), [m.upper() for m in metrics])
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    for ext in ["png", "pdf"]:
        out_path = f"{out_prefix}.{ext}"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Wrote {out_path}")

    plt.close()


def find_json_for_base(base):
    """
    Try several filename patterns so we don't care how run_transfer saved files.
    Priority:
      1) artifacts/dee_transfer_report_<base>.json
      2) artifacts/<base>/dee_transfer_report.json
      3) artifacts/dee_transfer_report.json (legacy for cicids2018)
    """
    candidates = [
        os.path.join("artifacts", f"dee_transfer_report_{base}.json"),
        os.path.join("artifacts", base, "dee_transfer_report.json"),
    ]

    if base.lower() == "cicids2018":
        candidates.append(os.path.join("artifacts", "dee_transfer_report.json"))

    for path in candidates:
        if path and os.path.exists(path):
            return path

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot baseline vs transfer learning performance "
                    "from dee_transfer_report JSON files."
    )
    parser.add_argument(
        "--json",
        default=None,
        help=(
            "Single JSON path (backwards-compatible mode), e.g. "
            "artifacts/dee_transfer_report.json. "
            "If provided, --datasets is ignored."
        ),
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=["cicids2018", "cicddos2019", "cicaptiiot"],
        help=(
            "Dataset bases. For each <base>, the script will look for "
            "dee_transfer_report_<base>.json etc."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default="artifacts/transfer_performance",
        help=(
            "Output prefix for single-json mode. "
            "For multi-dataset mode, figures are saved as "
            "artifacts/transfer_<base>.*"
        ),
    )

    args = parser.parse_args()

    # -------- single-file mode (original behavior) ----------
    if args.json is not None:
        print(f"[INFO] Single-file mode: {args.json}")
        baseline, transfer = load_metrics(args.json)
        plot_transfer(
            baseline,
            transfer,
            title="TIPSO-GAN: Baseline vs Transfer Learning",
            out_prefix=args.out_prefix,
        )
        return

    # -------- multi-dataset mode ----------
    print("[INFO] Multi-dataset mode. Datasets:", args.datasets)

    for base in args.datasets:
        json_path = find_json_for_base(base)
        if json_path is None:
            print(f"[WARN] No dee_transfer_report found for {base}; tried multiple patterns.")
            continue

        try:
            baseline, transfer = load_metrics(json_path)
        except Exception as e:
            print(f"[WARN] Could not load metrics for {base} from {json_path}: {e}")
            continue

        title = f"TIPSO-GAN Transfer Learning – {base.upper()}"
        out_prefix = os.path.join("artifacts", f"transfer_{base}")
        plot_transfer(baseline, transfer, title, out_prefix)

    print("\n[DONE] Transfer learning plots generated.")


if __name__ == "__main__":
    main()
