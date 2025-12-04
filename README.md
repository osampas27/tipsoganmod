# TIPSO-GAN: Malicious Network Traffic Detection Using a Novel Optimized Generative Adversarial Network

This artifact accompanies the NDSS 2026 submission:

> **TIPSO-GAN: Malicious Network Traffic Detection Using a Novel Optimized Generative Adversarial Network**

The package provides a fully executable Python environment to reproduce the paper’s key experiments on CPU (and GPU if available). It includes runnable scripts, configuration, full CICIDS-2018 dataset, and outputs to validate results/claims in the paper.

---

## 📦 Contents

| Path | What it contains |
|---|---|
| `ndss_tipso_artifact/tipso_gan/` | Core implementation (trainer, loader, metrics, simple attack routines, config). |
| `ndss_tipso_artifact/*.py` | Experiment drivers: `run_repro_perf.py`, `run_unseen_attack_eval.py`, `run_transfer.py`, `run_balance_eval.py`, `run_compare_baselines.py`, `run_cost_profile.py`, `run_ablation_pso.py`, `run_ablation_attention.py`, `run_adaptive_attacks.py`, 'plot_adaptive_attacks.py', 'plot_cost_profile.py', 'plot_loss_curves.py', 'plot_transfer_performance.py'. |
| `cicids2018.csv` | CICIDS-2018 tabular dataset |
| `cicddos2019.csv` | link to cicddos-2019 tabular dataset |
| `cicaptiiot.csv` | link to CICAPT-IIoT2024 tabular dataset |
| `artifacts/` | All generated results (CSV/JSON) are written here by each script. |
| `ndss_tipso_artifact/requirements.txt` | Python dependencies (TF 2.15, sklearn, numpy, pandas). |
| `ndss_tipso_artifact/README.md` | This file. |

> Note: For review, the artifact is uploaded via the NDSS HotCRP portal and also made public (https://github.com/osampas27/tipsoganmod).

---

## ⚙️ Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / Ubuntu 22.04 | Ubuntu 22.04 |
| Python | 3.10–3.11 | 3.11 (conda env) |
| TensorFlow | 2.15 (CPU) | 2.15 (GPU build if available) |
| RAM | 32–64 GB | HIGHER GB |
| GPU | Optional | NVIDIA GTX 1080 Ti |
| Disk | 20+ GB |

---

## 🪜 Installation

```bash
cd ndss_tipso_artifact

# Create & activate an environment
conda create -n tipso python=3.11 -y
conda activate tipso

# Install deps
pip install -r requirements.txt

# If TensorFlow import fails (Windows CPU):
python -m pip install tensorflow==2.15.0
```

---

## ▶️ Quick Start

Run the minimal working example (uses the included lightweight dataset):

```bash
python ndss_tipso_artifact/run_repro_perf.py
```

You should see training logs and this message at the end:

```
Wrote artifacts/perf_summary.json, confusion_matrix.json, loss_history.csv
```

## 📥 Using Full CICIDS-2018 (or Your Own CSVs)

Scripts accept CSV paths via CLI or environment variable. Columns must be `f0..fK` feature columns and a `label` column (`0=benign`, `1=attack`).

**Option A — CLI**

```bash
python ndss_tipso_artifact/run_repro_perf.py --data /path/to/cicids2018.csv
```

**Option B — Environment variable**

```bash
set TIPSO_DATA=/path/to/cicids2018.csv          # Windows
export TIPSO_DATA=/path/to/cicids2018.csv       # Linux/macOS
python ndss_tipso_artifact/run_repro_perf.py
```

You can pass multiple CSVs (comma or semicolon separated) and they’ll be concatenated.

---

## 🧩 Configuration

Training knobs live in `tipso_gan/config.py` (names may vary slightly by branch):

* `epochs_pretrain` (default ~5): proxy pretrain epochs  
* `epochs_tipso` (default ~15): main detector epochs  
* `batch_size` (default ~64)  
* `lr` (default `2e-4`)

> Dataset selection is **not** set in config; use `--data` or `TIPSO_DATA` (see above).

---

## 📊 Expected Outputs

After running the suite you’ll see:

```
ndss_tipso_artifact/artifacts/
 ├── perf_summary.json
 ├── confusion_matrix.json
 ├── loss_history.csv
 │
 ├── balance_grid.csv
 ├── baselines_perf.json
 ├── cost_metrics.json
 │
 ├── dee_transfer_report.json
 ├── dee_transfer_report_cicids2018.json
 ├── dee_transfer_report_cicddos2019.json
 ├── dee_transfer_report_cicaptiiot.json
 │
 ├── adaptive_attacks_report.json
 ├── adaptive_attacks_summary.csv
 │
 ├── plots/
 │   ├── transfer_performance_cicids2018.png
 │   ├── transfer_performance_cicddos2019.png
 │   ├── transfer_performance_cicaptiiot.png
 │   │
 │   ├── cost_profile.png
 │   ├── cost_profile.pdf
 │   │
 │   ├── loss_curve_cicids2018.png
 │   ├── loss_curve_cicddos2019.png
 │   ├── loss_curve_cicaptiiot.png
 │   │
 │   ├── adaptive_fgsm.png
 │   ├── adaptive_bim.png
 │   ├── adaptive_pgd.png
 │   └── adaptive_summary_bar.png
 │
 └── y_test.npy   (safe to keep, harmless if needed later)

```

* `perf_summary.json`: accuracy/precision/recall/F1 + train/test wall times (C1) 
* `loss_history.csv`: per-epoch proxy/classifier losses (for plotting C2)  
* `adaptive_attacks`: clean vs FGSM/BIM/PGD across TIPSO/LR/RF (C3)

---

## 🧾 Mapping to Paper Results
Each `run_*.py` script validates one or more paper claims (C1–C5). All outputs land in `artifacts/`.

All scripts map directly to paper claims (C1–C5), and all outputs are saved
into per-dataset folders under artifacts/.

| Claim | Summary | Validated By | Output Files |
|------|---------|--------------|--------------|
| C1 | TIPSO-GAN achieves strong detection metrics (Table II). | run_repro_perf.py, run_compare_baselines.py | perf_summary_multi.json, baselines_perf_*.json, confusion_matrix_*.json |
| C2 | Training curves of TIPSO-GAN (Fig. 7). | run_loss_curves.py | loss_history_*.csv, loss_curves_*.png |
| C3 | Robustness under FGSM, BIM, PGD (Fig. 8). | run_adaptive_attacks.py | adaptive_attacks_report.json, adaptive_attacks_summary.csv, adaptive_attacks_*.png |
| C4 | Transfer-learning benefit (Fig. 9). | run_transfer.py | dee_transfer_report_*.json, transfer_*.png |
| C5 | Effect of class-balance strategies (Fig. 10). | run_balance_eval.py | balance_grid_*.csv, preds_*.npy |
| — | Cost/latency analysis. | run_cost_profile.py | cost_metrics_*.json, cost_latency.png |
---

## ⏱ Time Budget

Times scale with dataset size and hardware resource capacity; bigger datasets takes longer.

---

## 🧩 Troubleshooting

| Issue | Fix |
| ------ | --- |
| `ModuleNotFoundError: tensorflow` | Ensure the env is active (`conda activate tipso`) and install `tensorflow==2.15.0`. |
| No GPU visible | The artifact targets CPU by default; GPU is optional. |
| Different numbers vs paper | Minor differences are expected due to random seeds/hardware; trends should match. |
| Slow training | Lower `epochs_pretrain` / `epochs_tipso` in `tipso_gan/config.py`. |
| CSV schema mismatch | Ensure features are `f0..fK` and there’s a `label` column |

---

## Datasets and CSV configuration files

For CIC-DDoS2019 and CICAPT-IIoT, the CSV files included here serve as placeholders and contain only hyperlinks to the official dataset repositories. 
Users should download the full datasets from the linked sources before running the corresponding scripts.
We provide **the full dataset for cicids2018**, For cicddos2019 and cicaptiiot, each contain:

- The official download links for the dataset files
- Any comments/notes needed to prepare them for TIPSO-GAN

The following CSV configuration files are included:

- `cicids2018.csv`
- `cicddos2019.csv`
- `cicaptiiot.csv`

By default, the scripts process all three datasets in one run.
Each evaluation script accepts one or more of these CSV configuration files via the `--data/-d` argument:

```python
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
---
© 2025 The Authors. All rights reserved.

The authors retain full copyright over all materials contained in this repository and its associated Zenodo record (DOI: https://doi.org/10.5281/zenodo.17759517).

*End of README.md*
