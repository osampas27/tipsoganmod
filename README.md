# TIPSO-GAN: Malicious Network Traffic Detection Using a Novel Optimized Generative Adversarial Network

This artifact accompanies the NDSS 2026 submission:

> **TIPSO-GAN: Malicious Network Traffic Detection Using a Novel Optimized Generative Adversarial Network**

The package provides a fully executable Python environment to reproduce the paper’s key experiments on CPU (and GPU if available). It includes runnable scripts, configuration, a lightweight CICIDS-2018–style sample dataset, and outputs to validate claims.

---

## 📦 Contents

| Path | What it contains |
|---|---|
| `ndss_tipso_artifact/tipso_gan/` | Core implementation (trainer, loader, metrics, simple attack routines, config). |
| `ndss_tipso_artifact/*.py` | Experiment drivers: `run_repro_perf.py`, `run_unseen_attack_eval.py`, `run_transfer.py`, `run_balance_eval.py`, `run_compare_baselines.py`, `run_cost_profile.py`, `run_ablation_pso.py`, `run_ablation_attention.py`, `run_adaptive_attacks.py`. |
| `ndss_tipso_artifact/data/sample_cicids_small.csv` | Minimal CICIDS-2018 tabular dataset (`f0..f19,label`) for fast CPU tests. |
| `ndss_tipso_artifact/artifacts/` | All generated results (CSV/JSON) are written here by each script. |
| `ndss_tipso_artifact/requirements.txt` | Python dependencies (TF 2.15, sklearn, numpy, pandas). |
| `ndss_tipso_artifact/artifact_appendix.tex` | NDSS Artifact Appendix (≤2 pages). |
| `ndss_tipso_artifact/README.md` | This file. |

> Note: For review, the artifact is uploaded via the NDSS HotCRP portal (not public GitHub).

---

## ⚙️ Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / Ubuntu 22.04 | Ubuntu 22.04 |
| Python | 3.10–3.11 | 3.11 (conda env) |
| TensorFlow | 2.15 (CPU) | 2.15 (GPU build if available) |
| RAM | 8–16 GB | 16–32 GB |
| GPU | Optional | NVIDIA GTX 1080 Ti |
| Disk | ~1 GB (sample) | 20+ GB for full CICIDS-2018 |

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

---

## 🧠 Experiment Workflow (Scripts → Claims)

Each `run_*.py` script validates one or more paper claims (C1–C8). All outputs land in `artifacts/`.

| Experiment | Script | Outputs | Claim(s) |
| ----------- | ------- | -------- | -------- |
| Detection performance (Acc/Prec/Rec/F1, CM, times) | `run_repro_perf.py` | `perf_summary.json`, `confusion_matrix.json`, `loss_history.csv` | C1, C2 |
| Transfer learning effect (DeePred) | `run_transfer.py` | `dee_transfer_report.json` | C3 |
| Class balancing effectiveness | `run_balance_eval.py` | `balance_grid.csv` | C4 |
| Unseen attack generalization | `run_unseen_attack_eval.py` | `unseen_report.json` | C5 |
| PSO proxy ablation | `run_ablation_pso.py` | `pso_vs_static.csv` | C6 |
| Attention ablation (toy toggle) | `run_ablation_attention.py` | `attention_ablation.csv` | C6 |
| Baselines comparison (LR/RF) | `run_compare_baselines.py` | `baselines_perf.json` | C1, C7 |
| Cost profile (train/infer time, params) | `run_cost_profile.py` | `cost_metrics.json` | C7 |
| Adaptive attacks (FGSM/BIM/PGD) vs TIPSO/LR/RF** | `run_adaptive_attacks.py` | `adaptive_attacks_report.json`, `adaptive_attacks_summary.csv` | C8 |

Typical CPU runtime: ~5–10 min per script on the sample dataset.

---

## 📥 Using Full CICIDS-2018 (or Your Own CSVs)

Scripts accept CSV paths via CLI or environment variable. Columns must be `f0..fK` feature columns and a `label` column (`0=benign`, `1=attack`).

**Option A — CLI**

```bash
python ndss_tipso_artifact/run_repro_perf.py --data /path/to/CICIDS2018.csv
```

**Option B — Environment variable**

```bash
set TIPSO_DATA=/path/to/CICIDS2018.csv          # Windows
export TIPSO_DATA=/path/to/CICIDS2018.csv       # Linux/macOS
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
 ├── dee_transfer_report.json
 ├── unseen_report.json
 ├── balance_grid.csv
 ├── pso_vs_static.csv
 ├── attention_ablation.csv
 ├── baselines_perf.json
 ├── cost_metrics.json
 ├── adaptive_attacks_report.json
 └── adaptive_attacks_summary.csv
```

* `perf_summary.json`: accuracy/precision/recall/F1 + train/test wall times  
* `loss_history.csv`: per-epoch proxy/classifier losses (for plotting C2)  
* `adaptive_attacks`: clean vs FGSM/BIM/PGD across TIPSO/LR/RF (C8)

---

## 🧾 Mapping to Paper Claims

| Claim | Summary | Validated by |
| ------ | -------- | ------------ |
| C1 | TIPSO-GAN achieves competitive detection metrics. | `run_repro_perf.py`, `run_compare_baselines.py` |
| C2 | Training curves behave as expected within ~100 epochs. | `run_repro_perf.py` → `loss_history.csv` |
| C3 | DeePred transfer learning improves performance. | `run_transfer.py` |
| C4 | Class balancing improves minority detection. | `run_balance_eval.py` |
| C5 | Robustness on unseen attack types. | `run_unseen_attack_eval.py` |
| C6 | PSO proxy & attention components are beneficial. | `run_ablation_pso.py`, `run_ablation_attention.py` |
| C7 | Computational cost comparable to GAN baselines. | `run_cost_profile.py` |
| C8 | Performance under adaptive attacks (FGSM/BIM/PGD) vs baselines. | `run_adaptive_attacks.py` |

---

## ⏱ Time Budget (Sample Dataset)

| Phase | Human | Compute (CPU) |
| ------ | ------ | ------------- |
| Setup | ~10 min | — |
| Minimal example | ~2 min | ~3–5 min |
| Full suite (all runs) | ~10–15 min | ~30–60 min |

Times scale with dataset size; full CICIDS-2018 takes longer.

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

*End of README.md*
