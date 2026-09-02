# CausalForge — Causal Inference Engine

> End-to-end causal inference pipeline combining PSM, IPW, Double Machine Learning, Causal Forest, Synthetic Control, and Sensitivity Analysis on Criteo-scale data.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![EconML](https://img.shields.io/badge/EconML-Microsoft-orange)
![DoWhy](https://img.shields.io/badge/DoWhy-Microsoft-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Masalah

*"Correlation is not causation"* — tapi bagaimana membuktikannya secara kuantitatif?

Dataset observational (non-RCT) mengandung confounding yang membuat naive comparison menyesatkan. CausalForge mendemonstrasikan secara empiris — dengan ground truth RCT yang diketahui — seberapa besar bias ini dan bagaimana berbagai metode causal inference mengatasinya.

---

---
## Arsitektur Pipeline
Criteo Dataset (100K rows, RCT ground truth)
│
├──► Confounding Injection (strength=2.0)
│ └── Simulasi data observational dengan ground truth tersimpan
│
├──► Overlap Check (Propensity Score)
│ └── Verifikasi asumsi positivity secara empiris
│
├──► Naive Estimation → Bias 98.3%
│ └── Demonstrasi konkret mengapa korelasi ≠ kausalitas
│
├──► Propensity Score Methods
│ ├── PSM (Nearest-Neighbor, caliper=0.05) → Bias 61.7%
│ └── IPW (Stabilized weights, trimming) → Bias 85.0%
│
├──► Double Machine Learning (Chernozhukov et al. 2018)
│ ├── Manual implementation (Frisch-Waugh-Lovell)
│ └── EconML LinearDML → Bias 93.3%
│
├──► Causal Forest (Wager & Athey 2018)
│ ├── CATE per individu (heterogeneous treatment effects)
│ ├── Uplift curve + policy recommendation
│ └── Top 20% CATE = 4.7x lebih tinggi dari rata-rata
│
├──► Synthetic Control (Abadie & Gardeazabal 2003)
│ ├── Panel data estimation → Bias 0.1%
│ └── In-space placebo test
│
└──► Sensitivity & Validation
├── Manski Bounds (partial identification)
├── Robustness Value (Cinelli & Hazlett 2020)
└── DoWhy Refutation Tests (2/3 passed)
---


---

## Hasil Utama

### Method Comparison

| Method | ATE Estimate | Bias (%) | Assumption |
|---|---|---|---|
| **Ground Truth (RCT)** | **0.0060** | **0%** | Randomization |
| PSM | 0.0023 | 61.7% | Unconfoundedness + Overlap |
| IPW | 0.0009 | 85.0% | Unconfoundedness + Positivity |
| DML (EconML) | 0.0004 | 93.3% | Unconfoundedness + Partial Linearity |
| Causal Forest | 0.0003 | 95.0% | Unconfoundedness + Overlap |
| **Naive** | **0.0001** | **98.3%** | None (biased) |
| **Synthetic Control** | **2.0016** | **0.1%** | Common trends (panel data) |

### Heterogeneous Treatment Effects
- CATE std: 0.0102 — heterogenitas efek signifikan
- Top 20% unit: CATE = 0.0126 (4.7x rata-rata)
- Policy recommendation: targetkan 20,000 unit teratas

### Sensitivity Analysis
- Manski Bounds: [-0.499, 0.501] → DML mempersempit ke [-0.001, 0.002]
- Robustness Value: 0.000 (fragile — confounding sangat kuat)
- Refutation Tests: 2/3 passed (random cause ✅, placebo ✅, subset ❌)

---

## Quickstart

```bash
git clone https://github.com/thoriqalkatiri710-oss/causalforge.git
cd causalforge
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Generate data
python src/causal_foundations/data_loader.py

# Overlap check
python src/causal_foundations/overlap_check.py

# Naive estimation (demonstrasi bias)
python src/causal_foundations/naive_estimation.py

# Propensity methods
python src/estimation/propensity_methods.py

# Double Machine Learning
python src/estimation/dml_estimator.py

# Causal Forest
python src/forests/causal_forest.py

# Synthetic Control
python src/synthetic_control/synthetic_control.py

# Sensitivity Analysis
python src/sensitivity/sensitivity_analysis.py

# Final comparison
python src/estimation/comparison.py

# Refutation tests
python src/estimation/refutation.py
```

---

## Output Files

| File | Deskripsi |
|---|---|
| `results/figures/overlap_rct.png` | Propensity score distribution (RCT vs observational) |
| `results/figures/bias_comparison.png` | ATE estimates vs ground truth |
| `results/figures/psm_balance.png` | Covariate balance after PSM (Love plot) |
| `results/figures/dml_residuals.png` | DML partial regression plot |
| `results/figures/cate_distribution.png` | CATE distribution + uplift curve |
| `results/figures/causal_forest_importance.png` | Feature importance (heterogeneity drivers) |
| `results/figures/synthetic_control.png` | Synthetic control vs treated unit |
| `results/figures/synthetic_control_placebo.png` | In-space placebo test |
| `results/figures/sensitivity_contour.png` | Sensitivity contour plot |
| `results/figures/identification_bounds.png` | Manski bounds vs DML CI |
| `results/figures/final_comparison.png` | Final method comparison |
| `results/method_comparison.csv` | Summary table semua metode |
| `results/ate_estimates.json` | ATE estimates per metode |
| `results/refutation_results.csv` | Refutation test results |

---


---
## Struktur Project
causalforge/
├── src/
│ ├── causal_foundations/ # Data loader, confounding, overlap check
│ ├── estimation/ # PSM, IPW, DML, comparison, refutation
│ ├── forests/ # Causal Forest + CATE analysis
│ ├── sensitivity/ # Manski bounds, robustness value
│ └── synthetic_control/ # SCM + placebo test
├── data/
│ ├── raw/ # Criteo dataset (atau simulasi)
│ └── processed/ # Confounded data + propensity scores
├── results/
│ ├── figures/ # 11 visualisasi
│ └── *.json, *.csv # Estimates dan validation tables
├── notebooks/ # Eksplorasi
├── tests/ # Unit tests
└── docs/ # Dokumentasi
---


---

## Keterbatasan

- Data simulasi (bukan Criteo asli) karena keterbatasan akses
- True ATE sangat kecil (0.006) → signal-to-noise rendah → semua metode bias tinggi
- Synthetic Control hanya valid untuk panel data (unit × waktu), bukan cross-sectional
- Robustness value = 0 bukan berarti metode salah, tapi efek memang tidak signifikan

---

## Rujukan Akademik

1. Chernozhukov et al. (2018) — Double/Debiased Machine Learning
2. Wager & Athey (2018) — Estimation and Inference of Heterogeneous Treatment Effects
3. Abadie & Gardeazabal (2003) — Synthetic Control Method
4. Manski (1990) — Nonparametric Bounds on Treatment Effects
5. Cinelli & Hazlett (2020) — Making Sense of Sensitivity
6. Rosenbaum & Rubin (1983) — Propensity Score Matching
7. Hirano & Imbens (2001) — Inverse Propensity Weighting

---

## Library Utama

- **EconML** (Microsoft Research) — DML, Causal Forest
- **DoWhy** (Microsoft Research) — Causal graph + refutation
- **doubleml** — Production-grade DML
- **scikit-learn** — PSM, nuisance models
- **statsmodels** — Regression, inference