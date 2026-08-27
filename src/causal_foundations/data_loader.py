import pandas as pd
import numpy as np
from pathlib import Path


# ── 2.1.1 Data Loader ────────────────────────────────────────────────────────

def load_criteo_data(path: str = "data/raw/criteo-uplift.csv",
                     sample_n: int = 100_000,
                     seed: int = 42) -> pd.DataFrame:
    """
    Load Criteo Uplift Modeling Dataset.
    Sample subset untuk efisiensi komputasi lokal.
    Kolom: treatment, visit, conversion, f0-f11
    """
    df = pd.read_csv(path)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    print(f"Loaded: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Treatment rate: {df['treatment'].mean():.3f}")
    return df


# ── 2.1.2 Inject Confounding ──────────────────────────────────────────────────

def inject_confounding(df: pd.DataFrame,
                       confounder_col: str = "f1",
                       strength: float = 2.0,
                       seed: int = 42) -> pd.DataFrame:
    """
    Suntikkan confounding buatan untuk mensimulasikan data observational.
    Simpan treatment RCT asli sebagai ground-truth pembanding.

    Desain metodologis: karena kita TAHU ground truth ATE dari RCT,
    kita bisa validasi apakah DML/Causal Forest berhasil memulihkan
    estimasi yang benar meski bekerja dari data observational yang bias.
    """
    rng = np.random.default_rng(seed)
    df  = df.copy()

    df["treatment_rct_original"] = df["treatment"]  # simpan ground truth

    propensity_true = 1 / (1 + np.exp(
        -strength * (df[confounder_col] - df[confounder_col].mean())
    ))

    df["treatment_observational"] = rng.binomial(1, propensity_true)
    df["true_propensity"]         = propensity_true

    print(f"\nConfounding injected (strength={strength}):")
    print(f"  RCT treatment rate         : {df['treatment_rct_original'].mean():.3f}")
    print(f"  Observational treatment rate: {df['treatment_observational'].mean():.3f}")
    print(f"  Propensity range           : [{propensity_true.min():.3f}, {propensity_true.max():.3f}]")

    return df


def simulate_criteo_data(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """
    Simulasi data mirip Criteo jika file asli tidak tersedia.
    Distribusi disesuaikan dengan statistik publik Criteo dataset.
    """
    rng = np.random.default_rng(seed)

    n_features = 12
    X = rng.normal(0, 1, size=(n, n_features))
    feature_cols = {f"f{i}": X[:, i] for i in range(n_features)}

    treatment = rng.binomial(1, 0.5, size=n)

    true_effect = 0.03 + 0.02 * X[:, 0] + 0.01 * X[:, 1]
    baseline    = 0.04 + 0.01 * X[:, 2]
    visit_prob  = np.clip(baseline + treatment * true_effect, 0, 1)
    visit       = rng.binomial(1, visit_prob)

    conv_prob   = np.clip(0.01 + 0.005 * treatment + 0.003 * X[:, 3], 0, 1)
    conversion  = rng.binomial(1, conv_prob)

    df = pd.DataFrame(feature_cols)
    df["treatment"]  = treatment
    df["visit"]      = visit
    df["conversion"] = conversion

    print(f"Simulated Criteo-like data: {df.shape}")
    print(f"Treatment rate: {df['treatment'].mean():.3f}")
    print(f"Visit rate    : {df['visit'].mean():.3f}")
    print(f"Conversion rate: {df['conversion'].mean():.3f}")

    return df


if __name__ == "__main__":
    # Coba load file asli, fallback ke simulasi
    criteo_path = Path("data/raw/criteo-uplift.csv")

    if criteo_path.exists():
        df = load_criteo_data(str(criteo_path), sample_n=100_000)
    else:
        print("File Criteo tidak ditemukan — menggunakan data simulasi")
        df = simulate_criteo_data(n=100_000)
        df.to_csv("data/raw/criteo_simulated.csv", index=False)

    # Inject confounding
    df_obs = inject_confounding(df, confounder_col="f1", strength=2.0)

    df_obs.to_csv("data/processed/criteo_with_confounding.csv", index=False)
    print(f"\n✅ Saved: data/processed/criteo_with_confounding.csv")
    print(f"Shape: {df_obs.shape}")