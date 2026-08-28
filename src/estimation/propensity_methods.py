import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# ── 4.1.1 Propensity Score Matching ──────────────────────────────────────────

def propensity_score_matching(df: pd.DataFrame,
                               treatment_col: str,
                               propensity_col: str,
                               caliper: float = 0.05) -> tuple:
    """
    Nearest-neighbor PSM dengan caliper.
    Caliper: buang match yang terlalu jauh (mencegah bad matches).
    """
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1).fit(control[[propensity_col]])
    distances, indices = nn.kneighbors(treated[[propensity_col]])

    valid           = distances.flatten() <= caliper
    matched_treated = treated[valid].reset_index(drop=True)
    matched_control = control.iloc[indices.flatten()[valid]].reset_index(drop=True)

    match_rate = len(matched_treated) / len(treated) * 100
    print(f"\n── PSM Results ──")
    print(f"Matched pairs  : {len(matched_treated)} dari {len(treated)} treated ({match_rate:.1f}%)")
    print(f"Caliper        : {caliper}")

    return matched_treated, matched_control


def check_covariate_balance(matched_treated: pd.DataFrame,
                             matched_control: pd.DataFrame,
                             covariates: list) -> pd.DataFrame:
    """
    Verifikasi covariate balance setelah matching.
    SMD < 0.1 = balanced (standar industri).
    Langkah wajib yang sering dilewatkan di portofolio lain.
    """
    results = []
    for col in covariates:
        mean_t     = matched_treated[col].mean()
        mean_c     = matched_control[col].mean()
        pooled_std = np.sqrt((matched_treated[col].var() + matched_control[col].var()) / 2)
        smd        = (mean_t - mean_c) / (pooled_std + 1e-10)
        results.append({
            "covariate": col,
            "mean_treated": round(mean_t, 4),
            "mean_control": round(mean_c, 4),
            "smd":          round(smd, 3),
            "balanced":     abs(smd) < 0.1
        })

    df_balance = pd.DataFrame(results)
    n_balanced = df_balance["balanced"].sum()
    print(f"\n── Covariate Balance ──")
    print(f"Balanced (SMD<0.1): {n_balanced}/{len(covariates)}")
    print(df_balance[["covariate", "smd", "balanced"]].to_string(index=False))

    return df_balance


def plot_balance(df_balance: pd.DataFrame, save_path: str = None):
    """Love plot — visualisasi standar untuk covariate balance."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#22C55E" if b else "#EF4444" for b in df_balance["balanced"]]
    ax.barh(df_balance["covariate"], df_balance["smd"].abs(), color=colors)
    ax.axvline(0.1, color="black", linestyle="--", label="SMD=0.1 threshold")
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_title("Covariate Balance After PSM\n(Hijau = balanced, Merah = tidak balanced)")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


def psm_ate(matched_treated: pd.DataFrame,
            matched_control: pd.DataFrame,
            outcome_col: str) -> float:
    """Hitung ATE dari matched sample."""
    return matched_treated[outcome_col].mean() - matched_control[outcome_col].mean()


# ── 4.2.1 Inverse Propensity Weighting ───────────────────────────────────────

def ipw_estimator(df: pd.DataFrame,
                  treatment_col: str,
                  outcome_col: str,
                  propensity_col: str,
                  trim_threshold: float = 0.01) -> dict:
    """
    IPW estimator untuk ATE.
    Formula: ATE = (1/n) Σ [T·Y/e(X) - (1-T)·Y/(1-e(X))]
    Trim propensity ekstrem untuk mencegah variance explosion.

    Kelemahan IPW (motivasi DML): sangat sensitif terhadap propensity
    ekstrem → variance besar. Ini motivasi utama beralih ke DML.
    """
    df_trim = df[
        (df[propensity_col] > trim_threshold) &
        (df[propensity_col] < 1 - trim_threshold)
    ].copy()

    n_trimmed = len(df) - len(df_trim)

    weights_treated = df_trim[treatment_col] / df_trim[propensity_col]
    weights_control = (1 - df_trim[treatment_col]) / (1 - df_trim[propensity_col])

    ate = (
        (df_trim[outcome_col] * weights_treated).sum() / weights_treated.sum() -
        (df_trim[outcome_col] * weights_control).sum() / weights_control.sum()
    )

    # Weight diagnostics
    all_weights = np.where(
        df_trim[treatment_col] == 1,
        weights_treated,
        weights_control
    )

    print(f"\n── IPW Results ──")
    print(f"ATE IPW        : {ate:.4f}")
    print(f"N trimmed      : {n_trimmed} ({n_trimmed/len(df)*100:.1f}%)")
    print(f"Max weight     : {all_weights.max():.1f}")
    print(f"Weight std     : {all_weights.std():.2f}")

    if all_weights.max() > 50:
        print("⚠️  Extreme weights detected — high variance risk")

    return {
        "ate_ipw":   round(float(ate), 4),
        "n_trimmed": n_trimmed,
        "max_weight": round(float(all_weights.max()), 2),
    }


if __name__ == "__main__":
    import json

    df = pd.read_csv("data/processed/criteo_with_propensity.csv")
    print(f"Loaded: {df.shape}")

    feature_cols = [f"f{i}" for i in range(12)]

    # ── PSM ──
    matched_t, matched_c = propensity_score_matching(
        df, treatment_col="treatment_observational",
        propensity_col="propensity_obs", caliper=0.05
    )
    df_balance = check_covariate_balance(matched_t, matched_c, feature_cols)
    plot_balance(df_balance, save_path="results/figures/psm_balance.png")

    ate_psm = psm_ate(matched_t, matched_c, outcome_col="conversion")
    print(f"\nATE PSM: {ate_psm:.4f}")

    # ── IPW ──
    ipw_result = ipw_estimator(
        df, treatment_col="treatment_observational",
        outcome_col="conversion", propensity_col="propensity_obs"
    )

    # Update estimates
    with open("results/ate_estimates.json") as f:
        data = json.load(f)

    data["estimates"]["PSM"]  = round(ate_psm, 4)
    data["estimates"]["IPW"]  = ipw_result["ate_ipw"]

    with open("results/ate_estimates.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n── All Estimates So Far ──")
    for method, val in data["estimates"].items():
        print(f"  {method:<30}: {val:.4f}")

    print("\n✅ Saved: results/ate_estimates.json")