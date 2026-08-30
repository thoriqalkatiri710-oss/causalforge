import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from econml.grf import CausalForest
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


# ── 6.1.1 Causal Forest ───────────────────────────────────────────────────────

def fit_causal_forest(df: pd.DataFrame,
                      treatment_col: str,
                      outcome_col: str,
                      covariates: list,
                      n_estimators: int = 200,
                      seed: int = 42) -> dict:
    """
    Causal Forest (Wager & Athey 2018) untuk estimasi CATE.
    Berbeda dari DML: menghasilkan DISTRIBUSI efek per individu,
    bukan hanya rata-rata ATE.

    Kelebihan vs DML:
    - Mendeteksi heterogenitas efek treatment
    - Tidak perlu spesifikasi bentuk fungsional
    - Honest estimation (split sample untuk inference)
    """
    X = df[covariates].values
    Y = df[outcome_col].values
    T = df[treatment_col].values

    print(f"Fitting Causal Forest ({n_estimators} trees)...")

    cf = CausalForest(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1
    )
    cf.fit(X, T, Y)

    # CATE per individu
    cate = cf.predict(X).flatten()

    # ATE = rata-rata CATE
    ate_cf = float(np.mean(cate))

    # Inference
    cate_lb, cate_ub = cf.predict_interval(X, alpha=0.05)

    print(f"\n── Causal Forest Results ──")
    print(f"ATE (mean CATE) : {ate_cf:.4f}")
    print(f"CATE std        : {np.std(cate):.4f}")
    print(f"CATE min        : {np.min(cate):.4f}")
    print(f"CATE max        : {np.max(cate):.4f}")
    print(f"% positive CATE : {(cate > 0).mean():.1%}")

    return {
        "model":    cf,
        "cate":     cate,
        "ate_cf":   round(ate_cf, 4),
        "cate_lb":  cate_lb.flatten(),
        "cate_ub":  cate_ub.flatten(),
        "X":        X,
        "covariates": covariates,
    }


# ── 6.1.2 CATE Distribution Plot ─────────────────────────────────────────────

def plot_cate_distribution(cate: np.ndarray, ate_cf: float,
                            true_ate: float = None,
                            save_path: str = None):
    """
    Distribusi CATE — visualisasi heterogenitas efek treatment.
    Ini yang membedakan Causal Forest dari DML:
    kita bisa lihat siapa yang paling diuntungkan treatment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel kiri: histogram CATE
    axes[0].hist(cate, bins=50, color="#3B82F6", alpha=0.7, edgecolor="white")
    axes[0].axvline(ate_cf, color="red", linestyle="--",
                    linewidth=2, label=f"ATE = {ate_cf:.4f}")
    if true_ate:
        axes[0].axvline(true_ate, color="green", linestyle="-",
                        linewidth=2, label=f"Ground Truth = {true_ate:.4f}")
    axes[0].axvline(0, color="black", linestyle=":", alpha=0.5, label="Zero effect")
    axes[0].set_xlabel("Estimated CATE (Individual Treatment Effect)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribusi CATE — Heterogenitas Efek Treatment")
    axes[0].legend()

    # Panel kanan: CATE sorted (uplift curve)
    sorted_cate = np.sort(cate)[::-1]
    cumulative  = np.cumsum(sorted_cate) / np.arange(1, len(sorted_cate)+1)
    axes[1].plot(np.linspace(0, 100, len(cumulative)), cumulative,
                 color="#3B82F6", linewidth=2)
    axes[1].axhline(ate_cf, color="red", linestyle="--", label=f"ATE = {ate_cf:.4f}")
    axes[1].axhline(0, color="black", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("% Population (sorted by CATE, high to low)")
    axes[1].set_ylabel("Cumulative Average CATE")
    axes[1].set_title("Uplift Curve — Prioritasi Target Treatment")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


# ── 6.1.3 Feature Importance ─────────────────────────────────────────────────

def plot_cate_feature_importance(cf_result: dict, save_path: str = None):
    """
    Feature importance dari Causal Forest — berbeda dari standard ML:
    ini mengukur seberapa besar tiap fitur mempengaruhi HETEROGENITAS efek,
    bukan seberapa besar pengaruhnya terhadap outcome.
    """
    model      = cf_result["model"]
    covariates = cf_result["covariates"]

    importances = model.feature_importances_
    idx         = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([covariates[i] for i in idx],
            importances[idx], color="#8B5CF6", edgecolor="white")
    ax.set_xlabel("Feature Importance (Heterogeneity Driver)")
    ax.set_title("Causal Forest Feature Importance\n"
                 "(Fitur yang paling menentukan siapa yang diuntungkan treatment)")
    ax.grid(axis="x", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


# ── 6.2 Policy Learning ───────────────────────────────────────────────────────

def policy_recommendation(df: pd.DataFrame, cate: np.ndarray,
                           covariates: list, top_pct: float = 0.20) -> pd.DataFrame:
    """
    Berdasarkan CATE, identifikasi top X% unit yang paling diuntungkan treatment.
    Ini adalah output actionable untuk tim bisnis:
    'Targetkan treatment ke segmen ini untuk ROI tertinggi.'
    """
    df_policy = df[covariates].copy()
    df_policy["estimated_cate"] = cate
    df_policy["treat_recommended"] = cate >= np.quantile(cate, 1 - top_pct)

    print(f"\n── Policy Recommendation (Top {top_pct:.0%}) ──")
    print(f"Units recommended for treatment: {df_policy['treat_recommended'].sum():,}")
    print(f"Avg CATE recommended group     : {df_policy[df_policy['treat_recommended']]['estimated_cate'].mean():.4f}")
    print(f"Avg CATE non-recommended group : {df_policy[~df_policy['treat_recommended']]['estimated_cate'].mean():.4f}")

    return df_policy


if __name__ == "__main__":
    import json

    df = pd.read_csv("data/processed/criteo_with_propensity.csv")
    print(f"Loaded: {df.shape}")

    feature_cols = [f"f{i}" for i in range(12)]

    # Fit Causal Forest
    cf_result = fit_causal_forest(
        df, treatment_col="treatment_observational",
        outcome_col="conversion", covariates=feature_cols,
        n_estimators=200
    )

    # Load true ATE
    with open("results/ate_estimates.json") as f:
        data = json.load(f)
    true_ate = data["true_ate"]

    # Plot CATE distribution
    plot_cate_distribution(
        cf_result["cate"], cf_result["ate_cf"],
        true_ate=true_ate,
        save_path="results/figures/cate_distribution.png"
    )

    # Feature importance
    plot_cate_feature_importance(
        cf_result,
        save_path="results/figures/causal_forest_importance.png"
    )

    # Policy recommendation
    df_policy = policy_recommendation(df, cf_result["cate"], feature_cols, top_pct=0.20)

    # Update estimates
    data["estimates"]["Causal Forest"] = cf_result["ate_cf"]
    with open("results/ate_estimates.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n── All Estimates ──")
    for method, val in data["estimates"].items():
        bias = abs(val - true_ate) / abs(true_ate) * 100
        print(f"  {method:<30}: {val:.4f} (bias {bias:.1f}%)")

    # Save CATE scores
    df["cate_estimate"] = cf_result["cate"]
    df.to_csv("data/processed/criteo_with_cate.csv", index=False)
    print("\n✅ Saved: criteo_with_cate.csv")