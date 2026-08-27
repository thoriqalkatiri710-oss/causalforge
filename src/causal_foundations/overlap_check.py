import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# ── 2.2.1 Overlap Check ───────────────────────────────────────────────────────

def check_overlap(df: pd.DataFrame,
                  treatment_col: str,
                  covariates: list,
                  save_path: str = None) -> tuple:
    """
    Verifikasi asumsi positivity secara empiris.
    Jika extreme_pct > 10-15%: pelanggaran positivity serius
    → pertimbangkan trimming sebelum estimasi.
    """
    X = df[covariates]
    T = df[treatment_col]

    propensity_model  = LogisticRegression(max_iter=1000).fit(X, T)
    propensity_scores = propensity_model.predict_proba(X)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel kiri: distribusi propensity score
    axes[0].hist(propensity_scores[T==1], bins=50, alpha=0.6,
                 label="Treated", density=True, color="#3B82F6")
    axes[0].hist(propensity_scores[T==0], bins=50, alpha=0.6,
                 label="Control", density=True, color="#EF4444")
    axes[0].axvline(0.05, color="orange", linestyle="--", alpha=0.7, label="Extreme threshold")
    axes[0].axvline(0.95, color="orange", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Estimated Propensity Score")
    axes[0].set_title("Overlap Check — Common Support")
    axes[0].legend()

    # Panel kanan: propensity score vs outcome
    axes[1].scatter(propensity_scores[T==1],
                    df.loc[T==1, "visit"].values,
                    alpha=0.02, s=5, color="#3B82F6", label="Treated")
    axes[1].scatter(propensity_scores[T==0],
                    df.loc[T==0, "visit"].values,
                    alpha=0.02, s=5, color="#EF4444", label="Control")
    axes[1].set_xlabel("Propensity Score")
    axes[1].set_ylabel("Outcome (visit)")
    axes[1].set_title("Propensity Score vs Outcome")
    axes[1].legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()

    # Statistik overlap
    extreme_pct  = ((propensity_scores < 0.05) | (propensity_scores > 0.95)).mean() * 100
    overlap_mean = propensity_scores.mean()
    overlap_std  = propensity_scores.std()

    print(f"\n── Overlap Check Results ──")
    print(f"Propensity score mean  : {overlap_mean:.3f}")
    print(f"Propensity score std   : {overlap_std:.3f}")
    print(f"Propensity range       : [{propensity_scores.min():.3f}, {propensity_scores.max():.3f}]")
    print(f"Extreme units (<5% or >95%): {extreme_pct:.1f}%")

    if extreme_pct > 15:
        print("⚠️  Pelanggaran positivity serius — pertimbangkan trimming")
    elif extreme_pct > 5:
        print("⚠️  Overlap moderat — monitor dengan hati-hati")
    else:
        print("✅ Overlap baik — asumsi positivity terpenuhi")

    return propensity_scores, fig


def trim_extreme_propensity(df: pd.DataFrame,
                             propensity_scores: np.ndarray,
                             lower: float = 0.05,
                             upper: float = 0.95) -> pd.DataFrame:
    """
    Trim sampel di region propensity ekstrem.
    Digunakan jika extreme_pct > 15%.
    """
    mask   = (propensity_scores >= lower) & (propensity_scores <= upper)
    df_trimmed = df[mask].reset_index(drop=True)
    print(f"Trimmed: {len(df)} → {len(df_trimmed)} rows ({mask.mean():.1%} retained)")
    return df_trimmed


if __name__ == "__main__":
    df = pd.read_csv("data/processed/criteo_with_confounding.csv")
    print(f"Loaded: {df.shape}")

    feature_cols = [f"f{i}" for i in range(12)]

    print("\n── RCT Treatment ──")
    ps_rct, _ = check_overlap(
        df, treatment_col="treatment_rct_original",
        covariates=feature_cols,
        save_path="results/figures/overlap_rct.png"
    )

    print("\n── Observational Treatment (with confounding) ──")
    ps_obs, _ = check_overlap(
        df, treatment_col="treatment_observational",
        covariates=feature_cols,
        save_path="results/figures/overlap_observational.png"
    )

    df["propensity_rct"] = ps_rct
    df["propensity_obs"] = ps_obs
    df.to_csv("data/processed/criteo_with_propensity.csv", index=False)
    print("\n✅ Saved: criteo_with_propensity.csv")