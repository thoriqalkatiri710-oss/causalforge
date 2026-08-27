import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 3.1.1 Naive ATE ───────────────────────────────────────────────────────────

def naive_ate(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> float:
    """
    Naive difference-in-means estimator.
    BUKAN estimator kausal yang valid pada data observational —
    digunakan sebagai baseline untuk menunjukkan magnitude bias confounding.
    """
    return (df[df[treatment_col] == 1][outcome_col].mean() -
            df[df[treatment_col] == 0][outcome_col].mean())


def compute_bias_analysis(df: pd.DataFrame, outcome_col: str = "conversion") -> dict:
    """
    Hitung dan bandingkan ATE naive vs ground truth RCT.
    Hasil ini menjadi pembuka narasi laporan:
    'Naive estimator meleset X% — berikut bagaimana DML memulihkan estimasi benar.'
    """
    ate_naive_obs = naive_ate(df, "treatment_observational", outcome_col)
    ate_true_rct  = naive_ate(df, "treatment_rct_original",  outcome_col)
    ate_naive_rct = naive_ate(df, "treatment_rct_original",  outcome_col)

    bias_abs = ate_naive_obs - ate_true_rct
    bias_pct = abs(bias_abs) / abs(ate_true_rct) * 100 if ate_true_rct != 0 else 0

    print(f"\n── Naive Estimation Bias Analysis ──")
    print(f"ATE Ground Truth (RCT)          : {ate_true_rct:.4f}")
    print(f"ATE Naive (RCT data)            : {ate_naive_rct:.4f}")
    print(f"ATE Naive (observational/biased): {ate_naive_obs:.4f}")
    print(f"Bias absolut                    : {bias_abs:+.4f}")
    print(f"Bias relatif                    : {bias_pct:.1f}%")

    if bias_pct > 50:
        print("⚠️  Bias SANGAT BESAR — naive estimator tidak dapat dipercaya")
    elif bias_pct > 20:
        print("⚠️  Bias signifikan — causal inference diperlukan")
    else:
        print("⚠️  Bias moderat — tetap perlu koreksi")

    return {
        "ATE Ground Truth (RCT)":           round(ate_true_rct, 4),
        "ATE Naive (observational)":         round(ate_naive_obs, 4),
        "Bias Absolut":                      round(bias_abs, 4),
        "Bias Relatif (%)":                  round(bias_pct, 1),
    }


# ── 3.1.2 Bias Comparison Plot ────────────────────────────────────────────────

def plot_bias_comparison(estimates: dict, true_ate: float,
                          save_path: str = None):
    """
    Bar chart perbandingan ATE antar metode vs ground truth.
    Akan diperkaya progresif seiring Bagian 4-6 menghasilkan estimasi baru.
    Merah = bias besar (>10%), Hijau = bias kecil.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(estimates.keys())
    values  = list(estimates.values())
    colors  = [
        "#EF4444" if abs(v - true_ate) > 0.10 * abs(true_ate) else "#22C55E"
        for v in values
    ]

    bars = ax.barh(methods, values, color=colors, edgecolor="white", height=0.6)
    ax.axvline(true_ate, color="black", linestyle="--", linewidth=2,
               label=f"Ground Truth ATE = {true_ate:.4f}")

    # Label nilai di setiap bar
    for bar, val in zip(bars, values):
        ax.text(val + 0.0001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Estimated ATE")
    ax.set_title("Perbandingan Bias Antar Metode Estimasi\n"
                 "(Merah = bias >10%, Hijau = bias ≤10%)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


if __name__ == "__main__":
    df = pd.read_csv("data/processed/criteo_with_propensity.csv")
    print(f"Loaded: {df.shape}")

    # Hitung bias
    bias_results = compute_bias_analysis(df, outcome_col="conversion")

    # Plot awal — hanya naive estimator (akan diperkaya nanti)
    true_ate = bias_results["ATE Ground Truth (RCT)"]
    estimates = {
        "Naive (Observational)": bias_results["ATE Naive (observational)"],
        "Ground Truth (RCT)":    true_ate,
    }

    plot_bias_comparison(
        estimates, true_ate,
        save_path="results/figures/bias_comparison.png"
    )

    # Simpan hasil untuk dipakai di bagian selanjutnya
    import json
    with open("results/ate_estimates.json", "w") as f:
        json.dump({"true_ate": true_ate, "estimates": estimates}, f, indent=2)
    print("✅ Saved: results/ate_estimates.json")