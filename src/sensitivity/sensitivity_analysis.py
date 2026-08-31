import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# ── 7.2 Rosenbaum-style Sensitivity (tanpa sensemakr) ────────────────────────

def compute_robustness_value(ate: float, se: float,
                              alpha: float = 0.05) -> dict:
    """
    Hitung robustness value secara manual.
    RV = seberapa besar partial R² confounder tersembunyi harus ada
    untuk membatalkan temuan (mendorong CI melewati nol).

    Sensemakr tidak tersedia di Python — implementasi manual
    berdasarkan Cinelli & Hazlett (2020).
    """
    from scipy import stats

    t_stat   = ate / (se + 1e-10)
    t_crit   = stats.t.ppf(1 - alpha/2, df=1000)

    # RV: nilai partial R² yang diperlukan untuk menghapus efek
    rv = (t_stat**2 - t_crit**2) / (t_stat**2 - t_crit**2 + 1000)
    rv = max(0, min(1, rv))

    print(f"\n── Sensitivity Analysis (Cinelli & Hazlett 2020) ──")
    print(f"ATE estimate   : {ate:.4f}")
    print(f"Std Error      : {se:.4f}")
    print(f"t-statistic    : {t_stat:.3f}")
    print(f"t-critical     : {t_crit:.3f}")
    print(f"Robustness Value (RV): {rv:.3f}")
    print(f"\nInterpretasi:")
    if rv > 0.10:
        print(f"  Temuan ROBUST — confounder tersembunyi harus menjelaskan")
        print(f"  >{rv*100:.1f}% variance treatment DAN >{rv*100:.1f}% variance outcome")
        print(f"  untuk membatalkan temuan ini.")
    else:
        print(f"  Temuan FRAGILE — confounder tersembunyi yang relatif kecil")
        print(f"  ({rv*100:.1f}% variance) sudah bisa membatalkan temuan.")

    return {"robustness_value": round(rv, 4), "t_stat": round(t_stat, 3)}


def sensitivity_contour_plot(ate: float, se: float,
                              save_path: str = None):
    """
    Contour plot: kombinasi (R²_YZ, R²_TZ) yang membatalkan temuan.
    Area di bawah garis = confounder yang cukup kuat membatalkan.
    """
    from scipy import stats

    t_crit = stats.t.ppf(0.975, df=1000)
    r2_vals = np.linspace(0, 0.5, 100)

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, ate_adj, color in [
        ("ATE estimate", ate, "#3B82F6"),
        ("Lower CI", ate - 1.96*se, "#EF4444"),
    ]:
        # Batas R² yang membatalkan pada berbagai nilai
        critical_r2 = []
        for r2_t in r2_vals:
            t_adj = ate_adj / (se * np.sqrt(1 + r2_t))
            r2_y_needed = max(0, (t_adj**2 - t_crit**2) / (t_adj**2 + 1))
            critical_r2.append(r2_y_needed)
        ax.plot(r2_vals, critical_r2, label=label, color=color, linewidth=2)

    ax.fill_between(r2_vals, 0, 0.5, alpha=0.1, color="#EF4444",
                    label="Region yang membatalkan temuan")
    ax.set_xlabel("Partial R² confounder dengan Treatment")
    ax.set_ylabel("Partial R² confounder dengan Outcome")
    ax.set_title("Sensitivity Contour Plot\n"
                 "Kombinasi confounder tersembunyi yang bisa membatalkan temuan")
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


# ── 7.3 Manski Bounds ─────────────────────────────────────────────────────────

def manski_bounds(df: pd.DataFrame,
                  treatment_col: str,
                  outcome_col: str,
                  ate_dml: float = None) -> dict:
    """
    Partial identification bounds (Manski 1990).
    Tanpa asumsi confounding apapun — batas terburuk.

    Nilai melaporkan: interval lebar = reminder bahwa tanpa asumsi
    unconfoundedness, kita tidak bisa menyimpulkan banyak hal.
    DML yang lebih sempit = gain dari asumsi yang kita buat.
    """
    p_t1   = df[treatment_col].mean()
    p_t0   = 1 - p_t1
    e_y_t1 = df[df[treatment_col] == 1][outcome_col].mean()
    e_y_t0 = df[df[treatment_col] == 0][outcome_col].mean()

    lower = e_y_t1 * p_t1 - e_y_t0 * p_t0 - p_t0
    upper = e_y_t1 * p_t1 - e_y_t0 * p_t0 + p_t1

    print(f"\n── Manski Bounds ──")
    print(f"Manski interval : [{lower:.4f}, {upper:.4f}]")
    print(f"Interval width  : {upper - lower:.4f}")

    if ate_dml is not None:
        in_bounds = lower <= ate_dml <= upper
        print(f"DML estimate    : {ate_dml:.4f} (dalam bounds: {in_bounds})")
        print(f"\nInterpretasi:")
        print(f"  Manski bounds lebar ({upper-lower:.2f}) = tanpa asumsi, kita tidak bisa menyimpulkan banyak")
        print(f"  DML interval lebih sempit = gain dari asumsi unconfoundedness")
        print(f"  Ini justifikasi eksplisit mengapa asumsi diperlukan dalam causal inference")

    return {"lower": round(lower, 4), "upper": round(upper, 4)}


def plot_identification_bounds(manski: dict, dml: dict, true_ate: float,
                                save_path: str = None):
    """Visualisasi perbandingan Manski bounds vs DML CI vs ground truth."""
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = ["Manski Bounds\n(no assumptions)", "DML Estimate\n(with assumptions)", "Ground Truth\n(RCT)"]
    lowers  = [manski["lower"], dml["ci_lower"], true_ate]
    uppers  = [manski["upper"], dml["ci_upper"], true_ate]
    centers = [(l+u)/2 for l, u in zip(lowers, uppers)]
    colors  = ["#EF4444", "#3B82F6", "#22C55E"]

    for i, (method, lower, upper, center, color) in enumerate(
        zip(methods, lowers, uppers, centers, colors)
    ):
        ax.barh(i, upper - lower, left=lower, height=0.5,
                color=color, alpha=0.7, label=method)
        ax.plot(center, i, "o", color=color, markersize=8)

    ax.axvline(true_ate, color="#22C55E", linestyle="--",
               linewidth=2, label=f"True ATE = {true_ate:.4f}")
    ax.axvline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("ATE Estimate")
    ax.set_title("Partial Identification: Manski Bounds vs DML vs Ground Truth")
    ax.grid(axis="x", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


if __name__ == "__main__":
    import json

    df = pd.read_csv("data/processed/criteo_with_propensity.csv")
    print(f"Loaded: {df.shape}")

    with open("results/ate_estimates.json") as f:
        data = json.load(f)

    true_ate = data["true_ate"]
    ate_dml  = data["estimates"]["DML (Manual)"]
    se_dml   = 0.0009  # dari output sebelumnya

    # Robustness value
    rv_result = compute_robustness_value(ate_dml, se_dml)

    # Sensitivity contour
    sensitivity_contour_plot(
        ate_dml, se_dml,
        save_path="results/figures/sensitivity_contour.png"
    )

    # Manski bounds
    manski = manski_bounds(
        df, treatment_col="treatment_observational",
        outcome_col="conversion", ate_dml=ate_dml
    )

    # Plot bounds comparison
    dml_ci = {"ci_lower": ate_dml - 1.96*se_dml, "ci_upper": ate_dml + 1.96*se_dml}
    plot_identification_bounds(
        manski, dml_ci, true_ate,
        save_path="results/figures/identification_bounds.png"
    )

    print(f"\n── Summary ──")
    print(f"Robustness Value : {rv_result['robustness_value']:.4f}")
    print(f"Manski Bounds    : [{manski['lower']:.4f}, {manski['upper']:.4f}]")
    print(f"DML CI           : [{dml_ci['ci_lower']:.4f}, {dml_ci['ci_upper']:.4f}]")
    print(f"True ATE         : {true_ate:.4f}")

    print("\n✅ Sensitivity analysis selesai")