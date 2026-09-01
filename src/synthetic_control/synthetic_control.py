import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ── 9.1.1 Synthetic Control Method ───────────────────────────────────────────

def generate_panel_data(n_units: int = 20,
                         n_periods: int = 40,
                         treatment_unit: int = 0,
                         treatment_period: int = 20,
                         true_effect: float = 2.0,
                         seed: int = 42) -> pd.DataFrame:
    """
    Generate panel data untuk synthetic control.
    Unit 0 = treated (mendapat intervensi di period 20).
    Unit 1-19 = donor pool (tidak pernah mendapat treatment).
    """
    rng = np.random.default_rng(seed)

    rows = []
    unit_fe  = rng.normal(0, 2, n_units)
    period_fe = rng.normal(0, 1, n_periods)
    common_trend = np.cumsum(rng.normal(0.1, 0.3, n_periods))

    for unit in range(n_units):
        for period in range(n_periods):
            treated = int(unit == treatment_unit and period >= treatment_period)
            outcome = (
                unit_fe[unit] +
                period_fe[period] +
                common_trend[period] +
                true_effect * treated +
                rng.normal(0, 0.5)
            )
            rows.append({
                "unit":    unit,
                "period":  period,
                "outcome": outcome,
                "treated": treated,
                "post":    int(period >= treatment_period),
            })

    return pd.DataFrame(rows)


def fit_synthetic_control(df: pd.DataFrame,
                           treatment_unit: int = 0,
                           treatment_period: int = 20) -> dict:
    """
    Synthetic Control Method (Abadie & Gardeazabal 2003).

    Cari bobot W untuk donor units sehingga synthetic control
    (weighted average donor) cocok dengan treated unit di pre-period.

    Kelebihan vs DID:
    - Tidak butuh parallel trends assumption
    - Data-driven pemilihan pembanding
    - Visualisasi counterfactual yang intuitif
    """
    # Pivot ke wide format
    pivot = df.pivot(index="period", columns="unit", values="outcome")

    pre_mask  = df["period"].unique() < treatment_period
    pre_periods = df["period"].unique()[pre_mask]

    Y_treated = pivot[treatment_unit].values
    Y_donors  = pivot.drop(columns=[treatment_unit]).values

    pre_treated = Y_treated[:treatment_period]
    pre_donors  = Y_donors[:treatment_period, :]

    n_donors = Y_donors.shape[1]

    # Optimasi bobot: minimize ||Y_treated_pre - W·Y_donors_pre||²
    def objective(W):
        synthetic_pre = pre_donors @ W
        return np.sum((pre_treated - synthetic_pre) ** 2)

    constraints = [{"type": "eq", "fun": lambda W: np.sum(W) - 1}]
    bounds      = [(0, 1)] * n_donors
    W0          = np.ones(n_donors) / n_donors

    result = minimize(objective, W0, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    W_opt = result.x

    # Synthetic control series
    synthetic = Y_donors @ W_opt

    # Treatment effect estimate (post-period)
    effect_post = Y_treated[treatment_period:] - synthetic[treatment_period:]
    ate_sc      = float(np.mean(effect_post))

    print(f"\n── Synthetic Control Results ──")
    print(f"Optimal weights (top 5):")
    donor_ids = [u for u in pivot.columns if u != treatment_unit]
    weight_df = pd.DataFrame({"unit": donor_ids, "weight": W_opt})
    print(weight_df.nlargest(5, "weight").to_string(index=False))
    print(f"\nATE (post-period avg): {ate_sc:.4f}")
    print(f"True effect          : 2.0000")
    print(f"Bias                 : {abs(ate_sc - 2.0):.4f}")

    return {
        "weights":    W_opt,
        "synthetic":  synthetic,
        "Y_treated":  Y_treated,
        "ate_sc":     round(ate_sc, 4),
        "pivot":      pivot,
        "donor_ids":  donor_ids,
        "treatment_period": treatment_period,
    }


def plot_synthetic_control(sc_result: dict, save_path: str = None):
    """
    Visualisasi synthetic control vs treated unit.
    Gap di post-period = estimated treatment effect.
    """
    Y_treated        = sc_result["Y_treated"]
    synthetic        = sc_result["synthetic"]
    treatment_period = sc_result["treatment_period"]
    ate_sc           = sc_result["ate_sc"]
    n_periods        = len(Y_treated)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel kiri: treated vs synthetic
    periods = np.arange(n_periods)
    axes[0].plot(periods, Y_treated, label="Treated Unit",
                 color="#3B82F6", linewidth=2)
    axes[0].plot(periods, synthetic, label="Synthetic Control",
                 color="#EF4444", linewidth=2, linestyle="--")
    axes[0].axvline(treatment_period, color="black", linestyle=":",
                    linewidth=1.5, label="Treatment Start")
    axes[0].fill_between(periods[treatment_period:],
                          synthetic[treatment_period:],
                          Y_treated[treatment_period:],
                          alpha=0.2, color="#22C55E",
                          label=f"Estimated Effect (ATE={ate_sc:.2f})")
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("Outcome")
    axes[0].set_title("Synthetic Control vs Treated Unit")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel kanan: gap plot (treatment effect over time)
    gap = Y_treated - synthetic
    axes[1].plot(periods, gap, color="#8B5CF6", linewidth=2)
    axes[1].axvline(treatment_period, color="black", linestyle=":",
                    linewidth=1.5, label="Treatment Start")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].axhline(ate_sc, color="#22C55E", linestyle="--",
                    linewidth=1.5, label=f"Avg Post Effect = {ate_sc:.2f}")
    axes[1].fill_between(periods[treatment_period:], 0, gap[treatment_period:],
                          alpha=0.2, color="#8B5CF6")
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("Gap (Treated - Synthetic)")
    axes[1].set_title("Treatment Effect Over Time (Gap Plot)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Synthetic Control Method — Causal Effect Estimation", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


# ── 9.1.2 Placebo Test ────────────────────────────────────────────────────────

def placebo_test(df: pd.DataFrame, sc_result: dict,
                  treatment_unit: int = 0,
                  treatment_period: int = 20,
                  save_path: str = None):
    """
    In-space placebo test: terapkan synthetic control ke setiap donor unit
    seolah-olah mereka yang mendapat treatment.
    Jika efek treated unit jauh lebih besar dari placebo → signifikan.
    """
    pivot    = sc_result["pivot"]
    all_units = pivot.columns.tolist()
    donor_units = [u for u in all_units if u != treatment_unit]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot placebo gaps
    placebo_gaps = []
    for placebo_unit in donor_units[:10]:  # ambil 10 donor saja
        try:
            placebo_donors = [u for u in all_units if u != placebo_unit]
            Y_p    = pivot[placebo_unit].values
            Y_d    = pivot[placebo_donors].values

            pre_p = Y_p[:treatment_period]
            pre_d = Y_d[:treatment_period, :]

            n_d = Y_d.shape[1]

            def obj(W): return np.sum((pre_p - pre_d @ W)**2)
            cons   = [{"type": "eq", "fun": lambda W: np.sum(W) - 1}]
            bounds = [(0, 1)] * n_d
            res    = minimize(obj, np.ones(n_d)/n_d, method="SLSQP",
                              bounds=bounds, constraints=cons)

            synthetic_p = Y_d @ res.x
            gap_p       = Y_p - synthetic_p
            placebo_gaps.append(gap_p)

            ax.plot(gap_p, color="gray", alpha=0.3, linewidth=0.8)
        except Exception:
            continue

    # Plot treated gap
    treated_gap = sc_result["Y_treated"] - sc_result["synthetic"]
    ax.plot(treated_gap, color="#EF4444", linewidth=2.5,
            label=f"Treated unit (ATE={sc_result['ate_sc']:.2f})")

    ax.axvline(treatment_period, color="black", linestyle=":",
               linewidth=1.5, label="Treatment Start")
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Period")
    ax.set_ylabel("Gap (Unit - Synthetic)")
    ax.set_title("Placebo Test (In-Space)\n"
                 "Jika treated unit (merah) jauh dari placebo → efek signifikan")
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


if __name__ == "__main__":
    print("── Synthetic Control Method ──")

    # Generate panel data
    df_panel = generate_panel_data(
        n_units=20, n_periods=40,
        treatment_unit=0, treatment_period=20,
        true_effect=2.0
    )

    print(f"Panel data: {df_panel.shape}")
    print(f"Units: {df_panel['unit'].nunique()}, Periods: {df_panel['period'].nunique()}")

    # Fit synthetic control
    sc_result = fit_synthetic_control(df_panel, treatment_unit=0, treatment_period=20)

    # Plot
    plot_synthetic_control(
        sc_result,
        save_path="results/figures/synthetic_control.png"
    )

    # Placebo test
    print("\nRunning placebo test...")
    placebo_test(
        df_panel, sc_result,
        treatment_unit=0, treatment_period=20,
        save_path="results/figures/synthetic_control_placebo.png"
    )

    print(f"\n── Summary ──")
    print(f"True effect  : 2.0000")
    print(f"SC estimate  : {sc_result['ate_sc']:.4f}")
    print(f"Bias         : {abs(sc_result['ate_sc'] - 2.0):.4f}")
    print(f"Bias %       : {abs(sc_result['ate_sc'] - 2.0)/2.0*100:.1f}%")