import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LinearRegression


# ── 5.1.1 Manual DML (Frisch-Waugh-Lovell) ───────────────────────────────────

def dml_ate_manual(df: pd.DataFrame,
                   treatment_col: str,
                   outcome_col: str,
                   covariates: list,
                   n_folds: int = 5,
                   seed: int = 42) -> dict:
    """
    Double Machine Learning (Chernozhukov et al. 2018) — implementasi manual.

    Algoritma:
    1. Residualize Y: tilde_Y = Y - E[Y|X] (pakai ML)
    2. Residualize T: tilde_T = T - E[T|X] (pakai ML)
    3. Regress tilde_Y ~ tilde_T → koefisien = ATE

    Kunci: cross-fitting (K-fold) mencegah overfitting bias
    yang terjadi jika fit dan predict pakai data yang sama.
    """
    X = df[covariates].values
    Y = df[outcome_col].values
    T = df[treatment_col].values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    Y_res = np.zeros(len(df))
    T_res = np.zeros(len(df))

    print(f"Running DML with {n_folds}-fold cross-fitting...")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Fit outcome model E[Y|X]
        m_model = GradientBoostingRegressor(n_estimators=100, random_state=seed)
        m_model.fit(X[train_idx], Y[train_idx])
        Y_res[test_idx] = Y[test_idx] - m_model.predict(X[test_idx])

        # Fit treatment model E[T|X]
        g_model = GradientBoostingClassifier(n_estimators=100, random_state=seed)
        g_model.fit(X[train_idx], T[train_idx])
        T_res[test_idx] = T[test_idx] - g_model.predict_proba(X[test_idx])[:, 1]

        print(f"  Fold {fold+1}/{n_folds} done")

    # Final regression: tilde_Y ~ tilde_T
    from sklearn.linear_model import LinearRegression
    final_model = LinearRegression()
    final_model.fit(T_res.reshape(-1, 1), Y_res)
    ate_dml = final_model.coef_[0]

    # Standard error via sandwich estimator
    residuals = Y_res - T_res * ate_dml
    se = np.sqrt(
        np.mean(residuals**2 * T_res**2) /
        (np.mean(T_res**2)**2 * len(df))
    )
    ci_lower = ate_dml - 1.96 * se
    ci_upper = ate_dml + 1.96 * se

    print(f"\n── DML Results ──")
    print(f"ATE DML        : {ate_dml:.4f}")
    print(f"Std Error      : {se:.4f}")
    print(f"95% CI         : [{ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        "ate_dml":  round(float(ate_dml), 4),
        "se":       round(float(se), 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "T_residuals": T_res,
        "Y_residuals": Y_res,
    }


# ── 5.1.2 EconML DML ─────────────────────────────────────────────────────────

def dml_ate_econml(df: pd.DataFrame,
                   treatment_col: str,
                   outcome_col: str,
                   covariates: list,
                   seed: int = 42) -> dict:
    """
    DML via EconML — pakai LinearDML dengan discrete_treatment=True
    untuk treatment binary.
    """
    from econml.dml import LinearDML

    X = df[covariates].values
    Y = df[outcome_col].values
    T = df[treatment_col].values

    model = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=100, random_state=seed),
        model_t=GradientBoostingRegressor(n_estimators=100, random_state=seed),
        discrete_treatment=False,
        cv=5,
        random_state=seed
    )
    model.fit(Y, T, X=X)

    ate = model.ate(X)

    print(f"\n── EconML LinearDML Results ──")
    print(f"ATE EconML     : {ate:.4f}")

    return {
        "ate_econml": round(float(ate), 4),
        "ci_lower":   None,
        "ci_upper":   None,
        "model":      model,
    }

# ── Residual Plot ─────────────────────────────────────────────────────────────

def plot_dml_residuals(T_res: np.ndarray, Y_res: np.ndarray,
                        ate: float, save_path: str = None):
    """
    Scatter plot tilde_T vs tilde_Y — slope = ATE.
    Visualisasi intuitif mekanisme DML.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(T_res, Y_res, alpha=0.02, s=5, color="#3B82F6")

    x_line = np.linspace(T_res.min(), T_res.max(), 100)
    ax.plot(x_line, ate * x_line, color="red", linewidth=2,
            label=f"ATE = {ate:.4f} (slope)")

    ax.set_xlabel("Residual Treatment (tilde_T = T - E[T|X])")
    ax.set_ylabel("Residual Outcome (tilde_Y = Y - E[Y|X])")
    ax.set_title("DML: Partial Linear Regression\n"
                 "Slope = Causal Effect of Treatment on Outcome")
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


if __name__ == "__main__":
    import json

    df = pd.read_csv("data/processed/criteo_with_propensity.csv")
    print(f"Loaded: {df.shape}")

    feature_cols = [f"f{i}" for i in range(12)]

    # ── Manual DML ──
    dml_result = dml_ate_manual(
        df, treatment_col="treatment_observational",
        outcome_col="conversion", covariates=feature_cols
    )

    plot_dml_residuals(
        dml_result["T_residuals"], dml_result["Y_residuals"],
        dml_result["ate_dml"],
        save_path="results/figures/dml_residuals.png"
    )

    # ── EconML DML ──
    econml_result = dml_ate_econml(
        df, treatment_col="treatment_observational",
        outcome_col="conversion", covariates=feature_cols
    )

    # Update estimates
    with open("results/ate_estimates.json") as f:
        data = json.load(f)

    data["estimates"]["DML (Manual)"]  = dml_result["ate_dml"]
    data["estimates"]["DML (EconML)"]  = econml_result["ate_econml"]

    with open("results/ate_estimates.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n── All Estimates So Far ──")
    for method, val in data["estimates"].items():
        bias = abs(val - data["true_ate"]) / abs(data["true_ate"]) * 100
        print(f"  {method:<30}: {val:.4f} (bias {bias:.1f}%)")

    print("\n✅ Done")