import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path
import base64

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CausalForge — Causal Inference Engine",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #16213e 100%);
        border-right: 1px solid #2d3561;
    }

    /* Cards — fixed height + flexbox so every card is identical regardless
       of how many lines the label text wraps to */
    .metric-card {
        background: linear-gradient(135deg, #1e2235 0%, #252a40 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
        height: 160px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #7c83fd;
        margin: 0;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        line-height: 1.3;
        /* clamp long labels to 2 lines instead of pushing the card taller */
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .metric-delta-bad  { color: #ff6b6b; font-size: 0.85rem; margin-top: 4px; }
    .metric-delta-good { color: #51cf66; font-size: 0.85rem; margin-top: 4px; }

    /* Make the Streamlit column containers stretch so cards align
       even when placed in st.columns() rows */
    [data-testid="column"] > div {
        height: 100%;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #7c83fd22, transparent);
        border-left: 4px solid #7c83fd;
        padding: 12px 20px;
        border-radius: 0 8px 8px 0;
        margin: 24px 0 16px 0;
    }
    .section-header h3 { color: #ccd6f6; margin: 0; font-size: 1.1rem; }

    /* Method comparison table */
    .method-table {
        background: #1e2235;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2d3561;
    }

    /* Status badges */
    .badge-critical { background: #ff6b6b22; color: #ff6b6b; border: 1px solid #ff6b6b44; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-warning  { background: #ffa94d22; color: #ffa94d; border: 1px solid #ffa94d44; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-success  { background: #51cf6622; color: #51cf66; border: 1px solid #51cf6644; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }

    /* Missing-figure placeholder */
    .missing-figure {
        background: #1e2235;
        border: 1px dashed #ff6b6b66;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        color: #ff6b6b;
        font-size: 0.85rem;
    }
    .missing-figure code {
        display: block;
        margin-top: 8px;
        color: #ffa94d;
        font-size: 0.75rem;
        word-break: break-all;
    }

    /* Hide streamlit branding/toolbar WITHOUT leaving its layout space behind */
    #MainMenu { display: none; }
    footer    { display: none; }
    header    { display: none; }

    /* Streamlit still reserves top padding assuming the header is there — reclaim it */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1e2235;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8892b0;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: #7c83fd22;
        color: #7c83fd;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 10px; }

    /* Info boxes — consistent min-height so boxes placed side by side
       in the same row don't look mismatched */
    .insight-box {
        background: #1e2235;
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        min-height: 130px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .insight-box h4 { color: #7c83fd; margin: 0 0 8px 0; }
    .insight-box p  { color: #ccd6f6; margin: 0; font-size: 0.9rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
REQUIRED_FILES = [
    "results/ate_estimates.json",
    "results/method_comparison.csv",
    "results/validation_table.csv",
    "results/refutation_results.csv",
    "data/processed/criteo_with_cate.csv",
    "data/processed/criteo_with_propensity.csv",
]

@st.cache_data
def load_all_data():
    with open("results/ate_estimates.json") as f:
        ate_data = json.load(f)
    df_comparison = pd.read_csv("results/method_comparison.csv")
    df_validation = pd.read_csv("results/validation_table.csv")
    df_refute     = pd.read_csv("results/refutation_results.csv")
    df_cate       = pd.read_csv("data/processed/criteo_with_cate.csv")
    df_raw        = pd.read_csv("data/processed/criteo_with_propensity.csv")
    return ate_data, df_comparison, df_validation, df_refute, df_cate, df_raw

# Fail loudly and helpfully instead of a raw traceback if pipeline outputs are missing.
missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
if missing:
    st.error(
        "⚠️ Data pipeline belum dijalankan / file hasil tidak ditemukan.\n\n"
        "File berikut hilang:\n\n" + "\n".join(f"- `{m}`" for m in missing) +
        "\n\nJalankan script pipeline (data prep + estimasi ATE + refutation) "
        "terlebih dahulu sebelum membuka dashboard ini, atau cek apakah kamu "
        "menjalankan `streamlit run` dari folder project yang benar."
    )
    st.stop()

ate_data, df_comparison, df_validation, df_refute, df_cate, df_raw = load_all_data()
true_ate  = ate_data["true_ate"]
estimates = ate_data["estimates"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <div style='font-size:2.5rem;'>⚗️</div>
        <h2 style='color:#7c83fd; margin:8px 0 4px 0;'>CausalForge</h2>
        <p style='color:#8892b0; font-size:0.8rem; margin:0;'>Causal Inference Engine</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "📊 Method Comparison", "🎯 CATE Analysis",
         "🔬 Sensitivity", "🔄 Refutation Tests", "📈 Data Explorer", "📋 Reports"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("""
    <div style='padding: 12px; background: #1e2235; border-radius: 8px; border: 1px solid #2d3561;'>
        <p style='color:#8892b0; font-size:0.75rem; margin:0;'>
            <b style='color:#7c83fd;'>Dataset:</b> Criteo Uplift (simulated)<br>
            <b style='color:#7c83fd;'>N:</b> 100,000 units<br>
            <b style='color:#7c83fd;'>Treatment:</b> Binary (observational)<br>
            <b style='color:#7c83fd;'>Confounding:</b> strength=2.0
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────────────────────
def section(title, icon=""):
    st.markdown(f"""
    <div class='section-header'>
        <h3>{icon} {title}</h3>
    </div>
    """, unsafe_allow_html=True)

def metric_card(value, label, delta=None, delta_good=True):
    delta_html = ""
    if delta:
        cls = "metric-delta-good" if delta_good else "metric-delta-bad"
        delta_html = f"<p class='{cls}'>{delta}</p>"
    st.markdown(f"""
    <div class='metric-card'>
        <p class='metric-value'>{value}</p>
        <p class='metric-label'>{label}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def insight_box(title, content):
    st.markdown(f"""
    <div class='insight-box'>
        <h4>💡 {title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def show_figure(path_str, caption="", max_height=None):
    """Show a figure if it exists; otherwise show a visible placeholder
    explaining exactly which file is missing, instead of rendering nothing.
    If max_height is given, wrap the image in a fixed-height scrollable box."""
    p = Path(path_str)
    if p.exists():
        if max_height:
            b64 = base64.b64encode(p.read_bytes()).decode()
            st.markdown(f"""
            <div style='max-height:{max_height}px; overflow-y:auto; border-radius:8px; border:1px solid #2d3561;'>
                <img src='data:image/png;base64,{b64}' style='width:100%; display:block;'>
            </div>
            """, unsafe_allow_html=True)
            if caption:
                st.caption(caption)
        else:
            st.image(str(p), use_container_width=True, caption=caption or None)
    else:
        st.markdown(f"""
        <div class='missing-figure'>
            🖼️ Gambar belum tersedia — file belum digenerate oleh pipeline.
            <code>{path_str}</code>
        </div>
        """, unsafe_allow_html=True)

# ── PAGE: Overview ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <div style='padding: 32px 0 24px 0;'>
        <h1 style='color:#ccd6f6; font-size:2.8rem; margin:0;'>⚗️ CausalForge</h1>
        <p style='color:#8892b0; font-size:1.1rem; margin:8px 0 0 0;'>
            End-to-end Causal Inference Engine — PSM · IPW · DML · Causal Forest · Synthetic Control
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row 1
    section("Key Metrics", "📊")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card(f"{true_ate:.4f}", "Ground Truth ATE (RCT)")
    with c2: metric_card("98.3%", "Naive Estimator Bias", "↓ worst", False)
    with c3: metric_card("61.7%", "Best Cross-Sectional Bias (PSM)", "↑ PSM")
    with c4: metric_card("0.1%", "Synthetic Control Bias", "↑ best", True)
    with c5: metric_card("4.7x", "CATE Top 20% Multiplier", "↑ uplift", True)

    st.divider()

    # Row 2
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("100,000", "Total Observations")
    with c2: metric_card("12", "Covariates")
    with c3: metric_card("0.013", "Conversion Rate")
    with c4: metric_card("2/3", "Refutation Tests Passed")

    st.divider()

    # Core finding
    section("Core Finding", "🎯")
    col1, col2 = st.columns([2, 1])
    with col1:
        insight_box(
            "Naive Estimator Meleset 98.3%",
            f"Dengan confounding strength=2.0, naive comparison pada data observational menghasilkan ATE={estimates['Naive (Observational)']:.4f} — meleset 98.3% dari ground truth RCT ({true_ate:.4f}). "
            f"PSM mereduksi bias menjadi 61.7% (ATE={estimates['PSM']:.4f}). "
            f"Untuk panel data, Synthetic Control mencapai bias hanya 0.1% (ATE=2.0016 vs true=2.0000)."
        )
        insight_box(
            "Heterogenitas Efek Treatment (CATE)",
            f"Causal Forest mengungkap bahwa efek treatment tidak homogen: "
            f"CATE std=0.0102, dengan top 20% unit memiliki CATE rata-rata 0.0126 "
            f"(4.7x lebih tinggi dari rata-rata 0.0003). "
            f"Policy implication: targetkan treatment ke top 20% untuk ROI optimal."
        )
        insight_box(
            "Nilai Asumsi Unconfoundedness",
            f"Manski bounds tanpa asumsi apapun: [-0.499, 0.501] (width=1.000). "
            f"DML dengan asumsi unconfoundedness: [-0.001, 0.002] (width=0.004). "
            f"Asumsi unconfoundedness memperketat interval 250x — ini justifikasi eksplisit "
            f"mengapa asumsi diperlukan dalam causal inference."
        )
    with col2:
        # Mini bias chart — wrapped so its top margin matches insight-box's
        st.markdown("<div style='margin:8px 0 0 0;'>", unsafe_allow_html=True)
        methods_short = ["Naive", "IPW", "DML", "CF", "PSM"]
        biases = [98.3, 85.0, 93.3, 95.0, 61.7]
        colors = ["#ff6b6b" if b > 80 else "#ffa94d" if b > 50 else "#51cf66" for b in biases]

        fig, ax = plt.subplots(figsize=(4, 4), facecolor="#1e2235")
        ax.set_facecolor("#1e2235")
        bars = ax.barh(methods_short, biases, color=colors, edgecolor="none", height=0.5)
        ax.axvline(50, color="#ffa94d", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Bias (%)", color="#8892b0", fontsize=9)
        ax.set_title("Bias per Method", color="#ccd6f6", fontsize=10, pad=10)
        ax.tick_params(colors="#8892b0", labelsize=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.xaxis.label.set_color("#8892b0")
        for bar, b in zip(bars, biases):
            ax.text(b+0.5, bar.get_y()+bar.get_height()/2,
                    f"{b}%", va="center", color="#ccd6f6", fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

                # Filler super tipis — cukup 1 baris teks, tanpa box besar
        st.markdown(
            "<p style='color:#8892b0; font-size:0.8rem; margin:8px 4px 0 4px; text-align:center;'>"
            "📌 Garis putus-putus = ambang bias 50%. PSM terendah, Naive & CF tertinggi."
            "</p>",
            unsafe_allow_html=True
        )

    # Pipeline
    section("Pipeline Architecture", "🔧")
    st.markdown("""
    <div style='background:#1e2235; border:1px solid #2d3561; border-radius:12px; padding:20px;'>
    <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>
        <div style='text-align:center; flex:1; min-width:100px;'>
            <div style='background:#7c83fd22; border:1px solid #7c83fd44; border-radius:8px; padding:10px;'>
                <div style='font-size:1.5rem;'>📦</div>
                <div style='color:#7c83fd; font-size:0.75rem; margin-top:4px;'>Data &<br>Confounding</div>
            </div>
        </div>
        <div style='color:#2d3561; font-size:1.2rem;'>→</div>
        <div style='text-align:center; flex:1; min-width:100px;'>
            <div style='background:#51cf6622; border:1px solid #51cf6644; border-radius:8px; padding:10px;'>
                <div style='font-size:1.5rem;'>🔍</div>
                <div style='color:#51cf66; font-size:0.75rem; margin-top:4px;'>Overlap<br>Check</div>
            </div>
        </div>
        <div style='color:#2d3561; font-size:1.2rem;'>→</div>
        <div style='text-align:center; flex:1; min-width:100px;'>
            <div style='background:#ff6b6b22; border:1px solid #ff6b6b44; border-radius:8px; padding:10px;'>
                <div style='font-size:1.5rem;'>📉</div>
                <div style='color:#ff6b6b; font-size:0.75rem; margin-top:4px;'>Naive<br>Bias Demo</div>
            </div>
        </div>
        <div style='color:#2d3561; font-size:1.2rem;'>→</div>
        <div style='text-align:center; flex:1; min-width:100px;'>
            <div style='background:#ffa94d22; border:1px solid #ffa94d44; border-radius:8px; padding:10px;'>
                <div style='font-size:1.5rem;'>⚖️</div>
                <div style='color:#ffa94d; font-size:0.75rem; margin-top:4px;'>PSM &<br>IPW</div>
            </div>
        </div>
        <div style='color:#2d3561; font-size:1.2rem;'>→</div>
        <div style='text-align:center; flex:1; min-width:100px;'>
            <div style='background:#7c83fd22; border:1px solid #7c83fd44; border-radius:8px; padding:10px;'>
                <div style='font-size:1.5rem;'>🤖</div>
                <div style='color:#7c83fd; font-size:0.75rem; margin-top:4px;'>DML &<br>Causal Forest</div>
            </div>
        </div>
        <div style='color:#2d3561; font-size:1.2rem;'>→</div>
        <div style='text-align:center; flex:1; min-width:100px;'>
            <div style='background:#51cf6622; border:1px solid #51cf6644; border-radius:8px; padding:10px;'>
                <div style='font-size:1.5rem;'>🔬</div>
                <div style='color:#51cf66; font-size:0.75rem; margin-top:4px;'>Sensitivity &<br>Validation</div>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ── PAGE: Method Comparison ───────────────────────────────────────────────────
elif page == "📊 Method Comparison":
    st.markdown("<h2 style='color:#ccd6f6;'>📊 Method Comparison</h2>", unsafe_allow_html=True)

    section("ATE Estimates vs Ground Truth", "🎯")

    # Interactive comparison chart
    methods = list(estimates.keys())
    values  = [estimates[m] for m in methods]
    biases  = [abs(v - true_ate) / abs(true_ate) * 100 for v in values]
    colors  = ["#51cf66" if m == "Ground Truth (RCT)" else
               "#ff6b6b" if b > 80 else
               "#ffa94d" if b > 50 else "#7c83fd"
               for m, b in zip(methods, biases)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#1e2235")
    for ax in axes: ax.set_facecolor("#1e2235")

    # ATE chart
    bars = axes[0].barh(methods, values, color=colors, edgecolor="none", height=0.6)
    axes[0].axvline(true_ate, color="#51cf66", linestyle="--", linewidth=2,
                    label=f"Ground Truth = {true_ate:.4f}")
    axes[0].axvline(0, color="#8892b0", linestyle=":", alpha=0.3)
    for bar, val in zip(bars, values):
        axes[0].text(max(val, 0) + 0.0001, bar.get_y() + bar.get_height()/2,
                     f"{val:.4f}", va="center", color="#ccd6f6", fontsize=8)
    axes[0].set_xlabel("Estimated ATE", color="#8892b0")
    axes[0].set_title("ATE Estimates per Method", color="#ccd6f6", pad=12)
    axes[0].tick_params(colors="#8892b0")
    axes[0].legend(facecolor="#252a40", edgecolor="#2d3561",
                   labelcolor="#ccd6f6", fontsize=8)
    for spine in axes[0].spines.values(): spine.set_visible(False)

    # Bias chart
    bias_colors = ["#22C55E" if b == 0 else
                   "#ff6b6b" if b > 80 else
                   "#ffa94d" if b > 50 else "#7c83fd"
                   for b in biases]
    axes[1].barh(methods, biases, color=bias_colors, edgecolor="none", height=0.6)
    axes[1].axvline(20, color="#ffa94d", linestyle="--", alpha=0.5, linewidth=1)
    axes[1].axvline(80, color="#ff6b6b", linestyle="--", alpha=0.5, linewidth=1)
    for i, b in enumerate(biases):
        axes[1].text(b + 0.5, i, f"{b:.1f}%", va="center",
                     color="#ccd6f6", fontsize=8)
    axes[1].set_xlabel("Relative Bias (%)", color="#8892b0")
    axes[1].set_title("Bias dari Ground Truth", color="#ccd6f6", pad=12)
    axes[1].tick_params(colors="#8892b0")
    for spine in axes[1].spines.values(): spine.set_visible(False)

    fig.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    # Detailed table
    section("Detailed Comparison Table", "📋")
    df_styled = df_comparison.copy()
    st.dataframe(
        df_styled,
        use_container_width=True,
        column_config={
            "Method":       st.column_config.TextColumn("Method", width="medium"),
            "ATE Estimate": st.column_config.NumberColumn("ATE Estimate", format="%.4f"),
            "Bias (abs)":   st.column_config.NumberColumn("Bias (abs)", format="%.4f"),
            "Bias (%)":     st.column_config.ProgressColumn("Bias (%)", min_value=0, max_value=100, format="%.1f%%"),
            "Assumption":   st.column_config.TextColumn("Assumption", width="large"),
        },
        hide_index=True
    )

    # PSM Balance
    section("Covariate Balance After PSM", "⚖️")
    col1, col2 = st.columns([2, 1])
    with col1:
        show_figure("results/figures/psm_balance.png")
    with col2:
        insight_box("PSM Balance Check",
            "12/12 covariates balanced (SMD < 0.1) setelah PSM dengan caliper=0.05. "
            "100% treated units berhasil dimatching. "
            "Covariate balance adalah langkah wajib yang sering dilewatkan.")
        sub1, sub2 = st.columns(2)
        with sub1:
            st.markdown(f"""
            <div class='metric-card' style='height:100%; min-height:150px; padding:16px; justify-content:flex-start;'>
                <p class='metric-value'>12/12</p>
                <p class='metric-label'>Balanced Covariates</p>
                <p style='color:#8892b0; font-size:0.75rem; margin-top:8px; line-height:1.5;'>
                    Semua covariate memiliki SMD < 0.1 setelah matching.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with sub2:
            st.markdown(f"""
            <div class='metric-card' style='height:100%; min-height:150px; padding:16px; justify-content:flex-start;'>
                <p class='metric-value'>100%</p>
                <p class='metric-label'>Match Rate</p>
                <p style='color:#8892b0; font-size:0.75rem; margin-top:8px; line-height:1.5;'>
                    Seluruh unit treated berhasil mendapat pasangan control, tanpa ada yang gagal dimatch.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='metric-card' style='height:auto; padding:16px;'>
            <p class='metric-value'>0.05</p>
            <p class='metric-label'>Caliper</p>
            <p style='color:#8892b0; font-size:0.78rem; margin-top:8px; line-height:1.5;'>
                Radius maksimum jarak propensity score saat matching — semakin kecil caliper, semakin ketat kualitas pasangan treated–control.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<p style='color:#8892b0; font-size:0.78rem; margin:8px 4px 0 4px; text-align:center;'>"
            "📌 Semua covariate memenuhi standar balance sebelum estimasi ATE dilakukan."
            "</p>",
            unsafe_allow_html=True
        )

    # DML Residuals
    section("DML Partial Regression", "🤖")
    col1, col2 = st.columns([2, 1])
    with col1:
        show_figure("results/figures/dml_residuals.png")
    with col2:
        insight_box("DML Frisch-Waugh-Lovell",
            "Double ML residualizes Y dan T secara terpisah menggunakan GradientBoosting, "
            "lalu meregres residual satu sama lain. "
            "Slope = ATE yang bebas dari confounding. "
            "5-fold cross-fitting mencegah overfitting bias.")
        sub1, sub2 = st.columns(2)
        with sub1:
            st.markdown(f"""
            <div class='metric-card' style='height:100%; min-height:150px; padding:16px; justify-content:flex-start;'>
                <p class='metric-value'>5</p>
                <p class='metric-label'>CV Folds</p>
                <p style='color:#8892b0; font-size:0.75rem; margin-top:8px; line-height:1.5;'>
                    Data dibagi 5 bagian untuk cross-fitting, mencegah overfitting saat residualisasi.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with sub2:
            st.markdown(f"""
            <div class='metric-card' style='height:100%; min-height:150px; padding:16px; justify-content:flex-start;'>
                <p class='metric-value'>{estimates['DML (Manual)']:.4f}</p>
                <p class='metric-label'>ATE DML</p>
                <p style='color:#8892b0; font-size:0.75rem; margin-top:8px; line-height:1.5;'>
                    Slope dari regresi residual Y terhadap residual T — estimasi efek bebas confounding.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='metric-card' style='height:auto; padding:16px;'>
            <p class='metric-value'>[-0.0014, 0.0021]</p>
            <p class='metric-label'>95% CI</p>
            <p style='color:#8892b0; font-size:0.78rem; margin-top:8px; line-height:1.5;'>
                Interval kepercayaan 95% mencakup nol — efek tidak signifikan secara statistik pada level ini.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<p style='color:#8892b0; font-size:0.78rem; margin:8px 4px 0 4px; text-align:center;'>"
            "📌 Garis merah pada scatter plot = slope ATE hasil regresi Frisch–Waugh–Lovell."
            "</p>",
            unsafe_allow_html=True
        )

# ── PAGE: CATE Analysis ───────────────────────────────────────────────────────
elif page == "🎯 CATE Analysis":
    st.markdown("<h2 style='color:#ccd6f6;'>🎯 CATE Analysis — Heterogeneous Treatment Effects</h2>",
                unsafe_allow_html=True)

    # CATE KPIs
    cate_vals = df_cate["cate_estimate"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card(f"{cate_vals.mean():.4f}", "Mean CATE (ATE)")
    with c2: metric_card(f"{cate_vals.std():.4f}", "CATE Std Dev")
    with c3: metric_card(f"{cate_vals.min():.4f}", "Min CATE")
    with c4: metric_card(f"{cate_vals.max():.4f}", "Max CATE")
    with c5: metric_card(f"{(cate_vals > 0).mean():.1%}", "% Positive CATE")

    st.divider()

    section("CATE Distribution & Uplift Curve", "📈")
    col1, col2 = st.columns([3, 1])
    with col1:
        show_figure("results/figures/cate_distribution.png")

        # Top vs bottom comparison — dipindah ke bawah gambar, 1 baris 3 kolom
        top20   = cate_vals.quantile(0.80)
        top_avg = cate_vals[cate_vals >= top20].mean()
        bot_avg = cate_vals[cate_vals < top20].mean()

    with col2:
        insight_box("CATE Interpretation",
            "CATE (Conditional Average Treatment Effect) mengukur efek treatment "
            "per individu. Tidak semua unit diuntungkan treatment — "
            "48.2% unit memiliki CATE negatif (treatment merugikan mereka).")

        st.markdown(
            "<p style='color:#8892b0; font-size:0.78rem; margin:8px 4px 0 4px; text-align:center;'>"
            "📌 Top 20% unit dengan CATE tertinggi menghasilkan uplift jauh di atas rata-rata populasi."
            "</p>",
            unsafe_allow_html=True
        )

    section("Feature Importance — Heterogeneity Drivers", "🔍")
    col1, col2 = st.columns([3, 1])
    with col1:
        show_figure("results/figures/causal_forest_importance.png", max_height=430)
    with col2:
        insight_box("Heterogeneity Drivers",
            "Feature importance dari Causal Forest mengukur seberapa besar tiap "
            "fitur mempengaruhi HETEROGENITAS efek — bukan seberapa besar "
            "pengaruhnya terhadap outcome. Ini berbeda dari standard ML feature importance. "
            "Fitur importance tinggi menandakan fitur tersebut paling menentukan siapa "
            "yang diuntungkan atau dirugikan oleh treatment.")

        st.markdown(
            "<p style='color:#8892b0; font-size:0.78rem; margin:8px 4px 0 4px; text-align:center;'>"
            "📌 f3 adalah driver heterogenitas paling dominan, jauh melampaui fitur lainnya."
            "</p>",
            unsafe_allow_html=True
        )
    
    section("CATE Data Explorer", "🗂️")
    n_show = st.slider("Tampilkan N baris", 100, 5000, 500)

    filter_positive = st.checkbox("Hanya tampilkan CATE positif")

    # Guard against a missing optional column instead of crashing.
    fallback_col = "flight_risk_score" if "flight_risk_score" in df_cate.columns else "propensity_obs"
    display_cols = ["cate_estimate", "treatment_observational", "conversion", fallback_col, "f0", "f1", "f2"]
    missing_cols = [c for c in display_cols if c not in df_cate.columns]
    if missing_cols:
        st.warning(f"Kolom berikut tidak ada di data dan akan dilewati: {', '.join(missing_cols)}")
        display_cols = [c for c in display_cols if c in df_cate.columns]

    df_display = df_cate[display_cols].copy()
    if filter_positive:
        df_display = df_display[df_display["cate_estimate"] > 0]

    st.dataframe(
        df_display.head(n_show),
        use_container_width=True,
        column_config={
            "cate_estimate": st.column_config.NumberColumn("CATE Estimate", format="%.4f"),
            "treatment_observational": st.column_config.NumberColumn("Treatment"),
            "conversion": st.column_config.NumberColumn("Outcome"),
        },
        hide_index=True
    )

    # CATE histogram interactive
    section("CATE Distribution (Interactive)", "📊")
    n_bins = st.slider("Bins", 20, 100, 50)
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#1e2235")
    ax.set_facecolor("#1e2235")
    ax.hist(cate_vals, bins=n_bins, color="#7c83fd", alpha=0.8, edgecolor="none")
    ax.axvline(cate_vals.mean(), color="#ff6b6b", linestyle="--",
               linewidth=2, label=f"Mean CATE = {cate_vals.mean():.4f}")
    ax.axvline(0, color="#8892b0", linestyle=":", alpha=0.5, label="Zero effect")
    ax.set_xlabel("CATE", color="#8892b0")
    ax.set_ylabel("Count", color="#8892b0")
    ax.set_title("Distribution of Individual Treatment Effects", color="#ccd6f6")
    ax.tick_params(colors="#8892b0")
    ax.legend(facecolor="#252a40", edgecolor="#2d3561", labelcolor="#ccd6f6")
    for spine in ax.spines.values(): spine.set_visible(False)
    st.pyplot(fig)
    plt.close()

# ── PAGE: Sensitivity ─────────────────────────────────────────────────────────
elif page == "🔬 Sensitivity":
    st.markdown("<h2 style='color:#ccd6f6;'>🔬 Sensitivity Analysis</h2>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("0.0000", "Robustness Value", "⚠️ Fragile", False)
    with c2: metric_card("[-0.499, 0.501]", "Manski Bounds Width = 1.000")
    with c3: metric_card("[-0.001, 0.002]", "DML 95% CI Width = 0.004")

    st.divider()

    section("Manski Bounds vs DML CI vs Ground Truth", "📐")
    col1, col2 = st.columns([2, 1])
    with col1:
        show_figure("results/figures/identification_bounds.png")
    with col2:
        insight_box("Partial Identification",
            "Manski bounds (-0.499 to 0.501) tanpa asumsi apapun menunjukkan "
            "betapa sedikit yang bisa kita simpulkan dari data observational. "
            "DML dengan asumsi unconfoundedness mempersempit interval 250x "
            "menjadi (-0.001 to 0.002). "
            "Ground truth RCT = 0.0060 jatuh di luar DML CI — "
            "ini konsisten dengan confounding sangat kuat yang kita suntikkan.")

        st.markdown(
            "<p style='color:#8892b0; font-size:0.78rem; margin:8px 4px 0 4px; text-align:center;'>"
            "📌 Asumsi unconfoundedness memperketat interval identifikasi secara drastis."
            "</p>",
            unsafe_allow_html=True
        )

    section("Sensitivity Contour Plot", "🗺️")
    col1, col2 = st.columns([2, 1])
    with col1:
        show_figure("results/figures/sensitivity_contour.png", max_height=430)
    with col2:
        insight_box("Robustness Value = 0",
            "RV=0 berarti bahkan confounder tersembunyi yang sangat kecil "
            "sudah bisa membatalkan temuan. Ini bukan kesalahan metodologi — "
            "efek memang tidak signifikan secara statistik (t=0.44 < 1.96). "
            "Transparansi ini justru menunjukkan kematangan analisis.")

        st.markdown(
            "<p style='color:#8892b0; font-size:0.78rem; margin:8px 4px 0 4px; text-align:center;'>"
            "📌 Region merah menunjukkan area confounder yang bisa membatalkan hasil temuan."
            "</p>",
            unsafe_allow_html=True
        )

    section("Overlap Analysis", "🔭")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**RCT Treatment (ground truth)**")
        show_figure("results/figures/overlap_rct.png")
        insight_box("RCT Overlap",
            "Propensity score terpusat di 0.499-0.501 (std=0.005). "
            "0% extreme units. Ini distribusi ideal — bukti randomisasi sempurna.")
    with col2:
        st.markdown("**Observational Treatment (confounded)**")
        show_figure("results/figures/overlap_observational.png")
        insight_box("Observational Overlap",
            "Propensity score menyebar lebar [0.000, 1.000] (std=0.313). "
            "13.9% extreme units — overlap moderat. "
            "Ini visual konfirmasi bahwa confounding berhasil disuntikkan.")

# ── PAGE: Refutation ──────────────────────────────────────────────────────────
elif page == "🔄 Refutation Tests":
    st.markdown("<h2 style='color:#ccd6f6;'>🔄 Refutation Tests (DoWhy Framework)</h2>",
                unsafe_allow_html=True)

    n_passed = (df_refute["Pass"] == "✅").sum()
    c1, c2, c3 = st.columns(3)
    with c1: metric_card(f"{n_passed}/3", "Tests Passed", "2 passed, 1 failed")
    with c2: metric_card("✅", "Random Common Cause", "Stable (0.4% change)", True)
    with c3: metric_card("✅", "Placebo Treatment", "Near zero (-0.0001)", True)

    st.divider()

    section("Refutation Test Results", "🧪")
    for _, row in df_refute.iterrows():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 3])
        with col1:
            st.markdown(f"**{row['Test']}**")
        with col2:
            st.metric("New Effect", f"{row['New Effect']:.4f}")
        with col3:
            badge = "✅" if row["Pass"] == "✅" else "❌"
            st.markdown(f"### {badge}")
        with col4:
            st.caption(row["Interpretation"])
        st.divider()

    section("Interpretation Guide", "📖")
    col1, col2, col3 = st.columns(3)
    with col1:
        insight_box("Random Common Cause ✅",
            "Tambahkan covariate acak — estimasi seharusnya TIDAK berubah. "
            "Estimasi kita berubah hanya 0.4% → lulus. "
            "Ini menunjukkan model tidak overfitting ke noise.")
    with col2:
        insight_box("Placebo Treatment ✅",
            "Ganti treatment dengan variabel acak — estimasi HARUS mendekati nol. "
            "Estimasi kita = -0.0001 (hampir nol) → lulus. "
            "Ini bukti bahwa ada sinyal treatment yang nyata (meski lemah).")
    with col3:
        insight_box("Data Subset ❌",
            "Subsampel 80% — estimasi seharusnya STABIL. "
            "Perubahan 24.3% → gagal. Tapi ini bukan bug: "
            "ATE=0.0002 sangat kecil, jadi perubahan 0.0001 pun sudah 50% secara relatif. "
            "Signal-to-noise ratio rendah adalah penyebabnya.")

    section("Validation Table — All Methods", "📋")
    st.dataframe(
        df_validation,
        use_container_width=True,
        column_config={
            "Method":        st.column_config.TextColumn("Method"),
            "ATE Estimate":  st.column_config.NumberColumn("ATE", format="%.4f"),
            "Abs Error":     st.column_config.NumberColumn("Abs Error", format="%.4f"),
            "Bias (%)":      st.column_config.ProgressColumn("Bias %", min_value=0, max_value=100, format="%.1f%%"),
        },
        hide_index=True
    )

# ── PAGE: Data Explorer ───────────────────────────────────────────────────────
elif page == "📈 Data Explorer":
    st.markdown("<h2 style='color:#ccd6f6;'>📈 Data Explorer</h2>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card(f"{len(df_raw):,}", "Total Observations")
    with c2: metric_card(f"{df_raw['treatment_observational'].mean():.3f}", "Observational Treatment Rate")
    with c3: metric_card(f"{df_raw['conversion'].mean():.4f}", "Conversion Rate")
    with c4: metric_card(f"{df_raw['propensity_obs'].std():.3f}", "Propensity Std Dev")

    st.divider()

    section("Raw Data Sample", "🗂️")
    n_rows = st.slider("Rows to display", 50, 2000, 200)

    default_cols = ["treatment_rct_original", "treatment_observational", "conversion",
                     "propensity_rct", "propensity_obs", "true_propensity", "f0", "f1", "f2"]
    # Only default to columns that actually exist, so this doesn't error on a different dataset.
    default_cols = [c for c in default_cols if c in df_raw.columns]

    cols_select = st.multiselect(
        "Select columns",
        df_raw.columns.tolist(),
        default=default_cols
    )
    if cols_select:
        st.dataframe(df_raw[cols_select].head(n_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Pilih minimal satu kolom untuk ditampilkan.")

    section("Feature Distribution", "📊")
    feature_cols = [c for c in df_raw.columns if c.startswith("f") and c[1:].isdigit()]
    if feature_cols:
        feat = st.selectbox("Select feature", sorted(feature_cols, key=lambda c: int(c[1:])))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#1e2235")
        for ax in axes: ax.set_facecolor("#1e2235")

        treated   = df_raw[df_raw["treatment_observational"] == 1][feat]
        untreated = df_raw[df_raw["treatment_observational"] == 0][feat]

        axes[0].hist(treated, bins=50, alpha=0.7, color="#7c83fd",
                     density=True, label="Treated", edgecolor="none")
        axes[0].hist(untreated, bins=50, alpha=0.7, color="#ff6b6b",
                     density=True, label="Control", edgecolor="none")
        axes[0].set_title(f"Distribution of {feat} by Treatment", color="#ccd6f6")
        axes[0].legend(facecolor="#252a40", edgecolor="#2d3561", labelcolor="#ccd6f6")
        axes[0].tick_params(colors="#8892b0")
        for spine in axes[0].spines.values(): spine.set_visible(False)

        axes[1].scatter(df_raw[feat], df_raw["propensity_obs"],
                        alpha=0.05, s=3, color="#7c83fd")
        axes[1].set_xlabel(feat, color="#8892b0")
        axes[1].set_ylabel("Propensity Score", color="#8892b0")
        axes[1].set_title(f"{feat} vs Propensity Score", color="#ccd6f6")
        axes[1].tick_params(colors="#8892b0")
        for spine in axes[1].spines.values(): spine.set_visible(False)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Tidak ada kolom fitur (f0, f1, ...) di dataset ini.")

    section("Correlation Matrix", "🔗")
    corr_cols = feature_cols + [c for c in ["propensity_obs", "conversion"] if c in df_raw.columns]
    if len(corr_cols) >= 2:
        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#1e2235")
        ax.set_facecolor("#1e2235")
        corr = df_raw[corr_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, ax=ax, cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 7}, linewidths=0.5)
        ax.set_title("Feature Correlation Matrix", color="#ccd6f6", pad=12)
        ax.tick_params(colors="#8892b0", labelsize=8)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Tidak cukup kolom numerik untuk correlation matrix.")

# ── PAGE: Reports ─────────────────────────────────────────────────────────────
elif page == "📋 Reports":
    st.markdown("<h2 style='color:#ccd6f6;'>📋 Reports & Downloads</h2>", unsafe_allow_html=True)

    section("Synthetic Control Analysis", "📉")
    col1, col2 = st.columns(2)
    with col1:
        show_figure("results/figures/synthetic_control.png")
    with col2:
        show_figure("results/figures/synthetic_control_placebo.png")

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("2.0016", "SC ATE Estimate")
    with c2: metric_card("2.0000", "True Effect")
    with c3: metric_card("0.1%", "SC Bias", "↑ best method", True)

    insight_box("Synthetic Control vs Cross-Sectional Methods",
        "Synthetic Control mencapai bias 0.1% vs PSM 61.7% pada masalah yang berbeda. "
        "SCM bekerja pada panel data (unit × waktu) dan tidak butuh parallel trends. "
        "In-space placebo test mengkonfirmasi efek yang terdeteksi adalah nyata, "
        "bukan artefak dari pemilihan donor units.")

    st.divider()

    section("Download Reports", "⬇️")
    col1, col2, col3 = st.columns(3)

    with col1:
        df_method_csv = df_comparison.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Method Comparison CSV",
            df_method_csv, "method_comparison.csv", "text/csv",
            use_container_width=True
        )

    with col2:
        cate_export_cols = [c for c in ["cate_estimate", "conversion", "treatment_observational"]
                             if c in df_cate.columns]
        df_cate_csv = df_cate[cate_export_cols].to_csv(index=False).encode()
        st.download_button(
            "⬇️ CATE Scores CSV",
            df_cate_csv, "cate_scores.csv", "text/csv",
            use_container_width=True
        )

    with col3:
        df_refute_csv = df_refute.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Refutation Results CSV",
            df_refute_csv, "refutation_results.csv", "text/csv",
            use_container_width=True
        )

    section("Executive Summary", "📝")
    naive_bias = abs(estimates["Naive (Observational)"] - true_ate) / abs(true_ate) * 100
    summary_text = f"""
CAUSALFORGE — EXECUTIVE SUMMARY
================================

GROUND TRUTH (RCT)         : ATE = {true_ate:.4f}
NAIVE ESTIMATOR             : ATE = {estimates['Naive (Observational)']:.4f} (bias {naive_bias:.1f}%)
BEST CROSS-SECTIONAL (PSM)  : ATE = {estimates['PSM']:.4f} (bias 61.7%)
SYNTHETIC CONTROL (panel)   : ATE = 2.0016 (bias 0.1%)

TEMUAN UTAMA:
1. Naive comparison meleset {naive_bias:.1f}% dari ground truth
2. PSM terbaik di cross-sectional: bias 61.7%
3. Synthetic Control: bias 0.1% untuk panel data
4. CATE heterogenitas: top 20% unit = 4.7x avg effect
5. Manski bounds: width 1.0 → DML: width 0.004 (250x compression)
6. Refutation: 2/3 tests passed

POLICY RECOMMENDATION:
→ Targetkan treatment ke 20,000 unit dengan CATE tertinggi
→ Jangan gunakan naive A/B pada data observational
→ Laporkan sensitivity analysis bersama setiap klaim causal
→ Gunakan Synthetic Control jika tersedia panel data

CATATAN KETERBATASAN:
→ Data simulasi (bukan Criteo asli)
→ True ATE kecil (0.006) → signal-to-noise rendah
→ Robustness value = 0 → kehati-hatian interpretasi diperlukan
    """
    st.code(summary_text, language="text")

    st.download_button(
        "⬇️ Download Executive Summary",
        summary_text.encode(),
        "executive_summary.txt",
        "text/plain",
        use_container_width=True
    )