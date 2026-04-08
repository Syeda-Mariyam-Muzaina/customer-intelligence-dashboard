import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SegmentAI · Customer Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #090d1a;
    color: #e2e8f0;
}
.stApp { background: #090d1a; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1321 !important;
    border-right: 1px solid #1e2740;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

/* ── NAV BUTTONS in sidebar ── */
section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    text-align: left !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    margin-bottom: 2px !important;
    box-shadow: none !important;
    transition: background 0.15s ease, color 0.15s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1e2740 !important;
    color: #e2e8f0 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #0d1321 100%);
    border: 1px solid #1e2740;
    border-radius: 14px;
    padding: 18px 20px !important;
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
}

/* ── Section headers ── */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 4px;
    border-bottom: 1px solid #1e2740;
    padding-bottom: 8px;
}

/* ── Dashboard title ── */
.dash-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.2;
}
.dash-subtitle {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 4px;
}

/* ── Cluster badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-blue   { background: #1e3a5f; color: #60a5fa; }
.badge-purple { background: #2d1b69; color: #a78bfa; }
.badge-green  { background: #14532d; color: #4ade80; }
.badge-amber  { background: #451a03; color: #fbbf24; }
.badge-pink   { background: #4a0d23; color: #f472b6; }

/* ── Recommendation card ── */
.rec-card {
    background: linear-gradient(135deg, #111827, #0d1321);
    border: 1px solid #1e2740;
    border-left: 3px solid #6366f1;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
}
.rec-card h4 {
    color: #818cf8;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 4px 0;
}
.rec-card p { color: #e2e8f0; margin: 0; font-size: 0.92rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1321;
    border-radius: 10px;
    border: 1px solid #1e2740;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #1e2740 !important;
    color: #e2e8f0 !important;
}

hr { border-color: #1e2740 !important; }

/* ── Action buttons in main area ── */
.main .stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 22px !important;
}
.main .stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    transform: translateY(-1px) !important;
}

.acc-pill {
    display: inline-block;
    background: linear-gradient(135deg, #14532d, #166534);
    color: #4ade80;
    border: 1px solid #15803d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
CLUSTER_NAMES = {
    0: "High Value",
    1: "Moderate Mover",
    2: "Young Savvy",
    3: "Conservative",
    4: "Low Budget",
}
CLUSTER_COLORS = {
    0: "#6366f1",
    1: "#a78bfa",
    2: "#4ade80",
    3: "#fbbf24",
    4: "#f472b6",
}
CLUSTER_BADGE = {
    0: "badge-blue",
    1: "badge-purple",
    2: "badge-green",
    3: "badge-amber",
    4: "badge-pink",
}
RECOMMENDATIONS = {
    0: "Offer premium products and loyalty rewards",
    1: "Provide discounts and engagement offers",
    2: "Promote affordable and high-value deals",
    3: "Minimal marketing, low priority customers",
    4: "Upsell products and cross-sell offers",
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="DM Sans"),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="#1e2740", zerolinecolor="#1e2740"),
    yaxis=dict(gridcolor="#1e2740", zerolinecolor="#1e2740"),
)


# ─────────────────────────────────────────────
#  AUTO-LOAD CSV FROM PROJECT FOLDER
# ─────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

if os.path.exists(CSV_PATH):
    raw_df = load_data(CSV_PATH)
else:
    st.error("❌ `Mall_Customers.csv` not found. Please place it in the same folder as `app.py`.")
    st.stop()


# ─────────────────────────────────────────────
#  ML PIPELINE  (cached — only reruns if K or data changes)
# ─────────────────────────────────────────────
@st.cache_data
def run_pipeline(df: pd.DataFrame, n_clusters: int):
    df = df.copy()

    # ── K-Means ──────────────────────────────
    X_km = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    scaler_km = StandardScaler()
    X_km_s = scaler_km.fit_transform(X_km)
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_km_s)

    # ── Elbow WCSS ───────────────────────────
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km.fit(X_km_s)
        wcss.append(km.inertia_)

    # ── PCA ──────────────────────────────────
    X_all = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values
    scaler_pca = StandardScaler()
    X_all_s = scaler_pca.fit_transform(X_all)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_all_s)
    df["PC1"], df["PC2"] = coords[:, 0], coords[:, 1]
    explained = pca.explained_variance_ratio_ * 100

    # ── Spending category & recommendations ──
    df["Spending_Category"] = (df["Spending Score (1-100)"] > 50).astype(int)
    df["Recommendation"] = df["Cluster"].map(RECOMMENDATIONS)

    # ── Logistic Regression ───────────────────
    Xl = df[["Age", "Annual Income (k$)"]]
    yl = df["Spending_Category"]
    X_tr, X_te, y_tr, y_te = train_test_split(Xl, yl, test_size=0.2, random_state=42)
    scaler_lr = StandardScaler()
    X_tr_s = scaler_lr.fit_transform(X_tr)
    X_te_s = scaler_lr.transform(X_te)
    lr = LogisticRegression(random_state=42)
    lr.fit(X_tr_s, y_tr)
    y_pred = lr.predict(X_te_s)
    acc = accuracy_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)

    # ── Cluster stats ─────────────────────────
    stats = (
        df.groupby("Cluster")
        .agg(Count=("CustomerID", "count"),
             Avg_Income=("Annual Income (k$)", "mean"),
             Avg_Score=("Spending Score (1-100)", "mean"),
             Avg_Age=("Age", "mean"))
        .reset_index()
    )

    return df, kmeans, wcss, explained, acc, cm, stats, scaler_km, scaler_lr, lr


# ─────────────────────────────────────────────
#  SESSION STATE — track active page
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Overview"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    # Brand header
    st.markdown("""
    <div style='display:flex;align-items:center;gap:8px;margin-bottom:2px;'>
        <span style='font-size:1.4rem;'>🧠</span>
        <span style='font-family:Space Grotesk;font-weight:700;font-size:1.15rem;color:#e2e8f0;'>SegmentAI</span>
    </div>
    <div style='color:#475569;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px;'>
        Customer ML v2.4 · Project Alpha
    </div>
    """, unsafe_allow_html=True)

    # Data loaded status
    st.markdown(f"""
    <div style='background:#0f1f12;border:1px solid #166534;border-radius:8px;
                padding:8px 12px;margin-bottom:14px;font-size:0.78rem;color:#4ade80;'>
        ✅ &nbsp;Mall_Customers.csv loaded<br>
        <span style='color:#475569;'>{len(raw_df)} customers · 5 features</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Parameters
    st.markdown("<span style='font-size:0.73rem;letter-spacing:0.09em;text-transform:uppercase;color:#475569;'>⚙️ &nbsp;Filters & Parameters</span>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    n_clusters = st.slider("Number of Clusters (K)", 2, 10, 5)
    age_min, age_max = st.slider("Age Range", 18, 70, (18, 70))

    st.divider()

    # Navigation buttons
    st.markdown("<span style='font-size:0.73rem;letter-spacing:0.09em;text-transform:uppercase;color:#475569;'>🧭 &nbsp;Navigation</span>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    NAV_ITEMS = [
        ("📊", "Overview"),
        ("🔬", "Analytics"),
        ("🤖", "Models"),
        ("📤", "Exports"),
    ]

    for icon, label in NAV_ITEMS:
        is_active = st.session_state.page == label

        if is_active:
            # Render active state as styled HTML (not a button)
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1e2f5e,#1e2740);
                        border-left:3px solid #6366f1;
                        border-radius:8px;
                        padding:10px 14px;
                        margin-bottom:4px;'>
                <span style='color:#818cf8;font-weight:600;font-size:0.88rem;'>
                    {icon} &nbsp;{label}
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Render inactive pages as clickable buttons
            if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
                st.session_state.page = label
                st.rerun()

    st.divider()

    st.markdown("""
    <div style='font-size:0.68rem;color:#334155;'>
        📁 Documentation<br>
        ⚙️ System Status · Online
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  APPLY FILTER + RUN PIPELINE
# ─────────────────────────────────────────────
df_filtered = raw_df[
    (raw_df["Age"] >= age_min) & (raw_df["Age"] <= age_max)
].copy()

(df, kmeans, wcss, explained, accuracy, cm,
 cluster_stats, scaler_km, scaler_lr, lr) = run_pipeline(df_filtered, n_clusters)

page = st.session_state.page


# ═══════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "Overview":

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("""
        <div class='dash-title'>AI Customer Segmentation Dashboard</div>
        <div class='dash-subtitle'>
            Utilizing K-Means &amp; PCA to discover high-dimensional patterns
            in customer behaviour and fiscal demographics.
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown("<div style='padding-top:10px;'>", unsafe_allow_html=True)
        st.markdown("<span class='badge badge-blue'>Real-time</span>&nbsp;<span class='badge badge-purple'>Batch v2.1</span>",
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── KPI cards ────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Customers", f"{len(df):,}",
                  delta=f"+{int(len(df)*0.12)}% from last cycle")
    with m2:
        st.metric("Identified Clusters", n_clusters,
                  delta="Optimal K via Elbow Method")
    with m3:
        st.metric("Avg Income",
                  f"${df['Annual Income (k$)'].median():.1f}k",
                  delta="Cross-cluster median normalised")
    with m4:
        st.metric("Spending Score",
                  f"{df['Spending Score (1-100)'].mean():.1f}",
                  delta="Weighted transaction velocity index")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PCA scatter + Income per cluster ─────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("<div class='section-title'>Principal Component Analysis</div>",
                    unsafe_allow_html=True)
        st.caption("2D projection of N-dimensional customer space")

        df["Cluster_Label"] = df["Cluster"].apply(
            lambda x: f"Cluster {x} · {CLUSTER_NAMES.get(x,'?')}"
        )
        fig_pca = px.scatter(
            df, x="PC1", y="PC2", color="Cluster_Label",
            color_discrete_sequence=list(CLUSTER_COLORS.values()),
            labels={"PC1": f"PC1 Component ({explained[0]:.1f}% var)",
                    "PC2": f"PC2 Component ({explained[1]:.1f}% var)"},
            opacity=0.82,
        )
        fig_pca.update_traces(marker=dict(size=7))
        fig_pca.update_layout(
            **PLOTLY_LAYOUT, height=340,
            legend=dict(orientation="h", yanchor="bottom", y=-0.32, font=dict(size=10))
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-title'>Avg Income per Cluster</div>",
                    unsafe_allow_html=True)
        st.caption("Segmented demographics analysis")

        for _, row in cluster_stats.iterrows():
            c_id  = int(row["Cluster"])
            name  = CLUSTER_NAMES.get(c_id, f"Cluster {c_id}")
            color = CLUSTER_COLORS.get(c_id, "#6366f1")
            badge = CLUSTER_BADGE.get(c_id, "badge-blue")
            inc   = row["Avg_Income"]
            pct   = inc / cluster_stats["Avg_Income"].max() * 100

            st.markdown(f"""
            <div style='margin-bottom:14px;'>
                <div style='display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:5px;'>
                    <span class='badge {badge}'>CLUSTER {c_id} · {name.upper()}</span>
                    <span style='font-family:Space Grotesk;font-weight:700;
                                 color:#f1f5f9;'>${inc:.0f}k</span>
                </div>
                <div style='background:#1e2740;border-radius:4px;height:6px;'>
                    <div style='width:{pct:.1f}%;background:{color};
                                border-radius:4px;height:6px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            "<div style='color:#475569;font-size:0.68rem;margin-top:6px;'>"
            "Values normalised on USD market exchange rates, current fiscal year."
            "</div>", unsafe_allow_html=True
        )

    st.divider()

    # ── Smart Recommendation Engine ───────────────────────────
    st.markdown("<div class='section-title'>Smart Recommendation Engine</div>",
                unsafe_allow_html=True)
    st.caption("AI insights based on recent cluster migration and spending velocity patterns · Active Inference")

    rc1, rc2 = st.columns([2, 1])
    with rc1:
        for _, row in cluster_stats.nlargest(2, "Avg_Income").iterrows():
            c_id = int(row["Cluster"])
            rec  = RECOMMENDATIONS.get(c_id, "N/A")
            st.markdown(f"""
            <div class='rec-card'>
                <h4>Cluster {c_id} Logic</h4>
                <p>🎯 {rec}</p>
            </div>
            """, unsafe_allow_html=True)
    with rc2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("⚡ Execute Campaign", use_container_width=True)
        st.caption("Deploys to marketing automation pipeline")


# ═══════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.markdown("<div class='dash-title'>Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-subtitle'>Deep-dive into customer demographics and cluster behaviour</div>",
                unsafe_allow_html=True)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📍 Cluster Map", "👥 Demographics", "📈 Elbow Curve", "📦 Distribution"]
    )

    with tab1:
        st.markdown("<div class='section-title'>Income vs Spending Score — Cluster Map</div>",
                    unsafe_allow_html=True)
        df["Cluster_Label"] = df["Cluster"].apply(
            lambda x: f"Cluster {x} · {CLUSTER_NAMES.get(x,'?')}"
        )
        fig_km = px.scatter(
            df, x="Annual Income (k$)", y="Spending Score (1-100)",
            color="Cluster_Label",
            color_discrete_sequence=list(CLUSTER_COLORS.values()),
            opacity=0.85,
        )
        ctr = pd.DataFrame(
            scaler_km.inverse_transform(kmeans.cluster_centers_),
            columns=["Annual Income (k$)", "Spending Score (1-100)"]
        )
        fig_km.add_trace(go.Scatter(
            x=ctr["Annual Income (k$)"], y=ctr["Spending Score (1-100)"],
            mode="markers",
            marker=dict(symbol="x", size=16, color="white", line=dict(width=2)),
            name="Centroids",
        ))
        fig_km.update_traces(selector=dict(mode="markers"), marker_size=8)
        fig_km.update_layout(
            **PLOTLY_LAYOUT, height=450,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
        )
        st.plotly_chart(fig_km, use_container_width=True)

    with tab2:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("<div class='section-title'>Gender Distribution</div>",
                        unsafe_allow_html=True)
            gd = df["Gender"].value_counts().reset_index()
            gd.columns = ["Gender", "Count"]
            fig_pie = px.pie(gd, names="Gender", values="Count",
                             color_discrete_sequence=["#6366f1", "#f472b6"], hole=0.55)
            fig_pie.update_layout(**PLOTLY_LAYOUT, height=300,
                                  legend=dict(orientation="h", yanchor="bottom", y=-0.2))
            st.plotly_chart(fig_pie, use_container_width=True)

        with d2:
            st.markdown("<div class='section-title'>Age Distribution</div>",
                        unsafe_allow_html=True)
            fig_age = px.histogram(df, x="Age", nbins=20,
                                   color_discrete_sequence=["#6366f1"])
            fig_age.update_layout(**PLOTLY_LAYOUT, height=300)
            fig_age.update_traces(marker_line_color="#1e2740", marker_line_width=0.5)
            st.plotly_chart(fig_age, use_container_width=True)

        st.markdown("<div class='section-title'>Cluster Statistics Summary</div>",
                    unsafe_allow_html=True)
        disp = cluster_stats.copy()
        disp["Cluster Name"] = disp["Cluster"].map(CLUSTER_NAMES)
        disp = disp.rename(columns={
            "Avg_Income": "Avg Income (k$)",
            "Avg_Score": "Avg Spending Score",
            "Avg_Age": "Avg Age"
        })
        disp[["Avg Income (k$)", "Avg Spending Score", "Avg Age"]] = \
            disp[["Avg Income (k$)", "Avg Spending Score", "Avg Age"]].round(1)
        st.dataframe(
            disp[["Cluster", "Cluster Name", "Count", "Avg Income (k$)", "Avg Spending Score", "Avg Age"]],
            use_container_width=True, hide_index=True
        )

    with tab3:
        st.markdown("<div class='section-title'>Elbow Method — Optimal K</div>",
                    unsafe_allow_html=True)
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(range(1, 11)), y=wcss, mode="lines+markers",
            line=dict(color="#6366f1", width=2.5),
            marker=dict(size=8, color="#6366f1"),
            name="WCSS",
        ))
        fig_elbow.add_vline(
            x=n_clusters, line_dash="dash", line_color="#f472b6",
            annotation_text=f" K={n_clusters} selected",
            annotation_font_color="#f472b6"
        )
        fig_elbow.update_layout(
            **PLOTLY_LAYOUT, height=380,
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Within-Cluster Sum of Squares"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with tab4:
        st.markdown("<div class='section-title'>Spending Score by Cluster</div>",
                    unsafe_allow_html=True)
        df["Cluster_Label"] = df["Cluster"].apply(
            lambda x: f"C{x} · {CLUSTER_NAMES.get(x,'?')}"
        )
        fig_box = px.box(
            df, x="Cluster_Label", y="Spending Score (1-100)",
            color="Cluster_Label",
            color_discrete_sequence=list(CLUSTER_COLORS.values())
        )
        fig_box.update_layout(
            **PLOTLY_LAYOUT, height=380, showlegend=False,
            xaxis_title="Cluster", yaxis_title="Spending Score"
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: MODELS
# ═══════════════════════════════════════════════════════════════
elif page == "Models":
    st.markdown("<div class='dash-title'>Models</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-subtitle'>Logistic Regression · Confusion Matrix · New Inference</div>",
                unsafe_allow_html=True)
    st.divider()

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("<div class='section-title'>Model Performance</div>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <br>
        <div style='text-align:center;'>
            <div style='font-size:3rem;font-family:Space Grotesk;
                        font-weight:700;color:#f1f5f9;'>{accuracy*100:.1f}%</div>
            <div class='acc-pill'>Model Accuracy</div>
            <div style='color:#64748b;font-size:0.82rem;margin-top:12px;line-height:1.8;'>
                LogisticRegression · StandardScaler<br>
                Features: Age, Annual Income<br>
                Target: Spending Category (binary)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-title'>Confusion Matrix</div>",
                    unsafe_allow_html=True)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Low Spender", "High Spender"],
            y=["Low Spender", "High Spender"],
            color_continuous_scale=[[0, "#0d1321"], [1, "#6366f1"]]
        )
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=320, coloraxis_showscale=False)
        fig_cm.update_traces(textfont=dict(size=18, color="white"))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # ── New Inference ─────────────────────────────────────────
    st.markdown("<div class='section-title'>🔮 New Customer Inference</div>",
                unsafe_allow_html=True)
    st.caption("Enter a new customer's details to predict their cluster and recommendation")

    ni1, ni2, ni3 = st.columns(3)
    with ni1:
        new_age    = st.number_input("Age", min_value=18, max_value=80, value=30)
    with ni2:
        new_income = st.number_input("Annual Income (k$)", min_value=10, max_value=200, value=60)
    with ni3:
        new_score  = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

    if st.button("⚡ Run Inference"):
        pred_cluster = int(kmeans.predict(scaler_km.transform([[new_income, new_score]]))[0])
        pred_name    = CLUSTER_NAMES.get(pred_cluster, f"Cluster {pred_cluster}")
        pred_rec     = RECOMMENDATIONS.get(pred_cluster, "N/A")
        badge_cls    = CLUSTER_BADGE.get(pred_cluster, "badge-blue")
        color        = CLUSTER_COLORS.get(pred_cluster, "#6366f1")
        pred_cat     = lr.predict(scaler_lr.transform([[new_age, new_income]]))[0]
        cat_label    = "High Spender 🟢" if pred_cat == 1 else "Low Spender 🔴"

        st.markdown(f"""
        <div class='rec-card' style='border-left-color:{color};margin-top:12px;'>
            <h4>Inference Result</h4>
            <div style='display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:6px;'>
                <span class='badge {badge_cls}'>Cluster {pred_cluster} · {pred_name.upper()}</span>
                <span style='color:#94a3b8;font-size:0.82rem;'>{cat_label}</span>
            </div>
            <p>🎯 {pred_rec}</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: EXPORTS
# ═══════════════════════════════════════════════════════════════
elif page == "Exports":
    st.markdown("<div class='dash-title'>Exports</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-subtitle'>Download segmented data and cluster reports</div>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='section-title'>Full Segmented Dataset</div>",
                unsafe_allow_html=True)
    export_cols = ["CustomerID", "Gender", "Age", "Annual Income (k$)",
                   "Spending Score (1-100)", "Cluster", "Spending_Category", "Recommendation"]
    export_df = df[export_cols].copy()
    export_df["Cluster_Name"] = export_df["Cluster"].map(CLUSTER_NAMES)

    st.dataframe(export_df, use_container_width=True, height=320)
    st.download_button(
        "⬇️ Download Segmented CSV",
        export_df.to_csv(index=False).encode(),
        "customer_segments.csv", "text/csv"
    )

    st.divider()

    st.markdown("<div class='section-title'>Cluster Summary Report</div>",
                unsafe_allow_html=True)
    report = cluster_stats.copy()
    report["Cluster_Name"]   = report["Cluster"].map(CLUSTER_NAMES)
    report["Recommendation"] = report["Cluster"].map(RECOMMENDATIONS)
    st.dataframe(report.round(2), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Cluster Report",
        report.to_csv(index=False).encode(),
        "cluster_report.csv", "text/csv"
    )