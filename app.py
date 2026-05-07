import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# ─── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Global Health & Development Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b0f1a;
        color: #c9d1d9;
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #1c2333;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #8b949e !important;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* ── Headings ────────────────────────────────────────────── */
    h1 {
        color: #e6edf3 !important;
        font-weight: 800 !important;
        font-size: 1.85rem !important;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        color: #c9d1d9 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }

    /* ── Metric cards ────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #161b22, #1c2333);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px 24px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #30363d;
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    div[data-testid="stMetric"] label {
        color: #6e7681 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-size: 1.5rem !important;
        font-weight: 700;
    }

    /* ── Tabs ─────────────────────────────────────────────────── */
    button[data-baseweb="tab"] {
        color: #6e7681 !important;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.01em;
        padding-bottom: 12px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #e6edf3 !important;
        border-bottom-color: #58a6ff !important;
    }

    /* ── Dataframe ────────────────────────────────────────────── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Divider ──────────────────────────────────────────────── */
    hr {
        border-color: #21262d !important;
    }

    /* ── Form controls ───────────────────────────────────────── */
    .stRadio > label, .stSelectbox > label,
    .stMultiSelect > label, .stSlider > label {
        color: #8b949e !important;
        font-weight: 500;
        font-size: 0.85rem;
    }

    /* ── Hide chrome ─────────────────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Plotly theme helper ─────────────────────────────────────────────
_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c9d1d9"),
    margin=dict(l=24, r=24, t=36, b=44),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=11),
    ),
)

PALETTE = [
    "#58a6ff", "#f78166", "#3fb950", "#d2a8ff",
    "#79c0ff", "#ffa657", "#56d364", "#bc8cff",
    "#a5d6ff",
]

RADAR_COLOR_A = "rgba(88, 166, 255, 0.72)"   # vivid blue, 72 % opacity
RADAR_COLOR_B = "rgba(247, 129, 102, 0.72)"   # vivid orange-red, 72 % opacity
RADAR_LINE_A  = "#58a6ff"
RADAR_LINE_B  = "#f78166"


def apply_layout(fig, **extra):
    merged = {**_LAYOUT_DEFAULTS, **extra}
    fig.update_layout(**merged)
    return fig


# ─── Data Loading & Preprocessing ────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("UnifiedDataset.csv")
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    df.drop_duplicates(inplace=True)
    countries = [
        "India", "Pakistan", "Bangladesh", "Nepal",
        "Bhutan", "Sri Lanka", "China", "Afghanistan", "Myanmar",
    ]
    return df[df["Country"].isin(countries)].copy()


df = load_and_preprocess()

# ─── Sidebar ─────────────────────────────────────────────────────────
st.sidebar.markdown("## Filters")

selected_countries = st.sidebar.multiselect(
    "Countries",
    options=sorted(df["Country"].unique()),
    default=sorted(df["Country"].unique()),
)

min_yr, max_yr = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider("Year Range", min_yr, max_yr, (min_yr, max_yr))

gender_options = sorted(df["Gender"].unique())
selected_gender = st.sidebar.selectbox("Gender", gender_options, index=0)

mask = (
    df["Country"].isin(selected_countries)
    & df["Year"].between(*year_range)
    & (df["Gender"] == selected_gender)
)
fdf = df[mask].copy()

# ─── Header ──────────────────────────────────────────────────────────
st.markdown("# Global Health & Development Dashboard")
st.caption(
    "Life expectancy, mortality, and socio-economic indicators "
    "across South & East Asian countries — 1990 to 2019."
)
st.divider()

# ─── KPI Row ─────────────────────────────────────────────────────────
if fdf.empty:
    st.warning("No data matches the current filters. Adjust the sidebar.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg Life Expectancy", f"{fdf['Life Expectancy'].mean():.1f} yrs")
k2.metric("Avg Infant Mortality", f"{fdf['Infant Mortality Rate'].mean():.1f}")
k3.metric("Avg Under-5 Mortality", f"{fdf['Under 5 Mortality Rate'].mean():.1f}")
if "GDP per Capita" in fdf.columns:
    k4.metric("Avg GDP / Capita", f"${fdf['GDP per Capita'].mean():,.0f}")

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────
tab_overview, tab_compare, tab_radar, tab_predict = st.tabs(
    ["Overview", "Country Comparison", "Radar Chart", "ML Predictions"]
)

# ═════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═════════════════════════════════════════════════════════════════════
with tab_overview:
    col_a, col_b = st.columns(2)

    # — Life-expectancy trends (line) ─────────────────────────────
    with col_a:
        st.markdown("### Life Expectancy Trends")
        trend = fdf.groupby(["Year", "Country"])["Life Expectancy"].mean().reset_index()
        fig1 = px.line(
            trend, x="Year", y="Life Expectancy", color="Country",
            markers=True,
            color_discrete_sequence=PALETTE,
        )
        apply_layout(fig1, legend=dict(orientation="h", y=-0.22,
                                       bgcolor="rgba(0,0,0,0)",
                                       font=dict(color="#8b949e", size=11)))
        st.plotly_chart(fig1, use_container_width=True)

    # — Box plot ──────────────────────────────────────────────────
    with col_b:
        st.markdown("### Life Expectancy Distribution")
        fig2 = px.box(
            fdf, x="Country", y="Life Expectancy", color="Country",
            color_discrete_sequence=PALETTE,
        )
        apply_layout(fig2, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # — Scatter ───────────────────────────────────────────────────
    st.markdown("### Life Expectancy vs Infant Mortality")
    fig3 = px.scatter(
        fdf, x="Infant Mortality Rate", y="Life Expectancy",
        color="Country", size="Under 5 Mortality Rate",
        hover_data=["Year"],
        color_discrete_sequence=PALETTE,
    )
    apply_layout(fig3)
    st.plotly_chart(fig3, use_container_width=True)

    # — Bar chart ─────────────────────────────────────────────────
    st.markdown("### Average Life Expectancy by Country")
    bar_data = (
        fdf.groupby("Country")["Life Expectancy"]
        .mean()
        .sort_values()
        .reset_index()
    )
    fig4 = px.bar(
        bar_data, x="Country", y="Life Expectancy", color="Country",
        color_discrete_sequence=PALETTE,
    )
    apply_layout(fig4, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 2 — Country Comparison
# ═════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### Compare Two Countries")
    comp_cols = st.columns(2)
    cty_a = comp_cols[0].selectbox("Country A", sorted(df["Country"].unique()), index=0)
    cty_b = comp_cols[1].selectbox("Country B", sorted(df["Country"].unique()), index=1)

    numeric_cols_for_compare = [
        "Life Expectancy", "Infant Mortality Rate", "Under 5 Mortality Rate",
        "GDP per Capita", "Birth Rate", "Death Rate",
    ]
    numeric_cols_for_compare = [c for c in numeric_cols_for_compare if c in df.columns]

    compare_metric = st.selectbox("Metric", numeric_cols_for_compare)

    comp_mask = (
        df["Country"].isin([cty_a, cty_b])
        & df["Year"].between(*year_range)
        & (df["Gender"] == selected_gender)
    )
    comp_df = df[comp_mask]

    comp_trend = comp_df.groupby(["Year", "Country"])[compare_metric].mean().reset_index()
    fig_comp = px.line(
        comp_trend, x="Year", y=compare_metric, color="Country",
        markers=True,
        color_discrete_sequence=[RADAR_LINE_A, RADAR_LINE_B],
    )
    apply_layout(fig_comp)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("### Side-by-Side Metrics (Latest Year in Range)")
    latest_year = comp_df["Year"].max()
    latest = comp_df[comp_df["Year"] == latest_year]

    mc1, mc2 = st.columns(2)
    for metric_name in numeric_cols_for_compare:
        val_a = latest[latest["Country"] == cty_a][metric_name].values
        val_b = latest[latest["Country"] == cty_b][metric_name].values
        va = val_a[0] if len(val_a) else 0
        vb = val_b[0] if len(val_b) else 0
        mc1.metric(f"{cty_a}  |  {metric_name}", f"{va:,.2f}")
        mc2.metric(f"{cty_b}  |  {metric_name}", f"{vb:,.2f}")


# ═════════════════════════════════════════════════════════════════════
# TAB 3 — Radar Chart
# ═════════════════════════════════════════════════════════════════════
with tab_radar:
    st.markdown("### Radar Comparison")

    radar_cols = st.columns(2)
    rc_a = radar_cols[0].selectbox(
        "Country A", sorted(df["Country"].unique()), index=0, key="ra"
    )
    rc_b = radar_cols[1].selectbox(
        "Country B", sorted(df["Country"].unique()), index=1, key="rb"
    )

    radar_metrics = [
        "Life Expectancy", "Infant Mortality Rate",
        "Under 5 Mortality Rate", "Birth Rate", "Death Rate",
    ]
    radar_metrics = [c for c in radar_metrics if c in df.columns]

    radar_year = st.slider("Year", min_yr, max_yr, max_yr, key="radar_yr")

    radar_mask = (
        df["Country"].isin([rc_a, rc_b])
        & (df["Year"] == radar_year)
        & (df["Gender"] == selected_gender)
    )
    rdf = df[radar_mask].groupby("Country")[radar_metrics].mean()

    if rdf.empty:
        st.info("No data for selected parameters.")
    else:
        # --- normalise each metric across ALL countries in the dataset
        # so the two traces sit on the same absolute scale
        all_mask = (df["Year"] == radar_year) & (df["Gender"] == selected_gender)
        all_stats = df[all_mask].groupby("Country")[radar_metrics].mean()
        global_min = all_stats.min()
        global_max = all_stats.max()
        rdf_norm = (rdf - global_min) / (global_max - global_min + 1e-9)

        fig_radar = go.Figure()

        # Country A — blue
        if rc_a in rdf_norm.index:
            vals_a = rdf_norm.loc[rc_a].tolist()
            vals_a += vals_a[:1]
            cats = radar_metrics + [radar_metrics[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_a,
                theta=cats,
                fill="toself",
                fillcolor=RADAR_COLOR_A,
                line=dict(color=RADAR_LINE_A, width=2.5),
                name=rc_a,
            ))

        # Country B — orange-red
        if rc_b in rdf_norm.index:
            vals_b = rdf_norm.loc[rc_b].tolist()
            vals_b += vals_b[:1]
            cats = radar_metrics + [radar_metrics[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_b,
                theta=cats,
                fill="toself",
                fillcolor=RADAR_COLOR_B,
                line=dict(color=RADAR_LINE_B, width=2.5),
                name=rc_b,
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=True,
                    tickfont=dict(color="#484f58", size=10),
                    gridcolor="#21262d",
                    linecolor="#21262d",
                ),
                angularaxis=dict(
                    gridcolor="#21262d",
                    linecolor="#21262d",
                    tickfont=dict(color="#8b949e", size=12),
                ),
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=80, r=80, t=60, b=60),
            title=dict(
                text=f"{rc_a}  vs  {rc_b}  ({radar_year})",
                font=dict(size=15, color="#c9d1d9"),
                x=0.5,
            ),
            legend=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center",
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9", size=13),
            ),
            height=520,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("#### Raw Values")
        st.dataframe(rdf.style.format("{:.2f}"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 4 — ML Predictions
# ═════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Life Expectancy Prediction")
    st.markdown(
        "Three regression models are trained on the filtered data and "
        "evaluated using Mean Absolute Error."
    )

    feature_candidates = [
        "Year", "Infant Mortality Rate", "Under 5 Mortality Rate",
        "Birth Rate", "Death Rate", "GDP per Capita",
        "Diet Calories Fat", "Diet Calories Carbohydrates",
        "Diet Calories Animal Protein", "Diet Calories Plant Protein",
    ]
    feature_candidates = [c for c in feature_candidates if c in fdf.columns]

    if len(feature_candidates) < 2:
        st.warning("Not enough numeric features for modelling with current data.")
    else:
        ml_df = fdf[feature_candidates + ["Life Expectancy"]].dropna()
        if ml_df.shape[0] < 20:
            st.warning("Not enough rows after dropping NaN. Try broader filters.")
        else:
            X = ml_df[feature_candidates]
            y = ml_df["Life Expectancy"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(
                    n_estimators=100, random_state=42
                ),
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                results[name] = {"mae": mae, "preds": preds}

            res_cols = st.columns(3)
            for idx, (name, res) in enumerate(results.items()):
                res_cols[idx].metric(f"{name}  MAE", f"{res['mae']:.3f}")

            st.markdown("### Actual vs Predicted")
            chosen_model = st.radio(
                "Model", list(models.keys()), horizontal=True,
            )
            chosen_preds = results[chosen_model]["preds"]

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=y_test.values,
                y=chosen_preds,
                mode="markers",
                marker=dict(
                    color=RADAR_LINE_A, size=8,
                    opacity=0.75,
                    line=dict(width=1, color="#0b0f1a"),
                ),
                name="Predictions",
            ))
            min_val = min(y_test.min(), chosen_preds.min())
            max_val = max(y_test.max(), chosen_preds.max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color=RADAR_LINE_B, dash="dash", width=2),
                name="Perfect Fit",
            ))
            apply_layout(
                fig_pred,
                xaxis_title="Actual Life Expectancy",
                yaxis_title="Predicted Life Expectancy",
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Risk Categorisation
            st.markdown("### Risk Categorisation")

            def categorize(val):
                if val > 70:
                    return "Low Risk"
                elif val >= 60:
                    return "Medium Risk"
                else:
                    return "High Risk"

            risk_df = fdf.copy()
            risk_df["Risk"] = risk_df["Life Expectancy"].apply(categorize)

            fig_risk = px.histogram(
                risk_df, x="Risk", color="Risk",
                category_orders={
                    "Risk": ["High Risk", "Medium Risk", "Low Risk"]
                },
                color_discrete_map={
                    "High Risk": "#f85149",
                    "Medium Risk": "#d29922",
                    "Low Risk": "#3fb950",
                },
            )
            apply_layout(fig_risk, showlegend=False,
                         title="Risk Category Distribution")
            st.plotly_chart(fig_risk, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#484f58;font-size:0.78rem;"
    "letter-spacing:0.04em;'>"
    "Global Health & Development Dashboard  ·  Source: UnifiedDataset.csv"
    "</p>",
    unsafe_allow_html=True,
)
