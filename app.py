import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Restaurant Growth Potential Modeling",
    page_icon="🍽️",
    layout="wide"
)

st.markdown("""
<style>
.main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
.subtitle { font-size: 1rem; color: #555; margin-bottom: 2rem; }
.metric-card { background: #f8f9fa; border-radius: 10px; padding: 1rem; text-align: center; }
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Synthetic Dataset ──────────────────────────────────────────────
@st.cache_data
def generate_data(n=500, seed=42):
    np.random.seed(seed)
    cuisines = ["Indian", "Chinese", "Italian", "Mexican", "Continental", "South Indian", "Fast Food"]
    locations = ["Urban Core", "Suburban", "Rural", "Mall", "Airport", "Highway"]

    df = pd.DataFrame({
        "restaurant_id": [f"R{i:04d}" for i in range(n)],
        "restaurant_name": [f"Restaurant {i}" for i in range(n)],
        "cuisine_type": np.random.choice(cuisines, n),
        "location_type": np.random.choice(locations, n),
        "avg_order_value": np.random.uniform(150, 1200, n).round(2),
        "monthly_orders": np.random.randint(200, 5000, n),
        "customer_rating": np.random.uniform(2.5, 5.0, n).round(1),
        "num_reviews": np.random.randint(10, 3000, n),
        "seating_capacity": np.random.randint(20, 300, n),
        "num_staff": np.random.randint(5, 80, n),
        "delivery_radius_km": np.random.uniform(1, 15, n).round(1),
        "years_in_operation": np.random.uniform(0.5, 20, n).round(1),
        "marketing_spend_monthly": np.random.uniform(500, 50000, n).round(0),
        "food_cost_pct": np.random.uniform(25, 45, n).round(1),
        "repeat_customer_rate": np.random.uniform(0.1, 0.8, n).round(2),
        "online_presence_score": np.random.uniform(1, 10, n).round(1),
        "competitor_density": np.random.randint(1, 30, n),
        "local_population_density": np.random.uniform(500, 15000, n).round(0),
    })

    # Derived features
    df["monthly_revenue"] = (df["avg_order_value"] * df["monthly_orders"]).round(0)
    df["revenue_per_seat"] = (df["monthly_revenue"] / df["seating_capacity"]).round(2)
    df["staff_efficiency"] = (df["monthly_orders"] / df["num_staff"]).round(2)

    # Growth potential score (0–100)
    score = (
        0.20 * (df["customer_rating"] / 5) * 100 +
        0.15 * (df["repeat_customer_rate"]) * 100 +
        0.15 * (df["online_presence_score"] / 10) * 100 +
        0.10 * (np.log1p(df["monthly_revenue"]) / np.log1p(df["monthly_revenue"].max())) * 100 +
        0.10 * (df["staff_efficiency"] / df["staff_efficiency"].max()) * 100 +
        0.10 * (1 - df["food_cost_pct"] / 100) * 100 +
        0.10 * (df["marketing_spend_monthly"] / df["marketing_spend_monthly"].max()) * 100 +
        0.10 * (df["local_population_density"] / df["local_population_density"].max()) * 100
    )
    df["growth_potential_score"] = score.round(2)

    # Strategic tier
    df["strategic_tier"] = pd.cut(
        df["growth_potential_score"],
        bins=[0, 40, 60, 80, 100],
        labels=["Low Potential", "Moderate Potential", "High Potential", "Star Performer"]
    )
    return df

df = generate_data()

# ── Clustering ─────────────────────────────────────────────────────
@st.cache_data
def run_clustering(data, k=4):
    features = ["avg_order_value", "monthly_orders", "customer_rating",
                "repeat_customer_rate", "online_presence_score", "staff_efficiency",
                "revenue_per_seat", "marketing_spend_monthly"]
    X = data[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return clusters, coords, pca.explained_variance_ratio_

clusters, pca_coords, var_ratio = run_clustering(df)
df["cluster"] = clusters.astype(str)
df["pca_x"] = pca_coords[:, 0]
df["pca_y"] = pca_coords[:, 1]

cluster_labels = {
    "0": "Steady Locals",
    "1": "Growth Stars",
    "2": "Budget Runners",
    "3": "Premium Players"
}
df["cluster_label"] = df["cluster"].map(cluster_labels)

# ── Growth Model ───────────────────────────────────────────────────
@st.cache_data
def train_model(data):
    feat_cols = ["avg_order_value", "monthly_orders", "customer_rating", "num_reviews",
                 "seating_capacity", "num_staff", "delivery_radius_km", "years_in_operation",
                 "marketing_spend_monthly", "food_cost_pct", "repeat_customer_rate",
                 "online_presence_score", "competitor_density", "local_population_density"]
    X = data[feat_cols]
    y = data["growth_potential_score"]
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=True)
    return model, importances, feat_cols

model, importances, feat_cols = train_model(df)

# ── UI ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🍽️ Restaurant Growth Potential Modeling</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Strategic Classification System — Data Science & ML Pipeline</p>', unsafe_allow_html=True)

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Restaurants", f"{len(df):,}")
col2.metric("Avg Growth Score", f"{df['growth_potential_score'].mean():.1f}/100")
col3.metric("Star Performers", f"{(df['strategic_tier']=='Star Performer').sum()}")
col4.metric("Avg Monthly Revenue", f"₹{df['monthly_revenue'].mean():,.0f}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Overview", "🔵 Cluster Analysis", "📈 Growth Scoring",
    "🤖 Predict My Restaurant", "📋 Strategic Report"
])

# ── Tab 1: Dataset ─────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Overview")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(df, x="growth_potential_score", color="strategic_tier",
                           nbins=40, title="Distribution of Growth Potential Scores",
                           color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tier_counts = df["strategic_tier"].value_counts().reset_index()
        fig2 = px.pie(tier_counts, names="strategic_tier", values="count",
                      title="Strategic Tier Distribution",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        avg_by_cuisine = df.groupby("cuisine_type")["growth_potential_score"].mean().sort_values()
        fig3 = px.bar(avg_by_cuisine, orientation="h",
                      title="Avg Growth Score by Cuisine Type",
                      labels={"value": "Avg Score", "index": "Cuisine"})
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        avg_by_loc = df.groupby("location_type")["growth_potential_score"].mean().sort_values()
        fig4 = px.bar(avg_by_loc, orientation="h",
                      title="Avg Growth Score by Location Type",
                      labels={"value": "Avg Score", "index": "Location"})
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(50), use_container_width=True)

# ── Tab 2: Cluster Analysis ────────────────────────────────────────
with tab2:
    st.subheader("Unsupervised Clustering (K-Means + PCA)")
    st.info(f"PCA explains {var_ratio[0]*100:.1f}% + {var_ratio[1]*100:.1f}% = {sum(var_ratio)*100:.1f}% of variance")

    fig_c = px.scatter(df, x="pca_x", y="pca_y", color="cluster_label",
                       hover_data=["restaurant_name", "cuisine_type", "growth_potential_score"],
                       title="Restaurant Clusters in PCA Space",
                       color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig_c, use_container_width=True)

    cluster_summary = df.groupby("cluster_label").agg(
        Count=("restaurant_id", "count"),
        Avg_Growth_Score=("growth_potential_score", "mean"),
        Avg_Revenue=("monthly_revenue", "mean"),
        Avg_Rating=("customer_rating", "mean"),
        Avg_Repeat_Rate=("repeat_customer_rate", "mean")
    ).round(2)
    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary, use_container_width=True)

# ── Tab 3: Feature Importance ──────────────────────────────────────
with tab3:
    st.subheader("Growth Potential Scoring Model")
    st.markdown("**Gradient Boosting Regressor** trained on 500 synthetic restaurants.")

    fig_imp = px.bar(importances, orientation="h",
                     title="Feature Importance for Growth Score Prediction",
                     labels={"value": "Importance", "index": "Feature"},
                     color=importances.values,
                     color_continuous_scale="Teal")
    st.plotly_chart(fig_imp, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_s = px.scatter(df, x="customer_rating", y="growth_potential_score",
                           color="strategic_tier", size="monthly_revenue",
                           title="Rating vs Growth Score",
                           color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig_s, use_container_width=True)

    with c2:
        fig_s2 = px.scatter(df, x="repeat_customer_rate", y="growth_potential_score",
                            color="strategic_tier", size="monthly_orders",
                            title="Repeat Rate vs Growth Score",
                            color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig_s2, use_container_width=True)

# ── Tab 4: Predictor ───────────────────────────────────────────────
with tab4:
    st.subheader("🤖 Predict Your Restaurant's Growth Potential")
    st.markdown("Fill in your restaurant's details to get an AI-powered growth score and strategic recommendation.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            avg_order = st.number_input("Avg Order Value (₹)", 100, 5000, 350)
            monthly_ord = st.number_input("Monthly Orders", 50, 10000, 800)
            rating = st.slider("Customer Rating", 1.0, 5.0, 4.0, 0.1)
            num_rev = st.number_input("Number of Reviews", 10, 5000, 200)
        with c2:
            seats = st.number_input("Seating Capacity", 10, 500, 60)
            staff = st.number_input("Number of Staff", 2, 100, 15)
            del_radius = st.slider("Delivery Radius (km)", 1.0, 20.0, 5.0, 0.5)
            years_op = st.slider("Years in Operation", 0.5, 25.0, 3.0, 0.5)
        with c3:
            mktg = st.number_input("Monthly Marketing Spend (₹)", 0, 100000, 5000)
            food_cost = st.slider("Food Cost %", 20, 60, 33)
            repeat_rate = st.slider("Repeat Customer Rate", 0.05, 0.95, 0.35, 0.05)
            online_score = st.slider("Online Presence Score (1–10)", 1.0, 10.0, 6.0, 0.5)
            comp_density = st.slider("Competitor Density (nearby)", 1, 30, 10)
            pop_density = st.number_input("Local Population Density", 500, 20000, 5000)

        submitted = st.form_submit_button("🔍 Predict Growth Potential", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "avg_order_value": avg_order, "monthly_orders": monthly_ord,
            "customer_rating": rating, "num_reviews": num_rev,
            "seating_capacity": seats, "num_staff": staff,
            "delivery_radius_km": del_radius, "years_in_operation": years_op,
            "marketing_spend_monthly": mktg, "food_cost_pct": food_cost,
            "repeat_customer_rate": repeat_rate, "online_presence_score": online_score,
            "competitor_density": comp_density, "local_population_density": pop_density
        }])
        score = model.predict(input_data)[0]
        score = float(np.clip(score, 0, 100))

        if score >= 80:
            tier, color, advice = "⭐ Star Performer", "success", "Your restaurant has exceptional growth potential. Focus on expanding delivery zones and franchising."
        elif score >= 60:
            tier, color, advice = "📈 High Potential", "info", "Strong fundamentals. Invest in online presence and loyalty programs to reach Star level."
        elif score >= 40:
            tier, color, advice = "📊 Moderate Potential", "warning", "Decent base. Focus on customer retention and marketing ROI to grow score."
        else:
            tier, color, advice = "⚠️ Low Potential", "error", "Review pricing, reduce food cost %, and boost your online presence immediately."

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Growth Potential Score", f"{score:.1f} / 100")
        m2.metric("Strategic Tier", tier)
        m3.metric("Est. Monthly Revenue", f"₹{avg_order * monthly_ord:,.0f}")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#2ecc71"},
                   "steps": [
                       {"range": [0, 40], "color": "#e74c3c"},
                       {"range": [40, 60], "color": "#f39c12"},
                       {"range": [60, 80], "color": "#3498db"},
                       {"range": [80, 100], "color": "#2ecc71"},
                   ]},
            title={"text": "Growth Potential Score"}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if color == "success":
            st.success(advice)
        elif color == "info":
            st.info(advice)
        elif color == "warning":
            st.warning(advice)
        else:
            st.error(advice)

# ── Tab 5: Strategic Report ────────────────────────────────────────
with tab5:
    st.subheader("📋 Strategic Classification Report")

    tier_filter = st.selectbox("Filter by Strategic Tier", ["All"] + list(df["strategic_tier"].unique()))
    loc_filter = st.selectbox("Filter by Location", ["All"] + list(df["location_type"].unique()))

    filtered = df.copy()
    if tier_filter != "All":
        filtered = filtered[filtered["strategic_tier"] == tier_filter]
    if loc_filter != "All":
        filtered = filtered[filtered["location_type"] == loc_filter]

    st.markdown(f"**Showing {len(filtered)} restaurants**")

    display_cols = ["restaurant_name", "cuisine_type", "location_type", "growth_potential_score",
                    "strategic_tier", "cluster_label", "monthly_revenue", "customer_rating"]
    st.dataframe(
        filtered[display_cols].sort_values("growth_potential_score", ascending=False).reset_index(drop=True),
        use_container_width=True
    )

    csv = filtered[display_cols].to_csv(index=False)
    st.download_button("⬇️ Download Report as CSV", csv, "restaurant_report.csv", "text/csv")

st.divider()
st.markdown("<center style='color:gray; font-size:12px;'>Restaurant Growth Potential Modeling & Strategic Classification System | Data Science Project</center>", unsafe_allow_html=True)
