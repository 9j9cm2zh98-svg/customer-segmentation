"""
app.py — Customer Segmentation Streamlit Dashboard
====================================================
Run this AFTER train.py has been executed:
    streamlit run app.py

What this dashboard shows:
  - Sidebar filter to select which clusters to display
  - KPI cards: total customers, average income, average spending score
  - A scatter plot of Annual Income vs Spending Score coloured by cluster
  - A bar chart showing how many customers are in each cluster
  - A filterable data table
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide",
)

# ─────────────────────────────────────────────
# Guard: make sure train.py has been run first
# ─────────────────────────────────────────────
CLUSTERED_DATA_PATH = "data/customers_clustered.csv"

if not os.path.exists(CLUSTERED_DATA_PATH):
    st.error(
        "⚠️ Clustered data not found.\n\n"
        "Please run the training script first:\n\n"
        "```\npython train.py\n```"
    )
    st.stop()

# ─────────────────────────────────────────────
# Load clustered data
# ─────────────────────────────────────────────
df = pd.read_csv(CLUSTERED_DATA_PATH)

# Consistent colour palette — same order as train.py
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

# ─────────────────────────────────────────────
# Sidebar — cluster filter
# ─────────────────────────────────────────────
st.sidebar.title("🔎 Filters")
st.sidebar.markdown("Select one or more clusters to explore.")

all_clusters = sorted(df["Cluster"].unique())
selected_clusters = st.sidebar.multiselect(
    label="Cluster",
    options=all_clusters,
    default=all_clusters,
    format_func=lambda x: f"Cluster {x}",
)

# Fall back to all clusters if user deselects everything
if not selected_clusters:
    selected_clusters = all_clusters

filtered_df = df[df["Cluster"].isin(selected_clusters)]

# ─────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────
st.title("👥 Customer Segmentation Dashboard")
st.markdown(
    "K-Means clustering (K=5) applied to **Annual Income** and **Spending Score** "
    "from the Mall Customer Dataset. Use the sidebar to filter by cluster."
)
st.divider()

# ─────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────
st.subheader("📊 Summary")
st.caption("Metrics are calculated for the currently selected clusters.")

col1, col2, col3 = st.columns(3)

col1.metric(
    label="Total Customers",
    value=f"{len(filtered_df):,}",
)
col2.metric(
    label="Avg Annual Income",
    value=f"${filtered_df['Annual Income (k$)'].mean():.1f}k",
)
col3.metric(
    label="Avg Spending Score",
    value=f"{filtered_df['Spending Score (1-100)'].mean():.1f} / 100",
)

st.divider()

# ─────────────────────────────────────────────
# Charts — side by side
# ─────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

# ── Scatter plot ──
with chart_col1:
    st.subheader("🗺️ Cluster Scatter Plot")
    st.caption(
        "Each dot is one customer. The X-axis is Annual Income, the Y-axis is "
        "Spending Score. Dots with the same colour belong to the same cluster. "
        "Black ✕ marks show the centre of each cluster (centroid)."
    )

    fig1, ax1 = plt.subplots(figsize=(6, 5))

    for cluster_id in selected_clusters:
        mask = filtered_df["Cluster"] == cluster_id
        ax1.scatter(
            filtered_df.loc[mask, "Annual Income (k$)"],
            filtered_df.loc[mask, "Spending Score (1-100)"],
            c=COLORS[cluster_id],
            label=f"Cluster {cluster_id}",
            s=60,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

    # Overlay centroids for selected clusters
    for cluster_id in selected_clusters:
        cluster_data = df[df["Cluster"] == cluster_id]
        cx = cluster_data["Annual Income (k$)"].mean()
        cy = cluster_data["Spending Score (1-100)"].mean()
        ax1.scatter(cx, cy, c="black", marker="X", s=180, zorder=5)

    ax1.set_xlabel("Annual Income (k$)", fontsize=11)
    ax1.set_ylabel("Spending Score (1-100)", fontsize=11)
    ax1.set_title("Annual Income vs Spending Score", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# ── Bar chart ──
with chart_col2:
    st.subheader("📦 Customers per Cluster")
    st.caption(
        "This bar chart shows how many customers fall into each cluster. "
        "A taller bar means more customers were assigned to that group."
    )

    cluster_counts = (
        filtered_df["Cluster"]
        .value_counts()
        .reindex(selected_clusters, fill_value=0)
        .sort_index()
    )

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    bar_colors = [COLORS[c] for c in cluster_counts.index]

    bars = ax2.bar(
        [f"Cluster {c}" for c in cluster_counts.index],
        cluster_counts.values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
    )

    # Label each bar with its count
    for bar, count in zip(bars, cluster_counts.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_xlabel("Cluster", fontsize=11)
    ax2.set_ylabel("Number of Customers", fontsize=11)
    ax2.set_title("Customer Count per Cluster", fontsize=12)
    ax2.set_ylim(0, cluster_counts.max() * 1.15 if len(cluster_counts) > 0 else 10)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

# ─────────────────────────────────────────────
# Data table
# ─────────────────────────────────────────────
st.subheader("📋 Dataset Preview")
st.caption(
    "The table below shows the raw customer records for the selected clusters. "
    "The **Cluster** column was added by K-Means — it was not in the original data."
)

display_cols = ["CustomerID", "Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"]
st.dataframe(
    filtered_df[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=350,
)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Dataset: Mall Customer Segmentation Data (Kaggle) · "
    "Algorithm: K-Means (K=5) · Features: Annual Income, Spending Score"
)
