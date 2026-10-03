"""
train.py — Customer Segmentation ML Pipeline
=============================================
This script runs the full K-Means clustering pipeline step by step.

Run this script FIRST before launching the Streamlit dashboard:
    python train.py

What this script does:
  1. Loads the raw customer data from data/customers.csv
  2. Cleans the data (checks for missing values)
  3. Selects two features: Annual Income and Spending Score
  4. Scales the features so K-Means works correctly
  5. Uses the Elbow Method to confirm K=5 is the best number of clusters
  6. Trains the final K-Means model with K=5
  7. Saves the clustered data, trained model, scaler, and charts to disk
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Step 1: Create output directories if needed
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)

# ─────────────────────────────────────────────
# Step 2: Load the dataset
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 1 — Loading data")
print("=" * 50)

df = pd.read_csv("data/customers.csv")

print(f"Dataset shape : {df.shape}  ({df.shape[0]} rows, {df.shape[1]} columns)")
print("\nFirst 5 rows:")
print(df.head())

# ─────────────────────────────────────────────
# Step 3: Clean the data
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2 — Cleaning data")
print("=" * 50)

print(f"Null values before cleaning:\n{df.isnull().sum()}")
df = df.dropna()
print(f"\nNull values after cleaning:\n{df.isnull().sum()}")
print(f"\nRows remaining after cleaning: {len(df)}")

# ─────────────────────────────────────────────
# Step 4: Select features
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3 — Selecting features")
print("=" * 50)

# We use only these two features for the classic 2D K-Means view.
# Age is kept in the dataframe but NOT used for clustering.
FEATURE_COLS = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[FEATURE_COLS].values

print(f"Features selected: {FEATURE_COLS}")
print(f"Feature matrix shape: {X.shape}")

# ─────────────────────────────────────────────
# Step 5: Scale the features
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4 — Scaling features")
print("=" * 50)

# K-Means uses Euclidean distance. Without scaling, Annual Income (range 15–137)
# would dominate Spending Score (range 1–100) purely because of its larger numbers.
# StandardScaler transforms each feature to have mean=0 and std=1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features scaled with StandardScaler (mean=0, std=1).")
print(f"Scaled feature sample (first 3 rows):\n{X_scaled[:3]}")

# ─────────────────────────────────────────────
# Step 6: Elbow Method — find the best K
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5 — Elbow Method (K=1 to K=10)")
print("=" * 50)

inertia_values = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X_scaled)
    inertia_values.append(km.inertia_)

print(f"\n{'K':>4} | {'Inertia':>12}")
print("-" * 20)
for k, inertia in zip(K_range, inertia_values):
    marker = " ← elbow" if k == 5 else ""
    print(f"{k:>4} | {inertia:>12.2f}{marker}")

# Save the Elbow curve chart
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia_values, marker="o", color="#3b82d4", linewidth=2, markersize=8)
plt.axvline(x=5, color="#e74c3c", linestyle="--", linewidth=1.5, label="Chosen K=5")
plt.title("Elbow Method — Choosing the Best Number of Clusters", fontsize=14, pad=15)
plt.xlabel("Number of Clusters (K)", fontsize=12)
plt.ylabel("Inertia (lower = tighter clusters)", fontsize=12)
plt.xticks(K_range)
plt.legend()
plt.tight_layout()
plt.savefig("charts/elbow_curve.png", dpi=150)
plt.close()
print("\nSaved → charts/elbow_curve.png")

# ─────────────────────────────────────────────
# Step 7: Train the final K-Means model (K=5)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6 — Training K-Means with K=5")
print("=" * 50)

K = 5
kmeans = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
kmeans.fit(X_scaled)

# Assign cluster labels back to the original dataframe
df["Cluster"] = kmeans.labels_

print(f"K-Means trained with K={K}.")
print(f"\nCluster sizes:")
cluster_counts = df["Cluster"].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"  Cluster {cluster_id}: {count} customers")

# ─────────────────────────────────────────────
# Step 8: Save clustered data and model
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 7 — Saving outputs")
print("=" * 50)

# Save the labelled dataframe
df.to_csv("data/customers_clustered.csv", index=False)
print("Saved → data/customers_clustered.csv")

# Save the trained model
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)
print("Saved → models/kmeans_model.pkl")

# Save the scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved → models/scaler.pkl")

# ─────────────────────────────────────────────
# Step 9: Save cluster scatter plot
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 8 — Saving cluster scatter plot")
print("=" * 50)

# Five distinct colours — one per cluster
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

fig, ax = plt.subplots(figsize=(9, 6))

for cluster_id in range(K):
    mask = df["Cluster"] == cluster_id
    ax.scatter(
        df.loc[mask, "Annual Income (k$)"],
        df.loc[mask, "Spending Score (1-100)"],
        c=COLORS[cluster_id],
        label=f"Cluster {cluster_id}",
        s=60,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )

# Plot centroids in the original (unscaled) space
# Inverse-transform the scaled centroids back to original units
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(
    centroids_original[:, 0],
    centroids_original[:, 1],
    c="black",
    marker="X",
    s=200,
    zorder=5,
    label="Centroids",
)

ax.set_title("Customer Segments — K-Means Clustering (K=5)", fontsize=14, pad=15)
ax.set_xlabel("Annual Income (k$)", fontsize=12)
ax.set_ylabel("Spending Score (1-100)", fontsize=12)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("charts/cluster_scatter.png", dpi=150)
plt.close()
print("Saved → charts/cluster_scatter.png")

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("ALL DONE ✓")
print("=" * 50)
print("Next step: run the dashboard with:")
print("  streamlit run app.py")
