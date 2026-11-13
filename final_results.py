
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

Path("outputs").mkdir(exist_ok=True)

# =====================================
# 1. LOAD REGIONAL DATA
# =====================================

data = pd.read_csv(
    "outputs/regional_mobility.csv",
    index_col=0
)

# =====================================
# 2. FULL DATASET: 36 REGIONS, K=3
# =====================================

scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(data)

full_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=50
)

full_labels = full_model.fit_predict(X_full)

full_results = data.copy()
full_results["Cluster"] = full_labels

full_results.to_csv(
    "outputs/final_full_dataset_clusters.csv"
)

full_profiles = full_results.groupby("Cluster").mean()

full_profiles.to_csv(
    "outputs/final_full_cluster_profiles.csv"
)

# =====================================
# 3. SENSITIVITY: 35 REGIONS, K=2
# =====================================

sensitivity_data = data.drop(
    index="Jammu and Kashmir"
)

scaler_sensitivity = StandardScaler()

X_sensitivity = scaler_sensitivity.fit_transform(
    sensitivity_data
)

sensitivity_model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=50
)

sensitivity_labels = sensitivity_model.fit_predict(
    X_sensitivity
)

sensitivity_results = sensitivity_data.copy()

sensitivity_results["Cluster"] = sensitivity_labels

sensitivity_results.to_csv(
    "outputs/final_sensitivity_clusters.csv"
)

sensitivity_profiles = (
    sensitivity_results.groupby("Cluster").mean()
)

sensitivity_profiles.to_csv(
    "outputs/final_sensitivity_profiles.csv"
)

# =====================================
# 4. CREATE PCA COORDINATES
# =====================================

pca = PCA(n_components=2)

coordinates = pca.fit_transform(X_full)

pca_results = pd.DataFrame({
    "Region": data.index,
    "PC1": coordinates[:, 0],
    "PC2": coordinates[:, 1],
    "Cluster": full_labels.astype(str)
})

pca_results.to_csv(
    "outputs/final_pca_coordinates.csv",
    index=False
)

# =====================================
# 5. SAVE FINAL METRICS
# =====================================

metrics = pd.DataFrame([
    {
        "Experiment": "All 36 regions",
        "Method": "StandardScaler + K-Means",
        "K": 3,
        "Regions": 36,
        "Silhouette": silhouette_score(
            X_full, full_labels
        ),
        "Davies_Bouldin": davies_bouldin_score(
            X_full, full_labels
        ),
        "WCSS": full_model.inertia_
    },
    {
        "Experiment": "35 regions sensitivity",
        "Method": "StandardScaler + K-Means",
        "K": 2,
        "Regions": 35,
        "Silhouette": silhouette_score(
            X_sensitivity, sensitivity_labels
        ),
        "Davies_Bouldin": davies_bouldin_score(
            X_sensitivity, sensitivity_labels
        ),
        "WCSS": sensitivity_model.inertia_
    }
])

metrics.to_csv(
    "outputs/final_selected_metrics.csv",
    index=False
)

# =====================================
# 6. DISPLAY FINAL RESULTS
# =====================================

print("\n========== FINAL MODEL RESULTS ==========")

print(metrics.round(4).to_string(index=False))

print("\nFull dataset cluster sizes:")
print(full_results["Cluster"].value_counts().sort_index())

print("\nSensitivity cluster sizes:")
print(sensitivity_results["Cluster"].value_counts().sort_index())

print("\nPCA explained variance:")
print(
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

print("\nAll final results saved successfully!")

print("\nGenerated files:")
print("1. final_full_dataset_clusters.csv")
print("2. final_full_cluster_profiles.csv")
print("3. final_sensitivity_clusters.csv")
print("4. final_sensitivity_profiles.csv")
print("5. final_pca_coordinates.csv")
print("6. final_selected_metrics.csv")
