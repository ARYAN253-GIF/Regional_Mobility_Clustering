
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score
)

# Load original regional data
data = pd.read_csv(
    "outputs/regional_mobility.csv",
    index_col=0
)

print("Original regional dataset:", data.shape)

# Create both scaled datasets
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

X_standard = standard_scaler.fit_transform(data)
X_robust = robust_scaler.fit_transform(data)

results = []

print("\n========== SCALING COMPARISON ==========")

for k in [2, 3, 4, 5, 6]:

    # StandardScaler model
    standard_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    standard_labels = standard_model.fit_predict(X_standard)

    # RobustScaler model
    robust_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    robust_labels = robust_model.fit_predict(X_robust)

    results.append({
        "K": k,
        "Standard_Silhouette":
            silhouette_score(X_standard, standard_labels),
        "Robust_Silhouette":
            silhouette_score(X_robust, robust_labels),
        "Standard_DB":
            davies_bouldin_score(X_standard, standard_labels),
        "Robust_DB":
            davies_bouldin_score(X_robust, robust_labels),
        "Agreement_ARI":
            adjusted_rand_score(
                standard_labels, robust_labels
            )
    })

    print(f"\nCompleted k = {k}")

comparison = pd.DataFrame(results)

print("\n========== FINAL COMPARISON ==========")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    "outputs/scaling_comparison.csv",
    index=False
)

# Examine cluster assignments using RobustScaler
print("\n========== ROBUST SCALER CLUSTERING ==========")

robust_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

robust_labels = robust_model.fit_predict(X_robust)

regional_results = data.copy()
regional_results["Cluster"] = robust_labels

print("\nCluster distribution:")
print(regional_results["Cluster"].value_counts().sort_index())

print("\nJammu and Kashmir assignment:")
print(regional_results.loc[
    "Jammu and Kashmir", "Cluster"
])

regional_results.to_csv(
    "outputs/robust_cluster_results.csv"
)

print("\nScaling comparison completed!")
