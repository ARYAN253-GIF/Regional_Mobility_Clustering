
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score
)

# =====================================
# 1. LOAD PREPROCESSED DATA
# =====================================

print("Loading scaled mobility data...")

X = pd.read_csv(
    "outputs/scaled_mobility.csv",
    index_col=0
)

print("Dataset shape:", X.shape)

if X.isna().any().any():
    raise ValueError("Missing values found in scaled dataset.")

# =====================================
# 2. TEST DIFFERENT CLUSTER COUNTS
# =====================================

results = []
models = {}

print("\nTraining K-Means models...")

for k in [2, 3, 4, 5, 6]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X)

    # Calculate evaluation metrics
    wcss = model.inertia_

    silhouette = silhouette_score(X, labels)

    db_index = davies_bouldin_score(X, labels)

    results.append({
        "Clusters": k,
        "WCSS": wcss,
        "Silhouette": silhouette,
        "Davies_Bouldin": db_index
    })

    models[k] = model

    print(f"\nModel completed for k = {k}")

# =====================================
# 3. DISPLAY MODEL COMPARISON
# =====================================

scores = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(scores.round(4).to_string(index=False))

# =====================================
# 4. SELECT PROVISIONAL MODEL
# =====================================

best_k = int(
    scores.loc[scores["Silhouette"].idxmax(), "Clusters"]
)

best_model = models[best_k]

print("\nProvisional best cluster count:", best_k)

# =====================================
# 5. SAVE BEST MODEL CLUSTER LABELS
# =====================================

labels = best_model.labels_

regional = pd.read_csv(
    "outputs/regional_mobility.csv",
    index_col=0
)

regional["Cluster"] = labels

regional.to_csv("outputs/clustered_regions.csv")

# =====================================
# 6. DISPLAY CLUSTER DISTRIBUTION
# =====================================

print("\n========== CLUSTER DISTRIBUTION ==========")

print(regional["Cluster"].value_counts().sort_index())

# =====================================
# 7. CREATE CLUSTER PROFILES
# =====================================

profiles = regional.groupby("Cluster").mean()

print("\n========== CLUSTER PROFILES ==========")

print(profiles.round(2).to_string())

profiles.to_csv("outputs/cluster_profiles.csv")

# =====================================
# 8. TEST MODEL STABILITY
# =====================================

print("\n========== STABILITY TEST ==========")

stability_scores = []

for seed in [1, 7, 21, 42, 99]:

    model = KMeans(
        n_clusters=best_k,
        random_state=seed,
        n_init=20
    )

    new_labels = model.fit_predict(X)

    ari = adjusted_rand_score(
        labels,
        new_labels
    )

    stability_scores.append(ari)

    print(f"Seed {seed}: ARI = {ari:.4f}")

print(
    "\nAverage stability:",
    round(np.mean(stability_scores), 4)
)

# =====================================
# 9. SAVE MODEL EVALUATION
# =====================================

scores.to_csv(
    "outputs/model_scores.csv",
    index=False
)

pd.DataFrame({
    "Random_Seed": [1, 7, 21, 42, 99],
    "ARI": stability_scores
}).to_csv(
    "outputs/stability_scores.csv",
    index=False
)

print("\n====================================")
print("K-MEANS CLUSTERING COMPLETED!")
print("====================================")

print("\nFiles generated:")
print("1. model_scores.csv")
print("2. clustered_regions.csv")
print("3. cluster_profiles.csv")
print("4. stability_scores.csv")
