
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    davies_bouldin_score
)

# =====================================
# 1. LOAD DATA
# =====================================

data = pd.read_csv(
    "outputs/regional_mobility.csv",
    index_col=0
)

X = StandardScaler().fit_transform(data)

# =====================================
# 2. K-MEDIANS IMPLEMENTATION
# =====================================

def kmedians(X, k, random_state=42, max_iter=200):

    rng = np.random.default_rng(random_state)

    # Select initial observations as centers
    indices = rng.choice(
        len(X),
        size=k,
        replace=False
    )

    centers = X[indices].copy()

    for iteration in range(max_iter):

        # Manhattan distances
        distances = np.abs(
            X[:, None, :] - centers[None, :, :]
        ).sum(axis=2)

        labels = distances.argmin(axis=1)

        new_centers = centers.copy()

        for cluster in range(k):

            members = X[labels == cluster]

            if len(members) > 0:
                new_centers[cluster] = np.median(
                    members,
                    axis=0
                )

            else:
                # Reinitialize empty cluster
                new_centers[cluster] = X[
                    rng.integers(len(X))
                ]

        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    # Final assignments
    distances = np.abs(
        X[:, None, :] - centers[None, :, :]
    ).sum(axis=2)

    labels = distances.argmin(axis=1)

    objective = distances[
        np.arange(len(X)), labels
    ].sum()

    return labels, centers, objective


# =====================================
# 3. COMPARE BOTH METHODS
# =====================================

results = []
saved_labels = {}

for k in [2, 3, 4, 5, 6]:

    # Euclidean K-Means
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    euclidean_labels = km.fit_predict(X)

    # Manhattan K-Medians
    manhattan_labels, centers, objective = kmedians(
        X, k, random_state=42
    )

    # Silhouette using each method's distance
    euclidean_sil = silhouette_score(
        X,
        euclidean_labels,
        metric="euclidean"
    )

    manhattan_sil = silhouette_score(
        X,
        manhattan_labels,
        metric="manhattan"
    )

    # Davies-Bouldin is Euclidean-based.
    # Report it as a supplementary common metric.
    euclidean_db = davies_bouldin_score(
        X, euclidean_labels
    )

    manhattan_db = davies_bouldin_score(
        X, manhattan_labels
    )

    results.append({
        "K": k,
        "Euclidean_WCSS": km.inertia_,
        "Manhattan_Objective": objective,
        "Euclidean_Silhouette": euclidean_sil,
        "Manhattan_Silhouette": manhattan_sil,
        "Euclidean_DB": euclidean_db,
        "Manhattan_DB_Supplementary": manhattan_db,
        "ARI_Agreement": adjusted_rand_score(
            euclidean_labels,
            manhattan_labels
        )
    })

    saved_labels[k] = {
        "Euclidean": euclidean_labels,
        "Manhattan": manhattan_labels
    }

# =====================================
# 4. DISPLAY RESULTS
# =====================================

comparison = pd.DataFrame(results)

print("\n========== EUCLIDEAN VS MANHATTAN ==========")

print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    "outputs/manhattan_comparison.csv",
    index=False
)

# =====================================
# 5. SAVE CLUSTER ASSIGNMENTS
# =====================================

assignment = pd.DataFrame(
    index=data.index
)

for k in [2, 3, 4, 5, 6]:

    assignment[f"K{k}_Euclidean"] = (
        saved_labels[k]["Euclidean"]
    )

    assignment[f"K{k}_Manhattan"] = (
        saved_labels[k]["Manhattan"]
    )

assignment.to_csv(
    "outputs/distance_assignments.csv"
)

print("\nComparison completed successfully!")
print("Saved Manhattan comparison results.")
