
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Load regional data
data = pd.read_csv(
    "outputs/regional_mobility.csv",
    index_col=0
)

X = StandardScaler().fit_transform(data)


# K-Medians algorithm
def kmedians(X, k, seed, max_iter=300):

    rng = np.random.default_rng(seed)

    centers = X[
        rng.choice(len(X), k, replace=False)
    ].copy()

    for _ in range(max_iter):

        distances = np.abs(
            X[:, None, :] - centers[None, :, :]
        ).sum(axis=2)

        labels = distances.argmin(axis=1)

        new_centers = centers.copy()

        for cluster in range(k):

            members = X[labels == cluster]

            if len(members) > 0:
                new_centers[cluster] = np.median(
                    members, axis=0
                )

        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    distances = np.abs(
        X[:, None, :] - centers[None, :, :]
    ).sum(axis=2)

    labels = distances.argmin(axis=1)

    return labels


# Test multiple initializations
for k in [3, 5, 6]:

    print("\n========== K =", k, "==========")

    all_labels = []

    for seed in range(10):

        labels = kmedians(X, k, seed)

        all_labels.append(labels)

        sizes = pd.Series(labels).value_counts().sort_index()

        silhouette = silhouette_score(
            X, labels, metric="manhattan"
        )

        print(
            f"Seed {seed}: "
            f"Silhouette={silhouette:.4f}, "
            f"Sizes={sizes.to_dict()}"
        )

    ari_values = []

    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):

            ari_values.append(
                adjusted_rand_score(
                    all_labels[i],
                    all_labels[j]
                )
            )

    print(
        "Average pairwise ARI:",
        round(np.mean(ari_values), 4)
    )

print("\nStability investigation completed!")
