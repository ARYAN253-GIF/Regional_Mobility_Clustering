
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

# Load data
data = pd.read_csv(
    "outputs/regional_mobility.csv",
    index_col=0
)

X = StandardScaler().fit_transform(data)

results = []
assignments = {}

# Test k from 2 to 6
for k in [2, 3, 4, 5, 6]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=50
    )

    labels = model.fit_predict(X)

    sizes = pd.Series(labels).value_counts()

    results.append({
        "K": k,
        "WCSS": model.inertia_,
        "Silhouette": silhouette_score(X, labels),
        "Davies_Bouldin": davies_bouldin_score(X, labels),
        "Smallest_Cluster": sizes.min(),
        "Largest_Cluster": sizes.max(),
        "Singleton_Clusters": (sizes == 1).sum()
    })

    assignments[k] = labels

    print(f"\nK = {k}")
    print("Cluster sizes:", sizes.to_dict())

# Save comparison
results_df = pd.DataFrame(results)

print("\n========== FINAL MODEL COMPARISON ==========")

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "outputs/final_model_comparison.csv",
    index=False
)

# Save all cluster assignments
assignment_df = pd.DataFrame(
    index=data.index
)

for k, labels in assignments.items():
    assignment_df[f"K{k}"] = labels

assignment_df.to_csv(
    "outputs/all_model_assignments.csv"
)

print("\nFinal model comparison saved!")
