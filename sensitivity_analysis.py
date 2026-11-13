
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score
)

# =====================================
# 1. LOAD DATA
# =====================================

data = pd.read_csv(
    "outputs/regional_mobility.csv",
    index_col=0
)

# Two experimental datasets
datasets = {
    "All_36_Regions": data,
    "Without_Jammu_Kashmir": data.drop(
        index="Jammu and Kashmir"
    )
}

all_results = []
all_assignments = []

# =====================================
# 2. TEST BOTH DATASETS
# =====================================

for dataset_name, subset in datasets.items():

    print("\n===================================")
    print(dataset_name)
    print("===================================")

    scaler = StandardScaler()
    X = scaler.fit_transform(subset)

    for k in [2, 3, 4, 5, 6]:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=50
        )

        labels = model.fit_predict(X)

        sizes = pd.Series(labels).value_counts()

        silhouette = silhouette_score(X, labels)

        db = davies_bouldin_score(X, labels)

        # Test stability using five random seeds
        stability = []

        for seed in [1, 7, 21, 42, 99]:

            test_model = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=50
            )

            test_labels = test_model.fit_predict(X)

            ari = adjusted_rand_score(
                labels,
                test_labels
            )

            stability.append(ari)

        average_stability = np.mean(stability)

        print(f"\nK = {k}")
        print("Cluster sizes:", sizes.to_dict())
        print("WCSS:", round(model.inertia_, 4))
        print("Silhouette:", round(silhouette, 4))
        print("Davies-Bouldin:", round(db, 4))
        print("Average stability:", round(average_stability, 4))

        all_results.append({
            "Dataset": dataset_name,
            "K": k,
            "Regions": len(subset),
            "WCSS": model.inertia_,
            "Silhouette": silhouette,
            "Davies_Bouldin": db,
            "Smallest_Cluster": sizes.min(),
            "Largest_Cluster": sizes.max(),
            "Singleton_Clusters": (sizes == 1).sum(),
            "Average_Stability": average_stability
        })

        # Save regional assignments
        for region, cluster in zip(subset.index, labels):

            all_assignments.append({
                "Dataset": dataset_name,
                "K": k,
                "Region": region,
                "Cluster": cluster
            })

# =====================================
# 3. SAVE EXPERIMENT RESULTS
# =====================================

results = pd.DataFrame(all_results)

assignments = pd.DataFrame(all_assignments)

results.to_csv(
    "outputs/sensitivity_comparison.csv",
    index=False
)

assignments.to_csv(
    "outputs/sensitivity_assignments.csv",
    index=False
)

print("\n===================================")
print("FINAL SENSITIVITY COMPARISON")
print("===================================")

print(results.round(4).to_string(index=False))

print("\nFiles saved:")
print("1. sensitivity_comparison.csv")
print("2. sensitivity_assignments.csv")

print("\nSensitivity analysis completed!")
