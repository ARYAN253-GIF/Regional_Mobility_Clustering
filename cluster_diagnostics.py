
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

# Load regional mobility data
data = pd.read_csv(
    "outputs/regional_mobility.csv",
    index_col=0
)

# Test both scaling methods
scalers = {
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler()
}

all_results = []

for scaler_name, scaler in scalers.items():

    X = scaler.fit_transform(data)

    print("\n================================")
    print("SCALER:", scaler_name)
    print("================================")

    for k in [2, 3, 4, 5, 6]:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X)

        sizes = pd.Series(labels).value_counts().sort_index()

        silhouette = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)

        print(f"\nK = {k}")
        print("Cluster sizes:", sizes.to_dict())
        print("Silhouette:", round(silhouette, 4))
        print("Davies-Bouldin:", round(db, 4))

        # Show the regions in each cluster
        for cluster in range(k):

            regions = data.index[labels == cluster].tolist()

            print(f"Cluster {cluster}: {regions}")

        all_results.append({
            "Scaler": scaler_name,
            "K": k,
            "Silhouette": silhouette,
            "Davies_Bouldin": db,
            "Smallest_Cluster": sizes.min(),
            "Largest_Cluster": sizes.max()
        })

# Save all results
results = pd.DataFrame(all_results)

results.to_csv(
    "outputs/cluster_diagnostics.csv",
    index=False
)

print("\n================================")
print("DIAGNOSTIC ANALYSIS COMPLETED")
print("================================")

print(results.round(4).to_string(index=False))

