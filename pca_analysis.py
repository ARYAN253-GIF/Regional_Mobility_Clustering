
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load regional mobility data
data = pd.read_csv(
    "outputs/regional_mobility.csv",
    index_col=0
)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
coordinates = pca.fit_transform(X)

# Feature contributions
loadings = pd.DataFrame(
    pca.components_.T,
    index=data.columns,
    columns=["PC1", "PC2"]
)

print("\n========== PCA VARIANCE ==========")

print("PC1:", round(pca.explained_variance_ratio_[0] * 100, 2), "%")
print("PC2:", round(pca.explained_variance_ratio_[1] * 100, 2), "%")

print(
    "Total:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

print("\n========== PCA FEATURE LOADINGS ==========")
print(loadings.round(4))

# Save loadings
loadings.to_csv("outputs/pca_loadings.csv")

# Save PCA coordinates
pca_results = pd.DataFrame(
    coordinates,
    columns=["PC1", "PC2"],
    index=data.index
)

pca_results.to_csv("outputs/pca_coordinates.csv")

print("\nPCA analysis completed!")
