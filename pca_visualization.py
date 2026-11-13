
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load regional data
data = pd.read_csv(
    "outputs/regional_mobility.csv",
    index_col=0
)

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Use k=3 for this exploratory visualization
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

labels = model.fit_predict(X)

# Apply PCA
pca = PCA(n_components=2)
coordinates = pca.fit_transform(X)

# Create visualization dataframe
plot_data = pd.DataFrame({
    "Region": data.index,
    "PC1": coordinates[:, 0],
    "PC2": coordinates[:, 1],
    "Cluster": labels.astype(str)
})

# Interactive scatter plot
fig = px.scatter(
    plot_data,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_name="Region",
    title="Indian Regional Mobility Clusters",
    labels={
        "PC1": "Principal Component 1",
        "PC2": "Principal Component 2"
    },
    template="plotly_white"
)

fig.update_traces(marker=dict(size=12))

fig.update_layout(
    width=1000,
    height=650,
    title_x=0.5
)

fig.write_html("outputs/pca_visualization.html")

fig.show()

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print(
    "\nTotal variance explained:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

print("\nPCA visualization saved!")
