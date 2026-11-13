
import pandas as pd

# Load clustering results
clusters = pd.read_csv(
    "outputs/clustered_regions.csv",
    index_col=0
)

# Find unusual region in Cluster 1
print("\n========== UNUSUAL REGION ==========")

outlier = clusters[clusters["Cluster"] == 1]

print(outlier.to_string())

# Load original Google dataset
df = pd.read_csv(
    "data/mobility.csv",
    low_memory=False
)

df["date"] = pd.to_datetime(df["date"])

features = [
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline"
]

# Investigate each unusual region
for region in outlier.index:

    print("\n====================================")
    print("INVESTIGATING:", region)
    print("====================================")

    raw = df[
        (df["sub_region_1"] == region) &
        (df["sub_region_2"].isna()) &
        (df["date"].between("2022-01-01", "2022-10-15"))
    ].copy()

    print("\nNumber of original records:", len(raw))

    print("\nMobility statistics:")
    print(raw[features].describe().round(2))

    print("\nHighest mobility observations:")
    print(
        raw.nlargest(
            10,
            "retail_and_recreation_percent_change_from_baseline"
        )[["date"] + features].to_string(index=False)
    )

    print("\nAverage mobility:")
    print(raw[features].mean().round(2))

    print("\nMedian mobility:")
    print(raw[features].median().round(2))
