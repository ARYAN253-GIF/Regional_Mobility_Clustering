
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# =====================================
# 1. CREATE OUTPUT FOLDER
# =====================================

Path("outputs").mkdir(exist_ok=True)

# =====================================
# 2. LOAD DATASET
# =====================================

print("Loading Google Mobility Dataset...")

df = pd.read_csv("data/mobility.csv", low_memory=False)

print("Original dataset shape:", df.shape)

# =====================================
# 3. SELECT MOBILITY FEATURES
# =====================================

features = [
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline"
]

# Check required columns
missing_columns = [col for col in features if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# =====================================
# 4. CONVERT DATE FORMAT
# =====================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# =====================================
# 5. FILTER INDIA STATE-LEVEL DATA
# =====================================

df = df[
    (df["country_region_code"].astype(str).str.strip() == "IN") &
    (df["sub_region_1"].notna()) &
    (df["sub_region_2"].isna()) &
    (df["date"].between("2022-01-01", "2022-10-15"))
].copy()

print("\nRows after filtering:", len(df))

if df.empty:
    raise ValueError("No records found after filtering.")

# =====================================
# 6. REMOVE DUPLICATES
# =====================================

df = df.drop_duplicates(
    subset=["sub_region_1", "date"]
)

# =====================================
# 7. CONVERT FEATURES TO NUMERIC
# =====================================

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =====================================
# 8. CREATE REGIONAL AVERAGES
# =====================================

regional = df.groupby("sub_region_1")[features].mean()

# Remove regions with no mobility information
regional = regional.dropna(how="all")

if len(regional) < 3:
    raise ValueError(
        "Not enough regions for clustering. Check your filters."
    )

print("\nNumber of regions:", len(regional))

# =====================================
# 9. HANDLE MISSING VALUES
# =====================================

print("\nMissing values before treatment:")
print(regional.isnull().sum())

# Remove features that have no available values
regional = regional.dropna(axis=1, how="all")

if regional.shape[1] < 2:
    raise ValueError("Not enough usable mobility features.")

# Fill remaining missing values using feature medians
regional = regional.fillna(regional.median())

# Verify missing values
print("\nMissing values after treatment:")
print(regional.isnull().sum())

# =====================================
# 10. SCALE NUMERICAL FEATURES
# =====================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(regional)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=regional.columns,
    index=regional.index
)

# =====================================
# 11. SAVE PROCESSED DATA
# =====================================

regional.to_csv("outputs/regional_mobility.csv")

scaled_df.to_csv("outputs/scaled_mobility.csv")

# Save preprocessing summary
summary = pd.DataFrame({
    "Total_Regions": [len(regional)],
    "Total_Features": [len(regional.columns)],
    "Start_Date": ["2022-01-01"],
    "End_Date": ["2022-10-15"]
})

summary.to_csv("outputs/preprocessing_summary.csv", index=False)

# =====================================
# 12. DISPLAY FINAL RESULTS
# =====================================

print("\n================================")
print("PREPROCESSING COMPLETED!")
print("================================")

print("\nFinal regional dataset shape:")
print(regional.shape)

print("\nSelected features:")
print(regional.columns.tolist())

print("\nFirst five regional records:")
print(regional.head())

print("\nScaled dataset:")
print(scaled_df.head())

print("\nFiles saved successfully:")
print("1. outputs/regional_mobility.csv")
print("2. outputs/scaled_mobility.csv")
print("3. outputs/preprocessing_summary.csv")
