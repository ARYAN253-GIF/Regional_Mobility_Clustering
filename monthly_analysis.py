
import pandas as pd

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

df = df[
    (df["country_region_code"] == "IN") &
    (df["sub_region_1"] == "Jammu and Kashmir") &
    (df["sub_region_2"].isna()) &
    (df["date"].between("2022-01-01", "2022-10-15"))
].copy()

df["Month"] = df["date"].dt.to_period("M").astype(str)

monthly = df.groupby("Month")[features].agg(
    ["mean", "median", "count"]
)

print("\nMONTHLY MOBILITY ANALYSIS")
print(monthly.round(2).to_string())

monthly.to_csv("outputs/jammu_monthly_analysis.csv")

print("\nMonthly analysis saved successfully!")
