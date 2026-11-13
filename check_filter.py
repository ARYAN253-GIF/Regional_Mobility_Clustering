
import pandas as pd

df = pd.read_csv("data/mobility.csv", low_memory=False)

print("Original rows:", len(df))

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print("\nValid dates:", df["date"].notna().sum())
    print("Minimum date:", df["date"].min())
    print("Maximum date:", df["date"].max())

if "country_region_code" in df.columns:
    print("\nRows with IN country code:",
          (df["country_region_code"].astype(str).str.strip() == "IN").sum())

if "country_region" in df.columns:
    print("\nCountry names:")
    print(df["country_region"].value_counts().head(10))

if "sub_region_1" in df.columns:
    print("\nRows with state information:",
          df["sub_region_1"].notna().sum())

if "sub_region_2" in df.columns:
    print("\nRows with empty sub-region 2:",
          df["sub_region_2"].isna().sum())

if "date" in df.columns:
    period = df["date"].between("2020-03-01", "2020-06-30")
    print("\nRows within selected date period:", period.sum())

print("\n========== CHECK COMPLETED ==========")
