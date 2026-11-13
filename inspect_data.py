import pandas as pd
import os

print("Current folder:", os.getcwd())

print("Checking CSV file...")

df = pd.read_csv("data/mobility.csv", low_memory=False)

print("\nDataset loaded successfully!")

print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())