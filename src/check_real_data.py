import pandas as pd

df = pd.read_csv("data/pune_daily_rainfall.csv")

print(df.info())
print(df.describe())

print("\nRainfall values:")
print(df["rainfall_mm"].value_counts().head(10))

print("\nMissing values:")
print(df.isna().sum())

print("\nDays with rainfall >= 64.5 mm:")
print((df["rainfall_mm"] >= 64.5).sum())

print("\nDays with rainfall > 0:")
print((df["rainfall_mm"] > 0).sum())