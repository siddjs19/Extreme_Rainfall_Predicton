import xarray as xr
import pandas as pd
import numpy as np

# Load IMD dataset
ds = xr.open_dataset("data/RF25_ind2025_rfp25.nc")

# Pune coordinates
PUNE_LAT = 18.52
PUNE_LON = 73.85

# Select nearest IMD grid point
rainfall = ds["RAINFALL"].sel(
    LATITUDE=PUNE_LAT,
    LONGITUDE=PUNE_LON,
    method="nearest"
)

# Convert to DataFrame
df = rainfall.to_dataframe(name="rainfall_mm").reset_index()

# Rename date column
df = df.rename(columns={"TIME": "date"})

# Remove missing values and invalid negative rainfall
df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0)
df = df.dropna(subset=["rainfall_mm"])

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

print(df.head())
print(df.tail())
print("\nRows:", len(df))
print("Nearest latitude:", rainfall.LATITUDE.item())
print("Nearest longitude:", rainfall.LONGITUDE.item())
print("Total rainfall:", df["rainfall_mm"].sum())

# Save extracted location data
df.to_csv("data/pune_daily_rainfall.csv", index=False)

print("\nSaved: data/pune_daily_rainfall.csv")