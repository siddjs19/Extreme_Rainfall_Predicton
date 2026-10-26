import xarray as xr
import pandas as pd
import glob
import os

# Find rainfall files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

files = sorted(glob.glob(os.path.join(DATA_DIR, "RF25_ind*_rfp25.nc")))

print("Files found:")
for file in files:
    print(os.path.basename(file))

if not files:
    raise FileNotFoundError(f"No NetCDF files found in {DATA_DIR}")

# Open all files
datasets = [xr.open_dataset(file) for file in files]

# Inspect dimensions of the first file
print("\nOriginal dimensions:")
print(datasets[0].dims)

# Combine along TIME
combined_ds = xr.concat(
    datasets,
    dim="TIME",
    join="override"
)

print("\nCombined dimensions:")
print(combined_ds.dims)

# Find rainfall variable
print("\nVariables:")
print(list(combined_ds.data_vars))

# Replace this if your rainfall variable has a different name
rainfall_var = list(combined_ds.data_vars)[0]

# Calculate spatial average across the India grid
rainfall_daily = combined_ds[rainfall_var].mean(
    dim=["LATITUDE", "LONGITUDE"],
    skipna=True
)

# Convert ONLY the reduced daily data to DataFrame
df = rainfall_daily.to_dataframe(
    name="rainfall_mm"
).reset_index()

# Standardize date column
df["TIME"] = pd.to_datetime(df["TIME"])

df = df.rename(columns={"TIME": "time"})

# Sort and remove duplicates
df = df.sort_values("time")
df = df.drop_duplicates(subset=["time"])
df = df.reset_index(drop=True)

print("\nFinal dataset:")
print(df.head())
print("\nShape:", df.shape)
print("Start:", df["time"].min())
print("End:", df["time"].max())

# Save combined processed dataset
output_path = os.path.join(DATA_DIR, "imd_rainfall_2021_2025.csv")
df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")