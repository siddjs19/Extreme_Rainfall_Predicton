import xarray as xr

file_path = "data/RF25_ind2025_rfp25.nc"

ds = xr.open_dataset(file_path)

print(ds)
print(ds.data_vars)
print(ds.coords)