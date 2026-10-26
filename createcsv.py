import numpy as np
import pandas as pd

np.random.seed(42)

n = 100

df = pd.DataFrame({
    "date": pd.date_range("2024-06-01", periods=n),
    "rainfall_mm": np.random.gamma(2, 10, n),
    "temp_c": np.random.normal(27, 3, n),
    "humidity": np.random.normal(80, 10, n),
    "pressure": np.random.normal(1005, 5, n),
    "wind_speed": np.random.normal(5, 2, n),
    "cape": np.random.normal(1200, 500, n),
})

# Generate next-day rainfall approximately for testing
df["rainfall_next_24h"] = (
    df["rainfall_mm"] * 0.5
    + (df["humidity"] - 70) * 1.2
    + df["cape"] * 0.015
    + np.random.normal(0, 15, n)
).clip(lower=0)

df.to_csv("data/sample_weather.csv", index=False)