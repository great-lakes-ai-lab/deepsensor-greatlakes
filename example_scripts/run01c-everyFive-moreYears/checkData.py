import xarray as xr
import numpy as np
import pandas as pd
import os

# --- Assume glsea_raw is loaded and standardized as per your notebook setup ---
# For this diagnostic script to run, ensure glsea_raw is accessible.
# If running this standalone, you might need to add:
glsea_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA_combined.zarr'
glsea_raw = xr.open_zarr(glsea_path, chunks={'time': 366, 'lat': 200, 'lon': 200})
from deepsensor_greatlakes.utils import standardize_dates
glsea_raw = standardize_dates(glsea_raw)
# --- End of standalone setup example ---

# Use the data_range from your *new* strategy (1995-01-01 to 2022-12-31)
data_range = ("1995-01-01", "2022-12-31")

print(f"--- Checking for truly missing dates in GLSEA data from {data_range[0]} to {data_range[1]} ---")

# 1. Generate a complete, expected daily date range for the full data_range
start_date_expected = pd.to_datetime(data_range[0]).normalize()
end_date_expected = pd.to_datetime(data_range[1]).normalize()
all_expected_dates_pd = pd.date_range(start=start_date_expected, end=end_date_expected, freq='D')
all_expected_dates_dt64 = all_expected_dates_pd.values.astype('datetime64[D]')

print(f"Total expected unique days in the full data range: {len(all_expected_dates_dt64)}")

# 2. Get the actual dates present in glsea_raw
# Ensure they are also normalized to date-only, matching your preprocessing
actual_glsea_dates_dt64 = glsea_raw['time'].values.astype('datetime64[D]')

print(f"Total actual unique days found in glsea_raw dataset: {len(actual_glsea_dates_dt64)}")

# Convert to sets for efficient difference calculation
set_all_expected = set(all_expected_dates_dt64)
set_actual_glsea = set(actual_glsea_dates_dt64)

# 3. Find missing dates: dates that are expected but not in the actual data
missing_dates_set = set_all_expected - set_actual_glsea

# Convert the set of missing dates back to a sorted list for easy viewing
missing_dates_list = sorted(list(missing_dates_set))

if missing_dates_list:
    print(f"\n--- Result: Found {len(missing_dates_list)} truly MISSING dates in GLSEA data! ---")
    print("These dates are expected in the range but are NOT present in glsea_raw:")
    for i, date in enumerate(missing_dates_list):
        if i < 20: # Print only the first 20 to avoid spamming the console
            print(f"  - {date}")
        elif i == 20:
            print(f"  ... (and {len(missing_dates_list) - 20} more missing dates)")
            break
    print("\nThis confirms that the `KeyError` is indeed due to actual data gaps for these specific dates.")
    print("The pre-filtering strategy in `gen_tasks` (to only use dates present in `glsea_raw`) is necessary to handle these gaps gracefully.")
else:
    print("\n--- Result: No truly missing dates found in GLSEA data within the specified range. ---")
    print("If you are still getting a KeyError with `date_subsample_factor=5`, there might be a subtle precision issue")
    print("or an edge case with DeepSensor's `time_slice_variable` function where the `delta_t` causes a lookup for a date not in the index.")
    print("However, based on the previous error, missing dates are the most probable cause.")
