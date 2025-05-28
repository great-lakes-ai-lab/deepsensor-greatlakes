#!/usr/bin/env python
# coding: utf-8

# # Prediction with Trained ConvNP Model
# 
#  This notebook demonstrates how to load a pre-trained ConvNP model,
#  perform predictions on new data, and visualize the model's mean
#  and standard deviation predictions for Great Lakes SST.
# 
#  We will:
#  1. Load the pre-trained ConvNP model from disk.
#  2. Prepare a prediction task for a specific date, including context
#     (e.g., random sensor observations) and the full lake grid as target.
#  3. Use the model to generate mean and standard deviation predictions.
#  4. Apply the lake mask to the predictions for clear visualization.
#  5. Create and save high-quality plots of the masked predictions.
#  6. Perform a time series prediction for a specific point in Lake Superior.

# ## Step 1: Import Packages and Set Up Environment
# 
#  We import necessary libraries for data handling, plotting, and DeepSensor.
#  We also make sure that the GPU is set as the default device.

# In[ ]:


import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # For geographical plots
import cartopy.feature as cfeature
import os
import sys # For sys.stdout.flush()
from tqdm import tqdm # For progress bar in time series loop

import deepsensor.torch
from deepsensor.data import DataProcessor, TaskLoader, construct_circ_time_ds
from deepsensor.model import ConvNP
from deepsensor.train import set_gpu_default_device
from deepsensor.data import Task # Explicitly import Task for manual construction

# Local package utilities
from deepsensor_greatlakes.utils import standardize_dates, generate_random_coordinates, apply_mask_to_prediction

# --- IMPORTANT: Make sure that load_convnp_model is correctly imported.
# Assuming it's in deepsensor_greatlakes.model as previously.
from deepsensor_greatlakes.model import load_convnp_model

# --- Batch Mode: Define Output Directory ---
PREDICTION_OUTPUT_DIR = 'prediction_plots' # Specific output directory for prediction results
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
# --- End Batch Mode Config ---

set_gpu_default_device()

print("Environment setup complete.")
sys.stdout.flush()


# ## Step 2: User Inputs - Paths and Prediction Dates/Locations
# 
#  Define the paths to your raw data and the folder where your trained model is saved.
#  Also, specify the dates/locations for which you want to make predictions.

# In[ ]:


# Paths to the files on U-M HPC
bathymetry_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/bathymetry/interpolated_bathymetry.nc'
mask_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/masks/lakemask.nc'
glsea_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA_combined.zarr'

# Path to your trained model's folder
# --- IMPORTANT: This should point to the output of your training script, e.g., run00/ ---
model_folder_path = "." # Assuming model is saved here
deepsensor_config_path = "../../deepsensor_config/" # Path to DataProcessor config

# Date for a single full-map prediction
SINGLE_PREDICTION_DATE = "2022-07-15" # Example date from the val_range "2022-01-01" to "2022-12-31"

# Number of random context points to simulate sparse observations for all predictions
N_CONTEXT_POINTS = 100

# --- NEW: Time Series Prediction Configuration ---
TARGET_LON_TS = -87.5 # Approximate middle of Lake Superior
TARGET_LAT_TS = 47.5 # Approximate middle of Lake Superior
TS_START_DATE = "2022-01-01"
TS_END_DATE = "2022-12-31" # Using your validation range
# --- END NEW ---

print(f"Single prediction configured for date: {SINGLE_PREDICTION_DATE} with {N_CONTEXT_POINTS} context points.")
print(f"Time series prediction configured for Lat: {TARGET_LAT_TS}, Lon: {TARGET_LON_TS} from {TS_START_DATE} to {TS_END_DATE}.")
sys.stdout.flush()


# ## Step 3: Load Raw Data and Preprocess
# 
#  We load the raw GLSEA (SST) data, bathymetry, and the lake mask. The data processor
#  (whose configuration is loaded from `deepsensor_config_path`) will handle normalization.

# In[ ]:


# Open the Zarr store for GLSEA data
glsea_raw = xr.open_zarr(glsea_path, chunks={'time': 366, 'lat': 200, 'lon': 200})
glsea_raw = standardize_dates(glsea_raw)
if 'crs' in glsea_raw.variables: # Check if 'crs' variable exists before dropping
    glsea_raw = glsea_raw.drop_vars('crs')

print(f"Type of glsea_raw after loading and initial processing: {type(glsea_raw)}")
sys.stdout.flush()

# Open the NetCDF files for bathymetry and lake mask
bathymetry_raw = xr.open_dataset(bathymetry_path)
lakemask_raw = xr.open_dataset(mask_path)

# --- NEW: Create a set of available dates from glsea_raw for fast lookup ---
glsea_available_dates = set(glsea_raw['time'].values.astype('datetime64[D]'))
print(f"GLSEA dataset has {len(glsea_available_dates)} unique available dates within its loaded time range.")
sys.stdout.flush()
# --- END NEW ---

print("Raw data loaded and preprocessed.")
sys.stdout.flush()


# ## Step 4: Initialize DataProcessor and TaskLoader
# 
#  The `DataProcessor` is essential for normalizing data, and the `TaskLoader`
#  for creating prediction tasks consistent with how the model was trained.

# In[ ]:


data_processor = DataProcessor(deepsensor_config_path)
print("DataProcessor initialized:", data_processor)
sys.stdout.flush()

# Process auxiliary data (like bathymetry, and day-of-year features)
dates_full_range = pd.date_range(glsea_raw.time.values.min(), glsea_raw.time.values.max(), freq="D")
dates_full_range = pd.to_datetime(dates_full_range).normalize()

doy_ds = construct_circ_time_ds(dates_full_range, freq="D")

# These were context variables during training, so they need to be processed here
aux_ds_for_taskloader, lakemask_ds_for_taskloader = data_processor([bathymetry_raw, lakemask_raw], method="min_max")
aux_ds_for_taskloader["cos_D"] = standardize_dates(doy_ds["cos_D"])
aux_ds_for_taskloader["sin_D"] = standardize_dates(doy_ds["sin_D"])

# Process glsea data. This will be a DataArray or Dataset depending on glsea_raw.
glsea_processed = data_processor(glsea_raw)

# TaskLoader context sets must precisely match your training setup
task_loader = TaskLoader(context=[glsea_processed, aux_ds_for_taskloader, lakemask_ds_for_taskloader], target=glsea_processed)
print("TaskLoader initialized:", task_loader)
sys.stdout.flush()


# ## Step 5: Load Trained Model
# 
# Load the ConvNP model weights and configuration from your saved `model_folder_path`.

# In[ ]:


model = load_convnp_model(model_folder_path, data_processor, task_loader)
print("Model loaded successfully!")
sys.stdout.flush()


# ## Step 6: Single Full-Map Prediction
# 
#  A `Task` object is created for the `SINGLE_PREDICTION_DATE`.
#  We'll use `N_CONTEXT_POINTS` random locations within the lake as context points,
#  and the full `glsea_raw` grid as the target locations (`X_t`).

# In[ ]:


print(f"\n--- Starting Full-Map Prediction for {SINGLE_PREDICTION_DATE} ---")
sys.stdout.flush()

# Filter SINGLE_PREDICTION_DATE against glsea_available_dates
if np.datetime64(SINGLE_PREDICTION_DATE, 'D') not in glsea_available_dates:
    print(f"Warning: SINGLE_PREDICTION_DATE {SINGLE_PREDICTION_DATE} not found in GLSEA data. Skipping full-map prediction.")
    sys.stdout.flush()
else:
    # Generate random context points within the lake mask for the prediction date
    random_lake_points_for_prediction = generate_random_coordinates(lakemask_raw, N_CONTEXT_POINTS, data_processor)

    # Create the prediction task
    prediction_task = task_loader(
        SINGLE_PREDICTION_DATE,
        context_sampling=random_lake_points_for_prediction, # Use the N random points as context
        target_sampling="all" # Predict over the entire grid
    )
    prediction_task = prediction_task.remove_context_nans() # Ensure no NaNs in context

    print(f"Prediction task created for {SINGLE_PREDICTION_DATE}.")
    print(f"DEBUG: Type of prediction_task['X_c']: {type(prediction_task['X_c'])}")
    if isinstance(prediction_task['X_c'], list) and len(prediction_task['X_c']) > 0:
        print(f"DEBUG: Type of prediction_task['X_c'][0]: {type(prediction_task['X_c'][0])}")
        print(f"Number of context points: {len(prediction_task['X_c'][0])}")
    else:
        print("DEBUG: prediction_task['X_c'] is not a list or is empty for context points.")

    print(f"DEBUG: Type of prediction_task['X_t']: {type(prediction_task['X_t'])}")
    if isinstance(prediction_task['X_t'], list) and len(prediction_task['X_t']) > 0:
        print(f"DEBUG: Type of prediction_task['X_t'][0]: {type(prediction_task['X_t'][0])}")
        print(f"Number of target points: {len(prediction_task['X_t'][0])}")
    else:
        print("DEBUG: prediction_task['X_t'] is not a list or is empty for target points.")
    sys.stdout.flush()


    # Perform Prediction
    # X_t argument to model.predict should match the data structure used for training.
    # It will typically be the raw Xarray object that defines the target grid.
    prediction_ds = model.predict(prediction_task, X_t=glsea_raw)

    print("Full-map prediction completed.")
    print(prediction_ds)
    sys.stdout.flush()

    # Apply Lake Mask and Save Plots
    # Pass the entire Dataset for the 'sst' variable to the masking function.
    # The 'prediction_ds' returned by model.predict is a dict like {'sst': Dataset_with_mean_and_std}
    # The apply_mask_to_prediction function expects a Dataset as its first argument.
    masked_prediction_output_ds = apply_mask_to_prediction(
        prediction_ds['sst'], lakemask_raw
    )
    # Now extract the masked mean and std DataArrays from the returned Dataset.
    prediction_ds_masked_mean = masked_prediction_output_ds['mean']
    prediction_ds_masked_std = masked_prediction_output_ds['std']

    # Plotting the masked mean prediction
    plt.figure(figsize=(10, 8))
    ax_mean = plt.axes(projection=ccrs.PlateCarree())
    prediction_ds_masked_mean.plot(
        ax=ax_mean,
        cmap='viridis',
        cbar_kwargs={'label': 'Predicted Mean SST (°C)'}
    )
    ax_mean.add_feature(cfeature.COASTLINE)
    ax_mean.add_feature(cfeature.BORDERS, linestyle=':')
    ax_mean.add_feature(cfeature.LAKES, alpha=0.5)
    ax_mean.add_feature(cfeature.RIVERS)
    ax_mean.set_title(f'Masked Predicted Mean SST for {SINGLE_PREDICTION_DATE}')
    ax_mean.set_xlabel('Longitude')
    ax_mean.set_ylabel('Latitude')
    plt.tight_layout()
    mean_plot_filename = os.path.join(PREDICTION_OUTPUT_DIR, f'mean_sst_prediction_{SINGLE_PREDICTION_DATE}.png')
    plt.savefig(mean_plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {mean_plot_filename}")
    sys.stdout.flush()


    # Plotting the masked standard deviation prediction
    plt.figure(figsize=(10, 8))
    ax_std = plt.axes(projection=ccrs.PlateCarree())
    prediction_ds_masked_std.plot(
        ax=ax_std,
        cmap='plasma', # A good cmap for uncertainty
        cbar_kwargs={'label': 'Predicted Std SST (°C)'}
    )
    ax_std.add_feature(cfeature.COASTLINE)
    ax_std.add_feature(cfeature.BORDERS, linestyle=':')
    ax_std.add_feature(cfeature.LAKES, alpha=0.5)
    ax_std.add_feature(cfeature.RIVERS)
    ax_std.set_title(f'Masked Predicted Std SST for {SINGLE_PREDICTION_DATE}')
    ax_std.set_xlabel('Longitude')
    ax_std.set_ylabel('Latitude')
    plt.tight_layout()
    std_plot_filename = os.path.join(PREDICTION_OUTPUT_DIR, f'std_sst_prediction_{SINGLE_PREDICTION_DATE}.png')
    plt.savefig(std_plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {std_plot_filename}")
    sys.stdout.flush()

print("Full-map prediction section completed.")
sys.stdout.flush()


# ## Step 7: Time Series Prediction for a Single Point
# 
#  Predict mean and standard deviation over a time range for a specific geographical point.

# In[ ]:


print("\n--- Starting Time Series Prediction ---")
sys.stdout.flush()

ts_dates_full = pd.date_range(TS_START_DATE, TS_END_DATE).normalize()

# Filter ts_dates against glsea_available_dates
ts_dates_filtered = [d for d in ts_dates_full if np.datetime64(d, 'D') in glsea_available_dates]
print(f"Time series will be predicted for {len(ts_dates_filtered)} dates out of {len(ts_dates_full)} available in GLSEA.")
sys.stdout.flush()

ts_means = []
ts_stds = []
actual_ts_dates = [] # To store dates for which prediction was successful

# Define the single target point for the time series
# This creates a Dataset with the 'sst' variable containing a single NaN value
# at the target lat/lon, and a time coordinate that will be updated in the loop.
target_point_ds_template = xr.Dataset(
    {'sst': (['lat', 'lon'], [[np.nan]])},
    coords={'lat': [TARGET_LAT_TS], 'lon': [TARGET_LON_TS]}
)

for date_dt64 in tqdm(ts_dates_filtered, desc="Predicting Time Series"):
    try:
        # Create context: Use the predefined N_CONTEXT_POINTS, but sample Y_c from glsea_raw for this date
        # context_data_for_date is already a Task object from task_loader
        ts_task = task_loader(
            date_dt64,
            context_sampling=random_lake_points_for_prediction,
            target_sampling=None # Not getting target for context creation
        ).remove_context_nans()

        # Create target for this specific date and point
        current_target_point_ds = target_point_ds_template.copy()
        current_target_point_ds['time'] = np.datetime64(date_dt64, 'D')

        # Set the target locations (X_t) for this specific point and date on the existing Task object
        ts_task.X_t = current_target_point_ds # This sets the X_t *within* the Task object

        # Perform prediction. Explicitly pass X_t, as the model.predict method requires it.
        ts_prediction_ds = model.predict(ts_task, X_t=current_target_point_ds)

        # Extract values, unnormalize, and convert to scalar
        mean_val = data_processor.map_array(ts_prediction_ds['sst']['mean'], 'sst', unnorm=True).item()
        std_val = data_processor.map_array(ts_prediction_ds['sst']['std'], 'sst', unnorm=True).item()

        ts_means.append(mean_val)
        ts_stds.append(std_val)
        actual_ts_dates.append(date_dt64)
    except Exception as e:
        print(f"Warning: Could not predict for date {date_dt64} due to error: {e}. Skipping.")
        sys.stdout.flush()
        continue


if not ts_means:
    print("No data points successfully predicted for time series. Skipping plot.")
    sys.stdout.flush()
else:
    # Convert lists to pandas DataFrame for plotting
    ts_df = pd.DataFrame({
        'mean': ts_means,
        'std': ts_stds
    }, index=pd.to_datetime(actual_ts_dates))

    # Save time series data to CSV
    ts_csv_filename = os.path.join(PREDICTION_OUTPUT_DIR, f'time_series_sst_prediction_LSuperior_{TARGET_LAT_TS}_{TARGET_LON_TS}.csv')
    ts_df.to_csv(ts_csv_filename)
    print(f"Saved: {ts_csv_filename}")
    sys.stdout.flush()

    # Plotting the time series
    plt.figure(figsize=(12, 6))
    plt.plot(ts_df.index, ts_df['mean'], label='Predicted Mean SST', color='blue')
    plt.fill_between(ts_df.index,
                     ts_df['mean'] - ts_df['std'],
                     ts_df['mean'] + ts_df['std'],
                     color='lightblue', alpha=0.6, label='Mean +/- Std Dev')

    plt.title(f'SST Prediction Time Series at Lake Superior (Lat: {TARGET_LAT_TS}, Lon: {TARGET_LON_TS})')
    plt.xlabel('Date')
    plt.ylabel('SST (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ts_plot_filename = os.path.join(PREDICTION_OUTPUT_DIR, f'time_series_sst_prediction_LSuperior_{TARGET_LAT_TS}_{TARGET_LON_TS}.png')
    plt.savefig(ts_plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {ts_plot_filename}")
    sys.stdout.flush()

print("Time series prediction completed.")
sys.stdout.flush()


# ## Conclusion
# 
#  This script successfully loaded a trained DeepSensor model, performed
#  a full-map prediction for a specific date, and generated a time series
#  prediction for a single point in Lake Superior, including uncertainty.

# ## Reproducibility
# 
#  Below we list some aspects of the computing environment for better reproduciblity.

# In[ ]:


print("\n--- Reproducibility Information ---")
print("Python Executable:", sys.executable)
print("Python Version:", sys.version)
sys.stdout.flush()

