#!/usr/bin/env python
# coding: utf-8

# # Training the ConvNP Model
#
# In this notebook, we will preprocess Great Lakes data using an existing data processor, generate tasks for model training, and set up a training loop to train a **ConvNP** model using DeepSensor. We will:
# 1. Load and preprocess temporal and static datasets like **SST**, **Ice Concentration**, **Lake Mask**, and **Bathymetry**.
# 2. Load and use an existing **DataProcessor** to handle data normalization.
# 3. Generate tasks using **TaskLoader** and train the **ConvNP** model.
# 4. Monitor validation performance and track model training losses and RMSE (Root Mean Squared Error).
#
# Let's begin by importing necessary packages and defining helper functions.
#

# ## Step 1: Import Packages and Define Helper Functions
#
# We import the libraries required for:
# - Data manipulation and visualization (`xarray`, `pandas`, `matplotlib`).
# - Geospatial operations (`cartopy`).
# - Efficient computation with Dask (`dask`).
# - DeepSensor for data processing and model training (`deepsensor`).
#
# Additionally, we import local helper functions such as `standardize_dates`, which standardizes the 'time' dimension in the dataset to a date-only format (`datetime64[D]`). We also define `generate_random_coordinates` and custom save and load functions, as the default functions in DeepSensor appear to be broken in this environment.
#

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da
from tqdm import tqdm
import gcsfs
import sys
import os

import deepsensor.torch
from deepsensor.data import DataProcessor, TaskLoader, construct_circ_time_ds
from deepsensor.data.sources import get_era5_reanalysis_data, get_earthenv_auxiliary_data, \
    get_gldas_land_mask
from deepsensor.model import ConvNP
from deepsensor.train import Trainer, set_gpu_default_device

# Local package utilities
from deepsensor_greatlakes.utils import standardize_dates, generate_random_coordinates, apply_mask_to_prediction
from deepsensor_greatlakes.model import save_model, load_convnp_model

# --- Batch Mode: Define Plot Output Directory ---
PLOT_OUTPUT_DIR = 'model_training_plots'
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
# --- End Batch Mode Config ---


set_gpu_default_device()


# ## Step 2: Data Inventory and Preprocessing
#
# In this section, we load the required environmental datasets for model training:
# - **Ice Concentration**: A dataset of ice cover over time in the Great Lakes.
# - **GLSEA (Sea Surface Temperature)**: A dataset of sea surface temperature.
# - **Bathymetry**: A dataset representing the underwater topography of the lakes.
# - **Lake Mask**: A binary mask indicating water presence.
#
# These datasets are loaded from storage and preprocessed by converting time into date-only format and handling missing data.
#

# ### User Inputs - Select Training and Validation Ranges

# Training/data config (adapted for Great Lakes)
data_range = ("2009-01-01", "2022-12-31")
train_range = ("2009-01-01", "2021-12-31")
val_range = ("2022-01-01", "2022-12-31")
date_subsample_factor = 5

# Just two years for demo purposes
#data_range = ("2009-01-01", "2010-12-31")
#train_range = ("2009-01-01", "2009-12-31")
#val_range = ("2010-01-01", "2010-12-31")
#date_subsample_factor = 30


# Path to the Zarr stores (NOTE: This won't work on U-M HPC. Paths must be changed)
#bathymetry_path = 'gs://great-lakes-osd/context/interpolated_bathymetry.nc'
#mask_path = 'gs://great-lakes-osd/context/lakemask.nc'
#ice_concentration_path = 'gs://great-lakes-osd/ice_concentration.zarr'
#glsea_path = 'gs://great-lakes-osd/GLSEA_combined.zarr'
#glsea3_path = 'gs://great-lakes-osd/GLSEA3_combined.zarr'

# Path to the files on U-M HPC
bathymetry_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/bathymetry/interpolated_bathymetry.nc'
mask_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/masks/lakemask.nc'
ice_concentration_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/NSIDC/ice_concentration.zarr'
glsea_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA_combined.zarr'
glsea3_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA3_combined.zarr'

# Paths to saved configurations
deepsensor_folder = '../../deepsensor_config/'


# ## Step 3: Loading Temporal Datasets (Ice Concentration and GLSEA)
#
# In this section, we load the **Ice Concentration** and **GLSEA** datasets stored in Zarr format. These datasets contain critical temporal information on ice cover and sea surface temperature.
#
# We perform the following preprocessing:
# 1. Replace invalid land values (denoted by `-1`) with `NaN`.
# 2. Standardize the time dimension to date-only precision.
# 3. Drop unnecessary variables like **CRS**.
#
# Let’s load and preprocess the data now.
#

# Open the Zarr stores
ice_concentration_raw = xr.open_zarr(ice_concentration_path, chunks={'time': 366, 'lat': 200, 'lon': 200})
glsea_raw = xr.open_zarr(glsea_path, chunks={'time': 366, 'lat': 200, 'lon': 200})
glsea3_raw = xr.open_zarr(glsea3_path, chunks={'time': 366, 'lat': 200, 'lon': 200})

# Replace -1 (land value) with NaN
ice_concentration_raw = ice_concentration_raw.where(ice_concentration_raw != -1, float('nan'))

# Convert all times to date-only format, removing the time component
ice_concentration_raw = standardize_dates(ice_concentration_raw)
glsea_raw = standardize_dates(glsea_raw)
glsea3_raw = standardize_dates(glsea3_raw)

# Drop CRS - not needed
glsea_raw = glsea_raw.drop_vars('crs')
glsea3_raw = glsea3_raw.drop_vars('crs')

# New: Create a set of available dates from glsea_raw for fast lookup
glsea_available_dates = set(glsea_raw['time'].values.astype('datetime64[D]'))
print(f"GLSEA dataset has {len(glsea_available_dates)} unique available dates within its loaded time range.")

# ## Step 4: Loading Static Datasets (Bathymetry and Lake Mask)
#
# Next, we load two static datasets:
# - **Bathymetry**: The underwater features of the Great Lakes.
# - **Lake Mask**: A binary mask indicating water bodies within the lakes.
#
# These datasets are loaded from NetCDF files and undergo basic preprocessing.
#

# Open the NetCDF files using xarray
bathymetry_raw = xr.open_dataset(bathymetry_path)
lakemask_raw = xr.open_dataset(mask_path)

# Name the bathymetry variable (only needed if reading from GCP)
#bathymetry_raw = bathymetry_raw.rename({'__xarray_dataarray_variable__': 'bathymetry'})


# ## Step 5: Initialize the Data Processor
#
# The **DataProcessor** from DeepSensor is used to preprocess and normalize the datasets, getting them ready for model training. It applies scaling and transformation techniques to the datasets, such as **min-max scaling**.
#
# We initialize the **DataProcessor** and apply it to the datasets. Below we load the `data_processor` that we fit in the last notebook.
#

data_processor = DataProcessor(deepsensor_folder)
print(data_processor)


glsea = data_processor(glsea_raw)

# process the bathymetry and lake
aux_ds, lakemask_ds = data_processor([bathymetry_raw, lakemask_raw], method="min_max")


# Select a subset of the ice concentration data to compute normalization parameters
#_ = data_processor(ice_concentration_raw.sel(time=slice("2009-01-01", "2009-12-31")))

# Now apply the normalization parameters to the full ice concentration dataset
#ice_concentration = data_processor(ice_concentration_raw, method="min_max")


print(data_processor.config)


dates = pd.date_range(glsea_raw.time.values.min(), glsea_raw.time.values.max(), freq="D")
dates = pd.to_datetime(dates).normalize() # This will set all times to 00:00:00


doy_ds = construct_circ_time_ds(dates, freq="D")
aux_ds["cos_D"] = standardize_dates(doy_ds["cos_D"])
aux_ds["sin_D"] = standardize_dates(doy_ds["sin_D"])
print(aux_ds)


# ## Step 6: Generating Random Coordinates within the Lake Mask
#
# We generate random coordinates within the **lake mask**. These coordinates represent sampling points inside the Great Lakes region. The **DataProcessor** is used to normalize these coordinates, ensuring that they are suitable for training the model.
#
# We will generate `N` random coordinates and plot them to visualize their distribution within the lake.
#

# Example usage
N = 200 # Number of random points
random_lake_points = generate_random_coordinates(lakemask_raw, N, data_processor)


# --- Batch Mode: Save Scatter Plot ---
latitudes = random_lake_points[0, :]
longitudes = random_lake_points[1, :]

plt.figure(figsize=(8, 6))
plt.scatter(longitudes, latitudes, color='blue', alpha=0.5, s=10)
plt.title("Scatter plot of N Random Coordinates within Lake Mask")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
scatter_plot_filename = os.path.join(PLOT_OUTPUT_DIR, 'random_lake_points_scatter.png')
plt.savefig(scatter_plot_filename, dpi=300, bbox_inches='tight')
plt.close() # Close the figure
print(f"Saved: {scatter_plot_filename}")
# --- End Batch Mode ---


# ## Step 7: Task Generation for Model Training
#
# In this section, we use **TaskLoader** to generate tasks. A task consists of context data (input features like sea surface temperature, bathymetry, etc.) and target data (what we want the model to predict, such as ice concentration).
#
# We generate tasks for training by sampling from the datasets. Each task represents a training example that the model will learn from.
#

task_loader = TaskLoader(context=[glsea, aux_ds, lakemask_ds], target=glsea)
print(task_loader)
#task_loader = TaskLoader(context=glsea, target=glsea)


from tqdm import tqdm

from tqdm import tqdm

# Define how Tasks are generated
def gen_tasks(dates_to_generate, glsea_available_dates_set, progress=True):
    tasks = []
    actual_dates_used = [] # To keep track of dates that successfully form tasks

    for date_pd in tqdm(dates_to_generate, disable=not progress):
        # Ensure the date is in datetime64[D] format for consistent comparison
        date_dt64 = np.datetime64(date_pd, 'D')

        # --- Check if the date exists in the GLSEA data's time index ---
        if date_dt64 not in glsea_available_dates_set:
            # You can uncomment this print statement if you want to see which dates are skipped
            # print(f"Skipping date {date_dt64} as it is not found in the GLSEA dataset's time index.")
            continue # Skip this date if it's not present in the actual data

        try:
            # Create task with context and target sampling, use date_dt64
            task = task_loader(date_dt64, context_sampling=random_lake_points, target_sampling="all")

            # Remove NaNs from the target data (Y_t) in the task
            # Target data cannot have NaNs
            task = task.remove_target_nans()

            # Append the processed task to the list
            tasks.append(task)
            actual_dates_used.append(date_dt64) # Add to list only if task generation succeeds
        except Exception as e:
            # Catch any other potential errors during task generation for robustness
            print(f"Error processing date {date_dt64}: {e}. Skipping this date.")
        sys.stdout.flush() # Ensure print statements appear immediately, useful for HPC logs

    print(f"Finished generating {len(tasks)} tasks out of {len(dates_to_generate)} requested dates.")
    print(f"Actual dates successfully used in tasks: {len(actual_dates_used)}")
    return tasks

# Generate training and validation tasks
train_dates = pd.date_range(train_range[0], train_range[1])[::date_subsample_factor]
val_dates = pd.date_range(val_range[0], val_range[1])[::date_subsample_factor]

# Standardize the dates so they are datetime64[D] (date only, no time)
train_dates = pd.to_datetime(train_dates).normalize() # This will set the time to 00:00:00
val_dates = pd.to_datetime(val_dates).normalize()     # This will set the time to 00:00:00

# Generate the tasks
train_tasks = gen_tasks(train_dates, glsea_available_dates)
val_tasks = gen_tasks(val_dates, glsea_available_dates)

print(train_tasks[10])


# --- Batch Mode: Save Task Plot ---
fig = deepsensor.plot.task(val_tasks[2], task_loader)
task_plot_filename = os.path.join(PLOT_OUTPUT_DIR, 'example_validation_task_plot.png')
fig.savefig(task_plot_filename, dpi=300, bbox_inches='tight') # deepsensor.plot.task returns a matplotlib figure
plt.close(fig) # Close the specific figure object
print(f"Saved: {task_plot_filename}")
# --- End Batch Mode ---


# ## Step 8: Model Setup and Training
#
# We now set up the **ConvNP** model, a neural process-based model from **DeepSensor**. We use the **DataProcessor** and **TaskLoader** as inputs to the model, which allows the model to handle context and target data properly during training.
#
# The model is then trained for a set number of epochs, and we monitor its performance by tracking the training loss and validation RMSE (Root Mean Squared Error).
#
# At the end of the training loop, we save the best-performing model.
#

# Set up model
model = ConvNP(data_processor, task_loader)


# Define the Trainer and training loop
trainer = Trainer(model, lr=5e-5)


# Monitor validation performance
def compute_val_rmse(model, val_tasks):
    errors = []
    target_var_ID = task_loader.target_var_IDs[0][0] # assuming 1st target set and 1D
    for task in val_tasks:
        mean = data_processor.map_array(model.mean(task), target_var_ID, unnorm=True)
        true = data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
        errors.extend(np.abs(mean - true))
    return np.sqrt(np.mean(np.concatenate(errors) ** 2))


# Track the losses and validation RMSEs
losses = []
val_rmses = []
val_rmse_best = np.inf

# Start the training loop
for epoch in tqdm(range(50), desc="Training Epochs"): # Training for 50 epochs
    # Generate tasks for training
    batch_losses = trainer(train_tasks)
    losses.append(np.mean(batch_losses))

    # Compute the validation RMSE
    val_rmse = compute_val_rmse(model, val_tasks)
    val_rmses.append(val_rmse)

    # Save the model if it performs better
    if val_rmse < val_rmse_best:
        val_rmse_best = val_rmse
        #model.save(deepsensor_folder)
        save_model(model, '.')

# --- Batch Mode: Save Training Progress Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses)
axes[0].set_xlabel('Epoch')
axes[0].set_title('Training Loss')

axes[1].plot(val_rmses)
axes[1].set_xlabel('Epoch')
axes[1].set_title('Validation RMSE')

training_progress_filename = os.path.join(PLOT_OUTPUT_DIR, 'training_progress.png')
plt.savefig(training_progress_filename, dpi=300, bbox_inches='tight')
plt.close(fig) # Close the specific figure object
print(f"Saved: {training_progress_filename}")
# --- End Batch Mode ---


# To load it later:
# Assuming you have data_processor and task_loader instantiated in your notebook
loaded_model = load_convnp_model(deepsensor_folder, data_processor, task_loader)
print("Model loaded successfully with custom function!")


# ## Step 9: Prediction
#
# Now that we have a trained model, we can use it to make a prediction. Notice that we get both a mean and standard deviation from this prediciton.

date = "2010-02-14"
test_task = task_loader(date, context_sampling=random_lake_points, target_sampling="all")
prediction_ds = loaded_model.predict(test_task, X_t=glsea_raw)
print(prediction_ds)


prediction_ds_masked = apply_mask_to_prediction(prediction_ds['sst'], lakemask_raw)
print(prediction_ds_masked)


# Note that the prediction produces both a mean prediction and a standard deviation, which is a characteristic of a Gaussian Process approach.

# --- Batch Mode: Save Masked Prediction Map Plots ---
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
prediction_ds_masked['mean'].plot(cmap='viridis', cbar_kwargs={'label': 'Predicted Mean SST'})
plt.title(f'Masked Predicted Mean SST for {str(prediction_ds["time"].values[0])[:10]}') # Use actual time from DS
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.subplot(1, 2, 2)
prediction_ds_masked['std'].plot(cmap='plasma', cbar_kwargs={'label': 'Predicted Std SST'})
plt.title(f'Masked Predicted Std SST for {str(prediction_ds["time"].values[0])[:10]}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.tight_layout()
prediction_map_filename = os.path.join(PLOT_OUTPUT_DIR, f'masked_prediction_maps_{str(prediction_ds["time"].values[0])[:10]}.png')
plt.savefig(prediction_map_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {prediction_map_filename}")
# --- End Batch Mode ---


# The above plot looks really bizarre because it has only been trained on two years of data! DeepSensor's models are data hungry...

# # Conclusion
#
# In this notebook, we:
# 1. Loaded and preprocessed several Great Lakes datasets for training a **ConvNP** model.
# 2. Generated tasks using **TaskLoader** and visualized data to perform sanity checks.
# 3. Trained the **ConvNP** model and monitored its performance.
#
# Next, we will explore the active learning component of **DeepSensor**.
#
