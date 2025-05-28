#!/usr/bin/env python

# # Active Learning with ConvNP Model
#
# In this notebook, we implement **active learning** to improve model training by selecting the most informative data points for the model to focus on. Specifically, we:
# 1. Preprocess the dataset and define helper functions for data normalization.
# 2. Use a **Greedy Algorithm** for active learning to iteratively sample the most uncertain data points.
# 3. Implement **Standard Deviation Acquisition Function** to guide the model to focus on uncertain regions in the data.
# 4. Visualize the placement of new data points selected by active learning.
#
# Let’s begin by importing necessary packages and defining helper functions.
#

# ## Step 1: Import Packages
#
# We import the libraries required for:
# - Data manipulation and visualization (`xarray`, `pandas`, `matplotlib`).
# - Geospatial operations (`cartopy`).
# - Active learning (`deepsensor`).
# - PyTorch for model training and inference.
#
# We also import local helper functions like `standardize_dates` for normalizing the 'time' dimension to date-only precision and `generate_random_coordinates` to sample random coordinates from the lake mask.
#

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da
import os
import sys # Added for sys.stdout.flush()

import json
import torch

import deepsensor.torch
from deepsensor.data import DataProcessor, TaskLoader, construct_circ_time_ds
from deepsensor.model.nps import construct_neural_process
from deepsensor.model import ConvNP
from deepsensor.train import Trainer, set_gpu_default_device
from deepsensor.active_learning import GreedyAlgorithm
from deepsensor.active_learning.acquisition_fns import Stddev # Added explicitly as used

# Local package utilities
from deepsensor_greatlakes.utils import standardize_dates, generate_random_coordinates
from deepsensor_greatlakes.model import save_model, load_convnp_model

# --- Batch Mode: Define Output Directory ---
AL_OUTPUT_DIR = 'active_learning_output' # Specific output directory for AL results
os.makedirs(AL_OUTPUT_DIR, exist_ok=True)
# --- End Batch Mode Config ---

set_gpu_default_device()


# ## Step 2: Data Inventory and Preprocessing
#
# In this section, we load and preprocess the required datasets for active learning:
# - **Ice Concentration**: A dataset containing information about ice cover over time in the Great Lakes.
# - **GLSEA (Sea Surface Temperature)**: A dataset representing sea surface temperature over time.
# - **Bathymetry**: A dataset of underwater topography.
# - **Lake Mask**: A binary mask indicating water presence.
#
# These datasets are loaded from U-M HPC storage and undergo preprocessing, such as replacing invalid land values with `NaN` and standardizing the time dimension.
#

# Training/data config (adapted for Great Lakes)
# Note: This notebook uses date_subsample_factor = 10 for validation dates,
# which is used for generating `placement_tasks`.
data_range = ("2009-01-01", "2022-12-31")
train_range = ("2009-01-01", "2021-12-31")
val_range = ("2022-01-01", "2022-12-31")
# --- MODIFIED LINE: Changed to 5 for consistency ---
date_subsample_factor = 5
extent = "great_lakes"

# --- IMPORTANT: Ensure this path is correct for where your trained model is saved ---
deepsensor_folder = "../../deepsensor_config/" # This was for the previous training notebook


# Path to the files on U-M HPC
bathymetry_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/bathymetry/interpolated_bathymetry.nc'
mask_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/masks/lakemask.nc'
ice_concentration_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/NSIDC/ice_concentration.zarr'
glsea_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA_combined.zarr'
glsea3_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA3_combined.zarr'


# ## Step 3: Loading Temporal Datasets (Ice Concentration and GLSEA)
#
# In this section, we load the **Ice Concentration** and **GLSEA** datasets stored in Zarr format. These datasets contain temporal data on ice cover and sea surface temperature.
#
# Preprocessing steps include:
# 1. Replacing invalid land values (denoted by `-1`) with `NaN`.
# 2. Converting the time dimension to date-only format.
# 3. Dropping unnecessary variables, such as **CRS**.
#
# We proceed by loading and preprocessing these datasets.
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

# --- NEW: Create a set of available dates from glsea_raw for fast lookup ---
glsea_available_dates = set(glsea_raw['time'].values.astype('datetime64[D]'))
print(f"GLSEA dataset has {len(glsea_available_dates)} unique available dates within its loaded time range.")
sys.stdout.flush()
# --- END NEW ---


# ## Step 4: Loading Static Datasets (Lake Mask and Bathymetry)
#
# We now load two static datasets:
# - **Bathymetry**: The underwater features of the Great Lakes.
# - **Lake Mask**: A binary mask that identifies water areas.
#
# These datasets are opened from NetCDF files and processed similarly to the temporal datasets.
#

# Open the NetCDF files using xarray with gcsfs
bathymetry_raw = xr.open_dataset(bathymetry_path)
lakemask_raw = xr.open_dataset(mask_path)


# ## Step 5: Initialize the Data Processor
#
# The **DataProcessor** from DeepSensor is used to preprocess and normalize the data. It standardizes and normalizes the datasets for model training, ensuring consistency in feature scaling across training examples.
#
# We initialize the **DataProcessor** and apply it to the datasets.


data_processor = DataProcessor("../../deepsensor_config/") # This is the path to the config, not the model weights
print(data_processor)
sys.stdout.flush()


glsea = data_processor(glsea_raw)

# process the bathymetry and lake
aux_ds, lakemask = data_processor([bathymetry_raw, lakemask_raw], method="min_max")


dates = pd.date_range(glsea_raw.time.values.min(), glsea_raw.time.values.max(), freq="D")
dates = pd.to_datetime(dates).normalize() # This will set all times to 00:00:00


# Generate training and validation dates (only used for creating `placement_tasks` in this script)
train_dates = pd.date_range(train_range[0], train_range[1])[::date_subsample_factor]
val_dates = pd.date_range(val_range[0], val_range[1])[::date_subsample_factor]

# Standardize the dates so they are datetime64[D] (date only, no time)
train_dates = pd.to_datetime(train_dates).normalize() # This will set the time to 00:00:00
val_dates = pd.to_datetime(val_dates).normalize()     # This will set the time to 00:00:00


doy_ds = construct_circ_time_ds(dates, freq="D")
aux_ds["cos_D"] = standardize_dates(doy_ds["cos_D"])
aux_ds["sin_D"] = standardize_dates(doy_ds["sin_D"])
print(aux_ds)
sys.stdout.flush()


# ## Step 6: Generate Random Coordinates within the Lake Mask
#
# We generate random coordinates from within the **lake mask**. These coordinates represent sampling points in the Great Lakes region. These points are normalized using the **DataProcessor** and will later be used in active learning.
#
# We generate `N` random points, and plot them to visualize the sampling distribution inside the lake.
#

# Example usage
N = 100 # Number of random points
random_lake_points = generate_random_coordinates(lakemask_raw, N, data_processor)


task_loader = TaskLoader(context=[glsea, aux_ds, lakemask], target=glsea)


# ## Step 7: Custom Save and Load Functions
#
# Since the built-in save and load functions are broken, we define custom save and load functions to handle the **ConvNP** model and its configuration. The functions will:
# 1. **Save the model**: Save both the model weights and configuration to disk.
# 2. **Load the model**: Reload the model from disk, including weights and configuration, to resume training or inference.
#
# These functions are defined near the top of this notebook.
#

# --- IMPORTANT: PWD must contain the saved model from the training run ---
loaded_model = load_convnp_model('.', data_processor, task_loader)
print("Model loaded successfully!")
sys.stdout.flush()


print(loaded_model.config)
sys.stdout.flush()


# Load model (assign the loaded model to `model` variable for consistency)
model = loaded_model


# ## Step 8: Active Learning with Greedy Algorithm
#
# Active learning helps the model focus on the most informative data points by selecting points where it is most uncertain. We use the **Greedy Algorithm** from DeepSensor to select the most uncertain points based on the current model's predictions.
#
# In this section, we set up the **GreedyAlgorithm** with:
# - **Context and target data**: GLSEA data for the context and the same GLSEA for the target.
# - **Mask data**: To restrict the selection to valid areas within the lake.
#
# We run the active learning process, selecting new data points for training based on model uncertainty.
#

alg = GreedyAlgorithm(
    model,
    X_s=glsea_raw,
    X_t=glsea_raw,
    X_s_mask=lakemask_raw['mask'],
    X_t_mask=lakemask_raw['mask'],
    context_set_idx=0,
    target_set_idx=0,
    N_new_context=3,
    progress_bar=True, # tqdm progress bar will still print to stdout
)


# ## Step 9: Standard Deviation Acquisition Function
#
# The **Standard Deviation (Stddev)** acquisition function is used to guide the model towards uncertain regions. The idea is to select data points where the model has high variability in its predictions, which often indicates areas where the model is unsure.
#
# We use the **Stddev** acquisition function to select uncertain data points during the active learning process.
#

acquisition_fn = Stddev(model, context_set_idx=0, target_set_idx=0)


# --- MODIFIED: Filtering `val_dates` for `placement_tasks` ---
# Filter `val_dates` to only include dates present in `glsea_raw`
filtered_placement_dates = [d for d in val_dates if np.datetime64(d, 'D') in glsea_available_dates]
print(f"Generating placement tasks for {len(filtered_placement_dates)} dates (out of {len(val_dates)} initial).")
sys.stdout.flush()

placement_tasks = task_loader(filtered_placement_dates, context_sampling=[random_lake_points, "all", "all"], seed_override=0)
# --- END MODIFICATION ---

# Perform active learning query
X_new_df, acquisition_fn_ds = alg(acquisition_fn, placement_tasks)


print("X_new_df (new context points selected):")
print(X_new_df)
sys.stdout.flush()

print("\nAcquisition function output (uncertainty map):")
print(acquisition_fn_ds)
sys.stdout.flush()

# --- NEW: Save active learning results ---
# Save X_new_df (the selected context points)
x_new_df_filename = os.path.join(AL_OUTPUT_DIR, 'active_learning_new_points.csv')
X_new_df.to_csv(x_new_df_filename)
print(f"Saved new context points to: {x_new_df_filename}")
sys.stdout.flush()

# Save acquisition_fn_ds (the uncertainty map)
acquisition_ds_filename = os.path.join(AL_OUTPUT_DIR, 'active_learning_uncertainty_map.nc')
acquisition_fn_ds.to_netcdf(acquisition_ds_filename)
print(f"Saved uncertainty map to: {acquisition_ds_filename}")
sys.stdout.flush()
# --- END NEW ---


# ## Step 10: Visualizing Active Learning Placements
#
# After performing active learning, we visualize the new data points selected by the algorithm. These points are likely to be in areas where the model is most uncertain.
#
# We plot the selected points on a map to visualize where the model has focused its attention.
#

# --- Batch Mode: Save Placements Plot ---
fig = deepsensor.plot.placements(placement_tasks[0], X_new_df, data_processor,
                                 crs=ccrs.PlateCarree())
placements_plot_filename = os.path.join(AL_OUTPUT_DIR, 'active_learning_placements.png')
fig.savefig(placements_plot_filename, dpi=300, bbox_inches='tight')
plt.close(fig) # Close the specific figure object
print(f"Saved: {placements_plot_filename}")
sys.stdout.flush()
# --- End Batch Mode ---


# Although this plot isn't the easiest to read, it seems that the GreedyAlgorithm is recommending that we focus on Western Lake Erie, which is good to see, as this is exactly where the current network does have a high density of instruments. So to first order, the existing observing network isn't too bad.

# # Conclusion
#
# In this notebook, we implemented active learning to improve model training by focusing on the most informative data points. We:
# 1. Loaded and preprocessed several datasets for training.
# 2. Used the **GreedyAlgorithm** to select uncertain data points for model training.
# 3. Visualized the placement of these points on the map.
#
# Active learning helps ensure that the model is trained on the most challenging examples, which can improve its performance.
#

# # Reproducibility
#
# Below we list some aspects of the computing environment for better reproduciblity.

import sys
print("Python Executable:", sys.executable)
print("Python Version:", sys.version)
sys.stdout.flush()
# !pip freeze > requirements.txt # Commented out, not for batch execution
# print("requirements.txt generated!")
