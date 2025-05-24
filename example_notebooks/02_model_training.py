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

# In[1]:


import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da
import gcsfs
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


# In[2]:


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

# In[3]:


# Training/data config (adapted for Great Lakes)
#data_range = ("2009-01-01", "2022-12-31")
#train_range = ("2009-01-01", "2021-12-31")
#val_range = ("2022-01-01", "2022-12-31")
#date_subsample_factor = 10

# Just two years for demo purposes
data_range = ("2009-01-01", "2010-12-31")
train_range = ("2009-01-01", "2009-12-31")
val_range = ("2010-01-01", "2010-12-31")
date_subsample_factor = 30


# In[4]:


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
deepsensor_folder = '../deepsensor_config/'


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

# In[5]:


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


# ## Step 4: Loading Static Datasets (Bathymetry and Lake Mask)
# 
# Next, we load two static datasets:
# - **Bathymetry**: The underwater features of the Great Lakes.
# - **Lake Mask**: A binary mask indicating water bodies within the lakes.
# 
# These datasets are loaded from NetCDF files and undergo basic preprocessing. 
# 

# In[6]:


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

# In[7]:


data_processor = DataProcessor(deepsensor_folder)
print(data_processor)


# In[8]:


glsea = data_processor(glsea_raw)

# process the bathymetry and lake
aux_ds, lakemask_ds = data_processor([bathymetry_raw, lakemask_raw], method="min_max")


# In[9]:


# Select a subset of the ice concentration data to compute normalization parameters
#_ = data_processor(ice_concentration_raw.sel(time=slice("2009-01-01", "2009-12-31")))

# Now apply the normalization parameters to the full ice concentration dataset
#ice_concentration = data_processor(ice_concentration_raw, method="min_max")


# In[10]:


data_processor.config


# In[11]:


dates = pd.date_range(glsea_raw.time.values.min(), glsea_raw.time.values.max(), freq="D")
dates = pd.to_datetime(dates).normalize()  # This will set all times to 00:00:00


# In[12]:


doy_ds = construct_circ_time_ds(dates, freq="D")
aux_ds["cos_D"] = standardize_dates(doy_ds["cos_D"])
aux_ds["sin_D"] = standardize_dates(doy_ds["sin_D"])
aux_ds


# ## Step 6: Generating Random Coordinates within the Lake Mask
# 
# We generate random coordinates within the **lake mask**. These coordinates represent sampling points inside the Great Lakes region. The **DataProcessor** is used to normalize these coordinates, ensuring that they are suitable for training the model.
# 
# We will generate `N` random coordinates and plot them to visualize their distribution within the lake.
# 

# In[13]:


# Example usage
N = 100  # Number of random points
random_lake_points = generate_random_coordinates(lakemask_raw, N, data_processor)


# In[14]:


import matplotlib.pyplot as plt

# Assuming random_coords is the (2, N) array from the previous step
latitudes = random_lake_points[0, :]
longitudes = random_lake_points[1, :]

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(longitudes, latitudes, color='blue', alpha=0.5, s=10)
plt.title("Scatter plot of N Random Coordinates within Lake Mask")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


# ## Step 7: Task Generation for Model Training
# 
# In this section, we use **TaskLoader** to generate tasks. A task consists of context data (input features like sea surface temperature, bathymetry, etc.) and target data (what we want the model to predict, such as ice concentration).
# 
# We generate tasks for training by sampling from the datasets. Each task represents a training example that the model will learn from.
# 

# In[15]:


task_loader = TaskLoader(context=[glsea, aux_ds, lakemask_ds], target=glsea)
task_loader
#task_loader = TaskLoader(context=glsea, target=glsea)


# In[16]:


from tqdm import tqdm

# Define how Tasks are generated
def gen_tasks(dates, progress=True):
    tasks = []
    for date in tqdm(dates, disable=not progress):
        # Create task with context and target sampling
        task = task_loader(date, context_sampling=random_lake_points, target_sampling="all")
        
        # Remove NaNs from the target data (Y_t) in the task 
        # Target data cannot have NaNs
        task = task.remove_target_nans()
        
        # Append the processed task to the list
        tasks.append(task)
        
    return tasks


# In[17]:


# Generate training and validation tasks
train_dates = pd.date_range(train_range[0], train_range[1])[::date_subsample_factor]
val_dates = pd.date_range(val_range[0], val_range[1])[::date_subsample_factor]

# Standardize the dates so they are datetime64[D] (date only, no time)
train_dates = pd.to_datetime(train_dates).normalize()  # This will set the time to 00:00:00
val_dates = pd.to_datetime(val_dates).normalize()      # This will set the time to 00:00:00

# Generate the tasks
train_tasks = gen_tasks(train_dates)
val_tasks = gen_tasks(val_dates)


# In[18]:


train_tasks[10]


# In[19]:


fig = deepsensor.plot.task(val_tasks[2], task_loader)
plt.show()


# ## Step 8: Model Setup and Training
# 
# We now set up the **ConvNP** model, a neural process-based model from **DeepSensor**. We use the **DataProcessor** and **TaskLoader** as inputs to the model, which allows the model to handle context and target data properly during training.
# 
# The model is then trained for a set number of epochs, and we monitor its performance by tracking the training loss and validation RMSE (Root Mean Squared Error).
# 
# At the end of the training loop, we save the best-performing model.
# 

# In[20]:


# Set up model
model = ConvNP(data_processor, task_loader)


# In[21]:


# Define the Trainer and training loop
trainer = Trainer(model, lr=5e-5)


# In[22]:


# Monitor validation performance
def compute_val_rmse(model, val_tasks):
    errors = []
    target_var_ID = task_loader.target_var_IDs[0][0]  # assuming 1st target set and 1D
    for task in val_tasks:
        mean = data_processor.map_array(model.mean(task), target_var_ID, unnorm=True)
        true = data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
        errors.extend(np.abs(mean - true))
    return np.sqrt(np.mean(np.concatenate(errors) ** 2))


# In[23]:


# Track the losses and validation RMSEs
losses = []
val_rmses = []
val_rmse_best = np.inf

# Start the training loop
for epoch in tqdm(range(50), desc="Training Epochs"):  # Training for 50 epochs
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
        save_model(model, deepsensor_folder)

# Plot training losses and validation RMSE
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses)
axes[0].set_xlabel('Epoch')
axes[0].set_title('Training Loss')

axes[1].plot(val_rmses)
axes[1].set_xlabel('Epoch')
axes[1].set_title('Validation RMSE')

plt.show()


# In[24]:


# To load it later:
# Assuming you have data_processor and task_loader instantiated in your notebook
loaded_model = load_convnp_model(deepsensor_folder, data_processor, task_loader)
print("Model loaded successfully with custom function!")


# ## Step 9: Prediction
# 
# Now that we have a trained model, we can use it to make a prediction. Notice that we get both a mean and standard deviation from this prediciton. 

# In[25]:


date = "2010-02-14"
test_task = task_loader(date, context_sampling=random_lake_points, target_sampling="all")
prediction_ds = loaded_model.predict(test_task, X_t=glsea_raw)
prediction_ds


# In[35]:


prediction_ds_masked = apply_mask_to_prediction(prediction_ds['sst'], lakemask_raw)
prediction_ds_masked


# Note that the prediction produces both a mean prediction and a standard deviation, which is a characteristic of a Gaussian Process approach. 

# In[39]:


plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
prediction_ds_masked['mean'].plot(cmap='viridis', cbar_kwargs={'label': 'Predicted Mean SST'})
plt.title(f'Masked Predicted Mean SST for Single Day')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.subplot(1, 2, 2) 
prediction_ds_masked['std'].plot(cmap='plasma', cbar_kwargs={'label': 'Predicted Std SST'})
plt.title(f'Masked Predicted Std SST for Single Day')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.tight_layout()
plt.show()


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
