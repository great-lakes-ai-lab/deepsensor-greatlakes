"""
Module for loading Great Lakes datasets from GCP in formats compatible with DeepSensor.

Functions:
    load_metadata(metadata_path): Load dataset metadata from a JSON file.
    load_dataset(dataset_name, metadata, year=None): Load a dataset from GCP and convert to xarray format.
    load_glsea_data_combined(metadata, years=None): Load GLSEA data as a single xarray Dataset.
    load_glsea3_data_combined(metadata, years=None): Load GLSEA3 data as a single xarray Dataset.
    load_context_data(metadata): Load static context data as xarray Datasets.

Example Usage:
    # Load metadata
    metadata = load_metadata()

    # Load GLSEA data for specific years as a single Dataset
    glsea_ds = load_glsea_data_combined(metadata, years=[1995, 1996, 1997])

    # Try accessing SST data for a specific time period
    print("\nSelecting time slice 1995-01-01 to 1996-12-31:")
    sst_data = glsea_ds.sst.sel(time=slice('1995-01-01', '1996-12-31'))
    print(f"Shape: {sst_data.shape}")

    # Load GLSEA3 data for recent years
    glsea3_ds = load_glsea3_data_combined(metadata, years=[2020, 2021, 2022])

    # Load context data
    context_data = load_context_data(metadata)
"""

import json
import fsspec
import zarr
import xarray as xr
import numpy as np
import pandas as pd

def load_metadata(metadata_path="../data/config/dataset_metadata.json"):
    """Load JSON metadata (including GCP paths and other necessary info)."""
    with open(metadata_path, "r") as file:
        return json.load(file)

def load_dataset(dataset_name, metadata, year=None):
    """Load a dataset from GCP and convert to xarray format."""
    dataset_info = metadata["datasets"].get(dataset_name)
    if not dataset_info:
        raise ValueError(f"Dataset '{dataset_name}' not found in metadata.")
    
    base_path = dataset_info["path"]
    if year:
        path = f"{base_path.rstrip('/')}/{dataset_name.upper()}_{year}.zarr"
    else:
        path = base_path

    # Load the dataset
    format = dataset_info["format"]
    if format == "zarr":
        print(f"Loading {dataset_name}" + (f" for {year}" if year else ""))
        store = fsspec.get_mapper(path)
        zarr_ds = zarr.open(store)
        
        # Get array shapes
        arrays_info = {name: array.shape for name, array in zarr_ds.arrays()}
        
        # Create coordinates
        coords = {}
        # If we have time dimension (checking first non-coordinate array)
        array_shapes = {k: v for k, v in arrays_info.items() 
                       if k not in ['lat', 'lon', 'time', 'crs']}
        if array_shapes:
            first_array_shape = next(iter(array_shapes.values()))
            if len(first_array_shape) == 3:  # time, lat, lon
                n_times = first_array_shape[0]
                if year:
                    # Create daily timestamps
                    dates = pd.date_range(start=f"{year}-01-01", periods=n_times, freq='D')
                    coords['time'] = ('time', dates)
        
        # Add lat/lon coordinates
        if 'lat' in zarr_ds and 'lon' in zarr_ds:
            coords['lat'] = ('lat', zarr_ds['lat'][...])
            coords['lon'] = ('lon', zarr_ds['lon'][...])
        
        # Create data variables
        data_vars = {}
        for name, array in zarr_ds.arrays():
            if name in ['lat', 'lon', 'time', 'crs']:
                continue
            
            shape = array.shape
            if len(shape) == 3:
                dims = ('time', 'lat', 'lon')
            elif len(shape) == 2:
                dims = ('lat', 'lon')
            elif len(shape) == 1:
                dims = ('time',)
            else:
                continue
            
            data = array[...]
            data_vars[name] = (dims, data)
        
        # Create dataset and replace fill values
        ds = xr.Dataset(data_vars, coords=coords)
        ds = ds.where(ds != -99999)
        
    elif format == "netcdf":
        print(f"Loading {dataset_name}")
        with fsspec.open(path, mode='rb') as f:
            ds = xr.load_dataset(f)
    else:
        raise ValueError(f"Unsupported format '{format}' for dataset '{dataset_name}'.")
    
    return ds

def load_glsea_data_combined(metadata, years=None):
    """Load GLSEA data as a single xarray Dataset with proper time dimension."""
    if years:
        yearly_datasets = []
        for year in years:
            ds = load_dataset("glsea", metadata, year)
            if not isinstance(ds.time.values[0], np.datetime64):
                ds['time'] = pd.date_range(start=f"{year}-01-01", periods=len(ds.time), freq='D')
            yearly_datasets.append(ds)
    else:
        yearly_datasets = [load_dataset("glsea", metadata, year) 
                         for year in range(1995, 2023)]
    
    # Concatenate along time dimension
    print("\nCombining GLSEA datasets...")
    combined = xr.concat(yearly_datasets, dim='time')
    
    # Print validation statistics
    print("\nGLSEA Data Validation:")
    print(f"Time range: {combined.time.values[0]} to {combined.time.values[-1]}")
    print(f"Spatial dimensions: {combined.lat.size} x {combined.lon.size}")
    print(f"SST range: {combined.sst.min().values:.2f}°C to {combined.sst.max().values:.2f}°C")
    print(f"Water coverage: {(~np.isnan(combined.sst)).mean().values * 100:.1f}%")
        
    return combined

def load_glsea3_data_combined(metadata, years=None):
    """Load GLSEA3 data as a single xarray Dataset with proper time dimension."""
    if years:
        yearly_datasets = []
        for year in years:
            ds = load_dataset("glsea3", metadata, year)
            if not isinstance(ds.time.values[0], np.datetime64):
                ds['time'] = pd.date_range(start=f"{year}-01-01", periods=len(ds.time), freq='D')
            yearly_datasets.append(ds)
    else:
        yearly_datasets = [load_dataset("glsea3", metadata, year) 
                         for year in range(2006, 2023)]
    
    # Concatenate along time dimension
    print("\nCombining GLSEA3 datasets...")
    combined = xr.concat(yearly_datasets, dim='time')
    
    # Print validation statistics
    print("\nGLSEA3 Data Validation:")
    print(f"Time range: {combined.time.values[0]} to {combined.time.values[-1]}")
    print(f"Spatial dimensions: {combined.lat.size} x {combined.lon.size}")
    print(f"SST range: {combined.sst.min().values:.2f}°C to {combined.sst.max().values:.2f}°C")
    print(f"Water coverage: {(~np.isnan(combined.sst)).mean().values * 100:.1f}%")
        
    return combined

def load_context_data(metadata):
    """Load static context data (bathymetry, lakemask) as xarray Datasets."""
    context_data = {}
    
    print("\nLoading context data:")
    if "bathymetry" in metadata["datasets"]:
        bathymetry = load_dataset("bathymetry", metadata)
        context_data["bathymetry"] = bathymetry
        
        # Check the variable name and assign a more meaningful one if needed
        var_name = list(bathymetry.data_vars)[0]
        
        # Rename the variable for clarity
        if var_name == '__xarray_dataarray_variable__':
            bathymetry = bathymetry.rename({var_name: 'bathymetry'})  
            var_name = 'bathymetry'  
        
        data = bathymetry[var_name].compute()
        
        print("\nBathymetry Validation:")
        print(f"Dimensions: {' x '.join(str(bathymetry[var_name].sizes[dim]) for dim in bathymetry[var_name].dims)}")
        print(f"Depth range: {data.min().values:.2f}m to {data.max().values:.2f}m")
        print(f"Coverage: {(~np.isnan(data)).mean().values * 100:.1f}%")
    
    if "lakemask" in metadata["datasets"]:
        lakemask = load_dataset("lakemask", metadata)
        context_data["lakemask"] = lakemask
        print("\nLakemask Validation:")
        print(f"Dimensions: {' x '.join(str(lakemask.mask.sizes[dim]) for dim in lakemask.mask.dims)}")
        print(f"Values: {np.unique(lakemask.mask.values)}")
        print(f"Coverage: {(~np.isnan(lakemask.mask)).mean().values * 100:.1f}%")
    
    return context_data
