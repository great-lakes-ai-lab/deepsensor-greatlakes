"""
Module for loading datasets from GCP and other sources, particularly in Zarr or NetCDF format.

Functions:
    load_metadata(metadata_path): Load dataset metadata from a JSON file.
    load_dataset(dataset_name, metadata, year=None): Load a dataset from the GCP bucket, supporting year-based datasets.
    load_glsea_data(metadata, years=None): Load GLSEA data across multiple years.
    load_glsea3_data(metadata, years=None): Load GLSEA3 data across multiple years.
    load_context_data(metadata): Load context data (e.g., bathymetry, lakemask).

Example Usage:
    # Load metadata
    metadata = load_metadata()

    # Load GLSEA data for specific years
    glsea_data = load_glsea_data(metadata, years=[1995, 1996, 1997])

    # Access SST data for 1995
    sst_data_1995 = glsea_data[1995]["sst"][:]
    print(sst_data_1995)

    # Load GLSEA3 data for all years
    glsea3_data = load_glsea3_data(metadata)

    # Load context data (bathymetry and lakemask)
    context_data = load_context_data(metadata)
"""

import json
import fsspec
import zarr
import xarray as xr
import numpy as np

def load_metadata(metadata_path="../data/config/dataset_metadata.json"):
    """Load JSON metadata (including GCP paths and other necessary info)."""
    with open(metadata_path, "r") as file:
        return json.load(file)

def load_dataset(dataset_name, metadata, year=None):
    """Load a dataset from the GCP bucket (including handling years)."""
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
        print(f"Opening Zarr store from path: {path}")  # Debug print
        store = fsspec.get_mapper(path)
        return zarr.open(store)
    elif format == "netcdf":
        print(f"Opening NetCDF file from path: {path}")  # Debug print
        # Use fsspec to open the NetCDF file
        with fsspec.open(path, mode='rb') as f:
            return xr.open_dataset(f)
    else:
        raise ValueError(f"Unsupported format '{format}' for dataset '{dataset_name}'.")

    return ds

def load_glsea_data(metadata, years=None):
    """Load GLSEA data across multiple years."""
    glsea_data = {}
    if years:
        for year in years:
            glsea_data[year] = load_dataset("glsea", metadata, year)
    else:
        glsea_data = {
            year: load_dataset("glsea", metadata, year)
            for year in range(1995, 2023)
        }
    return glsea_data

def load_glsea3_data(metadata, years=None):
    """Load GLSEA3 data across multiple years."""
    glsea3_data = {}
    if years:
        for year in years:
            glsea3_data[year] = load_dataset("glsea3", metadata, year)
    else:
        glsea3_data = {
            year: load_dataset("glsea3", metadata, year)
            for year in range(2006, 2023)
        }
    return glsea3_data

def load_context_data(metadata):
    """Load context data (e.g., bathymetry, lakemask)."""
    context_data = {}
    bathymetry_data = load_dataset("bathymetry", metadata) if "bathymetry" in metadata["datasets"] else None
    lakemask_data = load_dataset("lakemask", metadata) if "lakemask" in metadata["datasets"] else None
    
    if bathymetry_data:
        context_data["bathymetry"] = bathymetry_data
    if lakemask_data:
        context_data["lakemask"] = lakemask_data

    return context_data

