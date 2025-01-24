# Project Configuration

This directory contains configuration files that enable loading of datasets. It also contains a description of available datasets. 

## Configuration Files

### `dataset_metadata.json`
- Comprehensive dataset metadata
- Includes details for GLSEA and GLSEA3 datasets
- Specifies data paths, formats, and time ranges
- Tracks project metadata

### `gcp_paths.yaml`
- Defines Google Cloud Storage paths for datasets
- Specifies default chunk sizes for data loading
- Includes project metadata and GCS access options

## Key Features

- Centralized configuration management
- Support for multiple datasets
- Flexible data access configuration
- Cloud storage integration

## Dataset Details

| **Dataset Name**     | **Description**                                              | **Data Path**                                                       | **Format** | **Consolidated** | **Variables**                            | **Time Range**           |
|----------------------|--------------------------------------------------------------|--------------------------------------------------------------------|------------|------------------|------------------------------------------|--------------------------|
| **glsea**            | Great Lakes Surface Environmental Analysis data (GLSEA)      | gs://great-lakes-osd/zarr_experimental/glsea                        | zarr       | No               | sst, lat, lon, time, crs                 | 1995-01-01 to 2023-12-31 |
| **glsea_umhpc**      | Great Lakes Surface Environmental Analysis data (GLSEA) on U-M HPC | /nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA_NETCDF       | netcdf     | No               | sst, lat, lon, time, crs                 | 1995-01-01 to 2023-12-31 |
| **glsea3**           | Great Lakes Surface Environmental Analysis data (GLSEA3)     | gs://great-lakes-osd/zarr_experimental/glsea3                       | zarr       | No               | sst, lat, lon, time, crs                 | 2006-01-01 to 2023-12-31 |
| **glsea3_umhpc**     | Great Lakes Surface Environmental Analysis data (GLSEA3) on U-M HPC | /nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA3_NETCDF      | netcdf     | No               | sst, lat, lon, time, crs                 | 2006-01-01 to 2023-12-31 |
| **bathymetry**       | Interpolated bathymetry data for the Great Lakes             | gs://great-lakes-osd/context/interpolated_bathymetry.nc             | netcdf     | N/A              | N/A                                      | N/A                      |
| **bathymetry_umhpc** | Interpolated bathymetry data for the Great Lakes on U-M HPC     | /nfs/turbo/seas-dannes/SST-sensor-placement-input/bathymetry/interpolated_bathymetry.nc | netcdf     | N/A              | N/A                                      | N/A                      |
| **lakemask**         | Lake mask data for the Great Lakes                          | gs://great-lakes-osd/context/lakemask.nc                            | netcdf     | N/A              | N/A                                      | N/A                      |
| **lakemask_umhpc**   | Lake mask data for the Great Lakes on U-M HPC                  | /nfs/turbo/seas-dannes/SST-sensor-placement-input/masks/lakemask.nc | netcdf     | N/A              | N/A                                      | N/A                      |

### Accessing Datasets via ArrayLake

Datasets from this project, including the GLSEA and GLSEA3 datasets, can be accessed using the ArrayLake library for cloud-based data management. ArrayLake simplifies data loading and management for large datasets stored on cloud servers. 

To access the datasets:

1. **Install ArrayLake**:
   Install ArrayLake using pip if it's not already installed:

   ```bash
   pip install arraylake
   ```
2. **Complete Authentication Process**:
   You will need to be added to the `great-lakes-ai-lab` ArrayLake organizaiton for this next step to work. Use this command:
   ```bash
   arraylake auth login

   # Or, if running from a remote environment
    arraylake auth login --no-browser
   ```
   To complete the authentication process. For more information, see the [ArrayLake documentation](https://docs.earthmover.io/).

3. **Load Data with ArrayLake**: 
   You can load the datasets into an xarray object with the following code:
    ```python
    from arraylake import Client
    client = Client()

    # Initialize ArrayLake repo object
    repo = client.get_repo('great-lakes-ai-lab/glsea3')

    # Load GLSEA3 dataset
    ds = repo.to_xarray()

    # Explore the dataset
    print(ds)
    ```


## Access and Usage

Configurations can be loaded using standard Python libraries:
- JSON: `json.load()`
- YAML: `yaml.safe_load()`

## Metadata

- Owner: Dani Jones
- Last Updated: 2025-01-06
- Project: DeepSensor Great Lakes

## Adding a New Dataset

To add a new dataset to the configuration, follow these steps:

### 1. Update `dataset_metadata.json`

Add a new entry under the `datasets` key:

```json
"new_dataset_name": {
    "description": "Descriptive name of the dataset",
    "path": "gs://bucket-name/path/to/dataset",
    "format": "zarr", // or "netcdf"
    "consolidated": false, // optional
    "variables": ["variable1", "variable2", "lat", "lon", "time"],
    "time_range": {
        "start": "YYYY-MM-DD",
        "end": "YYYY-MM-DD"
    }
}
```

### 2. Update `gcp_paths.yaml`

Add the dataset under the `datasets` section:

```yaml
datasets:
  new_dataset_name:
    description: "Description of the new dataset"
    path: "gs://bucket-name/path/to/dataset"
    consolidated: false
```

### 3. Update Chunk Sizes (Optional)

In `gcp_paths.yaml`, adjust default chunk sizes if needed:

```yaml
default_chunk_size:
  time: 46
  latitude: 105
  longitude: 148
  # Add or modify chunk sizes as necessary
```

### Example: Adding a New Sea Surface Temperature Dataset

Let's say we're adding a new SST dataset from the North Atlantic:

```json
"north_atlantic_sst": {
    "description": "North Atlantic Sea Surface Temperature",
    "path": "gs://ocean-data/north-atlantic-sst",
    "format": "zarr",
    "consolidated": true,
    "variables": ["sst", "lat", "lon", "time"],
    "time_range": {
        "start": "1980-01-01",
        "end": "2022-12-31"
    }
}
```

### Best Practices

- Use consistent naming conventions
- Provide clear, descriptive metadata
- Verify the dataset path and accessibility
- Ensure all required variables are specified
- Update metadata fields like `last_updated`

### Validation

After adding a new dataset:
1. Verify the JSON and YAML are valid
2. Test loading the dataset in your code
3. Update any relevant documentation

