# `data/` Folder

This folder is used to organize configuration, model artifacts, and related files.

## Folder Structure

```plaintext
data/
├── README.md              # Documentation for the data folder
├── config/                # Configuration files for data access
│   ├── README.md          # Explains configuration details
│   ├── gcp_paths.yaml     # Paths to datasets stored in GCP buckets
│   └── dataset_metadata.json  # Metadata about datasets
├── processors/            # DataProcessor configurations
│   └── README.md          # Guidelines for data processors
└── seasonal_cycles/       # Seasonal cycle configurations
    └── README.md          # Information about seasonal cycle storage
```

## Purpose of Each Subdirectory

### `config/`
- Stores configuration files for data access
- Contains metadata about datasets
- Defines Google Cloud Storage paths
- Provides project-wide configuration details

### `processors/`
- Stores DataProcessor configurations from DeepSensor
- Maintains normalization and preprocessing parameters
- Enables reproducible data transformations

### `seasonal_cycles/`
- Stores seasonal cycle calculations
- Preserves monthly climatological means
- Supports anomaly computation across different datasets

## Dataset Details

### Datasets on Google Cloud Platform (GCP)

| **Dataset Name**       | **Description**                                              | **Data Path**                                                       | **Format** | **Consolidated** | **Variables**                            | **Time Range**           |
|------------------------|--------------------------------------------------------------|--------------------------------------------------------------------|------------|------------------|------------------------------------------|--------------------------|
| **glsea**              | Great Lakes Surface Environmental Analysis data (GLSEA)      | gs://great-lakes-osd/zarr_experimental/glsea                        | zarr       | No               | sst, lat, lon, time, crs                 | 1995-01-01 to 2023-12-31 |
| **glsea3**             | Great Lakes Surface Environmental Analysis data (GLSEA3)     | gs://great-lakes-osd/zarr_experimental/glsea3                       | zarr       | No               | sst, lat, lon, time, crs                 | 2006-01-01 to 2023-12-31 |
| **bathymetry**         | Interpolated bathymetry data for the Great Lakes             | gs://great-lakes-osd/context/interpolated_bathymetry.nc             | netcdf     | N/A              | N/A                                      | N/A                      |
| **lakemask**           | Lake mask data for the Great Lakes                          | gs://great-lakes-osd/context/lakemask.nc                            | netcdf     | N/A              | N/A                                      | N/A                      |
| **ice_concentration**  | Ice concentration data for the Great Lakes                   | gs://great-lakes-osd/ice_concentration.zarr                        | zarr       | Yes              | ice_concentration, lat, lon, time        | 1972-01-01 to 2023-05-21 |
| **ermask**                | Mask for Lake Erie                            | gs://great-lakes-osd/context/ermask.nc                              | netcdf     | N/A              | mask                                    | N/A                      |
| **hurmask**               | Mask for Lake Huron             | gs://great-lakes-osd/context/hurmask.nc                             | netcdf     | N/A              | mask                                    | N/A                      |                    |                   |
| **michmask**              | Mask for Lake Michigan                              | gs://great-lakes-osd/context/michmask.nc                            | netcdf     | N/A              | mask                                    | N/A                      |
| **ontmask**               | Mask for Lake Ontario                               | gs://great-lakes-osd/context/ontmask.nc                             | netcdf     | N/A              | mask                                    | N/A                      |
| **supmask**               | Mask for Lake Superior                              | gs://great-lakes-osd/context/supmask.nc                             | netcdf     | N/A              | mask                                    | N/A                      |


### Datasets on U-M HPC Turbo Research Storage

| **Dataset Name**         | **Description**                                              | **Data Path**                                                       | **Format** | **Consolidated** | **Variables**                            | **Time Range**           |
|--------------------------|--------------------------------------------------------------|--------------------------------------------------------------------|------------|------------------|------------------------------------------|--------------------------|
| **glsea**          | Great Lakes Surface Environmental Analysis data (GLSEA) | /nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA_NETCDF       | netcdf     | No               | sst, lat, lon, time, crs                 | 1995-01-01 to 2023-12-31 |
| **glsea (zarr)**   | Great Lakes Surface Environmental Analysis data (GLSEA) | /nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA_combined.zarr | zarr       | Yes              | sst, lat, lon, time, crs                 | 1995-01-01 to 2023-12-31 |
| **glsea3**         | Great Lakes Surface Environmental Analysis data (GLSEA3) | /nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA3_NETCDF      | netcdf     | No               | sst, lat, lon, time, crs                 | 2006-01-01 to 2023-12-31 |
| **glsea3 (zarr)**  | Great Lakes Surface Environmental Analysis data (GLSEA3) | /nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA3_combined.zarr | zarr       | Yes              | sst, lat, lon, time, crs                 | 2006-01-01 to 2023-12-31 |
| **bathymetry**     | Interpolated bathymetry data for the Great Lakes    | /nfs/turbo/seas-dannes/SST-sensor-placement-input/bathymetry/interpolated_bathymetry.nc | netcdf | N/A              | N/A                                      | N/A                      |
| **lakemask**       | Lake mask data for the Great Lakes                  | /nfs/turbo/seas-dannes/SST-sensor-placement-input/masks/lakemask.nc | netcdf     | N/A              | N/A                                      | N/A                      |
| **ice_concentration** | Ice concentration data for the Great Lakes      | /nfs/turbo/seas-dannes/SST-sensor-placement-input/NSIDC/ice_concentration.zarr             | Zarr     | N/A              | ice_concentration, lat, lon, time        | 1972-01-01 to 2023-05-21 |
| **ermask**                | Mask for Lake Erie                            | /nfs/turbo/seas-dannes/SST-sensor-placement-input/NSIDC/ermask.nc    | netcdf     | N/A              | mask                                    | N/A                      |
| **hurmask**               | Mask for Lake Huron             | /nfs/turbo/seas-dannes/SST-sensor-placement-input/NSIDC/hurmask.nc   | netcdf     | N/A              | mask                                    | N/A                      |
| **michmask**              | Mask for Lake Michigan                              | /nfs/turbo/seas-dannes/SST-sensor-placement-input/NSIDC/michmask.nc  | netcdf     | N/A              | mask                                    | N/A                      |
| **ontmask**               | Mask for Lake Ontario                               | /nfs/turbo/seas-dannes/SST-sensor-placement-input/NSIDC/ontmask.nc   | netcdf     | N/A              | mask                                    | N/A                      |
| **supmask**               | Mask for Lake Superior                              | /nfs/turbo/seas-dannes/SST-sensor-placement-input/NSIDC/supmask.nc   | netcdf     | N/A              | mask                                    | N/A                      |


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

## Working with Configurations

### Accessing Configurations

```python
import json
import yaml

# Load dataset metadata
with open('data/config/dataset_metadata.json', 'r') as f:
    dataset_metadata = json.load(f)

# Load GCP paths
with open('data/config/gcp_paths.yaml', 'r') as f:
    gcp_paths = yaml.safe_load(f)
```

## Best Practices

1. **Do not commit large files** to the repository
2. Use `.gitkeep` or README files to maintain directory structure
3. Document any changes to configurations
4. Ensure reproducibility of data processing steps

## Adding New Configurations

Refer to the README in the `config/` directory for detailed instructions on adding new datasets or modifying existing configurations.

## Dependencies

- `json`
- `yaml`
- `deepsensor`
- `fsspec`
