# Project Configuration

This directory contains configuration files

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

### GLSEA
- Time Range: 1995-01-01 to 2023-12-31
- Variables: SST, Latitude, Longitude, Time, CRS

### GLSEA3
- Time Range: 2006-01-01 to 2023-12-31
- Variables: SST, Latitude, Longitude, Time, CRS

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

