# `data/` Folder

This folder is used to organize configuration, model artifacts, and related files for the Great Lakes Sea Surface Temperature (SST) Project.

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
