# Data Processors

This directory stores DataProcessor configurations. 

## Directory Structure

- Each DataProcessor configuration is saved in its own folder
- Configuration includes normalization parameters and coordinate mappings

## Purpose

DataProcessor configurations are used to:
- Standardize and normalize input data
- Maintain consistent preprocessing across different models and experiments
- Enable reproducible machine learning workflows

## Best Practices

- Use descriptive names for processor configurations
- Document the source data and preprocessing steps
- Ensure reproducibility by saving complete configuration details

## Example Configuration Names

- `glsea_sst_anomalies`
- `ml_model_v1_preprocessor`

## Accessing Processors

Processors can be loaded using the `DataProcessor` class from the DeepSensor library.

Example:
```python
from deepsensor.data import DataProcessor

# Load a specific processor configuration
processor = DataProcessor("path/to/processor/config")