# Seasonal Cycles

This directory stores seasonal cycle configurations.

## Directory Structure

- Each seasonal cycle is saved with a descriptive filename
- Filename format: `{dataset}_{start_year}_{end_year}.nc`

## Purpose

Seasonal cycles represent the monthly climatological mean for each grid cell. These are critical for:
- Anomaly calculation
- Understanding spatial variability of temperature patterns
- Preprocessing for machine learning models

## Best Practices

- Do not commit large or sensitive data files to version control
- Use unique identifiers for each seasonal cycle configuration
- Document the source and method of calculation

## Example Filenames

- `glsea_1995_1996.nc`
- `glsea3_2010_2020.nc`

## Accessing Seasonal Cycles

Seasonal cycles can be loaded using the `SeasonalCycleProcessor` in the project's preprocessing module.