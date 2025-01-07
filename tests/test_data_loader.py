import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from deepsensor_greatlakes.dataset_loader import (
    load_metadata,
    load_dataset,
    load_glsea_data_combined,
    load_glsea3_data_combined,
    load_context_data,
)

class TestDatasetLoader(unittest.TestCase):
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data=json.dumps({
        "datasets": {
            "glsea": {
                "path": "gs://great-lakes-osd/zarr_experimental/glsea",
                "format": "zarr",
            },
            "glsea3": {
                "path": "gs://great-lakes-osd/zarr_experimental/glsea3",
                "format": "zarr",
            },
            "bathymetry": {
                "path": "gs://great-lakes-osd/context/interpolated_bathymetry.nc",
                "format": "netcdf",
            },
            "lakemask": {
                "path": "gs://great-lakes-osd/context/lakemask.nc",
                "format": "netcdf",
            },
        },
    }))
    def test_load_metadata(self, mock_open):
        metadata = load_metadata()
        self.assertIn("glsea", metadata["datasets"])
        self.assertEqual(metadata["datasets"]["glsea"]["path"], "gs://great-lakes-osd/zarr_experimental/glsea")

    @patch("fsspec.get_mapper")
    @patch("zarr.open")
    def test_load_dataset(self, mock_zarr_open, mock_fsspec_get_mapper):
        # Create a more sophisticated mock Zarr array
        class MockZarrArray:
            def __init__(self, data):
                self._data = data
            
            def __getitem__(self, key):
                if key == ...:
                    return self._data
                try:
                    return self._data[key]
                except TypeError:
                    return self._data
            
            @property
            def shape(self):
                return self._data.shape
            
            def __contains__(self, key):
                return key in ['data']  # Simulate Zarr array's attribute check

        # Create a mock Zarr dataset that has an .arrays() method
        class MockZarrDataset:
            def __init__(self):
                lat_data = np.linspace(0, 9, 10)
                lon_data = np.linspace(0, 9, 10)
                sst_data = np.random.rand(365, 10, 10)
                
                self._data = {
                    'sst': MockZarrArray(sst_data),
                    'lat': MockZarrArray(lat_data),
                    'lon': MockZarrArray(lon_data)
                }
            
            def arrays(self):
                return list(self._data.items())
            
            def __getitem__(self, key):
                return self._data[key]
            
            def __contains__(self, key):
                return key in ['sst', 'lat', 'lon']

        # Setup mocks
        mock_store = MagicMock()
        mock_fsspec_get_mapper.return_value = mock_store

        # Create mock zarr dataset
        mock_zarr_dataset = MockZarrDataset()
        mock_zarr_open.return_value = mock_zarr_dataset

        # Prepare metadata
        metadata = {
            "datasets": {
                "glsea": {
                    "path": "gs://great-lakes-osd/zarr_experimental/glsea",
                    "format": "zarr",
                }
            }
        }
        
        # Call the function
        ds = load_dataset("glsea", metadata, year=1995)

        # Verify the dataset
        self.assertIsInstance(ds, xr.Dataset)
        
        # Check data variables
        self.assertIn("sst", ds.data_vars)
        self.assertEqual(ds.sst.shape, (365, 10, 10))
        
        # Check coordinates
        self.assertIn("time", ds.coords)
        self.assertIn("lat", ds.coords)
        self.assertIn("lon", ds.coords)
        
        # Check time coordinate
        self.assertEqual(len(ds.time), 365)
        self.assertEqual(ds.time[0].dt.year.values, 1995)
        self.assertEqual(ds.time[0].dt.month.values, 1)
        self.assertEqual(ds.time[0].dt.day.values, 1)

    @patch("deepsensor_greatlakes.dataset_loader.load_dataset")
    def test_load_glsea_data_combined(self, mock_load_dataset):
        mock_ds = xr.Dataset(
            {
                "sst": (("time", "lat", "lon"), np.random.rand(365, 10, 10)),
            },
            coords={
                "time": pd.date_range("1995-01-01", periods=365),
                "lat": np.linspace(0, 9, 10),
                "lon": np.linspace(0, 9, 10),
            },
        )
        mock_load_dataset.return_value = mock_ds

        metadata = {
            "datasets": {
                "glsea": {
                    "path": "gs://great-lakes-osd/zarr_experimental/glsea",
                    "format": "zarr",
                }
            }
        }
        combined_ds = load_glsea_data_combined(metadata, years=[1995, 1996])

        self.assertIn("sst", combined_ds.data_vars)
        self.assertEqual(combined_ds.time.size, 365 * 2)  # Combined two years
        self.assertEqual(combined_ds.sst.shape, (365 * 2, 10, 10))

    @patch("deepsensor_greatlakes.dataset_loader.load_dataset")
    def test_load_context_data(self, mock_load_dataset):
        mock_bathymetry = xr.Dataset(
            {
                "depth": (("lat", "lon"), np.random.rand(10, 10)),
            },
            coords={
                "lat": np.linspace(0, 9, 10),
                "lon": np.linspace(0, 9, 10),
            },
        )
        mock_lakemask = xr.Dataset(
            {
                "mask": (("lat", "lon"), np.random.randint(0, 2, (10, 10))),
            },
            coords={
                "lat": np.linspace(0, 9, 10),
                "lon": np.linspace(0, 9, 10),
            },
        )
        mock_load_dataset.side_effect = [mock_bathymetry, mock_lakemask]

        metadata = {
            "datasets": {
                "bathymetry": {"path": "gs://great-lakes-osd/context/interpolated_bathymetry.nc", "format": "netcdf"},
                "lakemask": {"path": "gs://great-lakes-osd/context/lakemask.nc", "format": "netcdf"},
            }
        }
        context_data = load_context_data(metadata)

        self.assertIn("bathymetry", context_data)
        self.assertIn("lakemask", context_data)
        self.assertEqual(context_data["bathymetry"].depth.shape, (10, 10))
        self.assertEqual(context_data["lakemask"].mask.shape, (10, 10))

if __name__ == "__main__":
    unittest.main()