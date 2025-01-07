import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os

# Add the src directory to the path to import your module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the functions from your dataset_loader module
from dataset_loader import load_metadata, load_dataset, load_glsea_data_and_context

class TestDatasetLoader(unittest.TestCase):

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data=json.dumps({
        "datasets": {
            "glsea": {
                "path": "gs://great-lakes-osd/zarr_experimental/glsea",
                "format": "zarr",
                "consolidated": False,
                "variables": ["sst", "lat", "lon", "time", "crs"],
                "time_range": {
                    "start": "1995-01-01",
                    "end": "2023-12-31"
                }
            },
            "bathymetry": {
                "path": "gs://great-lakes-osd/context/interpolated_bathymetry.nc",
                "format": "netcdf"
            },
            "lakemask": {
                "path": "gs://great-lakes-osd/context/lakemask.nc",
                "format": "netcdf"
            }
        },
        "metadata": {
            "owner": "Dani Jones",
            "last_updated": "2025-01-06",
            "project_name": "DeepSensor Great Lakes",
            "description": "Configuration for accessing Great Lakes datasets on GCP"
        }
    }))
    def test_load_metadata(self, mock_open):
        """Test loading metadata from a JSON file."""
        metadata = load_metadata()
        
        # Check that the function is reading the correct metadata
        self.assertEqual(metadata["datasets"]["glsea"]["path"], "gs://great-lakes-osd/zarr_experimental/glsea")
        self.assertEqual(metadata["datasets"]["bathymetry"]["format"], "netcdf")

        # Ensure open was called with the correct file path
        mock_open.assert_called_once_with("../data/config/dataset_metadata.json", "r")

    @patch("dataset_loader.load_dataset")
    def test_load_glsea_data_and_context(self, mock_load_dataset):
        """Test loading both GLSEA data and context data (bathymetry, lakemask)."""
        # Mock the datasets
        mock_glsea_data = {
            1995: {"sst": MagicMock()},
            1996: {"sst": MagicMock()},
            1997: {"sst": MagicMock()},
        }
        mock_context_data = {
            "bathymetry": MagicMock(),
            "lakemask": MagicMock(),
        }

        # Set up mock calls for loading datasets
        mock_load_dataset.side_effect = lambda dataset_name, metadata, year=None: mock_glsea_data[year] if dataset_name == "glsea" else mock_context_data[dataset_name]

        # Prepare mock metadata
        mock_metadata = {
            "datasets": {
                "glsea": {"path": "gs://great-lakes-osd/zarr_experimental/glsea", "format": "zarr"},
                "bathymetry": {"path": "gs://great-lakes-osd/context/interpolated_bathymetry.nc", "format": "netcdf"},
                "lakemask": {"path": "gs://great-lakes-osd/context/lakemask.nc", "format": "netcdf"}
            }
        }

        # Load the GLSEA and context data
        data = load_glsea_data_and_context(mock_metadata, years=[1995, 1996, 1997])

        # Assertions
        self.assertIn(1995, data["glsea"])
        self.assertIn("bathymetry", data)
        self.assertIn("lakemask", data)

        # Check if SST data for 1995 is loaded correctly
        sst_data_1995 = data["glsea"][1995]["sst"]
        self.assertIsNotNone(sst_data_1995)

        # Check if bathymetry data is loaded
        bathymetry_data = data["bathymetry"]
        self.assertIsNotNone(bathymetry_data)

        # Check if lakemask data is loaded
        lakemask_data = data["lakemask"]
        self.assertIsNotNone(lakemask_data)

    @patch("dataset_loader.load_dataset")
    def test_load_glsea_data_for_specific_year(self, mock_load_dataset):
        """Test loading GLSEA data for a specific year."""
        mock_glsea_data = {
            1995: {"sst": MagicMock()},
        }
        mock_load_dataset.side_effect = lambda dataset_name, metadata, year=None: mock_glsea_data[year] if dataset_name == "glsea" else None

        # Prepare mock metadata
        mock_metadata = {
            "datasets": {
                "glsea": {"path": "gs://great-lakes-osd/zarr_experimental/glsea", "format": "zarr"}
            }
        }

        # Load GLSEA data for 1995
        data = load_glsea_data_and_context(mock_metadata, years=[1995])

        # Check if the data for 1995 is correctly loaded
        self.assertIn(1995, data["glsea"])
        self.assertIsNotNone(data["glsea"][1995]["sst"])

if __name__ == "__main__":
    unittest.main()
