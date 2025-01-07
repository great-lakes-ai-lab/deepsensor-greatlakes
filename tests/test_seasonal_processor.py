import unittest
import pytest
import pandas as pd
import xarray as xr
import numpy as np
import os
import tempfile
import shutil

from deepsensor_greatlakes.preprocessor import SeasonalCycleProcessor

class TestSeasonalCycleProcessor:
    def setup_method(self):
        # Create a sample dataset for testing
        times = pd.date_range(start='1995-01-01', end='1996-12-31', freq='D')
        lats = np.linspace(38, 50, 10)
        lons = np.linspace(-92, -75, 15)
        
        # Create a dataset with some spatial and temporal variation
        data = np.random.rand(len(times), len(lats), len(lons)) * 20 + 10
        self.test_ds = xr.Dataset(
            data_vars={
                'sst': (('time', 'lat', 'lon'), data)
            },
            coords={
                'time': times,
                'lat': lats,
                'lon': lons
            }
        )
        
        # Temporary directory for saving/loading
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_calculate_seasonal_cycle(self):
        # Calculate seasonal cycle
        processor = SeasonalCycleProcessor()
        processor.calculate(self.test_ds)
        
        # Check dimensions
        assert 'month' in processor.seasonal_cycle.dims
        assert processor.seasonal_cycle.month.size == 12
        
        # Check spatial dimensions maintained
        assert processor.seasonal_cycle.lat.size == self.test_ds.lat.size
        assert processor.seasonal_cycle.lon.size == self.test_ds.lon.size

    def test_save_and_load(self):
        # Calculate and save
        processor = SeasonalCycleProcessor()
        processor.calculate(self.test_ds)
        saved_paths = processor.save(base_dir=self.test_dir)
        
        # Load and verify
        loaded_processor = SeasonalCycleProcessor.load(
            saved_paths['seasonal_cycle_path'],
            saved_paths['metadata_path']
        )
        
        # Check data equality
        xr.testing.assert_allclose(
            processor.seasonal_cycle, 
            loaded_processor.seasonal_cycle
        )
        
        # Check metadata
        assert 'id' in loaded_processor.metadata
        assert 'calculation_date' in loaded_processor.metadata

    def test_compute_anomalies(self):
        # Calculate seasonal cycle
        processor = SeasonalCycleProcessor()
        processor.calculate(self.test_ds)
        
        # Compute anomalies
        anomalies = processor.compute_anomalies(self.test_ds)
        
        # Check dimensions maintained
        assert anomalies.time.size == self.test_ds.time.size
        assert anomalies.lat.size == self.test_ds.lat.size
        assert anomalies.lon.size == self.test_ds.lon.size
        
        # Verify anomaly computation
        # The mean of anomalies for each month should be close to zero
        for month in range(1, 13):
            month_anomalies = anomalies.sel(time=anomalies.time.dt.month == month)
            
            # Compute mean for each grid cell first, then overall mean
            month_mean = month_anomalies.mean(dim=['lat', 'lon']).sst.values.mean()
            
            # Check if mean is close to zero
            np.testing.assert_almost_equal(
                month_mean, 
                0, 
                decimal=1  # Allow some small floating-point variations
            )

if __name__ == "__main__":
    pytest.main()