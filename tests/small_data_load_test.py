import fsspec
import numpy as np
import zarr

# Define the path to your public bucket data
bucket_path = "gs://great-lakes-osd/zarr_experimental/glsea/GLSEA_1995.zarr"

# Use fsspec to get the mapper for the GCP bucket
fs = fsspec.filesystem('gcs')
store = fs.get_mapper(bucket_path)

# Open the Zarr dataset
zarr_data = zarr.open(store)

# Access a variable from the dataset (e.g., sst from day one)
sst_data = zarr_data['sst'][...,0]

# Replace the fill value of -99999 with NaN
sst_data[sst_data == -99999.] = np.nan

# Replace with NaN
print(sst_data)