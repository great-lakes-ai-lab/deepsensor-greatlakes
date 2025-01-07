import fsspec
import zarr

# Define the path to your public bucket data
bucket_path = "gs://great-lakes-osd/zarr_experimental/glsea/GLSEA_1995.zarr"

# Use fsspec to get the mapper for the GCP bucket
fs = fsspec.filesystem('gcs')
store = fs.get_mapper(bucket_path)

# Open the Zarr dataset
zarr_data = zarr.open(store)

# Access a variable from the dataset (e.g., sst)
sst_array = zarr_data['sst']

# Print the chunking information
print("Chunking for 'sst':", sst_array.chunks)
