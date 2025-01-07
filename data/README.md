```markdown
# `data/` Folder  

This folder is used to organize datasets and related files for the project. Below is a description of its structure and how to use the data stored here.

---

## Folder Structure

```plaintext
data/
├── README.md              # Documentation for the data folder
├── config/                # Configuration files for data access
│   ├── gcp_paths.yaml     # Paths to datasets stored in GCP buckets
│   └── dataset_metadata.json  # Metadata about datasets
├── raw/                   # Placeholder for raw datasets
│   ├── glsea/             # Local or symlink to `great-lakes-osd/zarr_experimental/glsea`
│   └── glsea3/            # Local or symlink to `great-lakes-osd/zarr_experimental/glsea3`
├── processed/             # Preprocessed or intermediate datasets
│   ├── glsea_subset.zarr/ # Example subset of GLSEA data
│   └── other_processed.nc
├── test/                  # Minimal or mock datasets for testing
│   ├── small_glsea.zarr/  # Tiny subset of GLSEA for unit tests
│   └── sample_input.nc
└── outputs/               # Output files from experiments or processing scripts
    └── analysis_results.csv
```

---

## Using the Datasets

### **Accessing Datasets in GCP Buckets**  
The following datasets are stored in public Google Cloud Storage buckets:  
1. **GLSEA Dataset**: `great-lakes-osd/zarr_experimental/glsea`  
2. **GLSEA3 Dataset**: `great-lakes-osd/zarr_experimental/glsea3`

These datasets can be accessed programmatically using Python with the `fsspec` and `zarr` libraries. Example code:

```python
import zarr
import fsspec

store = fsspec.get_mapper('gs://great-lakes-osd/zarr_experimental/glsea')
ds = zarr.open_consolidated(store)
print(ds.tree())
```

---

### **Working with Local Test Datasets**  
For testing or development, you can use the smaller datasets stored in the `test/` directory. Example:

```python
import zarr

test_store = zarr.DirectoryStore('data/test/small_glsea.zarr')
test_ds = zarr.open(test_store)
print(test_ds.tree())
```

---

## Adding Data to This Folder
- **Raw Datasets**: Place symlinks or stubs in `raw/` for large datasets stored externally. Avoid storing large files directly.
- **Processed Data**: Save intermediate or preprocessed files to the `processed/` folder.
- **Testing Data**: Include minimal mock data in the `test/` folder for unit tests.

---

## Notes
1. **Large Datasets**: Avoid committing large files to the repository. Use `.gitignore` to exclude unnecessary files.
2. **Documentation**: Update this README if new datasets or files are added to the folder structure.
3. **Sensitive Data**: Do not store credentials or sensitive data in this folder.
```
