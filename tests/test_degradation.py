import os
import h5py
import pandas as pd
import pytest

DEGRADED_HDF5 = "data/processed/cxr_degraded.h5"
MANIFEST_CSV = "data/processed/degradation_manifest.csv"

@pytest.mark.skipif(not os.path.exists(DEGRADED_HDF5), reason="Dataset file not generated yet")
def test_dataset_integrity():
    # 1. Ensure the generated metadata file exists
    assert os.path.exists(MANIFEST_CSV), "Manifest sheet missing"
    df = pd.read_csv(MANIFEST_CSV)
    
    # 2. Assert exact pipeline generation proportions (1 baseline + 8 corruptions)
    clean_count = len(df[df['severity'] == 0])
    degraded_count = len(df[df['severity'] != 0])
    assert clean_count * 8 == degraded_count, "Data engineering balance error: Tiers are uneven"
    
    # 3. Pull a sample dataset to check structural and mathematical properties
    sample_uid = df.iloc[0]['uid']
    with h5py.File(DEGRADED_HDF5, 'r') as h5f:
        assert sample_uid in h5f, f"Key reference {sample_uid} missing from database map"
        img_array = h5f[sample_uid][:]
        
        # Verify shape matches input requirements (1024x1024)
        assert img_array.shape == (1024, 1024), f"Array dimensions mismatch: {img_array.shape}"
        assert img_array.dtype in ['float32', 'float64'], "Array precision is not a float format"
        
        # Verify pixel normalization thresholds are strictly kept
        assert img_array.min() >= 0.0, "Normalization failure: Pixel scaling dropped below 0.0"
        assert img_array.max() <= 1.0, "Normalization failure: Pixel scaling spiked above 1.0"