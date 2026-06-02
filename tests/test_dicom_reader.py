import pytest
import numpy as np
import os
from src.io.dicom_reader import DICOMReader

reader = DICOMReader()

# Path to your 10 sample images downloaded from Kaggle
SAMPLE_DIR = "data/raw/sample_dicoms"


def get_sample_files():
    if not os.path.exists(SAMPLE_DIR):
        pytest.skip(f"Sample images not found at {SAMPLE_DIR}")
    files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".png")]
    if not files:
        pytest.skip("No PNG files found in sample dir")
    return [os.path.join(SAMPLE_DIR, f) for f in files[:5]]


def test_output_shape():
    """Image must always be exactly 1024x1024."""
    for path in get_sample_files():
        image, _ = reader.load_from_png(path)
        assert image.shape == (1024, 1024), f"Wrong shape for {path}: {image.shape}"


def test_output_dtype():
    """Image must be float32."""
    for path in get_sample_files():
        image, _ = reader.load_from_png(path)
        assert image.dtype == np.float32, f"Wrong dtype: {image.dtype}"


def test_output_range():
    """All pixel values must be between 0.0 and 1.0."""
    for path in get_sample_files():
        image, _ = reader.load_from_png(path)
        assert image.min() >= 0.0, f"Pixel below 0: {image.min()}"
        assert image.max() <= 1.0, f"Pixel above 1: {image.max()}"


def test_metadata_keys():
    """PNG fallback metadata must have all required keys."""
    _, metadata = reader.load_from_png(get_sample_files()[0])
    required_keys = [
        "study_uid", "modality", "view_position", "body_part",
        "exposure_index", "deviation_index", "kvp", "patient_id"
    ]
    for key in required_keys:
        assert key in metadata, f"Missing metadata key: {key}"


def test_no_nan_values():
    """Image must not contain NaN or Inf values."""
    for path in get_sample_files():
        image, _ = reader.load_from_png(path)
        assert not np.isnan(image).any(), f"NaN found in {path}"
        assert not np.isinf(image).any(), f"Inf found in {path}"