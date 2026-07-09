# tests/test_input_validator.py
"""
Harden and test the fail-safe layer.

Builds a deliberately malformed set of inputs and confirms the validator
rejects each one with a specific, identifiable reason — not a generic
crash, and not a silent pass-through that would let a scorer run on
garbage.

Run with: pytest tests/test_input_validator.py -v
"""
import numpy as np
import pytest

from src.validation.input_validator import validate_input, ValidationResult


GOOD_METADATA = {
    "study_uid": "TEST001",
    "modality": "CR",
    "view_position": "PA",
    "body_part": "CHEST",
    "exposure_index": 1850.0,
    "deviation_index": 0.2,
    "kvp": 110,
}


def _good_image(size=1024):
    """A well-formed, non-blank, in-range float32 image — the control case."""
    rng = np.random.default_rng(42)
    img = rng.uniform(0.05, 0.95, size=(size, size)).astype(np.float32)
    return img


# -----------------------------------------------------------------------
# Control case — must pass
# -----------------------------------------------------------------------

def test_well_formed_input_passes():
    result = validate_input(_good_image(), dict(GOOD_METADATA))
    assert result.is_valid is True
    assert result.failed_checks == []


# -----------------------------------------------------------------------
# Wrong shape / dimensionality
# -----------------------------------------------------------------------

def test_wrong_dimensions_too_small():
    img = np.random.uniform(0, 1, size=(8, 8)).astype(np.float32)
    result = validate_input(img, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert any("image_too_small" in c for c in result.failed_checks)


def test_non_square_image_rejected_when_resize_expected():
    img = np.random.uniform(0, 1, size=(1024, 768)).astype(np.float32)
    result = validate_input(img, dict(GOOD_METADATA), expect_square=True)
    assert result.is_valid is False
    assert any("image_not_square" in c for c in result.failed_checks)


def test_non_square_image_allowed_when_resize_disabled():
    """
    Fix applied after review: if the pipeline's resize step is disabled
    (config.image.resize is None), a non-square image is expected, valid
    behaviour — it must not be wrongly rejected as a structural defect.
    """
    img = np.random.uniform(0.05, 0.95, size=(1024, 768)).astype(np.float32)
    result = validate_input(img, dict(GOOD_METADATA), expect_square=False)
    assert not any("image_not_square" in c for c in result.failed_checks)


def test_implausibly_large_image():
    # Simulates a corrupt decode that produced a garbage huge array.
    img = np.zeros((20000, 20000), dtype=np.float32)
    result = validate_input(img, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert any("image_too_large" in c for c in result.failed_checks)


def test_unexpected_channel_count():
    img = np.random.uniform(0, 1, size=(1024, 1024, 7)).astype(np.float32)
    result = validate_input(img, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert any("image_unexpected_channels" in c for c in result.failed_checks)


# -----------------------------------------------------------------------
# Corrupted / unreadable file simulation
# -----------------------------------------------------------------------

def test_none_image_simulates_failed_decode():
    result = validate_input(None, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert "image_is_none" in result.failed_checks


def test_wrong_type_not_ndarray():
    result = validate_input([[1, 2], [3, 4]], dict(GOOD_METADATA))
    assert result.is_valid is False
    assert any("image_wrong_type" in c for c in result.failed_checks)


def test_image_with_nan_values():
    img = _good_image()
    img[0, 0] = np.nan
    result = validate_input(img, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert "contains_nan" in result.failed_checks


def test_image_with_inf_values():
    img = _good_image()
    img[0, 0] = np.inf
    result = validate_input(img, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert "contains_inf" in result.failed_checks


# -----------------------------------------------------------------------
# Blank image (decoded successfully but contains nothing useful)
# -----------------------------------------------------------------------

def test_all_black_image_is_rejected():
    img = np.zeros((1024, 1024), dtype=np.float32)
    result = validate_input(img, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert any("blank_or_flat_image" in c for c in result.failed_checks)


def test_all_white_image_is_rejected():
    img = np.ones((1024, 1024), dtype=np.float32)
    result = validate_input(img, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert any("blank_or_flat_image" in c for c in result.failed_checks)


def test_uniform_mid_gray_image_is_rejected():
    img = np.full((1024, 1024), 0.5, dtype=np.float32)
    result = validate_input(img, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert any("blank_or_flat_image" in c for c in result.failed_checks)


# -----------------------------------------------------------------------
# Wrong dtype / out-of-range values (e.g. unscaled DICOM slipped through
# without VOI-LUT/normalisation applied)
# -----------------------------------------------------------------------

def test_wrong_dtype_uint16_rejected():
    # Simulates raw DICOM pixel_array passed through without normalisation —
    # exactly the bug class flagged in the review re: bit-depth handling.
    img = (np.random.uniform(0, 4095, size=(1024, 1024))).astype(np.uint16)
    result = validate_input(img, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert any("unexpected_dtype" in c for c in result.failed_checks)


def test_values_above_expected_range():
    img = _good_image()
    img[10, 10] = 255.0  # someone forgot to divide by 255 for this pixel
    result = validate_input(img, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert any("value_above_range" in c for c in result.failed_checks)


def test_values_below_expected_range():
    img = _good_image()
    img[10, 10] = -5.0
    result = validate_input(img, dict(GOOD_METADATA))
    assert result.is_valid is False
    assert any("value_below_range" in c for c in result.failed_checks)


# -----------------------------------------------------------------------
# Missing / malformed metadata
# -----------------------------------------------------------------------

def test_missing_metadata_entirely():
    result = validate_input(_good_image(), None)
    assert result.is_valid is False
    assert "metadata_is_none" in result.failed_checks


def test_missing_required_metadata_key():
    meta = dict(GOOD_METADATA)
    del meta["modality"]
    result = validate_input(_good_image(), meta)
    assert result.is_valid is False
    assert any("missing_metadata_key:modality" in c for c in result.failed_checks)


def test_empty_string_metadata_value():
    meta = dict(GOOD_METADATA)
    meta["study_uid"] = ""
    result = validate_input(_good_image(), meta)
    assert result.is_valid is False
    assert any("empty_metadata_value:study_uid" in c for c in result.failed_checks)


# -----------------------------------------------------------------------
# Multiple simultaneous failures — confirms the validator reports ALL
# failures, not just the first one (important for debugging upstream issues)
# -----------------------------------------------------------------------

def test_multiple_failures_all_reported():
    img = np.zeros((4, 4), dtype=np.float32)  # too small AND blank
    meta = dict(GOOD_METADATA)
    del meta["kvp"]  # also missing metadata
    result = validate_input(img, meta)
    assert result.is_valid is False
    assert len(result.failed_checks) >= 3
    assert any("image_too_small" in c for c in result.failed_checks)
    assert any("blank_or_flat_image" in c for c in result.failed_checks)
    assert any("missing_metadata_key:kvp" in c for c in result.failed_checks)


# -----------------------------------------------------------------------
# Result shape sanity
# -----------------------------------------------------------------------

def test_to_dict_serializes_cleanly():
    result = validate_input(np.zeros((4, 4), dtype=np.float32), None)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "is_valid" in d
    assert "failed_checks" in d
    assert isinstance(d["failed_checks"], list)