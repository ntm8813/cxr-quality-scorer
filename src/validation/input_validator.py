# src/validation/input_validator.py
"""
Input validation / fail-safe layer for the CXR Quality Scorer pipeline.

For the current project phase, PNG images are the primary input.
PNG images naturally contain far less metadata than DICOM images, so the
validator adapts its metadata requirements accordingly.

PNG:
    Required metadata:
        - study_uid

DICOM:
    Required metadata:
        - study_uid
        - modality
        - view_position
        - body_part
        - exposure_index
        - deviation_index
        - kvp

All structural image validation remains identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


DICOM_REQUIRED_METADATA = [
    "study_uid",
    "modality",
    "view_position",
    "body_part",
    "exposure_index",
    "deviation_index",
    "kvp",
]

PNG_REQUIRED_METADATA = [
    "study_uid",
]

EXPECTED_DTYPE = np.float32

MIN_REASONABLE_DIM = 64
MAX_REASONABLE_DIM = 8192

EXPECTED_VALUE_MIN = 0.0
EXPECTED_VALUE_MAX = 1.0
VALUE_RANGE_TOLERANCE = 1e-3

BLANK_IMAGE_STD_THRESHOLD = 1e-4


@dataclass
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None
    failed_checks: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "is_valid": self.is_valid,
            "reason": self.reason,
            "failed_checks": self.failed_checks,
            "details": self.details,
        }


def _check_array_basic(image, expect_square=True):
    failures = []

    if image is None:
        failures.append("image_is_none")
        return failures

    if not isinstance(image, np.ndarray):
        failures.append(f"image_wrong_type:{type(image).__name__}")
        return failures

    if image.ndim not in (2, 3):
        failures.append(f"image_wrong_ndim:{image.ndim}")
        return failures

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        failures.append(f"image_unexpected_channels:{image.shape[2]}")

    h, w = image.shape[:2]

    if h < MIN_REASONABLE_DIM or w < MIN_REASONABLE_DIM:
        failures.append(f"image_too_small:{h}x{w}")

    if h > MAX_REASONABLE_DIM or w > MAX_REASONABLE_DIM:
        failures.append(f"image_too_large:{h}x{w}")

    if expect_square and h != w:
        failures.append(f"image_not_square:{h}x{w}")

    return failures


def _check_dtype(image):
    if image.dtype != EXPECTED_DTYPE:
        return [f"unexpected_dtype:{image.dtype}"]
    return []


def _check_value_range(image):
    failures = []

    if np.isnan(image).any():
        failures.append("contains_nan")

    if np.isinf(image).any():
        failures.append("contains_inf")

    finite = image[np.isfinite(image)]

    if finite.size == 0:
        failures.append("no_finite_pixels")
        return failures

    vmin = float(finite.min())
    vmax = float(finite.max())

    if vmin < EXPECTED_VALUE_MIN - VALUE_RANGE_TOLERANCE:
        failures.append(f"value_below_range:{vmin:.4f}")

    if vmax > EXPECTED_VALUE_MAX + VALUE_RANGE_TOLERANCE:
        failures.append(f"value_above_range:{vmax:.4f}")

    return failures


def _check_blank_image(image):
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return []

    std = float(finite.std())

    if std < BLANK_IMAGE_STD_THRESHOLD:
        return [f"blank_or_flat_image:std={std:.6f}"]

    return []


def _required_metadata(metadata):

    if metadata is None:
        return []

    # PNG loader supplies file_path.
    if "file_path" in metadata:
        return PNG_REQUIRED_METADATA

    # Otherwise assume DICOM.
    return DICOM_REQUIRED_METADATA


def _check_metadata(metadata):
    failures = []

    if metadata is None:
        failures.append("metadata_is_none")
        return failures

    required = _required_metadata(metadata)

    for key in required:

        if key not in metadata:
            failures.append(f"missing_metadata_key:{key}")

        elif metadata[key] is None or metadata[key] == "":
            failures.append(f"empty_metadata_value:{key}")

    return failures


def validate_input(
    image,
    metadata,
    expect_square=True,
):

    failed_checks = []

    basic = _check_array_basic(
        image,
        expect_square=expect_square,
    )

    failed_checks.extend(basic)

    fatal = any(
        f.startswith(
            (
                "image_is_none",
                "image_wrong_type",
                "image_wrong_ndim",
            )
        )
        for f in basic
    )

    if not fatal:
        failed_checks.extend(_check_dtype(image))
        failed_checks.extend(_check_value_range(image))
        failed_checks.extend(_check_blank_image(image))

    failed_checks.extend(_check_metadata(metadata))

    if failed_checks:
        return ValidationResult(
            is_valid=False,
            reason=f"{len(failed_checks)} validation check(s) failed",
            failed_checks=failed_checks,
            details={
                "image_shape": getattr(image, "shape", None),
                "image_dtype": str(getattr(image, "dtype", None)),
                "metadata_keys_present": list(metadata.keys()) if metadata else [],
            },
        )

    return ValidationResult(
        is_valid=True,
        reason=None,
        failed_checks=[],
    )