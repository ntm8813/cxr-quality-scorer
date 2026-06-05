import hashlib
import numpy as np
import cv2


# Applied angles per severity level (magnitude; direction is per-image)
ANGLES = {1: 8.0, 2: 15.0}


def get_applied_angle(severity: int, base_uid: str = None) -> float:
    """
    Return the signed angle that will be applied to this image.

    Direction is deterministic per base_uid (same as before).
    Exposing this as a standalone function lets run_degradations.py
    store the angle as an HDF5 attribute for ground-truth evaluation.
    """
    if severity == 0:
        return 0.0
    magnitude = ANGLES[severity]
    if base_uid:
        direction = 1 if int(hashlib.md5(str(base_uid).encode()).hexdigest(), 16) % 2 == 0 else -1
    else:
        direction = 1
    return magnitude * direction


def apply_rotation(image: np.ndarray, severity: int, base_uid: str = None) -> np.ndarray:
    """
    Simulates patient rotation.  Severity 0/1/2 → 0/±8/±15 degrees.
    Direction is deterministic per base_uid.
    """
    angle = get_applied_angle(severity, base_uid)
    if angle == 0.0:
        return image.copy()

    h, w  = image.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.clip(rotated, 0.0, 1.0)