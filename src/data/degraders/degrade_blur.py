from __future__ import annotations

import numpy as np
import cv2


def apply_blur(image: np.ndarray, severity: int, base_uid: str | None = None) -> np.ndarray:
    """
    Applies calibrated Gaussian blur.
    Severity:
        0 -> no-op
        1 -> mild
        2 -> strong
    """
    if severity == 0:
        return np.asarray(image, dtype=np.float32).copy()

    img = np.asarray(image, dtype=np.float32)
    sigma = 1.35 if severity == 1 else 2.75
    ksize = int(2 * round(3 * sigma) + 1)
    ksize = max(3, ksize | 1)

    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred.astype(np.float32), 0.0, 1.0)