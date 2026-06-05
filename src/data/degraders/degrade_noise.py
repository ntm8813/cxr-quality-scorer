from __future__ import annotations

import numpy as np

from src.scorers.quality_utils import stable_seed


def apply_noise(image: np.ndarray, severity: int, base_uid: str | None = None) -> np.ndarray:
    """
    Deterministic Gaussian noise.
    Severity:
        0 -> no-op
        1 -> mild
        2 -> heavy
    """
    img = np.asarray(image, dtype=np.float32)
    if severity == 0:
        return img.copy()

    std_dev = 0.025 if severity == 1 else 0.055
    rng = np.random.default_rng(stable_seed(base_uid, severity, "noise"))
    noise = rng.normal(0.0, std_dev, size=img.shape).astype(np.float32)

    noisy_image = img + noise
    return np.clip(noisy_image, 0.0, 1.0)