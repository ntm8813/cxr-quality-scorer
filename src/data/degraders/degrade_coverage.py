import numpy as np


def apply_coverage(image: np.ndarray, severity: int) -> np.ndarray:
    """
    Simulates anatomical coverage loss by progressively cropping image edges
    and padding back to the original size with border replication.

    Severity 0: no change.
    Severity 1: crop 5% from each edge (borderline — apices/sulci slightly cut).
    Severity 2: crop 12% from each edge (repeat — costophrenic angles missing).

    The crop + repad pattern mirrors clinical coverage defects (patient
    positioned too close to the edge of the detector).
    """
    if severity == 0:
        return image.copy()

    h, w = image.shape[:2]
    crop_frac = 0.05 if severity == 1 else 0.12
    cy = int(h * crop_frac)
    cx = int(w * crop_frac)

    # Crop the interior
    cropped = image[cy: h - cy, cx: w - cx]

    # Replicate-pad back to original size
    padded = np.pad(
        cropped,
        ((cy, cy), (cx, cx)),
        mode="edge"
    )

    # Ensure original shape (handles rounding)
    padded = padded[:h, :w]
    return np.clip(padded.astype(np.float32), 0.0, 1.0)