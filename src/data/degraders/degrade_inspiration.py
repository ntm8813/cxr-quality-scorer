import numpy as np
import cv2


def apply_inspiration(image: np.ndarray, severity: int) -> np.ndarray:
    """
    Simulates poor inspiratory effort by vertically compressing the lower lung
    zone (diaphragm raised) and replacing the vacated space with background.

    Severity 0: no change.
    Severity 1: lower 20% of image compressed to 12% height (borderline — 7-8 ribs).
    Severity 2: lower 35% compressed to 15% height (repeat — <7 ribs visible).

    This reduces the effective lung area ratio detected by the segmentation
    model, giving InspirationScorer a continuous, monotone signal.
    """
    if severity == 0:
        return image.copy()

    h, w = image.shape[:2]

    if severity == 1:
        compress_start = int(h * 0.80)   # bottom 20% is affected
        target_height  = int(h * 0.12)
    else:
        compress_start = int(h * 0.65)   # bottom 35% is affected
        target_height  = int(h * 0.15)

    lower_zone = image[compress_start:, :]

    # Compress the lower zone vertically
    compressed = cv2.resize(
        lower_zone,
        (w, target_height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Fill the freed-up space with background (mean intensity of the bottom strip)
    background_val = float(np.mean(image[h - 10:, :]))
    result = image.copy()
    fill_start = compress_start + target_height
    result[compress_start:fill_start, :] = compressed
    result[fill_start:, :] = background_val

    return np.clip(result.astype(np.float32), 0.0, 1.0)