import numpy as np
import cv2

def apply_blur(image: np.ndarray, severity: int) -> np.ndarray:
    """Applies Gaussian motion blur. Severity: 0 (none), 1 (σ=5), 2 (σ=10)."""
    if severity == 0:
        return image.copy()
    
    sigma = 5 if severity == 1 else 10
    ksize = int(2 * round(3 * sigma) + 1)
    
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return np.clip(blurred, 0.0, 1.0)