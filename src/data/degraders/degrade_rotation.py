import numpy as np
import cv2
import random

def apply_rotation(image: np.ndarray, severity: int) -> np.ndarray:
    """Simulates patient rotation/tilt. Severity: 0 (none), 1 (±8°), 2 (±15°)."""
    if severity == 0:
        return image.copy()
    
    angle = random.choice([-8, 8]) if severity == 1 else random.choice([-15, 15])
    h, w = image.shape
    center = (w // 2, h // 2)
    
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Use borderMode=BORDER_REPLICATE to avoid black triangles at edges
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return np.clip(rotated, 0.0, 1.0)