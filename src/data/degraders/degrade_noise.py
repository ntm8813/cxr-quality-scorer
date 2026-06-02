import numpy as np

def apply_noise(image: np.ndarray, severity: int) -> np.ndarray:
    """Applies Gaussian noise simulating low dose/SNR. Severity: 0 (none), 1 (mild), 2 (heavy)."""
    if severity == 0:
        return image.copy()
    
    # SNR roughly mapped to standard deviation
    std_dev = 0.05 if severity == 1 else 0.15
    noise = np.random.normal(0, std_dev, image.shape).astype(np.float32)
    
    noisy_image = image + noise
    return np.clip(noisy_image, 0.0, 1.0)