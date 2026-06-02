import numpy as np
import random

def apply_exposure(image: np.ndarray, severity: int) -> np.ndarray:
    """Simulates over/under penetration. Severity: 0 (none), 1 (borderline), 2 (repeat)."""
    if severity == 0:
        return image.copy()
    
    # Randomly choose between under-exposed (multiplier < 1) or over-exposed (multiplier > 1)
    if severity == 1:
        multiplier = random.choice([0.5, 1.8])
    else: # severity == 2
        multiplier = random.choice([0.1, 3.5])
        
    exposed = image * multiplier
    return np.clip(exposed, 0.0, 1.0)