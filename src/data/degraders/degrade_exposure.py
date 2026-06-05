import hashlib
import numpy as np


def apply_exposure(image: np.ndarray, severity: int, base_uid: str = None) -> np.ndarray:
    """
    Simulates over- or under-penetration.

    Severity 0: no change.
    Severity 1: moderate shift (0.45x under or 2.0x over).
    Severity 2: severe shift (0.15x under or 3.5x over).

    Direction (under vs over) is deterministic per base_uid so that
    severity-1 and severity-2 variants of the same image are always
    on the same side, guaranteeing a monotone rank ordering for the scorer.
    """
    if severity == 0:
        return image.copy()

    # Deterministic direction: same image always goes the same way.
    if base_uid:
        direction = (int(hashlib.md5(str(base_uid).encode()).hexdigest(), 16) % 2 == 0)
    else:
        direction = True  # default: over-exposed

    if direction:
        # Over-exposure: multiply > 1 (more at higher severity)
        multiplier = 2.0 if severity == 1 else 3.5
    else:
        # Under-exposure: multiply < 1 (less at higher severity)
        multiplier = 0.45 if severity == 1 else 0.15

    exposed = image * multiplier
    return np.clip(exposed, 0.0, 1.0)