from __future__ import annotations

import hashlib
import math
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def stable_seed(*parts: object) -> int:
    payload = "|".join("" if p is None else str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def as_float_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr.mean(axis=-1)

    if arr.size == 0:
        raise ValueError("Empty image array provided")

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    min_v = float(finite.min())
    max_v = float(finite.max())

    if min_v < -0.05 or max_v > 1.05:
        lo, hi = np.percentile(finite, [1.0, 99.0])
        if float(hi) > float(lo) + 1e-6:
            arr = (arr - float(lo)) / (float(hi) - float(lo))

    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    mask = np.ascontiguousarray(binary_mask.astype(np.uint8))
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    flood_inv = 1 - flood
    filled = np.maximum(mask, flood_inv)
    return filled.astype(np.uint8)


def keep_largest_components(binary_mask: np.ndarray, max_components: int = 1) -> np.ndarray:
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1 or max_components <= 0:
        return np.zeros_like(mask, dtype=np.uint8)

    areas = []
    for label in range(1, num_labels):
        areas.append((int(stats[label, cv2.CC_STAT_AREA]), label))

    areas.sort(reverse=True)
    keep = {label for _, label in areas[:max_components]}

    out = np.zeros_like(mask, dtype=np.uint8)
    for label in keep:
        out[labels == label] = 1
    return out


def morphological_cleanup(binary_mask: np.ndarray, keep_components: int = 1) -> np.ndarray:
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    if mask.size == 0:
        return mask

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = _fill_holes(mask)
    mask = keep_largest_components(mask, max_components=keep_components)
    return mask.astype(np.uint8)


def anatomical_foreground_mask(image: np.ndarray) -> np.ndarray:
    img = as_float_image(image)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    u8 = (blur * 255.0).astype(np.uint8)

    _, otsu = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = otsu > 0

    ratio = float(mask.mean())
    if ratio < 0.05 or ratio > 0.98:
        threshold = max(0.04, float(np.percentile(blur, 20.0)))
        mask = blur > threshold

    ratio = float(mask.mean())
    if ratio < 0.05:
        threshold = max(0.02, float(np.percentile(blur, 10.0)))
        mask = blur > threshold

    mask = morphological_cleanup(mask, keep_components=1)

    if float(mask.mean()) < 0.02:
        mask = np.ones_like(img, dtype=np.uint8)

    return mask.astype(bool)


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    pad_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    img = as_float_image(image)
    m = np.asarray(mask).astype(bool)

    if m.shape != img.shape:
        raise ValueError("Mask and image must have the same shape")

    ys, xs = np.where(m)
    if xs.size == 0 or ys.size == 0:
        full_mask = np.ones_like(img, dtype=bool)
        return img.copy(), full_mask

    h, w = img.shape[:2]
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    pad_x = int(max(8, round((x1 - x0) * pad_ratio)))
    pad_y = int(max(8, round((y1 - y0) * pad_ratio)))

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)

    cropped_image = img[y0:y1, x0:x1].copy()
    cropped_mask = m[y0:y1, x0:x1].copy()
    return cropped_image, cropped_mask


def normalized_hist_entropy(x, bins=64):
    import numpy as np

    x = np.asarray(x, dtype=np.float32)

    # faster histogram computation
    hist, _ = np.histogram(x, bins=bins, range=(0.0, 1.0), density=True)

    hist = hist + 1e-8  # avoid log(0)

    return float(-np.sum(hist * np.log(hist)))


def high_frequency_energy_ratio(image: np.ndarray, radius_fraction: float = 0.22) -> float:
    img = as_float_image(image)
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return 0.0

    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    centered = img * window
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(spectrum) ** 2
    total = float(power.sum())
    if total <= 0.0:
        return 0.0

    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = np.sqrt(cy**2 + cx**2)
    normalized_radius = radius / max_radius

    high_mask = normalized_radius >= float(radius_fraction)
    ratio = float(power[high_mask].sum() / total)
    return float(np.clip(ratio, 0.0, 1.0))


def weighted_orientation_deg(image: np.ndarray) -> Tuple[float, float, int]:
    """
    Returns:
        signed_angle_deg: dominant axis in degrees, normalized to [-90, 90)
        confidence: eigenvalue separation confidence in [0, 1]
        foreground_count: number of weighted pixels used
    """
    img = as_float_image(image)
    u8 = (img * 255.0).astype(np.uint8)
    blurred = cv2.GaussianBlur(u8, (5, 5), 0)
    eq = cv2.equalizeHist(blurred)

    gx = cv2.Sobel(eq, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    positive = mag[mag > 0]
    if positive.size == 0:
        return 0.0, 0.0, 0

    thresh = float(np.percentile(positive, 75.0))
    ys, xs = np.where(mag >= thresh)
    weights = mag[ys, xs]

    if xs.size < 120:
        edges = cv2.Canny(eq, 30, 90)
        ys, xs = np.where(edges > 0)
        weights = np.ones(xs.size, dtype=np.float64)

    if xs.size < 40:
        return 0.0, 0.0, int(xs.size)

    coords = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-6)

    center = np.average(coords, axis=0, weights=weights)
    coords = coords - center

    cov = np.cov(coords, rowvar=False, aweights=weights)
    if not np.all(np.isfinite(cov)):
        return 0.0, 0.0, int(xs.size)

    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_vec = eigvecs[:, int(np.argmax(eigvals))]

    signed_angle_deg = float(np.degrees(np.arctan2(principal_vec[1], principal_vec[0])))
    signed_angle_deg = ((signed_angle_deg + 90.0) % 180.0) - 90.0

    ev_sorted = np.sort(np.asarray(eigvals, dtype=np.float64))
    if ev_sorted[-1] <= 1e-8:
        confidence = 0.0
    else:
        confidence = float(np.clip((ev_sorted[-1] - ev_sorted[0]) / ev_sorted[-1], 0.0, 1.0))

    return signed_angle_deg, confidence, int(xs.size)
# src/scorers/quality_utils.py
# ADD this function anywhere after weighted_orientation_deg() in the existing file.
# Do not delete or modify any existing function — only append this one.

def mask_anchored_orientation_deg(
    image: np.ndarray,
    lung_mask: np.ndarray,
) -> Tuple[float, float, int]:
    """
    Estimates patient rotation using the LINE CONNECTING THE TWO LUNG
    CENTROIDS, not the mask's overall PCA principal axis.

    Why this replaced the original PCA-based version: the original
    function assumed a correctly positioned patient's combined lung
    mask is VERTICALLY elongated. This is false. Two separate lung
    fields side-by-side with a mediastinal gap between them produce a
    mask that is roughly as WIDE as it is TALL (confirmed on real U-Net
    output: bbox aspect ratios of 0.66-1.30 across multiple real CXRs,
    not the >1.5 vertical elongation the PCA approach required). PCA on
    that shape locks onto a near-horizontal axis regardless of actual
    patient rotation, producing 70-85 degree "rotation" on visually
    upright images.

    The correct anatomical invariant: in an upright, unrotated patient,
    the LEFT lung centroid and RIGHT lung centroid sit at the same
    vertical height. The line between them is horizontal. Patient
    rotation tilts this line. This is robust to the mediastinal gap and
    to left/right lung volume asymmetry, because it only depends on
    each lung's center of mass, not the combined mask's overall shape.

    Returns:
        signed_angle_deg : tilt of the inter-centroid line from
                            horizontal, in degrees. 0 = upright,
                            range approximately [-90, 90].
        confidence       : based on left/right size balance and
                            horizontal separation. Low confidence means
                            the two components are too unequal in size
                            or too close together to trust the angle.
        foreground_count : total mask pixels used.
    """
    mask = np.asarray(lung_mask).astype(bool).astype(np.uint8)
    total_fg = int(mask.sum())

    if total_fg < 50:
        return 0.0, 0.0, total_fg

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # label 0 is background; collect real components
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        components.append((area, cx, cy))

    if len(components) < 2:
        # Only one connected blob (lungs touching/merged, or segmentation
        # didn't separate them) -- no reliable left/right pair available.
        return 0.0, 0.0, total_fg

    # Keep the two largest components -- these should be left and right lung
    components.sort(key=lambda c: c[0], reverse=True)
    (area_a, cx_a, cy_a), (area_b, cx_b, cy_b) = components[0], components[1]

    # Order them left-to-right by x position for a consistent sign convention
    if cx_a <= cx_b:
        left_cx, left_cy = cx_a, cy_a
        right_cx, right_cy = cx_b, cy_b
        left_area, right_area = area_a, area_b
    else:
        left_cx, left_cy = cx_b, cy_b
        right_cx, right_cy = cx_a, cy_a
        left_area, right_area = area_b, area_a

    dx = right_cx - left_cx
    dy = right_cy - left_cy

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0, 0.0, total_fg

    # Angle of the inter-centroid line from horizontal.
    # 0 = perfectly horizontal (upright patient).
    # Positive = right lung centroid lower than left (clockwise tilt).
    signed_angle_deg = float(np.degrees(np.arctan2(dy, dx)))

    # Normalize to [-90, 90] -- a line and its reverse direction are
    # the same line, so wrap into a single sign-consistent range.
    if signed_angle_deg > 90.0:
        signed_angle_deg -= 180.0
    elif signed_angle_deg < -90.0:
        signed_angle_deg += 180.0

    # ── Confidence ──────────────────────────────────────────────────
    # Two factors reduce trust in this angle:
    #  1. Size imbalance between left/right components (one might be
    #     a segmentation artifact, not a real lung)
    #  2. The two centroids being too close together horizontally
    #     (small dx means a tiny rotation in pixel-noise terms swings
    #     the angle wildly -- division-by-small-number instability)
    size_ratio = min(left_area, right_area) / max(left_area, right_area, 1)

    img_w = image.shape[1] if image.ndim >= 2 else 1024
    separation_frac = float(np.clip(abs(dx) / max(img_w * 0.15, 1.0), 0.0, 1.0))

    confidence = float(np.clip(size_ratio * separation_frac, 0.0, 1.0))

    return signed_angle_deg, confidence, total_fg


def border_contact_fraction(binary_mask: np.ndarray) -> float:
    mask = np.asarray(binary_mask).astype(bool)
    if mask.size == 0:
        return 0.0

    top = mask[0, :]
    bottom = mask[-1, :]
    left = mask[:, 0]
    right = mask[:, -1]

    border = np.concatenate([top, bottom, left, right])
    return float(np.mean(border))


def mask_bbox(binary_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    mask = np.asarray(binary_mask).astype(bool)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return x0, y0, x1, y1


def _model_device(model) -> torch.device:
    try:
        param = next(model.parameters())
        return param.device
    except Exception:
        return torch.device("cpu")


def heuristic_lung_mask(image: np.ndarray) -> np.ndarray:
    """
    Conservative fallback:
    - derive body foreground
    - invert intensity to emphasize lung-like dark regions
    - keep two largest connected components inside the body
    """
    img = as_float_image(image)
    body = anatomical_foreground_mask(img)

    if body.sum() <= 0:
        body = np.ones_like(img, dtype=bool)

    inv = 1.0 - img
    body_values = inv[body]
    if body_values.size == 0:
        body_values = inv.ravel()

    threshold = float(np.percentile(body_values, 70.0))
    threshold = max(0.35, threshold)

    candidate = (inv >= threshold) & body
    candidate = morphological_cleanup(candidate, keep_components=2)

    if float(candidate.mean()) < 0.005:
        threshold = float(np.percentile(body_values, 60.0))
        candidate = (inv >= threshold) & body
        candidate = morphological_cleanup(candidate, keep_components=2)

    if float(candidate.mean()) < 0.005:
        threshold = float(np.percentile(inv, 65.0))
        candidate = inv >= threshold
        candidate = morphological_cleanup(candidate, keep_components=2)

    return candidate.astype(bool)


def infer_lung_mask(model, image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Returns:
        mask: binary lung mask
        detected: True if the model produced a non-empty, usable mask
    """
    img = as_float_image(image)

    if model is None:
        return heuristic_lung_mask(img), False

    try:
        device = _model_device(model)
        tensor = torch.from_numpy(img[None, None, ...]).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return heuristic_lung_mask(img), False
            probs = torch.sigmoid(output).detach().cpu().numpy()

        probs = np.squeeze(probs)
        if probs.ndim != 2:
            return heuristic_lung_mask(img), False

        peak = float(np.max(probs)) if probs.size else 0.0
        binary = probs > 0.5
        binary = morphological_cleanup(binary, keep_components=2)

        area = float(binary.mean())
        if peak < 0.05 or area < 0.001:
            return np.zeros_like(binary, dtype=bool), False

        if area > 0.90 or area < 0.01:
            return heuristic_lung_mask(img), True

        border = border_contact_fraction(binary)
        if border > 0.60 and area > 0.70:
            return heuristic_lung_mask(img), True

        return binary.astype(bool), True

    except Exception:
        return heuristic_lung_mask(img), False