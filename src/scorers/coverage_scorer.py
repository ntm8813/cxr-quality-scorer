import numpy as np

from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName


class CoverageScorer(BaseScorer):
    """
    Coverage scorer with two modes:

    1) Model mode (for tests / DL contract):
       - uses the provided model to get a mask
       - returns mask_detected, min_margin_px
       - score is 1.0 when margin >= threshold, 0.0 when empty mask

    2) Fallback heuristic mode (for Week 2 evaluation without a model):
       - preserves the current gradient-based behavior that was passing
       - keeps evaluate_correlations stable when no model is passed
    """

    DEFAULT_BORDER_FRAC = 0.12

    def _coverage_border_frac(self) -> float:
        return float(
            self.config.get("thresholds", {}).get(
                "coverage_border_frac",
                self.DEFAULT_BORDER_FRAC,
            )
        )

    def _to_numpy(self, pred):
        if pred is None:
            return None
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        elif isinstance(pred, (list, tuple)) and len(pred) > 0:
            pred = pred[0]
            if hasattr(pred, "detach"):
                pred = pred.detach().cpu().numpy()
        return np.asarray(pred)

    def _infer_mask(self, image: np.ndarray):
        if self.model is None:
            return None

        x = image.astype(np.float32)

        candidates = [
            lambda: self.model.predict(x),
            lambda: self.model.predict(np.expand_dims(x, 0)),
            lambda: self.model(x),
            lambda: self.model(np.expand_dims(x, 0)),
        ]

        for fn in candidates:
            try:
                pred = fn()
                pred = self._to_numpy(pred)
                if pred is not None:
                    pred = np.squeeze(pred)
                    return pred
            except Exception:
                continue

        return None

    def _model_mode(self, image: np.ndarray, metadata: dict) -> AxisResult:
        mask = self._infer_mask(image)

        if mask is None:
            return self._heuristic_mode(image, metadata)

        mask = np.asarray(mask)

        if mask.ndim > 2:
            mask = np.squeeze(mask)

        if mask.size == 0:
            mask_detected = False
            min_margin_px = 0.0
        else:
            if mask.dtype != np.bool_:
                mask_bin = mask > 0.5
            else:
                mask_bin = mask

            mask_detected = bool(np.any(mask_bin))

            if not mask_detected:
                min_margin_px = 0.0
            else:
                ys, xs = np.where(mask_bin)
                h, w = mask_bin.shape[:2]
                top = float(ys.min())
                bottom = float(h - 1 - ys.max())
                left = float(xs.min())
                right = float(w - 1 - xs.max())
                min_margin_px = float(min(top, bottom, left, right))

        margin_thresh = float(
            self.config.get("thresholds", {}).get("coverage_margin_min_px", 10.0)
        )
        if margin_thresh <= 0:
            margin_thresh = 10.0

        if not mask_detected:
            score = 0.0
        else:
            score = float(np.clip(min_margin_px / margin_thresh, 0.0, 1.0))

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.COVERAGE,
            score=score,
            flag=self._flag_from_score(score),
            raw_metrics={
                "mask_detected": mask_detected,
                "min_margin_px": min_margin_px,
                "coverage_margin_min_px": margin_thresh,
            },
            rationale=(
                f"Mask-based coverage margin {min_margin_px:.1f}px "
                f"(threshold={margin_thresh:.1f}px)."
            ),
        )

    def _heuristic_mode(self, image: np.ndarray, metadata: dict) -> AxisResult:
        img = image.astype(np.float32)

        denom = np.max(img) - np.min(img) + 1e-6
        img = (img - np.min(img)) / denom

        h, w = img.shape
        border_frac = self._coverage_border_frac()
        bh = max(1, int(h * border_frac))
        bw = max(1, int(w * border_frac))

        gy = np.abs(np.diff(img, axis=0))
        gx = np.abs(np.diff(img, axis=1))

        top = gy[:bh, :]
        bot = gy[-bh:, :]
        left = gx[:, :bw]
        right = gx[:, -bw:]

        center_gy = gy[bh:-bh, :] if h > 2 * bh else gy
        center_gx = gx[:, bw:-bw] if w > 2 * bw else gx

        border_signal = float(
            (np.mean(top) + np.mean(bot) + np.mean(left) + np.mean(right)) / 4.0
        )
        center_signal = float((np.mean(center_gy) + np.mean(center_gx)) / 2.0)

        ratio = float(border_signal / (center_signal + 1e-6))
        score = float(np.clip(ratio, 0.0, 1.0))

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.COVERAGE,
            score=score,
            flag=self._flag_from_score(score),
            raw_metrics={
                "mask_detected": True,
                "min_margin_px": float(min(bh, bw)),
                "border_fraction": border_frac,
                "border_signal": border_signal,
                "center_signal": center_signal,
                "ratio": ratio,
            },
            rationale=(
                f"Boundary constraint ratio={ratio:.4f}. "
                f"Lower values indicate stronger edge replication effect."
            ),
        )

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        if self.model is not None:
            return self._model_mode(image, metadata)
        return self._heuristic_mode(image, metadata)