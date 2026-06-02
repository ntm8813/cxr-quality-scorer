import torch
import numpy as np
from monai.networks.nets import UNet
from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName

class CoverageScorer(BaseScorer):
    """Processes deep learning segmentation output matrices to flag anatomical truncation."""

    def __init__(self, config, model_path: str):
        super().__init__(config) 
        self.model = UNet(spatial_dims=2, in_channels=1, out_channels=1, 
                         channels=(16,32,64,128), strides=(2,2,2), num_res_units=2) 
        self.model.load_state_dict(torch.load(model_path, map_location="cpu")) 
        self.model.eval() 

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        margin_threshold = self.config["thresholds"]["coverage_margin_px"] 

        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float() 
        with torch.no_grad(): 
            mask = torch.sigmoid(self.model(tensor)).squeeze().numpy() 
        binary_mask = (mask > 0.5).astype(np.uint8) 

        rows = np.any(binary_mask, axis=1) 
        cols = np.any(binary_mask, axis=0) 

        if not rows.any(): 
            return AxisResult(
                study_uid=metadata.get("study_uid","unknown"), 
                axis=AxisName.COVERAGE, 
                score=0.0, flag="repeat", 
                raw_metrics={}, rationale="No lung field detected." 
            )

        top = int(np.argmax(rows)) 
        bottom = int(len(rows) - np.argmax(rows[::-1]) - 1) 
        left = int(np.argmax(cols)) 
        right = int(len(cols) - np.argmax(cols[::-1]) - 1) 

        h, w = binary_mask.shape 
        margins = {
            "top": top, "bottom": h - bottom, 
            "left": left, "right": w - right 
        }
        min_margin = min(margins.values()) 
        truncated_sides = [k for k, v in margins.items() if v < margin_threshold] 

        raw_score = 1.0 if not truncated_sides else max(0.0, min_margin / margin_threshold) 

        return AxisResult(
            study_uid=metadata.get("study_uid","unknown"), 
            axis=AxisName.COVERAGE, 
            score=float(raw_score), 
            flag=self._flag_from_score(raw_score), 
            raw_metrics={"margins_px": margins, "truncated_sides": truncated_sides}, 
            rationale=f"Min margin {min_margin}px. Truncated: {truncated_sides or 'none'}." 
        )