from __future__ import annotations

import os
from typing import Any, Dict

import torch
import yaml
from monai.networks.nets import UNet


class ModelRegistry:
    """
    Manages and loads trained model checkpoints via versioned config references.
    Falls back to a valid randomly initialized architecture if weights are missing.
    """

    def __init__(self, config_path: str = "configs/model_versions.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.registry = yaml.safe_load(f)

    @staticmethod
    def _build_lung_unet() -> UNet:
        return UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
        )

    @staticmethod
    def _sanitize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            cleaned[new_key] = value
        return cleaned

    def load_lung_segmentation(self, device: str = "cpu") -> UNet:
        config = self.registry["models"]["lung_segmentation"]
        active = config["active_version"]
        model_path = config["versions"][active]["path"]

        model = self._build_lung_unet().to(device)

        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                if isinstance(checkpoint, dict):
                    checkpoint = self._sanitize_state_dict(checkpoint)
                    model.load_state_dict(checkpoint, strict=False)
            except Exception:
                pass

        model.eval()
        return model