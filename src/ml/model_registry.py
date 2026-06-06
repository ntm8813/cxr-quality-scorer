# src/ml/model_registry.py  — full replacement
from __future__ import annotations

import os
from typing import Any, Dict
import torch
import yaml
from monai.networks.nets import UNet
import timm


class ModelRegistry:
    """
    Manages and loads trained model checkpoints via versioned config references.
    Falls back to randomly initialized architecture if weights are missing.

    Supports:
      - MONAI U-Net          (lung segmentation)
      - EfficientNet-B0      (blur classifier)
      - EfficientNet-B0      (artifact classifier)
    """

    def __init__(self, config_path: str = "configs/model_versions.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.registry = yaml.safe_load(f)

    # ── Internal helpers ──────────────────────────────────────

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
    def _build_efficientnet_b0():
        return timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=1,
            in_chans=3,
        )

    @staticmethod
    def _sanitize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for key, value in state_dict.items():
            nk = key
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            cleaned[nk] = value
        return cleaned

    def _get_path(self, model_key: str) -> str:
        cfg = self.registry["models"][model_key]
        active = cfg["active_version"]
        return cfg["versions"][active]["path"]

    def _extract_state_dict(self, checkpoint: Any) -> Dict[str, Any]:
        """
        Handles common PyTorch checkpoint formats safely.
        """
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
        return checkpoint

    def _load_efficientnet(self, model_key: str, device: str = "cpu"):
        path = self._get_path(model_key)

        model = self._build_efficientnet_b0().to(device)

        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=device)
                sd = self._extract_state_dict(checkpoint)

                if isinstance(sd, dict):
                    sd = self._sanitize_state_dict(sd)
                    model.load_state_dict(sd, strict=True)

                print(f"[ModelRegistry] Loaded {model_key} from {path}")

            except Exception as e:
                print(f"[ModelRegistry] Warning loading {model_key}: {e}")

        else:
            print(f"[ModelRegistry] Warning: {path} not found — using random weights")

        model.eval()
        return model

    # ── Public loaders ────────────────────────────────────────

    def load_lung_segmentation(self, device: str = "cpu") -> UNet:
        """Existing loader — unchanged."""
        path = self._get_path("lung_segmentation")

        model = self._build_lung_unet().to(device)

        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=device)

                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]

                if isinstance(checkpoint, dict):
                    checkpoint = self._sanitize_state_dict(checkpoint)
                    model.load_state_dict(checkpoint, strict=False)

                print("[ModelRegistry] Loaded lung_segmentation")

            except Exception as e:
                print(f"[ModelRegistry] Warning loading lung_segmentation: {e}")

        model.eval()
        return model

    def load_blur_classifier(self, device: str = "cpu"):
        """Returns EfficientNet-B0 loaded with blur classifier weights."""
        return self._load_efficientnet("blur_classifier", device)

    def load_artifact_classifier(self, device: str = "cpu"):
        """Returns EfficientNet-B0 loaded with artifact classifier weights."""
        return self._load_efficientnet("artifact_classifier", device)