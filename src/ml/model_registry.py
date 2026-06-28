# src/ml/model_registry.py
"""
Loads and caches all model checkpoints used by the pipeline:
  - lung segmentation U-Net
  - blur classifier
  - artifact classifier

All loaders are defensive: if a checkpoint fails to load (wrong path,
wrong architecture, key mismatch), they raise loudly rather than
silently running an untrained model. This was added after a real
production bug where strict=False masked a complete key-prefix
mismatch (0/49 keys loaded) and the U-Net ran fully randomly
initialized for an entire validation cycle with no error shown.
"""

import os
import yaml
import torch
import timm
from monai.networks.nets import UNet


class ModelRegistry:
    def __init__(self, config_path: str = "configs/model_versions.yaml"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    # ── path lookup ──────────────────────────────────────────────

    def _get_path(self, model_key: str) -> str:
            model_entry = self.config["models"][model_key]
            active_version = model_entry["active_version"]
            return model_entry["versions"][active_version]["path"]

    # ── architecture builders ────────────────────────────────────

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
    def _build_classifier(num_classes: int = 1):
        return timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=num_classes,
        )

    # ── checkpoint key normalization ─────────────────────────────

    @staticmethod
    def _sanitize_state_dict(state_dict: dict, reference_keys: set = None) -> dict:
        """
        Normalizes state dict key prefixes so checkpoints saved with or
        without a wrapping prefix (module., model.) load correctly.

        Handles three cases:
          1. Checkpoint has 'module.' prefix (DataParallel) -> strip it
          2. Checkpoint has 'model.' prefix already -> leave as-is
          3. Checkpoint has NO prefix but model expects 'model.' -> add it

        Without case 3, a checkpoint saved via
        torch.save(raw_model.state_dict()) against a bare nn.Module
        silently fails to load into a wrapper that expects a 'model.'
        prefix (e.g. MONAI's UNet wraps its Sequential in a top-level
        'model' attribute) -- strict=False then loads ZERO keys with
        no error, leaving the network at random initialization.
        """
        keys = list(state_dict.keys())

        # Strip DataParallel wrapper prefix if present
        if any(k.startswith("module.") for k in keys):
            state_dict = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            keys = list(state_dict.keys())

        if reference_keys is not None:
            has_model_prefix_in_ckpt = any(k.startswith("model.") for k in keys)
            has_model_prefix_expected = any(
                k.startswith("model.") for k in reference_keys
            )

            if has_model_prefix_expected and not has_model_prefix_in_ckpt:
                state_dict = {f"model.{k}": v for k, v in state_dict.items()}
            elif has_model_prefix_in_ckpt and not has_model_prefix_expected:
                state_dict = {
                    (k[len("model."):] if k.startswith("model.") else k): v
                    for k, v in state_dict.items()
                }

        return state_dict

    # ── loaders ───────────────────────────────────────────────────

    def load_lung_segmentation(self, device: str = "cpu") -> UNet:
        path = self._get_path("lung_segmentation")
        model = self._build_lung_unet().to(device)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ModelRegistry] lung_segmentation checkpoint not found: {path}"
            )

        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        reference_keys = set(model.state_dict().keys())
        checkpoint = self._sanitize_state_dict(checkpoint, reference_keys=reference_keys)

        missing, unexpected = model.load_state_dict(checkpoint, strict=False)

        if len(missing) == len(reference_keys):
            raise RuntimeError(
                "[ModelRegistry] lung_segmentation: ALL keys failed to load "
                f"({len(missing)}/{len(reference_keys)}). Checkpoint architecture "
                "does not match model. Refusing to silently run an untrained model."
            )

        if missing or unexpected:
            print(
                f"[ModelRegistry] WARNING lung_segmentation: "
                f"{len(missing)} missing keys, {len(unexpected)} unexpected keys. "
                f"Model may be partially untrained."
            )
        else:
            print("[ModelRegistry] Loaded lung_segmentation (all keys matched)")

        model.eval()
        return model

    def load_blur_classifier(self, device: str = "cpu"):
        path = self._get_path("blur_classifier")
        model = self._build_classifier(num_classes=1).to(device)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ModelRegistry] blur_classifier checkpoint not found: {path}"
            )

        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        reference_keys = set(model.state_dict().keys())
        checkpoint = self._sanitize_state_dict(checkpoint, reference_keys=reference_keys)

        missing, unexpected = model.load_state_dict(checkpoint, strict=False)

        if len(missing) == len(reference_keys):
            raise RuntimeError(
                "[ModelRegistry] blur_classifier: ALL keys failed to load "
                f"({len(missing)}/{len(reference_keys)}). Checkpoint architecture "
                "does not match model."
            )

        if missing or unexpected:
            print(
                f"[ModelRegistry] WARNING blur_classifier: "
                f"{len(missing)} missing keys, {len(unexpected)} unexpected keys."
            )
        else:
            print(f"[ModelRegistry] Loaded blur_classifier from {path}")

        model.eval()
        return model

    def load_artifact_classifier(self, device: str = "cpu"):
        path = self._get_path("artifact_classifier")
        model = self._build_classifier(num_classes=1).to(device)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ModelRegistry] artifact_classifier checkpoint not found: {path}"
            )

        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        reference_keys = set(model.state_dict().keys())
        checkpoint = self._sanitize_state_dict(checkpoint, reference_keys=reference_keys)

        missing, unexpected = model.load_state_dict(checkpoint, strict=False)

        if len(missing) == len(reference_keys):
            raise RuntimeError(
                "[ModelRegistry] artifact_classifier: ALL keys failed to load "
                f"({len(missing)}/{len(reference_keys)}). Checkpoint architecture "
                "does not match model."
            )

        if missing or unexpected:
            print(
                f"[ModelRegistry] WARNING artifact_classifier: "
                f"{len(missing)} missing keys, {len(unexpected)} unexpected keys."
            )
        else:
            print(f"[ModelRegistry] Loaded artifact_classifier from {path}")

        model.eval()
        return model