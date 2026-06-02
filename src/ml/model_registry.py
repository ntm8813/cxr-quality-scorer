# src/ml/model_registry.py
import yaml
import torch
from monai.networks.nets import UNet

class ModelRegistry:
    """Manages and loads trained model checkpoints via versioned config references."""
    
    def __init__(self, config_path: str = "configs/model_versions.yaml"):
        with open(config_path, "r") as f:
            self.registry = yaml.safe_load(f)

    def load_lung_segmentation(self, device: str = "cpu") -> UNet:
        config = self.registry["models"]["lung_segmentation"]
        active = config["active_version"]
        model_path = config["versions"][active]["path"]
        
        # Instantiate architectural parameters matching your trained weights
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model