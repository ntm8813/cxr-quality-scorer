import yaml
import numpy as np
from PIL import Image

from src.ml.model_registry import ModelRegistry
from src.scorers.exposure_scorer import ExposureScorer
from src.scorers.sharpness_scorer import SharpnessScorer
from src.scorers.metadata_scorer import MetadataScorer
from src.scorers.rotation_scorer import RotationScorer
from src.scorers.coverage_scorer import CoverageScorer
from src.scorers.inspiration_scorer import InspirationScorer
from src.fusion.score_fusion import ScoreFusion
from schemas.study_result import StudyResult


def load_image_any(path: str):

    # PNG / JPG / JPEG support (YOUR DATASET)
    if path.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(path).convert("L")
        img = img.resize((256, 256))
        image = np.array(img, dtype=np.float32) / 255.0

        metadata = {
            "study_uid": path.split("\\")[-1].split("/")[-1],
            "file_path": path
        }

        return image, metadata

    # DICOM fallback (optional future support)
    if path.lower().endswith(".dcm"):
        from src.io.dicom_reader import DICOMReader
        return DICOMReader().load(path)

    raise ValueError(f"Unsupported file format: {path}")


def run_pipeline(image_path: str, config_path: str = "configs/v1.yaml") -> StudyResult:

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # STEP 1: LOAD IMAGE (FIXED)
    image, metadata = load_image_any(image_path)

    # STEP 2: MODEL
    registry = ModelRegistry()
    unet_model = registry.load_lung_segmentation()

    # STEP 3: SCORERS
    scorers = [
        ExposureScorer(config),
        SharpnessScorer(config),
        MetadataScorer(config),
        RotationScorer(config),
        CoverageScorer(config, model=unet_model),
        InspirationScorer(config, model=unet_model)
    ]

    axis_results = []
    for s in scorers:
        try:
            axis_results.append(s.score(image, metadata))
        except Exception as e:
            # prevents single scorer crash from killing pipeline
            print(f"[SCORER ERROR] {s.__class__.__name__}: {e}")

    # STEP 4: FUSION
    fusion = ScoreFusion(config_path)
    return fusion.fuse(metadata.get("study_uid", "unknown"), axis_results)