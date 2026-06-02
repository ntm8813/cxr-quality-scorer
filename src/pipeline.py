# src/pipeline.py
import yaml
from src.io.dicom_reader import DICOMReader
from src.ml.model_registry import ModelRegistry
from src.scorers.exposure_scorer import ExposureScorer
from src.scorers.sharpness_scorer import SharpnessScorer
from src.scorers.metadata_scorer import MetadataScorer
from src.scorers.rotation_scorer import RotationScorer
from src.scorers.coverage_scorer import CoverageScorer
from src.scorers.inspiration_scorer import InspirationScorer
from src.fusion.score_fusion import ScoreFusion
from schemas.study_result import StudyResult

def run_pipeline(dicom_path: str, config_path: str = "configs/v1.yaml") -> StudyResult:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Standardize and load incoming image array
    reader = DICOMReader()
    image, metadata = reader.load(dicom_path)

    # 2. Initialize model instance via Registry to prevent multiple loading loops
    registry = ModelRegistry()
    unet_model = registry.load_lung_segmentation()

    # 3. Orchester scoring execution matrix
    scorers = [
        ExposureScorer(config),
        SharpnessScorer(config),
        MetadataScorer(config),
        RotationScorer(config),
        CoverageScorer(config, model=unet_model),
        InspirationScorer(config, model=unet_model)
    ]

    axis_results = [s.score(image, metadata) for s in scorers]
    
    # 4. Fuse scores into final schema payload
    fusion = ScoreFusion(config_path)
    return fusion.fuse(metadata.get("study_uid", "unknown"), axis_results)