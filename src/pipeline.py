import yaml
import numpy as np
from src.io.dicom_reader import DICOMReader
from src.scorers.exposure_scorer import ExposureScorer
from src.scorers.sharpness_scorer import SharpnessScorer
from src.scorers.metadata_scorer import MetadataScorer
from src.scorers.rotation_scorer import RotationScorer
from src.scorers.coverage_scorer import CoverageScorer  # 1. IMPORT REAL COVERAGE SCORER
from src.fusion.score_fusion import ScoreFusion
from schemas.study_result import StudyResult

def run_pipeline(dicom_path: str, config_path: str = "configs/v1.yaml") -> StudyResult:
    with open(config_path) as f: 
        config = yaml.safe_load(f) 

    reader = DICOMReader() 
    image, metadata = reader.load(dicom_path) 

    # 2. INJECT INSTANTIATED MODEL COVERAGE SCORER WITH TARGET WEIGHTS PATH
    scorers = [
        ExposureScorer(config), 
        SharpnessScorer(config), 
        MetadataScorer(config), 
        RotationScorer(config),
        CoverageScorer(config, model_path="weights/best_lung_unet.pth")
    ]

    axis_results = [s.score(image, metadata) for s in scorers] 
    fusion = ScoreFusion(config_path) 
    return fusion.fuse(metadata.get("study_uid", "unknown"), axis_results)