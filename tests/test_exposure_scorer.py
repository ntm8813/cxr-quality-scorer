import h5py
import yaml
import pytest
from src.scorers.exposure_scorer import ExposureScorer

@pytest.fixture
def config():
    with open("configs/v1.yaml") as f:
        return yaml.safe_load(f)

def test_exposure_scorer_ground_truth(config):
    scorer = ExposureScorer(config)
    
    # 1. Test against expected structural formats 
    clean_metadata = {"study_uid": "test_clean", "deviation_index": 0.0}
    mock_clean_img = h5py.File("data/processed/cxr_degraded.h5", "r")
    
    # Locate a sample key with severity 0 or feed mock clean matrix
    # Assert acceptable flag
