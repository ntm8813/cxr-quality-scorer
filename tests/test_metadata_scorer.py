import pytest
import numpy as np
import yaml
from src.scorers.metadata_scorer import MetadataScorer

@pytest.fixture
def config():
    with open("configs/v1.yaml") as f:
        return yaml.safe_load(f)

def test_metadata_all_present(config):
    scorer = MetadataScorer(config)
    valid_metadata = {
        "study_uid": "1.2.3", "modality": "DX", "view_position": "PA", 
        "body_part": "CHEST", "patient_id": "P12", "bits_stored": 12 
    }
    res = scorer.score(np.zeros((1024, 1024)), valid_metadata)
    assert res.flag == "acceptable" 

def test_metadata_empty_trigger_repeat(config):
    scorer = MetadataScorer(config)
    broken_metadata = {"study_uid": "", "modality": "", "view_position": ""} 
    res = scorer.score(np.zeros((1024, 1024)), broken_metadata)
    assert res.flag == "acceptable"
    assert res.score == 1.0