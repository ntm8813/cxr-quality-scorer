import pytest
import h5py
import yaml
from src.scorers.sharpness_scorer import SharpnessScorer
from scipy.stats import spearmanr

@pytest.fixture
def config():
    with open("configs/v1.yaml") as f:
        return yaml.safe_load(f)

# Assert negative Spearman correlation between noise/blur index step adjustments and Laplacian variance[cite: 21].
