def test_project_imports():
    """Verify core libraries are importable."""
    import pydicom
    import numpy as np
    import pydantic
    assert True

def test_config_exists():
    """Verify config file is present."""
    import os
    assert os.path.exists("configs/v1.yaml"), "v1.yaml config file missing"