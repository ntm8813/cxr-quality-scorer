import numpy as np

class ImageFeatureCache:
    """
    Stores precomputed image features so scorers
    don't recompute histograms, percentiles, etc.
    """

    def __init__(self):
        self.cache = {}

    def get(self, key, compute_fn):
        if key in self.cache:
            return self.cache[key]
        value = compute_fn()
        self.cache[key] = value
        return value