"""
harness/preprocessing.py -- feature standardization for the neural models.

Raw MSLR features live on wildly different scales (click counts in the
thousands next to ratios in [0, 1]); an unscaled network usually fails to
train. Fit on TRAIN rows only and reuse the same statistics everywhere --
including inside each cross-validation fold (fit on that fold's training
part only; fitting on all the data first leaks the validation fold).
"""
from typing import Dict

import numpy as np


class Standardizer:
    def __init__(self, mean: np.ndarray = None, std: np.ndarray = None):
        self.mean = mean
        self.std = std

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0).astype(np.float64)
        std = X.std(axis=0).astype(np.float64)
        std[std < 1e-8] = 1.0  # constant columns: leave centred, don't divide by ~0
        self.std = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Standardizer.fit() has not been called")
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "Standardizer":
        return cls(np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64))
