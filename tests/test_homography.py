import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.homography import normalize_points


def test_normalize_points():
    points = np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 100.0],
        [0.0, 100.0]
    ])

    normalized, transform = normalize_points(points)

    assert normalized.shape == (4, 2)
    assert transform.shape == (3, 3)
    assert np.isfinite(normalized).all()
    assert np.isfinite(transform).all()