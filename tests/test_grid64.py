from pathlib import Path
import numpy as np

from detector.convert_to_grid64 import (
    CLASS_CROP,
    CLASS_SOIL,
    CLASS_WEED,
    convert_to_grid64,
    rgb_to_class_map,
)


def test_grid64_shape_dtype_and_classes():
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[64:, :64] = [0, 255, 0]
    rgb[64:, 64:] = [255, 0, 0]

    class_map = rgb_to_class_map(rgb)
    grid = convert_to_grid64(class_map, grid_size=64, weed_threshold=0.05)

    assert grid.shape == (64, 64)
    assert grid.dtype == np.uint8
    assert set(np.unique(grid).tolist()).issubset({CLASS_SOIL, CLASS_CROP, CLASS_WEED})
    assert CLASS_CROP in grid
    assert CLASS_WEED in grid


def test_saved_parcel_grids():
    npy_files = sorted(Path("data/grids_64").glob("*_64x64.npy"))
    assert len(npy_files) == 5

    for npy_path in npy_files:
        grid = np.load(npy_path)
        assert grid.shape == (64, 64)
        assert grid.dtype == np.uint8
        assert set(np.unique(grid).tolist()).issubset({CLASS_SOIL, CLASS_CROP, CLASS_WEED})