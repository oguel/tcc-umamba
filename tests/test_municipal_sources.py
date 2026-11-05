"""A conferência da janela deve detectar pixels ou grids alterados."""

import numpy as np
import pytest

from scripts.audit_municipal_sources import compare_window


def test_source_window_checks_pixels_and_grid(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    transform = from_origin(342130, 7635470, 10, 10)
    data = np.arange(64, dtype=np.uint16).reshape(1, 8, 8)
    profile = {
        "driver": "GTiff",
        "width": 8,
        "height": 8,
        "count": 1,
        "dtype": "uint16",
        "transform": transform,
        "crs": "EPSG:32723",
    }
    source = tmp_path / "source.tif"
    patch = tmp_path / "patch.tif"
    with rasterio.open(source, "w", **profile) as dst:
        dst.write(data)
    profile.update(width=4, height=4, transform=from_origin(342150, 7635450, 10, 10))
    with rasterio.open(patch, "w", **profile) as dst:
        dst.write(data[:, 2:6, 2:6])
    with rasterio.open(source) as src, rasterio.open(patch) as crop:
        assert compare_window(src, crop, 2, 2)
        assert not compare_window(src, crop, 1, 2)
        assert not compare_window(src, crop, 7, 7)
    with rasterio.open(patch, "r+") as dst:
        dst.write(np.zeros((1, 4, 4), dtype=np.uint16))
    with rasterio.open(source) as src, rasterio.open(patch) as crop:
        assert not compare_window(src, crop, 2, 2)
