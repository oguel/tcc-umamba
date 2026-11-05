"""Testes sintéticos de segmentação regional e vazamento espacial."""
import pytest

from src.data.emater_dataset import RasterPair, validate_mask_values
from src.data.emater_spatial import spatial_folds


def test_mask_values():
    import numpy as np
    validate_mask_values(np.array([0, 1, 255], dtype="uint8"))
    with pytest.raises(ValueError):
        validate_mask_values(np.array([0, 2], dtype="uint8"))


def test_spatial_purge(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin

    pairs = []
    for i in range(12):
        path = tmp_path / f"patch_{i}.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=10, width=10, count=1, dtype="uint16",
            crs="EPSG:32723", transform=from_origin(i * 20000, 0, 10, 10),
        ) as dst:
            dst.write(np.zeros((1, 10, 10), dtype="uint16"))
        pairs.append(RasterPair(path, path, f"municipio_{i%2}"))

    folds = spatial_folds(pairs, k=3, block_m=10000)
    assert len(folds) == 3
    for train, val in folds:
        assert set(train).isdisjoint(val)
        assert train and val
        assert len(train) + len(val) <= len(pairs)
