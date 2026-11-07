import numpy as np
import rasterio
import torch

from scripts.infer_new_sentinel import grid_profile, predict, read_asset, write_tiff


def test_scene_windows_preserve_band_order_and_invalid_pixels():
    class Model(torch.nn.Module):
        def forward(self, image):
            return image[:, 3:4] - image[:, 2:3]

    image = np.zeros((4, 256, 512), dtype=np.float32)
    image[3, :, :256] = 1
    image[2, :, 256:] = 1
    valid = np.ones((256, 512), dtype=bool)
    valid[0, 0] = False
    result = predict(Model(), image, valid, "cpu")
    assert result[0, 0] == -9999
    assert (result[:, :256][valid[:, :256]] > 0.5).all()
    assert (result[:, 256:] < 0.5).all()


def test_geotiff_preserves_grid_and_nodata(tmp_path):
    grid = grid_profile()
    grid.update(height=2, width=2)
    path = tmp_path / "classes.tif"
    values = np.array([[0, 1], [3, 255]], dtype=np.uint8)
    write_tiff(path, values, grid, 255)
    with rasterio.open(path) as image:
        assert image.transform == grid["transform"]
        assert image.crs.to_epsg() == 32723
        assert image.nodata == 255
        np.testing.assert_array_equal(image.read(1), values)


def test_rejects_nonpublic_asset():
    import pytest

    with pytest.raises(ValueError, match="HTTPS"):
        read_asset({"href": "s3://restricted/path"}, grid_profile())
