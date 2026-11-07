from urllib.error import URLError

import numpy as np
import rasterio
import torch
from shapely.geometry import box

from scripts import infer_new_sentinel
from scripts.infer_new_sentinel import grid_mask, grid_profile, predict, read_asset, write_tiff


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
    grid = grid_profile(box(300000, 7600000, 300020, 7600020))
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
        read_asset({"href": "s3://restricted/path"}, {})


def test_grid_covers_whole_polygon_and_excludes_padding():
    region = box(300010, 7600010, 302600, 7602580)
    grid = grid_profile(region)
    assert grid["width"] == grid["height"] == 512
    assert grid_mask(region, grid).sum() == 259 * 257


def test_municipal_timeout_leaves_regional_mask_available(monkeypatch):
    def unavailable(*args, **kwargs):
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr(infer_new_sentinel, "fetch_json", unavailable)
    inside = np.array([[True, False]])
    labels, members, reason = infer_new_sentinel.optional_municipality_grid({}, inside)
    assert "timed out" in reason
    assert infer_new_sentinel.coverage_by_municipality(labels, members, inside) is None
    np.testing.assert_array_equal(inside, [[True, False]])


def test_mosaic_preserves_first_date_and_fills_other_municipality(monkeypatch):
    region = box(300000, 7600000, 300040, 7600040)
    grid = grid_profile(region)
    grid.update(height=4, width=4)
    inside = grid_mask(region, grid)
    labels = np.tile([1, 1, 2, 2], (4, 1)).astype(np.uint8)
    members = [{"id": 1, "nome": "Oeste"}, {"id": 2, "nome": "Leste"}]
    items = []
    for day in (1, 2):
        assets = {
            key: {
                "href": f"https://example.com/{day}/{key}",
                "day": day,
                "key": key,
                "raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}],
            }
            for key in (*infer_new_sentinel.BANDS, "scl")
        }
        items.append(
            {
                "id": str(day),
                "collection": infer_new_sentinel.COLLECTION,
                "properties": {"datetime": f"2024-07-0{day}T12:00:00Z"},
                "assets": assets,
            }
        )

    def read(asset, profile):
        values = np.full(
            (profile["height"], profile["width"]),
            4 if asset["key"] == "scl" else 1000 + asset["day"] * 1000,
            dtype=np.uint16,
        )
        if asset["key"] == "scl":
            half = profile["width"] // 2
            values[:, half:] = 0 if asset["day"] == 1 else 4
            values[:, :half] = 4 if asset["day"] == 1 else 0
        return values

    monkeypatch.setattr(infer_new_sentinel, "catalog_items", lambda _: items)
    monkeypatch.setattr(infer_new_sentinel, "read_asset", read)
    image, sources, records, _ = infer_new_sentinel.load_mosaic(
        grid, region, inside, labels, members
    )
    np.testing.assert_array_equal(sources, labels)
    np.testing.assert_allclose(image[:, :, :2], 0.1)
    np.testing.assert_allclose(image[:, :, 2:], 0.2)
    assert [record["pixels"] for record in records] == [8, 8]
