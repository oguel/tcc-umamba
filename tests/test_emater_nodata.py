"""Regressão: bordas Fora0_M0 não são verdadeiros negativos."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.emater_dataset import EMATERRGIDataset, RasterPair
from src.losses import BCEDiceLoss
from src.metrics import BinaryConfusion


def _write_pair(tmp_path, *, offset="-1000", descriptions=True):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    image_path = tmp_path / "image.tif"
    mask_path = tmp_path / "mask.tif"
    transform = from_origin(322850, 7654420, 10, 10)
    image = np.full((10, 8, 8), 1800, dtype=np.uint16)
    image[:, :, :4] = 0
    # 32 pixels faltantes têm 0 na máscara original, embora não sejam não-café.
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 4:] = 1
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=10,
        dtype="uint16",
        nodata=0,
        transform=transform,
        crs="EPSG:32723",
    ) as dst:
        dst.write(image)
        dst.update_tags(BOA_ADD_OFFSET=offset)
        if descriptions:
            dst.descriptions = (
                "B02", "B03", "B04", "B08", "B05",
                "B06", "B07", "B8A", "B11", "B12",
            )
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        nodata=255,
        transform=transform,
        crs="EPSG:32723",
    ) as dst:
        dst.write(mask, 1)
    return RasterPair(image_path, mask_path, "guaxupe_2022")


def test_pixels_sem_imagem_sao_ignorados(tmp_path):
    pair = _write_pair(tmp_path)
    image, mask = EMATERRGIDataset([pair])[0]
    assert image.shape == (4, 8, 8)
    assert torch.all(mask[:, :, :4] == 255)
    assert torch.all(mask[:, :, 4:] == 1)
    assert torch.all(image[:, :, :4] == 0)
    assert torch.all(image[:, :, 4:] > 0)
    logits = torch.zeros_like(mask)[None]
    confusion = BinaryConfusion()
    confusion.update(logits, mask[None])
    assert confusion.as_dict() == {"tp": 32, "fp": 0, "tn": 0, "fn": 0}
    loss = BCEDiceLoss()(logits, mask[None])
    assert torch.isfinite(loss)


def test_metadados_de_banda_deveriam_ser_verificados(tmp_path):
    pair = _write_pair(tmp_path, descriptions=False)
    with pytest.raises(ValueError, match="Ordem de bandas"):
        _ = EMATERRGIDataset([pair])[0]


def test_offset_ausente_ou_invalido_bloqueia_treino(tmp_path):
    pair = _write_pair(tmp_path, offset="invalido")
    with pytest.raises(ValueError, match="BOA_ADD_OFFSET"):
        _ = EMATERRGIDataset([pair])[0]
