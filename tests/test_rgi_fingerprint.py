"""Fingerprint nao depende do caminho absoluto nem da ordem dos patches."""

from pathlib import Path

import pytest

from src.data.emater_dataset import RasterPair
from src.data.rgi_fingerprint import dataset_fingerprint


def _pair(root: Path, city: str, name: str, data: bytes) -> RasterPair:
    image = root / city / "img" / name
    mask = root / city / "mask" / name
    image.parent.mkdir(parents=True, exist_ok=True)
    mask.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(data)
    mask.write_bytes(b"mask_" + data)
    return RasterPair(image, mask, city)


def test_fingerprint_estavel_mesmo_com_outros_caminhos(tmp_path):
    a = _pair(tmp_path / "first", "guaxupe_2022", "a.tif", b"imagem_a")
    b = _pair(tmp_path / "first", "juruaia_2022", "b.tif", b"imagem_b")
    a2 = _pair(tmp_path / "second", "guaxupe_2022", "a.tif", b"imagem_a")
    b2 = _pair(tmp_path / "second", "juruaia_2022", "b.tif", b"imagem_b")
    assert dataset_fingerprint([a, b]) == dataset_fingerprint([b2, a2])


def test_fingerprint_detecta_alteracao_de_um_pixel(tmp_path):
    a = _pair(tmp_path, "guaxupe_2022", "a.tif", b"imagem_a")
    initial = dataset_fingerprint([a])
    a.mask.write_bytes(b"mask_modificada")
    assert initial != dataset_fingerprint([a])


def test_fingerprint_bloqueia_duplicata_ou_dataset_vazio(tmp_path):
    a = _pair(tmp_path, "guaxupe_2022", "a.tif", b"imagem_a")
    with pytest.raises(ValueError, match="duplicado"):
        dataset_fingerprint([a, a])
    with pytest.raises(ValueError, match="vazio"):
        dataset_fingerprint([])
