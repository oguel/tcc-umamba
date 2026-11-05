"""Pares de um mesmo patch podem estar em partes diferentes do download."""

from __future__ import annotations

import pytest

from src.data.emater_dataset import find_rgi_pairs


def _put(
    root,
    part,
    side,
    name="patch_y0_x0.tif",
    data=b"x",
    municipality="guaxupe_2022",
    variant="NN",
):
    folders = {
        "NN": "Patches_P512_S300_NN",
        "Fora": "Patches_P512_S300_NN_Fora0_M0",
    }
    leaf = {
        "img": "Imagens_Individuais",
        "mask": "Mascaras_Individuais",
    }
    path = (
        root / f"parte_{part}" / "Municipios" / municipality / folders[variant] / leaf[side] / name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_pares_reunidos_entre_partes_sem_mover_arquivos(tmp_path):
    img_a = _put(tmp_path, "001", "img", data=b"imagem_a")
    mask_a = _put(tmp_path, "002", "mask", data=b"mascara_a")
    img_b = _put(tmp_path, "002", "img", "patch_y0_x300.tif", b"imagem_b")
    mask_b = _put(tmp_path, "001", "mask", "patch_y0_x300.tif", b"mascara_b")
    _put(tmp_path, "001", "img", variant="Fora", data=b"outra")
    _put(tmp_path, "001", "img", municipality="tres_pontas_2024")
    _put(tmp_path, "002", "mask", municipality="tres_pontas_2024")

    pairs = find_rgi_pairs(tmp_path)
    assert [(p.image, p.mask) for p in pairs] == [
        (img_a, mask_a),
        (img_b, mask_b),
    ]
    assert len(find_rgi_pairs(tmp_path, include_external=True)) == 3


def test_uma_parte_so_deve_falhar(tmp_path):
    _put(tmp_path, "001", "img")
    with pytest.raises(ValueError, match="AMBAS"):
        find_rgi_pairs(tmp_path)


def test_duplicata_identica_so_conta_uma_vez(tmp_path):
    for part in ("001", "002"):
        _put(tmp_path, part, "img", data=b"igual")
        _put(tmp_path, part, "mask", data=b"igual")
    assert len(find_rgi_pairs(tmp_path)) == 1


def test_duplicata_divergente_e_proibida(tmp_path):
    _put(tmp_path, "001", "img", data=b"a")
    _put(tmp_path, "002", "img", data=b"b")
    _put(tmp_path, "001", "mask")
    with pytest.raises(ValueError, match="divergentes"):
        find_rgi_pairs(tmp_path)
