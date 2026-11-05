"""Os downloads 001/002 dividem pares de um mesmo patch entre diretórios."""

from __future__ import annotations

import pytest

from src.data.emater_dataset import find_rgi_pairs


def _file(root, part, municipality, variant, role, filename, value=b"x"):
    path = (
        root
        / part
        / "Municipios"
        / municipality
        / variant
        / role
        / filename
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def test_pares_sao_reunidos_entre_partes_sem_mover_arquivos(tmp_path):
    variant = "Patches_P512_S300_NN"
    img_a = _file(
        tmp_path, "parte_001", "guaxupe_2022", variant,
        "Imagens_Individuais", "patch_y0_x0.tif", b"imagem_a",
    )
    mask_a = _file(
        tmp_path, "parte_002", "guaxupe_2022", variant,
        "Mascaras_Individuais", "patch_y0_x0.tif", b"mascara_a",
    )
    img_b = _file(
        tmp_path, "parte_002", "guaxupe_2022", variant,
        "Imagens_Individuais", "patch_y0_x300.tif", b"imagem_b",
    )
    mask_b = _file(
        tmp_path, "parte_001", "guaxupe_2022", variant,
        "Mascaras_Individuais", "patch_y0_x300.tif", b"mascara_b",
    )
    # A variante alternativa NÃO constitui outra amostra.
    _file(
        tmp_path, "parte_001", "guaxupe_2022",
        "Patches_P512_S300_NN_Fora0_M0",
        "Imagens_Individuais", "patch_y0_x0.tif", b"variante_outra",
    )
    # Três Pontas é reservada para teste externo.
    _file(
        tmp_path, "parte_001", "tres_pontas_2024", variant,
        "Imagens_Individuais", "patch_y0_x0.tif", b"externo",
    )
    _file(
        tmp_path, "parte_002", "tres_pontas_2024", variant,
        "Mascaras_Individuais", "patch_y0_x0.tif", b"rotulo_externo",
    )

    pairs = find_rgi_pairs(tmp_path)
    assert len(pairs) == 2
    assert [(p.image, p.mask) for p in pairs] == [
        (img_a, mask_a),
        (img_b, mask_b),
    ]
    assert len(find_rgi_pairs(tmp_path, include_external=True)) == 3


def test_exige_as_duas_partes_quando_par_ainda_esta_ausente(tmp_path):
    _file(
        tmp_path, "parte_001", "guaxupe_2022",
        "Patches_P512_S300_NN", "Imagens_Individuais",
        "patch_y0_x0.tif",
    )
    with pytest.raises(ValueError, match="AMBAS"):
        find_rgi_pairs(tmp_path)


def test_duplicata_identica_nao_conta_duas_amostras(tmp_path):
    for part in ("parte_001", "parte_002"):
        for role in ("Imagens_Individuais", "Mascaras_Individuais"):
            _file(
                tmp_path, part, "guaxupe_2022",
                "Patches_P512_S300_NN", role,
                "patch_y0_x0.tif", b"identico",
            )
    assert len(find_rgi_pairs(tmp_path)) == 1


def test_duplicata_divergente_interrompe_preparacao(tmp_path):
    for part, value in (("parte_001", b"a"), ("parte_002", b"b")):
        _file(
            tmp_path, part, "guaxupe_2022",
            "Patches_P512_S300_NN", "Imagens_Individuais",
            "patch_y0_x0.tif", value,
        )
    _file(
        tmp_path, "parte_001", "guaxupe_2022",
        "Patches_P512_S300_NN", "Mascaras_Individuais",
        "patch_y0_x0.tif", b"rotulo",
    )
    with pytest.raises(ValueError, match="divergentes"):
        find_rgi_pairs(tmp_path)
