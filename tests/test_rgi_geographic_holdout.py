"""Split candidato: zonas geograficas sem vazamento e cobertura do treino."""

from __future__ import annotations

import pytest

from src.data.rgi_geographic_holdout import (
    DEFAULT_TEST_CENTER,
    DEFAULT_VALIDATION_CENTER,
    intersects,
    region_holdout_from_bounds,
)


def _rect(x: float, y: float, side: float = 5120.0):
    return (x - side / 2, y - side / 2, x + side / 2, y + side / 2)


def _dataset():
    bounds = []
    cities = []
    for j in range(5):
        for i in range(8):
            bounds.append(_rect(480000 + j * 40000 + i * 7000, 7700000))
            cities.append(f"city{j}")
    for i in range(18):
        bounds.append(_rect(DEFAULT_VALIDATION_CENTER[0] + i * 10, DEFAULT_VALIDATION_CENTER[1]))
        cities.append("city0")
    for i in range(18):
        bounds.append(_rect(DEFAULT_TEST_CENTER[0] + i * 10, DEFAULT_TEST_CENTER[1]))
        cities.append("city4")
    return bounds, cities


def test_holdout_tem_todos_municipios_no_treino():
    bounds, cities = _dataset()
    result = region_holdout_from_bounds(bounds, cities)
    assert len(result.train) == 40
    assert len(result.validation) == 18
    assert len(result.test) == 18
    assert len(result.purged) == 0
    for a, b in [
        (result.train, result.validation),
        (result.train, result.test),
        (result.validation, result.test),
    ]:
        assert all(not intersects(bounds[i], bounds[j]) for i in a for j in b)


def test_holdout_independe_da_ordem_dos_arquivos():
    bounds, cities = _dataset()
    split = region_holdout_from_bounds(bounds, cities)
    rev = region_holdout_from_bounds(bounds[::-1], cities[::-1])
    assert {len(split.train), len(split.validation), len(split.test)} == {40, 18}
    assert len(rev.train) == len(split.train)
    assert len(rev.validation) == len(split.validation)
    assert len(rev.test) == len(split.test)


def test_intersecao_na_borda_nao_conta_como_vazamento():
    assert not intersects((0, 0, 10, 10), (10, 2, 20, 10))
    assert intersects((0, 0, 10, 10), (9, 2, 20, 10))


def test_falha_se_uma_cidade_nao_tiver_treino():
    bounds, cities = _dataset()
    with pytest.raises(ValueError, match="Cobertura municipal"):
        region_holdout_from_bounds(
            bounds, cities, min_train_per_municipality=9
        )


def test_falha_se_validacao_e_teste_sobrepuserem():
    bounds, cities = _dataset()
    with pytest.raises(ValueError, match="compartilham"):
        region_holdout_from_bounds(
            bounds, cities, validation_center=DEFAULT_TEST_CENTER
        )


def test_falha_se_raster_ou_bounds_estiverem_incorretos():
    bounds, cities = _dataset()
    bounds[0] = (1, 2, 1, 3)
    with pytest.raises(ValueError, match="Bounds invalidos"):
        region_holdout_from_bounds(bounds, cities)
