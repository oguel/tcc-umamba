"""Holdout geografico candidato para a RGI de Guaxupe (EPSG:32723).

Zonas espaciais compactas; nenhum patch intercepta outra divisao.
Os centros foram propostos com base na geometria dos manifestos, antes
de qualquer resultado de modelo. Congelar somente apos auditar os rasters.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot

from src.data.emater_dataset import RasterPair

Bounds = tuple[float, float, float, float]

# Coordenadas projetadas em metros (EPSG:32723), do lote de novembro/2026.
DEFAULT_TEST_CENTER = (359250.0, 7657850.0)
DEFAULT_VALIDATION_CENTER = (344690.0, 7620910.0)
DEFAULT_RADIUS_M = 9000.0


@dataclass(frozen=True)
class RegionalSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    purged: tuple[int, ...]


def intersects(a: Bounds, b: Bounds) -> bool:
    """Contato apenas na borda nao constitui area de sobreposicao."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def region_holdout_from_bounds(
    bounds: Sequence[Bounds],
    municipalities: Sequence[str],
    *,
    test_center: tuple[float, float] = DEFAULT_TEST_CENTER,
    validation_center: tuple[float, float] = DEFAULT_VALIDATION_CENTER,
    test_radius_m: float = DEFAULT_RADIUS_M,
    validation_radius_m: float = DEFAULT_RADIUS_M,
    min_train_per_municipality: int = 5,
    min_val: int = 15,
    min_test: int = 15,
) -> RegionalSplit:
    """Seleciona zonas de validacao/teste, e remove vizinhos do treinamento."""
    if not bounds or len(bounds) != len(municipalities):
        raise ValueError("Patches e municipios devem ter o mesmo tamanho nao-zero.")
    if test_radius_m <= 0 or validation_radius_m <= 0:
        raise ValueError("Raios de selecao espacial devem ser positivos.")
    for bbox in bounds:
        if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError(f"Bounds invalidos: {bbox}")

    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bounds]

    def selected(target: tuple[float, float], radius: float) -> tuple[int, ...]:
        return tuple(
            i
            for i, center in enumerate(centers)
            if hypot(center[0] - target[0], center[1] - target[1]) <= radius
        )

    test = selected(test_center, test_radius_m)
    validation = selected(validation_center, validation_radius_m)
    forbidden = set(test) | set(validation)
    if len(forbidden) != len(test) + len(validation):
        raise ValueError("Validacao e teste compartilham patches.")
    if len(test) < min_test or len(validation) < min_val:
        raise ValueError(
            f"Validacao/teste muito pequenos: {len(validation)} / {len(test)}."
        )
    if any(intersects(bounds[i], bounds[j]) for i in validation for j in test):
        raise ValueError("Validacao e teste interceptam a mesma area.")

    train = []
    purged = []
    for i, bbox in enumerate(bounds):
        if i in forbidden:
            continue
        if any(intersects(bbox, bounds[j]) for j in forbidden):
            purged.append(i)
        else:
            train.append(i)

    counts = Counter(municipalities[i] for i in train)
    insufficient = {
        name: counts.get(name, 0)
        for name in sorted(set(municipalities))
        if counts.get(name, 0) < min_train_per_municipality
    }
    if insufficient:
        raise ValueError(f"Cobertura municipal insuficiente no treino: {insufficient}")

    result = RegionalSplit(tuple(train), validation, test, tuple(purged))
    if sum(len(getattr(result, key)) for key in ("train", "validation", "test", "purged")) != len(
        bounds
    ):
        raise AssertionError("Cada patch deve ter exatamente um destino.")
    return result


def regional_split(
    pairs: Sequence[RasterPair],
    **kwargs,
) -> RegionalSplit:
    """Le os bounds dos GeoTIFFs originais, sem processar pixels nem alterar dados."""
    import rasterio

    bounds: list[Bounds] = []
    municipalities = []
    for pair in pairs:
        with rasterio.open(pair.image) as raster:
            if raster.crs is None or raster.crs.to_epsg() != 32723:
                raise ValueError(f"CRS nao homologado: {pair.image}")
            b = raster.bounds
            bounds.append((b.left, b.bottom, b.right, b.top))
        municipalities.append(pair.municipality)
    return region_holdout_from_bounds(bounds, municipalities, **kwargs)
