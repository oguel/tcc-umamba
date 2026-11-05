"""Particionamento espacial para a RGI inteira, sem um treino por município.

O bloco é uma região do mesmo CRS; patches de sobreposição cruzada são
excluídos do treinamento de cada fold para evitar vazamento por interseção.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from src.data.emater_dataset import RasterPair


def _spatial_index(pairs: Sequence[RasterPair], block_m: float):
    import rasterio
    rows = []
    crs_seen = None
    for pair in pairs:
        with rasterio.open(pair.image) as src:
            if src.crs is None or not src.crs.is_projected:
                raise ValueError("Divisão por blocos exige CRS projetado em metros.")
            if crs_seen is None:
                crs_seen = src.crs
            elif src.crs != crs_seen:
                raise ValueError("CRS divergentes: reprojete antes da divisão espacial.")
            bounds = src.bounds
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2
            block = (int(np.floor(center_x / block_m)), int(np.floor(center_y / block_m)))
            rows.append((block, (bounds.left, bounds.bottom, bounds.right, bounds.top)))
    return rows


def _intersects(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Contato apenas na borda (área zero) não constitui sobreposição."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def spatial_folds(
    pairs: Sequence[RasterPair], k: int = 5, block_m: float = 15000,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Retorna (índices treino, índices validação) para k folds espaciais.

    Os blocos são atribuídos às dobras com semente fixa, balanceando número de
    patches. Todos os recortes com interseção a uma área de validação são
    retirados do treino nessa dobra (purge). Não usa cidade como classe ou split.
    """
    if k < 2 or block_m <= 0:
        raise ValueError("Use k >= 2 e block_m positivo.")
    rows = _spatial_index(pairs, block_m)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (block, _) in enumerate(rows):
        groups[block].append(i)
    if len(groups) < k:
        raise ValueError(f"Apenas {len(groups)} blocos distintos para {k} folds.")
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]), reverse=True)
    fold_sizes = [0] * k
    block_fold = {}
    for key in keys:
        fold = min(range(k), key=lambda j: fold_sizes[j])
        block_fold[key] = fold
        fold_sizes[fold] += len(groups[key])

    result = []
    for fold in range(k):
        val = [i for i, (block, _) in enumerate(rows) if block_fold[block] == fold]
        validation_bounds = [rows[i][1] for i in val]
        train = [
            i for i, (block, bounds) in enumerate(rows)
            if block_fold[block] != fold
            and not any(_intersects(bounds, vb) for vb in validation_bounds)
        ]
        if not val or not train:
            raise ValueError(f"Fold {fold}: conjunto vazio após remoção de sobreposição.")
        result.append((train, val))
    return result
