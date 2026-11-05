"""Validação municipal balanceada por geometria, preservando o teste anterior."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence

import numpy as np

from src.data.rgi_geographic_holdout import Bounds, RegionalSplit


def split_digest(keys: Sequence[str], split: RegionalSplit) -> str:
    record = {
        name: sorted(keys[i] for i in getattr(split, name))
        for name in ("train", "validation", "test", "purged")
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def balanced_split(
    bounds: Sequence[Bounds],
    municipalities: Sequence[str],
    locked_test: Sequence[int],
    *,
    buffer_m: float = 1000,
    per_city: int = 4,
    trials: int = 5000,
    seed: int = 42,
) -> RegionalSplit:
    """Busca geométrica determinística; não recebe classes ou métricas de modelo.

    Os índices de entrada devem estar ordenados por identidade estável do patch.
    Maximiza retenção municipal mínima no treino, depois retenção total.
    """
    b = np.asarray(bounds, dtype=float)
    if b.shape != (len(municipalities), 4) or not np.isfinite(b).all():
        raise ValueError("Geometria inválida")
    if np.any(b[:, 0] >= b[:, 2]) or np.any(b[:, 1] >= b[:, 3]):
        raise ValueError("Bounds inválidos")
    if buffer_m <= 0 or per_city < 1 or trials < 1:
        raise ValueError("Parâmetros inválidos")
    test = sorted(set(locked_test))
    if not test or min(test) < 0 or max(test) >= len(b):
        raise ValueError("Teste reservado inválido")
    dx = np.maximum(np.maximum(b[:, None, 0] - b[None, :, 2], b[None, :, 0] - b[:, None, 2]), 0)
    dy = np.maximum(np.maximum(b[:, None, 1] - b[None, :, 3], b[None, :, 1] - b[:, None, 3]), 0)
    conflict = np.hypot(dx, dy) < buffer_m
    available = ~conflict[:, test].any(axis=1)
    cities = sorted(set(municipalities))
    city_masks = [np.array([m == city for m in municipalities]) for city in cities]
    centers = (b[:, :2] + b[:, 2:]) / 2
    candidates = []
    for city_mask in city_masks:
        ids = np.flatnonzero(city_mask & available).tolist()
        if len(ids) < per_city + 5:
            raise ValueError("Cobertura insuficiente fora do teste")
        clusters = set()
        for i in ids:
            nearest = sorted(ids, key=lambda j: (float(np.linalg.norm(centers[i] - centers[j])), j))
            clusters.add(tuple(sorted(nearest[:per_city])))
        candidates.append(sorted(clusters))
    rng = random.Random(seed)
    best, best_score = None, None
    for _ in range(trials):
        validation = sorted(i for clusters in candidates for i in rng.choice(clusters))
        train_mask = available & ~conflict[:, validation].any(axis=1)
        counts = [int(np.count_nonzero(train_mask & mask)) for mask in city_masks]
        if min(counts) < 5:
            continue
        retention = min(
            count / int(np.count_nonzero(available & mask))
            for count, mask in zip(counts, city_masks, strict=True)
        )
        score = (retention, sum(counts), tuple(-i for i in validation))
        if best_score is None or score > best_score:
            best_score = score
            train = np.flatnonzero(train_mask).tolist()
            assigned = set(train) | set(validation) | set(test)
            best = RegionalSplit(
                tuple(train),
                tuple(validation),
                tuple(test),
                tuple(i for i in range(len(b)) if i not in assigned),
            )
    if best is None:
        raise ValueError("Nenhuma divisão viável com cobertura municipal")
    return best
