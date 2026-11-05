"""Cobertura, distância, teste preservado e digest da divisão balanceada."""

from collections import Counter
from math import hypot

import pytest

from src.data.rgi_balanced_split import balanced_split, split_digest


def test_balanced_split_preserves_test_and_municipal_validation():
    bounds, cities = [], []
    for city in range(5):
        for i in range(12):
            x = city * 100000 + i * 5000
            bounds.append((x, 0, x + 1000, 1000))
            cities.append(str(city))
    result = balanced_split(bounds, cities, [11], per_city=2, trials=100)
    assert result.test == (11,)
    assert Counter(cities[i] for i in result.validation) == {str(c): 2 for c in range(5)}
    assert len(set(result.train + result.validation + result.test + result.purged)) == 60
    for a, b in [
        (result.train, result.validation),
        (result.train, result.test),
        (result.validation, result.test),
    ]:
        for i in a:
            for j in b:
                u, v = bounds[i], bounds[j]
                assert (
                    hypot(max(u[0] - v[2], v[0] - u[2], 0), max(u[1] - v[3], v[1] - u[3], 0))
                    >= 1000
                )
    assert balanced_split(bounds, cities, [11], per_city=2, trials=100) == result
    keys = [str(i) for i in range(60)]
    assert len(split_digest(keys, result)) == 64
    keys[0] = "changed"
    assert split_digest(keys, result) != split_digest([str(i) for i in range(60)], result)


def test_balancing_fails_instead_of_silently_dropping_city():
    with pytest.raises(ValueError, match="Cobertura"):
        balanced_split([(i * 5000, 0, i * 5000 + 1000, 1000) for i in range(8)], ["city"] * 8, [0])
