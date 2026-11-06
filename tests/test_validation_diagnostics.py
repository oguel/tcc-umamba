"""Seleção deve retornar só validação e rejeitar alterações no teste/divisão."""

from copy import deepcopy
from pathlib import Path

import pytest

from src.data.emater_dataset import RasterPair
from src.data.rgi_balanced_split import split_digest
from src.data.rgi_geographic_holdout import RegionalSplit
from src.validation_diagnostics import patch_key, validation_pairs


def fixture():
    paths = [
        f"part1/city_2022/Patches_P512_S300_NN/Imagens_Individuais/patch_y0_x{i}.tif"
        for i in range(4)
    ]
    pairs = [
        RasterPair(Path(p), Path(p.replace("Imagens", "Mascaras")), "city_2022") for p in paths
    ]
    splits = {
        group: [paths[i]]
        for i, group in enumerate(("train", "validation", "test_locked", "purged"))
    }
    expected = split_digest([patch_key(p) for p in paths], RegionalSplit((0,), (1,), (2,), (3,)))
    return pairs, splits, expected


def test_only_validation_selected_independently_of_local_path_order():
    pairs, splits, expected = fixture()
    selected = validation_pairs(pairs[::-1], splits, expected)
    assert selected == [pairs[1]]
    assert pairs[2] not in selected


def test_overlap_or_missing_assignment_is_rejected():
    pairs, splits, expected = fixture()
    splits["validation"] += splits["test_locked"]
    with pytest.raises(ValueError, match="sobrepostas"):
        validation_pairs(pairs, splits, expected)


def test_reassigning_test_and_validation_breaks_frozen_hash():
    pairs, original, expected = fixture()
    splits = deepcopy(original)
    splits["validation"], splits["test_locked"] = splits["test_locked"], splits["validation"]
    with pytest.raises(ValueError, match="hash espacial"):
        validation_pairs(pairs, splits, expected)


def test_alternative_variant_is_never_selected():
    pairs, splits, expected = fixture()
    splits["validation"][0] = splits["validation"][0].replace(
        "Patches_P512_S300_NN/", "Patches_P512_S300_NN_Fora0_M0/"
    )
    with pytest.raises(ValueError, match="NN oficial"):
        validation_pairs(pairs, splits, expected)
