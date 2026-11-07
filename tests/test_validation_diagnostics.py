"""Seleção deve retornar só validação e rejeitar alterações no teste/divisão."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src import validation_diagnostics
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


@pytest.mark.parametrize("model", ["unet", "umamba"])
def test_checkpoint_audit_supports_both_models_without_importing_runtime(
    tmp_path, monkeypatch, model
):
    pairs, splits, digest = fixture()
    contract = {
        "model": model,
        "epoch_budget": 2,
        "dataset_fingerprint": "data",
        "split_sha256": digest,
    }
    history = [{"epoch": 1, "val_iou": 0.6}, {"epoch": 2, "val_iou": 0.5}]
    documents = {
        "metrics.json": {"model": model, "run_contract": contract, "best_epoch": 1},
        "invocation.json": contract,
        "history.json": history,
        "splits.json": splits,
        "status.json": {"status": "completed", "blind_test_evaluated": False},
        "progress.json": {"epochs_completed": 2},
    }
    for name, value in documents.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    best = {
        "run_contract": contract,
        "epoch": 1,
        "val_iou": 0.6,
        "model_state_dict": {"weight": torch.ones(1)},
    }
    torch.save(best, tmp_path / f"best_{model}_rgi.pt")
    torch.save({**best, "epoch": 2, "history": history}, tmp_path / "last.pt")
    monkeypatch.setattr(validation_diagnostics, "find_rgi_pairs", lambda root: pairs)
    monkeypatch.setattr(validation_diagnostics, "dataset_fingerprint", lambda pairs: "data")
    monkeypatch.setattr(
        validation_diagnostics,
        "get_config",
        lambda: {
            "official_dataset": {
                "expected_rgi_fingerprint": "data",
                "split": {"expected_split_sha256": digest},
            }
        },
    )
    selected, checkpoint, _, _ = validation_diagnostics.inspect_run(tmp_path, tmp_path)
    assert selected == [pairs[1]]
    assert checkpoint["epoch"] == 1


def test_umamba_diagnostic_requires_cuda_before_creating_output(tmp_path, monkeypatch):
    from scripts import inspect_validation

    monkeypatch.setattr(
        inspect_validation, "inspect_run", lambda *args: ([], {}, {"model": "umamba"}, [])
    )
    output = tmp_path / "should_not_exist"
    with pytest.raises(RuntimeError, match="requer CUDA"):
        inspect_validation.diagnose(tmp_path, tmp_path, output, "cpu")
    assert not output.exists()
