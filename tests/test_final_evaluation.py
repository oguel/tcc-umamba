"""Teste bloqueado, pesos fixos e pixels geográficos únicos em exemplos sintéticos."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine

from src import final_evaluation
from src.data.emater_dataset import RasterPair
from src.data.rgi_balanced_split import split_digest
from src.data.rgi_geographic_holdout import RegionalSplit
from src.final_evaluation import CHECKPOINTS, EXPECTED_CONTRACT, SceneMosaic, locked_test_pairs
from src.validation_diagnostics import patch_key


def test_locked_selection_returns_only_test_and_checks_all_roles():
    paths = [
        f"part/city_2022/Patches_P512_S300_NN/Imagens_Individuais/patch_y0_x{i}.tif"
        for i in range(4)
    ]
    pairs = [
        RasterPair(Path(p), Path(p.replace("Imagens", "Mascaras")), "city_2022") for p in paths
    ]
    splits = {
        group: [paths[i]]
        for i, group in enumerate(("train", "validation", "test_locked", "purged"))
    }
    digest = split_digest([patch_key(p) for p in paths], RegionalSplit((0,), (1,), (2,), (3,)))
    assert locked_test_pairs(pairs[::-1], splits, digest) == [pairs[2]]
    changed = deepcopy(splits)
    changed["train"], changed["test_locked"] = changed["test_locked"], changed["train"]
    with pytest.raises(ValueError, match="hash espacial"):
        locked_test_pairs(pairs, changed, digest)


def mosaic():
    return SceneMosaic(
        [(Affine(10, 0, 100, 0, -10, 200), 1, 2), (Affine(10, 0, 110, 0, -10, 200), 1, 2)],
        "EPSG:32723",
    )


def test_overlap_counts_once_and_averages_before_threshold():
    value = mosaic()
    value.add(0, np.array([[1, 1]]), np.array([[0.9, 0.2]]))
    value.add(1, np.array([[1, 0]]), np.array([[0.9, 0.1]]))
    probability, result = value.finish()
    np.testing.assert_allclose(probability, [[0.9, 0.55, 0.1]])
    assert result["confusion"] == {"tp": 2, "fp": 0, "tn": 1, "fn": 0}
    assert value.count.tolist() == [[1, 2, 1]]


def test_ignored_pixels_never_contribute_and_valid_reference_wins():
    value = mosaic()
    value.add(0, np.array([[255, 255]]), np.array([[0.9, 0.9]]))
    value.add(1, np.array([[0, 1]]), np.array([[0.1, 0.9]]))
    probability, result = value.finish()
    assert value.target.tolist() == [[255, 0, 1]]
    assert result["confusion"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert probability[0, 1] == pytest.approx(0.1)


def test_reference_conflict_duplicate_and_incomplete_are_rejected():
    value = mosaic()
    value.add(0, np.array([[0, 1]]), np.array([[0.1, 0.9]]))
    with pytest.raises(ValueError, match="incompleto"):
        value.finish()
    with pytest.raises(ValueError, match="duas vezes"):
        value.add(0, np.array([[0, 1]]), np.array([[0.1, 0.9]]))
    with pytest.raises(ValueError, match="conflitantes"):
        value.add(1, np.array([[0, 1]]), np.array([[0.1, 0.9]]))


@pytest.mark.parametrize(
    "transform", [Affine(10, 0, 105, 0, -10, 200), Affine(20, 0, 100, 0, -20, 200)]
)
def test_misaligned_or_different_resolution_grid_rejected(transform):
    with pytest.raises(ValueError, match="alinhada|incompatíveis"):
        SceneMosaic([(Affine(10, 0, 100, 0, -10, 200), 1, 2), (transform, 1, 2)], "EPSG:32723")


@pytest.mark.parametrize("probability", [np.array([[np.nan, 0.5]]), np.array([[0.1, 1.1]])])
def test_invalid_probabilities_rejected(probability):
    with pytest.raises(ValueError, match="Probabilidade"):
        mosaic().add(0, np.array([[0, 1]]), probability)


def fake_audit(monkeypatch, corrupt_hash=False):
    contract = {
        **EXPECTED_CONTRACT,
        "model": "unet",
        "dataset_fingerprint": "data",
        "split_sha256": "split",
    }
    report = {"model": "unet", "run_contract": contract}
    monkeypatch.setattr(final_evaluation, "inspect_run", lambda *a: ([], {}, report, []))
    monkeypatch.setattr(
        final_evaluation,
        "_file_sha256",
        lambda p: "modified" if corrupt_hash else CHECKPOINTS["unet"],
    )


def test_replacement_checkpoint_rejected_before_test_selection(monkeypatch):
    fake_audit(monkeypatch, corrupt_hash=True)
    with pytest.raises(ValueError, match="pesos selecionados"):
        final_evaluation.audit_final_inputs(
            Path("unused"), {"unet": Path("unet"), "umamba": Path("umamba")}
        )


def test_changed_budget_rejected_before_checkpoint_loading(monkeypatch):
    fake_audit(monkeypatch)
    report = {"model": "unet", "run_contract": {**EXPECTED_CONTRACT, "epoch_budget": 200}}
    monkeypatch.setattr(final_evaluation, "inspect_run", lambda *a: ([], {}, report, []))
    with pytest.raises(ValueError, match="configuração final"):
        final_evaluation.audit_final_inputs(
            Path("unused"), {"unet": Path("unet"), "umamba": Path("umamba")}
        )


def test_preflight_does_not_call_model_or_infer_test(tmp_path, monkeypatch):
    from scripts import evaluate_locked_test

    monkeypatch.setattr(
        evaluate_locked_test,
        "audit_final_inputs",
        lambda *a: {
            "unet": ([], {}, {"dataset_fingerprint": "data", "run_contract": {"split_sha256": "s"}})
        },
    )
    monkeypatch.setattr(evaluate_locked_test, "prepare_mosaics", lambda pairs: {})

    def forbidden(*args):
        raise AssertionError("Inferência não pode ser chamada no preflight")

    monkeypatch.setattr(evaluate_locked_test, "evaluate_model", forbidden)
    result = evaluate_locked_test.run(tmp_path, {}, tmp_path / "new", execute=False)
    assert result["test_patches_inferred"] == 0
    assert not (tmp_path / "new/AVALIACAO_TESTE.json").exists()


def test_execute_without_cuda_fails_before_creating_output(tmp_path, monkeypatch):
    from scripts import evaluate_locked_test

    monkeypatch.setattr(
        evaluate_locked_test,
        "audit_final_inputs",
        lambda *a: {
            "unet": ([], {}, {"dataset_fingerprint": "data", "run_contract": {"split_sha256": "s"}})
        },
    )
    monkeypatch.setattr(evaluate_locked_test, "prepare_mosaics", lambda pairs: {})
    monkeypatch.setattr(evaluate_locked_test.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Linux/CUDA"):
        evaluate_locked_test.run(tmp_path, {}, tmp_path / "new", execute=True)
    assert not (tmp_path / "new").exists()


def test_synthetic_inference_reconstruction_and_geotiff_export(tmp_path, monkeypatch):
    """Exercita I/O com um modelo sintético; não comprova kernels CUDA/Mamba."""
    from scripts import evaluate_locked_test

    class SyntheticModel(torch.nn.Module):
        def forward(self, image):
            return image[:, :1] * 10 - 5

    class SyntheticDataset:
        def __init__(self, pairs):
            pass

        def __getitem__(self, index):
            target = torch.ones(1, 512, 512)
            target[0, 0, 0] = 255
            return torch.ones(4, 512, 512), target

    transform = Affine(10, 0, 100, 0, -10, 200)
    pair = RasterPair(Path("patch.tif"), Path("mask.tif"), "synthetic_2026")
    monkeypatch.setattr(evaluate_locked_test, "UNet", lambda **kwargs: SyntheticModel())
    monkeypatch.setattr(evaluate_locked_test, "EMATERRGIDataset", SyntheticDataset)
    monkeypatch.setattr(
        evaluate_locked_test,
        "prepare_mosaics",
        lambda pairs: {"synthetic_2026": SceneMosaic([(transform, 512, 512)], "EPSG:32723")},
    )
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    for name in ("memory_allocated", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "synthetic_cpu_test")
    result = evaluate_locked_test.evaluate_model(
        "unet",
        ([pair], {"model_state_dict": {}, "epoch": 79}, {"dataset_fingerprint": "synthetic"}),
        tmp_path / "result",
        torch.device("cpu"),
    )
    assert result["primary_patch_pooled"]["confusion"] == {
        "tp": 512 * 512 - 1,
        "fp": 0,
        "tn": 0,
        "fn": 0,
    }
    assert result["unique_scene_pixels"] == 512 * 512 - 1
    assert result["duplicate_pixel_instances"] == 0
    assert result["forward_benchmark"]["views_256"] == 4
    with rasterio.open(tmp_path / "result/synthetic_2026_predicao.tif") as image:
        assert image.transform == transform
        assert image.read(1)[0, 0] == 255
        assert image.read(1)[511, 511] == 1
    assert (tmp_path / "result/paineis_teste/synthetic_2026_patch.png").exists()
