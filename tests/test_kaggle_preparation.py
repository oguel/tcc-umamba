"""Preparação não deve modificar dados nem aceitar lote diferente."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import package_rgi_dataset, preflight_training
from src.data.emater_dataset import RasterPair
from src.data.rgi_fingerprint import dataset_fingerprint
from src.models import umamba_runtime


def test_packaging_preserves_bytes_and_refuses_overwrite(tmp_path, monkeypatch):
    root = tmp_path / "data"
    variant = root / "part1/guaxupe_2022/Patches_P512_S300_NN"
    paths = [
        variant / leaf / "patch_y0_x0.tif"
        for leaf in ("Imagens_Individuais", "Mascaras_Individuais")
    ]
    for i, path in enumerate(paths):
        path.parent.mkdir(parents=True)
        path.write_bytes(bytes([i, 3, 7]))
    pair = RasterPair(*paths, "guaxupe_2022")
    monkeypatch.setattr(package_rgi_dataset, "find_rgi_pairs", lambda root: [pair])
    monkeypatch.setattr(
        package_rgi_dataset,
        "get_config",
        lambda: {"official_dataset": {"expected_rgi_fingerprint": dataset_fingerprint([pair])}},
    )
    output = tmp_path / "private.zip"
    summary = package_rgi_dataset.package(root, output)
    assert summary["pairs"] == 1
    with ZipFile(output) as archive:
        for path in paths:
            assert archive.read(path.relative_to(root).as_posix()) == path.read_bytes()
    with pytest.raises(FileExistsError):
        package_rgi_dataset.package(root, output)


def test_preflight_persists_mismatched_data_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_training, "audit", lambda root: {"dataset_fingerprint": "bad"})
    report = preflight_training.preflight(Path("unused"), "unet", tmp_path / "preflight.json")
    assert report["technical_ready"] is False
    assert "Hash" in report["error"]
    assert (tmp_path / "preflight.json").is_file()


def test_unet_stack_does_not_install_mamba_when_torch_matches(monkeypatch):
    monkeypatch.setattr(umamba_runtime, "_mamba_wheel_filename", lambda: "supported.whl")
    versions = {
        "torch": umamba_runtime.TORCH_LOCAL_PREFIX,
        "torchvision": umamba_runtime.TORCHVISION_LOCAL_PREFIX,
        "torchaudio": umamba_runtime.TORCHAUDIO_LOCAL_PREFIX,
    }
    monkeypatch.setattr(umamba_runtime, "_installed_version", versions.get)
    monkeypatch.setattr(
        umamba_runtime, "_pip_install", lambda *args: pytest.fail("Unexpected install")
    )
    assert umamba_runtime.install_torch_stack() is False
