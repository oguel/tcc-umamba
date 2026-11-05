"""Regressões de integridade e da aprovação indevida de auditoria candidata."""

import sys

import pytest

from scripts import validate_emater_rgi
from src.models import umamba_runtime


def test_expected_fingerprint_rejects_different_data(monkeypatch, tmp_path):
    monkeypatch.setattr(
        validate_emater_rgi,
        "audit",
        lambda *args: {"dataset_fingerprint": "actual", "preflight_approved": False},
    )
    monkeypatch.setattr(
        sys, "argv", ["audit", "--root", str(tmp_path), "--expected-fingerprint", "expected"]
    )
    with pytest.raises(SystemExit, match="FINGERPRINT DIVERGENTE"):
        validate_emater_rgi.main()


def test_mamba_installer_rejects_windows_before_mutation(monkeypatch):
    monkeypatch.setattr(umamba_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        umamba_runtime,
        "_pip_install",
        lambda *args: pytest.fail("Installer must not mutate unsupported platform"),
    )
    with pytest.raises(RuntimeError, match="Linux"):
        umamba_runtime.install_prebuilt_colab_stack()
