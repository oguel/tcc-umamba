"""Adaptador do TCC para a implementação oficial U-Mamba."""

from __future__ import annotations

import importlib
from typing import Sequence

import torch
from torch import nn

OFFICIAL_REPOSITORY = "https://github.com/bowang-lab/U-Mamba"


def _init_weights_he(module: nn.Module, negative_slope: float = 1e-2) -> None:
    """Replica a inicialização He aplicada pelo helper oficial do U-Mamba."""
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=negative_slope)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def _resolve_umamba_class():
    """Resolve UMambaEnc preparada no runtime; mantém fallback nnU-Net."""
    try:
        module = importlib.import_module("umamba_official_standalone")
        return module.UMambaEnc
    except (ImportError, AttributeError):
        pass

    try:
        from nnunetv2.nets.UMambaEnc_2d import UMambaEnc

        return UMambaEnc
    except ImportError as exc:
        raise RuntimeError(
            "U-Mamba ainda não foi preparada neste runtime. "
            "Execute ensure_umamba_runtime() de src.models.umamba_runtime."
        ) from exc


def umamba_available() -> bool:
    """Verifica se a classe U-Mamba e mamba_ssm estão funcionais/importáveis."""
    try:
        import mamba_ssm  # noqa: F401

        _resolve_umamba_class()
    except (ImportError, RuntimeError):
        return False
    return True


def build_official_umamba_enc_2d(
    input_channels: int = 3,
    num_classes: int = 1,
    input_size: tuple[int, int] = (256, 256),
    features_per_stage: Sequence[int] = (16, 32, 64, 128),
) -> nn.Module:
    """Constrói diretamente a arquitetura oficial UMambaEnc 2D."""
    UMambaEnc = _resolve_umamba_class()

    features = [int(value) for value in features_per_stage]
    n_stages = len(features)
    if n_stages < 3:
        raise ValueError("U-Mamba requer ao menos três estágios para este projeto.")

    model = UMambaEnc(
        input_size=input_size,
        input_channels=int(input_channels),
        n_stages=n_stages,
        features_per_stage=features,
        conv_op=nn.Conv2d,
        kernel_sizes=[(3, 3)] * n_stages,
        strides=[(1, 1)] + [(2, 2)] * (n_stages - 1),
        n_conv_per_stage=[2] * n_stages,
        num_classes=int(num_classes),
        n_conv_per_stage_decoder=[2] * (n_stages - 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
        stem_channels=None,
    )
    model.apply(_init_weights_he)
    return model
