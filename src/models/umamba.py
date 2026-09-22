"""Integração com a implementação oficial U-Mamba.

Este módulo não reimplementa U-Mamba. Ele constrói a classe UMambaEnc_2d do
repositório oficial bowang-lab/U-Mamba quando as dependências estão instaladas.
"""

from __future__ import annotations

from typing import Sequence

from torch import nn


OFFICIAL_REPOSITORY = "https://github.com/bowang-lab/U-Mamba"


def umamba_available() -> bool:
    """Verifica se a implementação oficial e mamba_ssm estão importáveis."""
    try:
        import mamba_ssm  # noqa: F401
        from nnunetv2.nets.UMambaEnc_2d import UMambaEnc  # noqa: F401
    except ImportError:
        return False
    return True


def build_official_umamba_enc_2d(
    input_channels: int = 3,
    num_classes: int = 1,
    input_size: tuple[int, int] = (256, 256),
    features_per_stage: Sequence[int] = (32, 64, 128, 256),
) -> nn.Module:
    """Constrói U-Mamba Encoder 2D diretamente da classe oficial.

    O construtor evita depender do PlansManager do nnU-Net para o smoke test,
    preservando a arquitetura UMambaEnc_2d publicada pelos autores.
    """
    try:
        from nnunetv2.nets.UMambaEnc_2d import UMambaEnc
    except ImportError as exc:
        raise RuntimeError(
            "U-Mamba oficial não está instalada. Execute primeiro o notebook "
            "11_umamba_environment.ipynb em um ambiente compatível."
        ) from exc

    features = [int(value) for value in features_per_stage]
    n_stages = len(features)
    if n_stages < 3:
        raise ValueError("U-Mamba requer ao menos três estágios para este projeto.")

    return UMambaEnc(
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
