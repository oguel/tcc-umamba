"""Funções de perda para segmentação binária com pixels 255 ignorados."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _valid_targets(targets: torch.Tensor, ignore_index: int = 255):
    valid = targets != float(ignore_index)
    clean = torch.where(valid, targets, torch.zeros_like(targets))
    if not torch.all((clean == 0) | (clean == 1)):
        raise ValueError("Máscaras devem ter apenas 0=não-café, 1=café, 255=ignorar.")
    return valid, clean


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Dice por amostra, considerando somente pixels rotulados."""
    if logits.shape != targets.shape:
        raise ValueError(f"Shapes diferentes: {logits.shape} vs {targets.shape}")
    valid, clean = _valid_targets(targets, ignore_index)
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, logits.ndim))
    valid_float = valid.to(logits.dtype)
    intersection = (probabilities * clean * valid_float).sum(dim=dims)
    denominator = ((probabilities + clean) * valid_float).sum(dim=dims)
    has_pixels = valid.any(dim=dims)
    dice = (2 * intersection + eps) / (denominator + eps)
    # Exclui amostras totalmente ignoradas.
    if not has_pixels.any():
        return logits.sum() * 0.0
    return (1.0 - dice[has_pixels]).mean()


class BCEDiceLoss(nn.Module):
    """Combina BCEWithLogits e Dice Loss sem contribuir com pixels ignorados."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError("Logits e máscaras precisam ter dimensões idênticas.")
        valid, clean = _valid_targets(targets, self.ignore_index)
        per_pixel = F.binary_cross_entropy_with_logits(logits, clean, reduction="none")
        # Um batch sem rótulos produz loss=0, preservando o grafo autograd.
        bce = (per_pixel * valid.to(per_pixel.dtype)).sum() / valid.sum().clamp(min=1)
        dice = dice_loss(logits, targets, ignore_index=self.ignore_index)
        return self.bce_weight * bce + self.dice_weight * dice
