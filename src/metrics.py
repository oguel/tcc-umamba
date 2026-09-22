"""Métricas agregadas para segmentação binária."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinaryConfusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> None:
        preds = torch.sigmoid(logits) >= threshold
        truth = targets >= 0.5

        self.tp += int(torch.logical_and(preds, truth).sum().item())
        self.fp += int(torch.logical_and(preds, ~truth).sum().item())
        self.tn += int(torch.logical_and(~preds, ~truth).sum().item())
        self.fn += int(torch.logical_and(~preds, truth).sum().item())

    def as_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def metrics_from_confusion(confusion: BinaryConfusion) -> dict[str, float]:
    """Calcula métricas positivas para a classe Café a partir da matriz agregada."""
    tp, fp, tn, fn = confusion.tp, confusion.fp, confusion.tn, confusion.fn
    return {
        "iou": _safe_div(tp, tp + fp + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "accuracy": _safe_div(tp + tn, tp + fp + tn + fn),
        "omission_error": _safe_div(fn, tp + fn),
        "commission_error": _safe_div(fp, tp + fp),
    }
