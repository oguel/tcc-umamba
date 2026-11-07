"""Comparação pareada de predições binárias, com referência 255 ignorada."""

from __future__ import annotations

import numpy as np

from src.metrics import BinaryConfusion, metrics_from_confusion


def compare_predictions(target: np.ndarray, unet: np.ndarray, umamba: np.ndarray) -> dict:
    if target.shape != unet.shape or target.shape != umamba.shape or target.ndim != 2:
        raise ValueError("Referência e predições devem compartilhar o mesmo grid 2D.")
    if not np.isin(target, (0, 1, 255)).all():
        raise ValueError("Referência deve conter apenas 0, 1 e 255.")
    valid = target != 255
    if not np.isin(unet[valid], (0, 1)).all() or not np.isin(umamba[valid], (0, 1)).all():
        raise ValueError("Predições válidas devem ser binárias.")
    truth = target[valid] == 1
    left, right = unet[valid] == 1, umamba[valid] == 1

    def evaluate(prediction):
        confusion = BinaryConfusion(
            tp=int(np.count_nonzero(prediction & truth)),
            fp=int(np.count_nonzero(prediction & ~truth)),
            tn=int(np.count_nonzero(~prediction & ~truth)),
            fn=int(np.count_nonzero(~prediction & truth)),
        )
        return {"confusion": confusion.as_dict(), "metrics": metrics_from_confusion(confusion)}

    correct_left, correct_right = left == truth, right == truth
    return {
        "valid_pixels": int(np.count_nonzero(valid)),
        "ignored_pixels": int(np.count_nonzero(~valid)),
        "unet": evaluate(left),
        "umamba": evaluate(right),
        "both_correct": int(np.count_nonzero(correct_left & correct_right)),
        "both_wrong": int(np.count_nonzero(~correct_left & ~correct_right)),
        "only_unet_correct": int(np.count_nonzero(correct_left & ~correct_right)),
        "only_umamba_correct": int(np.count_nonzero(~correct_left & correct_right)),
    }
