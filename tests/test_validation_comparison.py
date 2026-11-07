"""Mesmas métricas agregadas não implicam erros nos mesmos pixels."""

import numpy as np
import pytest

from src.validation_comparison import compare_predictions


def test_paired_errors_and_ignore_are_respected():
    target = np.array([[1, 0, 255], [1, 0, 1]])
    left = np.array([[1, 1, 9], [0, 0, 1]])
    right = np.array([[1, 0, 255], [1, 1, 0]])
    result = compare_predictions(target, left, right)
    assert result["valid_pixels"] == 5
    assert result["ignored_pixels"] == 1
    assert result["unet"]["confusion"] == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}
    assert result["unet"]["confusion"] == result["umamba"]["confusion"]
    assert result["both_correct"] == 1
    assert result["both_wrong"] == 0
    assert result["only_unet_correct"] == result["only_umamba_correct"] == 2


def test_mismatched_grids_cannot_be_compared():
    with pytest.raises(ValueError, match="grid"):
        compare_predictions(np.zeros((2, 3)), np.zeros((3, 2)), np.zeros((2, 3)))


def test_invalid_labels_or_predictions_are_rejected():
    with pytest.raises(ValueError, match="Referência"):
        compare_predictions(np.array([[2]]), np.array([[0]]), np.array([[1]]))
    with pytest.raises(ValueError, match="binárias"):
        compare_predictions(np.array([[1]]), np.array([[0.7]]), np.array([[1]]))


def test_empty_validity_does_not_invent_correct_pixels():
    result = compare_predictions(np.full((2, 2), 255), np.ones((2, 2)), np.zeros((2, 2)))
    assert result["valid_pixels"] == result["both_correct"] == 0
    assert result["ignored_pixels"] == 4
    assert result["unet"]["metrics"]["iou"] == 0
