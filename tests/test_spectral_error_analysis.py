"""Os estratos não confundem ignorados com bordas e cobrem os extremos de NDVI."""

import numpy as np
import pytest

from scripts.analyze_validation_spectra import cm, spectral_strata


def test_undefined_ndvi_and_extremes_partition_valid_pixels():
    image = np.zeros((4, 1, 5), dtype=np.float32)
    image[2, 0] = [1, 0, 0, 0.2, 0.5]
    image[3, 0] = [0, 1, 0, 0.8, 0.5]
    target = np.array([[0, 1, 0, 1, 255]])
    groups = spectral_strata(image, target)
    ndvi = [v for key, v in groups.items() if key.startswith("ndvi_")]
    np.testing.assert_array_equal(np.sum(ndvi, axis=0), target != 255)
    assert groups["ndvi_-1_0"][0, 0]
    assert groups["ndvi_0.8_1"][0, 1]
    assert groups["ndvi_undefined"][0, 2]


def test_nodata_is_not_a_class_boundary_and_brightness_is_partitioned():
    target = np.array([[1, 1, 255], [1, 1, 255]])
    groups = spectral_strata(np.ones((4, 2, 3), dtype=np.float32), target)
    assert not groups["boundary_within_2_pixels"].any()
    np.testing.assert_array_equal(groups["interior_more_than_2_pixels"], target != 255)
    brightness = [v for key, v in groups.items() if key.startswith("brightness_")]
    np.testing.assert_array_equal(np.sum(brightness, axis=0), target != 255)


def test_class_boundary_sides_and_confusion_denominators():
    target = np.array([[0, 0, 1, 1, 255]])
    groups = spectral_strata(np.ones((4, 1, 5), dtype=np.float32), target)
    assert groups["boundary_within_2_pixels"][0, 1]
    assert groups["boundary_within_2_pixels"][0, 2]
    result = cm(target, np.array([[0, 1, 0, 1, 99]]), np.ones(target.shape, dtype=bool))
    assert result == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_invalid_predictions_are_rejected_before_counting():
    with pytest.raises(ValueError, match="Classes"):
        cm(np.array([[1]]), np.array([[2]]), np.array([[True]]))
