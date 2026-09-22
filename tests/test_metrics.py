from src.metrics import BinaryConfusion, metrics_from_confusion


def test_metrics_perfect_prediction():
    confusion = BinaryConfusion(tp=10, fp=0, tn=20, fn=0)
    metrics = metrics_from_confusion(confusion)
    assert metrics["iou"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_metrics_counts_errors():
    confusion = BinaryConfusion(tp=5, fp=2, tn=10, fn=3)
    metrics = metrics_from_confusion(confusion)
    assert 0.0 < metrics["iou"] < 1.0
    assert metrics["omission_error"] > 0.0
    assert metrics["commission_error"] > 0.0
