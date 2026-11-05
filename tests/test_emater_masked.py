"""Regressões para ground truth da EMATER; sem acesso ao Drive."""
import pytest
import torch

from src.losses import BCEDiceLoss, dice_loss
from src.metrics import BinaryConfusion, metrics_from_confusion


def test_ignore_pixel_does_not_change_loss():
    logits = torch.tensor([[[[0.5, -1.0, 8.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0, 255.0]]]])
    reference_logits = logits[..., :2]
    reference_target = target[..., :2]
    loss_fn = BCEDiceLoss()
    got = loss_fn(logits, target)
    expected = loss_fn(reference_logits, reference_target)
    assert torch.allclose(got, expected, atol=1e-6)
    got.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 0, 2].item() == 0


def test_ignore_pixel_does_not_change_metrics():
    confusion = BinaryConfusion()
    confusion.update(torch.tensor([[[[9., -9., 9.]]]]),
                     torch.tensor([[[[1., 0., 255.]]]]))
    assert confusion.as_dict() == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert metrics_from_confusion(confusion)["iou"] == 1


def test_only_ignore_is_safe():
    logits = torch.ones((1, 1, 2, 2), requires_grad=True)
    labels = torch.full_like(logits, 255.)
    loss = BCEDiceLoss()(logits, labels)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_invalid_class_rejected():
    with pytest.raises(ValueError):
        dice_loss(torch.zeros((1, 1, 1, 1)),
                  torch.tensor([[[[3.0]]]]))
