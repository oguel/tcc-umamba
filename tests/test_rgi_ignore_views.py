"""Views 100% ignoradas nunca contaminam loss ou estatisticas do piloto."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.losses import BCEDiceLoss
from src.trainer import run_epoch


def _sample_batch():
    images = torch.ones((2, 4, 16, 16))
    masks = torch.full((2, 1, 16, 16), 255.0)
    masks[1] = 0
    masks[1, :, :8, :8] = 1
    return images, masks


def test_ignora_view_vazia_na_media_sem_alterar_loss():
    images, masks = _sample_batch()
    model = nn.Conv2d(4, 1, kernel_size=1, bias=False)
    nn.init.zeros_(model.weight)
    criterion = BCEDiceLoss()
    expected = criterion(model(images[1:2]), masks[1:2]).item()
    loader = DataLoader(TensorDataset(images, masks), batch_size=2)
    metrics = run_epoch(model, loader, criterion, torch.device("cpu"))
    assert metrics["loss"] == pytest.approx(expected, abs=1e-6)
    assert metrics["skipped_fully_ignored_views"] == 1.0
    assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == 256


def test_epoca_somente_ignorada_falha_sem_falsear_resultado():
    images, masks = _sample_batch()
    masks[:] = 255
    model = nn.Conv2d(4, 1, kernel_size=1)
    loader = DataLoader(TensorDataset(images, masks), batch_size=2)
    with pytest.raises(ValueError, match="sem nenhum pixel rotulado"):
        run_epoch(model, loader, BCEDiceLoss(), torch.device("cpu"))
