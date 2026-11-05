"""Retomada deve preservar pesos, otimizador, shuffle e RNG entre épocas."""

import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.losses import BCEDiceLoss
from src.trainer import fit_model


class RandomViews(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = torch.rand(1, 8, 8) + float(np.random.random()) + random.random()
        return image, torch.full((1, 8, 8), float(index % 2))


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Dropout2d(0.1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loaders = [
        DataLoader(
            RandomViews(),
            batch_size=2,
            shuffle=training,
            generator=torch.Generator().manual_seed(seed + i),
        )
        for i, training in enumerate((True, False))
    ]
    return model, optimizer, loaders


def run(parts, path, epochs, resume=None, contract=None):
    model, optimizer, loaders = parts
    return fit_model(
        model,
        *loaders,
        BCEDiceLoss(),
        optimizer,
        torch.device("cpu"),
        epochs,
        path,
        resume_path=resume,
        run_contract=contract,
    )


def test_resume_matches_uninterrupted_training(tmp_path):
    continuous = setup(42)
    reference = run(continuous, tmp_path / "continuous/best.pt", 2)
    interrupted = setup(42)
    path = tmp_path / "interrupted/best.pt"
    run(interrupted, path, 1)
    assert path.with_name("history.json").is_file()
    restored = setup(123)
    history = run(restored, path, 2, path.with_name("last.pt"))
    for key, value in continuous[0].state_dict().items():
        assert torch.equal(value, restored[0].state_dict()[key])
    assert [r["train_loss"] for r in reference] == [r["train_loss"] for r in history]
    assert [r["val_loss"] for r in reference] == [r["val_loss"] for r in history]
    state = torch.load(path.with_name("last.pt"), weights_only=True)
    assert state["epoch"] == 2
    assert len(state["history"]) == 2
    assert not list(path.parent.glob("*.tmp"))


def test_resume_rejects_modified_contract(tmp_path):
    path = tmp_path / "best.pt"
    run(setup(42), path, 1, contract={"split": "original"})
    with pytest.raises(ValueError, match="incompatível"):
        run(setup(42), path, 2, path.with_name("last.pt"), {"split": "changed"})
