"""Regressões para crop e augmentação online sem vazamento de máscaras."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset, Subset

from src.data.augmentation import RGIPatchViews


class SyntheticRaster(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, length: int = 2) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        y, x = torch.meshgrid(torch.arange(512), torch.arange(512), indexing="ij")
        mask = ((x + y + index) % 3 == 0).float()
        mask[:80, :80] = 255
        # Canal 0 espelha o rótulo (inclusive ignore) para verificar que
        # nenhuma transformação deslocou a máscara em relação à imagem.
        image = torch.stack(
            (mask, x.float(), y.float(), (x + y).float()), dim=0
        )
        return image, mask.unsqueeze(0)


def test_val_256_cobre_quatro_quadrantes_sem_augmentar():
    source = SyntheticRaster(1)
    views = RGIPatchViews(source, size=256, mode="eval")
    image_full, mask_full = source[0]
    assert len(views) == 4
    for idx in range(4):
        image, mask = views[idx]
        top, left = divmod(idx, 2)
        assert image.shape == (4, 256, 256)
        assert torch.equal(
            image, image_full[:, top * 256 : (top + 1) * 256, left * 256 : (left + 1) * 256]
        )
        assert torch.equal(
            mask, mask_full[:, top * 256 : (top + 1) * 256, left * 256 : (left + 1) * 256]
        )
        assert torch.equal(image[:1], mask)


def test_treino_256_augmentado_preserva_alinhamento_e_ignore():
    torch.manual_seed(111)
    views = RGIPatchViews(
        SyntheticRaster(2), size=256, mode="train", augment=True,
        train_views_per_image=4,
    )
    assert len(views) == 8
    for idx in range(8):
        image, mask = views[idx]
        assert image.shape == (4, 256, 256)
        assert mask.shape == (1, 256, 256)
        assert torch.equal(image[:1], mask)
        assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0, 255.0})


def test_treino_512_aug_preserva_todo_o_raster():
    torch.manual_seed(24)
    views = RGIPatchViews(SyntheticRaster(1), size=512, augment=True)
    image, mask = views[0]
    assert image.shape == (4, 512, 512)
    assert mask.shape == (1, 512, 512)
    assert torch.equal(image[:1], mask)
    assert int((mask == 255).sum()) == 80 * 80


def test_sem_aug_treino_256_deterministico_e_subsets_isolados():
    source = SyntheticRaster(3)
    views = RGIPatchViews(Subset(source, [2, 0]), size=256, train_views_per_image=4)
    assert len(views) == 8
    assert torch.equal(views[0][1], source[2][1][:, :256, :256])
    assert torch.equal(views[4][1], source[0][1][:, :256, :256])
    assert torch.equal(views[0][1], views[0][1])


def test_avaliacao_nao_pode_ter_augmentacao():
    with pytest.raises(ValueError, match="validação"):
        RGIPatchViews(SyntheticRaster(1), size=256, mode="eval", augment=True)


def test_512_sem_aug_nao_pode_duplicar_uma_amostra():
    with pytest.raises(ValueError, match="duplicar"):
        RGIPatchViews(SyntheticRaster(1), size=512, train_views_per_image=4)


def test_crop_ignorado_e_detectado():
    class Unlabelled(Dataset[tuple[torch.Tensor, torch.Tensor]]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros((4, 512, 512)), torch.full((1, 512, 512), 255.0)

    with pytest.raises(ValueError, match="Nenhum pixel rotulado"):
        RGIPatchViews(
            Unlabelled(), size=256, mode="train", augment=True,
        )[0]
