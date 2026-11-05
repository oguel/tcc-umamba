"""Recortes e augmentação online para segmentação multiespectral da RGI.

Não altera os GeoTIFFs. Aplica a MESMA geometria na imagem e na máscara.
Separação espacial deve ocorrer ANTES de instanciar RGIPatchViews.
"""

from __future__ import annotations

from typing import Literal, Protocol

import torch
from torch.utils.data import Dataset


class RasterSource(Protocol):
    """Contrato mínimo de Dataset com acesso por índice e comprimento."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]: ...


class RGIPatchViews(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Converte patches originais de 512 em views de 256/512.

    Treino com augment=True: crops aleatórios + flips + rotações de 90 graus.
    Treino sem augment: views determinísticas (centro ou quadrantes).
    Validação/teste: sempre determinísticos; para 256, 4 quadrantes sem
    sobreposição, cobrindo todos os 512x512 pixels originais.

    Atenção: train_views_per_image é o número de VISÕES por patch,
    não aumenta o número de amostras geográficas independentes.
    """

    def __init__(
        self,
        source: RasterSource,
        *,
        size: Literal[256, 512] = 512,
        mode: Literal["train", "eval"] = "train",
        augment: bool = False,
        train_views_per_image: int = 1,
        min_valid_fraction: float = 0.10,
        crop_attempts: int = 12,
    ) -> None:
        if size not in (256, 512):
            raise ValueError("Somente recortes 256 ou 512 são suportados.")
        if mode not in ("train", "eval"):
            raise ValueError("mode deve ser train ou eval.")
        if augment and mode != "train":
            raise ValueError("Não aplique augmentação à validação ou ao teste.")
        if train_views_per_image < 1 or train_views_per_image > 4:
            raise ValueError("train_views_per_image deve estar entre 1 e 4.")
        if not 0 <= min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction deve estar entre 0 e 1.")
        if crop_attempts < 1:
            raise ValueError("crop_attempts deve ser positivo.")
        if len(source) == 0:
            raise ValueError("O conjunto de origem está vazio.")
        if mode == "train" and size == 512 and not augment and train_views_per_image > 1:
            raise ValueError("512 sem augmentação: duplicar views idênticas não é útil.")
        self.source = source
        self.size = size
        self.mode = mode
        self.augment = augment
        self.train_views_per_image = train_views_per_image
        self.min_valid_fraction = min_valid_fraction
        self.crop_attempts = crop_attempts
        self.views_per_image = (
            train_views_per_image if mode == "train" else (4 if size == 256 else 1)
        )

    def __len__(self) -> int:
        return len(self.source) * self.views_per_image

    @staticmethod
    def _quadrant(index: int, size: int) -> tuple[int, int]:
        return (index // 2) * size, (index % 2) * size

    def _random_crop(self, mask: torch.Tensor, height: int, width: int) -> tuple[int, int]:
        size = self.size
        best = (-1.0, 0, 0)
        for _ in range(self.crop_attempts):
            top = int(torch.randint(height - size + 1, (1,)).item())
            left = int(torch.randint(width - size + 1, (1,)).item())
            fraction = float((mask[:, top : top + size, left : left + size] != 255).float().mean())
            if fraction > best[0]:
                best = (fraction, top, left)
            if fraction >= self.min_valid_fraction:
                return top, left
        if best[0] <= 0:
            raise ValueError("Nenhum pixel rotulado válido encontrado no crop.")
        return best[1], best[2]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        base_index, view_index = divmod(index, self.views_per_image)
        image, mask = self.source[base_index]
        if image.ndim != 3 or mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError("Esperado imagem (C,H,W) e máscara (1,H,W).")
        height, width = image.shape[-2:]
        if (height, width) != mask.shape[-2:] or (height, width) != (512, 512):
            raise ValueError("A entrada oficial deve ser 512x512 e alinhada à máscara.")

        if self.size == 256:
            if self.mode == "train" and self.augment:
                top, left = self._random_crop(mask, height, width)
            elif self.mode == "train" and self.views_per_image == 1:
                top, left = 128, 128
            else:
                top, left = self._quadrant(view_index, 256)
            image = image[:, top : top + 256, left : left + 256]
            mask = mask[:, top : top + 256, left : left + 256]

        if self.mode == "train" and self.augment:
            if bool(torch.rand(()) < 0.5):
                image, mask = torch.flip(image, (-1,)), torch.flip(mask, (-1,))
            if bool(torch.rand(()) < 0.5):
                image, mask = torch.flip(image, (-2,)), torch.flip(mask, (-2,))
            k = int(torch.randint(4, (1,)).item())
            if k:
                image, mask = torch.rot90(image, k, (-2, -1)), torch.rot90(mask, k, (-2, -1))

        return image.contiguous(), mask.contiguous()
