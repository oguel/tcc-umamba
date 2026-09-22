"""Dataset PyTorch para o conjunto experimental de segmentação de café."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class CoffeeSegmentationDataset(Dataset):
    """Carrega pares PNG de imagem RGB e máscara binária."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        image_size: int = 256,
        augment: bool = False,
        mask_threshold: int = 127,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = int(image_size)
        self.augment = augment
        self.mask_threshold = int(mask_threshold)

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Pasta de imagens inexistente: {self.images_dir}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Pasta de máscaras inexistente: {self.masks_dir}")

        image_names = {path.name for path in self.images_dir.glob("*.png")}
        mask_names = {path.name for path in self.masks_dir.glob("*.png")}

        missing_masks = sorted(image_names - mask_names)
        missing_images = sorted(mask_names - image_names)
        if missing_masks or missing_images:
            raise ValueError(
                "Pares imagem/máscara inconsistentes. "
                f"Sem máscara: {missing_masks[:5]}; sem imagem: {missing_images[:5]}"
            )

        self.names = sorted(image_names)
        if not self.names:
            raise ValueError(f"Nenhum PNG encontrado em {self.images_dir}")

    def __len__(self) -> int:
        return len(self.names)

    def _load_pair(self, name: str) -> tuple[Image.Image, Image.Image]:
        image = Image.open(self.images_dir / name).convert("RGB")
        mask = Image.open(self.masks_dir / name).convert("L")

        size = (self.image_size, self.image_size)
        image = image.resize(size, resample=Image.Resampling.BILINEAR)
        mask = mask.resize(size, resample=Image.Resampling.NEAREST)
        return image, mask

    def _augment_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        rotations = random.randint(0, 3)
        if rotations:
            angle = 90 * rotations
            image = image.rotate(angle)
            mask = mask.rotate(angle)

        if random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.35:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[index]
        image, mask = self._load_pair(name)

        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.uint8)
        mask_array = (mask_array > self.mask_threshold).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).contiguous()
        return image_tensor, mask_tensor
