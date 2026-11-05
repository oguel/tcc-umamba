"""Smoke test do pipeline: GeoTIFF -> augmentação -> U-Net -> loss -> backprop.

Este teste é DIAGNÓSTICO, não é treinamento científico, nem abre teste cego.
Sintético: python -m scripts.smoke_emater_rgi --synthetic --augment
Real: python -m scripts.smoke_emater_rgi --image IMAGEM.tif --mask MASCARA.tif
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.data.augmentation import RGIPatchViews
from src.data.emater_dataset import EMATERRGIDataset, RasterPair
from src.losses import BCEDiceLoss
from src.models.unet import UNet


def _synthetic_pair(directory: Path) -> RasterPair:
    """Cria um raster 10 bandas com mesmo contrato do lote, sem dados privados."""
    import rasterio
    from rasterio.transform import from_origin

    image_path = directory / "synthetic_image.tif"
    mask_path = directory / "synthetic_mask.tif"
    transform = from_origin(325000, 7660000, 10, 10)
    rng = np.random.default_rng(42)
    image = rng.integers(1200, 3600, size=(10, 512, 512), dtype=np.uint16)
    image[:, :40, :80] = 0
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[200:430, 140:380] = 1
    mask[:40, :80] = 255
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        height=512,
        width=512,
        count=10,
        dtype="uint16",
        nodata=0,
        crs="EPSG:32723",
        transform=transform,
    ) as dst:
        dst.write(image)
        dst.descriptions = (
            "B02", "B03", "B04", "B08", "B05",
            "B06", "B07", "B8A", "B11", "B12",
        )
        dst.update_tags(BOA_ADD_OFFSET="-1000")
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=512,
        width=512,
        count=1,
        dtype="uint8",
        nodata=255,
        crs="EPSG:32723",
        transform=transform,
    ) as dst:
        dst.write(mask, 1)
    return RasterPair(image_path, mask_path, "synthetic_2026")


def run_smoke(pair: RasterPair, size: int = 256, augment: bool = False) -> dict:
    """Realiza exatamente uma etapa com pesos pequenos; nunca é resultado final."""
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    base = EMATERRGIDataset([pair])
    views = RGIPatchViews(base, size=size, mode="train", augment=augment)
    image, mask = views[0]
    image_batch = image.unsqueeze(0)
    mask_batch = mask.unsqueeze(0)
    valid = int((mask_batch != 255).sum())
    if valid == 0:
        raise ValueError("Smoke test encontrou máscara sem pixels rotulados.")
    model = UNet(in_channels=4, out_channels=1, channels=(8, 16, 32, 64))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = BCEDiceLoss(ignore_index=255)
    optimizer.zero_grad(set_to_none=True)
    output = model(image_batch)
    loss = criterion(output, mask_batch)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Loss inválida no smoke test.")
    loss.backward()
    grads_valid = all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
    )
    if not grads_valid:
        raise FloatingPointError("Gradiente inválido no smoke test.")
    optimizer.step()
    result = {
        "kind": "technical_smoke_not_scientific_training",
        "image_shape": list(image_batch.shape),
        "mask_shape": list(mask_batch.shape),
        "output_shape": list(output.shape),
        "valid_pixels": valid,
        "ignored_pixels": int((mask_batch == 255).sum()),
        "loss_is_finite": bool(torch.isfinite(loss)),
        "gradients_are_finite": grads_valid,
        "one_optimizer_step_completed": True,
        "network": "small_unet_smoke_only",
        "patch_size": size,
        "augment_train_only": augment,
        "device": "cpu",
    }
    if output.shape != mask_batch.shape:
        raise ValueError("Saída da rede não corresponde à máscara.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--patch-size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.synthetic and (args.image is None or args.mask is None):
        parser.error("Passe --synthetic, ou informe --image e --mask.")
    if args.synthetic and (args.image is not None or args.mask is not None):
        parser.error("--synthetic e --image/--mask são alternativas.")

    if args.synthetic:
        with tempfile.TemporaryDirectory() as temporary:
            pair = _synthetic_pair(Path(temporary))
            report = run_smoke(pair, size=args.patch_size, augment=args.augment)
    else:
        report = run_smoke(
            RasterPair(args.image, args.mask, "technical_smoke"),
            size=args.patch_size,
            augment=args.augment,
        )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
