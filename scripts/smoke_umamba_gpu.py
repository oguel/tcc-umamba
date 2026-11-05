"""Prova tecnica de compatibilidade U-Mamba CUDA com quatro bandas.

Nao instala dependencias, nao acessa Drive e NAO e treinamento cientifico.
Exemplo: python -m scripts.smoke_umamba_gpu --patch-size 256
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from src.losses import BCEDiceLoss


def run_gpu_smoke(patch_size: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Nenhuma GPU CUDA disponivel neste processo. Execute em ambiente com GPU "
            "(Colab/Kaggle/local) e PyTorch CUDA compativel."
        )
    from src.models.umamba import build_official_umamba_enc_2d

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model = build_official_umamba_enc_2d(
        input_channels=4,
        num_classes=1,
        input_size=(patch_size, patch_size),
    ).to(device)
    model.train()
    images = torch.rand((1, 4, patch_size, patch_size), device=device)
    masks = torch.zeros((1, 1, patch_size, patch_size), device=device)
    masks[:, :, patch_size // 4 : 3 * patch_size // 4, patch_size // 4 :] = 1
    masks[:, :, :10, :10] = 255
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(images)
    if not isinstance(prediction, torch.Tensor) or prediction.shape != masks.shape:
        raise ValueError(f"U-Mamba retornou formato invalido: {type(prediction)}")
    loss = BCEDiceLoss(ignore_index=255)(prediction, masks)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Loss nao-finita na U-Mamba.")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("Gradientes invalidos ou ausentes na U-Mamba.")
    optimizer.step()
    torch.cuda.synchronize(device)
    try:
        mamba_version = version("mamba-ssm")
    except PackageNotFoundError:
        mamba_version = None
    return {
        "status": "technical_smoke_only_not_scientific_result",
        "model": "UMambaEnc_2d",
        "input_shape": list(images.shape),
        "output_shape": list(prediction.shape),
        "step_completed": True,
        "loss_is_finite": True,
        "gradients_are_finite": True,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1048576, 2),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "mamba_ssm": mamba_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_gpu_smoke(args.patch_size)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
