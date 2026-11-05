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


def run_gpu_smoke(patch_size: int, kind: str = "umamba", prepare_runtime: bool = False) -> dict:
    if patch_size not in (256, 512) or kind not in ("unet", "umamba"):
        raise ValueError("Modelo/tamanho inválido para smoke test.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Nenhuma GPU CUDA disponivel neste processo. Execute em ambiente com GPU "
            "(Colab/Kaggle/local) e PyTorch CUDA compativel."
        )
    runtime = None
    if kind == "umamba":
        from src.models.umamba_runtime import ensure_umamba_runtime

        if prepare_runtime:
            runtime = ensure_umamba_runtime()
        from src.models.umamba import build_official_umamba_enc_2d
    else:
        from src.models.unet import UNet

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # Inicializa contexto e allocator antes de consultar estatísticas no Windows.
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model = (
        build_official_umamba_enc_2d(
            input_channels=4,
            num_classes=1,
            input_size=(patch_size, patch_size),
        )
        if kind == "umamba"
        else UNet(in_channels=4, out_channels=1)
    ).to(device)
    before = next(model.parameters()).detach().clone()
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
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Pesos não finitos após atualização.")
    if torch.equal(before, next(model.parameters()).detach()):
        raise RuntimeError("Nenhuma atualização no primeiro tensor de pesos.")
    torch.cuda.synchronize(device)
    try:
        mamba_version = version("mamba-ssm")
    except PackageNotFoundError:
        mamba_version = None
    return {
        "status": "technical_smoke_only_not_scientific_result",
        "model": "UMambaEnc_2d" if kind == "umamba" else "UNet",
        "runtime_preparation": runtime,
        "input_shape": list(images.shape),
        "output_shape": list(prediction.shape),
        "step_completed": True,
        "loss_is_finite": True,
        "gradients_are_finite": True,
        "weights_updated_and_finite": True,
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
    parser.add_argument("--model", choices=("unet", "umamba"), default="umamba")
    parser.add_argument(
        "--prepare-runtime",
        action="store_true",
        help="Validar stack já instalado e carregar arquitetura oficial fixada.",
    )
    args = parser.parse_args()
    failure = None
    try:
        report = run_gpu_smoke(args.patch_size, args.model, args.prepare_runtime)
    except (RuntimeError, ValueError, ImportError) as exc:
        failure = str(exc)
        report = {
            "status": "failed_preflight_not_training",
            "step_completed": False,
            "model": args.model,
            "patch_size": args.patch_size,
            "error": failure,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
