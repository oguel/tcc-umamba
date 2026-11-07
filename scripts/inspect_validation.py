"""Reproduz métricas e exporta painéis de TODOS os patches de validação.

Não aceita --split=test, não treina e não altera os checkpoints recebidos.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.data.emater_dataset import EMATERRGIDataset
from src.data.rgi_fingerprint import _file_sha256
from src.losses import BCEDiceLoss
from src.metrics import BinaryConfusion, metrics_from_confusion
from src.models.unet import UNet
from src.training_state import save_json
from src.validation_diagnostics import inspect_run


def export_panel(image, target, probability, path: Path, title: str):
    valid = target != 255
    prediction = probability >= 0.5
    label = np.where(valid, target, 2)
    predicted = np.where(valid, prediction.astype(np.uint8), 2)
    error = np.zeros(target.shape, dtype=np.uint8)
    error[valid & prediction & (target == 1)] = 1
    error[valid & prediction & (target == 0)] = 2
    error[valid & ~prediction & (target == 1)] = 3
    error[~valid] = 4
    rgb = np.clip(image[[2, 1, 0]].transpose(1, 2, 0) / 0.3, 0, 1)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    axes[0].imshow(rgb)
    classes = ListedColormap(["#e5e7eb", "#15803d", "#555555"])
    axes[1].imshow(label, cmap=classes, vmin=0, vmax=2)
    axes[2].imshow(predicted, cmap=classes, vmin=0, vmax=2)
    colors = ["#e5e7eb", "#15803d", "#ea580c", "#9333ea", "#555555"]
    axes[3].imshow(error, cmap=ListedColormap(colors), vmin=0, vmax=4)
    for ax, name in zip(
        axes, ("RGB ilustrativo", "Referência", "Previsão ≥ 0,5", "Erros"), strict=True
    ):
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.legend(
        handles=[
            Patch(color=color, label=name)
            for color, name in zip(
                colors,
                ("Não café correto", "Café correto", "Falso café", "Café omitido", "Ignorado"),
                strict=True,
            )
        ],
        loc="lower center",
        ncol=5,
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.13, top=0.86, wspace=0.03)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def diagnose(
    root: Path, run_dir: Path, output: Path, device_name: str = "cpu", prepare_runtime: bool = False
):
    pairs, checkpoint, report, history = inspect_run(root, run_dir)
    kind = report["model"]
    device = torch.device(device_name)
    runtime = None
    if kind == "umamba":
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "Diagnóstico U-Mamba requer CUDA; execute no runtime Linux de treino."
            )
        if prepare_runtime:
            from src.models.umamba_runtime import ensure_umamba_runtime

            runtime = ensure_umamba_runtime()
        from src.models.umamba import build_official_umamba_enc_2d

        model = build_official_umamba_enc_2d(
            input_channels=4, num_classes=1, input_size=(report["patch_size"], report["patch_size"])
        ).to(device)
    else:
        model = UNet(in_channels=4, out_channels=1).to(device)
    output.mkdir(parents=True, exist_ok=False)
    panels = output / "paineis_validacao"
    panels.mkdir()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    size = report["patch_size"]
    if size not in (256, 512):
        raise ValueError("Tamanho não suportado.")
    source = EMATERRGIDataset(pairs)  # Somente os 20 pares de validação.
    global_confusion = BinaryConfusion()
    by_city: defaultdict[str, BinaryConfusion] = defaultdict(BinaryConfusion)
    per_patch, loss_sum, views = [], 0.0, 0
    with torch.inference_mode():
        for index, pair in enumerate(pairs):
            image, target = source[index]
            probability = np.zeros((512, 512), dtype=np.float32)
            confusion = BinaryConfusion()
            for top in range(0, 512, size):
                for left in range(0, 512, size):
                    x = image[:, top : top + size, left : left + size][None].to(device)
                    y = target[:, top : top + size, left : left + size][None].to(device)
                    if not bool((y != 255).any()):
                        continue
                    logits = model(x)
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Predição não finita.")
                    loss_sum += float(BCEDiceLoss()(logits, y).item())
                    views += 1
                    for meter in (confusion, global_confusion, by_city[pair.municipality]):
                        meter.update(logits, y)
                    probability[top : top + size, left : left + size] = (
                        torch.sigmoid(logits)[0, 0].cpu().numpy()
                    )
            values = metrics_from_confusion(confusion)
            name = f"{pair.municipality}_{pair.image.stem}"
            export_panel(
                image.numpy(),
                target[0].numpy(),
                probability,
                panels / f"{name}.png",
                f"{'U-Mamba' if kind == 'umamba' else 'U-Net'} • VALIDAÇÃO • "
                f"{pair.municipality} • {pair.image.stem} • "
                f"IoU {100 * values['iou']:.1f}%",
            )
            np.savez_compressed(
                panels / f"{name}.npz",
                probability=probability,
                prediction=(probability >= 0.5).astype(np.uint8),
                valid=target[0].numpy() != 255,
            )
            per_patch.append(
                {
                    "municipality": pair.municipality,
                    "filename": pair.image.name,
                    **values,
                    "confusion": confusion.as_dict(),
                }
            )
            print(
                f"Validação {index + 1}/{len(pairs)}: {pair.municipality} IoU={values['iou']:.4f}",
                flush=True,
            )
    result = {
        "stage": "validation_diagnostics_not_blind_test",
        "model": kind,
        "runtime": runtime,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": _file_sha256(run_dir / f"best_{kind}_rgi.pt"),
        "last_checkpoint_sha256": _file_sha256(run_dir / "last.pt"),
        "data_fingerprint": report["dataset_fingerprint"],
        "split_sha256": report["run_contract"]["split_sha256"],
        "device": str(device),
        "patches_inferred": len(pairs),
        "valid_views": views,
        "test_patches_inferred": 0,
        "threshold": 0.5,
        "metrics": metrics_from_confusion(global_confusion),
        "confusion": global_confusion.as_dict(),
        "mean_view_loss": loss_sum / views,
        "reported_val_iou": report["best_val_iou"],
        "by_municipality": {
            city: {**metrics_from_confusion(meter), "confusion": meter.as_dict()}
            for city, meter in sorted(by_city.items())
        },
        "per_patch": per_patch,
        "epochs_in_history": len(history),
        "all_validation_patches_exported": True,
        "spatial_overlap_within_validation_not_deduplicated": True,
    }
    result["iou_absolute_difference"] = abs(result["metrics"]["iou"] - result["reported_val_iou"])
    save_json(output / "diagnostico_validacao.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prepare-runtime", action="store_true")
    args = parser.parse_args()
    result = diagnose(args.root, args.run_dir, args.output, args.device, args.prepare_runtime)
    print({k: v for k, v in result.items() if k not in ("per_patch", "by_municipality")})


if __name__ == "__main__":
    main()
