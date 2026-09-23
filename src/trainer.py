"""Rotinas reutilizáveis de treinamento e avaliação."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import BinaryConfusion, metrics_from_confusion


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Executa uma época de treino ou avaliação."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    sample_count = 0
    confusion = BinaryConfusion()

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            sample_count += batch_size
            confusion.update(logits.detach(), masks)

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(sample_count, 1)
    metrics.update({key: float(value) for key, value in confusion.as_dict().items()})
    return metrics


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
) -> list[dict[str, Any]]:
    """Treina o modelo e salva o checkpoint com melhor IoU de validação."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_iou = -1.0

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - start

        peak_vram_mb = 0.0
        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        record: dict[str, Any] = {
            "epoch": epoch,
            "seconds": elapsed,
            "peak_vram_mb": peak_vram_mb,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"{elapsed:.1f}s | VRAM={peak_vram_mb:.1f} MB"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_iou": best_iou,
                },
                checkpoint_path,
            )

    return history
