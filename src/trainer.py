"""Rotinas reutilizáveis de treinamento e avaliação."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import BinaryConfusion, metrics_from_confusion
from src.training_state import capture_rng, restore_rng, save_checkpoint, save_json


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
    skipped_ignored_samples = 0
    confusion = BinaryConfusion()

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labeled = (masks != 255).flatten(start_dim=1).any(dim=1)
            skipped_ignored_samples += int((~labeled).sum().item())
            if not bool(labeled.any()):
                continue
            images = images[labeled]
            masks = masks[labeled]

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    "Loss NaN/Inf: verificar radiometria, rótulos, taxa de aprendizado "
                    "e pixels válidos antes de continuar o treinamento."
                )

            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf"), error_if_nonfinite=True
                )
                optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            sample_count += batch_size
            confusion.update(logits.detach(), masks)

    if sample_count == 0:
        raise ValueError("Epoca sem nenhum pixel rotulado; verificar masks/crops.")
    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / sample_count
    metrics["skipped_fully_ignored_views"] = float(skipped_ignored_samples)
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
    resume_path: str | Path | None = None,
    run_contract: dict | None = None,
) -> list[dict[str, Any]]:
    """Treina o modelo e salva o checkpoint com melhor IoU de validação."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_iou = -1.0
    best_checkpoint = None
    start_epoch = 1
    last_path = checkpoint_path.parent / "last.pt"
    if resume_path is not None:
        state = torch.load(resume_path, map_location=device, weights_only=True)
        if state.get("schema_version") != 1 or state.get("run_contract") != run_contract:
            raise ValueError("Checkpoint incompatível com contrato/versão desta execução.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        best_checkpoint = state["best_checkpoint"]
        best_iou = best_checkpoint["val_iou"]
        start_epoch = state["epoch"] + 1
        if epochs < state["epoch"]:
            raise ValueError("Épocas solicitadas anteriores ao checkpoint.")
        restore_rng(state["rng"], (train_loader, val_loader))
        save_checkpoint(checkpoint_path, best_checkpoint)

    for epoch in range(start_epoch, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
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
            best_checkpoint = {
                "model_state_dict": {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                },
                "epoch": epoch,
                "val_iou": best_iou,
                "run_contract": run_contract,
            }
            save_checkpoint(checkpoint_path, best_checkpoint)
        assert best_checkpoint is not None
        save_checkpoint(
            last_path,
            {
                "schema_version": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_checkpoint": best_checkpoint,
                "rng": capture_rng((train_loader, val_loader)),
                "run_contract": run_contract,
            },
        )
        save_json(checkpoint_path.parent / "history.json", history)
        save_json(
            checkpoint_path.parent / "progress.json",
            {
                "epochs_completed": epoch,
                "epochs_requested": epochs,
                "last_checkpoint": last_path.name,
                "best_epoch": best_checkpoint["epoch"],
                "blind_test_evaluated": False,
            },
        )

    return history
