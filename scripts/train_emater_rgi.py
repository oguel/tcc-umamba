"""Treinamento único sobre a RGI de Guaxupé, com teste espacial independente.

Exemplos:
python -m scripts.train_emater_rgi --root /kaggle/input/NOME --model unet --epochs 20
python -m scripts.train_emater_rgi --root /kaggle/input/NOME --model umamba --epochs 20
O comando U-Mamba exige instalação validada do runtime CUDA oficial ANTES.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import get_config
from src.data.augmentation import RGIPatchViews
from src.data.emater_dataset import EMATERRGIDataset, find_rgi_pairs
from src.data.rgi_fingerprint import dataset_fingerprint
from src.data.rgi_geographic_holdout import regional_split
from src.losses import BCEDiceLoss
from src.models.unet import UNet
from src.trainer import fit_model


def make_holdouts(pairs):
    """Proposta de holdout regional: teste totalmente fechado em pilotos."""
    result = regional_split(pairs)
    return list(result.train), list(result.validation), list(result.test), list(result.purged)


def build_model(kind: str, size: int) -> torch.nn.Module:
    if kind == "unet":
        return UNet(in_channels=4, out_channels=1)
    if kind == "umamba":
        from src.models.umamba import build_official_umamba_enc_2d

        return build_official_umamba_enc_2d(
            input_channels=4,
            num_classes=1,
            input_size=(size, size),
        )
    raise ValueError(f"Modelo desconhecido: {kind}")


def train(
    root: Path,
    kind: str,
    epochs: int,
    output: Path,
    seed: int,
    batch_size: int,
    patch_size: int = 512,
    augment: bool = False,
    train_views_per_image: int | None = None,
    num_workers: int = 2,
    evaluate_test: bool = False,
    expected_fingerprint: str | None = None,
    device_name: str = "auto",
    pilot_train_per_municipality: int = 0,
    pilot_val_limit: int = 0,
):
    if evaluate_test:
        raise ValueError(
            "Treinamento nunca avalia teste cego; use avaliador final separado "
            "somente depois de congelar o protocolo."
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pairs = find_rgi_pairs(root)
    fingerprint = dataset_fingerprint(pairs)
    if expected_fingerprint is None:
        expected_fingerprint = get_config()["official_dataset"]["expected_rgi_fingerprint"]
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise ValueError("Fingerprint divergente; piloto cancelado antes de carregar pixels.")
    train_ids, val_ids, test_ids, purged_ids = make_holdouts(pairs)
    full_train_count, full_val_count = len(train_ids), len(val_ids)
    if pilot_train_per_municipality:
        selected = []
        for municipality in sorted({pairs[i].municipality for i in train_ids}):
            selected.extend(
                [i for i in train_ids if pairs[i].municipality == municipality][
                    :pilot_train_per_municipality
                ]
            )
        train_ids = selected
    if pilot_val_limit:
        val_ids = val_ids[:pilot_val_limit]
    dataset = EMATERRGIDataset(pairs)
    size = patch_size
    if train_views_per_image is None:
        train_views_per_image = 4 if size == 256 else 1
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA solicitada, mas indisponível; não executar piloto em CPU silenciosamente."
        )
    if kind == "umamba" and device.type != "cuda":
        raise RuntimeError("U-Mamba requer runtime CUDA compatível.")

    def loader(ids, training=False):
        source = Subset(dataset, ids)
        views = RGIPatchViews(
            source,
            size=size,
            mode="train" if training else "eval",
            augment=augment if training else False,
            train_views_per_image=train_views_per_image if training else 1,
        )
        return DataLoader(
            views,
            batch_size=batch_size,
            shuffle=training,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(seed + (0 if training else 1)),
        )

    train_loader = loader(train_ids, training=True)
    val_loader = loader(val_ids)
    # Nao carregar o teste cego durante desenvolvimento.

    model = build_model(kind, size=size).to(device)
    criterion = BCEDiceLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Execuções independentes evitam sobrescrever pesos e métricas históricos.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output / f"{timestamp}_{kind}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    # A identidade espacial do teste é preservada, mas o conjunto NÃO é
    # avaliado durante exploração/seleção de hiperparâmetros.
    split_record = {
        "seed": seed,
        "split_method": get_config()["official_dataset"]["split"]["method"],
        "split_status": get_config()["official_dataset"]["split"]["status"],
        "split_sha256": get_config()["official_dataset"]["split"]["expected_split_sha256"],
        "dataset_fingerprint": fingerprint,
        "split_approved": False,  # aprovação científica não decorre do congelamento espacial
        "full_candidate_train_count": full_train_count,
        "full_candidate_validation_count": full_val_count,
        "pilot_train_per_municipality": pilot_train_per_municipality,
        "pilot_val_limit": pilot_val_limit,
        "source_root": str(root),
        "patch_size": size,
        "augment_train_only": augment,
        "train_views_per_image": train_views_per_image,
        "train": [str(pairs[i].image.relative_to(root)) for i in train_ids],
        "validation": [str(pairs[i].image.relative_to(root)) for i in val_ids],
        "test_locked": [str(pairs[i].image.relative_to(root)) for i in test_ids],
        "purged": [str(pairs[i].image.relative_to(root)) for i in purged_ids],
    }
    (run_dir / "splits.json").write_text(
        json.dumps(split_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    ckpt = run_dir / f"best_{kind}_rgi.pt"
    history = fit_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=epochs,
        checkpoint_path=ckpt,
    )
    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    # NUNCA inferir o teste nesta rotina. Criar script separado de avaliacao
    # final apenas apos congelar dataset, hiperparametros e checkpoints.
    metrics = None
    report = {
        "model": kind,
        "stage": "technical_pilot_not_scientific_result",
        "epochs_requested": epochs,
        "best_epoch": checkpoint["epoch"],
        "best_val_iou": checkpoint["val_iou"],
        "num_train": len(train_ids),
        "dataset_fingerprint": fingerprint,
        "num_val": len(val_ids),
        "num_test_locked": len(test_ids),
        "num_purged": len(purged_ids),
        "patch_size": size,
        "augmentation": "dihedral_flip_rotate90" if augment else "none",
        "train_views_per_image": train_views_per_image,
        "train_samples_per_epoch": len(train_loader.dataset),
        "validation_views": len(val_loader.dataset),
        "test_views_not_evaluated": len(test_ids) * (4 if size == 256 else 1),
        "excluded_external": "tres_pontas_2024",
        "device": str(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "seed": seed,
        "metrics": metrics,
        "checkpoint": str(ckpt),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("unet", "umamba"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, choices=(256, 512), default=512)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apenas no treino: crops aleatorios, flips e rotacoes em 90 graus.",
    )
    parser.add_argument(
        "--train-views-per-image",
        type=int,
        choices=(1, 2, 3, 4),
        help="Padrao: 4 views para 256; 1 para 512.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--expected-fingerprint")
    parser.add_argument("--pilot-train-per-municipality", type=int, default=0)
    parser.add_argument("--pilot-val-limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rgi_runs"))
    # Nao disponibilizar --evaluate-test aqui: evitar vazamento acidental.
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs e batch-size devem ser >= 1; num-workers deve ser >= 0")
    if args.pilot_train_per_municipality < 0 or args.pilot_val_limit < 0:
        parser.error("Limites do piloto devem ser >= 0")
    train(
        args.root,
        args.model,
        args.epochs,
        args.output,
        args.seed,
        args.batch_size,
        patch_size=args.patch_size,
        augment=args.augment,
        train_views_per_image=args.train_views_per_image,
        num_workers=args.num_workers,
        evaluate_test=False,
        expected_fingerprint=args.expected_fingerprint,
        device_name=args.device,
        pilot_train_per_municipality=args.pilot_train_per_municipality,
        pilot_val_limit=args.pilot_val_limit,
    )


if __name__ == "__main__":
    main()
