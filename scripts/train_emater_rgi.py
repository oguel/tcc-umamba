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

from src.data.augmentation import RGIPatchViews
from src.data.emater_dataset import EMATERRGIDataset, find_rgi_pairs
from src.data.emater_spatial import spatial_folds
from src.losses import BCEDiceLoss
from src.models.unet import UNet
from src.trainer import fit_model, run_epoch


def make_holdouts(pairs, k=5, block_m=15000, seed=42):
    """Dois folds espaciais distintos: um teste cego, um validação.

    Calcula a interseção dos conjuntos permitidos nos dois holdouts, para
    remover também patches que cruzam os limites entre teste e validação.
    """
    folds = spatial_folds(pairs, k=k, block_m=block_m, seed=seed)
    permitted_from_test, test = map(set, folds[0])
    permitted_from_val, candidates_val = map(set, folds[1])
    val = sorted(candidates_val & permitted_from_test)
    train = sorted(permitted_from_test & permitted_from_val)
    test = sorted(test)
    if not (train and val and test):
        raise ValueError("Divisões vazias: ajuste a granularidade do bloco espacial.")
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise AssertionError("Vazamento entre splits.")
    return train, val, test


def build_model(kind: str, size: int) -> torch.nn.Module:
    if kind == "unet":
        return UNet(in_channels=4, out_channels=1)
    if kind == "umamba":
        from src.models.umamba import build_official_umamba_enc_2d
        return build_official_umamba_enc_2d(
            input_channels=4, num_classes=1, input_size=(size, size),
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
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pairs = find_rgi_pairs(root)
    train_ids, val_ids, test_ids = make_holdouts(pairs, seed=seed)
    dataset = EMATERRGIDataset(pairs)
    size = patch_size
    if train_views_per_image is None:
        train_views_per_image = 4 if size == 256 else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    test_loader = loader(test_ids)

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
        "blocks_m": 15000,
        "k": 5,
        "source_root": str(root),
        "patch_size": size,
        "augment_train_only": augment,
        "train_views_per_image": train_views_per_image,
        "train": [str(pairs[i].image.relative_to(root)) for i in train_ids],
        "validation": [str(pairs[i].image.relative_to(root)) for i in val_ids],
        "test_locked": [str(pairs[i].image.relative_to(root)) for i in test_ids],
    }
    (run_dir / "splits.json").write_text(
        json.dumps(split_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    ckpt = run_dir / f"best_{kind}_rgi.pt"
    history = fit_model(
        model, train_loader, val_loader, criterion, optimizer,
        device, epochs=epochs, checkpoint_path=ckpt,
    )
    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    # O teste cego só será aberto na avaliação oficial, uma vez por modelo
    # após congelar arquitetura, configurações e melhor checkpoint.
    metrics = run_epoch(model, test_loader, criterion, device) if evaluate_test else None
    report = {
        "model": kind,
        "stage": "final_test" if evaluate_test else "development_validation",
        "epochs_requested": epochs,
        "best_epoch": checkpoint["epoch"],
        "best_val_iou": checkpoint["val_iou"],
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids),
        "patch_size": size,
        "augmentation": "dihedral_flip_rotate90" if augment else "none",
        "train_views_per_image": train_views_per_image,
        "train_samples_per_epoch": len(train_loader.dataset),
        "validation_views": len(val_loader.dataset),
        "test_views": len(test_loader.dataset),
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
        "--augment", action="store_true",
        help="Apenas no treino: crops aleatorios, flips e rotacoes em 90 graus.",
    )
    parser.add_argument(
        "--train-views-per-image", type=int, choices=(1, 2, 3, 4),
        help="Padrao: 4 views para 256; 1 para 512.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rgi_runs"))
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Abrir teste cego apenas na execução final após congelar o protocolo.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs e batch-size devem ser >= 1; num-workers deve ser >= 0")
    train(
        args.root, args.model, args.epochs, args.output, args.seed, args.batch_size,
        patch_size=args.patch_size,
        augment=args.augment,
        train_views_per_image=args.train_views_per_image,
        num_workers=args.num_workers,
        evaluate_test=args.evaluate_test,
    )


if __name__ == "__main__":
    main()
