"""Treinamento único sobre a RGI de Guaxupé, com teste espacial independente.

Exemplos:
python -m scripts.train_emater_rgi --root /kaggle/input/NOME --model unet --epochs 20
python -m scripts.train_emater_rgi --root /kaggle/input/NOME --model umamba --epochs 20
O comando U-Mamba exige instalação validada do runtime CUDA oficial ANTES.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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


def train(root: Path, kind: str, epochs: int, output: Path, seed: int, batch_size: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pairs = find_rgi_pairs(root)
    train_ids, val_ids, test_ids = make_holdouts(pairs, seed=seed)
    dataset = EMATERRGIDataset(pairs)
    size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kind == "umamba" and device.type != "cuda":
        raise RuntimeError("U-Mamba requer runtime CUDA compatível.")

    def loader(ids, shuffle=False):
        return DataLoader(
            Subset(dataset, ids), batch_size=batch_size, shuffle=shuffle,
            num_workers=2, pin_memory=device.type == "cuda",
        )

    train_loader = loader(train_ids, shuffle=True)
    val_loader = loader(val_ids)
    test_loader = loader(test_ids)

    model = build_model(kind, size=size).to(device)
    criterion = BCEDiceLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output.mkdir(parents=True, exist_ok=True)
    ckpt = output / f"best_{kind}_rgi.pt"
    history = fit_model(
        model, train_loader, val_loader, criterion, optimizer,
        device, epochs=epochs, checkpoint_path=ckpt,
    )
    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = run_epoch(model, test_loader, criterion, device)
    report = {
        "model": kind,
        "stage": "official_rgi_training",
        "epochs_requested": epochs,
        "best_epoch": checkpoint["epoch"],
        "best_val_iou": checkpoint["val_iou"],
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids),
        "excluded_external": "tres_pontas_2024",
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": seed,
        "metrics": metrics,
        "checkpoint": str(ckpt),
    }
    (output / f"{kind}_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / f"{kind}_history.json").write_text(
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/tcc_rgi"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs e batch-size precisam ser >= 1")
    train(args.root, args.model, args.epochs, args.output, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
