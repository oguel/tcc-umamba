"""Auditoria da base oficial: agrega todos os recortes da RGI antes de treinar.

Uso (Kaggle):
python -m scripts.validate_emater_rgi --root /kaggle/input/NOME_DO_DATASET
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.emater_dataset import (
    EMATERRGIDataset,
    find_rgi_pairs,
    validate_mask_values,
)
from src.data.emater_spatial import spatial_folds


def audit(root: Path, sample_limit: int = 0) -> dict:
    import rasterio

    pairs = find_rgi_pairs(root)
    external_pairs = find_rgi_pairs(root, include_external=True)
    excluded = sorted(set(p.municipality for p in external_pairs) -
                      set(p.municipality for p in pairs))
    years = Counter(p.municipality for p in pairs)
    checked = pairs if sample_limit <= 0 else pairs[:sample_limit]

    labels = Counter()
    channels = Counter()
    crs = Counter()
    shapes = Counter()
    for pair in checked:
        with rasterio.open(pair.image) as image, rasterio.open(pair.mask) as mask:
            if image.crs != mask.crs or image.transform != mask.transform:
                raise ValueError(f"Grid divergente: {pair.image}")
            if (image.height, image.width) != (mask.height, mask.width):
                raise ValueError(f"Tamanho divergente: {pair.image}")
            values, counts = np.unique(mask.read(1), return_counts=True)
            validate_mask_values(values)
            labels.update({str(v): int(c) for v, c in zip(values, counts)})
            channels[image.count] += 1
            crs[str(image.crs)] += 1
            shapes[f"{image.height}x{image.width}"] += 1

    folds = spatial_folds(pairs)
    report = {
        "study_area": "Regiao Geografica Imediata de Guaxupe / 310044",
        "pairs_rgi": len(pairs),
        "pairs_all_folders": len(external_pairs),
        "external_municipalities_excluded": excluded,
        "pairs_by_municipality_year": dict(sorted(years.items())),
        "labels_pixel_counts_sample": dict(labels),
        "image_channel_counts_sample": dict(channels),
        "crs_counts_sample": dict(crs),
        "shapes_sample": dict(shapes),
        "samples_audited": len(checked),
        "spatial_folds": [{"train": len(t), "val": len(v)} for t, v in folds],
        "dataset_loader": EMATERRGIDataset.__name__,
        "training_completed": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.sample_limit)
    result = json.dumps(report, indent=2, ensure_ascii=False)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
