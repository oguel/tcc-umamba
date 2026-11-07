"""Análise exploratória das bandas e erros da validação; sem modelo ou teste."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

from src.data.emater_dataset import EMATERRGIDataset, find_rgi_pairs
from src.data.rgi_fingerprint import dataset_fingerprint
from src.final_evaluation import CHECKPOINTS
from src.validation_diagnostics import patch_key, validation_pairs


def cm(target, prediction, included):
    if target.shape != prediction.shape or included.shape != target.shape:
        raise ValueError("Shapes divergentes.")
    if (
        not np.isin(target, (0, 1, 255)).all()
        or not np.isin(prediction[target != 255], (0, 1)).all()
    ):
        raise ValueError("Classes inválidas.")
    valid = included & (target != 255)
    bins = np.bincount(target[valid].astype(int) * 2 + prediction[valid].astype(int), minlength=4)
    return dict(zip(("tn", "fp", "fn", "tp"), map(int, bins), strict=True))


def spectral_strata(image, target):
    valid = target != 255
    denominator = image[3] + image[2]
    ndvi = np.zeros(target.shape, dtype=np.float32)
    np.divide(image[3] - image[2], denominator, out=ndvi, where=denominator > 0)
    brightness = image[:3].mean(axis=0)
    # Bordas apenas entre rótulos 0/1 válidos; 255 não cria contorno de classe.
    boundary = np.zeros(target.shape, dtype=bool)
    changed = valid[1:] & valid[:-1] & (target[1:] != target[:-1])
    boundary[1:] |= changed
    boundary[:-1] |= changed
    changed = valid[:, 1:] & valid[:, :-1] & (target[:, 1:] != target[:, :-1])
    boundary[:, 1:] |= changed
    boundary[:, :-1] |= changed
    distance = (
        distance_transform_edt(~boundary) if boundary.any() else np.full(target.shape, np.inf)
    )
    strata = {
        "all": valid,
        "ndvi_undefined": valid & (denominator == 0),
        "boundary_within_2_pixels": valid & (distance <= 2),
        "interior_more_than_2_pixels": valid & (distance > 2),
    }
    for lower, upper in [(-1, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.00001)]:
        strata[f"ndvi_{lower}_{min(upper, 1)}"] = (
            valid & (denominator > 0) & (ndvi >= lower) & (ndvi < upper)
        )
    for lower, upper in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.00001)]:
        strata[f"brightness_{lower}_{min(upper, 1)}"] = (
            valid & (brightness >= lower) & (brightness < upper)
        )
    return strata


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "unet-run", "unet-diagnostic", "umamba-diagnostic", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError("Use uma saída nova para preservar análises anteriores.")
    root = args.root
    pairs = find_rgi_pairs(root)
    split_file = args.unet_run / "splits.json"
    splits = json.loads(split_file.read_text())
    selected = validation_pairs(
        pairs, splits, "1d7327d89c0abc795364a9751223fbd6725d1d0997a080e9a07cc4d98cc02f72"
    )
    # Identidade estável: município/ano e filename; não tratar partes 001/002 como grupos.
    train_names = {patch_key(path) for path in splits["train"]}
    training = [p for p in pairs if p.municipality + "/" + p.image.name in train_names]
    if len(training) != 89 or len(selected) != 20:
        raise ValueError("Contagem de patches diverge da divisão congelada.")
    directories = {
        "unet": args.unet_diagnostic / "paineis_validacao",
        "umamba": args.umamba_diagnostic / "paineis_validacao",
    }
    reference = {}
    fingerprint = dataset_fingerprint(pairs)
    for kind, folder in directories.items():
        reference[kind] = json.loads(
            (folder.parent / "diagnostico_validacao.json").read_text(encoding="utf-8")
        )
        if reference[kind]["checkpoint_sha256"] != CHECKPOINTS[kind]:
            raise ValueError("Diagnóstico não corresponde ao checkpoint selecionado.")
        if reference[kind]["split_sha256"] != splits["split_sha256"]:
            raise ValueError("Divisão divergente no diagnóstico.")
        if reference[kind]["test_patches_inferred"] != 0:
            raise ValueError("Somente diagnósticos de validação são permitidos.")
        if fingerprint != reference[kind]["data_fingerprint"]:
            raise ValueError("Dados divergentes do diagnóstico.")
    groups = defaultdict(lambda: {"unet": Counter(), "umamba": Counter()})
    band_statistics = {}

    for role, members in [("train", training), ("validation", selected)]:
        stats = {band: Counter() for band in ("B02", "B03", "B04", "B08")}
        dataset = EMATERRGIDataset(members)
        for index, pair in enumerate(members):
            image, mask = dataset[index]
            image, target = image.numpy(), mask[0].numpy().astype(np.uint8)
            valid = target != 255
            with rasterio.open(pair.image) as raster:
                raw = raster.read().astype(np.float64)
                offset = float(raster.tags()["BOA_ADD_OFFSET"])
            for band_index, band in enumerate(stats):
                values = image[band_index][valid].astype(np.float64)
                stats[band]["valid_pixel_instances"] += int(valid.sum())
                stats[band]["sum"] += float(values.sum())
                stats[band]["sum_squared"] += float((values * values).sum())
                stats[band]["clipped_below_zero"] += int(
                    ((raw[band_index] + offset < 0) & valid).sum()
                )
                stats[band]["clipped_above_one"] += int(
                    ((raw[band_index] + offset > 10000) & valid).sum()
                )
            if role == "train":
                continue
            predictions = {}
            for kind, folder in directories.items():
                with np.load(
                    folder / f"{pair.municipality}_{pair.image.stem}.npz", allow_pickle=False
                ) as array:
                    probability = array["probability"]
                    if (
                        probability.shape != target.shape
                        or not np.array_equal(array["valid"], valid)
                        or not np.array_equal(array["prediction"], probability >= 0.5)
                        or not np.isfinite(probability).all()
                        or ((probability < 0) | (probability > 1)).any()
                    ):
                        raise ValueError("Grid, validade ou probabilidades divergentes.")
                    predictions[kind] = array["prediction"].copy()
            strata = spectral_strata(image, target)
            for label, included in strata.items():
                for kind, prediction in predictions.items():
                    groups[label][kind].update(cm(target, prediction, included))
        band_statistics[role] = {}
        for band, stat in stats.items():
            n = stat["valid_pixel_instances"]
            mean = stat["sum"] / n
            band_statistics[role][band] = {
                "valid_pixel_instances": n,
                "mean_reflectance": mean,
                "std_reflectance": float(np.sqrt(max(0, stat["sum_squared"] / n - mean**2))),
                "clipped_below_zero_count": stat["clipped_below_zero"],
                "clipped_above_one_count": stat["clipped_above_one"],
                "clipped_below_zero_fraction": stat["clipped_below_zero"] / n,
                "clipped_above_one_fraction": stat["clipped_above_one"] / n,
            }

    result = {}
    for label, models in groups.items():
        result[label] = {}
        for kind, c in models.items():
            positives, negatives = c["tp"] + c["fn"], c["tn"] + c["fp"]
            result[label][kind] = {
                "confusion": dict(c),
                "coffee_pixels": positives,
                "noncoffee_pixels": negatives,
                "false_positive_rate": c["fp"] / negatives if negatives else None,
                "false_negative_rate": c["fn"] / positives if positives else None,
                "iou": c["tp"] / (c["tp"] + c["fp"] + c["fn"])
                if c["tp"] + c["fp"] + c["fn"]
                else None,
            }
    # A análise deve reconstruir exatamente a validação já publicada.
    for kind in directories:
        if result["all"][kind]["confusion"] != reference[kind]["confusion"]:
            raise ValueError("A matriz global não reproduz o diagnóstico auditado.")
    report = {
        "stage": "exploratory_associations_not_causal_experiment",
        "data_fingerprint": fingerprint,
        "split_sha256": splits["split_sha256"],
        "checkpoint_sha256": CHECKPOINTS,
        "train_patches": len(training),
        "validation_patches": len(selected),
        "test_patches_analyzed": 0,
        "new_inference": False,
        "new_training": False,
        "overlapping_patch_pixels_not_deduplicated": True,
        "brightness_definition": "mean_normalized_B02_B03_B04_not_a_shadow_label",
        "ndvi_definition": "(B08-B04)/(B08+B04)_after_frozen_normalization",
        "boundary_definition": "valid_0_1_transitions_4_neighbors_distance_euclidean_pixels",
        "band_statistics": band_statistics,
        "validation_strata": result,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "ANALISE_ESPECTRAL_VALIDACAO.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
