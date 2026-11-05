"""Auditoria integral pré-treinamento da base regional em GeoTIFF.

Execute ANTES de qualquer treinamento e persista o JSON no armazenamento oficial.
O relatório aplica às máscaras a mesma regra de NoData do Dataset PyTorch.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import hypot, isfinite
from pathlib import Path

import numpy as np

from src.data.emater_dataset import find_rgi_pairs, validate_mask_values
from src.data.rgi_fingerprint import dataset_fingerprint
from src.data.rgi_geographic_holdout import regional_split


def audit(
    root: Path,
    sample_limit: int = 0,
    include_external: bool = False,
    variant: str = "Patches_P512_S300_NN",
) -> dict:
    """Audita rasters, classes e folds; amostragem é apenas diagnóstico rápido."""
    import rasterio

    if sample_limit < 0:
        raise ValueError("sample_limit deve ser >= 0")
    pairs = find_rgi_pairs(root, variant=variant, include_external=include_external)
    all_pairs = find_rgi_pairs(root, variant=variant, include_external=True)
    checked = pairs if sample_limit <= 0 else pairs[:sample_limit]
    report_counts = Counter(p.municipality for p in pairs)
    totals: Counter[str] = Counter()
    channels: Counter[int] = Counter()
    crs: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    offsets: Counter[str] = Counter()
    warnings = []
    per_patch = []

    for pair in checked:
        with rasterio.open(pair.image) as image, rasterio.open(pair.mask) as mask:
            if image.crs != mask.crs or image.transform != mask.transform:
                raise ValueError(f"Grid não alinhado: {pair.image}")
            if image.shape != mask.shape or mask.count != 1:
                raise ValueError(f"Dimensões/bandas de máscara inválidas: {pair.mask}")
            if image.shape != (512, 512):
                raise ValueError(f"Patch deveria ser 512x512: {pair.image}")
            if image.count != 10:
                raise ValueError(f"Esperadas dez bandas espectrais: {pair.image}")
            if set(image.dtypes) != {"uint16"} or mask.dtypes != ("uint8",):
                raise ValueError(f"Imagem não é DN uint16: {pair.image}")
            if tuple(image.descriptions) != (
                "B02",
                "B03",
                "B04",
                "B08",
                "B05",
                "B06",
                "B07",
                "B8A",
                "B11",
                "B12",
            ):
                raise ValueError(f"Ordem de bandas incorreta: {pair.image}")
            if not image.crs or image.crs.to_epsg() != 32723:
                raise ValueError(f"CRS inesperado: {pair.image}")
            if (
                abs(image.transform.a - 10) > 1e-6
                or abs(image.transform.e + 10) > 1e-6
                or image.transform.b != 0
                or image.transform.d != 0
            ):
                raise ValueError(f"Resolução diferente de 10 m: {pair.image}")
            offset = image.tags().get("BOA_ADD_OFFSET")
            if offset is None:
                raise ValueError(f"BOA_ADD_OFFSET ausente: {pair.image}")
            try:
                if not isfinite(float(offset)):
                    raise ValueError("offset não finito")
            except ValueError as exc:
                raise ValueError(f"BOA_ADD_OFFSET inválido: {pair.image}") from exc

            raw_mask = mask.read(1)
            validate_mask_values(raw_mask)
            # Descomprimir TODAS as bandas, incluindo as seis fora da entrada do modelo.
            full_image = image.read()
            observed = full_image[:4]
            valid = np.all(observed != 0, axis=0)
            valid &= np.all(image.read_masks((1, 2, 3, 4)) > 0, axis=0)
            valid &= raw_mask != 255
            valid &= mask.read_masks(1) > 0
            coffee = int(np.count_nonzero((raw_mask == 1) & valid))
            noncoffee = int(np.count_nonzero((raw_mask == 0) & valid))
            ignored = int(np.count_nonzero(~valid))
            fake_background = int(np.count_nonzero((raw_mask == 0) & ~valid))
            channels[image.count] += 1
            crs[str(image.crs)] += 1
            shapes[f"{image.height}x{image.width}"] += 1
            offsets[str(offset)] += 1
            totals.update(
                coffee=coffee,
                noncoffee=noncoffee,
                ignored=ignored,
                invalid_labelled_background=fake_background,
            )
            if not coffee:
                totals["patches_without_coffee"] += 1
            if not noncoffee:
                totals["patches_without_noncoffee"] += 1
            if not coffee and not noncoffee:
                raise ValueError(f"Sem pixels rotulados válidos: {pair.image}")
            per_patch.append(
                {
                    "municipality": pair.municipality,
                    "filename": pair.image.name,
                    "coffee_pixels": coffee,
                    "noncoffee_pixels": noncoffee,
                    "ignored_pixels": ignored,
                    "invalid_labelled_background": fake_background,
                    "bounds": list(image.bounds),
                    "image_tags": image.tags(),
                    "mask_tags": mask.tags(),
                    "image_nodata": image.nodata,
                    "mask_nodata": mask.nodata,
                    "band_descriptions": list(image.descriptions),
                }
            )

    if totals["patches_without_coffee"] == 0:
        warnings.append(
            "Nenhum patch somente não-café na amostra: adicionar negativos "
            "geoespaciais independentes antes do treinamento oficial."
        )
    if totals["invalid_labelled_background"] > 0:
        warnings.append(
            "Máscaras Fora0_M0 contêm 0 fora da imagem: loader deverá "
            "remapear esses pixels para 255 (implementado)."
        )
    if sample_limit > 0 and len(checked) < len(pairs):
        warnings.append("Auditoria PARCIAL. Não usar como aprovação do treino oficial.")

    try:
        regional_pairs = find_rgi_pairs(root)
        proposal = regional_split(regional_pairs)
        with_bounds = {}
        for index, pair in enumerate(regional_pairs):
            with rasterio.open(pair.image) as raster:
                with_bounds[index] = raster.bounds

        def distance(i, j):
            a, b = with_bounds[i], with_bounds[j]
            return hypot(
                max(a.left - b.right, b.left - a.right, 0),
                max(a.bottom - b.top, b.bottom - a.top, 0),
            )

        separations = {
            f"{a}_to_{b}": min(
                distance(i, j) for i in getattr(proposal, a) for j in getattr(proposal, b)
            )
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        }
        split_info = {
            "status": "candidate_not_frozen",
            "method": "geographic_zones_with_overlap_purge",
            "train": len(proposal.train),
            "validation": len(proposal.validation),
            "test_locked": len(proposal.test),
            "purged": len(proposal.purged),
            "minimum_envelope_distance_m": separations,
            "patch_assignments": {
                group: [
                    f"{regional_pairs[i].municipality}/{regional_pairs[i].image.name}"
                    for i in getattr(proposal, group)
                ]
                for group in ("train", "validation", "test", "purged")
            },
            "train_by_municipality": dict(
                Counter(regional_pairs[i].municipality for i in proposal.train)
            ),
            "validation_by_municipality": dict(
                Counter(regional_pairs[i].municipality for i in proposal.validation)
            ),
            "test_by_municipality": dict(
                Counter(regional_pairs[i].municipality for i in proposal.test)
            ),
        }
    except ValueError as exc:
        split_info = {}
        warnings.append(f"Split espacial inviavel: {exc}")

    return {
        "study_area": "RGI de Guaxupe 310044",
        "dataset_variant": variant,
        "pairs_rgi": len(find_rgi_pairs(root)),
        "audit_scope": "official_nn_including_external" if include_external else "rgi_only",
        "pairs_all_folders": len(all_pairs),
        "external_municipalities_excluded": sorted(
            {p.municipality for p in all_pairs} - {p.municipality for p in pairs}
        ),
        "pairs_by_municipality_year": dict(sorted(report_counts.items())),
        "samples_audited": len(checked),
        "image_channel_counts": dict(channels),
        "crs_counts": dict(crs),
        "shape_counts": dict(shapes),
        "radiometric_offsets": dict(offsets),
        "pixel_counts_effective": dict(totals),
        "candidate_spatial_holdout": split_info,
        "dataset_fingerprint": dataset_fingerprint(checked),
        "fingerprint_scope": "full" if len(checked) == len(pairs) else "partial",
        "warnings": warnings,
        "preflight_approved": False,
        "approval_pending": "Divisão candidata; protocolo científico ainda não congelado.",
        "per_patch": per_patch,
        "training_completed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument(
        "--variant",
        default="Patches_P512_S300_NN",
        choices=("Patches_P512_S300_NN", "Patches_P512_S300_NN_Fora0_M0"),
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Auditar também Três Pontas; nunca incluí-la no split regional.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected-fingerprint",
        help="SHA256 obtido na auditoria anterior; falha se o dataset for diferente.",
    )
    args = parser.parse_args()
    report = audit(args.root, args.sample_limit, args.include_external, args.variant)
    if args.expected_fingerprint and report["dataset_fingerprint"] != args.expected_fingerprint:
        raise SystemExit("FINGERPRINT DIVERGENTE: dados diferentes da versão esperada.")
    result = json.dumps(report, ensure_ascii=False, indent=2)
    print(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    if not report["preflight_approved"]:
        print("AUDITORIA: pendencias identificadas; corrigir antes do treino OFICIAL.")


if __name__ == "__main__":
    main()
