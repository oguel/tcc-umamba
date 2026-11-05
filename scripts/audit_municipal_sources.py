"""Confere mosaicos de origem e cada janela NN sem modificar os arquivos."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from src.data.emater_dataset import find_rgi_pairs
from src.data.rgi_fingerprint import _file_sha256


def compare_window(source, patch, x: int, y: int) -> bool:
    """Compara grid e todos os pixels/bandas de uma janela com seu mosaico."""
    window = Window(x, y, patch.width, patch.height)
    if x < 0 or y < 0 or x + patch.width > source.width or y + patch.height > source.height:
        return False
    return (
        source.crs == patch.crs
        and source.window_transform(window) == patch.transform
        and source.count == patch.count
        and np.array_equal(source.read(window=window), patch.read())
    )


def audit_sources(root: Path) -> dict:
    pairs = find_rgi_pairs(root, include_external=True)
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for municipality in sorted({p.municipality for p in pairs}):
        files = [
            p
            for p in root.rglob("*.tif")
            if p.parent.name == "Dados_Raster" and p.parent.parent.name == municipality
        ]
        images = [p for p in files if "_S2_" in p.name]
        masks = [p for p in files if p.name.endswith("_Mascara_EMATER.tif")]
        inside = [p for p in files if p.name.endswith("_Dentro_Municipio.tif")]
        if len(images) != 1 or len(masks) != 1 or len(inside) != 1:
            failures.append(f"{municipality}: origem ausente/duplicada")
            continue
        record: dict[str, Any] = {
            "municipality": municipality,
            "files": [],
            "patch_pairs_verified": 0,
        }
        for path in files:
            with rasterio.open(path) as raster:
                # Leitura integral em blocos limita o consumo de memória.
                values = set()
                for _, window in raster.block_windows(1):
                    array = raster.read(window=window)
                    if raster.count == 1:
                        values.update(np.unique(array).tolist())
                record["files"].append(
                    {
                        "filename": path.name,
                        "sha256": _file_sha256(path),
                        "shape": list(raster.shape),
                        "bands": raster.count,
                        "crs": str(raster.crs),
                        "nodata": raster.nodata,
                        "labels": sorted(values) if raster.count == 1 else None,
                    }
                )
                allowed = {0, 1, 255} if "Mascara" in path.name else {0, 1}
                if raster.count == 1 and not values.issubset(allowed):
                    failures.append(f"{municipality}/{path.name}: rótulos inválidos")
        with (
            rasterio.open(images[0]) as source_image,
            rasterio.open(masks[0]) as source_mask,
            rasterio.open(inside[0]) as source_inside,
        ):
            if any(
                source_image.crs != r.crs
                or source_image.transform != r.transform
                or source_image.shape != r.shape
                for r in (source_mask, source_inside)
            ):
                failures.append(f"{municipality}: grids municipais divergentes")
            for pair in [p for p in pairs if p.municipality == municipality]:
                match = re.fullmatch(r"patch_y(\d+)_x(\d+)", pair.image.stem)
                if not match:
                    failures.append(f"Nome de patch inválido: {municipality}/{pair.image.name}")
                    continue
                y, x = map(int, match.groups())
                with rasterio.open(pair.image) as image, rasterio.open(pair.mask) as mask:
                    if not compare_window(source_image, image, x, y):
                        failures.append(
                            f"{municipality}/{pair.image.name}: imagem difere da origem"
                        )
                    elif not compare_window(source_mask, mask, x, y):
                        failures.append(
                            f"{municipality}/{pair.image.name}: máscara difere da origem"
                        )
                    else:
                        record["patch_pairs_verified"] += 1
        results.append(record)
    return {
        "stage": "data_integrity_only_not_training",
        "variant": "Patches_P512_S300_NN",
        "municipalities": results,
        "failures": failures,
        "pairs_verified": sum(r["patch_pairs_verified"] for r in results),
        "source_rasters_read": sum(len(r["files"]) for r in results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_sources(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "municipalities"}, ensure_ascii=False)
    )
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
