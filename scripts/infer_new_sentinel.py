"""Aplica os pesos fixos a um recorte de outra data do Sentinel-2."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.transform import array_bounds, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform, transform_bounds

from src.data.emater_dataset import normalize_sentinel_dn
from src.data.rgi_fingerprint import _file_sha256
from src.final_evaluation import CHECKPOINTS, THRESHOLD
from src.models.unet import UNet

CATALOG = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
DATES = "2024-07-01T00:00:00Z/2024-09-30T23:59:59Z"
CRS = "EPSG:32723"
SIZE = 1024
PATCH_SIZE = 256
BANDS = dict(zip(("blue", "green", "red", "nir"), ("B02", "B03", "B04", "B08"), strict=True))


def grid_profile():
    east, north = transform("EPSG:4326", CRS, [-46.74], [-21.30])
    left = round(east[0] / 10) * 10 - SIZE * 5
    top = round(north[0] / 10) * 10 + SIZE * 5
    return {"crs": CRS, "transform": from_origin(left, top, 10, 10), "height": SIZE, "width": SIZE}


def catalog_items(grid):
    bbox = transform_bounds(
        CRS, "EPSG:4326", *array_bounds(grid["height"], grid["width"], grid["transform"])
    )
    query = urlencode(
        {
            "collections": COLLECTION,
            "bbox": ",".join(map(str, bbox)),
            "datetime": DATES,
            "limit": 100,
        }
    )
    with urlopen(f"{CATALOG}/search?{query}", timeout=45) as response:
        result = json.load(response)
    if any(link.get("rel") == "next" for link in result.get("links", [])):
        raise RuntimeError("Busca paginada. Reduza o intervalo antes de selecionar a cena.")
    return sorted(
        result["features"],
        key=lambda item: (item["properties"].get("eo:cloud_cover", 100), item["id"]),
    )


def read_asset(asset, grid):
    href = asset["href"]
    if not href.startswith("https://"):
        raise ValueError("O catálogo deve fornecer um GeoTIFF público por HTTPS.")
    with rasterio.open(href) as source:
        if source.count != 1 or source.crs is None:
            raise ValueError("Asset sem banda única ou sem CRS.")
        with WarpedVRT(source, **grid, nodata=0) as vrt:
            return vrt.read(1, masked=True).filled(0)


def load_scene(grid):
    """Escolhe a cena pela cobertura de pixels válidos antes da inferência."""
    attempts = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="45"):
        for item in catalog_items(grid):
            if item["collection"] != COLLECTION:
                raise ValueError("Coleção inesperada.")
            assets = item["assets"]
            if not (set(BANDS) | {"scl"}).issubset(assets):
                continue
            print(f"Conferindo cena {item['id']}", flush=True)
            scl = read_asset(assets["scl"], grid)
            valid = np.isin(scl, (4, 5, 6))
            candidate = {"id": item["id"], "scl_clear_fraction": float(valid.mean())}
            attempts.append(candidate)
            if valid.mean() < 0.90:
                continue
            channels = []
            radiometry = {}
            for key, band in BANDS.items():
                asset = assets[key]
                metadata = asset["raster:bands"][0]
                scale, offset = float(metadata["scale"]), float(metadata["offset"])
                if (scale, offset, metadata.get("nodata")) != (0.0001, -0.1, 0):
                    raise ValueError(f"Radiometria inesperada em {band}: {metadata}")
                values = read_asset(asset, grid)
                if values.dtype != np.uint16:
                    raise ValueError(f"Esperado DN uint16 em {band}.")
                valid &= values != 0
                channels.append(normalize_sentinel_dn(values, offset=offset / scale))
                radiometry[band] = {"scale": scale, "offset": offset, "url": asset["href"]}
            candidate["valid_fraction"] = float(valid.mean())
            if valid.mean() >= 0.90:
                image = np.stack(channels)
                image[:, ~valid] = 0
                return image, valid, item, attempts, radiometry
    raise RuntimeError(f"Nenhuma cena com 90% de pixels válidos. Candidatas: {attempts}")


def predict(model, image, valid, device):
    probability = np.full(valid.shape, -9999, dtype=np.float32)
    with torch.inference_mode():
        for row in range(0, valid.shape[0], PATCH_SIZE):
            for col in range(0, valid.shape[1], PATCH_SIZE):
                window = np.s_[row : row + PATCH_SIZE, col : col + PATCH_SIZE]
                if not valid[window].any():
                    continue
                inputs = torch.from_numpy(image[:, window[0], window[1]][None]).to(device)
                logits = model(inputs)
                if logits.shape != (1, 1, PATCH_SIZE, PATCH_SIZE):
                    raise ValueError("Dimensão de saída inesperada.")
                values = logits.sigmoid()[0, 0].cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Previsões não finitas.")
                probability[window] = np.where(valid[window], values, -9999)
    return probability


def write_tiff(path, values, grid, nodata):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=1,
        dtype=values.dtype,
        nodata=nodata,
        compress="deflate",
        **grid,
    ) as output:
        output.write(values, 1)


def compare_panel(path, image, valid, disagreement):
    colors = ["#eeeeee", "#2563eb", "#ea580c", "#15803d", "#555555"]
    labels = ["Nenhuma", "Só U-Net", "Só U-Mamba", "Ambas", "Inválido"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    rgb = np.clip(image[[2, 1, 0]].transpose(1, 2, 0) / 0.3, 0, 1)
    rgb[~valid] = 0.33
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 em RGB")
    axes[1].imshow(
        np.where(valid, disagreement, 4),
        cmap=ListedColormap(colors),
        vmin=0,
        vmax=4,
        interpolation="nearest",
    )
    axes[1].set_title("Onde cada modelo prevê café")
    for ax in axes:
        ax.axis("off")
    fig.legend([Patch(color=color) for color in colors], labels, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unet-run", required=True, type=Path)
    parser.add_argument("--umamba-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or torch.__version__ != "2.10.0+cu126":
        raise RuntimeError("Use a GPU com o stack CUDA do notebook.")
    paths = {
        kind: folder / f"best_{kind}_rgi.pt"
        for kind, folder in (("unet", args.unet_run), ("umamba", args.umamba_run))
    }
    for kind, path in paths.items():
        if _file_sha256(path) != CHECKPOINTS[kind]:
            raise ValueError(f"Checkpoint diferente do fixado para {kind}.")
    from src.models.umamba import build_official_umamba_enc_2d
    from src.models.umamba_runtime import ensure_umamba_runtime

    runtime = ensure_umamba_runtime()
    grid = grid_profile()
    image, valid, item, attempts, radiometry = load_scene(grid)
    print(f"Cena selecionada: {item['id']} ({valid.mean():.1%} válidos)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        output = Path(temporary) / "result"
        output.mkdir()
        predictions = {}
        for kind, path in paths.items():
            model = (
                UNet(in_channels=4, out_channels=1)
                if kind == "unet"
                else build_official_umamba_enc_2d(
                    input_channels=4, num_classes=1, input_size=(PATCH_SIZE, PATCH_SIZE)
                )
            )
            checkpoint = torch.load(path, weights_only=True, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            probability = predict(model.to("cuda").eval(), image, valid, "cuda")
            del model, checkpoint
            predictions[kind] = probability >= THRESHOLD
            write_tiff(output / f"{kind}_probabilidade.tif", probability, grid, -9999)
            write_tiff(
                output / f"{kind}_cafe.tif",
                np.where(valid, predictions[kind], 255).astype(np.uint8),
                grid,
                255,
            )
        disagreement = predictions["unet"].astype(np.uint8) + 2 * predictions["umamba"]
        write_tiff(
            output / "comparacao.tif",
            np.where(valid, disagreement, 255).astype(np.uint8),
            grid,
            255,
        )
        compare_panel(output / "comparacao.png", image, valid, disagreement)
        report = {
            "stage": "temporal_transfer_unlabeled_pilot",
            "scene": item["id"],
            "datetime": item["properties"]["datetime"],
            "collection": COLLECTION,
            "catalog": CATALOG,
            "search_interval": DATES,
            "selection": attempts,
            "bounds_utm": array_bounds(SIZE, SIZE, grid["transform"]),
            "crs": CRS,
            "radiometry": radiometry,
            "scl_valid": [4, 5, 6],
            "threshold": THRESHOLD,
            "patch_size": PATCH_SIZE,
            "checkpoints": CHECKPOINTS,
            "runtime": runtime,
            "valid_pixels": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "prediction_counts": dict(
                zip(
                    ("neither", "unet_only", "umamba_only", "both"),
                    np.bincount(disagreement[valid], minlength=4).tolist(),
                    strict=True,
                )
            ),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }
        (output / "relatorio.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (output / "cena_stac.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        output.rename(args.output)


if __name__ == "__main__":
    main()
