"""Aplica os pesos fixos a um mosaico Sentinel-2 da RGI de Guaxupé."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from math import ceil, floor
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode, urljoin
from urllib.request import urlopen

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from affine import Affine
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import array_bounds, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, transform_geom

from src.data.emater_dataset import normalize_sentinel_dn
from src.data.rgi_fingerprint import _file_sha256
from src.final_evaluation import CHECKPOINTS, THRESHOLD
from src.models.unet import UNet

CATALOG = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
DATES = "2024-07-01T00:00:00Z/2024-09-30T23:59:59Z"
CRS = "EPSG:32723"
REGION_CODE = "310044"
BOUNDARY = Path(__file__).resolve().parents[1] / (
    "data/external/ibge/mg_rg_immediatas_2025/mg_rg_immediatas_2025.shp"
)
IBGE = "https://servicodados.ibge.gov.br/api"
MAX_DATES = 4
PATCH_SIZE = 256
BANDS = dict(zip(("blue", "green", "red", "nir"), ("B02", "B03", "B04", "B08"), strict=True))


def region_geometry():
    boundaries = gpd.read_file(BOUNDARY)
    region = boundaries.loc[boundaries["CD_RGI"].astype(str) == REGION_CODE].to_crs(CRS)
    if len(region) != 1:
        raise ValueError("Esperada exatamente uma geometria da RGI 310044.")
    return region.geometry.iloc[0]


def grid_profile(region):
    west, south, east, north = region.bounds
    left, top = floor(west / 10) * 10, ceil(north / 10) * 10
    width = ceil((east - left) / (10 * PATCH_SIZE)) * PATCH_SIZE
    height = ceil((top - south) / (10 * PATCH_SIZE)) * PATCH_SIZE
    return {
        "crs": CRS,
        "transform": from_origin(left, top, 10, 10),
        "height": height,
        "width": width,
    }


def grid_mask(region, grid):
    return geometry_mask(
        [region],
        out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"],
        invert=True,
    )


def fetch_json(url):
    with urlopen(url, timeout=45) as response:
        return json.load(response)


def municipality_grid(grid, inside):
    members = fetch_json(f"{IBGE}/v1/localidades/regioes-imediatas/{REGION_CODE}/municipios")
    if len(members) != 9 or len({entry["id"] for entry in members}) != 9:
        raise ValueError("A lista de municípios diverge da composição de nove municípios da RGI.")
    shapes = []
    for index, member in enumerate(members, start=1):
        query = urlencode({"formato": "application/vnd.geo+json"})
        url = f"{IBGE}/v4/malhas/municipios/{member['id']}?{query}"
        member["geometry_url"] = url
        for feature in fetch_json(url)["features"]:
            shapes.append((transform_geom("EPSG:4326", CRS, feature["geometry"]), index))
    labels = rasterize(shapes, out_shape=inside.shape, transform=grid["transform"], dtype="uint8")
    labels[~inside] = 0
    if np.any(np.bincount(labels.ravel(), minlength=10)[1:] == 0):
        raise ValueError("Um município da RGI ficou sem pixels na malha municipal.")
    return labels, members


def coverage_by_municipality(labels, members, valid):
    total = np.bincount(labels.ravel(), minlength=len(members) + 1)
    covered = np.bincount(labels[valid], minlength=len(members) + 1)
    return [
        {
            "id": entry["id"],
            "nome": entry["nome"],
            "pixels": int(total[i]),
            "valid_pixels": int(covered[i]),
            "valid_fraction": float(covered[i] / total[i]),
        }
        for i, entry in enumerate(members, start=1)
    ]


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
    url = f"{CATALOG}/search?{query}"
    items = {}
    seen = set()
    while url:
        if not url.startswith(f"{CATALOG}/") or url in seen:
            raise ValueError("Paginação inesperada no catálogo.")
        seen.add(url)
        result = fetch_json(url)
        items.update((item["id"], item) for item in result["features"])
        following = next(
            (link["href"] for link in result.get("links", []) if link.get("rel") == "next"), None
        )
        url = urljoin(url, following) if following else None
    return sorted(
        items.values(),
        key=lambda item: (
            item["properties"]["eo:cloud_cover"]
            if item["properties"].get("eo:cloud_cover") is not None
            else 100,
            item["id"],
        ),
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


def load_mosaic(grid, region, inside, labels, members):
    """Prioriza a data com maior cobertura SCL e completa lacunas com outras datas."""
    by_day = defaultdict(list)
    for item in catalog_items(grid):
        if item["collection"] != COLLECTION:
            raise ValueError("Coleção inesperada.")
        if (set(BANDS) | {"scl"}).issubset(item["assets"]):
            by_day[item["properties"]["datetime"][:10]].append(item)
    scl_grid = {
        **grid,
        "width": grid["width"] // 2,
        "height": grid["height"] // 2,
        "transform": grid["transform"] * Affine.scale(2),
    }
    scl_inside = grid_mask(region, scl_grid)
    ranked = []
    image = np.zeros((4, *inside.shape), dtype=np.float32)
    source_index = np.zeros(inside.shape, dtype=np.uint16)
    sources = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="45"):
        for day, items in sorted(by_day.items()):
            print(f"Conferindo cobertura SCL de {day}", flush=True)
            clear = np.zeros_like(scl_inside)
            for item in items:
                clear |= np.isin(read_asset(item["assets"]["scl"], scl_grid), (4, 5, 6))
            fraction = float(np.count_nonzero(clear & scl_inside) / scl_inside.sum())
            if fraction > 0:
                ranked.append(
                    {
                        "day": day,
                        "scl_fraction_20m": fraction,
                        "scenes": [item["id"] for item in items],
                    }
                )
        ranked.sort(key=lambda row: (-row["scl_fraction_20m"], row["day"]))
        for candidate in ranked[:MAX_DATES]:
            for item in by_day[candidate["day"]]:
                assets = item["assets"]
                scl = read_asset(assets["scl"], grid)
                take = inside & (source_index == 0) & np.isin(scl, (4, 5, 6))
                if not take.any():
                    continue
                print(f"Compondo {item['id']}", flush=True)
                bands = []
                radiometry = {}
                for key, band in BANDS.items():
                    asset = assets[key]
                    metadata = asset["raster:bands"][0]
                    scale, offset = float(metadata["scale"]), float(metadata["offset"])
                    if (scale, offset, metadata.get("nodata")) != (0.0001, -0.1, 0):
                        raise ValueError(f"Radiometria inesperada em {band}: {metadata}")
                    raw = read_asset(asset, grid)
                    if raw.dtype != np.uint16:
                        raise ValueError(f"Esperado DN uint16 em {band}.")
                    take &= raw != 0
                    bands.append(raw)
                    radiometry[band] = {"scale": scale, "offset": offset, "url": asset["href"]}
                if take.any():
                    for channel, raw in enumerate(bands):
                        image[channel, take] = normalize_sentinel_dn(raw[take])
                    index = len(sources) + 1
                    source_index[take] = index
                    sources.append(
                        {
                            "index": index,
                            "pixels": int(take.sum()),
                            "stac": item,
                            "radiometry": radiometry,
                        }
                    )
                del bands, raw
            valid = source_index != 0
            coverage = coverage_by_municipality(labels, members, valid)
            if valid.sum() >= 0.99 * inside.sum() and all(
                entry["valid_fraction"] >= 0.98 for entry in coverage
            ):
                break
    if not sources:
        raise RuntimeError("Nenhuma cena cobriu pixels válidos na RGI no intervalo selecionado.")
    return image, source_index, sources, ranked


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


def compare_panel(path, image, inside, valid, disagreement):
    colors = ["#eeeeee", "#2563eb", "#ea580c", "#15803d", "#555555"]
    labels = ["Nenhuma", "Só U-Net", "Só U-Mamba", "Ambas", "Inválido"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    preview = np.s_[::6, ::6]
    rgb = np.clip(image[:, ::6, ::6][[2, 1, 0]].transpose(1, 2, 0) / 0.3, 0, 1)
    rgb[~valid[preview]] = 0.33
    rgb[~inside[preview]] = 1
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 em RGB")
    panel = np.ma.masked_where(~inside[preview], np.where(valid[preview], disagreement[preview], 4))
    axes[1].imshow(
        panel,
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
    region = region_geometry()
    grid = grid_profile(region)
    inside = grid_mask(region, grid)
    labels, members = municipality_grid(grid, inside)
    image, source_index, sources, ranked = load_mosaic(grid, region, inside, labels, members)
    valid = source_index != 0
    coverage = coverage_by_municipality(labels, members, valid)
    valid_fraction = float(valid.sum() / inside.sum())
    print(f"Cobertura válida da RGI: {valid_fraction:.1%}", flush=True)
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
            del probability
        disagreement = predictions["unet"].astype(np.uint8) + 2 * predictions["umamba"]
        write_tiff(
            output / "comparacao.tif",
            np.where(valid, disagreement, 255).astype(np.uint8),
            grid,
            255,
        )
        write_tiff(output / "cena_por_pixel.tif", source_index, grid, 0)
        write_tiff(
            output / "cobertura.tif",
            np.where(inside, np.where(valid, 1, 2), 0).astype(np.uint8),
            grid,
            0,
        )
        compare_panel(output / "comparacao.png", image, inside, valid, disagreement)
        report = {
            "stage": "regional_temporal_transfer_unlabeled",
            "region_code": REGION_CODE,
            "boundary": BOUNDARY.name,
            "collection": COLLECTION,
            "catalog": CATALOG,
            "search_interval": DATES,
            "selection": ranked,
            "max_dates": MAX_DATES,
            "bounds_utm": array_bounds(grid["height"], grid["width"], grid["transform"]),
            "crs": CRS,
            "scl_valid": [4, 5, 6],
            "threshold": THRESHOLD,
            "patch_size": PATCH_SIZE,
            "checkpoints": CHECKPOINTS,
            "runtime": runtime,
            "valid_pixels": int(valid.sum()),
            "region_pixels": int(inside.sum()),
            "valid_fraction": valid_fraction,
            "municipal_coverage": coverage,
            "municipal_boundary_sources": members,
            "region_pixels_unassigned_to_municipality": int((inside & (labels == 0)).sum()),
            "coverage_target_met": valid_fraction >= 0.99
            and all(row["valid_fraction"] >= 0.98 for row in coverage),
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
        (output / "cenas_stac.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
        output.rename(args.output)


if __name__ == "__main__":
    main()
