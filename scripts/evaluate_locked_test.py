"""Avalia os dois pesos fixos; padrão é preflight sem inferência do teste."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch

from scripts.inspect_validation import export_panel
from src.data.emater_dataset import EMATERRGIDataset
from src.data.rgi_fingerprint import _file_sha256
from src.final_evaluation import (
    CHECKPOINTS,
    PROTOCOL_ID,
    THRESHOLD,
    SceneMosaic,
    audit_final_inputs,
)
from src.metrics import BinaryConfusion, metrics_from_confusion
from src.models.unet import UNet
from src.training_state import save_json
from src.validation_comparison import compare_predictions


def protocol_record(audited):
    _, _, report = audited["unet"]
    return {
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": CHECKPOINTS,
        "dataset_fingerprint": report["dataset_fingerprint"],
        "split_sha256": report["run_contract"]["split_sha256"],
        "threshold": THRESHOLD,
        "primary": "pooled_patch_pixels_including_within_test_overlap",
        "secondary": "unique_pixels_per_municipality_scene_mean_probability_then_threshold",
        "selection": "best_validation_iou_already_fixed",
        "patch_size": 256,
        "evaluation_augmentation": False,
        "inference_batch_size": 1,
        "benchmark_warmup_synthetic_views": 3,
        "new_training": False,
        "test_used_for_tuning": False,
        "scope": "juruaia_2022_and_nova_resende_2023_not_entire_rgi_or_temporal_generalization",
    }


def prepare_mosaics(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        with rasterio.open(pair.image) as image:
            groups[pair.municipality].append((image.transform, image.height, image.width))
    mosaics = {}
    for city, grids in groups.items():
        members = [p for p in pairs if p.municipality == city]
        with rasterio.open(members[0].image) as image:
            crs = image.crs
        for pair in members:
            with rasterio.open(pair.image) as image:
                if image.crs != crs or image.shape != (512, 512):
                    raise ValueError("CRS/dimensão divergente na cena de teste.")
        mosaics[city] = SceneMosaic(grids, str(crs))
    if {city: len(grids) for city, grids in groups.items()} != {
        "juruaia_2022": 2,
        "nova_resende_2023": 18,
    }:
        raise ValueError("Composição do teste diverge do protocolo.")
    return mosaics


def export_mosaic(output, city, mosaic, probability):
    valid = mosaic.count > 0
    profile = {
        "driver": "GTiff",
        "height": mosaic.count.shape[0],
        "width": mosaic.count.shape[1],
        "count": 1,
        "crs": mosaic.crs,
        "transform": mosaic.transform,
        "compress": "deflate",
    }
    with rasterio.open(
        output / f"{city}_probabilidade.tif", "w", **profile, dtype="float32", nodata=-9999
    ) as dst:
        dst.write(np.where(valid, probability, -9999).astype(np.float32), 1)
    with rasterio.open(
        output / f"{city}_predicao.tif", "w", **profile, dtype="uint8", nodata=255
    ) as dst:
        dst.write(np.where(valid, probability >= THRESHOLD, 255).astype(np.uint8), 1)
    with rasterio.open(
        output / f"{city}_referencia.tif", "w", **profile, dtype="uint8", nodata=255
    ) as dst:
        dst.write(mosaic.target, 1)


def evaluate_model(kind, audited, output, device):
    pairs, checkpoint, report = audited
    mosaics = prepare_mosaics(pairs)
    runtime = None
    if kind == "umamba":
        from src.models.umamba import build_official_umamba_enc_2d
        from src.models.umamba_runtime import ensure_umamba_runtime

        runtime = ensure_umamba_runtime()
        model = build_official_umamba_enc_2d(
            input_channels=4, num_classes=1, input_size=(256, 256)
        ).to(device)
    else:
        model = UNet(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    parameter_count = sum(p.numel() for p in model.parameters())
    source = EMATERRGIDataset(pairs)
    output.mkdir(parents=True, exist_ok=False)
    panels = output / "paineis_teste"
    panels.mkdir()
    aggregate = BinaryConfusion()
    city_confusion = defaultdict(BinaryConfusion)
    city_indices = defaultdict(int)
    per_patch, forward_seconds = [], []
    with torch.inference_mode():
        warmup = torch.zeros((1, 4, 256, 256), device=device)
        for _ in range(3):
            model(warmup)
        del warmup
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        for index, pair in enumerate(pairs):
            image, target_tensor = source[index]
            target = target_tensor[0].numpy().astype(np.uint8)
            probability = np.zeros((512, 512), dtype=np.float32)
            for top in (0, 256):
                for left in (0, 256):
                    if not np.any(target[top : top + 256, left : left + 256] != 255):
                        continue
                    x = image[:, top : top + 256, left : left + 256][None].to(device)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    logits = model(x)
                    torch.cuda.synchronize(device)
                    forward_seconds.append(time.perf_counter() - start)
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Logits não finitos no teste.")
                    probability[top : top + 256, left : left + 256] = (
                        torch.sigmoid(logits)[0, 0].cpu().numpy()
                    )
                    del x, logits
            prediction = probability >= THRESHOLD
            values = compare_predictions(target, prediction, prediction)["unet"]
            for meter in (aggregate, city_confusion[pair.municipality]):
                for field, value in values["confusion"].items():
                    setattr(meter, field, getattr(meter, field) + value)
            mosaics[pair.municipality].add(city_indices[pair.municipality], target, probability)
            city_indices[pair.municipality] += 1
            name = f"{pair.municipality}_{pair.image.stem}"
            np.savez_compressed(
                panels / f"{name}.npz",
                probability=probability,
                prediction=prediction.astype(np.uint8),
                valid=target != 255,
            )
            export_panel(
                image.numpy(),
                target,
                probability,
                panels / f"{name}.png",
                f"{kind} • TESTE ESPACIAL FIXO • {pair.municipality} • {pair.image.stem}",
            )
            per_patch.append(
                {"municipality": pair.municipality, "filename": pair.image.name, **values}
            )
            print(f"{kind}: patch {index + 1}/{len(pairs)} processado", flush=True)
    peak_memory = torch.cuda.max_memory_allocated(device)
    unique = BinaryConfusion()
    unique_city = {}
    # Contagem de instâncias por patch (inclui interseções); não é área única.
    valid_instances = sum(aggregate.as_dict().values())
    unique_pixels = 0
    for city, mosaic in mosaics.items():
        probability, values = mosaic.finish()
        unique_city[city] = values
        unique_pixels += int((mosaic.count > 0).sum())
        for field, value in values["confusion"].items():
            setattr(unique, field, getattr(unique, field) + value)
        export_mosaic(output, city, mosaic, probability)
    result = {
        "model": kind,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": CHECKPOINTS[kind],
        "runtime": runtime,
        "primary_patch_pooled": {
            "metrics": metrics_from_confusion(aggregate),
            "confusion": aggregate.as_dict(),
        },
        "secondary_unique_scene_pixels": {
            "metrics": metrics_from_confusion(unique),
            "confusion": unique.as_dict(),
        },
        "by_municipality_patch_pooled": {
            city: {"metrics": metrics_from_confusion(meter), "confusion": meter.as_dict()}
            for city, meter in city_confusion.items()
        },
        "by_municipality_unique_scene_pixels": unique_city,
        "patches": len(pairs),
        "per_patch": per_patch,
        "valid_pixel_instances": valid_instances,
        "unique_scene_pixels": unique_pixels,
        "duplicate_pixel_instances": valid_instances - unique_pixels,
        "forward_benchmark": {
            "views_256": len(forward_seconds),
            "mean_ms_per_view": 1000 * float(np.mean(forward_seconds)),
            "median_ms_per_view": 1000 * float(np.median(forward_seconds)),
            "seconds_per_view": forward_seconds,
            "excludes": "io_transfers_sigmoid_mosaics_and_rendering",
        },
        "parameter_count": parameter_count,
        "baseline_allocated_mib": baseline_memory / 1024**2,
        "peak_allocated_mib": peak_memory / 1024**2,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "data_fingerprint": report["dataset_fingerprint"],
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    save_json(output / "resultado_modelo.json", result)
    return result


def run(root: Path, runs: dict[str, Path], output: Path, execute: bool = False):
    audited = audit_final_inputs(root, runs)
    # Metadados somente: valida os grids antes de permitir qualquer inferência.
    for value in audited.values():
        prepare_mosaics(value[0])
    protocol = protocol_record(audited)
    cuda_ready = torch.cuda.is_available() and platform.system() == "Linux"
    stack_ready = str(torch.__version__) == "2.10.0+cu126" and torch.version.cuda == "12.6"
    if execute and (not cuda_ready or not stack_ready):
        raise RuntimeError("Avaliação conjunta requer Linux/CUDA com torch 2.10.0+cu126.")
    if output.exists():
        raise FileExistsError("Preserve a avaliação existente; use uma pasta de saída nova.")
    if execute:
        # O teste não é aberto antes de verificar kernels e arquitetura da U-Mamba.
        from src.models.umamba_runtime import ensure_umamba_runtime

        ensure_umamba_runtime()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocolo.json", protocol)
    preflight = {
        "inputs_verified": True,
        "cuda_ready": cuda_ready,
        "torch_stack_ready": stack_ready,
        "test_patches_inferred": 0,
        "execution_requested": execute,
    }
    save_json(output / "preflight.json", preflight)
    if not execute:
        return preflight
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    save_json(output / "status.json", {"status": "running", "protocol_id": PROTOCOL_ID})
    results = {}
    try:
        for kind in ("unet", "umamba"):
            results[kind] = evaluate_model(kind, audited[kind], output / kind, torch.device("cuda"))
        for kind, folder in runs.items():
            if _file_sha256(folder / f"best_{kind}_rgi.pt") != CHECKPOINTS[kind]:
                raise ValueError("Checkpoint mudou durante a avaliação.")
        result = {
            "stage": "fixed_checkpoints_locked_spatial_test",
            "protocol": protocol,
            "code_revision": revision,
            "models": results,
            "test_evaluation_completed": True,
            "new_training_executed": False,
            "general_superiority_established": False,
        }
        save_json(output / "AVALIACAO_TESTE.json", result)
        save_json(
            output / "status.json",
            {
                "status": "completed",
                "test_evaluation_completed": True,
                "new_training_executed": False,
            },
        )
        return {"status": "completed", "results": str(output / "AVALIACAO_TESTE.json")}
    except Exception as error:
        save_json(
            output / "status.json",
            {
                "status": "failed",
                "error": str(error),
                "models_completed": list(results),
                "test_may_have_been_inferred": True,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "unet-run", "umamba-run", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Abre a primeira avaliação do teste com ambos os pesos fixos.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.root,
                {"unet": args.unet_run, "umamba": args.umamba_run},
                args.output,
                args.execute,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
