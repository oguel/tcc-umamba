"""Verifica dados, divisão e dispositivo sem executar treino ou teste cego."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import torch

from scripts.validate_emater_rgi import audit
from src.config import get_config
from src.training_state import save_json


def preflight(root: Path, model: str, output: Path) -> dict:
    config = get_config()["official_dataset"]
    report = {
        "model": model,
        "technical_ready": False,
        "scientific_evaluation_completed": False,
        "blind_test_evaluated": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }
    try:
        data = audit(root)
        save_json(output.parent / "audit.json", data)
        if data["dataset_fingerprint"] != config["expected_rgi_fingerprint"]:
            raise ValueError("Hash dos dados diferente do lote congelado.")
        split = data["candidate_spatial_holdout"]
        if split.get("split_sha256") != config["split"]["expected_split_sha256"]:
            raise ValueError("Divisão espacial não validada.")
        if not torch.cuda.is_available():
            raise RuntimeError("Selecione uma GPU CUDA; o treino não deve cair em CPU.")
        torch.ones(1, device="cuda").sum().item()
        if model == "umamba":
            from src.models.umamba_runtime import ensure_umamba_runtime

            report["umamba_runtime"] = ensure_umamba_runtime()
        report.update(
            technical_ready=True,
            dataset_fingerprint=data["dataset_fingerprint"],
            split=split,
            gpu=torch.cuda.get_device_name(0),
            limitations=data["warnings"],
        )
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        report["error"] = str(exc)
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", choices=("unet", "umamba"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = preflight(args.root, args.model, args.output)
    print({k: v for k, v in report.items() if k != "split"})
    if not report["technical_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
