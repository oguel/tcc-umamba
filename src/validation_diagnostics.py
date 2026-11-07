"""Diagnóstico de validação com contrato congelado; não disponibiliza teste."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import torch

from src.config import get_config
from src.data.emater_dataset import RasterPair, find_rgi_pairs
from src.data.rgi_balanced_split import split_digest
from src.data.rgi_fingerprint import dataset_fingerprint
from src.data.rgi_geographic_holdout import RegionalSplit


def patch_key(path: str) -> str:
    source = PurePosixPath(path.replace("\\", "/"))
    if len(source.parts) < 4 or source.parent.name != "Imagens_Individuais":
        raise ValueError("Identidade de imagem inválida no registro espacial.")
    if source.parents[1].name != "Patches_P512_S300_NN":
        raise ValueError("Diagnóstico requer a variante NN oficial.")
    return f"{source.parents[2].name}/{source.name}"


def validation_pairs(
    pairs: list[RasterPair], splits: dict, expected_digest: str
) -> list[RasterPair]:
    by_key = {f"{p.municipality}/{p.image.name}": p for p in pairs}
    if len(by_key) != len(pairs):
        raise ValueError("Identidades locais duplicadas.")
    groups = {
        g: [patch_key(path) for path in splits[g]]
        for g in ("train", "validation", "test_locked", "purged")
    }
    assigned = [key for group in groups.values() for key in group]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(by_key):
        raise ValueError("Divisões sobrepostas, ausentes ou divergentes da base local.")
    keys = sorted(by_key)
    indices = {key: i for i, key in enumerate(keys)}
    result = RegionalSplit(
        *(
            tuple(indices[k] for k in groups[g])
            for g in ("train", "validation", "test_locked", "purged")
        )
    )
    if split_digest(keys, result) != expected_digest:
        raise ValueError("Atribuições diferem do hash espacial congelado.")
    return [by_key[key] for key in groups["validation"]]


def inspect_run(root: Path, run_dir: Path):
    def read(name):
        return json.loads((run_dir / name).read_text(encoding="utf-8"))

    report, history, contract = read("metrics.json"), read("history.json"), read("invocation.json")
    status, progress, splits = read("status.json"), read("progress.json"), read("splits.json")
    if (
        contract != report["run_contract"]
        or contract["model"] not in ("unet", "umamba")
        or report["model"] != contract["model"]
    ):
        raise ValueError("Contrato/modelo incompatível com diagnóstico U-Net/U-Mamba.")
    if status["status"] != "completed" or status["blind_test_evaluated"]:
        raise ValueError("Execução incompleta ou teste já avaliado; requer revisão.")
    if [row["epoch"] for row in history] != list(range(1, contract["epoch_budget"] + 1)):
        raise ValueError("Histórico incompleto ou fora de ordem.")
    config = get_config()["official_dataset"]
    pairs = find_rgi_pairs(root)
    fingerprint = dataset_fingerprint(pairs)
    if (
        fingerprint != contract["dataset_fingerprint"]
        or fingerprint != config["expected_rgi_fingerprint"]
    ):
        raise ValueError("Bytes dos dados não correspondem à execução congelada.")
    if contract["split_sha256"] != config["split"]["expected_split_sha256"]:
        raise ValueError("Versão espacial divergente da configuração congelada.")
    selected = validation_pairs(pairs, splits, contract["split_sha256"])
    best = torch.load(
        run_dir / f"best_{contract['model']}_rgi.pt", weights_only=True, map_location="cpu"
    )
    last = torch.load(run_dir / "last.pt", weights_only=True, map_location="cpu")
    if best["run_contract"] != contract or last["run_contract"] != contract:
        raise ValueError("Checkpoint e contrato não correspondem.")
    chosen = max(history, key=lambda row: row["val_iou"])
    if best["epoch"] != chosen["epoch"] or report["best_epoch"] != best["epoch"]:
        raise ValueError("Checkpoint não corresponde ao melhor IoU de validação.")
    if abs(best["val_iou"] - chosen["val_iou"]) > 1e-12:
        raise ValueError("Métrica do checkpoint divergente do histórico.")
    if last["epoch"] != len(history) or progress["epochs_completed"] != len(history):
        raise ValueError("Última época não corresponde aos registros.")
    if last["history"] != history:
        raise ValueError("Histórico do último checkpoint diverge do JSON.")
    for state in (best["model_state_dict"], last["model_state_dict"]):
        if not all(bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("Pesos não finitos.")
    return selected, best, report, history
