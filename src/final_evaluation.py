"""Contrato final fixo e agregação geográfica; não seleciona nem treina modelos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from affine import Affine

from src.data.emater_dataset import RasterPair, find_rgi_pairs
from src.data.rgi_fingerprint import _file_sha256
from src.validation_comparison import compare_predictions
from src.validation_diagnostics import inspect_run, patch_key, validation_pairs

PROTOCOL_ID = "locked_test_v1_existing_checkpoints_20261106"
CHECKPOINTS = {
    "unet": "addd3b5f680c1495dfa73b7eae6332fe085015c8d3e7035d825dfab4adc1d2cf",
    "umamba": "038a100adc397ae7482bf12550f958c9046829d5ff3d4f16d56275f69a91f9bf",
}
THRESHOLD = 0.5
EXPECTED_CONTRACT = {
    "seed": 42,
    "batch_size": 1,
    "patch_size": 256,
    "augment": True,
    "train_views_per_image": 4,
    "num_workers": 0,
    "pilot_train_per_municipality": 0,
    "pilot_val_limit": 0,
    "run_mode": "training",
    "epoch_budget": 100,
    "optimizer": "AdamW",
    "lr": 0.0001,
    "loss": "BCE+Dice_ignore255",
    "torch": "2.10.0+cu126",
    "train_count": 89,
    "validation_count": 20,
}


def locked_test_pairs(pairs: list[RasterPair], splits: dict, digest: str) -> list[RasterPair]:
    # Verifica a partição INTEIRA e seu hash antes de selecionar o teste.
    validation_pairs(pairs, splits, digest)
    lookup = {f"{p.municipality}/{p.image.name}": p for p in pairs}
    return sorted(
        (lookup[patch_key(p)] for p in splits["test_locked"]),
        key=lambda p: (p.municipality, p.image.name),
    )


def audit_final_inputs(root: Path, runs: dict[str, Path]):
    if set(runs) != set(CHECKPOINTS):
        raise ValueError("São necessárias as duas execuções fixadas no protocolo.")
    audited = {}
    common_contract = None
    common_keys = None
    for kind, run in runs.items():
        _, checkpoint, report, _ = inspect_run(root, run)
        contract = report["run_contract"]
        if report["model"] != kind or any(
            contract.get(key) != value for key, value in EXPECTED_CONTRACT.items()
        ):
            raise ValueError("Execução difere da configuração final congelada.")
        checkpoint_path = run / f"best_{kind}_rgi.pt"
        if _file_sha256(checkpoint_path) != CHECKPOINTS[kind]:
            raise ValueError("Checkpoint difere dos pesos selecionados antes do teste.")
        common = {key: value for key, value in contract.items() if key != "model"}
        if common_contract is not None and common != common_contract:
            raise ValueError("Contratos das redes diferem além do modelo.")
        common_contract = common
        splits = json.loads((run / "splits.json").read_text(encoding="utf-8"))
        selected = locked_test_pairs(find_rgi_pairs(root), splits, contract["split_sha256"])
        keys = [(p.municipality, p.image.name) for p in selected]
        if len(selected) != 20 or (common_keys is not None and keys != common_keys):
            raise ValueError("Os modelos devem compartilhar os vinte patches reservados.")
        common_keys = keys
        audited[kind] = (selected, checkpoint, report)
    return audited


class SceneMosaic:
    """Média de probabilidades no mesmo pixel de cena; referência deve concordar.

    Instanciar separadamente por município/cena. Nunca combinar datas diferentes.
    O limite de alocação evita criar acidentalmente um mosaico regional enorme.
    """

    def __init__(self, grids: list[tuple[Affine, int, int]], crs: str):
        if not grids:
            raise ValueError("Nenhum grid para o mosaico.")
        self.crs = crs
        first = grids[0][0]
        if first.a != 10 or first.e != -10 or first.b != 0 or first.d != 0:
            raise ValueError("Requer grid norte-acima de 10m, sem rotação.")
        self.positions = []
        for transform, height, width in grids:
            if (transform.a, transform.b, transform.d, transform.e) != (10, 0, 0, -10):
                raise ValueError("Grids incompatíveis.")
            col = (transform.c - first.c) / first.a
            row = (transform.f - first.f) / first.e
            if abs(col - round(col)) > 1e-5 or abs(row - round(row)) > 1e-5:
                raise ValueError("Origem não alinhada ao grid comum.")
            if height <= 0 or width <= 0:
                raise ValueError("Dimensão de grid inválida.")
            self.positions.append((round(row), round(col), height, width))
        top = min(p[0] for p in self.positions)
        left = min(p[1] for p in self.positions)
        height = max(p[0] + p[2] for p in self.positions) - top
        width = max(p[1] + p[3] for p in self.positions) - left
        if height * width > 64_000_000:
            raise ValueError("Mosaico excede o limite de 64 milhões de pixels.")
        self.positions = [(r - top, c - left, h, w) for r, c, h, w in self.positions]
        self.transform = Affine(10, 0, first.c + left * 10, 0, -10, first.f - top * 10)
        self.target = np.full((height, width), 255, dtype=np.uint8)
        self.count = np.zeros((height, width), dtype=np.uint16)
        self.total = np.zeros((height, width), dtype=np.float64)
        self.added: set[int] = set()

    def add(self, index: int, target: np.ndarray, probability: np.ndarray):
        if index in self.added:
            raise ValueError("O mesmo patch não pode ser acumulado duas vezes.")
        row, col, height, width = self.positions[index]
        if target.shape != (height, width) or probability.shape != target.shape:
            raise ValueError("Shape difere do grid registrado.")
        if not np.isin(target, (0, 1, 255)).all():
            raise ValueError("Referência não binária.")
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Probabilidade deve ser finita e estar em [0,1].")
        area = np.s_[row : row + height, col : col + width]
        current = self.target[area]
        valid = target != 255
        if ((current != 255) & valid & (current != target)).any():
            raise ValueError("Referências conflitantes na sobreposição geográfica.")
        current[valid] = target[valid]
        self.count[area] += valid.astype(np.uint16)
        self.total[area] += np.where(valid, probability, 0)
        self.added.add(index)

    def finish(self):
        if len(self.added) != len(self.positions):
            raise ValueError("Mosaico incompleto.")
        valid = self.count > 0
        probability = np.zeros(self.count.shape, dtype=np.float32)
        np.divide(self.total, self.count, out=probability, where=valid, casting="unsafe")
        prediction = (probability >= THRESHOLD).astype(np.uint8)
        result = compare_predictions(self.target, prediction, prediction)["unet"]
        return probability, result
