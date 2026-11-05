"""Persistência atômica e estado aleatório para retomada entre épocas."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def capture_rng(loaders) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": [
            loader.generator.get_state() if loader.generator is not None else None
            for loader in loaders
        ],
    }


def restore_rng(state: dict, loaders) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    for loader, generator_state in zip(loaders, state["loaders"], strict=True):
        if generator_state is not None and loader.generator is not None:
            loader.generator.set_state(generator_state.cpu())
