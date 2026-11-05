"""A prova CUDA nao deve ser marcada como executada em CPU."""

import pytest
import torch

from scripts.smoke_umamba_gpu import run_gpu_smoke


def test_prova_umamba_bloqueia_execucao_sem_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Nenhuma GPU CUDA"):
        run_gpu_smoke(256)
