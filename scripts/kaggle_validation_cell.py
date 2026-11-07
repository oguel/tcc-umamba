"""Executar no Kaggle para diagnosticar o checkpoint U-Mamba já treinado.

Usa outro checkout, não treina e não modifica os pesos nem o código de treino.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    work = Path("/kaggle/working")
    run = work / "tcc_runs/20261106T223530560665Z_umamba_seed42"
    if not (run / "best_umamba_rgi.pt").is_file():
        raise RuntimeError("Execute na sessão Kaggle que contém o checkpoint da U-Mamba.")
    repo = work / "tcc_diagnostico"
    branch = "feat/diagnostico-validacao-unet-20261106"
    if not repo.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                branch,
                "https://github.com/oguel/tcc-umamba.git",
                str(repo),
            ],
            check=True,
        )
    else:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
        if dirty.strip():
            raise RuntimeError(
                "Checkout de diagnóstico tem alterações; preserve-as antes de continuar."
            )
        subprocess.run(["git", "fetch", "origin", branch], cwd=repo, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    print("Código do diagnóstico:", revision, flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = work / f"validacao_umamba_{stamp}"
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.inspect_validation",
            "--root",
            "/kaggle/input",
            "--run-dir",
            str(run),
            "--output",
            str(output),
            "--device",
            "cuda",
            "--prepare-runtime",
        ],
        cwd=repo,
        check=True,
    )
    archive = shutil.make_archive(
        str(work / "diagnostico_UMAMBA_VALIDACAO"), "zip", root_dir=output
    )
    print("ZIP DO DIAGNÓSTICO:", archive, flush=True)
    print((output / "diagnostico_validacao.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
