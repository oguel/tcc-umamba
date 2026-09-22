"""Bootstrap do workspace, executado antes do import de src.

O módulo obtém o repositório público do TCC, extrai src/, data/external/ e
requirements-runtime.txt para o workspace e adiciona o workspace ao sys.path.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tarfile
import urllib.request

REPO_URL = "https://github.com/oguel/tcc-umamba"
REPO_ARCHIVE_DIR = "tcc-umamba-main"
WORKSPACE_COLAB = pathlib.Path("/content")
WORKSPACE_KAGGLE = pathlib.Path("/kaggle/working")
DRIVE_MOUNT_POINT = pathlib.Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT_POINT / "MyDrive"


def detect_platform() -> str:
    """Identifica a plataforma de execução: colab, kaggle ou local."""
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401

        return "colab"
    except Exception:
        return "local"


def workspace_root() -> pathlib.Path:
    """Retorna a raiz do workspace de acordo com a plataforma."""
    platform = detect_platform()
    if platform == "colab":
        return WORKSPACE_COLAB
    if platform == "kaggle":
        return WORKSPACE_KAGGLE
    return pathlib.Path.cwd()


def _download_and_extract(workspace: pathlib.Path) -> None:
    """Baixa o tarball do repositório e entrega os arquivos necessários."""
    archive = workspace / "_repo_bootstrap.tar.gz"
    extract_dir = workspace / "_repo_bootstrap"
    try:
        urllib.request.urlretrieve(f"{REPO_URL}/archive/refs/heads/main.tar.gz", archive)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(extract_dir)

        repo_dir = extract_dir / REPO_ARCHIVE_DIR

        for rel in ("src", "data", "requirements-runtime.txt"):
            target = workspace / rel
            if target.is_dir():
                shutil.rmtree(target)
            elif target.is_file():
                target.unlink(missing_ok=True)

        for rel in ("src", "requirements-runtime.txt"):
            source = repo_dir / rel
            if source.is_dir():
                shutil.copytree(source, workspace / rel)
            elif source.is_file():
                shutil.copy(source, workspace / rel)

        external = repo_dir / "data" / "external"
        if external.is_dir():
            shutil.copytree(external, workspace / "data" / "external")
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)


def _seed_drive_mirror(workspace: pathlib.Path) -> None:
    """Cria ou atualiza o espelho MyDrive/tcc/repo no Colab."""
    if detect_platform() != "colab":
        return

    from google.colab import drive  # noqa: PLC0415

    if not DRIVE_ROOT.exists():
        drive.mount(str(DRIVE_MOUNT_POINT))

    mirror = DRIVE_ROOT / "tcc" / "repo"

    def _refresh(source: pathlib.Path, target: pathlib.Path) -> None:
        if target.exists():
            shutil.rmtree(target)
        if source.exists():
            shutil.copytree(source, target)

    _refresh(workspace / "src", mirror / "src")
    _refresh(workspace / "data" / "external", mirror / "data" / "external")


def bootstrap_workspace(workspace: pathlib.Path | None = None) -> pathlib.Path:
    """Executa o bootstrap completo e retorna o caminho do workspace."""
    if workspace is None:
        workspace = workspace_root()

    _download_and_extract(workspace)
    _seed_drive_mirror(workspace)

    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))

    return workspace
