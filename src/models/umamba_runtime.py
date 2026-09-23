"""Runtime reprodutível para U-Mamba no Google Colab.

O Colab atual pode fornecer PyTorch mais novo que a matriz de wheels CUDA do
mamba-ssm. Para evitar compilação local longa do selective_scan_cuda, este
módulo fixa um stack com wheel pré-compilada:
- PyTorch 2.10.0 + CUDA 12.6
- mamba-ssm 2.3.2.post1, wheel cu12 / torch2.10
- dynamic-network-architectures 0.3.1

A arquitetura UMambaEnc_2d continua vindo do repositório oficial U-Mamba,
fixada em commit específico para reprodutibilidade.
"""

from __future__ import annotations

import importlib
import importlib.util
from importlib.metadata import PackageNotFoundError, distribution, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
from urllib.parse import unquote

OFFICIAL_UMAMBA_REPOSITORY = "https://github.com/bowang-lab/U-Mamba"
OFFICIAL_UMAMBA_COMMIT = "28459e33ca03769800dd35e23c6e62491d1925b5"
OFFICIAL_ARCH_URL = (
    "https://raw.githubusercontent.com/bowang-lab/U-Mamba/"
    f"{OFFICIAL_UMAMBA_COMMIT}/umamba/nnunetv2/nets/UMambaEnc_2d.py"
)

TORCH_VERSION = "2.10.0"
TORCH_LOCAL_PREFIX = "2.10.0+cu126"
TORCHVISION_VERSION = "0.25.0"
TORCHVISION_LOCAL_PREFIX = "0.25.0+cu126"
TORCHAUDIO_VERSION = "2.10.0"
TORCHAUDIO_LOCAL_PREFIX = "2.10.0+cu126"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"
MAMBA_SSM_VERSION = "2.3.2.post1"
DNA_VERSION = "0.3.1"

RUNTIME_DIR = Path("/content/umamba_runtime") if Path("/content").exists() else Path(".umamba_runtime")
STANDALONE_MODULE_NAME = "umamba_official_standalone"
STANDALONE_FILE = RUNTIME_DIR / f"{STANDALONE_MODULE_NAME}.py"


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _pip_install(args: list[str], timeout_seconds: int = 1200) -> None:
    command = [sys.executable, "-m", "pip", "install", *args]
    print("$", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Instalação excedeu {timeout_seconds // 60} minutos e foi interrompida: "
            f"{' '.join(command)}"
        ) from exc


def _mamba_wheel_filename() -> str:
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if py_tag not in {"cp310", "cp311", "cp312", "cp313"}:
        raise RuntimeError(
            f"Python {sys.version_info.major}.{sys.version_info.minor} não possui wheel "
            "pré-compilada selecionada por este projeto."
        )
    if platform.machine() != "x86_64":
        raise RuntimeError(f"Arquitetura não suportada por este setup: {platform.machine()}")

    return (
        f"mamba_ssm-{MAMBA_SSM_VERSION}+cu12torch2.10cxx11abiTRUE-"
        f"{py_tag}-{py_tag}-linux_x86_64.whl"
    )


def _mamba_wheel_url() -> str:
    encoded = _mamba_wheel_filename().replace("+", "%2B")
    return (
        "https://github.com/state-spaces/mamba/releases/download/"
        f"v{MAMBA_SSM_VERSION}/{encoded}"
    )


def _mamba_install_is_target() -> bool:
    """Confirma que mamba-ssm veio da wheel binária selecionada.

    O campo Version do pacote não preserva necessariamente o sufixo local do
    nome do arquivo da wheel. Por isso a validação usa PEP 610 direct_url.json.
    """
    installed = _installed_version("mamba-ssm")
    if installed is None or not installed.startswith(MAMBA_SSM_VERSION):
        return False

    try:
        dist = distribution("mamba-ssm")
        direct_url_raw = dist.read_text("direct_url.json")
    except PackageNotFoundError:
        return False

    if not direct_url_raw:
        return False

    try:
        direct_url = json.loads(direct_url_raw).get("url", "")
    except json.JSONDecodeError:
        return False

    return _mamba_wheel_filename() in unquote(direct_url)


def stack_needs_restart() -> bool:
    """Indica se o PyTorch instalado precisa ser trocado antes de importar torch."""
    installed = _installed_version("torch")
    return installed is None or not installed.startswith(TORCH_LOCAL_PREFIX)


def install_prebuilt_colab_stack() -> bool:
    """Instala o stack binário do Colab.

    Retorna True quando o PyTorch foi alterado e o kernel deve ser reiniciado.
    """
    torch_before = _installed_version("torch")
    torch_changed = torch_before is None or not torch_before.startswith(TORCH_LOCAL_PREFIX)

    if torch_changed:
        print(
            f"PyTorch atual: {torch_before}. Instalando torch=={TORCH_VERSION} com CUDA 12.6 "
            "para usar uma wheel Mamba pré-compilada.",
            flush=True,
        )
        _pip_install([
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            f"torchaudio=={TORCHAUDIO_VERSION}",
            "--index-url",
            TORCH_INDEX_URL,
        ])
    else:
        print(f"PyTorch compatível já instalado: {torch_before}", flush=True)
        torchvision_before = _installed_version("torchvision")
        torchaudio_before = _installed_version("torchaudio")
        if (
            torchvision_before is None
            or not torchvision_before.startswith(TORCHVISION_LOCAL_PREFIX)
            or torchaudio_before is None
            or not torchaudio_before.startswith(TORCHAUDIO_LOCAL_PREFIX)
        ):
            _pip_install([
                f"torchvision=={TORCHVISION_VERSION}",
                f"torchaudio=={TORCHAUDIO_VERSION}",
                "--index-url",
                TORCH_INDEX_URL,
            ])

    if _installed_version("dynamic-network-architectures") != DNA_VERSION:
        _pip_install([f"dynamic-network-architectures=={DNA_VERSION}"])

    for package in ("einops", "ninja", "packaging", "transformers", "huggingface-hub"):
        if _installed_version(package) is None:
            _pip_install([package])

    mamba_before = _installed_version("mamba-ssm")
    if not _mamba_install_is_target():
        wheel_url = _mamba_wheel_url()
        print(
            "Instalando wheel CUDA pré-compilada do mamba-ssm; "
            "não haverá compilação local do selective_scan_cuda.",
            flush=True,
        )
        _pip_install(["--force-reinstall", "--no-deps", wheel_url])
    else:
        print(f"mamba-ssm já instalado: {mamba_before}", flush=True)

    return torch_changed


def restart_colab_runtime() -> None:
    """Encerra o kernel para que o Colab recarregue o PyTorch recém-instalado."""
    print(
        "\nStack instalado. O runtime será reiniciado agora. "
        "Quando o Colab reconectar, execute o notebook 11 novamente desde a primeira célula.",
        flush=True,
    )
    os.kill(os.getpid(), 9)


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está ativa. No Google Colab, use "
            "'Ambiente de execução > Alterar tipo de ambiente de execução > GPU' "
            "e execute novamente."
        )
    if torch.version.cuda is None:
        raise RuntimeError("O PyTorch carregado não possui suporte CUDA.")


def _mamba_forward_sanity() -> None:
    import torch
    from mamba_ssm import Mamba

    model = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda()
    x = torch.randn(1, 64, 16, device="cuda")
    with torch.inference_mode():
        y = model(x)
    if y.shape != x.shape:
        raise RuntimeError(f"Saída inesperada do bloco Mamba: {tuple(y.shape)}")
    del model, x, y
    torch.cuda.empty_cache()


def prepare_official_architecture() -> Path:
    """Prepara UMambaEnc_2d oficial sem os helpers de planejamento do nnU-Net."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    source = urllib.request.urlopen(OFFICIAL_ARCH_URL, timeout=60).read().decode("utf-8")

    for line in (
        "from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager\n",
        "from nnunetv2.utilities.network_initialization import InitWeights_He\n",
    ):
        source = source.replace(line, "")

    helper_marker = "\ndef get_umamba_enc_2d_from_plans("
    if helper_marker in source:
        source = source.split(helper_marker, 1)[0].rstrip() + "\n"

    header = (
        '"""Runtime adaptation of official U-Mamba UMambaEnc_2d.\n\n'
        f"Source: {OFFICIAL_UMAMBA_REPOSITORY}\n"
        f"Commit: {OFFICIAL_UMAMBA_COMMIT}\n"
        "License: Apache-2.0\n"
        "Modified only to remove unused nnU-Net planning imports/helper.\n"
        '"""\n\n'
    )
    STANDALONE_FILE.write_text(header + source, encoding="utf-8")

    runtime_str = str(RUNTIME_DIR)
    if runtime_str not in sys.path:
        sys.path.insert(0, runtime_str)

    importlib.invalidate_caches()
    sys.modules.pop(STANDALONE_MODULE_NAME, None)
    module = importlib.import_module(STANDALONE_MODULE_NAME)
    if not hasattr(module, "UMambaEnc"):
        raise RuntimeError("UMambaEnc não encontrada no arquivo oficial preparado.")
    return STANDALONE_FILE


def ensure_umamba_runtime() -> dict[str, str | bool]:
    """Valida GPU, Mamba e arquitetura oficial após o stack binário estar instalado."""
    import torch

    require_cuda()

    if not torch.__version__.startswith(TORCH_LOCAL_PREFIX):
        raise RuntimeError(
            f"PyTorch carregado é {torch.__version__}, mas o stack requer {TORCH_LOCAL_PREFIX}. "
            "Execute install_prebuilt_colab_stack() e reinicie o runtime."
        )

    abi_enabled = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    if not abi_enabled:
        raise RuntimeError(
            "O PyTorch carregado usa CXX11 ABI=False, mas a wheel Mamba selecionada "
            "para torch 2.10 usa ABI=True."
        )

    mamba_version = _installed_version("mamba-ssm")
    if mamba_version is None or not mamba_version.startswith(MAMBA_SSM_VERSION):
        raise RuntimeError(
            "mamba-ssm ainda não está instalado. "
            "Execute install_prebuilt_colab_stack()."
        )

    # A prova real de compatibilidade é carregar e executar o kernel CUDA.
    # Não dependemos do sufixo da versão retornado por importlib.metadata,
    # porque wheels publicadas podem expor somente a versão-base no METADATA.
    _mamba_forward_sanity()
    architecture_path = prepare_official_architecture()

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "torch_cxx11_abi": abi_enabled,
        "gpu": torch.cuda.get_device_name(0),
        "mamba_ssm": mamba_version or "unknown",
        "umamba_commit": OFFICIAL_UMAMBA_COMMIT,
        "architecture_file": str(architecture_path),
    }
