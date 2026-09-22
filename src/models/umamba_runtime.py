"""Preparação reprodutível do runtime para a arquitetura oficial U-Mamba.

A arquitetura 2D é obtida do repositório oficial bowang-lab/U-Mamba em um
commit fixado. Para o experimento parcial usamos somente as classes da rede,
sem o pipeline médico completo do nnU-Net.

A cópia de runtime remove apenas imports/helpers do nnU-Net que não são
necessários para construir UMambaEnc diretamente. A arquitetura UMambaEnc,
ResidualMambaEncoder, decoder e MambaLayer permanecem oriundas do arquivo
oficial UMambaEnc_2d.py.

U-Mamba: Apache-2.0
https://github.com/bowang-lab/U-Mamba
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import torch

OFFICIAL_UMAMBA_REPOSITORY = "https://github.com/bowang-lab/U-Mamba"
OFFICIAL_UMAMBA_COMMIT = "28459e33ca03769800dd35e23c6e62491d1925b5"
OFFICIAL_ARCH_URL = (
    "https://raw.githubusercontent.com/bowang-lab/U-Mamba/"
    f"{OFFICIAL_UMAMBA_COMMIT}/umamba/nnunetv2/nets/UMambaEnc_2d.py"
)
MAMBA_SSM_VERSION = "2.3.2.post1"
DNA_VERSION = "0.3.1"

RUNTIME_DIR = Path("/content/umamba_runtime") if Path("/content").exists() else Path(".umamba_runtime")
STANDALONE_MODULE_NAME = "umamba_official_standalone"
STANDALONE_FILE = RUNTIME_DIR / f"{STANDALONE_MODULE_NAME}.py"


def require_cuda() -> None:
    """Interrompe cedo quando o Colab não está usando uma GPU NVIDIA."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está ativa. No Google Colab, abra "
            "'Ambiente de execução > Alterar tipo de ambiente de execução' e "
            "selecione uma GPU (por exemplo, T4). Depois reinicie a sessão e "
            "execute este notebook desde a primeira célula."
        )
    if torch.version.cuda is None:
        raise RuntimeError("O PyTorch atual não possui suporte CUDA.")


def _pip_install(args: list[str], env: dict[str, str] | None = None) -> None:
    command = [sys.executable, "-m", "pip", "install", *args]
    print("$", " ".join(command))
    subprocess.run(command, check=True, env=env)


def _ensure_python_dependencies() -> None:
    """Instala dependências leves usadas pelo arquivo oficial da arquitetura."""
    try:
        import dynamic_network_architectures  # noqa: F401
    except ImportError:
        _pip_install([f"dynamic-network-architectures=={DNA_VERSION}"])

    for package, module in (
        ("einops", "einops"),
        ("ninja", "ninja"),
        ("packaging", "packaging"),
    ):
        if importlib.util.find_spec(module) is None:
            _pip_install([package])


def _mamba_forward_sanity() -> None:
    """Confirma que o bloco Mamba executa de fato na GPU."""
    from mamba_ssm import Mamba

    model = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda()
    x = torch.randn(1, 64, 16, device="cuda")
    with torch.inference_mode():
        y = model(x)
    if y.shape != x.shape:
        raise RuntimeError(f"Saída inesperada do Mamba: {tuple(y.shape)}")


def _ensure_mamba() -> None:
    """Instala Mamba com o backend CUDA quando necessário."""
    try:
        import mamba_ssm  # noqa: F401
        _mamba_forward_sanity()
        return
    except Exception as exc:
        print(f"Mamba ainda não funcional neste runtime: {exc}")

    env = os.environ.copy()
    env["MAMBA_KEEP_CUDA_BUILD"] = "TRUE"
    env.setdefault("MAX_JOBS", "2")

    _pip_install(
        [
            f"mamba-ssm=={MAMBA_SSM_VERSION}",
            "--no-build-isolation",
            "--no-cache-dir",
        ],
        env=env,
    )

    importlib.invalidate_caches()
    if "mamba_ssm" in sys.modules:
        del sys.modules["mamba_ssm"]
    _mamba_forward_sanity()


def prepare_official_architecture() -> Path:
    """Cria uma cópia de runtime da arquitetura UMambaEnc 2D oficial.

    Removemos somente imports e o helper baseado em PlansManager/nnU-Net,
    porque o TCC instancia a rede diretamente. Isso evita instalar todo o
    ecossistema médico apenas para executar a arquitetura.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    source = urllib.request.urlopen(OFFICIAL_ARCH_URL, timeout=60).read().decode("utf-8")

    removable_imports = (
        "from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager\n",
        "from nnunetv2.utilities.network_initialization import InitWeights_He\n",
    )
    for line in removable_imports:
        source = source.replace(line, "")

    helper_marker = "\ndef get_umamba_enc_2d_from_plans("
    if helper_marker in source:
        source = source.split(helper_marker, 1)[0].rstrip() + "\n"

    header = (
        '"""Runtime adaptation of the official U-Mamba UMambaEnc_2d architecture.\n\n'
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
    if STANDALONE_MODULE_NAME in sys.modules:
        del sys.modules[STANDALONE_MODULE_NAME]

    module = importlib.import_module(STANDALONE_MODULE_NAME)
    if not hasattr(module, "UMambaEnc"):
        raise RuntimeError("A classe UMambaEnc não foi encontrada no arquivo oficial preparado.")
    return STANDALONE_FILE


def ensure_umamba_runtime() -> dict[str, str]:
    """Prepara e valida o runtime necessário para os notebooks 11 e 12."""
    require_cuda()
    _ensure_python_dependencies()
    _ensure_mamba()
    architecture_path = prepare_official_architecture()

    import mamba_ssm

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "mamba_ssm": getattr(mamba_ssm, "__version__", MAMBA_SSM_VERSION),
        "umamba_commit": OFFICIAL_UMAMBA_COMMIT,
        "architecture_file": str(architecture_path),
    }
