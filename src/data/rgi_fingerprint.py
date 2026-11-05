"""Identificador de versao do dataset sem enviar as imagens ao GitHub."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from src.data.emater_dataset import RasterPair


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(pairs: Sequence[RasterPair]) -> str:
    """SHA256 independente dos caminhos absolutos e da ordem da lista.

    O SHA do dataset muda se qualquer pixel, metadado ou mascara mudar.
    Nao representa autorizacao de uso nem substitui auditoria de imagens.
    """
    if not pairs:
        raise ValueError("Nao e possivel obter fingerprint de dataset vazio.")
    lines = []
    keys = set()
    for pair in pairs:
        if pair.image.name != pair.mask.name:
            raise ValueError(f"Par com nomes divergentes: {pair.image}, {pair.mask}")
        key = (pair.municipality, pair.image.stem)
        if key in keys:
            raise ValueError(f"Patch duplicado no fingerprint: {key}")
        keys.add(key)
        lines.append(
            f"{pair.municipality}/{pair.image.stem}:"
            f"{_file_sha256(pair.image)}:{_file_sha256(pair.mask)}"
        )
    return sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
