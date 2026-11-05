"""Leitura da base geoespacial definitiva da RGI (GeoTIFF + máscara EMATER).

Os municípios identificam a proveniência dos recortes; NÃO definem treinos separados.
Máscaras: 0 = não-café; 1 = café; 255 = pixel ignorado.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RasterPair:
    image: Path
    mask: Path
    municipality: str


RGI_GUAXUPE_MUNICIPALITIES = frozenset(
    {
        "cabo_verde",
        "guaxupe",
        "juruaia",
        "muzambinho",
        "nova_resende",
    }
)


def _identical_files(a: Path, b: Path) -> bool:
    """Confere duplicatas em downloads fracionados sem misturar patches distintos."""
    if a.stat().st_size != b.stat().st_size:
        return False

    def digest(path: Path) -> str:
        h = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    return digest(a) == digest(b)


def find_rgi_pairs(
    dataset_root: str | Path,
    variant: str = "Patches_P512_S300_NN",
    include_external: bool = False,
) -> list[RasterPair]:
    """Combina LOGICAMENTE partes 001/002, sem copiar/alterar arquivos originais.

    As partes do lote espalham imagens e máscaras de um mesmo patch por
    diretórios DIFERENTES. Portanto é incorreto emparelhar dentro de cada
    pasta antes de reunir os nomes globalmente.

    Cada identidade de patch é (município_ano, nome_do_arquivo). A variante
    Fora0_M0 não deve ser combinada à NN: representam os MESMOS recortes.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Base RGI não encontrada: {root}")

    def collect(leaf_name: str) -> dict[tuple[str, str], Path]:
        found: dict[tuple[str, str], Path] = {}
        for folder in sorted(root.rglob(leaf_name)):
            if not folder.is_dir() or folder.parent.name != variant:
                continue
            municipality = folder.parent.parent.name
            geographic_name = municipality.rsplit("_", 1)[0]
            if (
                not include_external
                and geographic_name not in RGI_GUAXUPE_MUNICIPALITIES
            ):
                continue
            for path in sorted(folder.glob("*.tif")):
                key = (municipality, path.name)
                old_path = found.get(key)
                if old_path is not None:
                    if not _identical_files(old_path, path):
                        raise ValueError(
                            f"Arquivos divergentes para o patch {key} na variante "
                            f"{variant}: {old_path} vs {path}"
                        )
                    continue
                found[key] = path
        return found

    images = collect("Imagens_Individuais")
    masks = collect("Mascaras_Individuais")
    if not images and not masks:
        raise ValueError(
            f"Nenhum par GeoTIFF encontrado em {root} na variante {variant}. "
            "Informe a raiz que contém os dois pacotes Municipios-...-001/002."
        )
    without_mask = sorted(images.keys() - masks.keys())
    without_image = sorted(masks.keys() - images.keys())
    if without_mask or without_image:
        raise ValueError(
            "O conjunto parece incompleto. É obrigatório incluir AMBAS "
            "as partes 001 e 002 do pacote Municipios. "
            f"Sem máscara (exemplos): {without_mask[:6]}; "
            f"sem imagem (exemplos): {without_image[:6]}"
        )

    return [
        RasterPair(image=images[key], mask=masks[key], municipality=key[0])
        for key in sorted(images)
    ]

def validate_mask_values(mask: np.ndarray) -> None:
    """Não permite binarização por threshold: 255 nunca é café."""
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 1, 255}):
        raise ValueError(f"Rótulos inválidos na máscara EMATER: {sorted(values)}")


def normalize_sentinel_dn(image: np.ndarray, offset: float = -1000.0) -> np.ndarray:
    """DN -> reflectância BOA conforme metadata da cena; DN=0 é nodata.

    O offset -1000 é o observado no lote atual. Conferir metadados da
    fonte antes de reaproveitar em outros lotes.
    """
    nodata = image == 0
    reflectance = np.clip((image.astype(np.float32) + offset) / 10000.0, 0, 1)
    reflectance[nodata] = 0
    return reflectance


class EMATERRGIDataset(Dataset):
    """Carrega patches da mesma RGI usando bandas (B02, B03, B04, B08)."""

    def __init__(self, pairs: list[RasterPair], band_indexes: tuple[int, ...] = (1, 2, 3, 4)):
        self.pairs = list(pairs)
        self.band_indexes = tuple(band_indexes)
        if not self.pairs:
            raise ValueError("O Dataset não pode estar vazio.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        import rasterio

        pair = self.pairs[index]
        with rasterio.open(pair.image) as src:
            if src.count < max(self.band_indexes):
                raise ValueError(f"Bandas insuficientes em {pair.image}")
            image = src.read(self.band_indexes)
            valid_image = np.all(src.read_masks(self.band_indexes) > 0, axis=0)
            # A variante Fora0_M0 traz pixels externos ao município com
            # imagem=0 e MÁSCARA=0 (não-café). São regiões sem observação,
            # nunca negativos verdadeiros; ambos precisam sair do loss.
            valid_image &= np.all(image != 0, axis=0)
            expected_bands = ("B02", "B03", "B04", "B08")
            actual_bands = tuple(src.descriptions[i - 1] for i in self.band_indexes)
            if self.band_indexes == (1, 2, 3, 4) and actual_bands != expected_bands:
                raise ValueError(f"Ordem de bandas inesperada em {pair.image}: {actual_bands}")
            if image.dtype != np.dtype("uint16"):
                raise ValueError(
                    f"Imagens Sentinel-2 oficiais devem conter DN uint16: {pair.image}"
                )
            try:
                offset = float(src.tags()["BOA_ADD_OFFSET"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"BOA_ADD_OFFSET ausente/inválido no GeoTIFF: {pair.image}"
                ) from exc
            image_bounds = src.bounds
            image_crs = src.crs
            image_transform = src.transform

        with rasterio.open(pair.mask) as src:
            if src.count != 1:
                raise ValueError(f"Máscara deve ter 1 banda: {pair.mask}")
            mask = src.read(1)
            if src.crs != image_crs or src.transform != image_transform:
                raise ValueError(f"Imagem e máscara com grids distintos: {pair.image}")
            if src.bounds != image_bounds:
                raise ValueError(f"Imagem e máscara com áreas distintas: {pair.image}")

        if image.shape[1:] != mask.shape:
            raise ValueError(f"Dimensões não coincidem em {pair.image}")
        validate_mask_values(mask)
        mask = mask.copy()
        mask[~valid_image] = 255
        image = normalize_sentinel_dn(image, offset=offset)
        # 255 inclui borda/nodata de imagem mesmo se a máscara vier com valor 0.
        return torch.from_numpy(image.copy()), torch.from_numpy(mask.astype(np.float32)[None])
