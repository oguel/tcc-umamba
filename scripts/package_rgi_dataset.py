"""Empacota cópias fiéis dos 206 pares NN para upload privado manual no Kaggle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from src.config import get_config
from src.data.emater_dataset import find_rgi_pairs
from src.data.rgi_fingerprint import _file_sha256, dataset_fingerprint


def package(root: Path, output: Path) -> dict:
    pairs = find_rgi_pairs(root)
    fingerprint = dataset_fingerprint(pairs)
    if fingerprint != get_config()["official_dataset"]["expected_rgi_fingerprint"]:
        raise ValueError("Dataset difere do lote congelado.")
    files = sorted({path for pair in pairs for path in (pair.image, pair.mask)})
    files += sorted(
        p
        for p in root.rglob("metadados_municipio.json")
        if p.parent.name in {pair.municipality for pair in pairs}
    )
    manifest = {
        "dataset_fingerprint": fingerprint,
        "pairs": len(pairs),
        "variant": "Patches_P512_S300_NN",
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in files
        ],
        "external_excluded": "tres_pontas_2024",
        "originals_modified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_STORED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(root).as_posix())
        archive.writestr("MANIFESTO_PRIVADO.json", json.dumps(manifest, indent=2))
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Falha de integridade do ZIP.")
    summary = {
        "file": output.name,
        "bytes": output.stat().st_size,
        "sha256": _file_sha256(output),
        "dataset_fingerprint": fingerprint,
        "pairs": len(pairs),
        "files": len(files),
        "zip_crc_verified": True,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
