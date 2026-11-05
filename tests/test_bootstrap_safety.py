"""Bootstrap nunca deve apagar dados regionais ou artefatos do pesquisador."""

from __future__ import annotations

import io
import shutil
import tarfile

from src import bootstrap


def test_bootstrap_preserva_dados_do_usuario(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    (raw / "sentinel_original.tif").write_bytes(b"nao_apagar")
    (processed / "manifest.parquet").write_bytes(b"nao_apagar_tambem")

    archive = tmp_path / "source.tar.gz"
    payloads = {
        "tcc-umamba-main/src/__init__.py": b"",
        "tcc-umamba-main/src/config.py": b"# codigo\n",
        "tcc-umamba-main/requirements-runtime.txt": b"numpy\n",
        "tcc-umamba-main/data/external/ibge/info.txt": b"IBGE\n",
    }
    with tarfile.open(archive, "w:gz") as tf:
        for name, blob in payloads.items():
            item = tarfile.TarInfo(name)
            item.size = len(blob)
            tf.addfile(item, io.BytesIO(blob))

    def fake_download(url, destination):
        shutil.copyfile(archive, destination)
        return destination, None

    monkeypatch.setattr(bootstrap.urllib.request, "urlretrieve", fake_download)
    bootstrap._download_and_extract(root)

    assert (raw / "sentinel_original.tif").read_bytes() == b"nao_apagar"
    assert (processed / "manifest.parquet").read_bytes() == b"nao_apagar_tambem"
    assert (root / "src" / "config.py").read_text() == "# codigo\n"
    assert (root / "data" / "external" / "ibge" / "info.txt").read_text() == "IBGE\n"
