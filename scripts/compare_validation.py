"""Confere arrays de validação das duas redes e exporta painéis pareados.

Requer os backups extraídos e os dois diagnósticos; não treina nem infere teste.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

from src.data.emater_dataset import EMATERRGIDataset, find_rgi_pairs
from src.data.rgi_fingerprint import _file_sha256, dataset_fingerprint
from src.validation_comparison import compare_predictions
from src.validation_diagnostics import inspect_run, validation_pairs


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "root",
        "unet-run",
        "umamba-run",
        "unet-diagnostic",
        "umamba-diagnostic",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--no-panels", action="store_true")
    args = parser.parse_args()
    root, left_dir, right_dir, out = (
        args.root,
        args.unet_diagnostic,
        args.umamba_diagnostic,
        args.output,
    )
    out.mkdir(parents=True, exist_ok=False)
    panels = out / "paineis_pareados"
    panels.mkdir()
    left = json.loads((left_dir / "diagnostico_validacao.json").read_text(encoding="utf-8"))
    right = json.loads((right_dir / "diagnostico_validacao.json").read_text(encoding="utf-8"))
    for report, kind, run_dir in [
        (left, "unet", args.unet_run),
        (right, "umamba", args.umamba_run),
    ]:
        verified, best, original, _ = inspect_run(root, run_dir)
        if original["model"] != kind:
            raise ValueError("Modelo incorreto no diretório de execução")
        if report["checkpoint_sha256"] != _file_sha256(run_dir / f"best_{kind}_rgi.pt"):
            raise ValueError("SHA do checkpoint diverge do diagnóstico")
        if report["test_patches_inferred"] != 0 or report["threshold"] != 0.5:
            raise ValueError("Diagnóstico não corresponde ao escopo/limiar esperado")
        if report["data_fingerprint"] != original["dataset_fingerprint"]:
            raise ValueError("Dados do diagnóstico diferem da execução")
        if report["split_sha256"] != original["run_contract"]["split_sha256"]:
            raise ValueError("Divisão do diagnóstico difere da execução")
        if report["reported_val_iou"] != original["best_val_iou"]:
            raise ValueError("Métrica reportada difere do checkpoint selecionado")
    pairs = find_rgi_pairs(root)
    assert dataset_fingerprint(pairs) == right["data_fingerprint"] == left["data_fingerprint"]
    splits = json.loads((args.umamba_run / "splits.json").read_text(encoding="utf-8"))
    selected = validation_pairs(pairs, splits, right["split_sha256"])
    keys = {f"{p.municipality}/{p.image.name}" for p in selected}
    for report in (left, right):
        assert len(report["per_patch"]) == 20
        assert {p["municipality"] + "/" + p["filename"] for p in report["per_patch"]} == keys
    source = EMATERRGIDataset(selected)
    aggregate = {"unet": Counter(), "umamba": Counter()}
    paired = Counter()
    per_patch = []
    colors = ["#e5e7eb", "#15803d", "#ea580c", "#9333ea", "#555555"]
    class_cmap = ListedColormap([colors[0], colors[1], colors[4]])

    def error_map(target, prediction):
        valid = target != 255
        error = np.zeros(target.shape, dtype=np.uint8)
        error[valid & (prediction == 1) & (target == 1)] = 1
        error[valid & (prediction == 1) & (target == 0)] = 2
        error[valid & (prediction == 0) & (target == 1)] = 3
        error[~valid] = 4
        return error

    for index, pair in enumerate(selected):
        image, mask = source[index]
        target = mask[0].numpy()
        name = f"{pair.municipality}_{pair.image.stem}"
        predictions = []
        for folder in (left_dir, right_dir):
            array = folder / "paineis_validacao" / f"{name}.npz"
            with np.load(array, allow_pickle=False) as data:
                assert np.array_equal(data["valid"], target != 255)
                assert data["probability"].shape == (512, 512)
                assert np.isfinite(data["probability"]).all()
                assert ((data["probability"] >= 0) & (data["probability"] <= 1)).all()
                assert np.array_equal(data["prediction"], data["probability"] >= 0.5)
                predictions.append(data["prediction"].copy())
        result = compare_predictions(target, *predictions)
        for kind in aggregate:
            aggregate[kind].update(result[kind]["confusion"])
        for key in (
            "valid_pixels",
            "ignored_pixels",
            "both_correct",
            "both_wrong",
            "only_unet_correct",
            "only_umamba_correct",
        ):
            paired[key] += result[key]
        per_patch.append({"municipality": pair.municipality, "filename": pair.image.name, **result})
        if not args.no_panels:
            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            rgb = np.clip(image.numpy()[[2, 1, 0]].transpose(1, 2, 0) / 0.3, 0, 1)
            for row, (kind, prediction) in enumerate(
                zip(("unet", "umamba"), predictions, strict=True)
            ):
                axes[row, 0].imshow(rgb)
                axes[row, 1].imshow(
                    np.where(target != 255, target, 2), cmap=class_cmap, vmin=0, vmax=2
                )
                axes[row, 2].imshow(
                    np.where(target != 255, prediction, 2), cmap=class_cmap, vmin=0, vmax=2
                )
                axes[row, 3].imshow(
                    error_map(target, prediction), cmap=ListedColormap(colors), vmin=0, vmax=4
                )
                label = "U-Net" if kind == "unet" else "U-Mamba"
                for ax, title in zip(
                    axes[row],
                    (
                        "RGB ilustrativo",
                        "Referência",
                        f"{label} • IoU {100 * result[kind]['metrics']['iou']:.1f}%",
                        "Erros",
                    ),
                    strict=True,
                ):
                    ax.set_title(title, fontsize=11)
                    ax.axis("off")
            fig.suptitle(
                f"VALIDAÇÃO • {pair.municipality} • {pair.image.stem}\n"
                "Mesma referência e limiar 0,5 para as duas redes",
                fontsize=12,
            )
            fig.legend(
                handles=[
                    Patch(color=color, label=label)
                    for color, label in zip(
                        colors,
                        (
                            "Não café correto",
                            "Café correto",
                            "Falso café",
                            "Café omitido",
                            "Ignorado",
                        ),
                        strict=True,
                    )
                ],
                loc="lower center",
                ncol=5,
                fontsize=10,
            )
            fig.subplots_adjust(bottom=0.07, top=0.89, wspace=0.02, hspace=0.16)
            fig.savefig(panels / f"{name}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
        print(f"Painel pareado {index + 1}/20: {pair.municipality}", flush=True)

    for kind, report in [("unet", left), ("umamba", right)]:
        assert dict(aggregate[kind]) == report["confusion"]
        for city in report["by_municipality"]:
            city_sum = Counter()
            for item in per_patch:
                if item["municipality"] == city:
                    city_sum.update(item[kind]["confusion"])
            assert dict(city_sum) == report["by_municipality"][city]["confusion"]
        for path in (left_dir if kind == "unet" else right_dir).glob("paineis_validacao/*.png"):
            with Image.open(path) as im:
                im.verify()
    assert (
        sum(
            paired[k]
            for k in ("both_correct", "both_wrong", "only_unet_correct", "only_umamba_correct")
        )
        == paired["valid_pixels"]
    )
    summary = {
        "source": (
            "Arrays da U-Net gerados localmente e arrays U-Mamba recebidos do Kaggle, "
            "conferidos com referência local."
        ),
        "diagnostic_umamba_report_sha256": _file_sha256(right_dir / "diagnostico_validacao.json"),
        "data_fingerprint": right["data_fingerprint"],
        "split_sha256": right["split_sha256"],
        "test_patches_inferred": 0,
        "patches_compared": 20,
        "threshold": 0.5,
        "all_arrays_and_references_verified": True,
        "unet": {
            "metrics": left["metrics"],
            "confusion": dict(aggregate["unet"]),
            "checkpoint_sha256": left["checkpoint_sha256"],
        },
        "umamba": {
            "metrics": right["metrics"],
            "confusion": dict(aggregate["umamba"]),
            "checkpoint_sha256": right["checkpoint_sha256"],
        },
        "paired_pixel_counts": dict(paired),
        "patch_wins_unet": sum(
            p["unet"]["metrics"]["iou"] > p["umamba"]["metrics"]["iou"] for p in per_patch
        ),
        "patch_wins_umamba": sum(
            p["umamba"]["metrics"]["iou"] > p["unet"]["metrics"]["iou"] for p in per_patch
        ),
        "by_municipality": {
            city: {"unet": left["by_municipality"][city], "umamba": right["by_municipality"][city]}
            for city in left["by_municipality"]
        },
        "per_patch": per_patch,
        "within_validation_overlap_not_deduplicated": True,
        "no_statistical_independence_or_significance_claim": True,
    }
    (out / "COMPARACAO_CONFERIDA.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("per_patch", "by_municipality")},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
