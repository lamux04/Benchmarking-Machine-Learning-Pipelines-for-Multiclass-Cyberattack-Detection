#!/usr/bin/env python3
"""Genera matrices de confusion bonitas para los experimentos de CIC17."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_PATTERN = "*__holdout_confusion_matrix.csv"


def _read_confusion_matrix(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric)


def _annotation_text(values: np.ndarray) -> np.ndarray:
    annotations = np.empty(values.shape, dtype=object)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            count = int(values[i, j])
            annotations[i, j] = "" if count == 0 else f"{count:,}"

    return annotations


def _output_path(csv_path: Path, suffix: str, extension: str) -> Path:
    experiment_dir = csv_path.parent.parent
    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / f"{csv_path.stem}{suffix}.{extension}"


def plot_confusion_matrix(
    csv_path: Path,
    formats: tuple[str, ...],
    suffix: str,
    dpi: int,
) -> list[Path]:
    df = _read_confusion_matrix(csv_path)
    values = df.to_numpy(dtype=float)
    row_totals = values.sum(axis=1, keepdims=True)
    row_percentages = np.divide(
        values,
        row_totals,
        out=np.zeros_like(values, dtype=float),
        where=row_totals != 0,
    )

    display_labels = list(df.index.astype(str))
    annotations = _annotation_text(values)

    n_classes = len(display_labels)
    fig_size = max(7.5, n_classes * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size + 1.2, fig_size), constrained_layout=True)

    cmap = LinearSegmentedColormap.from_list(
        "cic17_blues",
        ["#f8fbff", "#d6e8f5", "#74a9cf", "#2b6c9e", "#08306b"],
    )
    image = ax.imshow(row_percentages, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    ax.set_xlabel("Etiqueta predicha", fontsize=13, labelpad=12)
    ax.set_ylabel("Etiqueta real", fontsize=13, labelpad=12)
    ax.set_xticks(np.arange(n_classes), labels=display_labels)
    ax.set_yticks(np.arange(n_classes), labels=display_labels)
    ax.tick_params(axis="x", labelrotation=0, labelsize=10, pad=2)
    ax.tick_params(axis="y", labelsize=10)
    plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor")

    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    threshold = 0.52
    for i in range(n_classes):
        for j in range(n_classes):
            text = annotations[i, j]
            if not text:
                continue
            color = "white" if row_percentages[i, j] >= threshold else "#1d2733"
            weight = "bold" if i == j else "normal"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                fontweight=weight,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Proporcion por clase real", rotation=270, labelpad=18, fontsize=11)
    colorbar.ax.tick_params(labelsize=9)

    saved_paths = []
    for extension in formats:
        output_path = _output_path(csv_path, suffix=suffix, extension=extension)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved_paths.append(output_path)

    plt.close(fig)
    return saved_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera figuras de matrices de confusion a partir de los CSV "
            "de hold-out de CIC17. Las figuras se guardan sin titulo."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Carpeta base donde buscar matrices CSV.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Patron glob para localizar matrices de confusion.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Formatos de salida.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Sufijo opcional para no sobrescribir las figuras existentes.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolucion usada para salidas raster como PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    csv_paths = sorted(root.rglob(args.pattern))

    if not csv_paths:
        raise SystemExit(f"No se encontraron matrices con el patron: {args.pattern}")

    generated = []
    for csv_path in csv_paths:
        generated.extend(
            plot_confusion_matrix(
                csv_path=csv_path,
                formats=tuple(args.formats),
                suffix=args.suffix,
                dpi=args.dpi,
            )
        )

    print(f"Matrices procesadas: {len(csv_paths)}")
    print(f"Figuras generadas: {len(generated)}")
    for output_path in generated:
        print(output_path.relative_to(root))


if __name__ == "__main__":
    main()
