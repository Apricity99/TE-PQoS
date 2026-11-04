#!/usr/bin/env python3
"""Plot the isolated-component view of the Reviewer 2 factorial ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "outputs/reports/20260716_reviewer2_component_factorial_ablation_v3"
    / "ISOLATED_COMPONENT_EFFECTS.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs/figures/20260716_reviewer2_component_ablation_v1"
)
DEFAULT_PAPER_PDF = (
    ROOT
    / "IEEE-Transactions-LaTeX2e-templates-and-instructions/picture"
    / "Fig6_component_ablation.pdf"
)

VARIANTS = [
    ("c0_b0_r0", "All off"),
    ("c1_b0_r0", "Calibration only"),
    ("c0_b1_r0", "p75 buffer only"),
    ("c0_b0_r1", "Edge rescue only"),
    ("c1_b1_r1", "All on"),
]
DEVICES = [("edge_3090", "RTX 3090"), ("edge_4090", "RTX 4090")]
COLORS = ["#7F7F7F", "#56B4E9", "#E69F00", "#009E73", "#0072B2"]
HATCHES = ["//", "xx", "..", "\\\\", ""]


def load_rows(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["variant_key"], row["device_id"])
            rows[key] = {
                "throughput": float(row["mean_actual_tok_per_s"]),
                "p95": float(row["mean_actual_p95_s"]),
            }

    missing = [
        (variant_key, device_key)
        for variant_key, _ in VARIANTS
        for device_key, _ in DEVICES
        if (variant_key, device_key) not in rows
    ]
    if missing:
        raise ValueError(f"Missing required component/device rows: {missing}")
    return rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_metric(
    ax: plt.Axes,
    rows: dict[tuple[str, str], dict[str, float]],
    metric: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> list[plt.Rectangle]:
    x = np.arange(len(DEVICES), dtype=float)
    width = 0.15
    bars_for_legend: list[plt.Rectangle] = []

    for index, (variant_key, _) in enumerate(VARIANTS):
        values = [rows[(variant_key, device_key)][metric] for device_key, _ in DEVICES]
        offset = (index - (len(VARIANTS) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[index],
            edgecolor="#333333",
            linewidth=0.6,
            hatch=HATCHES[index],
            zorder=3,
        )
        bars_for_legend.append(bars[0])

    ax.set_title(title, pad=7)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [label for _, label in DEVICES])
    ax.set_xlim(-0.55, len(DEVICES) - 0.45)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return bars_for_legend


def build_figure(rows: dict[tuple[str, str], dict[str, float]]) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.25))

    legend_handles = plot_metric(
        axes[0],
        rows,
        metric="throughput",
        title="Throughput",
        ylabel="Mean throughput (tok/s)",
        ylim=(180.0, 211.0),
    )
    plot_metric(
        axes[1],
        rows,
        metric="p95",
        title="Latency",
        ylabel="Mean p95 latency (s)",
        ylim=(0.67, 0.775),
    )

    for label, ax in zip(("(a)", "(b)"), axes):
        ax.text(-0.16, 1.04, label, transform=ax.transAxes, fontweight="bold")

    fig.legend(
        legend_handles,
        [label for _, label in VARIANTS],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        columnspacing=1.25,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.9, bottom=0.29, wspace=0.34)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-pdf", type=Path, default=DEFAULT_PAPER_PDF)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    figure = build_figure(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = args.output_dir / "Fig6_component_ablation.pdf"
    output_png = args.output_dir / "Fig6_component_ablation.png"
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(args.paper_pdf, bbox_inches="tight")
    plt.close(figure)

    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_png}")
    print(f"Wrote {args.paper_pdf}")


if __name__ == "__main__":
    main()
