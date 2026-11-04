from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = "20260714_reviewer2_component_ablation_v2"
MODEL_SOURCE = (
    ROOT
    / "final"
    / "unified_scheduler_dual_experiments_v1"
    / "models"
    / "short_trace_final500"
)
TRACE_CSV = ROOT / "data" / "workload_traces" / "wildchat_turn_trace_short_bucketed.csv"
CATALOG = ROOT / "scheduler_catalog.json"
RUNNER = ROOT / "scripts" / "evaluation" / "run_scheduler_policy_baselines_trace.py"


STAGES = [
    {
        "key": "raw",
        "label": "Raw prediction",
        "short_label": "Raw",
        "prediction_p95_col": "p95_hat_s",
        "zero_risk_buffer": True,
        "edge_rescue": False,
    },
    {
        "key": "calibration",
        "label": "+ residual calibration",
        "short_label": "+Cal.",
        "prediction_p95_col": "p95_calibrated_s",
        "zero_risk_buffer": True,
        "edge_rescue": False,
    },
    {
        "key": "risk_buffer",
        "label": "+ p75 risk buffer",
        "short_label": "+Buffer",
        "prediction_p95_col": "p95_calibrated_s",
        "zero_risk_buffer": False,
        "edge_rescue": False,
    },
    {
        "key": "edge_rescue",
        "label": "+ edge rescue",
        "short_label": "+Rescue",
        "prediction_p95_col": "p95_calibrated_s",
        "zero_risk_buffer": False,
        "edge_rescue": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Reviewer 2 calibration/risk/rescue component ablation."
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--max-trace-cases", type=int, default=500)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse completed stage summaries instead of refusing an existing output directory.",
    )
    return parser.parse_args()


def experiment_paths(name: str) -> Dict[str, Path]:
    return {
        "run": ROOT / "outputs" / "runs" / name,
        "report": ROOT / "outputs" / "reports" / name,
        "figure": ROOT / "outputs" / "figures" / name,
    }


def prepare_stage_model(stage: Dict[str, object], stage_root: Path) -> Path:
    model_dir = stage_root / "models"
    shutil.copytree(MODEL_SOURCE, model_dir)
    if bool(stage["zero_risk_buffer"]):
        table_path = model_dir / "combined_perf_with_pred.csv"
        frame = pd.read_csv(table_path, encoding="utf-8-sig")
        for column in ["p95_abs_err_s", "p95_calibrated_abs_err_s"]:
            if column in frame.columns:
                frame[column] = 0.0
        frame.to_csv(table_path, index=False, encoding="utf-8-sig")
    return model_dir


def stage_command(
    stage: Dict[str, object], model_dir: Path, result_dir: Path, max_trace_cases: int
) -> List[str]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--artifacts-dir",
        str(model_dir),
        "--catalog",
        str(CATALOG),
        "--workload-csv",
        str(TRACE_CSV),
        "--max-trace-cases",
        str(max_trace_cases),
        "--policies",
        "scheduler_pred",
        "--prediction-source",
        "table",
        "--prediction-table-csv",
        str(model_dir / "combined_perf_with_pred.csv"),
        "--prediction-p95-col",
        str(stage["prediction_p95_col"]),
        "--disable-learned-policy",
        "--optimize-for",
        "throughput_constrained_trace_distilled",
        "--cloud-throughput-mode",
        "service_effective",
        "--trace-small-latency-rescue-min-cloud-pressure-ratio",
        "0.20",
        "--trace-medium-edge-min-tps-ratio",
        "0.95",
        "--throughput-tie-ratio",
        "0.03",
        "--outdir",
        str(result_dir),
    ]
    if not bool(stage["edge_rescue"]):
        cmd.append("--disable-trace-edge-rescue")
    return cmd


def run_stage(
    stage: Dict[str, object], run_root: Path, max_trace_cases: int, reuse_existing: bool
) -> Path:
    stage_root = run_root / str(stage["key"])
    summary_path = stage_root / "results" / "baseline_policy_summary.csv"
    if summary_path.exists() and reuse_existing:
        return summary_path
    if stage_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite {stage_root}. Use --reuse-existing or a new --experiment-name."
        )

    stage_root.mkdir(parents=True)
    model_dir = prepare_stage_model(stage, stage_root)
    result_dir = stage_root / "results"
    result_dir.mkdir()
    cmd = stage_command(stage, model_dir, result_dir, max_trace_cases)
    (stage_root / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (stage_root / "command.log").write_text(completed.stdout, encoding="utf-8")
    return summary_path


def collect_rows(summary_paths: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    stage_index = {str(stage["key"]): i for i, stage in enumerate(STAGES)}
    stage_by_key = {str(stage["key"]): stage for stage in STAGES}
    for key, path in summary_paths.items():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame = frame[frame["policy"] == "scheduler_pred"]
        stage = stage_by_key[key]
        for _, row in frame.iterrows():
            rows.append(
                {
                    "stage_index": stage_index[key],
                    "stage_key": key,
                    "stage_label": stage["label"],
                    "short_label": stage["short_label"],
                    "device_id": row["device_id"],
                    "mean_actual_tok_per_s": float(row["mean_actual_tok_per_s"]),
                    "mean_actual_p95_s": float(row["mean_actual_p95_s"]),
                    "slo_hit_rate": float(row["slo_hit_rate"]),
                    "edge_capture_ratio": float(row["tcr"]),
                    "cases": int(row["cases"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["stage_index", "device_id"]).reset_index(drop=True)


def write_report(frame: pd.DataFrame, report_root: Path) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_root / "COMPONENT_ABLATION.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Reviewer 2 Component Ablation",
        "",
        "This incremental ablation isolates the effects of residual calibration,",
        "the p75 empirical risk buffer, and edge rescue on the fixed 500-case",
        "WildChat-derived replay. All stages use the same predictor artifacts,",
        "candidate catalog, SLO/memory grid, and unified scheduler parameters.",
        "",
        "| Stage | Device | Throughput (tok/s) | Mean p95 (s) | SLO hit | Edge capture | Cases |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            "| {stage_label} | {device_id} | {mean_actual_tok_per_s:.3f} | "
            "{mean_actual_p95_s:.3f} | {slo_hit_rate:.3f} | "
            "{edge_capture_ratio:.3f} | {cases} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The first transition changes only the p95 prediction column from raw",
            "to residual-calibrated values. The second restores the configuration-level",
            "p75 absolute-error buffer while rescue remains disabled. The final transition",
            "enables edge rescue while retaining calibration and risk buffering.",
        ]
    )
    (report_root / "COMPONENT_ABLATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_results(frame: pd.DataFrame, figure_root: Path) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
        }
    )
    colors = {"edge_3090": "#0072B2", "edge_4090": "#D55E00"}
    markers = {"edge_3090": "o", "edge_4090": "s"}
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.3), sharex=True)
    for device_id, device_frame in frame.groupby("device_id", sort=True):
        device_frame = device_frame.sort_values("stage_index")
        x = device_frame["stage_index"].to_numpy()
        label = device_id.replace("edge_", "RTX ")
        axes[0].plot(
            x,
            device_frame["mean_actual_tok_per_s"],
            color=colors[device_id],
            marker=markers[device_id],
            linewidth=1.2,
            markersize=4,
            label=label,
        )
        axes[1].plot(
            x,
            device_frame["slo_hit_rate"],
            color=colors[device_id],
            marker=markers[device_id],
            linewidth=1.2,
            markersize=4,
            label=label,
        )

    axes[0].set_ylabel("Throughput (tok/s)")
    axes[1].set_ylabel("SLO hit rate")
    axes[1].set_ylim(0.94, 1.005)
    axes[0].legend(frameon=False, ncol=2, loc="best")
    labels = (
        frame.sort_values("stage_index")
        .drop_duplicates("stage_index")["short_label"]
        .tolist()
    )
    axes[1].set_xticks(range(len(labels)), labels)
    for index, ax in enumerate(axes):
        ax.text(
            -0.15,
            1.03,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    fig.tight_layout(pad=0.8, h_pad=1.0)
    fig.savefig(figure_root / "component_ablation.pdf", bbox_inches="tight")
    fig.savefig(figure_root / "component_ablation.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = experiment_paths(args.experiment_name)
    if paths["run"].exists() and not args.reuse_existing:
        raise FileExistsError(
            f"Refusing to overwrite {paths['run']}. Choose a new --experiment-name."
        )

    summary_paths: Dict[str, Path] = {}
    for stage in STAGES:
        summary_paths[str(stage["key"])] = run_stage(
            stage,
            paths["run"],
            args.max_trace_cases,
            bool(args.reuse_existing),
        )

    frame = collect_rows(summary_paths)
    write_report(frame, paths["report"])
    plot_results(frame, paths["figure"])
    metadata = {
        "experiment_name": args.experiment_name,
        "max_trace_cases": args.max_trace_cases,
        "model_source": str(MODEL_SOURCE),
        "trace_csv": str(TRACE_CSV),
        "catalog": str(CATALOG),
        "stages": STAGES,
    }
    (paths["run"] / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(paths["report"] / "COMPONENT_ABLATION.md")


if __name__ == "__main__":
    main()
