from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = ROOT / "outputs/reports/20260716_cross_model_workload_validation_v2"
DATASETS = {
    "Llama-2-7B (AWQ/GPTQ)": (
        ROOT
        / "outputs/runs/20260711_llama_burstgpt_v1/results/burstgpt_stratified500/baseline_case_results.csv"
    ),
    "Qwen2.5-7B-Instruct-AWQ": (
        ROOT
        / "outputs/runs/20260716_cross_model_workload_validation_v2/qwen/results/burstgpt_stratified500/baseline_case_results.csv"
    ),
}
POLICIES = ["cloud_only", "scheduler_pred"]
METRICS = [
    "actual_tok_per_s",
    "actual_p95_s",
    "actual_within_slo",
    "total_cost",
    "tcr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate cross-model BurstGPT validation with workload-case bootstrap intervals."
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def add_case_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    block_size = frame.groupby(["slo_p95_s", "edge_mem_available_mb"]).ngroups
    if block_size != 16 or len(frame) % block_size:
        raise ValueError(f"Expected 16 grid rows per workload case, got {block_size}")
    frame["case_block"] = np.arange(len(frame)) // block_size
    uniqueness = frame.groupby("case_block")[["prompt_len", "gen_len", "concurrency"]].nunique()
    if int(uniqueness.to_numpy().max()) != 1:
        raise ValueError("Rows are not contiguous workload-case blocks")
    return frame


def bootstrap(
    frame: pd.DataFrame, iterations: int, rng: np.random.Generator
) -> dict[str, float]:
    frame = add_case_blocks(frame)
    case_means = frame.groupby("case_block")[METRICS].mean().to_numpy(dtype=float)
    n_cases = len(case_means)
    draws = rng.integers(0, n_cases, size=(iterations, n_cases))
    samples = case_means[draws].mean(axis=1)
    points = case_means.mean(axis=0)
    result: dict[str, float] = {"workload_cases": float(n_cases)}
    for index, metric in enumerate(METRICS):
        result[f"{metric}_mean"] = float(points[index])
        result[f"{metric}_lo"] = float(np.quantile(samples[:, index], 0.025))
        result[f"{metric}_hi"] = float(np.quantile(samples[:, index], 0.975))
    return result


def case_metric_matrix(frame: pd.DataFrame) -> np.ndarray:
    return (
        add_case_blocks(frame)
        .groupby("case_block")[METRICS]
        .mean()
        .to_numpy(dtype=float)
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    records = []
    for model, path in DATASETS.items():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        selected = frame[frame["policy"].isin(POLICIES)]
        for (device, policy), group in selected.groupby(["device_id", "policy"], sort=True):
            records.append(
                {
                    "model": model,
                    "device_id": device,
                    "policy": policy,
                    **bootstrap(group, args.iterations, rng),
                }
            )

    result = pd.DataFrame(records)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        args.report_dir / "CROSS_SCALE_BOOTSTRAP_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_rng = np.random.default_rng(args.seed + 1)
    paired_records = []
    for model, path in DATASETS.items():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        for device in sorted(frame["device_id"].unique()):
            cloud = case_metric_matrix(
                frame[(frame["device_id"] == device) & (frame["policy"] == "cloud_only")]
            )
            tepqos = case_metric_matrix(
                frame[(frame["device_id"] == device) & (frame["policy"] == "scheduler_pred")]
            )
            if cloud.shape != tepqos.shape:
                raise ValueError(f"Mismatched paired case shapes for {model}, {device}")
            delta = tepqos - cloud
            draws = paired_rng.integers(
                0, len(delta), size=(args.iterations, len(delta))
            )
            samples = delta[draws].mean(axis=1)
            record: dict[str, Any] = {"model": model, "device_id": device}
            for index, metric in enumerate(METRICS):
                record[f"{metric}_delta"] = float(delta[:, index].mean())
                record[f"{metric}_delta_lo"] = float(
                    np.quantile(samples[:, index], 0.025)
                )
                record[f"{metric}_delta_hi"] = float(
                    np.quantile(samples[:, index], 0.975)
                )
            paired_records.append(record)
    paired = pd.DataFrame(paired_records)
    paired.to_csv(
        args.report_dir / "CROSS_SCALE_PAIRED_DELTA_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# Cross-Model and Independent-Workload Validation",
        "",
        "Workload: deterministic proportionally stratified sample of 500 requests from all 4,956,058 valid BurstGPT rows. Each request is evaluated over the same 4-by-4 SLO/memory grid.",
        "",
        f"Bootstrap: {args.iterations} workload-case resamples, seed {args.seed} for per-policy intervals and seed {args.seed + 1} for paired differences, with empirical 95% intervals. All 16 grid rows of a request remain in the same resampled block.",
        "",
        "| Model | Device | Policy | Throughput [95% CI] | SLO hit [95% CI] | Mean p95 [95% CI] | Cost [95% CI] | Edge route |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result.to_dict(orient="records"):
        lines.append(
            "| {model} | {device_id} | {policy} | "
            "{actual_tok_per_s_mean:.2f} [{actual_tok_per_s_lo:.2f}, {actual_tok_per_s_hi:.2f}] | "
            "{actual_within_slo_mean:.3f} [{actual_within_slo_lo:.3f}, {actual_within_slo_hi:.3f}] | "
            "{actual_p95_s_mean:.3f} [{actual_p95_s_lo:.3f}, {actual_p95_s_hi:.3f}] | "
            "{total_cost_mean:.3f} [{total_cost_lo:.3f}, {total_cost_hi:.3f}] | "
            "{tcr_mean:.3f} |".format(**row)
        )

    lines.extend(["", "## Paired TE-PQoS minus cloud-only changes", ""])
    for (model, device), group in result.groupby(["model", "device_id"], sort=False):
        indexed = group.set_index("policy")
        cloud = indexed.loc["cloud_only"]
        tepqos = indexed.loc["scheduler_pred"]
        throughput_gain = 100.0 * (
            tepqos["actual_tok_per_s_mean"] / cloud["actual_tok_per_s_mean"] - 1.0
        )
        cost_change = 100.0 * (
            tepqos["total_cost_mean"] / cloud["total_cost_mean"] - 1.0
        )
        delta = paired[(paired["model"] == model) & (paired["device_id"] == device)].iloc[0]
        lines.append(
            f"- {model}, {device}: throughput {throughput_gain:+.1f}% "
            f"(paired absolute delta {delta['actual_tok_per_s_delta']:+.2f} "
            f"[{delta['actual_tok_per_s_delta_lo']:+.2f}, {delta['actual_tok_per_s_delta_hi']:+.2f}] tok/s), "
            f"cost {cost_change:+.1f}% (delta {delta['total_cost_delta']:+.3f} "
            f"[{delta['total_cost_delta_lo']:+.3f}, {delta['total_cost_delta_hi']:+.3f}]), "
            f"SLO-hit delta {delta['actual_within_slo_delta']:+.3f} "
            f"[{delta['actual_within_slo_delta_lo']:+.3f}, {delta['actual_within_slo_delta_hi']:+.3f}]."
        )

    lines.extend(
        [
            "",
            "Limitations: Qwen peak-memory fields are empty, so its scheduler feasibility test uses catalog memory estimates. The BurstGPT replay is an offline case-grid evaluation and does not model a live multi-tenant queue.",
        ]
    )
    report = args.report_dir / "CROSS_SCALE_VALIDATION.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
