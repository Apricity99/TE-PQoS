from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "final/unified_scheduler_dual_experiments_v1/results"
DEFAULT_REPORT_DIR = ROOT / "outputs/reports/20260714_reviewer2_reproducibility_v2"
PROFILES = {
    "main_controlled": FINAL / "main_controlled/baseline_case_results.csv",
    "short_trace_final500": FINAL / "short_trace_final500/baseline_case_results.csv",
}
POLICIES = ["cloud_only", "scheduler_pred"]
METRICS = ["actual_tok_per_s", "actual_p95_s", "actual_within_slo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster-bootstrap scheduler case metrics.")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def add_case_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    block_size = frame.groupby(["slo_p95_s", "edge_mem_available_mb"]).ngroups
    if block_size != 16 or len(frame) % block_size:
        raise ValueError(f"Expected 16 grid rows per workload case, got {block_size}")
    frame["case_block"] = np.arange(len(frame)) // block_size
    shape_counts = frame.groupby("case_block")[["prompt_len", "gen_len", "concurrency"]].nunique()
    if int(shape_counts.to_numpy().max()) != 1:
        raise ValueError("Rows are not ordered as contiguous workload-case blocks")
    return frame


def bootstrap_group(
    frame: pd.DataFrame, iterations: int, rng: np.random.Generator
) -> dict[str, float]:
    frame = add_case_blocks(frame)
    case_means = frame.groupby("case_block")[METRICS].mean().to_numpy(dtype=float)
    n_cases = len(case_means)
    draws = rng.integers(0, n_cases, size=(iterations, n_cases))
    samples = case_means[draws].mean(axis=1)
    point = case_means.mean(axis=0)
    result: dict[str, float] = {"workload_cases": n_cases, "grid_rows": len(frame)}
    for index, metric in enumerate(METRICS):
        result[f"{metric}_mean"] = float(point[index])
        result[f"{metric}_lo"] = float(np.quantile(samples[:, index], 0.025))
        result[f"{metric}_hi"] = float(np.quantile(samples[:, index], 0.975))
    return result


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for profile, path in PROFILES.items():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        for (device_id, policy), group in frame[frame["policy"].isin(POLICIES)].groupby(
            ["device_id", "policy"], sort=True
        ):
            rows.append(
                {
                    "profile": profile,
                    "device_id": device_id,
                    "policy": policy,
                    **bootstrap_group(group, args.iterations, rng),
                }
            )
    result = pd.DataFrame(rows)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.report_dir / "BOOTSTRAP_CASE_CI.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Workload-Case Bootstrap Confidence Intervals", "",
        f"Iterations: {args.iterations}; seed: {args.seed}; interval: empirical 2.5th--97.5th percentiles.",
        "The resampling unit is one workload case, with all 16 SLO-by-memory grid rows retained as a block.",
        "Deterministic policies are evaluated once over the full case grid; these intervals quantify case-set uncertainty, not independent hardware-run variability.",
        "Randomized baselines separately use seeds 0--9.", "",
        "| Profile | Device | Policy | Workload cases | Throughput mean [95% CI] | Mean per-case p95 [95% CI] | SLO hit [95% CI] |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in result.to_dict(orient="records"):
        lines.append(
            "| {profile} | {device_id} | {policy} | {workload_cases} | "
            "{actual_tok_per_s_mean:.2f} [{actual_tok_per_s_lo:.2f}, {actual_tok_per_s_hi:.2f}] | "
            "{actual_p95_s_mean:.3f} [{actual_p95_s_lo:.3f}, {actual_p95_s_hi:.3f}] | "
            "{actual_within_slo_mean:.3f} [{actual_within_slo_lo:.3f}, {actual_within_slo_hi:.3f}] |".format(**row)
        )
    report = args.report_dir / "BOOTSTRAP_CASE_CI.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
