from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = ROOT / "outputs/reports/20260716_profiling_cost_v1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct measured profiling request counts and active service time from benchmark CSVs."
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def dataset_groups() -> dict[str, list[dict[str, Any]]]:
    from tepqos import training as train
    from scripts.evaluation.run_qwen_scale_v1 import QWEN_DATASETS

    llama = [
        {
            "path": spec.path,
            "device_id": spec.device_id,
            "quant_method": spec.quant_method,
            "config_id": spec.config_id,
        }
        for spec in train.DEFAULT_DATASETS
    ]
    return {"Llama-2-7B": llama, "Qwen2.5-7B-Instruct": QWEN_DATASETS}


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def summarize_file(model: str, spec: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(spec["path"])
    frame = pd.read_csv(path, encoding="utf-8-sig")
    valid = (
        numeric(frame, "lat_p95_s").gt(0)
        & numeric(frame, "tok_per_s").gt(0)
        & numeric(frame, "prompt_len").notna()
        & numeric(frame, "gen_len").notna()
    )
    clean = (
        valid
        & numeric(frame, "error_reqs").fillna(0).eq(0)
        & numeric(frame, "timeout_reqs").fillna(0).eq(0)
    )
    requests = numeric(frame, "total_reqs")
    successes = numeric(frame, "success_reqs").fillna(requests)
    # Output-token throughput is aggregate over concurrent requests, so this ratio
    # reconstructs active measured service time for each row.
    active_seconds = successes * numeric(frame, "gen_len") / numeric(frame, "tok_per_s")
    return {
        "model": model,
        "device_id": str(spec["device_id"]),
        "quant_method": str(spec["quant_method"]),
        "config_id": str(spec["config_id"]),
        "source_csv": str(spec["path"]),
        "raw_grid_rows": int(len(frame)),
        "valid_grid_rows": int(valid.sum()),
        "clean_grid_rows": int(clean.sum()),
        "measured_requests": int(requests[valid].sum()),
        "successful_requests": int(successes[valid].sum()),
        "active_service_gpu_hours": float(active_seconds[valid].sum() / 3600.0),
        "clean_active_service_gpu_hours": float(active_seconds[clean].sum() / 3600.0),
        "warmup_recorded": "warmup" in frame.columns,
        "rounds_recorded": "rounds" in frame.columns,
    }


def main() -> None:
    args = parse_args()
    rows = [
        summarize_file(model, spec)
        for model, specs in dataset_groups().items()
        for spec in specs
    ]
    result = pd.DataFrame(rows)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        args.report_dir / "PROFILING_COST_BY_GROUP.csv",
        index=False,
        encoding="utf-8-sig",
    )
    totals = (
        result.groupby("model", sort=False)[
            [
                "raw_grid_rows",
                "valid_grid_rows",
                "clean_grid_rows",
                "measured_requests",
                "successful_requests",
                "active_service_gpu_hours",
                "clean_active_service_gpu_hours",
            ]
        ]
        .sum()
        .reset_index()
    )
    totals["device_config_groups"] = result.groupby("model", sort=False).size().to_numpy()
    totals.to_csv(
        args.report_dir / "PROFILING_COST_TOTALS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# Profiling Cost Reconstructed from Benchmark Logs",
        "",
        "Active service GPU-hours are reconstructed as the sum of `successful_requests * generation_length / aggregate_output_throughput` over measured grid rows. This includes the timed service interval represented by throughput but excludes model loading, recorded warmups, inter-run gaps, and failed requests; it is therefore a reproducible lower bound on end-to-end campaign time rather than a billing claim.",
        "",
        "| Model | Device/config groups | Raw rows | Clean rows | Measured requests | Successful requests | Active service GPU-hours | Clean-row GPU-hours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in totals.to_dict(orient="records"):
        lines.append(
            "| {model} | {device_config_groups:.0f} | {raw_grid_rows:.0f} | {clean_grid_rows:.0f} | "
            "{measured_requests:.0f} | {successful_requests:.0f} | {active_service_gpu_hours:.2f} | "
            "{clean_active_service_gpu_hours:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The core Llama campaign contains 64 coarse workload tuples per device/configuration group (prompt length, generation length, and concurrency) and 30 timed rounds per concurrency level. The 42 rows removed before training contain errors/timeouts or otherwise fail the validity filter. Qwen is a separate scale-validation campaign and is not counted in the 534-row main predictor dataset.",
        ]
    )
    report = args.report_dir / "PROFILING_COST.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
