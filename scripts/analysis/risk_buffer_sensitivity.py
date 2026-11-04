from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "outputs" / "runs" / "20260712_review_response_easy_items_v1" / "risk_buffer_sensitivity"
REPORT_ROOT = ROOT / "outputs" / "reports" / "20260712_review_response_easy_items_v1"
FINAL = ROOT / "final" / "unified_scheduler_dual_experiments_v1"


PROFILES = {
    "main_controlled": {
        "model_src": FINAL / "models" / "main_controlled",
        "eval_script": ROOT / "run_scheduler_policy_baselines.py",
        "eval_extra": [
            "--catalog",
            str(ROOT / "scheduler_catalog.json"),
            "--prediction-source",
            "table",
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
        ],
    },
    "short_trace_final500": {
        "model_src": FINAL / "models" / "short_trace_final500",
        "eval_script": ROOT / "run_scheduler_policy_baselines_trace.py",
        "eval_extra": [
            "--catalog",
            str(ROOT / "scheduler_catalog.json"),
            "--workload-csv",
            str(ROOT / "data" / "workload_traces" / "wildchat_turn_trace_short_bucketed.csv"),
            "--max-trace-cases",
            "500",
            "--policies",
            "scheduler_pred,cloud_only,robust_mef_adapted,llm_pq_adapted",
            "--prediction-source",
            "table",
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
        ],
    },
}

QUANTILES = {
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
}


def _error_column(columns: Iterable[str]) -> str:
    names = set(columns)
    for column in ["p95_calibrated_abs_err_s", "p95_abs_err_s"]:
        if column in names:
            return column
    raise ValueError("combined_perf_with_pred.csv has no p95 absolute-error column")


def prepare_model_variant(profile: str, quantile_name: str, quantile: float) -> Dict[str, float]:
    spec = PROFILES[profile]
    src = Path(spec["model_src"])
    dst = RUN_ROOT / profile / quantile_name / "models"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    table_path = dst / "combined_perf_with_pred.csv"
    frame = pd.read_csv(table_path)
    err_col = _error_column(frame.columns)
    frame[err_col] = pd.to_numeric(frame[err_col], errors="coerce")

    quantile_values: List[float] = []
    for (_, _), index in frame.groupby(["device_id", "config_id"]).groups.items():
        values = frame.loc[index, err_col].dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        q_value = float(np.quantile(values, quantile))
        frame.loc[index, err_col] = q_value
        quantile_values.append(q_value)

    frame.to_csv(table_path, index=False, encoding="utf-8-sig")

    return {
        "config_count": float(len(quantile_values)),
        "mean_buffer_s": float(np.mean(quantile_values)) if quantile_values else float("nan"),
        "median_buffer_s": float(np.median(quantile_values)) if quantile_values else float("nan"),
        "min_buffer_s": float(np.min(quantile_values)) if quantile_values else float("nan"),
        "max_buffer_s": float(np.max(quantile_values)) if quantile_values else float("nan"),
    }


def run_eval(profile: str, quantile_name: str) -> Path:
    spec = PROFILES[profile]
    model_dir = RUN_ROOT / profile / quantile_name / "models"
    result_dir = RUN_ROOT / profile / quantile_name / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    if (result_dir / "baseline_policy_summary.csv").exists():
        return result_dir
    cmd = [
        sys.executable,
        str(spec["eval_script"]),
        "--artifacts-dir",
        str(model_dir),
        "--prediction-table-csv",
        str(model_dir / "combined_perf_with_pred.csv"),
        "--outdir",
        str(result_dir),
        *spec["eval_extra"],
    ]
    (RUN_ROOT / profile / quantile_name / "command.txt").write_text(
        " ".join(str(part) for part in cmd) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    (RUN_ROOT / profile / quantile_name / "command.log").write_text(completed.stdout, encoding="utf-8")
    return result_dir


def read_scheduler_rows(profile: str, quantile_name: str, buffer_stats: Dict[str, float]) -> List[Dict[str, object]]:
    summary_path = RUN_ROOT / profile / quantile_name / "results" / "baseline_policy_summary.csv"
    rows: List[Dict[str, object]] = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("policy") != "scheduler_pred":
                continue
            ecr = row.get("mean_edge_capture_ratio", row.get("tcr", "nan"))
            tps = row.get("mean_tok_per_s", row.get("mean_actual_tok_per_s", "nan"))
            rows.append(
                {
                    "profile": profile,
                    "risk_buffer": quantile_name,
                    "device_id": row["device_id"],
                    "mean_total_cost": float(row["mean_total_cost"]),
                    "mean_p95_s": float(row["mean_actual_p95_s"]),
                    "mean_edge_capture_ratio": float(ecr),
                    "slo_hit_rate": float(row["slo_hit_rate"]),
                    "mean_tps": float(tps),
                    **buffer_stats,
                }
            )
    return rows


def write_summary(rows: List[Dict[str, object]]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_ROOT / "RISK_BUFFER_SENSITIVITY.csv"
    fieldnames = [
        "profile",
        "risk_buffer",
        "device_id",
        "mean_total_cost",
        "mean_p95_s",
        "mean_edge_capture_ratio",
        "slo_hit_rate",
        "mean_tps",
        "config_count",
        "mean_buffer_s",
        "median_buffer_s",
        "min_buffer_s",
        "max_buffer_s",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Risk-Buffer Sensitivity",
        "",
        "This report evaluates p50, p75, and p90 empirical absolute-error risk buffers",
        "using isolated copies of the final model artifacts. The official final",
        "artifacts are not modified.",
        "",
        "Method: for each profile and risk-buffer setting, the script copies the final",
        "model directory into `outputs/runs/20260712_review_response_easy_items_v1/`",
        "and rewrites only the copied `combined_perf_with_pred.csv` absolute-error",
        "column so that the scheduler's existing p75 statistics correspond to the",
        "requested original quantile. Prediction columns and truth tables are left",
        "unchanged.",
        "",
        "## Scheduler Results",
        "",
        "| Profile | Buffer | Device | Cost | p95 | ECR | SLO | Tok/s | Median buffer s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {risk_buffer} | {device_id} | {mean_total_cost:.3f} | "
            "{mean_p95_s:.3f} | {mean_edge_capture_ratio:.3f} | {slo_hit_rate:.3f} | "
            "{mean_tps:.2f} | {median_buffer_s:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- p50 is the least conservative buffer and may admit more edge choices.",
            "- p90 is the most conservative buffer and may reject more edge choices.",
            "- p75 is retained as the manuscript default when it preserves SLO behavior",
            "  while avoiding the most conservative throughput loss.",
        ]
    )
    (REPORT_ROOT / "RISK_BUFFER_SENSITIVITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, object] = {"profiles": {}, "quantiles": QUANTILES}
    rows: List[Dict[str, object]] = []
    for profile in PROFILES:
        metadata["profiles"][profile] = {}
        for quantile_name, quantile in QUANTILES.items():
            stats = prepare_model_variant(profile, quantile_name, quantile)
            result_dir = run_eval(profile, quantile_name)
            metadata["profiles"][profile][quantile_name] = {
                "models": str(RUN_ROOT / profile / quantile_name / "models"),
                "results": str(result_dir),
                "buffer_stats": stats,
            }
            rows.extend(read_scheduler_rows(profile, quantile_name, stats))
    (RUN_ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(rows)


if __name__ == "__main__":
    main()
