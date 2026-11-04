from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "outputs/runs/20260716_cloud_condition_sensitivity_v1"
DEFAULT_REPORT_ROOT = ROOT / "outputs/reports/20260716_cloud_condition_sensitivity_v1"
SOURCE_CATALOG = ROOT / "scheduler_catalog.json"
ARTIFACTS = ROOT / "outputs/runs/20260711_llama_burstgpt_v1/models/burstgpt_final500"
WORKLOAD = ROOT / "data/workload_traces/burstgpt_stratified500_bucketed.csv"
EVAL_SCRIPT = ROOT / "scripts" / "evaluation" / "run_scheduler_policy_baselines_trace.py"


SCENARIOS: dict[str, dict[str, float]] = {
    "favorable_cloud": {
        "base_latency_s": 0.12,
        "network_rtt_s": 0.02,
        "prompt_upload_per_token_s": 0.00009,
        "response_download_per_token_s": 0.00004,
        "prefill_per_token_s": 0.00025,
        "decode_tokens_per_s": 480.0,
    },
    "nominal": {},
    "high_rtt": {"network_rtt_s": 0.20},
    "low_bandwidth": {
        "prompt_upload_per_token_s": 0.00072,
        "response_download_per_token_s": 0.00032,
    },
    "slow_cloud_compute": {
        "prefill_per_token_s": 0.00070,
        "decode_tokens_per_s": 160.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TE-PQoS under isolated analytic cloud-condition changes."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--action", choices=["run", "report", "all"], default="all"
    )
    return parser.parse_args()


def scenario_catalog(source: dict[str, Any], changes: dict[str, float]) -> dict[str, Any]:
    catalog = json.loads(json.dumps(source))
    clouds = catalog.get("cloud_candidates", [])
    if len(clouds) != 1:
        raise ValueError(f"Expected exactly one cloud candidate, found {len(clouds)}")
    perf_model = clouds[0].get("perf_model", {})
    perf_model.update(changes)
    clouds[0]["note"] = (
        "Analytic cloud condition used only for one-factor sensitivity evaluation."
    )
    return catalog


def run_scenarios(run_root: Path) -> None:
    source = json.loads(SOURCE_CATALOG.read_text(encoding="utf-8"))
    config_dir = run_root / "configs"
    result_root = run_root / "results"
    config_dir.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    for name, changes in SCENARIOS.items():
        catalog_path = config_dir / f"scheduler_catalog_{name}.json"
        catalog_path.write_text(
            json.dumps(scenario_catalog(source, changes), indent=2), encoding="utf-8"
        )
        outdir = result_root / name
        command = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--artifacts-dir",
            str(ARTIFACTS),
            "--prediction-table-csv",
            str(ARTIFACTS / "combined_perf_with_pred.csv"),
            "--outdir",
            str(outdir),
            "--catalog",
            str(catalog_path),
            "--prediction-source",
            "table",
            "--disable-learned-policy",
            "--policies",
            "cloud_only,scheduler_pred,robust_mef_adapted",
            "--edge-mem-list",
            "6500,7500,8500,10000",
            "--static-policy-map",
            "edge_3090=gptq4_bs8_tok2048_kvauto,edge_4090=gptq4_bs8_tok2048_kvauto",
            "--workload-csv",
            str(WORKLOAD),
            "--max-trace-cases",
            "500",
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
        ]
        print(f"[scenario:{name}] {' '.join(command)}")
        subprocess.run(command, cwd=ROOT, check=True)

    manifest = {
        "experiment": run_root.name,
        "model": "Llama-2-7B AWQ/GPTQ predictor",
        "workload": str(WORKLOAD),
        "workload_cases": 500,
        "fixed_scheduler": "TE-PQoS with the same parameters and predictor in every scenario",
        "source_cloud_model": str(SOURCE_CATALOG),
        "scenarios": SCENARIOS,
    }
    (run_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_report(run_root: Path, report_root: Path) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE_CATALOG.read_text(encoding="utf-8"))
    nominal = source["cloud_candidates"][0]["perf_model"]
    rows: list[dict[str, str]] = []
    for scenario in SCENARIOS:
        summary = run_root / "results" / scenario / "baseline_policy_summary.csv"
        for row in read_summary(summary):
            if row["policy"] in {"cloud_only", "scheduler_pred"}:
                rows.append({"scenario": scenario, **row})

    csv_path = report_root / "CLOUD_CONDITION_SENSITIVITY.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Cloud-Condition Sensitivity",
        "",
        "The predictor, workload, scheduler parameters, SLO grid, and memory grid are fixed. Only the analytic cloud model changes.",
        "",
        "Nominal cloud coefficients: "
        + ", ".join(f"`{key}={value}`" for key, value in nominal.items())
        + ".",
        "",
        "| Scenario | Device | Policy | SLO hit | Edge route | Mean p95 s | Mean tok/s | Mean cost |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {device_id} | {policy} | {slo_hit_rate:.3f} | {tcr:.3f} | "
            "{mean_actual_p95_s:.3f} | {mean_actual_tok_per_s:.2f} | {mean_total_cost:.3f} |".format(
                scenario=row["scenario"],
                device_id=row["device_id"],
                policy=row["policy"],
                slo_hit_rate=float(row["slo_hit_rate"]),
                tcr=float(row["tcr"]),
                mean_actual_p95_s=float(row["mean_actual_p95_s"]),
                mean_actual_tok_per_s=float(row["mean_actual_tok_per_s"]),
                mean_total_cost=float(row["mean_total_cost"]),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation is intentionally limited to this controlled analytic reference; these scenarios are not measurements of a named cloud provider.",
        ]
    )
    report = report_root / "CLOUD_CONDITION_SENSITIVITY.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    if args.action in {"run", "all"}:
        run_scenarios(args.run_root)
    if args.action in {"report", "all"}:
        print(write_report(args.run_root, args.report_root))


if __name__ == "__main__":
    main()
