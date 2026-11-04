from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


PP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = PP_ROOT / "outputs" / "runs" / "20260708_qwen_scale_v1"
DEFAULT_REPORT_ROOT = PP_ROOT / "outputs" / "reports" / "20260708_qwen_scale_v1"
SOURCE_CATALOG = PP_ROOT / "configs" / "experiments" / "scheduler_catalog_qwen_scale_v1.json"
SHORT_TRACE_CSV = PP_ROOT / "data" / "workload_traces" / "wildchat_turn_trace_short_bucketed.csv"
BURSTGPT_TRACE_CSV = PP_ROOT / "data" / "workload_traces" / "burstgpt_without_fails_3_bucketed.csv"
BURSTGPT_STRATIFIED_TRACE_CSV = (
    PP_ROOT / "data" / "workload_traces" / "burstgpt_stratified500_bucketed.csv"
)

if str(PP_ROOT) not in sys.path:
    sys.path.insert(0, str(PP_ROOT))

QWEN_BS8 = "qwen2_5_7b_awq4_bs8_tok2048_kvauto"
QWEN_BS16 = "qwen2_5_7b_awq4_bs16_tok2048_kvauto"
EDGE_MEM_LIST = "9000,11000,13000,16000"
STATIC_POLICY_MAP = f"edge_3090={QWEN_BS8},edge_4090={QWEN_BS8}"


QWEN_DATASETS: List[Dict[str, Any]] = [
    {
        "path": "data/benchmarks/edge_3090/qwen_awq/edge_3090_qwen2_5_7b_awq4_bs8_tok2048_kvauto_valid.csv",
        "source_tag": "edge_3090_qwen2_5_awq_bs8_kvauto",
        "device_id": "edge_3090",
        "quant_method": "awq",
        "config_id": QWEN_BS8,
        "model_family": "qwen2_5_7b",
        "quant_bits": 4,
        "batch_size": 8,
        "max_tokens": 2048,
        "kv_strategy": "kvauto",
    },
    {
        "path": "data/benchmarks/edge_3090/qwen_awq/edge_3090_qwen2_5_7b_awq4_bs16_tok2048_kvauto_valid.csv",
        "source_tag": "edge_3090_qwen2_5_awq_bs16_kvauto",
        "device_id": "edge_3090",
        "quant_method": "awq",
        "config_id": QWEN_BS16,
        "model_family": "qwen2_5_7b",
        "quant_bits": 4,
        "batch_size": 16,
        "max_tokens": 2048,
        "kv_strategy": "kvauto",
    },
    {
        "path": "data/benchmarks/edge_4090/qwen_awq/edge_4090_qwen2_5_7b_awq4_bs8_tok2048_kvauto_valid.csv",
        "source_tag": "edge_4090_qwen2_5_awq_bs8_kvauto",
        "device_id": "edge_4090",
        "quant_method": "awq",
        "config_id": QWEN_BS8,
        "model_family": "qwen2_5_7b",
        "quant_bits": 4,
        "batch_size": 8,
        "max_tokens": 2048,
        "kv_strategy": "kvauto",
    },
    {
        "path": "data/benchmarks/edge_4090/qwen_awq/edge_4090_qwen2_5_7b_awq4_bs16_tok2048_kvauto_valid.csv",
        "source_tag": "edge_4090_qwen2_5_awq_bs16_kvauto",
        "device_id": "edge_4090",
        "quant_method": "awq",
        "config_id": QWEN_BS16,
        "model_family": "qwen2_5_7b",
        "quant_bits": 4,
        "batch_size": 16,
        "max_tokens": 2048,
        "kv_strategy": "kvauto",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Qwen2.5-7B-Instruct-AWQ scale experiment without touching existing formal results.",
    )
    parser.add_argument(
        "--action",
        choices=["commands", "train", "eval", "report", "all"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--profiles",
        default="all",
        help=(
            "Comma-separated profiles: main_controlled,short_trace_final500,"
            "burstgpt_final500,burstgpt_stratified500, or all."
        ),
    )
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    return parser.parse_args()


def parse_profiles(raw: str) -> List[str]:
    available = [
        "main_controlled",
        "short_trace_final500",
        "burstgpt_final500",
        "burstgpt_stratified500",
    ]
    text = str(raw or "all").strip()
    if not text or text == "all":
        return available
    selected = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown profiles: {unknown}")
    return selected


def profile_definitions(run_root: Path, catalog_path: Path) -> Dict[str, Dict[str, Any]]:
    shared_eval = [
        "--catalog",
        str(catalog_path),
        "--prediction-source",
        "table",
        "--disable-learned-policy",
        "--edge-mem-list",
        EDGE_MEM_LIST,
        "--static-policy-map",
        STATIC_POLICY_MAP,
    ]
    return {
        "main_controlled": {
            "model_dir": run_root / "models" / "main_controlled",
            "result_dir": run_root / "results" / "main_controlled",
            "train_extra": [],
            "eval_script": PP_ROOT / "scripts" / "evaluation" / "run_scheduler_policy_baselines.py",
            "eval_extra": [
                *shared_eval,
                "--optimize-for",
                "throughput_constrained",
                "--cloud-throughput-mode",
                "decode_only",
                "--throughput-tie-ratio",
                "0.03",
            ],
        },
        "short_trace_final500": {
            "model_dir": run_root / "models" / "short_trace_final500",
            "result_dir": run_root / "results" / "short_trace_final500",
            "train_extra": [
                "--trace-weight-csv",
                str(SHORT_TRACE_CSV),
                "--trace-weight-scope",
                "prompt_gen",
                "--trace-weight-alpha",
                "1.0",
                "--trace-weight-min",
                "0.25",
                "--trace-weight-max",
                "4.0",
            ],
            "eval_script": PP_ROOT / "scripts" / "evaluation" / "run_scheduler_policy_baselines_trace.py",
            "eval_extra": [
                *shared_eval,
                "--workload-csv",
                str(SHORT_TRACE_CSV),
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
            ],
        },
        "burstgpt_final500": {
            "model_dir": run_root / "models" / "burstgpt_final500",
            "result_dir": run_root / "results" / "burstgpt_final500",
            "train_extra": [
                "--trace-weight-csv",
                str(BURSTGPT_TRACE_CSV),
                "--trace-weight-scope",
                "prompt_gen",
                "--trace-weight-alpha",
                "1.0",
                "--trace-weight-min",
                "0.25",
                "--trace-weight-max",
                "4.0",
            ],
            "eval_script": PP_ROOT / "scripts" / "evaluation" / "run_scheduler_policy_baselines_trace.py",
            "eval_extra": [
                *shared_eval,
                "--workload-csv",
                str(BURSTGPT_TRACE_CSV),
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
            ],
        },
        "burstgpt_stratified500": {
            "model_dir": run_root / "models" / "burstgpt_stratified500",
            "result_dir": run_root / "results" / "burstgpt_stratified500",
            "train_extra": [
                "--trace-weight-csv",
                str(BURSTGPT_STRATIFIED_TRACE_CSV),
                "--trace-weight-scope",
                "prompt_gen",
                "--trace-weight-alpha",
                "1.0",
                "--trace-weight-min",
                "0.25",
                "--trace-weight-max",
                "4.0",
            ],
            "eval_script": PP_ROOT / "scripts" / "evaluation" / "run_scheduler_policy_baselines_trace.py",
            "eval_extra": [
                *shared_eval,
                "--workload-csv",
                str(BURSTGPT_STRATIFIED_TRACE_CSV),
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
            ],
        },
    }


def ensure_layout(run_root: Path, report_root: Path) -> None:
    for path in [
        run_root,
        run_root / "configs",
        run_root / "models",
        run_root / "results",
        report_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_enabled_catalog(run_root: Path) -> Path:
    catalog = json.loads(SOURCE_CATALOG.read_text(encoding="utf-8"))
    csv_by_candidate = {
        (item["device_id"], item["config_id"]): item["path"]
        for item in QWEN_DATASETS
    }
    catalog["status"] = "validated_for_scheduler_eval_without_measured_peak_memory"
    catalog["memory_note"] = (
        "Qwen benchmark CSVs have an empty peak_mem_mb column; memory_mb values are catalog estimates."
    )
    for candidate in catalog.get("edge_candidates", []):
        key = (str(candidate.get("device_id")), str(candidate.get("config_id")))
        candidate["enabled"] = True
        candidate["perf_source"] = "profiled_csv"
        candidate["profile_csv"] = csv_by_candidate.get(key)
        candidate["memory_source"] = "catalog_estimate_no_peak_mem_collected"
        candidate["note"] = (
            "Qwen2.5-7B-Instruct-AWQ profiled CSV validated; peak_mem_mb was not collected."
        )
    out_path = run_root / "configs" / "scheduler_catalog_qwen_scale_v1_enabled.json"
    out_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def qwen_dataset_specs() -> List[Any]:
    from tepqos import training as train

    return [train.DatasetSpec(**item) for item in QWEN_DATASETS]


def run_train(profile_name: str, profile: Dict[str, Any]) -> None:
    from tepqos import training as train

    model_dir = Path(profile["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    old_datasets = train.DEFAULT_DATASETS
    old_argv = sys.argv[:]
    try:
        train.DEFAULT_DATASETS = qwen_dataset_specs()
        sys.argv = [
            "train_gbdt_multi.py",
            "--outdir",
            str(model_dir),
            *profile["train_extra"],
        ]
        print(f"[train:{profile_name}] python {' '.join(shlex.quote(x) for x in sys.argv)}")
        train.main()
    finally:
        train.DEFAULT_DATASETS = old_datasets
        sys.argv = old_argv


def eval_command(profile: Dict[str, Any]) -> List[str]:
    model_dir = Path(profile["model_dir"])
    result_dir = Path(profile["result_dir"])
    return [
        sys.executable,
        str(profile["eval_script"]),
        "--artifacts-dir",
        str(model_dir),
        "--prediction-table-csv",
        str(model_dir / "combined_perf_with_pred.csv"),
        "--outdir",
        str(result_dir),
        *profile["eval_extra"],
    ]


def run_eval(profile_name: str, profile: Dict[str, Any]) -> None:
    result_dir = Path(profile["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = eval_command(profile)
    print(f"[eval:{profile_name}] {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=str(PP_ROOT), check=True)


def print_commands(profiles: Iterable[str], definitions: Dict[str, Dict[str, Any]]) -> None:
    for profile_name in profiles:
        profile = definitions[profile_name]
        train_cmd = [
            sys.executable,
            str(PP_ROOT / "scripts" / "evaluation" / "run_qwen_scale_v1.py"),
            "--action",
            "train",
            "--profiles",
            profile_name,
        ]
        print(f"[{profile_name}]")
        print(f"train: {shlex.join(train_cmd)}")
        print(f"eval : {shlex.join(eval_command(profile))}")
        print()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_summary_rows(summary_path: Path) -> List[Dict[str, str]]:
    policies = {
        "scheduler_pred",
        "robust_mef_adapted",
        "llm_pq_adapted",
        "slo_goodput_greedy",
        "cloud_relative_greedy",
        "best_static_edge",
        "cloud_only",
        "oracle_throughput_actual",
    }
    rows = [row for row in read_csv_rows(summary_path) if row.get("policy") in policies]
    rows.sort(key=lambda row: (row.get("device_id", ""), row.get("policy", "")))
    return rows


def format_float(value: str, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def write_report(profiles: Iterable[str], definitions: Dict[str, Dict[str, Any]], report_root: Path) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Qwen2.5 scale validation",
        "",
        "Model: Qwen2.5-7B-Instruct-AWQ. Devices: RTX 3090 and RTX 4090. Configurations: AWQ bs8/bs16, tok2048, kvauto.",
        "",
        "Important limitation: benchmark CSVs contain an empty `peak_mem_mb` column, so scheduler memory feasibility uses catalog estimates instead of measured peak memory.",
        "",
    ]
    for profile_name in profiles:
        profile = definitions[profile_name]
        model_dir = Path(profile["model_dir"])
        result_dir = Path(profile["result_dir"])
        summary_csv = result_dir / "baseline_policy_summary.csv"
        manifest_path = model_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        lines.extend(
            [
                f"## {profile_name}",
                "",
                f"- Model artifacts: `{model_dir}`",
                f"- Evaluation results: `{result_dir}`",
                f"- Rows after clean: `{manifest.get('rows_after_clean', 'missing')}`",
                "",
                "| device | policy | SLO hit | edge route/TCR | mean p95 s | mean tok/s | mean cost |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected_summary_rows(summary_csv):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.get("device_id", ""),
                        row.get("policy", ""),
                        format_float(row.get("slo_hit_rate", ""), 3),
                        format_float(row.get("tcr", ""), 3),
                        format_float(row.get("mean_actual_p95_s", ""), 3),
                        format_float(row.get("mean_actual_tok_per_s", ""), 2),
                        format_float(row.get("mean_total_cost", ""), 3),
                    ]
                )
                + " |"
            )
        lines.append("")

    out_path = report_root / "QWEN_RESULTS.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_experiment_manifest(run_root: Path, profiles: Iterable[str], definitions: Dict[str, Dict[str, Any]], catalog_path: Path) -> None:
    manifest = {
        "experiment": run_root.name,
        "model": "Qwen2.5-7B-Instruct-AWQ",
        "catalog": str(catalog_path),
        "datasets": QWEN_DATASETS,
        "profiles": {
            name: {
                "model_dir": str(definitions[name]["model_dir"]),
                "result_dir": str(definitions[name]["result_dir"]),
                "train_extra": definitions[name]["train_extra"],
                "eval_extra": definitions[name]["eval_extra"],
            }
            for name in profiles
        },
        "limitations": [
            "peak_mem_mb is empty in the benchmark CSVs; memory feasibility uses catalog estimates.",
            "The predictor is trained only on Qwen2.5 AWQ profiling data for this scale experiment.",
        ],
    }
    (run_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    os.chdir(PP_ROOT)
    run_root = Path(args.run_root)
    report_root = Path(args.report_root)
    profiles = parse_profiles(args.profiles)
    ensure_layout(run_root, report_root)
    catalog_path = write_enabled_catalog(run_root)
    definitions = profile_definitions(run_root, catalog_path)
    write_experiment_manifest(run_root, profiles, definitions, catalog_path)

    if args.action == "commands":
        print_commands(profiles, definitions)
        return

    if args.action in {"train", "all"}:
        for profile_name in profiles:
            run_train(profile_name, definitions[profile_name])

    if args.action in {"eval", "all"}:
        for profile_name in profiles:
            run_eval(profile_name, definitions[profile_name])

    if args.action in {"report", "all"}:
        report_path = write_report(profiles, definitions, report_root)
        print(json.dumps({"report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
