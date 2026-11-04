from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = "20260716_reviewer2_component_factorial_ablation_v3"
MODEL_SOURCE = (
    ROOT
    / "final"
    / "unified_scheduler_dual_experiments_v1"
    / "models"
    / "main_controlled"
)
CATALOG = ROOT / "scheduler_catalog.json"
RUNNER = ROOT / "scripts" / "evaluation" / "run_scheduler_batch_eval.py"
COMPONENTS = ("calibration", "risk_buffer", "edge_rescue")
CASE_KEYS = [
    "device_id",
    "prompt_len",
    "gen_len",
    "concurrency",
    "slo_p95_s",
    "edge_mem_available_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 2^3 factorial ablation of calibration, risk buffering, and edge rescue."
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse completed variants and regenerate the summaries.",
    )
    return parser.parse_args()


def experiment_paths(name: str) -> Dict[str, Path]:
    return {
        "run": ROOT / "outputs" / "runs" / name,
        "report": ROOT / "outputs" / "reports" / name,
    }


def variant_key(calibration: bool, risk_buffer: bool, edge_rescue: bool) -> str:
    return "c{calibration}_b{risk_buffer}_r{edge_rescue}".format(
        calibration=int(calibration),
        risk_buffer=int(risk_buffer),
        edge_rescue=int(edge_rescue),
    )


def variants() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for calibration, risk_buffer, edge_rescue in product([False, True], repeat=3):
        rows.append(
            {
                "key": variant_key(calibration, risk_buffer, edge_rescue),
                "calibration": calibration,
                "risk_buffer": risk_buffer,
                "edge_rescue": edge_rescue,
            }
        )
    return rows


def prepare_model_variant(variant: Dict[str, object], variant_root: Path) -> Path:
    model_dir = variant_root / "models"
    shutil.copytree(MODEL_SOURCE, model_dir)

    table_path = model_dir / "combined_perf_with_pred.csv"
    frame = pd.read_csv(table_path, encoding="utf-8-sig")
    risk_buffer = bool(variant["risk_buffer"])

    if not risk_buffer:
        for column in ["p95_abs_err_s", "p95_calibrated_abs_err_s"]:
            if column in frame.columns:
                frame[column] = 0.0
    frame.to_csv(table_path, index=False, encoding="utf-8-sig")
    return model_dir


def variant_command(
    variant: Dict[str, object], model_dir: Path, result_dir: Path
) -> List[str]:
    prediction_p95_col = (
        "p95_calibrated_s" if bool(variant["calibration"]) else "p95_hat_s"
    )
    cmd = [
        sys.executable,
        str(RUNNER),
        "--artifacts-dir",
        str(model_dir),
        "--catalog",
        str(CATALOG),
        "--prediction-source",
        "table",
        "--prediction-table-csv",
        str(model_dir / "combined_perf_with_pred.csv"),
        "--prediction-p95-col",
        prediction_p95_col,
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
    if not bool(variant["edge_rescue"]):
        cmd.append("--disable-trace-edge-rescue")
    return cmd


def run_variant(
    variant: Dict[str, object], run_root: Path, reuse_existing: bool
) -> Tuple[str, Path]:
    key = str(variant["key"])
    variant_root = run_root / key
    result_dir = variant_root / "results"
    summary_path = result_dir / "batch_policy_summary.csv"
    case_path = result_dir / "batch_case_results.csv"
    if summary_path.exists() and case_path.exists() and reuse_existing:
        return key, result_dir
    if variant_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite {variant_root}. Use --reuse-existing or a new experiment name."
        )

    variant_root.mkdir(parents=True)
    model_dir = prepare_model_variant(variant, variant_root)
    result_dir.mkdir()
    cmd = variant_command(variant, model_dir, result_dir)
    (variant_root / "command.txt").write_text(
        shlex.join(cmd) + "\n", encoding="utf-8"
    )
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (variant_root / "command.log").write_text(completed.stdout, encoding="utf-8")
    return key, result_dir


def scheduler_cases(result_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(result_dir / "batch_case_results.csv", encoding="utf-8-sig")
    return frame[frame["policy"] == "scheduler_pred"].copy()


def collect_factorial_summary(
    all_variants: Iterable[Dict[str, object]], result_dirs: Dict[str, Path]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for variant in all_variants:
        key = str(variant["key"])
        frame = pd.read_csv(
            result_dirs[key] / "batch_policy_summary.csv", encoding="utf-8-sig"
        )
        frame = frame[frame["policy"] == "scheduler_pred"]
        for _, row in frame.iterrows():
            rows.append(
                {
                    **variant,
                    "device_id": row["device_id"],
                    "cases": int(row["cases"]),
                    "mean_actual_tok_per_s": float(row["mean_actual_tok_per_s"]),
                    "slo_hit_rate": float(row["slo_hit_rate"]),
                    "edge_capture_ratio": float(row["edge_route_rate"]),
                    "mean_actual_p95_s": float(row["mean_actual_p95_s"]),
                }
            )
    return pd.DataFrame(rows).sort_values([*COMPONENTS, "device_id"]).reset_index(
        drop=True
    )


def decision_identity(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["selected_target"].fillna("").astype(str)
        + "|"
        + frame["selected_config_id"].fillna("").astype(str)
    )


def component_main_effects(
    all_variants: List[Dict[str, object]], result_dirs: Dict[str, Path]
) -> pd.DataFrame:
    cases_by_key = {
        str(variant["key"]): scheduler_cases(result_dirs[str(variant["key"])])
        for variant in all_variants
    }
    rows: List[Dict[str, object]] = []

    for component in COMPONENTS:
        other_components = [name for name in COMPONENTS if name != component]
        for device_id in ["edge_3090", "edge_4090"]:
            paired_rows: List[pd.DataFrame] = []
            decision_changes = 0
            pair_count = 0
            for other_values in product([False, True], repeat=2):
                flags = dict(zip(other_components, other_values))
                off_flags = {**flags, component: False}
                on_flags = {**flags, component: True}
                off_key = variant_key(**off_flags)
                on_key = variant_key(**on_flags)
                off = cases_by_key[off_key]
                on = cases_by_key[on_key]
                off = off[off["device_id"] == device_id].copy()
                on = on[on["device_id"] == device_id].copy()
                off["decision_off"] = decision_identity(off)
                on["decision_on"] = decision_identity(on)
                paired = off.merge(on, on=CASE_KEYS, suffixes=("_off", "_on"))
                if len(paired) != len(off) or len(paired) != len(on):
                    raise ValueError(f"case alignment failed for {component} on {device_id}")
                decision_changes += int(
                    (paired["decision_off"] != paired["decision_on"]).sum()
                )
                pair_count += len(paired)
                paired_rows.append(paired)

            combined = pd.concat(paired_rows, ignore_index=True)
            rows.append(
                {
                    "component": component,
                    "device_id": device_id,
                    "delta_tok_per_s": float(
                        (combined["actual_tok_per_s_on"] - combined["actual_tok_per_s_off"]).mean()
                    ),
                    "delta_slo_hit_rate": float(
                        (
                            combined["actual_within_slo_on"].astype(float)
                            - combined["actual_within_slo_off"].astype(float)
                        ).mean()
                    ),
                    "delta_edge_capture_ratio": float(
                        (
                            combined["route_is_edge_on"].astype(float)
                            - combined["route_is_edge_off"].astype(float)
                        ).mean()
                    ),
                    "delta_mean_p95_s": float(
                        (combined["actual_p95_s_on"] - combined["actual_p95_s_off"]).mean()
                    ),
                    "decision_changes": decision_changes,
                    "paired_decisions": pair_count,
                    "decision_change_rate": decision_changes / pair_count,
                }
            )
    return pd.DataFrame(rows)


def isolated_effects(
    summary: pd.DataFrame, result_dirs: Dict[str, Path]
) -> pd.DataFrame:
    baseline_key = variant_key(False, False, False)
    selected_keys = [
        baseline_key,
        variant_key(True, False, False),
        variant_key(False, True, False),
        variant_key(False, False, True),
        variant_key(True, True, True),
    ]
    labels = {
        baseline_key: "All off",
        variant_key(True, False, False): "Calibration only",
        variant_key(False, True, False): "p75 buffer only",
        variant_key(False, False, True): "Edge rescue only",
        variant_key(True, True, True): "All on",
    }
    baseline_cases = scheduler_cases(result_dirs[baseline_key])
    rows: List[Dict[str, object]] = []
    for key in selected_keys:
        current_cases = scheduler_cases(result_dirs[key])
        for device_id in ["edge_3090", "edge_4090"]:
            current_summary = summary[
                (summary["key"] == key) & (summary["device_id"] == device_id)
            ].iloc[0]
            off = baseline_cases[baseline_cases["device_id"] == device_id].copy()
            on = current_cases[current_cases["device_id"] == device_id].copy()
            off["decision_off"] = decision_identity(off)
            on["decision_on"] = decision_identity(on)
            paired = off.merge(on, on=CASE_KEYS, suffixes=("_off", "_on"))
            rows.append(
                {
                    "variant_key": key,
                    "variant_label": labels[key],
                    "device_id": device_id,
                    "mean_actual_tok_per_s": current_summary["mean_actual_tok_per_s"],
                    "slo_hit_rate": current_summary["slo_hit_rate"],
                    "edge_capture_ratio": current_summary["edge_capture_ratio"],
                    "mean_actual_p95_s": current_summary["mean_actual_p95_s"],
                    "decision_changes_vs_all_off": int(
                        (paired["decision_off"] != paired["decision_on"]).sum()
                    ),
                    "cases": int(current_summary["cases"]),
                }
            )
    return pd.DataFrame(rows)


def format_flag(value: object) -> str:
    return "on" if bool(value) else "off"


def write_report(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    isolated: pd.DataFrame,
    report_root: Path,
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        report_root / "FACTORIAL_ABLATION.csv", index=False, encoding="utf-8-sig"
    )
    effects.to_csv(
        report_root / "COMPONENT_MAIN_EFFECTS.csv", index=False, encoding="utf-8-sig"
    )
    isolated.to_csv(
        report_root / "ISOLATED_COMPONENT_EFFECTS.csv", index=False, encoding="utf-8-sig"
    )

    lines = [
        "# Calibration, Risk-Buffer, and Edge-Rescue Factorial Ablation",
        "",
        "This controlled replay uses a complete 2^3 design. Each component main effect",
        "is the mean of four matched on-minus-off comparisons while the other two",
        "component states are held fixed. The workload, candidate catalog, SLO grid,",
        "memory grid, cloud model, and all non-ablated scheduler thresholds are identical.",
        "",
        "## Component Main Effects",
        "",
        "| Component | Device | Delta tok/s | Delta SLO | Delta ECR | Delta p95 (s) | Changed decisions |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in effects.to_dict(orient="records"):
        lines.append(
            "| {component} | {device_id} | {delta_tok_per_s:+.3f} | "
            "{delta_slo_hit_rate:+.4f} | {delta_edge_capture_ratio:+.4f} | "
            "{delta_mean_p95_s:+.4f} | {decision_changes}/{paired_decisions} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Isolated Components and Full Method",
            "",
            "| Variant | Device | Tok/s | SLO | ECR | Mean p95 (s) | Decisions changed vs. all off |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in isolated.to_dict(orient="records"):
        lines.append(
            "| {variant_label} | {device_id} | {mean_actual_tok_per_s:.3f} | "
            "{slo_hit_rate:.4f} | {edge_capture_ratio:.4f} | {mean_actual_p95_s:.4f} | "
            "{decision_changes_vs_all_off}/{cases} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Complete Factorial Results",
            "",
            "| Calibration | Buffer | Rescue | Device | Tok/s | SLO | ECR | Mean p95 (s) |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.to_dict(orient="records"):
        formatted_row = {
            **row,
            "calibration": format_flag(row["calibration"]),
            "risk_buffer": format_flag(row["risk_buffer"]),
            "edge_rescue": format_flag(row["edge_rescue"]),
        }
        lines.append(
            "| {calibration} | {risk_buffer} | {edge_rescue} | {device_id} | "
            "{mean_actual_tok_per_s:.3f} | {slo_hit_rate:.4f} | "
            "{edge_capture_ratio:.4f} | {mean_actual_p95_s:.4f} |".format(
                **formatted_row
            )
        )

    lines.extend(
        [
            "",
            "All buffer-on variants use the same prespecified p75 residual-calibrated",
            "absolute-error buffer, so toggling calibration changes only the active p95",
            "prediction column. Buffer-off variants set only the copied error columns to",
            "zero. The source model artifacts and formal results are never modified.",
        ]
    )
    (report_root / "COMPONENT_FACTORIAL_ABLATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    paths = experiment_paths(args.experiment_name)
    if paths["run"].exists() and not args.reuse_existing:
        raise FileExistsError(
            f"Refusing to overwrite {paths['run']}. Choose a new experiment name."
        )
    paths["run"].mkdir(parents=True, exist_ok=True)

    all_variants = variants()
    result_dirs: Dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_variant, variant, paths["run"], bool(args.reuse_existing)
            ): str(variant["key"])
            for variant in all_variants
        }
        for future in as_completed(futures):
            key, result_dir = future.result()
            result_dirs[key] = result_dir
            print(f"completed {key}", flush=True)

    summary = collect_factorial_summary(all_variants, result_dirs)
    effects = component_main_effects(all_variants, result_dirs)
    isolated = isolated_effects(summary, result_dirs)
    write_report(summary, effects, isolated, paths["report"])

    metadata = {
        "experiment_name": args.experiment_name,
        "design": "2^3 full factorial",
        "components": list(COMPONENTS),
        "model_source": str(MODEL_SOURCE),
        "catalog": str(CATALOG),
        "runner": str(RUNNER),
        "workers": args.workers,
        "variants": all_variants,
    }
    (paths["run"] / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(paths["report"] / "COMPONENT_FACTORIAL_ABLATION.md")


if __name__ == "__main__":
    main()
