from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


RELEASE_DIR = Path(__file__).resolve().parent
# This launcher lives directly under ``experiments/`` in the source release.
PP_ROOT = RELEASE_DIR.parent
MODELS_DIR = RELEASE_DIR / "models"
RESULTS_DIR = RELEASE_DIR / "results"
SNAPSHOTS_DIR = RELEASE_DIR / "snapshots"
SUMMARY_DIR = RELEASE_DIR / "summary"


def _profile_definitions(*, strict_same_branch: bool = False) -> Dict[str, Dict[str, Any]]:
    if strict_same_branch:
        shared_trace_branch_args = [
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
        ]
        return {
            "main_controlled": {
                "description": "Controlled main experiment with main-data training and the strict same-branch unified scheduler.",
                "train_extra": [],
                "eval_script": PP_ROOT / "run_scheduler_policy_baselines.py",
                "eval_extra": [
                    "--catalog",
                    str(PP_ROOT / "scheduler_catalog.json"),
                    *shared_trace_branch_args,
                ],
                "existing_result_dir": None,
            },
            "short_trace_final500": {
                "description": "WildChat short final500 with short-trace-weighted training and the strict same-branch unified scheduler.",
                "train_extra": [
                    "--trace-weight-csv",
                    str(PP_ROOT / "data" / "workload_traces" / "wildchat_turn_trace_short_bucketed.csv"),
                    "--trace-weight-scope",
                    "prompt_gen",
                    "--trace-weight-alpha",
                    "1.0",
                    "--trace-weight-min",
                    "0.25",
                    "--trace-weight-max",
                    "4.0",
                ],
                "eval_script": PP_ROOT / "run_scheduler_policy_baselines_trace.py",
                "eval_extra": [
                    "--catalog",
                    str(PP_ROOT / "scheduler_catalog.json"),
                    "--workload-csv",
                    str(PP_ROOT / "data" / "workload_traces" / "wildchat_turn_trace_short_bucketed.csv"),
                    "--max-trace-cases",
                    "500",
                    *shared_trace_branch_args,
                ],
                "existing_result_dir": None,
            },
        }
    return {
        "main_controlled": {
            "description": "Controlled main experiment with main-data training and the unified scheduler core rule.",
            "train_extra": [],
            "eval_script": PP_ROOT / "run_scheduler_policy_baselines.py",
            "eval_extra": [
                "--catalog",
                str(PP_ROOT / "scheduler_catalog.json"),
                "--prediction-source",
                "table",
                "--disable-learned-policy",
                "--optimize-for",
                "throughput_constrained",
                "--cloud-throughput-mode",
                "decode_only",
                "--throughput-tie-ratio",
                "0.03",
            ],
            "existing_result_dir": PP_ROOT / "artifacts_gbdt_multi_unified_mainfit_v1" / "search_u3_main",
        },
        "short_trace_final500": {
            "description": "WildChat short final500 with short-trace-weighted training and trace-adaptive unified scheduling.",
            "train_extra": [
                "--trace-weight-csv",
                str(PP_ROOT / "data" / "workload_traces" / "wildchat_turn_trace_short_bucketed.csv"),
                "--trace-weight-scope",
                "prompt_gen",
                "--trace-weight-alpha",
                "1.0",
                "--trace-weight-min",
                "0.25",
                "--trace-weight-max",
                "4.0",
            ],
            "eval_script": PP_ROOT / "run_scheduler_policy_baselines_trace.py",
            "eval_extra": [
                "--catalog",
                str(PP_ROOT / "scheduler_catalog.json"),
                "--workload-csv",
                str(PP_ROOT / "data" / "workload_traces" / "wildchat_turn_trace_short_bucketed.csv"),
                "--max-trace-cases",
                "500",
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
            "existing_result_dir": PP_ROOT
            / "artifacts_gbdt_multi_wildchat_short_weighted"
            / "trace_tune_short_cloudservice020_medium095_tie003_final500_allpolicies_v1",
        },
    }


CURRENT_PROFILES = _profile_definitions()


def ensure_release_layout() -> None:
    for path in [MODELS_DIR, RESULTS_DIR, SNAPSHOTS_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def parse_profiles(raw: str, profiles_map: Dict[str, Dict[str, Any]]) -> List[str]:
    text = str(raw or "all").strip()
    if not text or text == "all":
        return list(profiles_map.keys())
    selected = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(profiles_map))
    if unknown:
        raise ValueError(f"unknown profiles: {unknown}")
    return selected


def profile_model_dir(profile_name: str) -> Path:
    return MODELS_DIR / profile_name


def profile_result_dir(profile_name: str) -> Path:
    return RESULTS_DIR / profile_name


def profile_snapshot_dir(profile_name: str) -> Path:
    return SNAPSHOTS_DIR / profile_name


def train_command(profile_name: str) -> List[str]:
    profile = CURRENT_PROFILES[profile_name]
    model_dir = profile_model_dir(profile_name)
    return [
        sys.executable,
        str(PP_ROOT / "train_gbdt_multi.py"),
        "--outdir",
        str(model_dir),
        *profile["train_extra"],
    ]


def eval_command(profile_name: str) -> List[str]:
    profile = CURRENT_PROFILES[profile_name]
    model_dir = profile_model_dir(profile_name)
    result_dir = profile_result_dir(profile_name)
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


def run_command(cmd: List[str], *, cwd: Path) -> None:
    print(f"$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def print_commands(profile_names: Iterable[str]) -> None:
    for profile_name in profile_names:
        print(f"[{profile_name}]")
        print(f"train: {shlex.join(train_command(profile_name))}")
        print(f"eval : {shlex.join(eval_command(profile_name))}")
        print()


def run_train(profile_names: Iterable[str]) -> None:
    ensure_release_layout()
    for profile_name in profile_names:
        profile_model_dir(profile_name).mkdir(parents=True, exist_ok=True)
        run_command(train_command(profile_name), cwd=PP_ROOT)


def run_eval(profile_names: Iterable[str]) -> None:
    ensure_release_layout()
    for profile_name in profile_names:
        profile_result_dir(profile_name).mkdir(parents=True, exist_ok=True)
        run_command(eval_command(profile_name), cwd=PP_ROOT)


def copy_existing_result(profile_name: str) -> Path:
    ensure_release_layout()
    existing_result_dir = CURRENT_PROFILES[profile_name].get("existing_result_dir")
    if not existing_result_dir:
        raise ValueError(f"profile '{profile_name}' has no existing validated snapshot for the current mode")
    src = Path(existing_result_dir)
    dst = profile_snapshot_dir(profile_name)
    if not src.exists():
        raise FileNotFoundError(f"existing result dir not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def collect_existing(profile_names: Iterable[str]) -> None:
    manifest: Dict[str, Any] = {}
    for profile_name in profile_names:
        dst = copy_existing_result(profile_name)
        manifest[profile_name] = {
            "source_dir": str(CURRENT_PROFILES[profile_name]["existing_result_dir"]),
            "snapshot_dir": str(dst),
        }
    manifest_path = SUMMARY_DIR / "existing_snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _baseline_summary_dir(profile_name: str) -> Path | None:
    candidates = [
        profile_result_dir(profile_name),
        profile_snapshot_dir(profile_name),
    ]
    for directory in candidates:
        if (directory / "baseline_policy_summary.csv").exists():
            return directory
    return None


def _extract_policy_rows(rows: List[Dict[str, str]], policies: Iterable[str]) -> List[Dict[str, str]]:
    wanted = set(policies)
    return [row for row in rows if row.get("policy") in wanted]


def build_summary(profile_names: Iterable[str]) -> None:
    ensure_release_layout()
    summary_json: Dict[str, Any] = {}
    lines: List[str] = [
        "# Unified Scheduler Dual Experiments Summary",
        "",
        "This folder packages one scheduler implementation with two experiment profiles:",
        "",
        "- `main_controlled`: separate training on controlled profiling data, evaluated on the controlled main experiment.",
        "- `short_trace_final500`: separate training with WildChat-short trace weighting, evaluated on short-trace final500.",
        "",
        "Both profiles use the same scheduler code in `scheduler.py`; they differ only in training data and profile-specific scheduler arguments.",
        "",
    ]
    for profile_name in profile_names:
        summary_dir = _baseline_summary_dir(profile_name)
        if summary_dir is None:
            lines.extend(
                [
                    f"## {profile_name}",
                    "",
                    "No result directory found yet.",
                    "",
                ]
            )
            continue

        rows = _read_csv_rows(summary_dir / "baseline_policy_summary.csv")
        selected = _extract_policy_rows(
            rows,
            [
                "scheduler_pred",
                "robust_mef_adapted",
                "slo_goodput_greedy",
                "cloud_relative_greedy",
                "llm_pq_adapted",
                "cloud_only",
                "mef_adapted",
                "melange_adapted",
                "best_static_edge",
                "oracle_throughput_actual",
            ],
        )
        summary_json[profile_name] = {
            "summary_dir": str(summary_dir),
            "rows": selected,
            "description": CURRENT_PROFILES[profile_name]["description"],
        }

        lines.extend(
            [
                f"## {profile_name}",
                "",
                CURRENT_PROFILES[profile_name]["description"],
                "",
                f"Result directory: `{summary_dir}`",
                "",
                "| device | policy | mean tok/s | mean p95 (s) | slo hit | tcr |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in selected:
            device = row.get("device_id", row.get("\ufeffdevice_id", ""))
            lines.append(
                "| "
                + " | ".join(
                    [
                        device,
                        row.get("policy", ""),
                        row.get("mean_actual_tok_per_s", ""),
                        row.get("mean_actual_p95_s", ""),
                        row.get("slo_hit_rate", ""),
                        row.get("tcr", ""),
                    ]
                )
                + " |"
            )
        lines.append("")

    (SUMMARY_DIR / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (SUMMARY_DIR / "final_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    global CURRENT_PROFILES
    parser = argparse.ArgumentParser(
        description="Release runner for one unified scheduler implementation with separate training for main and short-trace experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["show", "train", "eval", "all", "collect-existing", "summarize"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--profiles", default="all")
        sub.add_argument("--strict-same-branch", action="store_true")

    args = parser.parse_args()
    CURRENT_PROFILES = _profile_definitions(strict_same_branch=bool(args.strict_same_branch))
    profile_names = parse_profiles(args.profiles, CURRENT_PROFILES)

    if args.command == "show":
        print_commands(profile_names)
        return
    if args.command == "train":
        run_train(profile_names)
        return
    if args.command == "eval":
        run_eval(profile_names)
        return
    if args.command == "all":
        run_train(profile_names)
        run_eval(profile_names)
        build_summary(profile_names)
        return
    if args.command == "collect-existing":
        collect_existing(profile_names)
        build_summary(profile_names)
        return
    if args.command == "summarize":
        build_summary(profile_names)
        return


if __name__ == "__main__":
    main()
