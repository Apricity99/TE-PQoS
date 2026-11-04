from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tepqos import batch_eval
from tepqos import scheduler


DEFAULT_EXPERIMENT = "20260715_reviewer2_overhead_v2"
FINAL_MODELS = ROOT / "final/unified_scheduler_dual_experiments_v1/models"
MODEL_FILES = [
    "gbdt_p95.joblib", "gbdt_tps.joblib",
    "p95_calibrator.joblib", "pairwise_ranker.joblib",
]
CASES = [
    ("edge_3090", 16, 16, 1, 0.75, 8500.0),
    ("edge_3090", 96, 96, 4, 2.0, 10000.0),
    ("edge_4090", 48, 48, 2, 1.0, 8500.0),
    ("edge_4090", 192, 192, 8, 4.0, 10000.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Reviewer 2 overhead measurements.")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--full-repeats", type=int, default=100)
    parser.add_argument("--choice-repeats", type=int, default=1000)
    return parser.parse_args()


def scheduler_config() -> SimpleNamespace:
    return SimpleNamespace(
        edge_queue_delay_s=0.0,
        cloud_extra_delay_s=0.0,
        latency_weight=1.0,
        memory_weight=0.3,
        throughput_weight=0.2,
        cloud_route_penalty=0.8,
        min_throughput_tps=None,
        throughput_floor_ratio=0.85,
        throughput_tie_ratio=0.03,
        optimize_for="throughput_constrained_trace_distilled",
        cloud_throughput_mode="service_effective",
        trace_small_latency_rescue_min_cloud_pressure_ratio=0.20,
        trace_medium_edge_min_tps_ratio=0.95,
        reject_extrapolation=False,
        no_prefer_edge=False,
    )


def make_args(case: tuple[object, ...], config: SimpleNamespace) -> argparse.Namespace:
    device, prompt, generation, concurrency, slo, memory = case
    return batch_eval.make_scheduler_args(
        device_id=str(device), prompt_len=int(prompt), gen_len=int(generation),
        concurrency=int(concurrency), slo_p95_s=float(slo),
        edge_mem_available_mb=float(memory), config=config,
    )


def summarize_ms(values_s: list[float]) -> dict[str, float]:
    values_ms = np.asarray(values_s, dtype=float) * 1000.0
    return {
        "decisions": int(len(values_ms)),
        "mean_ms": float(values_ms.mean()),
        "median_ms": float(np.median(values_ms)),
        "p95_ms": float(np.quantile(values_ms, 0.95)),
    }


def train_profiles(run_root: Path) -> list[dict[str, object]]:
    profiles = {
        "main_controlled": [],
        "short_trace_weighted": [
            "--trace-weight-csv", str(ROOT / "data/workload_traces/wildchat_turn_trace_short_bucketed.csv"),
            "--trace-weight-scope", "prompt_gen", "--trace-weight-alpha", "1.0",
            "--trace-weight-min", "0.25", "--trace-weight-max", "4.0",
        ],
    }
    rows = []
    for name, extra in profiles.items():
        profile_root = run_root / f"train_{name}"
        model_dir = profile_root / "models"
        profile_root.mkdir(parents=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "training" / "train_gbdt_multi.py"), "--outdir", str(model_dir), *extra]
        (profile_root / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        started = time.perf_counter()
        completed = subprocess.run(
            cmd, cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        elapsed = time.perf_counter() - started
        (profile_root / "command.log").write_text(completed.stdout, encoding="utf-8")
        sizes = {filename: (model_dir / filename).stat().st_size for filename in MODEL_FILES}
        rows.append(
            {
                "profile": name,
                "training_s": elapsed,
                "rows": 534,
                "model_bytes": sum(sizes.values()),
                "model_mb_decimal": sum(sizes.values()) / 1_000_000.0,
                "model_files": sizes,
            }
        )
    return rows


def benchmark(full_repeats: int, choice_repeats: int) -> dict[str, object]:
    model_dir = FINAL_MODELS / "main_controlled"
    assets = scheduler.load_assets(str(model_dir), prediction_source="model")
    catalog = json.loads((ROOT / "scheduler_catalog.json").read_text(encoding="utf-8"))
    config = scheduler_config()
    prepared = []
    for case in CASES:
        args = make_args(case, config)
        candidates = batch_eval.build_predicted_candidates(
            catalog=catalog, assets=assets, args=args, prediction_source="model"
        )
        scheduler.choose_candidate(candidates, args)
        prepared.append((args, candidates))

    full_times: list[float] = []
    for _ in range(full_repeats):
        for case in CASES:
            args = make_args(case, config)
            started = time.perf_counter()
            candidates = batch_eval.build_predicted_candidates(
                catalog=catalog, assets=assets, args=args, prediction_source="model"
            )
            scheduler.choose_candidate(candidates, args)
            full_times.append(time.perf_counter() - started)

    choice_times: list[float] = []
    for _ in range(choice_repeats):
        for args, candidates in prepared:
            started = time.perf_counter()
            scheduler.choose_candidate(candidates, args)
            choice_times.append(time.perf_counter() - started)
    return {
        "request_shapes": len(CASES),
        "candidates_per_request": len(prepared[0][1]),
        "full": summarize_ms(full_times),
        "choice_only": summarize_ms(choice_times),
    }


def write_report(
    report_root: Path, training: list[dict[str, object]], benchmark_result: dict[str, object]
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
        },
        "training": training,
        "benchmark": benchmark_result,
    }
    (report_root / "OVERHEAD_RESULTS.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Predictor and Scheduler Overhead", "",
        "Fresh training runs use isolated output directories. Online timing begins after all model artifacts are loaded.", "",
        "## Training and Serialized Size", "",
        "| Profile | Clean rows | Training (s) | Four-model package (MB) |",
        "|---|---:|---:|---:|",
    ]
    for row in training:
        lines.append(
            f"| {row['profile']} | {row['rows']} | {row['training_s']:.2f} | {row['model_mb_decimal']:.3f} |"
        )
    full = benchmark_result["full"]
    choice = benchmark_result["choice_only"]
    lines += [
        "", "## Loaded-Model Online Microbenchmark", "",
        f"Four representative request shapes and {benchmark_result['candidates_per_request']} candidates per request are used.",
        "", "| Path | Decisions | Mean (ms) | Median (ms) | p95 (ms) |",
        "|---|---:|---:|---:|---:|",
        f"| Prediction + scoring + ranking + choice | {full['decisions']} | {full['mean_ms']:.3f} | {full['median_ms']:.3f} | {full['p95_ms']:.3f} |",
        f"| Choice only with prepared candidates | {choice['decisions']} | {choice['mean_ms']:.3f} | {choice['median_ms']:.3f} | {choice['p95_ms']:.3f} |",
        "", "These timings characterize the current single-threaded Python implementation, not an optimized serving integration.",
    ]
    (report_root / "OVERHEAD_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = ROOT / "outputs/runs" / args.experiment_name
    report_root = ROOT / "outputs/reports/20260714_reviewer2_reproducibility_v2"
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite {run_root}; choose a new --experiment-name")
    run_root.mkdir(parents=True)
    training = train_profiles(run_root)
    benchmark_result = benchmark(args.full_repeats, args.choice_repeats)
    write_report(report_root, training, benchmark_result)
    print(report_root / "OVERHEAD_RESULTS.md")


if __name__ == "__main__":
    main()
