import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import scheduler as scheduler_core
from .features import load_json

DEFAULT_ARTIFACTS_DIR = "artifacts_gbdt_multi"
DEFAULT_CATALOG = "scheduler_catalog.json"
DEFAULT_OUTDIR = os.path.join(DEFAULT_ARTIFACTS_DIR, "scheduler_batch_eval")
DEFAULT_DEVICES = "edge_3090,edge_4090"
DEFAULT_SLO_LIST = "0.75,1.0,2.0,4.0"
DEFAULT_EDGE_MEM_LIST = "6500,7500,8500,10000"
DEFAULT_PREDICTION_SOURCE = "model"
DEFAULT_PREDICTION_TABLE_CSV = os.path.join(DEFAULT_ARTIFACTS_DIR, "combined_perf_with_pred.csv")
DEFAULT_PREDICTION_P95_COL = "p95_calibrated_s"
DEFAULT_PREDICTION_TPS_COL = "tps_hat"


def parse_csv_list(raw_value: str, caster):
    values = []
    for item in str(raw_value or "").split(","):
        text = item.strip()
        if not text:
            continue
        values.append(caster(text))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline batch evaluation for the scheduler against static baselines.",
    )
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--devices", default=DEFAULT_DEVICES)
    parser.add_argument("--slo-list", default=DEFAULT_SLO_LIST)
    parser.add_argument("--edge-mem-list", default=DEFAULT_EDGE_MEM_LIST)
    parser.add_argument(
        "--workload-mode",
        default="intersection",
        choices=["intersection", "union"],
        help="Use only workloads common to all edge configs on each device, or the union of all profiled workloads.",
    )
    parser.add_argument("--prediction-source", default=DEFAULT_PREDICTION_SOURCE, choices=["model", "table"])
    parser.add_argument("--prediction-table-csv", default=DEFAULT_PREDICTION_TABLE_CSV)
    parser.add_argument("--prediction-p95-col", default=DEFAULT_PREDICTION_P95_COL)
    parser.add_argument("--prediction-tps-col", default=DEFAULT_PREDICTION_TPS_COL)
    parser.add_argument("--edge-queue-delay-s", type=float, default=0.0)
    parser.add_argument("--cloud-extra-delay-s", type=float, default=0.0)
    parser.add_argument("--latency-weight", type=float, default=1.0)
    parser.add_argument("--memory-weight", type=float, default=0.3)
    parser.add_argument("--throughput-weight", type=float, default=0.2)
    parser.add_argument("--cloud-route-penalty", type=float, default=0.8)
    parser.add_argument("--min-throughput-tps", type=float, default=None)
    parser.add_argument("--throughput-floor-ratio", type=float, default=0.85)
    parser.add_argument("--throughput-tie-ratio", type=float, default=scheduler_core.DEFAULT_THROUGHPUT_TIE_RATIO)
    parser.add_argument("--trace-small-max-prompt-len", type=int, default=scheduler_core.DEFAULT_TRACE_SMALL_MAX_PROMPT_LEN)
    parser.add_argument("--trace-small-max-gen-len", type=int, default=scheduler_core.DEFAULT_TRACE_SMALL_MAX_GEN_LEN)
    parser.add_argument("--trace-medium-max-prompt-len", type=int, default=scheduler_core.DEFAULT_TRACE_MEDIUM_MAX_PROMPT_LEN)
    parser.add_argument("--trace-medium-max-gen-len", type=int, default=scheduler_core.DEFAULT_TRACE_MEDIUM_MAX_GEN_LEN)
    parser.add_argument("--trace-small-edge-min-tps-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MIN_TPS_RATIO)
    parser.add_argument("--trace-medium-edge-min-tps-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MIN_TPS_RATIO)
    parser.add_argument("--trace-large-edge-min-tps-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MIN_TPS_RATIO)
    parser.add_argument("--trace-small-edge-max-slo-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MAX_SLO_RATIO)
    parser.add_argument("--trace-medium-edge-max-slo-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MAX_SLO_RATIO)
    parser.add_argument("--trace-large-edge-max-slo-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MAX_SLO_RATIO)
    parser.add_argument("--trace-small-edge-max-cloud-lat-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument("--trace-medium-edge-max-cloud-lat-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument("--trace-large-edge-max-cloud-lat-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument("--trace-small-latency-rescue-min-cloud-pressure-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_SMALL_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument("--trace-medium-latency-rescue-min-cloud-pressure-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_MEDIUM_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument("--trace-large-latency-rescue-min-cloud-pressure-ratio", type=float, default=scheduler_core.DEFAULT_TRACE_LARGE_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument("--disable-trace-edge-rescue", action="store_true")
    parser.add_argument("--cloud-throughput-mode", choices=["decode_only", "service_effective"], default=scheduler_core.DEFAULT_CLOUD_THROUGHPUT_MODE)
    parser.add_argument("--optimize-for", choices=["balanced", "latency", "throughput", "throughput_constrained", "throughput_constrained_trace", "throughput_constrained_trace_distilled", "latency_constrained"], default="balanced")
    parser.add_argument("--slo-safety-margin", type=float, default=1.0)
    parser.add_argument("--soft-slo-tolerance-ratio", type=float, default=0.05)
    parser.add_argument("--tie-cost-tolerance-ratio", type=float, default=0.05)
    parser.add_argument("--violation-tolerance-ratio", type=float, default=0.03)
    parser.add_argument("--memory-penalty-threshold", type=float, default=0.85)
    parser.add_argument("--reject-extrapolation", action="store_true")
    parser.add_argument("--no-prefer-edge", action="store_true")
    return parser.parse_args()


def make_scheduler_args(
    *,
    device_id: str,
    prompt_len: int,
    gen_len: int,
    concurrency: int,
    slo_p95_s: float,
    edge_mem_available_mb: float,
    config: argparse.Namespace,
) -> argparse.Namespace:
    return argparse.Namespace(
        edge_device_id=device_id,
        prompt_len=int(prompt_len),
        gen_len=int(gen_len),
        concurrency=int(concurrency),
        slo_p95_s=float(slo_p95_s),
        edge_mem_available_mb=float(edge_mem_available_mb),
        edge_queue_delay_s=float(config.edge_queue_delay_s),
        cloud_extra_delay_s=float(config.cloud_extra_delay_s),
        latency_weight=float(config.latency_weight),
        memory_weight=float(config.memory_weight),
        throughput_weight=float(config.throughput_weight),
        cloud_route_penalty=float(config.cloud_route_penalty),
        optimize_for=str(getattr(config, "optimize_for", "balanced")),
        min_throughput_tps=getattr(config, "min_throughput_tps", None),
        throughput_floor_ratio=float(getattr(config, "throughput_floor_ratio", 0.85)),
        throughput_tie_ratio=float(getattr(config, "throughput_tie_ratio", scheduler_core.DEFAULT_THROUGHPUT_TIE_RATIO)),
        trace_small_max_prompt_len=int(getattr(config, "trace_small_max_prompt_len", scheduler_core.DEFAULT_TRACE_SMALL_MAX_PROMPT_LEN)),
        trace_small_max_gen_len=int(getattr(config, "trace_small_max_gen_len", scheduler_core.DEFAULT_TRACE_SMALL_MAX_GEN_LEN)),
        trace_medium_max_prompt_len=int(getattr(config, "trace_medium_max_prompt_len", scheduler_core.DEFAULT_TRACE_MEDIUM_MAX_PROMPT_LEN)),
        trace_medium_max_gen_len=int(getattr(config, "trace_medium_max_gen_len", scheduler_core.DEFAULT_TRACE_MEDIUM_MAX_GEN_LEN)),
        trace_small_edge_min_tps_ratio=float(getattr(config, "trace_small_edge_min_tps_ratio", scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MIN_TPS_RATIO)),
        trace_medium_edge_min_tps_ratio=float(getattr(config, "trace_medium_edge_min_tps_ratio", scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MIN_TPS_RATIO)),
        trace_large_edge_min_tps_ratio=float(getattr(config, "trace_large_edge_min_tps_ratio", scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MIN_TPS_RATIO)),
        trace_small_edge_max_slo_ratio=float(getattr(config, "trace_small_edge_max_slo_ratio", scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MAX_SLO_RATIO)),
        trace_medium_edge_max_slo_ratio=float(getattr(config, "trace_medium_edge_max_slo_ratio", scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MAX_SLO_RATIO)),
        trace_large_edge_max_slo_ratio=float(getattr(config, "trace_large_edge_max_slo_ratio", scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MAX_SLO_RATIO)),
        trace_small_edge_max_cloud_lat_ratio=float(getattr(config, "trace_small_edge_max_cloud_lat_ratio", scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MAX_CLOUD_LAT_RATIO)),
        trace_medium_edge_max_cloud_lat_ratio=float(getattr(config, "trace_medium_edge_max_cloud_lat_ratio", scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MAX_CLOUD_LAT_RATIO)),
        trace_large_edge_max_cloud_lat_ratio=float(getattr(config, "trace_large_edge_max_cloud_lat_ratio", scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MAX_CLOUD_LAT_RATIO)),
        trace_small_latency_rescue_min_cloud_pressure_ratio=float(getattr(config, "trace_small_latency_rescue_min_cloud_pressure_ratio", scheduler_core.DEFAULT_TRACE_SMALL_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)),
        trace_medium_latency_rescue_min_cloud_pressure_ratio=float(getattr(config, "trace_medium_latency_rescue_min_cloud_pressure_ratio", scheduler_core.DEFAULT_TRACE_MEDIUM_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)),
        trace_large_latency_rescue_min_cloud_pressure_ratio=float(getattr(config, "trace_large_latency_rescue_min_cloud_pressure_ratio", scheduler_core.DEFAULT_TRACE_LARGE_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)),
        disable_trace_edge_rescue=bool(getattr(config, "disable_trace_edge_rescue", False)),
        cloud_throughput_mode=str(getattr(config, "cloud_throughput_mode", scheduler_core.DEFAULT_CLOUD_THROUGHPUT_MODE)),
        slo_safety_margin=float(getattr(config, "slo_safety_margin", 1.0)),
        soft_slo_tolerance_ratio=float(getattr(config, "soft_slo_tolerance_ratio", 0.05)),
        tie_cost_tolerance_ratio=float(getattr(config, "tie_cost_tolerance_ratio", 0.05)),
        violation_tolerance_ratio=float(getattr(config, "violation_tolerance_ratio", 0.03)),
        memory_penalty_threshold=float(getattr(config, "memory_penalty_threshold", 0.85)),
        utility_cost_weight=float(getattr(config, "utility_cost_weight", scheduler_core.DEFAULT_UTILITY_COST_WEIGHT)),
        utility_latency_weight=float(getattr(config, "utility_latency_weight", scheduler_core.DEFAULT_UTILITY_LATENCY_WEIGHT)),
        utility_throughput_weight=float(getattr(config, "utility_throughput_weight", scheduler_core.DEFAULT_UTILITY_THROUGHPUT_WEIGHT)),
        utility_risk_weight=float(getattr(config, "utility_risk_weight", scheduler_core.DEFAULT_UTILITY_RISK_WEIGHT)),
        utility_memory_weight=float(getattr(config, "utility_memory_weight", scheduler_core.DEFAULT_UTILITY_MEMORY_WEIGHT)),
        utility_queue_weight=float(getattr(config, "utility_queue_weight", scheduler_core.DEFAULT_UTILITY_QUEUE_WEIGHT)),
        utility_switch_penalty=float(getattr(config, "utility_switch_penalty", scheduler_core.DEFAULT_UTILITY_SWITCH_PENALTY)),
        utility_min_gain=float(getattr(config, "utility_min_gain", scheduler_core.DEFAULT_UTILITY_MIN_GAIN)),
        reject_extrapolation=bool(config.reject_extrapolation),
        no_prefer_edge=bool(config.no_prefer_edge),
        pretty=False,
    )


def make_global_args(args: argparse.Namespace) -> argparse.Namespace:
    global_args = copy.copy(args)
    global_args.no_prefer_edge = True
    return global_args


def load_truth_tables(artifacts_dir: str) -> Tuple[Dict[str, Any], Dict[Tuple[str, str, int, int, int], Dict[str, Any]]]:
    manifest = load_json(os.path.join(artifacts_dir, "dataset_manifest.json"))
    lookup: Dict[Tuple[str, str, int, int, int], Dict[str, Any]] = {}

    for spec in manifest.get("datasets", []):
        csv_path = Path(spec["path"])
        df = pd.read_csv(csv_path)

        if "device_id" not in df.columns:
            df["device_id"] = spec["device_id"]
        else:
            df["device_id"] = df["device_id"].fillna(spec["device_id"]).replace("", spec["device_id"])

        if "config_id" not in df.columns:
            df["config_id"] = spec["config_id"]
        else:
            df["config_id"] = df["config_id"].fillna(spec["config_id"]).replace("", spec["config_id"])

        for column in ["prompt_len", "gen_len", "concurrency", "lat_p95_s", "tok_per_s", "error_reqs", "timeout_reqs"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            else:
                df[column] = 0

        for row in df.to_dict(orient="records"):
            device_id = str(row["device_id"])
            config_id = str(row["config_id"])
            key = (
                device_id,
                config_id,
                int(row["prompt_len"]),
                int(row["gen_len"]),
                int(row["concurrency"]),
            )
            error_reqs = int(row.get("error_reqs", 0) or 0)
            timeout_reqs = int(row.get("timeout_reqs", 0) or 0)
            lat_p95_s = float(row.get("lat_p95_s", np.nan))
            tok_per_s = float(row.get("tok_per_s", np.nan))
            row["success"] = bool(
                error_reqs <= 0
                and timeout_reqs <= 0
                and np.isfinite(lat_p95_s)
                and np.isfinite(tok_per_s)
                and lat_p95_s > 0
                and tok_per_s > 0
            )
            lookup[key] = row

    return manifest, lookup


def device_catalog_candidates(catalog: Dict[str, Any], device_id: str) -> List[Dict[str, Any]]:
    return [
        candidate
        for candidate in catalog.get("edge_candidates", [])
        if bool(candidate.get("enabled", True)) and str(candidate.get("device_id")) == device_id
    ]


def workload_tuples_for_candidate(
    truth_lookup: Dict[Tuple[str, str, int, int, int], Dict[str, Any]],
    device_id: str,
    config_id: str,
) -> set[Tuple[int, int, int]]:
    tuples = set()
    for (dev, cfg, prompt_len, gen_len, concurrency), row in truth_lookup.items():
        if dev != device_id or cfg != config_id:
            continue
        if bool(row.get("success", False)):
            tuples.add((int(prompt_len), int(gen_len), int(concurrency)))
    return tuples


def load_prediction_lookup(
    table_csv: str,
) -> Dict[Tuple[str, str, int, int, int], Dict[str, Any]]:
    csv_path = Path(table_csv)
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    lookup: Dict[Tuple[str, str, int, int, int], Dict[str, Any]] = {}
    required = ['device_id', 'config_id', 'prompt_len', 'gen_len', 'concurrency']
    for column in required:
        if column not in df.columns:
            raise ValueError(f'missing required column in prediction table: {column}')

    for row in df.to_dict(orient='records'):
        key = (
            str(row['device_id']),
            str(row['config_id']),
            int(row['prompt_len']),
            int(row['gen_len']),
            int(row['concurrency']),
        )
        lookup[key] = row
    return lookup



def predict_edge_candidate_from_table(
    *,
    candidate: Dict[str, Any],
    args: argparse.Namespace,
    assets: Dict[str, Any],
    lookup_row: Dict[str, Any],
    p95_col: str,
    tps_col: str,
) -> Dict[str, Any]:
    core_p95_s = float(lookup_row.get(p95_col, np.nan))
    core_p95_raw_s = float(lookup_row.get('p95_hat_s', core_p95_s))
    predicted_tok_per_s = float(lookup_row.get(tps_col, np.nan))
    if not np.isfinite(core_p95_s):
        core_p95_s = core_p95_raw_s

    extra_latency_s = float(candidate.get('extra_latency_s', 0.0)) + float(args.edge_queue_delay_s)
    predicted_p95_s = core_p95_s + extra_latency_s
    predicted_p95_raw_s = core_p95_raw_s + extra_latency_s

    result = scheduler_core.static_candidate_fields(candidate)
    result.update(
        {
            'predicted_p95_s': predicted_p95_s,
            'predicted_p95_raw_s': predicted_p95_raw_s,
            'core_p95_s': core_p95_s,
            'core_p95_raw_s': core_p95_raw_s,
            'p95_calibration_applied': p95_col != 'p95_hat_s',
            'p95_calibration_delta_s': core_p95_s - core_p95_raw_s,
            'predicted_tok_per_s': predicted_tok_per_s,
            'estimated_memory_mb': candidate.get('memory_mb'),
            'within_slo': predicted_p95_s <= float(args.slo_p95_s),
            'within_safe_slo': predicted_p95_s <= scheduler_core.safe_slo_limit(args),
            'safe_slo_limit_s': scheduler_core.safe_slo_limit(args),
            'rejected_for_extrapolation': False,
            'extrapolation': {
                'device_id_unseen': False,
                'quant_method_unseen': False,
                'config_id_unseen': False,
                'kv_strategy_unseen': False,
            },
            'cost_bias': float(candidate.get('cost_bias', 0.0)),
            'extra_latency_s': extra_latency_s,
            'prediction_source': 'table',
        }
    )
    result.update(scheduler_core.runtime_constraints(result, args))
    scheduler_core.annotate_candidate_risk(
        result,
        args,
        risk_buffer_s=scheduler_core.lookup_candidate_risk_buffer_s(
            assets=assets,
            device_id=str(candidate.get('device_id')),
            config_id=str(candidate.get('config_id')),
            target=str(result['target']),
        ),
    )
    return result


def build_workload_cases(
    catalog: Dict[str, Any],
    truth_lookup: Dict[Tuple[str, str, int, int, int], Dict[str, Any]],
    devices: Sequence[str],
    workload_mode: str,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    for device_id in devices:
        edge_candidates = device_catalog_candidates(catalog, device_id)
        workload_sets = [
            workload_tuples_for_candidate(truth_lookup, device_id, str(candidate["config_id"]))
            for candidate in edge_candidates
        ]
        if not workload_sets:
            continue

        tuples = workload_sets[0].copy()
        if workload_mode == "intersection":
            for current in workload_sets[1:]:
                tuples &= current
        else:
            for current in workload_sets[1:]:
                tuples |= current

        for prompt_len, gen_len, concurrency in sorted(tuples):
            cases.append(
                {
                    "device_id": device_id,
                    "prompt_len": int(prompt_len),
                    "gen_len": int(gen_len),
                    "concurrency": int(concurrency),
                }
            )

    return cases


def build_predicted_candidates(
    *,
    catalog: Dict[str, Any],
    assets: Dict[str, Any],
    args: argparse.Namespace,
    prediction_source: str = DEFAULT_PREDICTION_SOURCE,
    prediction_lookup: Optional[Dict[Tuple[str, str, int, int, int], Dict[str, Any]]] = None,
    prediction_p95_col: str = DEFAULT_PREDICTION_P95_COL,
    prediction_tps_col: str = DEFAULT_PREDICTION_TPS_COL,
) -> List[Dict[str, Any]]:
    predicted: List[Dict[str, Any]] = []

    for candidate in catalog.get("edge_candidates", []):
        if not bool(candidate.get("edge_deployable", True)):
            continue
        if str(candidate.get("device_id")) != args.edge_device_id:
            continue

        if prediction_source == 'table' and prediction_lookup is not None:
            key = (
                str(candidate.get('device_id')),
                str(candidate.get('config_id')),
                int(args.prompt_len),
                int(args.gen_len),
                int(args.concurrency),
            )
            lookup_row = prediction_lookup.get(key)
            if lookup_row is not None:
                predicted.append(
                    predict_edge_candidate_from_table(
                        candidate=candidate,
                        args=args,
                        assets=assets,
                        lookup_row=lookup_row,
                        p95_col=prediction_p95_col,
                        tps_col=prediction_tps_col,
                    )
                )
                continue

        if prediction_source == 'table':
            scheduler_core.ensure_model_assets_loaded(assets)
        predicted.append(scheduler_core.predict_edge_candidate(candidate, args, assets))

    for candidate in catalog.get("cloud_candidates", []):
        predicted.append(scheduler_core.predict_cloud_candidate(candidate, args))

    scheduler_core.score_candidates(predicted, args)
    scheduler_core.apply_pairwise_rank_scores(predicted, args, assets)
    return predicted


def actualize_candidate(
    candidate: Dict[str, Any],
    *,
    truth_lookup: Dict[Tuple[str, str, int, int, int], Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    actual = copy.deepcopy(candidate)

    if actual["target"] == "cloud":
        actual["actual_status"] = "ok"
        actual["actual_p95_s"] = float(actual["predicted_p95_s"])
        actual["actual_tok_per_s"] = float(actual["predicted_tok_per_s"])
        actual["actual_error_reqs"] = 0
        actual["actual_timeout_reqs"] = 0
        actual["actual_peak_mem_mb"] = actual.get("estimated_memory_mb")
        return actual

    key = (
        str(actual["device_id"]),
        str(actual["config_id"]),
        int(args.prompt_len),
        int(args.gen_len),
        int(args.concurrency),
    )
    truth_row = truth_lookup.get(key)
    if truth_row is None:
        actual["actual_status"] = "missing_truth"
        actual["actual_p95_s"] = np.nan
        actual["actual_tok_per_s"] = np.nan
        actual["actual_error_reqs"] = np.nan
        actual["actual_timeout_reqs"] = np.nan
        actual["actual_peak_mem_mb"] = np.nan
        actual["predicted_p95_s"] = 1e9
        actual["predicted_tok_per_s"] = 0.0
        actual["within_slo"] = False
        return actual

    actual["actual_status"] = "ok" if bool(truth_row.get("success", False)) else "profile_error"
    actual["actual_p95_s"] = float(truth_row.get("lat_p95_s", np.nan))
    actual["actual_tok_per_s"] = float(truth_row.get("tok_per_s", np.nan))
    actual["actual_error_reqs"] = int(truth_row.get("error_reqs", 0) or 0)
    actual["actual_timeout_reqs"] = int(truth_row.get("timeout_reqs", 0) or 0)
    actual["actual_peak_mem_mb"] = truth_row.get("peak_mem_mb")

    if actual["actual_status"] != "ok":
        actual["predicted_p95_s"] = 1e9
        actual["predicted_tok_per_s"] = 0.0
        actual["within_slo"] = False
    else:
        actual["predicted_p95_s"] = float(actual["actual_p95_s"])
        actual["core_p95_s"] = float(actual["actual_p95_s"])
        actual["predicted_tok_per_s"] = float(actual["actual_tok_per_s"])
        actual["within_slo"] = float(actual["actual_p95_s"]) <= float(args.slo_p95_s)

    return actual


def find_candidate(candidates: Iterable[Dict[str, Any]], *, target: str, config_id: str) -> Optional[Dict[str, Any]]:
    for candidate in candidates:
        if str(candidate["target"]) == str(target) and str(candidate["config_id"]) == str(config_id):
            return candidate
    return None


def choose_fixed_policy(
    *,
    config_id: str,
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    predicted_edge = find_candidate(predicted_candidates, target="edge", config_id=config_id)
    if predicted_edge is None:
        predicted_cloud = find_candidate(predicted_candidates, target="cloud", config_id="fp16_bs16_tok2048")
        actual_cloud = find_candidate(actual_candidates, target="cloud", config_id="fp16_bs16_tok2048")
        return predicted_cloud, actual_cloud, "missing_edge_candidate_fallback_cloud"

    can_use_edge = bool(
        predicted_edge.get("within_memory", False)
        and predicted_edge.get("within_token_budget", False)
        and predicted_edge.get("within_batch_budget", False)
    )
    if can_use_edge:
        actual_edge = find_candidate(actual_candidates, target="edge", config_id=config_id)
        return predicted_edge, actual_edge, "fixed_edge_candidate"

    predicted_cloud = find_candidate(predicted_candidates, target="cloud", config_id="fp16_bs16_tok2048")
    actual_cloud = find_candidate(actual_candidates, target="cloud", config_id="fp16_bs16_tok2048")
    return predicted_cloud, actual_cloud, "runtime_fallback_to_cloud"


def selected_policy_row(
    *,
    policy: str,
    policy_type: str,
    selection_reason: str,
    case_info: Dict[str, Any],
    predicted_choice: Optional[Dict[str, Any]],
    actual_choice: Optional[Dict[str, Any]],
    global_oracle_choice: Optional[Dict[str, Any]],
    rule_oracle_choice: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    predicted_p95_s = np.nan
    predicted_tok_per_s = np.nan
    predicted_score = np.nan
    if predicted_choice is not None:
        predicted_p95_s = float(predicted_choice.get("predicted_p95_s", np.nan))
        predicted_tok_per_s = float(predicted_choice.get("predicted_tok_per_s", np.nan))
        predicted_score = float(predicted_choice.get("score", np.nan))

    actual_p95_s = np.nan
    actual_tok_per_s = np.nan
    actual_score = np.nan
    actual_status = "missing_choice"
    actual_within_slo = False
    selected_target = None
    selected_config_id = None
    selected_quant_method = None
    selected_batch_size = None
    selected_kv_strategy = None
    selected_estimated_memory_mb = np.nan
    actual_error_reqs = np.nan
    actual_timeout_reqs = np.nan
    actual_peak_mem_mb = np.nan
    within_memory = False
    within_token_budget = False
    within_batch_budget = False

    if actual_choice is not None:
        actual_status = str(actual_choice.get("actual_status", "missing_choice"))
        actual_p95_s = float(actual_choice.get("actual_p95_s", np.nan))
        actual_tok_per_s = float(actual_choice.get("actual_tok_per_s", np.nan))
        actual_score = float(actual_choice.get("score", np.nan))
        actual_within_slo = bool(actual_choice.get("within_slo", False))
        selected_target = actual_choice.get("target")
        selected_config_id = actual_choice.get("config_id")
        selected_quant_method = actual_choice.get("quant_method")
        selected_batch_size = actual_choice.get("batch_size")
        selected_kv_strategy = actual_choice.get("kv_strategy")
        selected_estimated_memory_mb = actual_choice.get("estimated_memory_mb")
        actual_error_reqs = actual_choice.get("actual_error_reqs")
        actual_timeout_reqs = actual_choice.get("actual_timeout_reqs")
        actual_peak_mem_mb = actual_choice.get("actual_peak_mem_mb")
        within_memory = bool(actual_choice.get("within_memory", False))
        within_token_budget = bool(actual_choice.get("within_token_budget", False))
        within_batch_budget = bool(actual_choice.get("within_batch_budget", False))

    global_oracle_p95_s = np.nan
    global_oracle_tok_per_s = np.nan
    global_oracle_score = np.nan
    global_oracle_target = None
    global_oracle_config_id = None
    if global_oracle_choice is not None:
        global_oracle_p95_s = float(global_oracle_choice.get("actual_p95_s", np.nan))
        global_oracle_tok_per_s = float(global_oracle_choice.get("actual_tok_per_s", np.nan))
        global_oracle_score = float(global_oracle_choice.get("score", np.nan))
        global_oracle_target = global_oracle_choice.get("target")
        global_oracle_config_id = global_oracle_choice.get("config_id")

    rule_oracle_target = None
    rule_oracle_config_id = None
    if rule_oracle_choice is not None:
        rule_oracle_target = rule_oracle_choice.get("target")
        rule_oracle_config_id = rule_oracle_choice.get("config_id")

    prediction_abs_err_p95_s = np.nan
    prediction_abs_err_tps = np.nan
    if np.isfinite(predicted_p95_s) and np.isfinite(actual_p95_s):
        prediction_abs_err_p95_s = abs(actual_p95_s - predicted_p95_s)
    if np.isfinite(predicted_tok_per_s) and np.isfinite(actual_tok_per_s):
        prediction_abs_err_tps = abs(actual_tok_per_s - predicted_tok_per_s)

    global_oracle_gap_score = np.nan
    global_oracle_gap_p95_s = np.nan
    global_oracle_gap_tps = np.nan
    if np.isfinite(actual_score) and np.isfinite(global_oracle_score):
        global_oracle_gap_score = actual_score - global_oracle_score
    if np.isfinite(actual_p95_s) and np.isfinite(global_oracle_p95_s):
        global_oracle_gap_p95_s = actual_p95_s - global_oracle_p95_s
    if np.isfinite(actual_tok_per_s) and np.isfinite(global_oracle_tok_per_s):
        global_oracle_gap_tps = global_oracle_tok_per_s - actual_tok_per_s

    matches_global_oracle = bool(
        selected_target is not None
        and global_oracle_target is not None
        and str(selected_target) == str(global_oracle_target)
        and str(selected_config_id) == str(global_oracle_config_id)
    )
    matches_rule_oracle = bool(
        selected_target is not None
        and rule_oracle_target is not None
        and str(selected_target) == str(rule_oracle_target)
        and str(selected_config_id) == str(rule_oracle_config_id)
    )

    row = dict(case_info)
    row.update(
        {
            "policy": policy,
            "policy_type": policy_type,
            "selection_reason": selection_reason,
            "selected_target": selected_target,
            "selected_config_id": selected_config_id,
            "selected_quant_method": selected_quant_method,
            "selected_batch_size": selected_batch_size,
            "selected_kv_strategy": selected_kv_strategy,
            "selected_estimated_memory_mb": selected_estimated_memory_mb,
            "actual_status": actual_status,
            "actual_p95_s": actual_p95_s,
            "actual_tok_per_s": actual_tok_per_s,
            "actual_score": actual_score,
            "actual_within_slo": actual_within_slo,
            "actual_error_reqs": actual_error_reqs,
            "actual_timeout_reqs": actual_timeout_reqs,
            "actual_peak_mem_mb": actual_peak_mem_mb,
            "within_memory": within_memory,
            "within_token_budget": within_token_budget,
            "within_batch_budget": within_batch_budget,
            "predicted_p95_s": predicted_p95_s,
            "predicted_tok_per_s": predicted_tok_per_s,
            "predicted_score": predicted_score,
            "prediction_abs_err_p95_s": prediction_abs_err_p95_s,
            "prediction_abs_err_tps": prediction_abs_err_tps,
            "global_oracle_target": global_oracle_target,
            "global_oracle_config_id": global_oracle_config_id,
            "global_oracle_p95_s": global_oracle_p95_s,
            "global_oracle_tok_per_s": global_oracle_tok_per_s,
            "global_oracle_score": global_oracle_score,
            "global_oracle_gap_score": global_oracle_gap_score,
            "global_oracle_gap_p95_s": global_oracle_gap_p95_s,
            "global_oracle_gap_tps": global_oracle_gap_tps,
            "rule_oracle_target": rule_oracle_target,
            "rule_oracle_config_id": rule_oracle_config_id,
            "matches_global_oracle": matches_global_oracle,
            "matches_rule_oracle": matches_rule_oracle,
            "route_is_edge": bool(selected_target == "edge"),
            "route_is_cloud": bool(selected_target == "cloud"),
        }
    )
    return row


def summarize_cases(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group_key, part in frame.groupby(list(group_cols), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(group_cols, group_key)}
        row.update(
            {
                "cases": int(len(part)),
                "ok_rate": float((part["actual_status"] == "ok").mean()),
                "slo_hit_rate": float(part["actual_within_slo"].fillna(False).mean()),
                "edge_route_rate": float(part["route_is_edge"].mean()),
                "cloud_route_rate": float(part["route_is_cloud"].mean()),
                "mean_actual_p95_s": float(part["actual_p95_s"].mean()),
                "median_actual_p95_s": float(part["actual_p95_s"].median()),
                "mean_actual_tok_per_s": float(part["actual_tok_per_s"].mean()),
                "mean_actual_score": float(part["actual_score"].mean()),
                "mean_global_oracle_gap_score": float(part["global_oracle_gap_score"].mean()),
                "mean_global_oracle_gap_p95_s": float(part["global_oracle_gap_p95_s"].mean()),
                "mean_global_oracle_gap_tps": float(part["global_oracle_gap_tps"].mean()),
                "global_oracle_match_rate": float(part["matches_global_oracle"].mean()),
                "rule_oracle_match_rate": float(part["matches_rule_oracle"].mean()),
                "mean_prediction_abs_err_p95_s": float(part["prediction_abs_err_p95_s"].mean()),
                "mean_prediction_abs_err_tps": float(part["prediction_abs_err_tps"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def best_static_baselines(summary: pd.DataFrame) -> List[Dict[str, Any]]:
    fixed = summary[summary["policy_type"] == "static"].copy()
    if fixed.empty:
        return []

    best_rows: List[Dict[str, Any]] = []
    for device_id, part in fixed.groupby("device_id", sort=True):
        best = part.sort_values(
            by=["mean_actual_score", "mean_global_oracle_gap_score", "mean_actual_p95_s", "policy"],
            ascending=[True, True, True, True],
        ).iloc[0]
        best_rows.append(
            {
                "device_id": device_id,
                "best_static_policy": best["policy"],
                "mean_actual_score": float(best["mean_actual_score"]),
                "slo_hit_rate": float(best["slo_hit_rate"]),
                "mean_actual_p95_s": float(best["mean_actual_p95_s"]),
                "mean_actual_tok_per_s": float(best["mean_actual_tok_per_s"]),
            }
        )
    return best_rows


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    devices = parse_csv_list(args.devices, str)
    slo_list = parse_csv_list(args.slo_list, float)
    edge_mem_list = parse_csv_list(args.edge_mem_list, float)

    assets = scheduler_core.load_assets(args.artifacts_dir, prediction_source=args.prediction_source)
    catalog = load_json(args.catalog)
    manifest, truth_lookup = load_truth_tables(args.artifacts_dir)
    prediction_lookup = {}
    if args.prediction_source == 'table':
        prediction_lookup = load_prediction_lookup(args.prediction_table_csv)
    workload_cases = build_workload_cases(
        catalog=catalog,
        truth_lookup=truth_lookup,
        devices=devices,
        workload_mode=args.workload_mode,
    )

    all_rows: List[Dict[str, Any]] = []

    for workload in workload_cases:
        device_id = str(workload["device_id"])
        fixed_config_ids = [
            str(candidate["config_id"])
            for candidate in device_catalog_candidates(catalog, device_id)
        ]

        for slo_p95_s in slo_list:
            for edge_mem_available_mb in edge_mem_list:
                scheduler_args = make_scheduler_args(
                    device_id=device_id,
                    prompt_len=int(workload["prompt_len"]),
                    gen_len=int(workload["gen_len"]),
                    concurrency=int(workload["concurrency"]),
                    slo_p95_s=float(slo_p95_s),
                    edge_mem_available_mb=float(edge_mem_available_mb),
                    config=args,
                )
                global_args = make_global_args(scheduler_args)

                predicted_candidates = build_predicted_candidates(
                    catalog=catalog,
                    assets=assets,
                    args=scheduler_args,
                    prediction_source=args.prediction_source,
                    prediction_lookup=prediction_lookup,
                    prediction_p95_col=args.prediction_p95_col,
                    prediction_tps_col=args.prediction_tps_col,
                )
                actual_candidates = [
                    actualize_candidate(candidate, truth_lookup=truth_lookup, args=scheduler_args)
                    for candidate in predicted_candidates
                ]
                scheduler_core.score_candidates(actual_candidates, scheduler_args)
                rule_oracle_choice, _, _ = scheduler_core.choose_candidate(actual_candidates, scheduler_args)
                global_oracle_choice, _, _ = scheduler_core.choose_candidate(actual_candidates, global_args)
                scheduler_choice_pred, scheduler_reason, _ = scheduler_core.choose_candidate(predicted_candidates, scheduler_args)
                scheduler_choice_actual = None
                if scheduler_choice_pred is not None:
                    scheduler_choice_actual = find_candidate(
                        actual_candidates,
                        target=str(scheduler_choice_pred["target"]),
                        config_id=str(scheduler_choice_pred["config_id"]),
                    )

                case_info = {
                    "device_id": device_id,
                    "prompt_len": int(workload["prompt_len"]),
                    "gen_len": int(workload["gen_len"]),
                    "concurrency": int(workload["concurrency"]),
                    "request_total_tokens": int(workload["prompt_len"]) + int(workload["gen_len"]),
                    "slo_p95_s": float(slo_p95_s),
                    "edge_mem_available_mb": float(edge_mem_available_mb),
                }

                all_rows.append(
                    selected_policy_row(
                        policy="scheduler_pred",
                        policy_type="adaptive",
                        selection_reason=scheduler_reason,
                        case_info=case_info,
                        predicted_choice=scheduler_choice_pred,
                        actual_choice=scheduler_choice_actual,
                        global_oracle_choice=global_oracle_choice,
                        rule_oracle_choice=rule_oracle_choice,
                    )
                )

                cloud_pred = find_candidate(predicted_candidates, target="cloud", config_id="fp16_bs16_tok2048")
                cloud_actual = find_candidate(actual_candidates, target="cloud", config_id="fp16_bs16_tok2048")
                all_rows.append(
                    selected_policy_row(
                        policy="cloud_only",
                        policy_type="static",
                        selection_reason="always_cloud",
                        case_info=case_info,
                        predicted_choice=cloud_pred,
                        actual_choice=cloud_actual,
                        global_oracle_choice=global_oracle_choice,
                        rule_oracle_choice=rule_oracle_choice,
                    )
                )

                for config_id in fixed_config_ids:
                    pred_choice, actual_choice, reason = choose_fixed_policy(
                        config_id=config_id,
                        predicted_candidates=predicted_candidates,
                        actual_candidates=actual_candidates,
                    )
                    all_rows.append(
                        selected_policy_row(
                            policy=f"fixed_{config_id}",
                            policy_type="static",
                            selection_reason=reason,
                            case_info=case_info,
                            predicted_choice=pred_choice,
                            actual_choice=actual_choice,
                            global_oracle_choice=global_oracle_choice,
                            rule_oracle_choice=rule_oracle_choice,
                        )
                    )

                all_rows.append(
                    selected_policy_row(
                        policy="oracle_rule_actual",
                        policy_type="oracle",
                        selection_reason="best_actual_candidate_with_edge_preference",
                        case_info=case_info,
                        predicted_choice=rule_oracle_choice,
                        actual_choice=rule_oracle_choice,
                        global_oracle_choice=global_oracle_choice,
                        rule_oracle_choice=rule_oracle_choice,
                    )
                )
                all_rows.append(
                    selected_policy_row(
                        policy="oracle_global_actual",
                        policy_type="oracle",
                        selection_reason="best_actual_candidate_global",
                        case_info=case_info,
                        predicted_choice=global_oracle_choice,
                        actual_choice=global_oracle_choice,
                        global_oracle_choice=global_oracle_choice,
                        rule_oracle_choice=rule_oracle_choice,
                    )
                )

    case_df = pd.DataFrame(all_rows)
    summary_policy = summarize_cases(case_df, ["device_id", "policy", "policy_type"]).sort_values(
        by=["device_id", "policy_type", "mean_actual_score", "policy"],
        ascending=[True, True, True, True],
    )
    summary_policy_slo_mem = summarize_cases(
        case_df,
        ["device_id", "policy", "policy_type", "slo_p95_s", "edge_mem_available_mb"],
    ).sort_values(
        by=["device_id", "policy_type", "policy", "slo_p95_s", "edge_mem_available_mb"],
        ascending=[True, True, True, True, True],
    )

    case_csv = os.path.join(args.outdir, "batch_case_results.csv")
    summary_csv = os.path.join(args.outdir, "batch_policy_summary.csv")
    summary_slo_mem_csv = os.path.join(args.outdir, "batch_policy_summary_by_slo_mem.csv")
    metadata_json = os.path.join(args.outdir, "batch_eval_metadata.json")
    best_static_json = os.path.join(args.outdir, "best_static_baselines.json")

    case_df.to_csv(case_csv, index=False, encoding="utf-8-sig")
    summary_policy.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_policy_slo_mem.to_csv(summary_slo_mem_csv, index=False, encoding="utf-8-sig")

    with open(metadata_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "devices": devices,
                "slo_list": slo_list,
                "edge_mem_list": edge_mem_list,
                "workload_mode": args.workload_mode,
                "workload_cases": len(workload_cases),
                "catalog_path": args.catalog,
                "artifacts_dir": args.artifacts_dir,
                "prediction_source": args.prediction_source,
                "prediction_table_csv": args.prediction_table_csv,
                "prediction_p95_col": args.prediction_p95_col,
                "prediction_tps_col": args.prediction_tps_col,
                "notes": [
                    "Edge memory values come from scheduler_catalog.json placeholders because peak_mem_mb was not collected in the current benchmark runs.",
                    "Cloud candidate uses the analytic model from scheduler_catalog.json as both predicted and actual reference.",
                    "oracle_rule_actual keeps scheduler prefer_edge behavior, while oracle_global_actual removes the edge-preference heuristic and picks the globally best feasible score.",
                ],
                "manifest_rows_after_clean": manifest.get("rows_after_clean"),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    best_static = best_static_baselines(summary_policy)
    with open(best_static_json, "w", encoding="utf-8") as handle:
        json.dump(best_static, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"case_csv": case_csv, "rows": int(len(case_df))}, ensure_ascii=False))
    print(json.dumps({"summary_csv": summary_csv, "summary_rows": int(len(summary_policy))}, ensure_ascii=False))
    print(json.dumps({"best_static_baselines": best_static}, ensure_ascii=False))


if __name__ == "__main__":
    main()


