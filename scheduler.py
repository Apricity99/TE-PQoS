import argparse
import csv
import json
import os
import sys
import types
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores", category=UserWarning)

def patch_optional_pyarrow() -> None:
    try:
        import pyarrow  # type: ignore  # noqa: F401
    except Exception:
        fake_module = types.ModuleType("pyarrow")
        fake_module.__version__ = "17.0.0"
        sys.modules["pyarrow"] = fake_module


patch_optional_pyarrow()



import joblib
import numpy as np

from perf_feature_utils import (
    build_feature_row,
    build_p95_calibration_row,
    build_pairwise_feature_row,
    known_values_from_manifest,
    load_json,
)

DEFAULT_ARTIFACTS_DIR = "artifacts_gbdt_multi"
DEFAULT_CATALOG = "scheduler_catalog.json"
DEFAULT_DEVICE_RATES = {
    "edge_3090": 1.0,
    "edge_4090": 1.3,
    "edge_4060": 0.85,
    "cloud_a100": 4.0,
}
DEFAULT_SLO_SAFETY_MARGIN = 1.0
DEFAULT_SOFT_SLO_TOLERANCE_RATIO = 0.05
DEFAULT_TIE_COST_TOLERANCE_RATIO = 0.05
DEFAULT_VIOLATION_TOLERANCE_RATIO = 0.03
DEFAULT_MEMORY_PENALTY_THRESHOLD = 0.85
DEFAULT_UTILITY_COST_WEIGHT = 1.0
DEFAULT_UTILITY_LATENCY_WEIGHT = 0.2
DEFAULT_UTILITY_THROUGHPUT_WEIGHT = 0.0
DEFAULT_UTILITY_RISK_WEIGHT = 3.0
DEFAULT_UTILITY_MEMORY_WEIGHT = 0.1
DEFAULT_UTILITY_QUEUE_WEIGHT = 0.1
DEFAULT_UTILITY_SWITCH_PENALTY = 0.08
DEFAULT_UTILITY_MIN_GAIN = 0.0
DEFAULT_LATENCY_SHORTLIST_RATIO = 0.12
DEFAULT_LATENCY_CRITICAL_RATIO = 0.78
DEFAULT_LATENCY_CRITICAL_SHORTLIST_RATIO = 0.04
DEFAULT_EDGE_SELECTION_BONUS = 0.02
DEFAULT_THROUGHPUT_FLOOR_RATIO = 0.85
DEFAULT_THROUGHPUT_TIE_RATIO = 0.0
DEFAULT_TRACE_SMALL_MAX_PROMPT_LEN = 48
DEFAULT_TRACE_SMALL_MAX_GEN_LEN = 48
DEFAULT_TRACE_MEDIUM_MAX_PROMPT_LEN = 96
DEFAULT_TRACE_MEDIUM_MAX_GEN_LEN = 96
DEFAULT_TRACE_SMALL_EDGE_MIN_TPS_RATIO = 0.60
DEFAULT_TRACE_MEDIUM_EDGE_MIN_TPS_RATIO = 0.80
DEFAULT_TRACE_LARGE_EDGE_MIN_TPS_RATIO = 1.00
DEFAULT_TRACE_SMALL_EDGE_MAX_SLO_RATIO = 1.00
DEFAULT_TRACE_MEDIUM_EDGE_MAX_SLO_RATIO = 0.98
DEFAULT_TRACE_LARGE_EDGE_MAX_SLO_RATIO = 0.95
DEFAULT_TRACE_SMALL_EDGE_MAX_CLOUD_LAT_RATIO = 1.00
DEFAULT_TRACE_MEDIUM_EDGE_MAX_CLOUD_LAT_RATIO = 0.98
DEFAULT_TRACE_LARGE_EDGE_MAX_CLOUD_LAT_RATIO = 1.00
DEFAULT_TRACE_SMALL_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO = 0.0
DEFAULT_TRACE_MEDIUM_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO = 0.0
DEFAULT_TRACE_LARGE_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO = 0.0
DEFAULT_CLOUD_THROUGHPUT_MODE = "decode_only"




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latency/throughput-aware online scheduler over edge/cloud compression candidates.",
    )
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--edge-device-id", required=True)
    parser.add_argument("--prompt-len", required=True, type=int)
    parser.add_argument("--gen-len", required=True, type=int)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--slo-p95-s", required=True, type=float)
    parser.add_argument("--edge-mem-available-mb", type=float, default=None)
    parser.add_argument("--edge-queue-delay-s", type=float, default=0.0)
    parser.add_argument("--cloud-extra-delay-s", type=float, default=0.0)
    parser.add_argument("--latency-weight", type=float, default=1.0)
    parser.add_argument("--memory-weight", type=float, default=0.3)
    parser.add_argument("--throughput-weight", type=float, default=0.2)
    parser.add_argument("--cloud-route-penalty", type=float, default=0.8)
    parser.add_argument("--slo-safety-margin", type=float, default=DEFAULT_SLO_SAFETY_MARGIN)
    parser.add_argument("--soft-slo-tolerance-ratio", type=float, default=DEFAULT_SOFT_SLO_TOLERANCE_RATIO)
    parser.add_argument("--tie-cost-tolerance-ratio", type=float, default=DEFAULT_TIE_COST_TOLERANCE_RATIO)
    parser.add_argument("--violation-tolerance-ratio", type=float, default=DEFAULT_VIOLATION_TOLERANCE_RATIO)
    parser.add_argument("--memory-penalty-threshold", type=float, default=DEFAULT_MEMORY_PENALTY_THRESHOLD)
    parser.add_argument("--utility-cost-weight", type=float, default=DEFAULT_UTILITY_COST_WEIGHT)
    parser.add_argument("--utility-latency-weight", type=float, default=DEFAULT_UTILITY_LATENCY_WEIGHT)
    parser.add_argument("--utility-throughput-weight", type=float, default=DEFAULT_UTILITY_THROUGHPUT_WEIGHT)
    parser.add_argument("--utility-risk-weight", type=float, default=DEFAULT_UTILITY_RISK_WEIGHT)
    parser.add_argument("--utility-memory-weight", type=float, default=DEFAULT_UTILITY_MEMORY_WEIGHT)
    parser.add_argument("--utility-queue-weight", type=float, default=DEFAULT_UTILITY_QUEUE_WEIGHT)
    parser.add_argument("--utility-switch-penalty", type=float, default=DEFAULT_UTILITY_SWITCH_PENALTY)
    parser.add_argument("--utility-min-gain", type=float, default=DEFAULT_UTILITY_MIN_GAIN)
    parser.add_argument("--latency-shortlist-ratio", type=float, default=DEFAULT_LATENCY_SHORTLIST_RATIO)
    parser.add_argument("--latency-critical-ratio", type=float, default=DEFAULT_LATENCY_CRITICAL_RATIO)
    parser.add_argument("--latency-critical-shortlist-ratio", type=float, default=DEFAULT_LATENCY_CRITICAL_SHORTLIST_RATIO)
    parser.add_argument("--edge-selection-bonus", type=float, default=DEFAULT_EDGE_SELECTION_BONUS)
    parser.add_argument("--min-throughput-tps", type=float, default=None)
    parser.add_argument("--throughput-floor-ratio", type=float, default=DEFAULT_THROUGHPUT_FLOOR_RATIO)
    parser.add_argument("--throughput-tie-ratio", type=float, default=DEFAULT_THROUGHPUT_TIE_RATIO)
    parser.add_argument("--trace-small-max-prompt-len", type=int, default=DEFAULT_TRACE_SMALL_MAX_PROMPT_LEN)
    parser.add_argument("--trace-small-max-gen-len", type=int, default=DEFAULT_TRACE_SMALL_MAX_GEN_LEN)
    parser.add_argument("--trace-medium-max-prompt-len", type=int, default=DEFAULT_TRACE_MEDIUM_MAX_PROMPT_LEN)
    parser.add_argument("--trace-medium-max-gen-len", type=int, default=DEFAULT_TRACE_MEDIUM_MAX_GEN_LEN)
    parser.add_argument("--trace-small-edge-min-tps-ratio", type=float, default=DEFAULT_TRACE_SMALL_EDGE_MIN_TPS_RATIO)
    parser.add_argument("--trace-medium-edge-min-tps-ratio", type=float, default=DEFAULT_TRACE_MEDIUM_EDGE_MIN_TPS_RATIO)
    parser.add_argument("--trace-large-edge-min-tps-ratio", type=float, default=DEFAULT_TRACE_LARGE_EDGE_MIN_TPS_RATIO)
    parser.add_argument("--trace-small-edge-max-slo-ratio", type=float, default=DEFAULT_TRACE_SMALL_EDGE_MAX_SLO_RATIO)
    parser.add_argument("--trace-medium-edge-max-slo-ratio", type=float, default=DEFAULT_TRACE_MEDIUM_EDGE_MAX_SLO_RATIO)
    parser.add_argument("--trace-large-edge-max-slo-ratio", type=float, default=DEFAULT_TRACE_LARGE_EDGE_MAX_SLO_RATIO)
    parser.add_argument("--trace-small-edge-max-cloud-lat-ratio", type=float, default=DEFAULT_TRACE_SMALL_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument("--trace-medium-edge-max-cloud-lat-ratio", type=float, default=DEFAULT_TRACE_MEDIUM_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument("--trace-large-edge-max-cloud-lat-ratio", type=float, default=DEFAULT_TRACE_LARGE_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument("--trace-small-latency-rescue-min-cloud-pressure-ratio", type=float, default=DEFAULT_TRACE_SMALL_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument("--trace-medium-latency-rescue-min-cloud-pressure-ratio", type=float, default=DEFAULT_TRACE_MEDIUM_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument("--trace-large-latency-rescue-min-cloud-pressure-ratio", type=float, default=DEFAULT_TRACE_LARGE_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument("--disable-trace-edge-rescue", action="store_true")
    parser.add_argument("--cloud-throughput-mode", choices=["decode_only", "service_effective"], default=DEFAULT_CLOUD_THROUGHPUT_MODE)
    parser.add_argument("--optimize-for", choices=["balanced", "latency", "throughput", "throughput_constrained", "throughput_constrained_trace", "throughput_constrained_trace_distilled", "latency_constrained"], default="balanced")
    parser.add_argument("--reject-extrapolation", action="store_true")
    parser.add_argument("--no-prefer-edge", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def load_error_statistics(artifacts_dir: str) -> Dict[str, Any]:
    table_csv = os.path.join(artifacts_dir, "combined_perf_with_pred.csv")
    stats: Dict[str, Any] = {
        "global_mean_abs_err_s": 0.05,
        "global_p75_abs_err_s": 0.08,
        "by_device_config": {},
    }
    if not os.path.exists(table_csv):
        return stats

    with open(table_csv, 'r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        abs_err_col = None
        for candidate_col in ["p95_calibrated_abs_err_s", "p95_abs_err_s"]:
            if candidate_col in fieldnames:
                abs_err_col = candidate_col
                break
        if abs_err_col is None:
            return stats

        global_errors: List[float] = []
        by_device_config: Dict[Tuple[str, str], List[float]] = {}
        for row in reader:
            raw_error = row.get(abs_err_col)
            try:
                abs_error = float(raw_error)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(abs_error):
                continue
            global_errors.append(abs_error)
            device_id = str(row.get('device_id') or '').strip()
            config_id = str(row.get('config_id') or '').strip()
            if device_id and config_id:
                by_device_config.setdefault((device_id, config_id), []).append(abs_error)

    if global_errors:
        global_errors.sort()
        stats["global_mean_abs_err_s"] = float(sum(global_errors) / len(global_errors))
        p75_idx = min(int(0.75 * (len(global_errors) - 1)), len(global_errors) - 1)
        stats["global_p75_abs_err_s"] = float(global_errors[p75_idx])

    for key, values in by_device_config.items():
        values.sort()
        p75_idx = min(int(0.75 * (len(values) - 1)), len(values) - 1)
        stats["by_device_config"][key] = {
            "mean_abs_err_s": float(sum(values) / len(values)),
            "p75_abs_err_s": float(values[p75_idx]),
        }
    return stats


def load_assets(artifacts_dir: str, prediction_source: str = "model") -> Dict[str, Any]:
    manifest = load_json(os.path.join(artifacts_dir, "dataset_manifest.json"))
    calibrator_path = os.path.join(artifacts_dir, "p95_calibrator.joblib")
    calibration_cols_path = os.path.join(artifacts_dir, "p95_calibration_feature_cols.joblib")
    pairwise_ranker_path = os.path.join(artifacts_dir, "pairwise_ranker.joblib")
    pairwise_feature_cols_path = os.path.join(artifacts_dir, "pairwise_rank_feature_cols.joblib")
    assets = {
        "_artifacts_dir": artifacts_dir,
        "pairwise_ranker": joblib.load(pairwise_ranker_path) if os.path.exists(pairwise_ranker_path) else None,
        "pairwise_rank_feature_cols": joblib.load(pairwise_feature_cols_path) if os.path.exists(pairwise_feature_cols_path) else None,
        "manifest": manifest,
        "error_stats": load_error_statistics(artifacts_dir),
    }
    if str(prediction_source) == "table":
        return assets
    assets.update(
        {
            "feature_cols": joblib.load(os.path.join(artifacts_dir, "feature_cols.joblib")),
            "p95_model": joblib.load(os.path.join(artifacts_dir, "gbdt_p95.joblib")),
            "p95_calibrator": joblib.load(calibrator_path) if os.path.exists(calibrator_path) else None,
            "p95_calibration_feature_cols": joblib.load(calibration_cols_path) if os.path.exists(calibration_cols_path) else None,
            "tps_model": joblib.load(os.path.join(artifacts_dir, "gbdt_tps.joblib")),
            "known_values": known_values_from_manifest(manifest),
        }
    )
    return assets


def ensure_model_assets_loaded(assets: Dict[str, Any]) -> Dict[str, Any]:
    if "feature_cols" in assets and "p95_model" in assets and "tps_model" in assets and "known_values" in assets:
        return assets
    artifacts_dir = str(assets.get("_artifacts_dir", ""))
    if not artifacts_dir:
        raise ValueError("missing artifacts directory for lazy model asset loading")
    calibrator_path = os.path.join(artifacts_dir, "p95_calibrator.joblib")
    calibration_cols_path = os.path.join(artifacts_dir, "p95_calibration_feature_cols.joblib")
    manifest = assets.get("manifest") or load_json(os.path.join(artifacts_dir, "dataset_manifest.json"))
    assets["manifest"] = manifest
    assets["feature_cols"] = joblib.load(os.path.join(artifacts_dir, "feature_cols.joblib"))
    assets["p95_model"] = joblib.load(os.path.join(artifacts_dir, "gbdt_p95.joblib"))
    assets["p95_calibrator"] = joblib.load(calibrator_path) if os.path.exists(calibrator_path) else None
    assets["p95_calibration_feature_cols"] = joblib.load(calibration_cols_path) if os.path.exists(calibration_cols_path) else None
    assets["tps_model"] = joblib.load(os.path.join(artifacts_dir, "gbdt_tps.joblib"))
    assets["known_values"] = known_values_from_manifest(manifest)
    return assets


def option(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def default_rental_rate(device_id: str, target: str) -> float:
    if device_id in DEFAULT_DEVICE_RATES:
        return float(DEFAULT_DEVICE_RATES[device_id])
    return 4.0 if target == "cloud" else 1.0


def candidate_rental_rate(candidate: Dict[str, Any]) -> float:
    target = str(candidate.get("target", "edge"))
    device_id = str(candidate.get("device_id", "unknown"))
    return float(candidate.get("rental_rate", default_rental_rate(device_id, target)))


def safe_slo_limit(args: argparse.Namespace) -> float:
    slo = max(float(args.slo_p95_s), 1e-6)
    ratio_limit = slo * float(option(args, "slo_safety_margin", DEFAULT_SLO_SAFETY_MARGIN))
    absolute_buffer = min(0.08, 0.10 * slo)
    return max(ratio_limit, slo - absolute_buffer)


def soft_slo_limit(args: argparse.Namespace) -> float:
    slo = max(float(args.slo_p95_s), 1e-6)
    return slo * (1.0 + max(float(option(args, "soft_slo_tolerance_ratio", DEFAULT_SOFT_SLO_TOLERANCE_RATIO)), 0.0))


def lookup_candidate_risk_buffer_s(
    *,
    assets: Optional[Dict[str, Any]],
    device_id: str,
    config_id: Optional[str],
    target: str,
) -> float:
    if target == "cloud":
        return 0.0
    if assets is None:
        return 0.05
    stats = assets.get("error_stats", {})
    by_device_config = stats.get("by_device_config", {})
    key = (str(device_id), str(config_id or ""))
    if key in by_device_config:
        return float(by_device_config[key].get("p75_abs_err_s", by_device_config[key].get("mean_abs_err_s", 0.05)))
    return float(stats.get("global_p75_abs_err_s", stats.get("global_mean_abs_err_s", 0.05)))


def annotate_candidate_risk(
    candidate: Dict[str, Any],
    args: argparse.Namespace,
    *,
    risk_buffer_s: Optional[float] = None,
) -> Dict[str, Any]:
    slo = max(float(args.slo_p95_s), 1e-6)
    buffer_s = float(candidate.get("risk_buffer_s", 0.0) if risk_buffer_s is None else risk_buffer_s)
    risk_adjusted_p95_s = float(candidate.get("predicted_p95_s", 1e9)) + max(buffer_s, 0.0)
    candidate["risk_buffer_s"] = buffer_s
    candidate["risk_adjusted_p95_s"] = risk_adjusted_p95_s
    candidate["within_risk_slo"] = bool(risk_adjusted_p95_s <= slo)
    candidate["within_risk_safe_slo"] = bool(risk_adjusted_p95_s <= safe_slo_limit(args))
    candidate["risk_violation_s"] = max(risk_adjusted_p95_s - slo, 0.0)
    candidate["risk_violation_ratio"] = float(candidate["risk_violation_s"]) / slo
    return candidate


def calibrate_p95(
    *,
    raw_p95_s: float,
    assets: Dict[str, Any],
    device_id: str,
    quant_method: str,
    prompt_len: int,
    gen_len: int,
    concurrency: int,
    config_id: Optional[str],
    batch_size: Optional[int],
    max_tokens: Optional[int],
    kv_strategy: Optional[str],
    quant_bits: Optional[int],
) -> Tuple[float, bool]:
    calibrator = assets.get("p95_calibrator")
    calibration_feature_cols = assets.get("p95_calibration_feature_cols")
    if calibrator is None or calibration_feature_cols is None:
        return float(raw_p95_s), False

    calibration_row = build_p95_calibration_row(
        calibration_feature_cols=calibration_feature_cols,
        base_feature_cols=assets["feature_cols"],
        raw_p95_s=raw_p95_s,
        device_id=device_id,
        quant_method=quant_method,
        prompt_len=prompt_len,
        gen_len=gen_len,
        concurrency=concurrency,
        config_id=config_id,
        batch_size=batch_size,
        max_tokens=max_tokens,
        kv_strategy=kv_strategy,
        quant_bits=quant_bits,
    )
    residual_hat = float(calibrator.predict(calibration_row)[0])
    calibrated_p95_s = float(np.clip(np.expm1(np.log1p(max(float(raw_p95_s), 1e-9)) + residual_hat), 1e-9, None))
    return calibrated_p95_s, True


def static_candidate_fields(candidate: Dict[str, Any]) -> Dict[str, Any]:
    target = str(candidate["target"])
    return {
        "config_id": candidate["config_id"],
        "target": target,
        "device_id": str(candidate["device_id"]),
        "quant_method": str(candidate["quant_method"]),
        "quant_bits": candidate.get("quant_bits"),
        "batch_size": candidate.get("batch_size"),
        "max_tokens": candidate.get("max_tokens"),
        "kv_strategy": candidate.get("kv_strategy"),
        "edge_deployable": bool(candidate.get("edge_deployable", target == "edge")),
        "enabled": bool(candidate.get("enabled", True)),
        "perf_source": candidate.get("perf_source", "unknown"),
        "note": candidate.get("note", ""),
        "rental_rate": candidate_rental_rate(candidate),
    }


def runtime_constraints(candidate: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    request_total_tokens = args.prompt_len + args.gen_len
    max_tokens = candidate.get("max_tokens")
    batch_size = candidate.get("batch_size")
    estimated_memory_mb = candidate.get("estimated_memory_mb")

    within_memory = True
    memory_pressure = 0.0
    memory_headroom_mb = np.nan
    if args.edge_mem_available_mb is not None and estimated_memory_mb is not None and candidate["target"] == "edge":
        within_memory = float(estimated_memory_mb) <= float(args.edge_mem_available_mb)
        memory_pressure = float(estimated_memory_mb) / max(float(args.edge_mem_available_mb), 1e-6)
        memory_headroom_mb = float(args.edge_mem_available_mb) - float(estimated_memory_mb)

    within_token_budget = True
    if max_tokens is not None:
        within_token_budget = request_total_tokens <= int(max_tokens)

    within_batch_budget = True
    if batch_size is not None:
        within_batch_budget = int(args.concurrency) <= int(batch_size)

    return {
        "within_memory": within_memory,
        "within_token_budget": within_token_budget,
        "within_batch_budget": within_batch_budget,
        "memory_pressure": memory_pressure,
        "memory_headroom_mb": memory_headroom_mb,
    }


def predict_edge_candidate(candidate: Dict[str, Any], args: argparse.Namespace, assets: Dict[str, Any]) -> Dict[str, Any]:
    known_values = assets["known_values"]
    device_id = str(candidate["device_id"])
    quant_method = str(candidate["quant_method"])
    config_id = str(candidate["config_id"])
    kv_strategy = candidate.get("kv_strategy")
    batch_size = candidate.get("batch_size")
    max_tokens = candidate.get("max_tokens")
    quant_bits = candidate.get("quant_bits")

    feature_row = build_feature_row(
        feature_cols=assets["feature_cols"],
        device_id=device_id,
        quant_method=quant_method,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        concurrency=args.concurrency,
        config_id=config_id,
        batch_size=batch_size,
        max_tokens=max_tokens,
        kv_strategy=kv_strategy,
        quant_bits=quant_bits,
    )

    core_p95_raw_s = float(np.expm1(assets["p95_model"].predict(feature_row))[0])
    core_p95_s, calibration_applied = calibrate_p95(
        raw_p95_s=core_p95_raw_s,
        assets=assets,
        device_id=device_id,
        quant_method=quant_method,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        concurrency=args.concurrency,
        config_id=config_id,
        batch_size=batch_size,
        max_tokens=max_tokens,
        kv_strategy=kv_strategy,
        quant_bits=quant_bits,
    )
    core_tps = float(np.expm1(assets["tps_model"].predict(feature_row))[0])
    extra_latency_s = float(candidate.get("extra_latency_s", 0.0)) + float(args.edge_queue_delay_s)
    predicted_p95_raw_s = core_p95_raw_s + extra_latency_s
    predicted_p95_s = core_p95_s + extra_latency_s
    extrapolation = {
        "device_id_unseen": device_id not in known_values["device_ids"],
        "quant_method_unseen": quant_method not in known_values["quant_methods"],
        "config_id_unseen": config_id not in known_values["config_ids"],
        "kv_strategy_unseen": str(kv_strategy) not in known_values["kv_strategies"],
    }
    rejected_for_extrapolation = bool(args.reject_extrapolation and any(extrapolation.values()))

    result = static_candidate_fields(candidate)
    result.update(
        {
            "predicted_p95_s": predicted_p95_s,
            "predicted_p95_raw_s": predicted_p95_raw_s,
            "core_p95_s": core_p95_s,
            "core_p95_raw_s": core_p95_raw_s,
            "p95_calibration_applied": calibration_applied,
            "p95_calibration_delta_s": core_p95_s - core_p95_raw_s,
            "predicted_tok_per_s": core_tps,
            "estimated_memory_mb": candidate.get("memory_mb"),
            "within_slo": predicted_p95_s <= float(args.slo_p95_s),
            "within_safe_slo": predicted_p95_s <= safe_slo_limit(args),
            "safe_slo_limit_s": safe_slo_limit(args),
            "rejected_for_extrapolation": rejected_for_extrapolation,
            "extrapolation": extrapolation,
            "cost_bias": float(candidate.get("cost_bias", 0.0)),
            "extra_latency_s": extra_latency_s,
        }
    )
    result.update(runtime_constraints(result, args))
    annotate_candidate_risk(
        result,
        args,
        risk_buffer_s=lookup_candidate_risk_buffer_s(
            assets=assets,
            device_id=device_id,
            config_id=config_id,
            target=str(result["target"]),
        ),
    )
    return result


def predict_cloud_candidate(candidate: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    perf_model = candidate["perf_model"]
    if perf_model.get("kind") != "analytic":
        raise ValueError(f"unsupported cloud perf model kind: {perf_model.get('kind')}")

    concurrency_scale = 1.0 + float(perf_model.get("concurrency_penalty", 0.0)) * max(args.concurrency - 1, 0)
    adjusted_tok_per_s = float(perf_model["decode_tokens_per_s"]) / concurrency_scale
    prefill_latency_s = float(perf_model["prefill_per_token_s"]) * args.prompt_len
    decode_latency_s = args.gen_len / adjusted_tok_per_s
    prompt_upload_latency_s = float(perf_model.get("prompt_upload_per_token_s", 0.0)) * args.prompt_len
    response_download_latency_s = float(perf_model.get("response_download_per_token_s", 0.0)) * args.gen_len
    extra_latency_s = float(perf_model.get("base_latency_s", 0.0))
    extra_latency_s += float(perf_model.get("network_rtt_s", 0.0))
    extra_latency_s += prompt_upload_latency_s
    extra_latency_s += response_download_latency_s
    extra_latency_s += float(candidate.get("extra_latency_s", 0.0))
    extra_latency_s += float(args.cloud_extra_delay_s)
    predicted_p95_s = prefill_latency_s + decode_latency_s + extra_latency_s
    effective_tok_per_s = float(args.gen_len) / max(predicted_p95_s, 1e-9)
    throughput_mode = str(option(args, "cloud_throughput_mode", DEFAULT_CLOUD_THROUGHPUT_MODE))
    if throughput_mode == "service_effective":
        selected_tok_per_s = effective_tok_per_s
    else:
        selected_tok_per_s = adjusted_tok_per_s

    result = static_candidate_fields(candidate)
    result.update(
        {
            "predicted_p95_s": predicted_p95_s,
            "predicted_p95_raw_s": predicted_p95_s,
            "core_p95_s": predicted_p95_s,
            "core_p95_raw_s": predicted_p95_s,
            "p95_calibration_applied": False,
            "p95_calibration_delta_s": 0.0,
            "predicted_tok_per_s": selected_tok_per_s,
            "predicted_tok_per_s_raw": adjusted_tok_per_s,
            "predicted_tok_per_s_service": effective_tok_per_s,
            "estimated_memory_mb": candidate.get("memory_mb"),
            "within_slo": predicted_p95_s <= float(args.slo_p95_s),
            "within_safe_slo": predicted_p95_s <= safe_slo_limit(args),
            "safe_slo_limit_s": safe_slo_limit(args),
            "rejected_for_extrapolation": False,
            "extrapolation": {
                "device_id_unseen": False,
                "quant_method_unseen": False,
                "config_id_unseen": False,
                "kv_strategy_unseen": False,
            },
            "cost_bias": float(candidate.get("cost_bias", 0.0)),
            "extra_latency_s": extra_latency_s,
            "cloud_prompt_upload_latency_s": prompt_upload_latency_s,
            "cloud_response_download_latency_s": response_download_latency_s,
            "cloud_throughput_mode": throughput_mode,
        }
    )
    result.update(runtime_constraints(result, args))
    annotate_candidate_risk(result, args, risk_buffer_s=0.0)
    return result


def score_candidates(candidates: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if not candidates:
        return

    max_tps = max(float(candidate.get("predicted_tok_per_s", 0.0)) for candidate in candidates)
    max_tps = max(max_tps, 1e-6)
    min_compare_latency = min(float(candidate.get("predicted_p95_s", 1e9)) for candidate in candidates)
    min_compare_latency = max(min_compare_latency, 1e-6)
    mem_threshold = float(option(args, "memory_penalty_threshold", DEFAULT_MEMORY_PENALTY_THRESHOLD))
    slo = max(float(args.slo_p95_s), 1e-6)

    for candidate in candidates:
        if "risk_adjusted_p95_s" not in candidate:
            annotate_candidate_risk(candidate, args)

        predicted_p95_s = float(candidate["predicted_p95_s"])
        predicted_tps = float(candidate.get("predicted_tok_per_s", 0.0))
        risk_latency = float(candidate.get("risk_adjusted_p95_s", predicted_p95_s))
        compare_latency = predicted_p95_s
        throughput_norm = predicted_tps / max_tps
        latency_advantage = min_compare_latency / max(compare_latency, 1e-6)
        safe_slack_ratio = max(slo - compare_latency, 0.0) / slo
        memory_pressure = float(candidate.get("memory_pressure", 0.0) or 0.0)
        near_limit_memory_penalty = 0.0
        if candidate["target"] == "edge" and memory_pressure > mem_threshold:
            near_limit_memory_penalty = ((memory_pressure - mem_threshold) / max(1.0 - mem_threshold, 1e-6)) ** 2

        rental_rate = float(candidate.get("rental_rate", candidate_rental_rate(candidate)))
        predicted_rental_cost = rental_rate * predicted_p95_s
        violation_s = max(predicted_p95_s - slo, 0.0)
        violation_ratio = violation_s / slo
        risk_violation_s = max(risk_latency - slo, 0.0)
        risk_violation_ratio = risk_violation_s / slo
        edge_bonus = float(option(args, "edge_selection_bonus", DEFAULT_EDGE_SELECTION_BONUS)) if candidate["target"] == "edge" and not bool(option(args, "no_prefer_edge", False)) else 0.0

        performance_score = (
            0.50 * latency_advantage
            + 0.30 * throughput_norm
            + 0.15 * safe_slack_ratio
            + edge_bonus
            - 0.08 * near_limit_memory_penalty
            - 0.60 * risk_violation_ratio
        )

        candidate["score"] = performance_score
        candidate["performance_score"] = performance_score
        candidate["predicted_rental_cost"] = predicted_rental_cost
        candidate["violation_s"] = violation_s
        candidate["violation_ratio"] = violation_ratio
        candidate["risk_violation_s"] = risk_violation_s
        candidate["risk_violation_ratio"] = risk_violation_ratio
        candidate["safe_slack_s"] = max(slo - compare_latency, 0.0)
        candidate["safe_slack_ratio"] = safe_slack_ratio
        candidate["near_limit_memory_penalty"] = near_limit_memory_penalty
        candidate["score_components"] = {
            "performance_score": performance_score,
            "latency_advantage": latency_advantage,
            "throughput_norm": throughput_norm,
            "safe_slack_ratio": safe_slack_ratio,
            "near_limit_memory_penalty": near_limit_memory_penalty,
            "predicted_p95_s": compare_latency,
            "risk_adjusted_p95_s": risk_latency,
            "risk_violation_ratio": risk_violation_ratio,
            "predicted_rental_cost": predicted_rental_cost,
        }

def apply_pairwise_rank_scores(candidates: List[Dict[str, Any]], args: argparse.Namespace, assets: Dict[str, Any]) -> None:
    for candidate in candidates:
        candidate["pairwise_rank_score"] = 0.0
        candidate["pairwise_rank_applied"] = False
        candidate["pairwise_rank_wins"] = 0.0

    ranker = assets.get("pairwise_ranker")
    feature_cols = assets.get("pairwise_rank_feature_cols")
    edge_candidates = [candidate for candidate in candidates if str(candidate.get("target")) == "edge"]
    if ranker is None or not feature_cols or len(edge_candidates) < 2:
        return

    score_totals = [0.0 for _ in edge_candidates]
    comparisons = [0 for _ in edge_candidates]

    for left_idx in range(len(edge_candidates)):
        for right_idx in range(left_idx + 1, len(edge_candidates)):
            left = edge_candidates[left_idx]
            right = edge_candidates[right_idx]
            x_left = build_pairwise_feature_row(
                feature_cols=feature_cols,
                device_id=str(left.get("device_id", args.edge_device_id)),
                prompt_len=int(args.prompt_len),
                gen_len=int(args.gen_len),
                concurrency=int(args.concurrency),
                left_candidate=left,
                right_candidate=right,
            )
            x_right = build_pairwise_feature_row(
                feature_cols=feature_cols,
                device_id=str(right.get("device_id", args.edge_device_id)),
                prompt_len=int(args.prompt_len),
                gen_len=int(args.gen_len),
                concurrency=int(args.concurrency),
                left_candidate=right,
                right_candidate=left,
            )
            left_prob = float(ranker.predict_proba(x_left)[0][1])
            right_prob = float(ranker.predict_proba(x_right)[0][1])
            left_win_prob = 0.5 * (left_prob + (1.0 - right_prob))
            score_totals[left_idx] += left_win_prob
            score_totals[right_idx] += 1.0 - left_win_prob
            comparisons[left_idx] += 1
            comparisons[right_idx] += 1

    for idx, candidate in enumerate(edge_candidates):
        candidate["pairwise_rank_applied"] = True
        candidate["pairwise_rank_wins"] = score_totals[idx]
        if comparisons[idx] > 0:
            candidate["pairwise_rank_score"] = score_totals[idx] / comparisons[idx]


def reference_cloud_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    runtime_clouds = [candidate for candidate in candidates if str(candidate.get("target")) == "cloud"]
    if not runtime_clouds:
        return None
    return min(
        runtime_clouds,
        key=lambda item: (
            float(item.get("predicted_rental_cost", 1e9)),
            float(item.get("risk_adjusted_p95_s", item.get("predicted_p95_s", 1e9))),
            -float(item.get("predicted_tok_per_s", 0.0)),
        ),
    )


def _utility_edge_execution_key(item: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
    batch_size = int(item.get("batch_size") or 0)
    return (
        float(item.get("predicted_rental_cost", 1e9)),
        float(item.get("risk_adjusted_p95_s", item.get("predicted_p95_s", 1e9))),
        -float(item.get("pairwise_rank_score", 0.0)),
        -float(item.get("predicted_tok_per_s", 0.0)),
        -batch_size,
    )


def _utility_candidate_key(item: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, int]:
    batch_size = int(item.get("batch_size") or 0)
    return (
        -float(item.get("utility_score", -1e9)),
        float(item.get("risk_adjusted_p95_s", item.get("predicted_p95_s", 1e9))),
        float(item.get("predicted_rental_cost", 1e9)),
        -float(item.get("pairwise_rank_score", 0.0)),
        -float(item.get("predicted_tok_per_s", 0.0)),
        float(item.get("near_limit_memory_penalty", 0.0)),
        -batch_size,
    )


def annotate_relative_utilities(
    candidates: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    reference_cloud: Optional[Dict[str, Any]] = None,
    device_queue_penalties: Optional[Dict[str, float]] = None,
    active_config_by_device: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return reference_cloud

    for candidate in candidates:
        if "risk_adjusted_p95_s" not in candidate:
            annotate_candidate_risk(candidate, args)
        if "predicted_rental_cost" not in candidate:
            candidate["predicted_rental_cost"] = float(candidate_rental_rate(candidate)) * float(candidate.get("predicted_p95_s", 1e9))
        if "near_limit_memory_penalty" not in candidate:
            memory_pressure = float(candidate.get("memory_pressure", 0.0) or 0.0)
            mem_threshold = float(option(args, "memory_penalty_threshold", DEFAULT_MEMORY_PENALTY_THRESHOLD))
            near_limit_memory_penalty = 0.0
            if str(candidate.get("target")) == "edge" and memory_pressure > mem_threshold:
                near_limit_memory_penalty = ((memory_pressure - mem_threshold) / max(1.0 - mem_threshold, 1e-6)) ** 2
            candidate["near_limit_memory_penalty"] = near_limit_memory_penalty
        candidate["batch_load_ratio"] = candidate_batch_load_ratio(candidate, args)

    cloud_reference = reference_cloud or reference_cloud_candidate(candidates)
    if cloud_reference is None:
        for candidate in candidates:
            candidate["utility_score"] = -1e9 if str(candidate.get("target")) == "edge" else 0.0
            candidate["utility_positive"] = False
        return None

    slo = max(float(args.slo_p95_s), 1e-6)
    cloud_cost = float(cloud_reference.get("predicted_rental_cost", 1e9))
    cloud_risk_p95 = float(cloud_reference.get("risk_adjusted_p95_s", cloud_reference.get("predicted_p95_s", 1e9)))
    cloud_tps = max(float(cloud_reference.get("predicted_tok_per_s", 0.0)), 1e-6)
    cloud_violation_ratio = float(cloud_reference.get("risk_violation_ratio", 0.0))

    for candidate in candidates:
        if str(candidate.get("target")) == "cloud":
            candidate["utility_score"] = 0.0
            candidate["utility_positive"] = False
            candidate["utility_reason"] = "cloud_reference"
            continue

        device_id = str(candidate.get("device_id"))
        config_id = str(candidate.get("config_id"))
        edge_cost = float(candidate.get("predicted_rental_cost", 1e9))
        edge_risk_p95 = float(candidate.get("risk_adjusted_p95_s", candidate.get("predicted_p95_s", 1e9)))
        edge_tps = float(candidate.get("predicted_tok_per_s", 0.0))
        edge_violation_ratio = float(candidate.get("risk_violation_ratio", 0.0))

        cost_gain = cloud_cost - edge_cost
        slack_bonus = max(slo - edge_risk_p95, 0.0) / slo if cloud_violation_ratio > 0.0 else 0.0
        throughput_gain = (edge_tps - cloud_tps) / cloud_tps
        risk_gap = max(edge_violation_ratio - cloud_violation_ratio, 0.0)
        memory_penalty = float(candidate.get("near_limit_memory_penalty", 0.0))
        queue_penalty = float((device_queue_penalties or {}).get(device_id, 0.0))

        switch_penalty = 0.0
        active_config = (active_config_by_device or {}).get(device_id)
        if active_config and active_config != config_id:
            switch_penalty = float(option(args, "utility_switch_penalty", DEFAULT_UTILITY_SWITCH_PENALTY))

        utility = 0.0
        utility += float(option(args, "utility_cost_weight", DEFAULT_UTILITY_COST_WEIGHT)) * cost_gain
        utility += float(option(args, "utility_latency_weight", DEFAULT_UTILITY_LATENCY_WEIGHT)) * slack_bonus
        utility += float(option(args, "utility_throughput_weight", DEFAULT_UTILITY_THROUGHPUT_WEIGHT)) * throughput_gain
        utility -= float(option(args, "utility_risk_weight", DEFAULT_UTILITY_RISK_WEIGHT)) * risk_gap
        utility -= float(option(args, "utility_memory_weight", DEFAULT_UTILITY_MEMORY_WEIGHT)) * memory_penalty
        utility -= float(option(args, "utility_queue_weight", DEFAULT_UTILITY_QUEUE_WEIGHT)) * queue_penalty
        utility -= switch_penalty

        admissible_latency = edge_risk_p95 <= soft_slo_limit(args) or edge_risk_p95 <= cloud_risk_p95 + slo * float(option(args, "violation_tolerance_ratio", DEFAULT_VIOLATION_TOLERANCE_RATIO))
        utility_positive = bool(utility > float(option(args, "utility_min_gain", DEFAULT_UTILITY_MIN_GAIN)) and admissible_latency)

        candidate["utility_score"] = utility
        candidate["utility_positive"] = utility_positive
        candidate["utility_reason"] = "positive_edge_gain" if utility_positive else "non_positive_edge_gain"
        candidate["utility_components"] = {
            "cost_gain": cost_gain,
            "slack_bonus": slack_bonus,
            "throughput_gain": throughput_gain,
            "risk_gap": risk_gap,
            "memory_penalty": memory_penalty,
            "queue_penalty": queue_penalty,
            "switch_penalty": switch_penalty,
        }

    return cloud_reference


def _tie_break_key(item: Dict[str, Any]) -> Tuple[float, float, float, float, float, int]:
    batch_size = int(item.get("batch_size") or 0)
    return (
        -float(item.get("pairwise_rank_score", 0.0)),
        float(item.get("predicted_p95_s", 1e9)),
        float(item.get("near_limit_memory_penalty", 0.0)),
        -float(item.get("predicted_tok_per_s", 0.0)),
        -float(item.get("safe_slack_ratio", -1e9)),
        -batch_size,
    )


def _apply_cost_shortlist(pool: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not pool:
        return []
    best_cost = min(float(item["predicted_rental_cost"]) for item in pool)
    tolerance_ratio = float(option(args, "tie_cost_tolerance_ratio", DEFAULT_TIE_COST_TOLERANCE_RATIO))
    cutoff = best_cost * (1.0 + max(tolerance_ratio, 0.0)) + 1e-9
    return [item for item in pool if float(item["predicted_rental_cost"]) <= cutoff]


def _prefer_edge_if_possible(pool: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], bool]:
    if bool(option(args, "no_prefer_edge", False)):
        return pool, False
    edge_pool = [item for item in pool if str(item.get("target")) == "edge"]
    if edge_pool:
        return edge_pool, True
    return pool, False


def _candidate_risk_latency(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("risk_adjusted_p95_s", candidate.get("predicted_p95_s", 1e9)))


def _candidate_compare_latency(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("predicted_p95_s", 1e9))


def _latency_shortlist(
    pool: List[Dict[str, Any]],
    *,
    shortlist_ratio: float,
) -> List[Dict[str, Any]]:
    if not pool:
        return []
    best_latency = min(_candidate_compare_latency(item) for item in pool)
    cutoff = best_latency * (1.0 + max(shortlist_ratio, 0.0)) + 1e-9
    shortlist = [item for item in pool if _candidate_compare_latency(item) <= cutoff]
    if shortlist:
        return shortlist
    return [min(pool, key=lambda item: (_candidate_compare_latency(item), -float(item.get("predicted_tok_per_s", 0.0))))]


def _resolve_throughput_floor_tps(pool: List[Dict[str, Any]], args: argparse.Namespace) -> float:
    explicit_floor = option(args, "min_throughput_tps", None)
    if explicit_floor is not None:
        return max(float(explicit_floor), 0.0)
    ratio = max(float(option(args, "throughput_floor_ratio", DEFAULT_THROUGHPUT_FLOOR_RATIO)), 0.0)
    if not pool:
        return 0.0
    best_tps = max(float(candidate.get("predicted_tok_per_s", 0.0)) for candidate in pool)
    return max(best_tps * ratio, 0.0)


def _throughput_floor_pool(pool: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], float]:
    if not pool:
        return [], 0.0
    floor_tps = _resolve_throughput_floor_tps(pool, args)
    constrained = [candidate for candidate in pool if float(candidate.get("predicted_tok_per_s", 0.0)) >= floor_tps - 1e-9]
    return constrained, floor_tps


def _throughput_shortlist(pool: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not pool:
        return []
    best_tps = max(float(candidate.get("predicted_tok_per_s", 0.0)) for candidate in pool)
    tie_ratio = max(float(option(args, "throughput_tie_ratio", DEFAULT_THROUGHPUT_TIE_RATIO)), 0.0)
    cutoff = best_tps * (1.0 - tie_ratio) - 1e-9
    shortlist = [candidate for candidate in pool if float(candidate.get("predicted_tok_per_s", 0.0)) >= cutoff]
    if shortlist:
        return shortlist
    return [max(pool, key=lambda item: float(item.get("predicted_tok_per_s", 0.0)))]


def _choose_throughput_candidate(pool: List[Dict[str, Any]], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    shortlist = _throughput_shortlist(pool, args)
    if not shortlist:
        return None
    return min(shortlist, key=lambda item: _tie_break_key(item))


def _latency_first_key(candidate: Dict[str, Any], args: argparse.Namespace) -> Tuple[float, int, float, float, float]:
    return (
        _candidate_risk_latency(candidate),
        0 if str(candidate.get("target")) == "edge" and not bool(option(args, "no_prefer_edge", False)) else 1,
        -float(candidate.get("predicted_tok_per_s", 0.0)),
        float(candidate.get("near_limit_memory_penalty", 0.0)),
        -float(candidate.get("pairwise_rank_score", 0.0)),
    )


def _balanced_selection_key(candidate: Dict[str, Any], args: argparse.Namespace) -> Tuple[float, int, float, float, float, float]:
    edge_bonus = float(option(args, "edge_selection_bonus", DEFAULT_EDGE_SELECTION_BONUS)) if str(candidate.get("target")) == "edge" and not bool(option(args, "no_prefer_edge", False)) else 0.0
    return (
        float(candidate.get("performance_score", 0.0)),
        1 if edge_bonus > 0.0 else 0,
        float(candidate.get("safe_slack_ratio", 0.0)),
        float(candidate.get("pairwise_rank_score", 0.0)),
        -_candidate_compare_latency(candidate),
        float(candidate.get("predicted_tok_per_s", 0.0)),
    )


def _throughput_selection_key(candidate: Dict[str, Any], args: argparse.Namespace) -> Tuple[float, float, float, float]:
    return (
        float(candidate.get("predicted_tok_per_s", 0.0)),
        -_candidate_risk_latency(candidate),
        float(candidate.get("pairwise_rank_score", 0.0)),
        float(candidate.get("performance_score", 0.0)),
    )


def _selection_tiers(candidates: List[Dict[str, Any]], args: argparse.Namespace) -> List[Tuple[str, List[Dict[str, Any]]]]:
    soft_limit = soft_slo_limit(args)
    return [
        ("risk_safe", [item for item in candidates if bool(item.get("within_risk_safe_slo", False))]),
        ("risk_ontime", [item for item in candidates if bool(item.get("within_risk_slo", False))]),
        ("soft_slo", [item for item in candidates if _candidate_risk_latency(item) <= soft_limit]),
        ("runtime_feasible", list(candidates)),
    ]


def _preferred_edge_tier_pool(candidates: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    edge_candidates = [item for item in candidates if str(item.get("target")) == "edge"]
    if not edge_candidates:
        return None, []

    for tier_name, pool in _selection_tiers(edge_candidates, args):
        if tier_name == "runtime_feasible":
            break
        if pool:
            return tier_name, pool
    return None, []


def _trace_request_bucket(args: argparse.Namespace) -> str:
    prompt_len = int(getattr(args, "prompt_len", 0))
    gen_len = int(getattr(args, "gen_len", 0))
    if (
        prompt_len <= int(option(args, "trace_small_max_prompt_len", DEFAULT_TRACE_SMALL_MAX_PROMPT_LEN))
        and gen_len <= int(option(args, "trace_small_max_gen_len", DEFAULT_TRACE_SMALL_MAX_GEN_LEN))
    ):
        return "small"
    if (
        prompt_len <= int(option(args, "trace_medium_max_prompt_len", DEFAULT_TRACE_MEDIUM_MAX_PROMPT_LEN))
        and gen_len <= int(option(args, "trace_medium_max_gen_len", DEFAULT_TRACE_MEDIUM_MAX_GEN_LEN))
    ):
        return "medium"
    return "large"


def _trace_bucket_thresholds(args: argparse.Namespace) -> Tuple[str, float, float, float, float]:
    bucket = _trace_request_bucket(args)
    if bucket == "small":
        return (
            bucket,
            float(option(args, "trace_small_edge_min_tps_ratio", DEFAULT_TRACE_SMALL_EDGE_MIN_TPS_RATIO)),
            float(option(args, "trace_small_edge_max_slo_ratio", DEFAULT_TRACE_SMALL_EDGE_MAX_SLO_RATIO)),
            float(option(args, "trace_small_edge_max_cloud_lat_ratio", DEFAULT_TRACE_SMALL_EDGE_MAX_CLOUD_LAT_RATIO)),
            float(option(args, "trace_small_latency_rescue_min_cloud_pressure_ratio", DEFAULT_TRACE_SMALL_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)),
        )
    if bucket == "medium":
        return (
            bucket,
            float(option(args, "trace_medium_edge_min_tps_ratio", DEFAULT_TRACE_MEDIUM_EDGE_MIN_TPS_RATIO)),
            float(option(args, "trace_medium_edge_max_slo_ratio", DEFAULT_TRACE_MEDIUM_EDGE_MAX_SLO_RATIO)),
            float(option(args, "trace_medium_edge_max_cloud_lat_ratio", DEFAULT_TRACE_MEDIUM_EDGE_MAX_CLOUD_LAT_RATIO)),
            float(option(args, "trace_medium_latency_rescue_min_cloud_pressure_ratio", DEFAULT_TRACE_MEDIUM_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)),
        )
    return (
        bucket,
        float(option(args, "trace_large_edge_min_tps_ratio", DEFAULT_TRACE_LARGE_EDGE_MIN_TPS_RATIO)),
        float(option(args, "trace_large_edge_max_slo_ratio", DEFAULT_TRACE_LARGE_EDGE_MAX_SLO_RATIO)),
        float(option(args, "trace_large_edge_max_cloud_lat_ratio", DEFAULT_TRACE_LARGE_EDGE_MAX_CLOUD_LAT_RATIO)),
        float(option(args, "trace_large_latency_rescue_min_cloud_pressure_ratio", DEFAULT_TRACE_LARGE_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)),
    )


def _best_cloud_candidate(pool: List[Dict[str, Any]], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    cloud_candidates = [item for item in pool if str(item.get("target")) == "cloud"]
    if not cloud_candidates:
        return None
    return max(cloud_candidates, key=lambda item: _throughput_selection_key(item, args))


def _trace_filter_edge_shortlist(
    edge_shortlist: List[Dict[str, Any]],
    cloud_reference: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if not edge_shortlist:
        return []

    bucket, required_tps_ratio, allowed_slo_ratio, allowed_cloud_lat_ratio, latency_rescue_min_cloud_pressure_ratio = _trace_bucket_thresholds(args)
    required_tps_ratio = max(required_tps_ratio, 0.0)
    allowed_slo_ratio = max(allowed_slo_ratio, 0.0)
    allowed_cloud_lat_ratio = max(allowed_cloud_lat_ratio, 0.0)
    latency_rescue_min_cloud_pressure_ratio = max(latency_rescue_min_cloud_pressure_ratio, 0.0)
    slo_limit = max(float(args.slo_p95_s), 1e-6) * allowed_slo_ratio
    cloud_tps = None
    cloud_risk_latency = None
    cloud_pressure_ratio = None
    if cloud_reference is not None:
        cloud_tps = max(float(cloud_reference.get("predicted_tok_per_s", 0.0)), 1e-6)
        cloud_risk_latency = _candidate_risk_latency(cloud_reference)
        cloud_pressure_ratio = cloud_risk_latency / max(float(args.slo_p95_s), 1e-6)

    filtered: List[Dict[str, Any]] = []
    for candidate in edge_shortlist:
        edge_risk_latency = _candidate_risk_latency(candidate)
        if edge_risk_latency > slo_limit + 1e-9:
            continue
        edge_tps = float(candidate.get("predicted_tok_per_s", 0.0))
        throughput_ok = True
        latency_ok = True
        if cloud_tps is not None:
            throughput_ok = edge_tps + 1e-9 >= cloud_tps * required_tps_ratio
        if cloud_risk_latency is not None:
            latency_ok = edge_risk_latency <= cloud_risk_latency * allowed_cloud_lat_ratio + 1e-9
        latency_rescue_ok = latency_ok
        if (
            latency_ok
            and not throughput_ok
            and cloud_pressure_ratio is not None
            and cloud_pressure_ratio + 1e-9 < latency_rescue_min_cloud_pressure_ratio
        ):
            latency_rescue_ok = False
        if not throughput_ok and not latency_rescue_ok:
            continue
        filtered.append(candidate)
    return filtered


def _trace_pick_best_edge_candidate(
    edge_shortlist: List[Dict[str, Any]],
    cloud_reference: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    filtered = _trace_filter_edge_shortlist(edge_shortlist, cloud_reference, args)
    if not filtered:
        return None
    return _choose_throughput_candidate(filtered, args)


def _ordered_considered_pool(candidates: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            not bool(item.get("within_risk_safe_slo", False)),
            not bool(item.get("within_risk_slo", False)),
            _candidate_risk_latency(item),
            -float(item.get("predicted_tok_per_s", 0.0)),
            float(item.get("near_limit_memory_penalty", 0.0)),
        ),
    )

def candidate_batch_load_units(candidate: Dict[str, Any], args: argparse.Namespace) -> float:
    if str(candidate.get("target")) == "cloud":
        return 0.0
    return 1.0


def candidate_batch_load_ratio(candidate: Dict[str, Any], args: argparse.Namespace) -> float:
    if str(candidate.get("target")) == "cloud":
        return 0.0
    batch_size = int(candidate.get("batch_size") or max(int(args.concurrency), 1))
    return 1.0 / max(batch_size, 1)


def candidate_service_satisfaction_ratio(candidate: Dict[str, Any], args: argparse.Namespace) -> float:
    slo = max(float(args.slo_p95_s), 1e-6)
    latency_ratio = float(candidate.get("predicted_p95_s", 1e9)) / slo
    memory_pressure = float(candidate.get("memory_pressure", 0.0) or 0.0)
    batch_ratio = candidate_batch_load_ratio(candidate, args)
    return max(latency_ratio, batch_ratio, memory_pressure)


def _annotate_aqs_candidate(candidate: Dict[str, Any], *, aqs_tier: int, args: argparse.Namespace) -> None:
    candidate["aqs_tier"] = int(aqs_tier)
    candidate["batch_load_units"] = candidate_batch_load_units(candidate, args)
    candidate["batch_load_ratio"] = candidate_batch_load_ratio(candidate, args)
    candidate["service_satisfaction_ratio"] = candidate_service_satisfaction_ratio(candidate, args)


def _task_preference_key(candidate: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, int]:
    batch_size = int(candidate.get("batch_size") or 0)
    return (
        float(candidate.get("aqs_tier", 99)),
        float(candidate.get("predicted_rental_cost", 1e9)),
        float(candidate.get("predicted_p95_s", 1e9)),
        float(candidate.get("service_satisfaction_ratio", 1e9)),
        -float(candidate.get("pairwise_rank_score", 0.0)),
        -float(candidate.get("predicted_tok_per_s", 0.0)),
        -batch_size,
    )


def build_feasible_candidate_set(candidates: List[Dict[str, Any]], args: argparse.Namespace, *, shortlist_edges: bool = True) -> Dict[str, Any]:
    active = [candidate for candidate in candidates if candidate["enabled"]]
    active = [candidate for candidate in active if not candidate["rejected_for_extrapolation"]]
    capacity_ok = [
        candidate
        for candidate in active
        if candidate["within_memory"] and candidate["within_token_budget"] and candidate["within_batch_budget"]
    ]

    result: Dict[str, Any] = {
        "active": active,
        "capacity_ok": capacity_ok,
        "selected_pool": [],
        "edge_pool": [],
        "cloud_pool": [],
        "reason": "no_enabled_candidate" if not active else "no_runtime_feasible_candidate",
    }
    if not capacity_ok:
        return result

    edge_capacity_ok = [candidate for candidate in capacity_ok if str(candidate.get("target")) == "edge"]
    cloud_capacity_ok = [candidate for candidate in capacity_ok if str(candidate.get("target")) == "cloud"]
    safe_edges = [candidate for candidate in edge_capacity_ok if bool(candidate.get("within_risk_safe_slo", candidate.get("within_safe_slo", False)))]
    ontime_edges = [candidate for candidate in edge_capacity_ok if bool(candidate.get("within_risk_slo", candidate.get("within_slo", False)))]
    soft_edges = [
        candidate
        for candidate in edge_capacity_ok
        if float(candidate.get("risk_adjusted_p95_s", candidate.get("predicted_p95_s", 1e9))) <= soft_slo_limit(args)
    ]

    edge_pool: List[Dict[str, Any]] = []
    reason = "aqs_cloud_fallback_only"
    edge_tier = 0
    if safe_edges:
        edge_pool = _apply_cost_shortlist(safe_edges, args) if shortlist_edges else list(safe_edges)
        reason = "aqs_safe_edge_candidates"
        edge_tier = 0
    elif ontime_edges:
        edge_pool = _apply_cost_shortlist(ontime_edges, args) if shortlist_edges else list(ontime_edges)
        reason = "aqs_ontime_edge_candidates"
        edge_tier = 1
    elif soft_edges:
        edge_pool = _apply_cost_shortlist(soft_edges, args) if shortlist_edges else list(soft_edges)
        reason = "aqs_soft_edge_candidates"
        edge_tier = 2

    selected_pool: List[Dict[str, Any]] = []
    if edge_pool:
        for candidate in edge_pool:
            _annotate_aqs_candidate(candidate, aqs_tier=edge_tier, args=args)
        for candidate in cloud_capacity_ok:
            _annotate_aqs_candidate(candidate, aqs_tier=edge_tier + 1, args=args)
        selected_pool = sorted(edge_pool + cloud_capacity_ok, key=_task_preference_key)
    elif cloud_capacity_ok:
        for candidate in cloud_capacity_ok:
            _annotate_aqs_candidate(candidate, aqs_tier=0, args=args)
        selected_pool = sorted(cloud_capacity_ok, key=_task_preference_key)
    elif edge_capacity_ok:
        min_violation = min(float(candidate.get("risk_violation_ratio", candidate.get("violation_ratio", 1e9))) for candidate in edge_capacity_ok)
        violation_tol = max(float(option(args, "violation_tolerance_ratio", DEFAULT_VIOLATION_TOLERANCE_RATIO)), 0.0)
        near_best_violation = [
            candidate
            for candidate in edge_capacity_ok
            if float(candidate.get("risk_violation_ratio", candidate.get("violation_ratio", 1e9))) <= min_violation + violation_tol + 1e-9
        ]
        edge_pool = _apply_cost_shortlist(near_best_violation, args) if shortlist_edges else list(near_best_violation)
        for candidate in edge_pool:
            _annotate_aqs_candidate(candidate, aqs_tier=3, args=args)
        selected_pool = sorted(edge_pool, key=_task_preference_key)
        reason = "aqs_best_effort_edge_only"

    result.update(
        {
            "selected_pool": selected_pool,
            "edge_pool": edge_pool,
            "cloud_pool": cloud_capacity_ok,
            "reason": reason,
        }
    )
    return result


def _edge_server_key(candidate: Dict[str, Any]) -> str:
    return f"edge::{candidate['device_id']}"


def _cloud_server_key(candidate: Dict[str, Any]) -> str:
    return f"cloud::{candidate['device_id']}::{candidate['config_id']}"


def _device_option_preference_key(device_option: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, int]:
    return _task_preference_key(device_option["preferred_candidate"])


def build_device_feasible_set(candidates: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    feasible = build_feasible_candidate_set(candidates, args, shortlist_edges=False)
    runtime_pool = list(feasible["capacity_ok"])
    cloud_reference = annotate_relative_utilities(runtime_pool, args)

    edge_positive = [
        candidate
        for candidate in runtime_pool
        if str(candidate.get("target")) == "edge" and bool(candidate.get("utility_positive", False))
    ]
    edge_by_device: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in edge_positive:
        edge_by_device.setdefault(str(candidate["device_id"]), []).append(candidate)

    device_options: List[Dict[str, Any]] = []
    for device_id, device_candidates in edge_by_device.items():
        ordered_candidates = sorted(device_candidates, key=_utility_edge_execution_key)
        preferred_candidate = max(device_candidates, key=lambda item: float(item.get("utility_score", -1e9)))
        device_options.append(
            {
                "target": "edge_device",
                "server_key": f"edge::{device_id}",
                "device_id": device_id,
                "preferred_candidate": preferred_candidate,
                "feasible_candidates": ordered_candidates,
            }
        )

    feasible["positive_edge_pool"] = edge_positive
    feasible["cloud_reference"] = cloud_reference
    feasible["device_options"] = sorted(device_options, key=lambda item: _utility_candidate_key(item["preferred_candidate"]))
    feasible["reason"] = "utility_positive_edge_candidates" if edge_positive else "utility_cloud_default"
    return feasible


def _server_preference_key(candidate: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    return (
        -float(candidate.get("utility_score", -1e9)),
        float(candidate.get("risk_adjusted_p95_s", candidate.get("predicted_p95_s", 1e9))),
        float(candidate.get("predicted_rental_cost", 1e9)),
        float(candidate.get("batch_load_ratio", 1e9)),
        -float(candidate.get("pairwise_rank_score", 0.0)),
        -float(candidate.get("predicted_tok_per_s", 0.0)),
    )


def _accepted_proposals_for_config(proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not proposals:
        return []
    candidate = proposals[0]["candidate"]
    capacity_units = float(max(int(candidate.get("batch_size") or 1), 1))
    accepted: List[Dict[str, Any]] = []
    used_units = 0.0
    for proposal in sorted(proposals, key=lambda item: _server_preference_key(item["candidate"])):
        load_units = float(proposal.get("load_units", 1.0))
        if used_units + load_units <= capacity_units + 1e-9:
            accepted.append(proposal)
            used_units += load_units
    return accepted


def _assign_requests_to_active_configs(
    proposals: List[Dict[str, Any]],
    active_config_ids: List[str],
) -> List[Dict[str, Any]]:
    state_by_request: Dict[str, Dict[str, Any]] = {}
    held_by_config: Dict[str, List[Dict[str, Any]]] = {}

    for proposal in proposals:
        request_id = str(proposal["request_id"])
        candidate_list = [
            candidate
            for candidate in proposal["device_option"]["feasible_candidates"]
            if str(candidate.get("config_id")) in active_config_ids
        ]
        if candidate_list:
            state_by_request[request_id] = {
                "request_id": request_id,
                "load_units": float(proposal.get("load_units", 1.0)),
                "candidate_list": candidate_list,
                "next_idx": 0,
                "held_config": None,
                "accepted_proposal": None,
            }

    unmatched = set(state_by_request.keys())
    while unmatched:
        proposals_by_config: Dict[str, List[Dict[str, Any]]] = {}
        progress = False

        for request_id in list(unmatched):
            state = state_by_request[request_id]
            if state["held_config"] is not None:
                continue
            if state["next_idx"] >= len(state["candidate_list"]):
                unmatched.discard(request_id)
                continue
            candidate = state["candidate_list"][state["next_idx"]]
            state["next_idx"] += 1
            progress = True
            proposals_by_config.setdefault(str(candidate["config_id"]), []).append(
                {
                    "request_id": request_id,
                    "candidate": candidate,
                    "load_units": state["load_units"],
                }
            )

        if not progress:
            break

        next_unmatched = set()
        for config_id, incoming in proposals_by_config.items():
            combined = held_by_config.get(config_id, []) + incoming
            accepted = _accepted_proposals_for_config(combined)
            accepted_ids = {proposal["request_id"] for proposal in accepted}
            held_by_config[config_id] = accepted

            for proposal in combined:
                request_id = proposal["request_id"]
                state = state_by_request[request_id]
                if request_id in accepted_ids:
                    state["held_config"] = config_id
                    state["accepted_proposal"] = proposal
                else:
                    if state.get("held_config") == config_id:
                        state["held_config"] = None
                        state["accepted_proposal"] = None
                    if state["next_idx"] < len(state["candidate_list"]):
                        next_unmatched.add(request_id)

        unmatched = next_unmatched

    accepted: List[Dict[str, Any]] = []
    for state in state_by_request.values():
        accepted_proposal = state.get("accepted_proposal")
        if accepted_proposal is not None:
            accepted.append(accepted_proposal)
    return accepted


def _edge_subset_objective(
    accepted: List[Dict[str, Any]],
    active_config_count: int,
    args: argparse.Namespace,
) -> Tuple[float, float, float, float, float]:
    if not accepted:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    total_utility = sum(float(item["candidate"].get("utility_score", 0.0)) for item in accepted)
    activation_penalty = float(option(args, "utility_switch_penalty", DEFAULT_UTILITY_SWITCH_PENALTY)) * max(active_config_count - 1, 0)
    total_cost = sum(float(item["candidate"].get("predicted_rental_cost", 1e9)) for item in accepted)
    total_risk_latency = sum(float(item["candidate"].get("risk_adjusted_p95_s", item["candidate"].get("predicted_p95_s", 1e9))) for item in accepted)
    total_rank_bonus = sum(float(item["candidate"].get("pairwise_rank_score", 0.0)) for item in accepted)
    return (
        -(total_utility - activation_penalty),
        -float(len(accepted)),
        total_cost,
        total_risk_latency - 0.01 * total_rank_bonus,
        float(active_config_count),
    )


def _select_edge_device_proposals(proposals: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not proposals:
        return []

    memory_budget = proposals[0].get("edge_mem_available_mb")
    memory_budget_mb = float(memory_budget) if memory_budget is not None else float("inf")

    config_memory_mb: Dict[str, float] = {}
    for proposal in proposals:
        for candidate in proposal["device_option"]["feasible_candidates"]:
            config_id = str(candidate.get("config_id"))
            if config_id in config_memory_mb:
                continue
            memory_value = candidate.get("estimated_memory_mb")
            if memory_value is None or not np.isfinite(float(memory_value)):
                config_memory_mb[config_id] = float("inf")
            else:
                config_memory_mb[config_id] = float(memory_value)

    config_ids = list(config_memory_mb.keys())
    best_subset: List[Dict[str, Any]] = []
    best_objective: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    for mask in range(1, 1 << len(config_ids)):
        active_configs = [config_ids[idx] for idx in range(len(config_ids)) if mask & (1 << idx)]
        total_memory = sum(config_memory_mb[config_id] for config_id in active_configs)
        if total_memory > memory_budget_mb + 1e-9:
            continue
        accepted_subset = _assign_requests_to_active_configs(proposals, active_configs)
        objective = _edge_subset_objective(accepted_subset, len(active_configs), args)
        if objective < best_objective:
            best_objective = objective
            best_subset = accepted_subset

    return best_subset


def schedule_batch_requests(batch_requests: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    states: Dict[str, Dict[str, Any]] = {}
    held_by_server: Dict[str, List[Dict[str, Any]]] = {}

    for request in batch_requests:
        request_id = str(request["request_id"])
        args = request["args"]
        candidates = request["predicted_candidates"]
        feasible = build_device_feasible_set(candidates, args)
        preference_list = list(feasible.get("device_options", []))
        cloud_fallback = feasible.get("cloud_reference")
        states[request_id] = {
            "request_id": request_id,
            "args": args,
            "preference_list": preference_list,
            "next_idx": 0,
            "assignment": None,
            "cloud_fallback": cloud_fallback,
            "held_server": None,
            "aqs_reason": feasible["reason"],
            "decision_reason": feasible["reason"],
        }

    unmatched = {request_id for request_id, state in states.items() if state["preference_list"]}

    while unmatched:
        proposals_by_server: Dict[str, List[Dict[str, Any]]] = {}
        progress = False

        for request_id in list(unmatched):
            state = states[request_id]
            if state["held_server"] is not None:
                continue
            if state["next_idx"] >= len(state["preference_list"]):
                unmatched.discard(request_id)
                continue

            device_option = state["preference_list"][state["next_idx"]]
            state["next_idx"] += 1
            progress = True

            server_key = str(device_option["server_key"])
            proposals_by_server.setdefault(server_key, []).append(
                {
                    "request_id": request_id,
                    "device_option": device_option,
                    "load_units": 1.0,
                    "edge_mem_available_mb": state["args"].edge_mem_available_mb,
                    "args": state["args"],
                }
            )

        if not progress:
            break

        next_unmatched = set()
        for server_key, proposals in proposals_by_server.items():
            args = proposals[0]["args"]
            combined = held_by_server.get(server_key, []) + proposals
            accepted = _select_edge_device_proposals(combined, args)
            accepted_ids = {proposal["request_id"] for proposal in accepted}
            held_by_server[server_key] = accepted

            for proposal in combined:
                request_id = proposal["request_id"]
                state = states[request_id]
                if request_id in accepted_ids:
                    accepted_proposal = next(item for item in accepted if item["request_id"] == request_id)
                    state["assignment"] = accepted_proposal["candidate"]
                    state["held_server"] = server_key
                    state["decision_reason"] = f"batch_{state['aqs_reason']}_utility_edge_admit"
                else:
                    if state.get("held_server") == server_key:
                        state["held_server"] = None
                        state["assignment"] = None
                    if state["next_idx"] < len(state["preference_list"]):
                        next_unmatched.add(request_id)

        unmatched = next_unmatched

    for state in states.values():
        if state["assignment"] is None:
            if state.get("cloud_fallback") is not None:
                state["assignment"] = state["cloud_fallback"]
                state["decision_reason"] = f"batch_{state['aqs_reason']}_cloud_fallback"
            else:
                state["decision_reason"] = f"batch_{state['aqs_reason']}_unassigned"

    return states


def choose_candidate(candidates: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Optional[Dict[str, Any]], str, List[Dict[str, Any]]]:
    active = [candidate for candidate in candidates if bool(candidate.get("enabled", True))]
    active = [candidate for candidate in active if not bool(candidate.get("rejected_for_extrapolation", False))]
    if not active:
        return None, "no_enabled_candidate", candidates

    capacity_ok = [
        candidate
        for candidate in active
        if bool(candidate.get("within_memory", False))
        and bool(candidate.get("within_token_budget", False))
        and bool(candidate.get("within_batch_budget", False))
    ]
    if not capacity_ok:
        return None, "no_runtime_feasible_candidate", active

    tier_name = "runtime_feasible"
    selected_pool: List[Dict[str, Any]] = []
    for current_tier_name, pool in _selection_tiers(capacity_ok, args):
        if pool:
            tier_name = current_tier_name
            selected_pool = pool
            break

    considered = _ordered_considered_pool(selected_pool, args)
    best_latency = min(_candidate_risk_latency(item) for item in selected_pool)
    slo = max(float(args.slo_p95_s), 1e-6)
    latency_pressure = best_latency / slo
    latency_critical = tier_name in {"soft_slo", "runtime_feasible"} or latency_pressure >= float(option(args, "latency_critical_ratio", DEFAULT_LATENCY_CRITICAL_RATIO))
    shortlist_ratio = float(option(args, "latency_critical_shortlist_ratio", DEFAULT_LATENCY_CRITICAL_SHORTLIST_RATIO)) if latency_critical else float(option(args, "latency_shortlist_ratio", DEFAULT_LATENCY_SHORTLIST_RATIO))
    shortlist = _latency_shortlist(selected_pool, shortlist_ratio=shortlist_ratio)

    optimize_for = str(option(args, "optimize_for", "balanced"))
    if optimize_for == "latency_constrained":
        constrained_pool, floor_tps = _throughput_floor_pool(selected_pool, args)
        if constrained_pool:
            chosen = min(constrained_pool, key=lambda item: _latency_first_key(item, args))
            return chosen, f"{tier_name}_latency_objective_throughput_constraint_{floor_tps:.3f}", considered
        chosen = min(selected_pool, key=lambda item: _latency_first_key(item, args))
        return chosen, f"{tier_name}_latency_objective_throughput_constraint_unmet", considered

    if optimize_for == "throughput_constrained":
        chosen = _choose_throughput_candidate(selected_pool, args)
        return chosen, f"{tier_name}_throughput_objective_latency_constraint", considered

    if optimize_for == "throughput_constrained_trace_distilled":
        main_choice = _choose_throughput_candidate(selected_pool, args)
        trace_bucket = _trace_request_bucket(args)
        if str(main_choice.get("target")) == "edge":
            return main_choice, f"{tier_name}_throughput_objective_distilled_main_{trace_bucket}", considered
        if bool(option(args, "disable_trace_edge_rescue", False)):
            return main_choice, f"{tier_name}_throughput_objective_distilled_cloud_{trace_bucket}", considered

        edge_tier_name, edge_pool = _preferred_edge_tier_pool(capacity_ok, args)
        if edge_pool:
            edge_best_latency = min(_candidate_risk_latency(item) for item in edge_pool)
            edge_latency_pressure = edge_best_latency / slo
            edge_latency_critical = edge_tier_name in {"soft_slo"} or edge_latency_pressure >= float(option(args, "latency_critical_ratio", DEFAULT_LATENCY_CRITICAL_RATIO))
            edge_shortlist_ratio = float(option(args, "latency_critical_shortlist_ratio", DEFAULT_LATENCY_CRITICAL_SHORTLIST_RATIO)) if edge_latency_critical else float(option(args, "latency_shortlist_ratio", DEFAULT_LATENCY_SHORTLIST_RATIO))
            edge_shortlist = _latency_shortlist(edge_pool, shortlist_ratio=edge_shortlist_ratio)
            rescued_edge = _trace_pick_best_edge_candidate(edge_shortlist, main_choice, args)
            if rescued_edge is not None:
                edge_considered = _ordered_considered_pool(edge_pool, args)
                return rescued_edge, f"{edge_tier_name}_throughput_objective_distilled_edge_rescue_{trace_bucket}", edge_considered

        return main_choice, f"{tier_name}_throughput_objective_distilled_cloud_{trace_bucket}", considered

    if optimize_for == "throughput_constrained_trace":
        cloud_reference = _best_cloud_candidate(selected_pool, args) or _best_cloud_candidate(capacity_ok, args)
        if bool(option(args, "disable_trace_edge_rescue", False)):
            trace_bucket = _trace_request_bucket(args)
            chosen = _choose_throughput_candidate(selected_pool, args)
            return chosen, f"{tier_name}_throughput_objective_cloud_preferred_{trace_bucket}", considered
        edge_tier_name, edge_pool = _preferred_edge_tier_pool(capacity_ok, args)
        if edge_pool:
            edge_best_latency = min(_candidate_risk_latency(item) for item in edge_pool)
            edge_latency_pressure = edge_best_latency / slo
            edge_latency_critical = edge_tier_name in {"soft_slo"} or edge_latency_pressure >= float(option(args, "latency_critical_ratio", DEFAULT_LATENCY_CRITICAL_RATIO))
            edge_shortlist_ratio = float(option(args, "latency_critical_shortlist_ratio", DEFAULT_LATENCY_CRITICAL_SHORTLIST_RATIO)) if edge_latency_critical else float(option(args, "latency_shortlist_ratio", DEFAULT_LATENCY_SHORTLIST_RATIO))
            edge_shortlist = _latency_shortlist(edge_pool, shortlist_ratio=edge_shortlist_ratio)
            edge_considered = _ordered_considered_pool(edge_pool, args)
            chosen = _trace_pick_best_edge_candidate(edge_shortlist, cloud_reference, args)
            if chosen is not None:
                trace_bucket = _trace_request_bucket(args)
                return chosen, f"{edge_tier_name}_throughput_objective_edge_preferred_{trace_bucket}", edge_considered

        chosen = cloud_reference or _choose_throughput_candidate(selected_pool, args)
        trace_bucket = _trace_request_bucket(args)
        return chosen, f"{tier_name}_throughput_objective_trace_guard_cloud_fallback_{trace_bucket}", considered

    if optimize_for == "latency" or latency_critical:
        chosen = min(shortlist, key=lambda item: _latency_first_key(item, args))
        return chosen, f"{tier_name}_latency_mode", considered

    if optimize_for == "throughput":
        chosen = _choose_throughput_candidate(shortlist, args)
        return chosen, f"{tier_name}_throughput_mode", considered

    chosen = max(shortlist, key=lambda item: _balanced_selection_key(item, args))
    return chosen, f"{tier_name}_balanced_performance", considered

def compact_candidate_view(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config_id": candidate["config_id"],
        "target": candidate["target"],
        "device_id": candidate["device_id"],
        "quant_method": candidate["quant_method"],
        "quant_bits": candidate.get("quant_bits"),
        "batch_size": candidate.get("batch_size"),
        "max_tokens": candidate.get("max_tokens"),
        "kv_strategy": candidate.get("kv_strategy"),
        "predicted_p95_s": candidate["predicted_p95_s"],
        "predicted_p95_raw_s": candidate.get("predicted_p95_raw_s"),
        "risk_adjusted_p95_s": candidate.get("risk_adjusted_p95_s"),
        "risk_buffer_s": candidate.get("risk_buffer_s"),
        "predicted_tok_per_s": candidate["predicted_tok_per_s"],
        "predicted_tok_per_s_raw": candidate.get("predicted_tok_per_s_raw"),
        "predicted_tok_per_s_service": candidate.get("predicted_tok_per_s_service"),
        "predicted_rental_cost": candidate.get("predicted_rental_cost"),
        "performance_score": candidate.get("performance_score"),
        "utility_score": candidate.get("utility_score"),
        "utility_positive": candidate.get("utility_positive"),
        "pairwise_rank_score": candidate.get("pairwise_rank_score"),
        "pairwise_rank_applied": candidate.get("pairwise_rank_applied"),
        "p95_calibration_applied": candidate.get("p95_calibration_applied"),
        "p95_calibration_delta_s": candidate.get("p95_calibration_delta_s"),
        "rental_rate": candidate.get("rental_rate"),
        "within_memory": candidate.get("within_memory"),
        "within_token_budget": candidate.get("within_token_budget"),
        "within_batch_budget": candidate.get("within_batch_budget"),
        "within_slo": candidate.get("within_slo"),
        "within_safe_slo": candidate.get("within_safe_slo"),
        "within_risk_slo": candidate.get("within_risk_slo"),
        "within_risk_safe_slo": candidate.get("within_risk_safe_slo"),
        "memory_pressure": candidate.get("memory_pressure"),
        "near_limit_memory_penalty": candidate.get("near_limit_memory_penalty"),
        "violation_ratio": candidate.get("violation_ratio"),
        "risk_violation_ratio": candidate.get("risk_violation_ratio"),
        "score": candidate.get("score"),
    }


def main() -> None:
    args = parse_args()
    assets = load_assets(args.artifacts_dir)
    catalog = load_json(args.catalog)

    candidate_predictions: List[Dict[str, Any]] = []
    for candidate in catalog.get("edge_candidates", []):
        if not bool(candidate.get("edge_deployable", True)):
            continue
        if str(candidate.get("device_id")) != args.edge_device_id:
            continue
        candidate_predictions.append(predict_edge_candidate(candidate, args, assets))

    for candidate in catalog.get("cloud_candidates", []):
        candidate_predictions.append(predict_cloud_candidate(candidate, args))

    score_candidates(candidate_predictions, args)
    apply_pairwise_rank_scores(candidate_predictions, args, assets)
    chosen, decision_reason, considered_pool = choose_candidate(candidate_predictions, args)

    decision = {
        "decision_status": "selected" if chosen is not None else "infeasible",
        "decision_reason": decision_reason,
        "selected_target": chosen["target"] if chosen is not None else None,
        "selected_config_id": chosen["config_id"] if chosen is not None else None,
        "selected_device_id": chosen["device_id"] if chosen is not None else None,
        "selected_quant_method": chosen["quant_method"] if chosen is not None else None,
        "selected_quant_bits": chosen.get("quant_bits") if chosen is not None else None,
        "selected_batch_size": chosen.get("batch_size") if chosen is not None else None,
        "selected_max_tokens": chosen.get("max_tokens") if chosen is not None else None,
        "selected_kv_strategy": chosen.get("kv_strategy") if chosen is not None else None,
        "predicted_p95_s": chosen["predicted_p95_s"] if chosen is not None else None,
        "predicted_p95_raw_s": chosen.get("predicted_p95_raw_s") if chosen is not None else None,
        "predicted_tok_per_s": chosen["predicted_tok_per_s"] if chosen is not None else None,
        "predicted_rental_cost": chosen.get("predicted_rental_cost") if chosen is not None else None,
        "pairwise_rank_score": chosen.get("pairwise_rank_score") if chosen is not None else None,
        "p95_calibration_applied": chosen.get("p95_calibration_applied") if chosen is not None else False,
        "p95_calibration_delta_s": chosen.get("p95_calibration_delta_s") if chosen is not None else None,
        "rental_rate": chosen.get("rental_rate") if chosen is not None else None,
        "estimated_memory_mb": chosen["estimated_memory_mb"] if chosen is not None else None,
        "within_slo": chosen["within_slo"] if chosen is not None else False,
        "within_safe_slo": chosen.get("within_safe_slo", False) if chosen is not None else False,
        "within_token_budget": chosen["within_token_budget"] if chosen is not None else False,
        "within_batch_budget": chosen["within_batch_budget"] if chosen is not None else False,
        "score": chosen["score"] if chosen is not None else None,
    }

    result = {
        "request": {
            "edge_device_id": args.edge_device_id,
            "prompt_len": args.prompt_len,
            "gen_len": args.gen_len,
            "concurrency": args.concurrency,
            "request_total_tokens": args.prompt_len + args.gen_len,
            "slo_p95_s": args.slo_p95_s,
            "edge_mem_available_mb": args.edge_mem_available_mb,
            "edge_queue_delay_s": args.edge_queue_delay_s,
            "cloud_extra_delay_s": args.cloud_extra_delay_s,
            "slo_safety_margin": option(args, "slo_safety_margin", DEFAULT_SLO_SAFETY_MARGIN),
            "soft_slo_tolerance_ratio": option(args, "soft_slo_tolerance_ratio", DEFAULT_SOFT_SLO_TOLERANCE_RATIO),
            "tie_cost_tolerance_ratio": option(args, "tie_cost_tolerance_ratio", DEFAULT_TIE_COST_TOLERANCE_RATIO),
            "memory_penalty_threshold": option(args, "memory_penalty_threshold", DEFAULT_MEMORY_PENALTY_THRESHOLD),
        },
        "decision": decision,
        "considered_pool": [compact_candidate_view(item) for item in considered_pool],
        "all_candidates": candidate_predictions,
        "known_training_values": assets["known_values"],
    }

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()




