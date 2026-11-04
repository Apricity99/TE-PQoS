import argparse
import json
import os
import sys
import types
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def patch_optional_pyarrow() -> None:
    try:
        import pyarrow  # type: ignore  # noqa: F401
    except Exception:
        fake_module = types.ModuleType("pyarrow")
        fake_module.__version__ = "17.0.0"

        class _ArrowStub:
            pass

        fake_module.Table = _ArrowStub
        fake_module.RecordBatch = _ArrowStub
        fake_module.Array = _ArrowStub
        fake_module.ChunkedArray = _ArrowStub
        sys.modules["pyarrow"] = fake_module


patch_optional_pyarrow()

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from perf_feature_utils import (
    BASE_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
    NUMERIC_CLEAN_COLS,
    TARGET_COLS,
    add_workload_features,
    build_feature_frame,
    build_p95_calibration_frame,
    build_pairwise_feature_frame,
    build_pairwise_feature_record,
    fill_or_default,
    normalize_model_family,
    parse_config_metadata,
)


OUTDIR = "artifacts_gbdt_multi"
DEVICE_RENTAL_RATES = {
    "edge_3090": 1.0,
    "edge_4090": 1.3,
    "edge_4060": 0.85,
}


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    source_tag: str
    device_id: str
    quant_method: str
    config_id: str
    model_family: str
    quant_bits: Optional[int] = None
    batch_size: Optional[int] = None
    max_tokens: Optional[int] = None
    kv_strategy: Optional[str] = None


DEFAULT_DATASETS = [
    DatasetSpec(
        path="data/benchmarks/edge_4060/awq/awq_marlin_len384_grid_dense_conc.csv",
        source_tag="edge_4060_awq_legacy",
        device_id="edge_4060",
        quant_method="awq",
        config_id="awq_marlin_len384",
        model_family="llama2_7b",
        quant_bits=4,
        batch_size=-1,
        max_tokens=384,
        kv_strategy="dense",
    ),
    DatasetSpec(
        path="data/benchmarks/edge_3090/awq/edge_3090_awq4_bs8_tok2048_kvauto_valid.csv",
        source_tag="edge_3090_awq_bs8_kvauto",
        device_id="edge_3090",
        quant_method="awq",
        config_id="awq4_bs8_tok2048_kvauto",
        model_family="llama2_7b",
        quant_bits=4,
        batch_size=8,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
    DatasetSpec(
        path="data/benchmarks/edge_3090/awq/edge_3090_awq4_bs16_tok2048_kvauto_valid.csv",
        source_tag="edge_3090_awq_bs16_kvauto",
        device_id="edge_3090",
        quant_method="awq",
        config_id="awq4_bs16_tok2048_kvauto",
        model_family="llama2_7b",
        quant_bits=4,
        batch_size=16,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
    DatasetSpec(
        path="data/benchmarks/edge_3090/gptq/edge_3090_gptq4_bs8_tok2048_kvauto_valid.csv",
        source_tag="edge_3090_gptq_bs8_kvauto",
        device_id="edge_3090",
        quant_method="gptq",
        config_id="gptq4_bs8_tok2048_kvauto",
        model_family="llama2_7b",
        quant_bits=4,
        batch_size=8,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
    DatasetSpec(
        path="data/benchmarks/edge_3090/gptq/edge_3090_gptq4_bs16_tok2048_kvauto_valid.csv",
        source_tag="edge_3090_gptq_bs16_kvauto",
        device_id="edge_3090",
        quant_method="gptq",
        config_id="gptq4_bs16_tok2048_kvauto",
        model_family="llama2_7b",
        quant_bits=4,
        batch_size=16,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
    DatasetSpec(
        path="data/benchmarks/edge_4090/awq/edge_4090_awq4_bs8_tok2048_kvauto_valid.csv",
        source_tag="edge_4090_awq_bs8_kvauto",
        device_id="edge_4090",
        quant_method="awq",
        config_id="awq4_bs8_tok2048_kvauto",
        model_family="llama2_7b",
        quant_bits=4,
        batch_size=8,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
    DatasetSpec(
        path="data/benchmarks/edge_4090/awq/edge_4090_awq4_bs16_tok2048_kvauto_valid.csv",
        source_tag="edge_4090_awq_bs16_kvauto",
        device_id="edge_4090",
        quant_method="awq",
        config_id="awq4_bs16_tok2048_kvauto",
        model_family="llama2_7b",
        quant_bits=4,
        batch_size=16,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
    DatasetSpec(
        path="data/benchmarks/edge_4090/gptq/edge_4090_gptq4_bs8_tok2048_kvauto_valid.csv",
        source_tag="edge_4090_gptq_bs8_kvauto",
        device_id="edge_4090",
        quant_method="gptq",
        config_id="gptq4_bs8_tok2048_kvauto",
        model_family="llama2_7b",
        quant_bits=4,
        batch_size=8,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
    DatasetSpec(
        path="data/benchmarks/edge_4090/gptq/edge_4090_gptq4_bs16_tok2048_kvauto_valid.csv",
        source_tag="edge_4090_gptq_bs16_kvauto",
        device_id="edge_4090",
        quant_method="gptq",
        config_id="gptq4_bs16_tok2048_kvauto",
        model_family="llama2_7b",
        quant_bits=4,
        batch_size=16,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-device GBDT performance model from benchmark CSV files.",
    )
    parser.add_argument("--outdir", default=OUTDIR, help="Directory to save artifacts")
    parser.add_argument(
        "--keep-error-rows",
        action="store_true",
        help="Keep rows with error_reqs > 0 or timeout_reqs > 0",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Random split test ratio for interpolation metrics",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    parser.add_argument(
        "--trace-weight-csv",
        default=None,
        help="Optional bucketed public trace CSV used to importance-weight training samples.",
    )
    parser.add_argument(
        "--trace-weight-scope",
        default="prompt_gen",
        choices=["prompt_gen", "prompt_gen_concurrency"],
        help="Granularity used to align trace distribution and training distribution.",
    )
    parser.add_argument(
        "--trace-weight-alpha",
        type=float,
        default=1.0,
        help="Exponent applied to the importance ratio before clipping.",
    )
    parser.add_argument(
        "--trace-weight-min",
        type=float,
        default=0.25,
        help="Minimum per-sample weight after clipping.",
    )
    parser.add_argument(
        "--trace-weight-max",
        type=float,
        default=4.0,
        help="Maximum per-sample weight after clipping.",
    )
    return parser.parse_args()


def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    frame = pd.read_csv(spec.path)
    frame["source_file"] = os.path.basename(spec.path)
    frame["source_tag"] = spec.source_tag
    frame["device_id"] = fill_or_default(frame, "device_id", spec.device_id).astype(str)
    frame["config_id"] = fill_or_default(frame, "config_id", spec.config_id).astype(str)
    frame["model_family"] = (
        fill_or_default(frame, "model_id", spec.model_family)
        .astype(str)
        .map(normalize_model_family)
    )
    model_id_series = frame["model_id"] if "model_id" in frame.columns else pd.Series([""] * len(frame), index=frame.index)

    parsed_rows = [
        parse_config_metadata(
            config_id=str(config_id),
            quant_method=str(spec.quant_method or model_id),
            quant_bits=spec.quant_bits,
            batch_size=spec.batch_size,
            max_tokens=spec.max_tokens,
            kv_strategy=spec.kv_strategy,
        )
        for model_id, config_id in zip(model_id_series, frame["config_id"])
    ]
    parsed = pd.DataFrame(parsed_rows, index=frame.index)
    for column in ["config_id", "quant_method", "quant_bits", "batch_size", "max_tokens", "kv_strategy"]:
        frame[column] = parsed[column]

    if "total_reqs" in frame.columns and "success_reqs" not in frame.columns:
        frame["success_reqs"] = frame["total_reqs"]
    if "error_reqs" not in frame.columns:
        frame["error_reqs"] = 0
    if "timeout_reqs" not in frame.columns:
        frame["timeout_reqs"] = 0

    return frame


def prepare_dataset(keep_error_rows: bool) -> pd.DataFrame:
    frames = [load_dataset(spec) for spec in DEFAULT_DATASETS]
    df = pd.concat(frames, ignore_index=True)

    for column in NUMERIC_CLEAN_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=TARGET_COLS + ["prompt_len", "gen_len", "concurrency"]).copy()
    df = df[(df["lat_p95_s"] > 0) & (df["tok_per_s"] > 0)].copy()

    if not keep_error_rows:
        has_error = pd.Series(False, index=df.index)
        if "error_reqs" in df.columns:
            has_error = has_error | (df["error_reqs"].fillna(0) > 0)
        if "timeout_reqs" in df.columns:
            has_error = has_error | (df["timeout_reqs"].fillna(0) > 0)
        df = df[~has_error].copy()

    df = add_workload_features(df)
    return df.reset_index(drop=True)


def align_columns(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns, fill_value=0)


def trace_weight_key_columns(scope: str) -> List[str]:
    if scope == "prompt_gen_concurrency":
        return ["prompt_len", "gen_len", "concurrency"]
    return ["prompt_len", "gen_len"]


def _normalized_frequency_map(frame: pd.DataFrame, key_cols: Sequence[str]) -> Dict[Tuple[int, ...], float]:
    if frame.empty:
        return {}
    counts = (
        frame.groupby(list(key_cols), dropna=False)
        .size()
        .astype(float)
    )
    total = float(counts.sum())
    if total <= 0:
        return {}
    return {
        tuple(int(v) for v in key): float(value / total)
        for key, value in counts.items()
    }


def build_trace_sample_weights(
    df: pd.DataFrame,
    *,
    trace_csv: str,
    scope: str,
    alpha: float,
    min_weight: float,
    max_weight: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    trace_df = pd.read_csv(trace_csv, encoding="utf-8-sig")
    key_cols = trace_weight_key_columns(scope)
    missing_cols = [column for column in key_cols if column not in trace_df.columns]
    if missing_cols:
        raise ValueError(f"trace weight csv missing columns: {missing_cols}")

    trace_view = trace_df[key_cols].copy()
    dataset_view = df[key_cols].copy()
    for column in key_cols:
        trace_view[column] = pd.to_numeric(trace_view[column], errors="coerce").fillna(-1).astype(int)
        dataset_view[column] = pd.to_numeric(dataset_view[column], errors="coerce").fillna(-1).astype(int)

    trace_prob = _normalized_frequency_map(trace_view, key_cols)
    dataset_prob = _normalized_frequency_map(dataset_view, key_cols)
    fallback = max(min_weight, 1e-6)

    weights: List[float] = []
    matched = 0
    for row in dataset_view.to_dict(orient="records"):
        key = tuple(int(row[column]) for column in key_cols)
        if key in trace_prob:
            matched += 1
        ratio = trace_prob.get(key, 0.0) / max(dataset_prob.get(key, 1e-12), 1e-12)
        if ratio <= 0.0:
            weight = fallback
        else:
            weight = float(np.clip(ratio ** alpha, min_weight, max_weight))
        weights.append(weight)

    weights_array = np.asarray(weights, dtype=float)
    metadata = {
        "trace_weight_csv": trace_csv,
        "trace_weight_scope": scope,
        "trace_weight_alpha": float(alpha),
        "trace_weight_min": float(min_weight),
        "trace_weight_max": float(max_weight),
        "matched_rows": int(matched),
        "total_rows": int(len(df)),
        "matched_ratio": float(matched / len(df)) if len(df) > 0 else float("nan"),
        "weight_summary": {
            "mean": float(np.mean(weights_array)) if len(weights_array) else float("nan"),
            "min": float(np.min(weights_array)) if len(weights_array) else float("nan"),
            "max": float(np.max(weights_array)) if len(weights_array) else float("nan"),
            "p50": float(np.quantile(weights_array, 0.5)) if len(weights_array) else float("nan"),
            "p90": float(np.quantile(weights_array, 0.9)) if len(weights_array) else float("nan"),
        },
        "trace_bucket_count": int(len(trace_prob)),
        "dataset_bucket_count": int(len(dataset_prob)),
    }
    return weights_array, metadata


def make_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.05,
        max_iter=1200,
        min_samples_leaf=4,
        l2_regularization=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=40,
        random_state=42,
    )


def make_p95_calibrator() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=500,
        min_samples_leaf=8,
        l2_regularization=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=43,
    )


def fit_p95_calibrator(
    X_train_base: pd.DataFrame,
    X_test_base: pd.DataFrame,
    y_p95_train: np.ndarray,
    raw_p95_train: np.ndarray,
    raw_p95_test: np.ndarray,
    sample_weight_train: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    calibrator = make_p95_calibrator()
    X_train_cal = build_p95_calibration_frame(X_train_base, raw_p95_train)
    X_test_cal = build_p95_calibration_frame(X_test_base, raw_p95_test)

    raw_log_train = np.log1p(np.clip(raw_p95_train, 1e-9, None))
    residual_target = np.log1p(np.clip(y_p95_train, 1e-9, None)) - raw_log_train
    calibrator.fit(X_train_cal, residual_target, sample_weight=sample_weight_train)

    train_residual_pred = calibrator.predict(X_train_cal)
    train_calibrated_log = np.log1p(np.clip(raw_p95_train, 1e-9, None)) + train_residual_pred
    train_calibrated_p95 = np.clip(np.expm1(train_calibrated_log), 1e-9, None)

    residual_pred = calibrator.predict(X_test_cal)
    calibrated_log = np.log1p(np.clip(raw_p95_test, 1e-9, None)) + residual_pred
    calibrated_p95 = np.clip(np.expm1(calibrated_log), 1e-9, None)

    return {
        "model": calibrator,
        "feature_cols": list(X_train_cal.columns),
        "train_p95_pred": train_calibrated_p95,
        "p95_pred": calibrated_p95,
        "residual_pred": residual_pred,
    }


def summarize_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    safe_weight = sample_weight if sample_weight is not None else None
    abs_pct = np.abs((y_true - y_pred) / y_true) * 100.0
    if safe_weight is None:
        mape_pct = float(np.mean(abs_pct))
    else:
        mape_pct = float(np.average(abs_pct, weights=safe_weight))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred, sample_weight=safe_weight)),
        "r2": float(r2_score(y_true, y_pred, sample_weight=safe_weight)),
        "mape_pct": mape_pct,
        "median_true": float(np.median(y_true)),
        "median_pred": float(np.median(y_pred)),
    }


def default_rental_rate(device_id: str) -> float:
    return float(DEVICE_RENTAL_RATES.get(str(device_id), 1.0))


def make_ranker() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=500,
        min_samples_leaf=8,
        l2_regularization=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=44,
    )


def _pairwise_target(left: Dict[str, object], right: Dict[str, object]) -> Optional[int]:
    left_p95 = float(left["lat_p95_s"])
    right_p95 = float(right["lat_p95_s"])
    p95_tol = max(1e-6, 0.005 * min(max(left_p95, 1e-6), max(right_p95, 1e-6)))
    if left_p95 < right_p95 - p95_tol:
        return 1
    if right_p95 < left_p95 - p95_tol:
        return 0

    left_tps = float(left["tok_per_s"])
    right_tps = float(right["tok_per_s"])
    tps_tol = max(1e-6, 0.005 * max(abs(left_tps), abs(right_tps), 1.0))
    if left_tps > right_tps + tps_tol:
        return 1
    if right_tps > left_tps + tps_tol:
        return 0
    return None


def _prepare_rank_frame(frame: pd.DataFrame, p95_col: str, tps_col: str) -> pd.DataFrame:
    ranked = frame.copy()
    ranked[p95_col] = pd.to_numeric(ranked[p95_col], errors="coerce")
    ranked[tps_col] = pd.to_numeric(ranked[tps_col], errors="coerce")
    ranked["predicted_rental_cost"] = ranked[p95_col] * ranked["device_id"].astype(str).map(default_rental_rate).astype(float)
    if "peak_mem_mb" in ranked.columns:
        ranked["estimated_memory_mb"] = pd.to_numeric(ranked["peak_mem_mb"], errors="coerce").fillna(0.0)
    else:
        ranked["estimated_memory_mb"] = 0.0
    if "p95_calibration_delta_s" not in ranked.columns:
        ranked["p95_calibration_delta_s"] = 0.0
    return ranked


def build_pairwise_examples(
    frame: pd.DataFrame,
    *,
    p95_col: str = "p95_calibrated_s",
    tps_col: str = "tps_hat",
) -> Dict[str, object]:
    ranked = _prepare_rank_frame(frame, p95_col, tps_col)
    records: List[Dict[str, object]] = []
    targets: List[int] = []
    sample_weights: List[float] = []
    workload_groups = 0
    undirected_pairs = 0

    for _, group in ranked.groupby(["device_id", "prompt_len", "gen_len", "concurrency"], sort=False):
        rows = group.sort_values("config_id").to_dict(orient="records")
        if len(rows) < 2:
            continue
        workload_groups += 1
        for left_idx in range(len(rows)):
            for right_idx in range(left_idx + 1, len(rows)):
                left = rows[left_idx]
                right = rows[right_idx]
                target = _pairwise_target(left, right)
                if target is None:
                    continue
                undirected_pairs += 1
                for first, second, label in ((left, right, target), (right, left, 1 - target)):
                    records.append(
                        build_pairwise_feature_record(
                            device_id=str(first["device_id"]),
                            prompt_len=int(first["prompt_len"]),
                            gen_len=int(first["gen_len"]),
                            concurrency=int(first["concurrency"]),
                            left_config_id=str(first["config_id"]),
                            right_config_id=str(second["config_id"]),
                            left_quant_method=str(first["quant_method"]),
                            right_quant_method=str(second["quant_method"]),
                            left_kv_strategy=str(first["kv_strategy"]),
                            right_kv_strategy=str(second["kv_strategy"]),
                            left_quant_bits=first.get("quant_bits"),
                            right_quant_bits=second.get("quant_bits"),
                            left_batch_size=first.get("batch_size"),
                            right_batch_size=second.get("batch_size"),
                            left_max_tokens=first.get("max_tokens"),
                            right_max_tokens=second.get("max_tokens"),
                            left_predicted_p95_s=float(first[p95_col]),
                            right_predicted_p95_s=float(second[p95_col]),
                            left_predicted_tok_per_s=float(first[tps_col]),
                            right_predicted_tok_per_s=float(second[tps_col]),
                            left_predicted_rental_cost=float(first["predicted_rental_cost"]),
                            right_predicted_rental_cost=float(second["predicted_rental_cost"]),
                            left_estimated_memory_mb=first.get("estimated_memory_mb"),
                            right_estimated_memory_mb=second.get("estimated_memory_mb"),
                            left_p95_calibration_delta_s=float(first.get("p95_calibration_delta_s", 0.0) or 0.0),
                            right_p95_calibration_delta_s=float(second.get("p95_calibration_delta_s", 0.0) or 0.0),
                        )
                    )
                    targets.append(int(label))
                    left_weight = float(first.get("sample_weight", 1.0) or 1.0)
                    right_weight = float(second.get("sample_weight", 1.0) or 1.0)
                    sample_weights.append(0.5 * (left_weight + right_weight))

    X_pair = build_pairwise_feature_frame(records) if records else pd.DataFrame()
    return {
        "X": X_pair,
        "y": np.asarray(targets, dtype=int),
        "sample_weight": np.asarray(sample_weights, dtype=float),
        "pair_rows": int(len(records)),
        "undirected_pairs": int(undirected_pairs),
        "workload_groups": int(workload_groups),
    }


def summarize_pairwise_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "pair_accuracy": float("nan"),
            "pair_rows": 0,
            "positive_rate": float("nan"),
            "mean_positive_prob": float("nan"),
        }
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "pair_accuracy": float(accuracy_score(y_true, y_pred)),
        "pair_rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "mean_positive_prob": float(np.mean(y_prob)),
    }


def evaluate_pairwise_top1(
    frame: pd.DataFrame,
    *,
    model: HistGradientBoostingClassifier,
    feature_cols: List[str],
    p95_col: str = "p95_calibrated_s",
    tps_col: str = "tps_hat",
) -> Dict[str, float]:
    ranked = _prepare_rank_frame(frame, p95_col, tps_col)
    correct = 0
    groups = 0

    for _, group in ranked.groupby(["device_id", "prompt_len", "gen_len", "concurrency"], sort=False):
        rows = group.sort_values("config_id").to_dict(orient="records")
        if len(rows) < 2:
            continue
        groups += 1
        scores = {str(row["config_id"]): 0.0 for row in rows}
        pair_records: List[Dict[str, object]] = []
        pair_meta: List[tuple[str, str]] = []

        for left_idx in range(len(rows)):
            for right_idx in range(left_idx + 1, len(rows)):
                left = rows[left_idx]
                right = rows[right_idx]
                pair_records.append(
                    build_pairwise_feature_record(
                        device_id=str(left["device_id"]),
                        prompt_len=int(left["prompt_len"]),
                        gen_len=int(left["gen_len"]),
                        concurrency=int(left["concurrency"]),
                        left_config_id=str(left["config_id"]),
                        right_config_id=str(right["config_id"]),
                        left_quant_method=str(left["quant_method"]),
                        right_quant_method=str(right["quant_method"]),
                        left_kv_strategy=str(left["kv_strategy"]),
                        right_kv_strategy=str(right["kv_strategy"]),
                        left_quant_bits=left.get("quant_bits"),
                        right_quant_bits=right.get("quant_bits"),
                        left_batch_size=left.get("batch_size"),
                        right_batch_size=right.get("batch_size"),
                        left_max_tokens=left.get("max_tokens"),
                        right_max_tokens=right.get("max_tokens"),
                        left_predicted_p95_s=float(left[p95_col]),
                        right_predicted_p95_s=float(right[p95_col]),
                        left_predicted_tok_per_s=float(left[tps_col]),
                        right_predicted_tok_per_s=float(right[tps_col]),
                        left_predicted_rental_cost=float(left["predicted_rental_cost"]),
                        right_predicted_rental_cost=float(right["predicted_rental_cost"]),
                        left_estimated_memory_mb=left.get("estimated_memory_mb"),
                        right_estimated_memory_mb=right.get("estimated_memory_mb"),
                        left_p95_calibration_delta_s=float(left.get("p95_calibration_delta_s", 0.0) or 0.0),
                        right_p95_calibration_delta_s=float(right.get("p95_calibration_delta_s", 0.0) or 0.0),
                    )
                )
                pair_records.append(
                    build_pairwise_feature_record(
                        device_id=str(right["device_id"]),
                        prompt_len=int(right["prompt_len"]),
                        gen_len=int(right["gen_len"]),
                        concurrency=int(right["concurrency"]),
                        left_config_id=str(right["config_id"]),
                        right_config_id=str(left["config_id"]),
                        left_quant_method=str(right["quant_method"]),
                        right_quant_method=str(left["quant_method"]),
                        left_kv_strategy=str(right["kv_strategy"]),
                        right_kv_strategy=str(left["kv_strategy"]),
                        left_quant_bits=right.get("quant_bits"),
                        right_quant_bits=left.get("quant_bits"),
                        left_batch_size=right.get("batch_size"),
                        right_batch_size=left.get("batch_size"),
                        left_max_tokens=right.get("max_tokens"),
                        right_max_tokens=left.get("max_tokens"),
                        left_predicted_p95_s=float(right[p95_col]),
                        right_predicted_p95_s=float(left[p95_col]),
                        left_predicted_tok_per_s=float(right[tps_col]),
                        right_predicted_tok_per_s=float(left[tps_col]),
                        left_predicted_rental_cost=float(right["predicted_rental_cost"]),
                        right_predicted_rental_cost=float(left["predicted_rental_cost"]),
                        left_estimated_memory_mb=right.get("estimated_memory_mb"),
                        right_estimated_memory_mb=left.get("estimated_memory_mb"),
                        left_p95_calibration_delta_s=float(right.get("p95_calibration_delta_s", 0.0) or 0.0),
                        right_p95_calibration_delta_s=float(left.get("p95_calibration_delta_s", 0.0) or 0.0),
                    )
                )
                pair_meta.append((str(left["config_id"]), str(right["config_id"])))

        if not pair_records:
            continue

        X_group = build_pairwise_feature_frame(pair_records).reindex(columns=feature_cols, fill_value=0.0)
        probs = model.predict_proba(X_group)[:, 1]
        for idx, (left_cfg, right_cfg) in enumerate(pair_meta):
            left_prob = float(probs[2 * idx])
            right_prob = float(probs[2 * idx + 1])
            left_win_prob = 0.5 * (left_prob + (1.0 - right_prob))
            scores[left_cfg] += left_win_prob
            scores[right_cfg] += 1.0 - left_win_prob

        predicted_best = min(
            rows,
            key=lambda row: (
                -scores[str(row["config_id"])],
                float(row[p95_col]),
                -float(row[tps_col]),
                str(row["config_id"]),
            ),
        )
        actual_best = min(
            rows,
            key=lambda row: (
                float(row["lat_p95_s"]),
                -float(row["tok_per_s"]),
                str(row["config_id"]),
            ),
        )
        if str(predicted_best["config_id"]) == str(actual_best["config_id"]):
            correct += 1

    return {
        "top1_accuracy": float(correct / groups) if groups > 0 else float("nan"),
        "top1_groups": int(groups),
    }


def fit_pairwise_ranker(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    p95_col: str = "p95_calibrated_s",
    tps_col: str = "tps_hat",
) -> Dict[str, object]:
    train_pairs = build_pairwise_examples(train_frame, p95_col=p95_col, tps_col=tps_col)
    test_pairs = build_pairwise_examples(test_frame, p95_col=p95_col, tps_col=tps_col)

    if train_pairs["X"].empty or len(np.unique(train_pairs["y"])) < 2:
        return {
            "model": None,
            "feature_cols": [],
            "metrics": {
                "pair_accuracy": float("nan"),
                "pair_rows": int(len(test_pairs["y"])),
                "positive_rate": float("nan"),
                "mean_positive_prob": float("nan"),
                "top1_accuracy": float("nan"),
                "top1_groups": 0,
                "train_pair_rows": int(len(train_pairs["y"])),
                "test_pair_rows": int(len(test_pairs["y"])),
                "train_workload_groups": int(train_pairs["workload_groups"]),
                "test_workload_groups": int(test_pairs["workload_groups"]),
            },
        }

    ranker = make_ranker()
    ranker.fit(train_pairs["X"], train_pairs["y"], sample_weight=train_pairs["sample_weight"])

    if test_pairs["X"].empty:
        pair_metrics = summarize_pairwise_metrics(np.asarray([], dtype=int), np.asarray([], dtype=float))
    else:
        test_prob = ranker.predict_proba(test_pairs["X"])[:, 1]
        pair_metrics = summarize_pairwise_metrics(test_pairs["y"], test_prob)
    top1_metrics = evaluate_pairwise_top1(
        test_frame,
        model=ranker,
        feature_cols=list(train_pairs["X"].columns),
        p95_col=p95_col,
        tps_col=tps_col,
    )

    metrics = {
        **pair_metrics,
        **top1_metrics,
        "train_pair_rows": int(len(train_pairs["y"])),
        "test_pair_rows": int(len(test_pairs["y"])),
        "train_workload_groups": int(train_pairs["workload_groups"]),
        "test_workload_groups": int(test_pairs["workload_groups"]),
    }
    return {
        "model": ranker,
        "feature_cols": list(train_pairs["X"].columns),
        "metrics": metrics,
    }


def fit_predict_pair(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_p95_train: np.ndarray,
    y_tps_train: np.ndarray,
    sample_weight_train: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    p95_model = make_regressor()
    tps_model = make_regressor()

    p95_model.fit(X_train, np.log1p(y_p95_train), sample_weight=sample_weight_train)
    tps_model.fit(X_train, np.log1p(y_tps_train), sample_weight=sample_weight_train)

    p95_pred = np.expm1(p95_model.predict(X_test))
    tps_pred = np.expm1(tps_model.predict(X_test))

    return {
        "p95_model": p95_model,
        "tps_model": tps_model,
        "p95_pred": p95_pred,
        "tps_pred": tps_pred,
    }


def evaluate_random_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    X_full = build_feature_frame(df)
    y_p95 = df["lat_p95_s"].to_numpy()
    y_tps = df["tok_per_s"].to_numpy()
    groups = df["source_tag"].to_numpy()
    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=groups,
    )

    X_train = X_full.iloc[train_idx].reset_index(drop=True)
    X_test = align_columns(X_full.iloc[test_idx].reset_index(drop=True), list(X_train.columns))

    fit_result = fit_predict_pair(
        X_train=X_train,
        X_test=X_test,
        y_p95_train=y_p95[train_idx],
        y_tps_train=y_tps[train_idx],
        sample_weight_train=sample_weight[train_idx] if sample_weight is not None else None,
    )
    raw_p95_train = np.expm1(fit_result["p95_model"].predict(X_train))
    tps_train_pred = np.expm1(fit_result["tps_model"].predict(X_train))
    calibration_result = fit_p95_calibrator(
        X_train_base=X_train,
        X_test_base=X_test,
        y_p95_train=y_p95[train_idx],
        raw_p95_train=raw_p95_train,
        raw_p95_test=fit_result["p95_pred"],
        sample_weight_train=sample_weight[train_idx] if sample_weight is not None else None,
    )

    train_predicted = df.iloc[train_idx].reset_index(drop=True).copy()
    train_predicted["p95_hat_s"] = raw_p95_train
    train_predicted["p95_calibrated_s"] = calibration_result["train_p95_pred"]
    train_predicted["p95_calibration_delta_s"] = train_predicted["p95_calibrated_s"] - train_predicted["p95_hat_s"]
    train_predicted["tps_hat"] = tps_train_pred
    train_predicted["sample_weight"] = sample_weight[train_idx] if sample_weight is not None else 1.0

    test_predicted = df.iloc[test_idx].reset_index(drop=True).copy()
    test_predicted["p95_hat_s"] = fit_result["p95_pred"]
    test_predicted["p95_calibrated_s"] = calibration_result["p95_pred"]
    test_predicted["p95_calibration_delta_s"] = test_predicted["p95_calibrated_s"] - test_predicted["p95_hat_s"]
    test_predicted["tps_hat"] = fit_result["tps_pred"]
    test_predicted["sample_weight"] = sample_weight[test_idx] if sample_weight is not None else 1.0

    pairwise_rank = fit_pairwise_ranker(train_predicted, test_predicted)

    return {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "feature_cols": list(X_train.columns),
        "p95_calibration_feature_cols": calibration_result["feature_cols"],
        "pairwise_rank_feature_cols": pairwise_rank["feature_cols"],
        "train_groups": sorted(df.iloc[train_idx]["source_tag"].unique().tolist()),
        "test_groups": sorted(df.iloc[test_idx]["source_tag"].unique().tolist()),
        "p95": summarize_metrics(y_p95[test_idx], fit_result["p95_pred"], sample_weight=sample_weight[test_idx] if sample_weight is not None else None),
        "p95_calibrated": summarize_metrics(y_p95[test_idx], calibration_result["p95_pred"], sample_weight=sample_weight[test_idx] if sample_weight is not None else None),
        "tps": summarize_metrics(y_tps[test_idx], fit_result["tps_pred"], sample_weight=sample_weight[test_idx] if sample_weight is not None else None),
        "pairwise_rank": pairwise_rank["metrics"],
        "split_idx": {
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
        },
    }


def evaluate_leave_one_group_out(df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None) -> List[Dict[str, object]]:
    X_full = build_feature_frame(df)
    y_p95 = df["lat_p95_s"].to_numpy()
    y_tps = df["tok_per_s"].to_numpy()
    results: List[Dict[str, object]] = []

    for held_out_group in sorted(df["source_tag"].unique().tolist()):
        test_mask = df["source_tag"] == held_out_group
        train_mask = ~test_mask

        X_train = X_full.loc[train_mask].reset_index(drop=True)
        X_test = align_columns(
            X_full.loc[test_mask].reset_index(drop=True),
            list(X_train.columns),
        )

        fit_result = fit_predict_pair(
            X_train=X_train,
            X_test=X_test,
            y_p95_train=y_p95[train_mask.to_numpy()],
            y_tps_train=y_tps[train_mask.to_numpy()],
            sample_weight_train=sample_weight[train_mask.to_numpy()] if sample_weight is not None else None,
        )
        raw_p95_train = np.expm1(fit_result["p95_model"].predict(X_train))
        calibration_result = fit_p95_calibrator(
            X_train_base=X_train,
            X_test_base=X_test,
            y_p95_train=y_p95[train_mask.to_numpy()],
            raw_p95_train=raw_p95_train,
            raw_p95_test=fit_result["p95_pred"],
            sample_weight_train=sample_weight[train_mask.to_numpy()] if sample_weight is not None else None,
        )

        results.append(
            {
                "held_out_group": held_out_group,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_groups": sorted(df.loc[train_mask, "source_tag"].unique().tolist()),
                "p95": summarize_metrics(y_p95[test_mask.to_numpy()], fit_result["p95_pred"], sample_weight=sample_weight[test_mask.to_numpy()] if sample_weight is not None else None),
                "p95_calibrated": summarize_metrics(y_p95[test_mask.to_numpy()], calibration_result["p95_pred"], sample_weight=sample_weight[test_mask.to_numpy()] if sample_weight is not None else None),
                "tps": summarize_metrics(y_tps[test_mask.to_numpy()], fit_result["tps_pred"], sample_weight=sample_weight[test_mask.to_numpy()] if sample_weight is not None else None),
            }
        )

    return results


def fit_full_models(df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None) -> Dict[str, object]:
    X_full = build_feature_frame(df)
    y_p95 = df["lat_p95_s"].to_numpy()
    y_tps = df["tok_per_s"].to_numpy()

    fit_result = fit_predict_pair(
        X_train=X_full,
        X_test=X_full,
        y_p95_train=y_p95,
        y_tps_train=y_tps,
        sample_weight_train=sample_weight,
    )
    calibration_result = fit_p95_calibrator(
        X_train_base=X_full,
        X_test_base=X_full,
        y_p95_train=y_p95,
        raw_p95_train=fit_result["p95_pred"],
        raw_p95_test=fit_result["p95_pred"],
        sample_weight_train=sample_weight,
    )

    predicted = df.copy()
    predicted["p95_hat_s"] = fit_result["p95_pred"]
    predicted["p95_calibrated_s"] = calibration_result["p95_pred"]
    predicted["p95_calibration_delta_s"] = predicted["p95_calibrated_s"] - predicted["p95_hat_s"]
    predicted["tps_hat"] = fit_result["tps_pred"]
    predicted["sample_weight"] = sample_weight if sample_weight is not None else 1.0
    predicted["p95_abs_err_s"] = (predicted["lat_p95_s"] - predicted["p95_hat_s"]).abs()
    predicted["p95_calibrated_abs_err_s"] = (predicted["lat_p95_s"] - predicted["p95_calibrated_s"]).abs()
    predicted["tps_abs_err"] = (predicted["tok_per_s"] - predicted["tps_hat"]).abs()

    pairwise_rank = fit_pairwise_ranker(predicted, predicted)

    return {
        "feature_cols": list(X_full.columns),
        "p95_calibration_feature_cols": calibration_result["feature_cols"],
        "pairwise_rank_feature_cols": pairwise_rank["feature_cols"],
        "p95_model": fit_result["p95_model"],
        "p95_calibrator": calibration_result["model"],
        "tps_model": fit_result["tps_model"],
        "pairwise_ranker": pairwise_rank["model"],
        "predicted": predicted,
        "p95_metrics": summarize_metrics(y_p95, fit_result["p95_pred"], sample_weight=sample_weight),
        "p95_calibrated_metrics": summarize_metrics(y_p95, calibration_result["p95_pred"], sample_weight=sample_weight),
        "pairwise_rank_metrics": pairwise_rank["metrics"],
    }


def source_summary(df: pd.DataFrame) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for source_tag, part in df.groupby("source_tag", sort=True):
        summary.append(
            {
                "source_tag": source_tag,
                "rows": int(len(part)),
                "device_id": sorted(part["device_id"].astype(str).unique().tolist()),
                "quant_method": sorted(part["quant_method"].astype(str).unique().tolist()),
                "config_id": sorted(part["config_id"].astype(str).unique().tolist()),
                "kv_strategy": sorted(part["kv_strategy"].astype(str).unique().tolist()),
                "batch_size": sorted(pd.to_numeric(part["batch_size"], errors="coerce").fillna(-1).astype(int).unique().tolist()),
                "max_tokens": sorted(pd.to_numeric(part["max_tokens"], errors="coerce").fillna(0).astype(int).unique().tolist()),
                "p95_median_s": float(part["lat_p95_s"].median()),
                "tps_median": float(part["tok_per_s"].median()),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = prepare_dataset(keep_error_rows=args.keep_error_rows)
    sample_weight = None
    trace_weight_metadata = None
    if args.trace_weight_csv:
        sample_weight, trace_weight_metadata = build_trace_sample_weights(
            df,
            trace_csv=args.trace_weight_csv,
            scope=args.trace_weight_scope,
            alpha=float(args.trace_weight_alpha),
            min_weight=float(args.trace_weight_min),
            max_weight=float(args.trace_weight_max),
        )
        df = df.copy()
        df["sample_weight"] = sample_weight
    random_metrics = evaluate_random_split(
        df=df,
        test_size=args.test_size,
        random_state=args.random_state,
        sample_weight=sample_weight,
    )
    logo_metrics = evaluate_leave_one_group_out(df, sample_weight=sample_weight)
    full_fit = fit_full_models(df, sample_weight=sample_weight)

    combined_csv = os.path.join(args.outdir, "combined_perf_dataset.csv")
    predicted_csv = os.path.join(args.outdir, "combined_perf_with_pred.csv")
    df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
    full_fit["predicted"].to_csv(predicted_csv, index=False, encoding="utf-8-sig")

    joblib.dump(full_fit["p95_model"], os.path.join(args.outdir, "gbdt_p95.joblib"))
    joblib.dump(full_fit["p95_calibrator"], os.path.join(args.outdir, "p95_calibrator.joblib"))
    joblib.dump(full_fit["tps_model"], os.path.join(args.outdir, "gbdt_tps.joblib"))
    if full_fit["pairwise_ranker"] is not None:
        joblib.dump(full_fit["pairwise_ranker"], os.path.join(args.outdir, "pairwise_ranker.joblib"))
    joblib.dump(full_fit["feature_cols"], os.path.join(args.outdir, "feature_cols.joblib"))
    joblib.dump(full_fit["p95_calibration_feature_cols"], os.path.join(args.outdir, "p95_calibration_feature_cols.joblib"))
    joblib.dump(full_fit["pairwise_rank_feature_cols"], os.path.join(args.outdir, "pairwise_rank_feature_cols.joblib"))
    joblib.dump(random_metrics["split_idx"], os.path.join(args.outdir, "random_split_idx.joblib"))

    metrics_random_split = dict(random_metrics)
    split_idx = metrics_random_split.pop("split_idx")
    metrics_p95_calibration = {
        "random_split": {
            "raw": random_metrics["p95"],
            "calibrated": random_metrics["p95_calibrated"],
        },
        "leave_one_group_out": [
            {
                "held_out_group": item["held_out_group"],
                "raw": item["p95"],
                "calibrated": item["p95_calibrated"],
            }
            for item in logo_metrics
        ],
        "full_fit": {
            "raw": full_fit["p95_metrics"],
            "calibrated": full_fit["p95_calibrated_metrics"],
        },
    }
    metrics_pairwise_ranker = {
        "random_split": random_metrics["pairwise_rank"],
        "full_fit": full_fit["pairwise_rank_metrics"],
    }

    with open(os.path.join(args.outdir, "metrics_random_split.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_random_split, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(args.outdir, "metrics_leave_one_group_out.json"), "w", encoding="utf-8") as handle:
        json.dump(logo_metrics, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(args.outdir, "metrics_p95_calibration.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_p95_calibration, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(args.outdir, "metrics_pairwise_ranker.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_pairwise_ranker, handle, indent=2, ensure_ascii=False)
    if trace_weight_metadata is not None:
        with open(os.path.join(args.outdir, "trace_weighting.json"), "w", encoding="utf-8") as handle:
            json.dump(trace_weight_metadata, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(args.outdir, "dataset_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "datasets": [asdict(spec) for spec in DEFAULT_DATASETS],
                "rows_after_clean": int(len(df)),
                "keep_error_rows": bool(args.keep_error_rows),
                "source_summary": source_summary(df),
                "categorical_feature_cols": CATEGORICAL_FEATURE_COLS,
                "base_feature_cols": BASE_FEATURE_COLS,
                "p95_calibration_feature_cols": full_fit["p95_calibration_feature_cols"],
                "pairwise_rank_feature_cols": full_fit["pairwise_rank_feature_cols"],
                "trace_weighting": trace_weight_metadata,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps({"rows_after_clean": int(len(df)), "outdir": args.outdir}, ensure_ascii=False))
    print(json.dumps({
        "random_split_p95_raw": random_metrics["p95"],
        "random_split_p95_calibrated": random_metrics["p95_calibrated"],
        "random_split_tps": random_metrics["tps"],
    }, ensure_ascii=False))
    print(json.dumps({
        "full_fit_p95_raw": full_fit["p95_metrics"],
        "full_fit_p95_calibrated": full_fit["p95_calibrated_metrics"],
        "full_fit_pairwise_rank": full_fit["pairwise_rank_metrics"],
    }, ensure_ascii=False))
    print(json.dumps({
        "random_split_pairwise_rank": random_metrics["pairwise_rank"],
        "leave_one_group_out": logo_metrics,
        "saved_split_idx_rows": len(split_idx["train_idx"]) + len(split_idx["test_idx"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
