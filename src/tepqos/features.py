import json
import re
from typing import Any, Dict, List, Optional

import numpy as np

BASE_FEATURE_COLS = [
    "prompt_len",
    "gen_len",
    "concurrency",
    "total_tokens",
    "decode_work",
    "prefill_work",
    "total_work",
    "ratio_decode",
    "batch_size",
    "max_tokens",
    "quant_bits",
    "batch_pressure",
    "token_pressure",
]
CATEGORICAL_FEATURE_COLS = [
    "device_id",
    "quant_method",
    "kv_strategy",
    "config_id",
]
PAIRWISE_NUMERIC_FEATURE_COLS = [
    "prompt_len",
    "gen_len",
    "concurrency",
    "total_tokens",
    "decode_work",
    "prefill_work",
    "total_work",
    "ratio_decode",
    "left_predicted_p95_s",
    "right_predicted_p95_s",
    "predicted_p95_diff",
    "predicted_p95_abs_diff",
    "predicted_p95_ratio",
    "left_predicted_tok_per_s",
    "right_predicted_tok_per_s",
    "predicted_tps_diff",
    "predicted_tps_abs_diff",
    "predicted_tps_ratio",
    "left_predicted_rental_cost",
    "right_predicted_rental_cost",
    "predicted_cost_diff",
    "predicted_cost_abs_diff",
    "left_batch_size",
    "right_batch_size",
    "batch_size_diff",
    "left_max_tokens",
    "right_max_tokens",
    "max_tokens_diff",
    "left_quant_bits",
    "right_quant_bits",
    "quant_bits_diff",
    "left_batch_pressure",
    "right_batch_pressure",
    "batch_pressure_diff",
    "left_token_pressure",
    "right_token_pressure",
    "token_pressure_diff",
    "left_estimated_memory_mb",
    "right_estimated_memory_mb",
    "estimated_memory_diff_mb",
    "left_p95_calibration_delta_s",
    "right_p95_calibration_delta_s",
    "p95_calibration_delta_diff",
]
PAIRWISE_CATEGORICAL_FEATURE_COLS = [
    "device_id",
    "left_config_id",
    "right_config_id",
    "left_quant_method",
    "right_quant_method",
    "left_kv_strategy",
    "right_kv_strategy",
    "ordered_config_pair",
]
TARGET_COLS = [
    "lat_p95_s",
    "tok_per_s",
]
NUMERIC_CLEAN_COLS = [
    "prompt_len",
    "gen_len",
    "concurrency",
    "lat_p95_s",
    "tok_per_s",
    "success_reqs",
    "error_reqs",
    "timeout_reqs",
    "batch_size",
    "max_tokens",
    "quant_bits",
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def normalize_model_family(raw_value: str) -> str:
    raw = str(raw_value or "").strip().lower()
    if "llama-2-7b" in raw or "llama2-7b" in raw or "llama2_7b" in raw:
        return "llama2_7b"
    return raw or "unknown"


def infer_quant_method(*values: str) -> str:
    joined = " ".join(str(value or "") for value in values).lower()
    if "gptq" in joined:
        return "gptq"
    if "awq" in joined:
        return "awq"
    if "fp16" in joined:
        return "fp16"
    if "fp8" in joined:
        return "fp8"
    if "int8" in joined:
        return "int8"
    return "unknown"


def fill_or_default(frame, column: str, default_value: object):
    import pandas as pd

    if column not in frame.columns:
        return pd.Series([default_value] * len(frame), index=frame.index)
    series = frame[column]
    return series.fillna(default_value).replace("", default_value)


def _extract_int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))


def _default_quant_bits(quant_method: str) -> int:
    if quant_method in {"awq", "gptq"}:
        return 4
    if quant_method == "fp16":
        return 16
    if quant_method in {"fp8", "int8"}:
        return 8
    return 0


def _default_batch_size(quant_method: str) -> int:
    if quant_method in {"awq", "gptq", "fp16"}:
        return 16
    return -1


def _default_max_tokens(quant_method: str) -> int:
    if quant_method in {"awq", "gptq", "fp16"}:
        return 2048
    return 0


def _default_kv_strategy(quant_method: str) -> str:
    if quant_method in {"awq", "gptq"}:
        return "kvauto"
    if quant_method == "fp16":
        return "dense"
    return "unknown"


def _normalize_kv_strategy(raw_value: str) -> str:
    raw = str(raw_value or "").strip().lower()
    if not raw:
        return "unknown"
    if "prefixon" in raw:
        return "prefixon"
    if "kvauto" in raw:
        return "kvauto"
    if re.search(r"(?:^|_)auto(?:$|_)", raw):
        return "kvauto"
    if "dense" in raw:
        return "dense"
    if "fp8" in raw:
        return "kvfp8"
    return raw


def synthesize_config_id(
    quant_method: str,
    quant_bits: int,
    batch_size: int,
    max_tokens: int,
    kv_strategy: str,
) -> str:
    if batch_size > 0 and max_tokens > 0 and quant_bits > 0 and kv_strategy not in {"", "unknown"}:
        return f"{quant_method}{quant_bits}_bs{batch_size}_tok{max_tokens}_{kv_strategy}"
    return f"{quant_method or 'unknown'}_custom"


def parse_config_metadata(
    *,
    config_id: str,
    quant_method: str = "",
    quant_bits: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    kv_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    raw_config_id = str(config_id or "").strip()
    inferred_quant_method = infer_quant_method(raw_config_id, quant_method)
    joined = f"{raw_config_id} {quant_method}".lower()

    resolved_quant_bits = quant_bits
    if resolved_quant_bits is None:
        resolved_quant_bits = _extract_int(r"(?:awq|gptq|fp|int)(\d+)", joined)
    if resolved_quant_bits is None:
        resolved_quant_bits = _default_quant_bits(inferred_quant_method)

    resolved_batch_size = batch_size
    if resolved_batch_size is None:
        resolved_batch_size = _extract_int(r"_bs(\d+)", joined)
    if resolved_batch_size is None:
        resolved_batch_size = _default_batch_size(inferred_quant_method)

    resolved_max_tokens = max_tokens
    if resolved_max_tokens is None:
        resolved_max_tokens = _extract_int(r"_tok(\d+)", joined)
    if resolved_max_tokens is None:
        resolved_max_tokens = _extract_int(r"len(\d+)", joined)
    if resolved_max_tokens is None:
        resolved_max_tokens = _default_max_tokens(inferred_quant_method)

    resolved_kv_strategy = _normalize_kv_strategy(kv_strategy or raw_config_id)
    if resolved_kv_strategy == "unknown":
        resolved_kv_strategy = _default_kv_strategy(inferred_quant_method)

    resolved_config_id = raw_config_id or synthesize_config_id(
        quant_method=inferred_quant_method,
        quant_bits=int(resolved_quant_bits or 0),
        batch_size=int(resolved_batch_size or -1),
        max_tokens=int(resolved_max_tokens or 0),
        kv_strategy=resolved_kv_strategy,
    )

    return {
        "config_id": resolved_config_id,
        "quant_method": inferred_quant_method,
        "quant_bits": int(resolved_quant_bits or 0),
        "batch_size": int(resolved_batch_size if resolved_batch_size is not None else -1),
        "max_tokens": int(resolved_max_tokens or 0),
        "kv_strategy": resolved_kv_strategy,
    }


def add_workload_features(frame):
    df = frame.copy()
    df["total_tokens"] = df["prompt_len"] + df["gen_len"]
    df["decode_work"] = df["gen_len"] * df["concurrency"]
    df["prefill_work"] = df["prompt_len"] * df["concurrency"]
    df["total_work"] = (df["prompt_len"] + df["gen_len"]) * df["concurrency"]
    df["ratio_decode"] = df["gen_len"] / (df["prompt_len"] + 1e-6)
    df["batch_pressure"] = 0.0
    batch_mask = df["batch_size"] > 0
    df.loc[batch_mask, "batch_pressure"] = df.loc[batch_mask, "concurrency"] / df.loc[batch_mask, "batch_size"]
    df["token_pressure"] = 0.0
    token_mask = df["max_tokens"] > 0
    df.loc[token_mask, "token_pressure"] = df.loc[token_mask, "total_tokens"] / df.loc[token_mask, "max_tokens"]
    return df


def build_feature_frame(df):
    import pandas as pd

    base = df[BASE_FEATURE_COLS].copy()
    categorical = pd.get_dummies(
        df[CATEGORICAL_FEATURE_COLS].astype(str),
        prefix=CATEGORICAL_FEATURE_COLS,
        dtype=np.int8,
    )
    return pd.concat([base, categorical], axis=1)


def build_feature_row(
    *,
    feature_cols: List[str],
    device_id: str,
    quant_method: str,
    prompt_len: int,
    gen_len: int,
    concurrency: int,
    config_id: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    kv_strategy: Optional[str] = None,
    quant_bits: Optional[int] = None,
) -> np.ndarray:
    config = parse_config_metadata(
        config_id=config_id or "",
        quant_method=quant_method,
        quant_bits=quant_bits,
        batch_size=batch_size,
        max_tokens=max_tokens,
        kv_strategy=kv_strategy,
    )

    row: Dict[str, float] = {column: 0.0 for column in feature_cols}
    row["prompt_len"] = float(prompt_len)
    row["gen_len"] = float(gen_len)
    row["concurrency"] = float(concurrency)
    row["total_tokens"] = float(prompt_len + gen_len)
    row["decode_work"] = float(gen_len * concurrency)
    row["prefill_work"] = float(prompt_len * concurrency)
    row["total_work"] = float((prompt_len + gen_len) * concurrency)
    row["ratio_decode"] = float(gen_len / (prompt_len + 1e-6))
    row["batch_size"] = float(config["batch_size"])
    row["max_tokens"] = float(config["max_tokens"])
    row["quant_bits"] = float(config["quant_bits"])
    row["batch_pressure"] = float(concurrency / config["batch_size"]) if config["batch_size"] > 0 else 0.0
    row["token_pressure"] = float((prompt_len + gen_len) / config["max_tokens"]) if config["max_tokens"] > 0 else 0.0

    for prefix, value in [
        ("device_id", device_id),
        ("quant_method", config["quant_method"]),
        ("kv_strategy", config["kv_strategy"]),
        ("config_id", config["config_id"]),
    ]:
        feature_name = f"{prefix}_{value}"
        if feature_name in row:
            row[feature_name] = 1.0

    return np.array([[row[column] for column in feature_cols]], dtype=float)


def build_p95_calibration_frame(base_frame, raw_p95_s):
    import pandas as pd

    calibration = base_frame.copy()
    calibration = calibration.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    calibration.insert(
        0,
        "raw_p95_log",
        np.log1p(np.clip(np.asarray(raw_p95_s, dtype=float), 1e-9, None)),
    )
    return calibration


def build_p95_calibration_row(
    *,
    calibration_feature_cols: List[str],
    base_feature_cols: List[str],
    raw_p95_s: float,
    device_id: str,
    quant_method: str,
    prompt_len: int,
    gen_len: int,
    concurrency: int,
    config_id: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    kv_strategy: Optional[str] = None,
    quant_bits: Optional[int] = None,
) -> np.ndarray:
    base_row = build_feature_row(
        feature_cols=base_feature_cols,
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
    )[0]

    row: Dict[str, float] = {column: 0.0 for column in calibration_feature_cols}
    row["raw_p95_log"] = float(np.log1p(max(float(raw_p95_s), 1e-9)))

    for idx, column in enumerate(base_feature_cols):
        if column in row:
            row[column] = float(base_row[idx])

    return np.array([[row[column] for column in calibration_feature_cols]], dtype=float)



def build_pairwise_feature_record(
    *,
    device_id: str,
    prompt_len: int,
    gen_len: int,
    concurrency: int,
    left_config_id: str,
    right_config_id: str,
    left_quant_method: str,
    right_quant_method: str,
    left_kv_strategy: str,
    right_kv_strategy: str,
    left_quant_bits: Optional[int],
    right_quant_bits: Optional[int],
    left_batch_size: Optional[int],
    right_batch_size: Optional[int],
    left_max_tokens: Optional[int],
    right_max_tokens: Optional[int],
    left_predicted_p95_s: float,
    right_predicted_p95_s: float,
    left_predicted_tok_per_s: float,
    right_predicted_tok_per_s: float,
    left_predicted_rental_cost: float,
    right_predicted_rental_cost: float,
    left_estimated_memory_mb: Optional[float] = None,
    right_estimated_memory_mb: Optional[float] = None,
    left_p95_calibration_delta_s: float = 0.0,
    right_p95_calibration_delta_s: float = 0.0,
) -> Dict[str, Any]:
    left_batch_size = int(left_batch_size or 0)
    right_batch_size = int(right_batch_size or 0)
    left_max_tokens = int(left_max_tokens or 0)
    right_max_tokens = int(right_max_tokens or 0)
    left_quant_bits = int(left_quant_bits or 0)
    right_quant_bits = int(right_quant_bits or 0)
    total_tokens = int(prompt_len + gen_len)
    left_batch_pressure = float(concurrency / left_batch_size) if left_batch_size > 0 else 0.0
    right_batch_pressure = float(concurrency / right_batch_size) if right_batch_size > 0 else 0.0
    left_token_pressure = float(total_tokens / left_max_tokens) if left_max_tokens > 0 else 0.0
    right_token_pressure = float(total_tokens / right_max_tokens) if right_max_tokens > 0 else 0.0

    return {
        "device_id": str(device_id),
        "prompt_len": float(prompt_len),
        "gen_len": float(gen_len),
        "concurrency": float(concurrency),
        "total_tokens": float(total_tokens),
        "decode_work": float(gen_len * concurrency),
        "prefill_work": float(prompt_len * concurrency),
        "total_work": float(total_tokens * concurrency),
        "ratio_decode": float(gen_len / (prompt_len + 1e-6)),
        "left_predicted_p95_s": float(left_predicted_p95_s),
        "right_predicted_p95_s": float(right_predicted_p95_s),
        "predicted_p95_diff": float(left_predicted_p95_s - right_predicted_p95_s),
        "predicted_p95_abs_diff": float(abs(left_predicted_p95_s - right_predicted_p95_s)),
        "predicted_p95_ratio": float(left_predicted_p95_s / max(right_predicted_p95_s, 1e-6)),
        "left_predicted_tok_per_s": float(left_predicted_tok_per_s),
        "right_predicted_tok_per_s": float(right_predicted_tok_per_s),
        "predicted_tps_diff": float(left_predicted_tok_per_s - right_predicted_tok_per_s),
        "predicted_tps_abs_diff": float(abs(left_predicted_tok_per_s - right_predicted_tok_per_s)),
        "predicted_tps_ratio": float(left_predicted_tok_per_s / max(right_predicted_tok_per_s, 1e-6)),
        "left_predicted_rental_cost": float(left_predicted_rental_cost),
        "right_predicted_rental_cost": float(right_predicted_rental_cost),
        "predicted_cost_diff": float(left_predicted_rental_cost - right_predicted_rental_cost),
        "predicted_cost_abs_diff": float(abs(left_predicted_rental_cost - right_predicted_rental_cost)),
        "left_batch_size": float(left_batch_size),
        "right_batch_size": float(right_batch_size),
        "batch_size_diff": float(left_batch_size - right_batch_size),
        "left_max_tokens": float(left_max_tokens),
        "right_max_tokens": float(right_max_tokens),
        "max_tokens_diff": float(left_max_tokens - right_max_tokens),
        "left_quant_bits": float(left_quant_bits),
        "right_quant_bits": float(right_quant_bits),
        "quant_bits_diff": float(left_quant_bits - right_quant_bits),
        "left_batch_pressure": float(left_batch_pressure),
        "right_batch_pressure": float(right_batch_pressure),
        "batch_pressure_diff": float(left_batch_pressure - right_batch_pressure),
        "left_token_pressure": float(left_token_pressure),
        "right_token_pressure": float(right_token_pressure),
        "token_pressure_diff": float(left_token_pressure - right_token_pressure),
        "left_estimated_memory_mb": float(left_estimated_memory_mb or 0.0),
        "right_estimated_memory_mb": float(right_estimated_memory_mb or 0.0),
        "estimated_memory_diff_mb": float((left_estimated_memory_mb or 0.0) - (right_estimated_memory_mb or 0.0)),
        "left_p95_calibration_delta_s": float(left_p95_calibration_delta_s),
        "right_p95_calibration_delta_s": float(right_p95_calibration_delta_s),
        "p95_calibration_delta_diff": float(left_p95_calibration_delta_s - right_p95_calibration_delta_s),
        "left_config_id": str(left_config_id),
        "right_config_id": str(right_config_id),
        "left_quant_method": str(left_quant_method),
        "right_quant_method": str(right_quant_method),
        "left_kv_strategy": str(left_kv_strategy),
        "right_kv_strategy": str(right_kv_strategy),
        "ordered_config_pair": f"{left_config_id}__vs__{right_config_id}",
    }


def build_pairwise_feature_frame(records):
    import pandas as pd

    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame(list(records))

    base = frame[PAIRWISE_NUMERIC_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    categorical = pd.get_dummies(
        frame[PAIRWISE_CATEGORICAL_FEATURE_COLS].astype(str),
        prefix=PAIRWISE_CATEGORICAL_FEATURE_COLS,
        dtype=np.int8,
    )
    return pd.concat([base, categorical], axis=1)


def build_pairwise_feature_row(
    *,
    feature_cols: List[str],
    device_id: str,
    prompt_len: int,
    gen_len: int,
    concurrency: int,
    left_candidate: Dict[str, Any],
    right_candidate: Dict[str, Any],
) -> np.ndarray:
    record = build_pairwise_feature_record(
        device_id=device_id,
        prompt_len=prompt_len,
        gen_len=gen_len,
        concurrency=concurrency,
        left_config_id=str(left_candidate.get("config_id", "")),
        right_config_id=str(right_candidate.get("config_id", "")),
        left_quant_method=str(left_candidate.get("quant_method", "unknown")),
        right_quant_method=str(right_candidate.get("quant_method", "unknown")),
        left_kv_strategy=str(left_candidate.get("kv_strategy", "unknown")),
        right_kv_strategy=str(right_candidate.get("kv_strategy", "unknown")),
        left_quant_bits=left_candidate.get("quant_bits"),
        right_quant_bits=right_candidate.get("quant_bits"),
        left_batch_size=left_candidate.get("batch_size"),
        right_batch_size=right_candidate.get("batch_size"),
        left_max_tokens=left_candidate.get("max_tokens"),
        right_max_tokens=right_candidate.get("max_tokens"),
        left_predicted_p95_s=float(left_candidate.get("predicted_p95_s", np.nan)),
        right_predicted_p95_s=float(right_candidate.get("predicted_p95_s", np.nan)),
        left_predicted_tok_per_s=float(left_candidate.get("predicted_tok_per_s", np.nan)),
        right_predicted_tok_per_s=float(right_candidate.get("predicted_tok_per_s", np.nan)),
        left_predicted_rental_cost=float(left_candidate.get("predicted_rental_cost", np.nan)),
        right_predicted_rental_cost=float(right_candidate.get("predicted_rental_cost", np.nan)),
        left_estimated_memory_mb=left_candidate.get("estimated_memory_mb"),
        right_estimated_memory_mb=right_candidate.get("estimated_memory_mb"),
        left_p95_calibration_delta_s=float(left_candidate.get("p95_calibration_delta_s", 0.0) or 0.0),
        right_p95_calibration_delta_s=float(right_candidate.get("p95_calibration_delta_s", 0.0) or 0.0),
    )
    row: Dict[str, float] = {column: 0.0 for column in feature_cols}
    for column in PAIRWISE_NUMERIC_FEATURE_COLS:
        if column in row:
            row[column] = float(record.get(column, 0.0) or 0.0)

    for prefix, value in [
        ("device_id", record["device_id"]),
        ("left_config_id", record["left_config_id"]),
        ("right_config_id", record["right_config_id"]),
        ("left_quant_method", record["left_quant_method"]),
        ("right_quant_method", record["right_quant_method"]),
        ("left_kv_strategy", record["left_kv_strategy"]),
        ("right_kv_strategy", record["right_kv_strategy"]),
        ("ordered_config_pair", record["ordered_config_pair"]),
    ]:
        feature_name = f"{prefix}_{value}"
        if feature_name in row:
            row[feature_name] = 1.0

    return np.array([[row[column] for column in feature_cols]], dtype=float)


def known_values_from_manifest(manifest: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        "device_ids": sorted({device for item in manifest.get("source_summary", []) for device in item.get("device_id", [])}),
        "quant_methods": sorted({quant for item in manifest.get("source_summary", []) for quant in item.get("quant_method", [])}),
        "config_ids": sorted({config for item in manifest.get("source_summary", []) for config in item.get("config_id", [])}),
        "kv_strategies": sorted({kv for item in manifest.get("source_summary", []) for kv in item.get("kv_strategy", [])}),
    }

