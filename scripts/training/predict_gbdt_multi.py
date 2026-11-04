import argparse
import json
import os
import sys
import types
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)


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

from tepqos.features import (
    build_feature_row,
    build_p95_calibration_row,
    known_values_from_manifest,
    load_json,
    parse_config_metadata,
)

DEFAULT_ARTIFACTS_DIR = "artifacts_gbdt_multi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict p95 latency and tok/s from the multi-device GBDT model.",
    )
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--quant-method", required=True)
    parser.add_argument("--prompt-len", required=True, type=int)
    parser.add_argument("--gen-len", required=True, type=int)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--config-id")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--kv-strategy")
    parser.add_argument("--quant-bits", type=int)
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir

    feature_cols = joblib.load(os.path.join(artifacts_dir, "feature_cols.joblib"))
    p95_model = joblib.load(os.path.join(artifacts_dir, "gbdt_p95.joblib"))
    tps_model = joblib.load(os.path.join(artifacts_dir, "gbdt_tps.joblib"))
    manifest = load_json(os.path.join(artifacts_dir, "dataset_manifest.json"))
    known_values = known_values_from_manifest(manifest)

    calibrator_path = os.path.join(artifacts_dir, "p95_calibrator.joblib")
    calibration_cols_path = os.path.join(artifacts_dir, "p95_calibration_feature_cols.joblib")
    p95_calibrator = joblib.load(calibrator_path) if os.path.exists(calibrator_path) else None
    p95_calibration_feature_cols = joblib.load(calibration_cols_path) if os.path.exists(calibration_cols_path) else None

    resolved_config = parse_config_metadata(
        config_id=args.config_id or "",
        quant_method=args.quant_method,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        kv_strategy=args.kv_strategy,
        quant_bits=args.quant_bits,
    )

    feature_row = build_feature_row(
        feature_cols=feature_cols,
        device_id=args.device_id,
        quant_method=args.quant_method,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        concurrency=args.concurrency,
        config_id=resolved_config["config_id"],
        batch_size=resolved_config["batch_size"],
        max_tokens=resolved_config["max_tokens"],
        kv_strategy=resolved_config["kv_strategy"],
        quant_bits=resolved_config["quant_bits"],
    )

    p95_raw_hat = float(np.expm1(p95_model.predict(feature_row))[0])
    p95_calibrated_hat = p95_raw_hat
    if p95_calibrator is not None and p95_calibration_feature_cols is not None:
        calibration_row = build_p95_calibration_row(
            calibration_feature_cols=p95_calibration_feature_cols,
            base_feature_cols=feature_cols,
            raw_p95_s=p95_raw_hat,
            device_id=args.device_id,
            quant_method=args.quant_method,
            prompt_len=args.prompt_len,
            gen_len=args.gen_len,
            concurrency=args.concurrency,
            config_id=resolved_config["config_id"],
            batch_size=resolved_config["batch_size"],
            max_tokens=resolved_config["max_tokens"],
            kv_strategy=resolved_config["kv_strategy"],
            quant_bits=resolved_config["quant_bits"],
        )
        residual_hat = float(p95_calibrator.predict(calibration_row)[0])
        p95_calibrated_hat = float(np.clip(np.expm1(np.log1p(max(p95_raw_hat, 1e-9)) + residual_hat), 1e-9, None))

    tps_hat = float(np.expm1(tps_model.predict(feature_row))[0])

    extrapolation = {
        "device_id_unseen": args.device_id not in known_values["device_ids"],
        "quant_method_unseen": resolved_config["quant_method"] not in known_values["quant_methods"],
        "config_id_unseen": resolved_config["config_id"] not in known_values["config_ids"],
        "kv_strategy_unseen": resolved_config["kv_strategy"] not in known_values["kv_strategies"],
    }

    result = {
        "input": {
            "device_id": args.device_id,
            "quant_method": resolved_config["quant_method"],
            "config_id": resolved_config["config_id"],
            "batch_size": resolved_config["batch_size"],
            "max_tokens": resolved_config["max_tokens"],
            "kv_strategy": resolved_config["kv_strategy"],
            "quant_bits": resolved_config["quant_bits"],
            "prompt_len": args.prompt_len,
            "gen_len": args.gen_len,
            "concurrency": args.concurrency,
        },
        "prediction": {
            "p95_s": p95_calibrated_hat,
            "p95_raw_s": p95_raw_hat,
            "p95_calibrated_s": p95_calibrated_hat,
            "p95_calibration_applied": bool(p95_calibrator is not None and p95_calibration_feature_cols is not None),
            "tok_per_s": tps_hat,
        },
        "known_training_values": known_values,
        "extrapolation": extrapolation,
    }

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()




