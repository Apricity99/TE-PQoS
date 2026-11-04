from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tepqos import training as train
from tepqos.features import TARGET_COLS, add_workload_features, build_feature_frame
from tepqos.model_baselines import metric_dict

DEFAULT_RUN_ROOT = ROOT / "outputs" / "runs" / "20260914_predictor_generalization_robustness_v1"
DEFAULT_REPORT_ROOT = ROOT / "outputs" / "reports" / "20260914_predictor_generalization_robustness_v1"

QWEN_DATASETS = [
    train.DatasetSpec(
        path="data/benchmarks/edge_3090/qwen_awq/edge_3090_qwen2_5_7b_awq4_bs8_tok2048_kvauto_valid.csv",
        source_tag="edge_3090_qwen2_5_awq_bs8_kvauto",
        device_id="edge_3090",
        quant_method="awq",
        config_id="qwen2_5_7b_awq4_bs8_tok2048_kvauto",
        model_family="qwen2_5_7b",
        quant_bits=4,
        batch_size=8,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
    train.DatasetSpec(
        path="data/benchmarks/edge_3090/qwen_awq/edge_3090_qwen2_5_7b_awq4_bs16_tok2048_kvauto_valid.csv",
        source_tag="edge_3090_qwen2_5_awq_bs16_kvauto",
        device_id="edge_3090",
        quant_method="awq",
        config_id="qwen2_5_7b_awq4_bs16_tok2048_kvauto",
        model_family="qwen2_5_7b",
        quant_bits=4,
        batch_size=16,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
    train.DatasetSpec(
        path="data/benchmarks/edge_4090/qwen_awq/edge_4090_qwen2_5_7b_awq4_bs8_tok2048_kvauto_valid.csv",
        source_tag="edge_4090_qwen2_5_awq_bs8_kvauto",
        device_id="edge_4090",
        quant_method="awq",
        config_id="qwen2_5_7b_awq4_bs8_tok2048_kvauto",
        model_family="qwen2_5_7b",
        quant_bits=4,
        batch_size=8,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
    train.DatasetSpec(
        path="data/benchmarks/edge_4090/qwen_awq/edge_4090_qwen2_5_7b_awq4_bs16_tok2048_kvauto_valid.csv",
        source_tag="edge_4090_qwen2_5_awq_bs16_kvauto",
        device_id="edge_4090",
        quant_method="awq",
        config_id="qwen2_5_7b_awq4_bs16_tok2048_kvauto",
        model_family="qwen2_5_7b",
        quant_bits=4,
        batch_size=16,
        max_tokens=2048,
        kv_strategy="kvauto",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predictor generalization and robustness for reviewer response.",
    )
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--wildchat-trace", default="data/workload_traces/wildchat_turn_trace_short_bucketed.csv")
    parser.add_argument("--burstgpt-trace", default="data/workload_traces/burstgpt_stratified500_bucketed.csv")
    parser.add_argument("--trace-weight-scope", default="prompt_gen", choices=["prompt_gen", "prompt_gen_concurrency"])
    parser.add_argument("--trace-weight-alpha", type=float, default=1.0)
    parser.add_argument("--trace-weight-min", type=float, default=0.25)
    parser.add_argument("--trace-weight-max", type=float, default=4.0)
    return parser.parse_args()


def prepare_qwen_dataset() -> pd.DataFrame:
    frames = [train.load_dataset(spec) for spec in QWEN_DATASETS]
    df = pd.concat(frames, ignore_index=True)
    for column in train.NUMERIC_CLEAN_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=TARGET_COLS + ["prompt_len", "gen_len", "concurrency"]).copy()
    df = df[(df["lat_p95_s"] > 0) & (df["tok_per_s"] > 0)].copy()
    has_error = pd.Series(False, index=df.index)
    if "error_reqs" in df.columns:
        has_error = has_error | (df["error_reqs"].fillna(0) > 0)
    if "timeout_reqs" in df.columns:
        has_error = has_error | (df["timeout_reqs"].fillna(0) > 0)
    df = df[~has_error].copy()
    return add_workload_features(df).reset_index(drop=True)


def length_bin(frame: pd.DataFrame) -> pd.Series:
    total = frame["prompt_len"].astype(int) + frame["gen_len"].astype(int)
    return pd.cut(
        total,
        bins=[0, 96, 192, 384, np.inf],
        labels=["short<=96", "medium<=192", "long<=384", "extra>384"],
        include_lowest=True,
    ).astype(str)


def prompt_bin(frame: pd.DataFrame) -> pd.Series:
    prompt = frame["prompt_len"].astype(int)
    return pd.cut(
        prompt,
        bins=[0, 48, 96, 192, np.inf],
        labels=["p<=48", "p<=96", "p<=192", "p>192"],
        include_lowest=True,
    ).astype(str)


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_weight_train: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    x_train = build_feature_frame(train_df).reset_index(drop=True)
    x_test = train.align_columns(build_feature_frame(test_df).reset_index(drop=True), list(x_train.columns))
    y_p95_train = train_df["lat_p95_s"].to_numpy()
    y_tps_train = train_df["tok_per_s"].to_numpy()
    fit_result = train.fit_predict_pair(
        X_train=x_train,
        X_test=x_test,
        y_p95_train=y_p95_train,
        y_tps_train=y_tps_train,
        sample_weight_train=sample_weight_train,
    )
    raw_p95_train = np.expm1(fit_result["p95_model"].predict(x_train))
    calibration = train.fit_p95_calibrator(
        X_train_base=x_train,
        X_test_base=x_test,
        y_p95_train=y_p95_train,
        raw_p95_train=raw_p95_train,
        raw_p95_test=fit_result["p95_pred"],
        sample_weight_train=sample_weight_train,
    )
    return {
        "p95_raw": np.asarray(fit_result["p95_pred"], dtype=float),
        "p95_calibrated": np.asarray(calibration["p95_pred"], dtype=float),
        "tps": np.asarray(fit_result["tps_pred"], dtype=float),
    }


def append_metric_rows(
    rows: List[Dict[str, Any]],
    *,
    dataset: str,
    protocol: str,
    held_out: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preds: Dict[str, np.ndarray],
    sample_weight_test: Optional[np.ndarray] = None,
) -> None:
    target_specs = [
        ("p95 latency", "calibrated", test_df["lat_p95_s"].to_numpy(), preds["p95_calibrated"]),
        ("p95 latency", "raw", test_df["lat_p95_s"].to_numpy(), preds["p95_raw"]),
        ("throughput", "raw", test_df["tok_per_s"].to_numpy(), preds["tps"]),
    ]
    for target, variant, y_true, y_pred in target_specs:
        metrics = metric_dict(y_true, y_pred, sample_weight_test)
        rows.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "held_out": held_out,
                "target": target,
                "variant": variant,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                **metrics,
            }
        )


def evaluate_group_holdout(
    rows: List[Dict[str, Any]],
    *,
    dataset: str,
    df: pd.DataFrame,
    group_col: str,
    protocol: str,
    min_train_groups: int = 1,
) -> None:
    groups = sorted(str(value) for value in df[group_col].dropna().unique().tolist())
    if len(groups) <= min_train_groups:
        return
    for group in groups:
        test_mask = df[group_col].astype(str) == group
        train_mask = ~test_mask
        train_df = df.loc[train_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue
        preds = fit_predict(train_df, test_df)
        append_metric_rows(
            rows,
            dataset=dataset,
            protocol=protocol,
            held_out=group,
            train_df=train_df,
            test_df=test_df,
            preds=preds,
        )


def evaluate_random_split(
    rows: List[Dict[str, Any]],
    *,
    dataset: str,
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    sample_weight: Optional[np.ndarray] = None,
    protocol: str = "random_split",
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=df["source_tag"].to_numpy() if df["source_tag"].nunique() > 1 else None,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    preds = fit_predict(
        train_df,
        test_df,
        sample_weight_train=sample_weight[train_idx] if sample_weight is not None else None,
    )
    append_metric_rows(
        rows,
        dataset=dataset,
        protocol=protocol,
        held_out="random_20pct",
        train_df=train_df,
        test_df=test_df,
        preds=preds,
        sample_weight_test=sample_weight[test_idx] if sample_weight is not None else None,
    )
    return train_idx, test_idx


def add_request_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["length_bin"] = length_bin(out)
    out["prompt_bin"] = prompt_bin(out)
    out["concurrency_group"] = out["concurrency"].astype(int).map(lambda value: f"c{value}")
    return out


def aggregate_summary(rows: pd.DataFrame) -> pd.DataFrame:
    keep = rows[rows["variant"].isin(["calibrated", "raw"])].copy()
    keep = keep[
        ((keep["target"] == "p95 latency") & (keep["variant"] == "calibrated"))
        | ((keep["target"] == "throughput") & (keep["variant"] == "raw"))
    ].copy()
    return (
        keep.groupby(["dataset", "protocol", "target"], as_index=False)
        .agg(
            folds=("held_out", "nunique"),
            test_rows=("test_rows", "sum"),
            mae=("mae", "mean"),
            r2=("r2", "mean"),
            mape_pct=("mape_pct", "mean"),
        )
        .sort_values(["dataset", "protocol", "target"])
    )


def fmt_metric(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_report(
    *,
    report_root: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: Dict[str, Any],
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Predictor Generalization and Robustness")
    lines.append("")
    lines.append("This report evaluates the predictor itself, separate from the scheduler.")
    lines.append("")
    lines.append("## Protocol Summary")
    lines.append("")
    lines.append("- Random split measures interpolation within the profiled configuration space.")
    lines.append("- LOGO holds out one device/configuration source group.")
    lines.append("- Device, quantization, request-length, prompt-length, and concurrency holdouts test structured shifts.")
    lines.append("- Qwen random/LOGO evaluates portability after refreshed profiling on a second 7B model family.")
    lines.append("- Llama-to-Qwen zero-shot is reported only as a boundary check, not as the claimed deployment mode.")
    lines.append("- WildChat and BurstGPT weighted splits measure robustness under shifted request distributions.")
    lines.append("")
    lines.append("## Aggregated Results")
    lines.append("")
    lines.append("| Dataset | Protocol | Target | Folds | Test rows | MAE | R2 | MAPE % |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in summary.to_dict(orient="records"):
        lines.append(
            "| {dataset} | {protocol} | {target} | {folds} | {test_rows} | {mae} | {r2} | {mape} |".format(
                dataset=row["dataset"],
                protocol=row["protocol"],
                target=row["target"],
                folds=int(row["folds"]),
                test_rows=int(row["test_rows"]),
                mae=fmt_metric(float(row["mae"])),
                r2=fmt_metric(float(row["r2"])),
                mape=fmt_metric(float(row["mape_pct"])),
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The strongest supported claim is not zero-shot transfer to an arbitrary unseen LLM. "
        "The supported claim is that the same predictor family remains usable across random splits, "
        "held-out source groups, structured request/configuration shifts, and after refreshed profiling "
        "on Qwen2.5-7B-Instruct-AWQ."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Full metrics CSV: `{metadata['metrics_csv']}`")
    lines.append(f"- Aggregated summary CSV: `{metadata['summary_csv']}`")
    lines.append(f"- Metadata JSON: `{metadata['metadata_json']}`")
    lines.append("")
    (report_root / "PREDICTOR_GENERALIZATION_ROBUSTNESS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    report_root = Path(args.report_root)
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    llama_df = add_request_bins(train.prepare_dataset(keep_error_rows=False))
    qwen_df = add_request_bins(prepare_qwen_dataset())
    all_rows: List[Dict[str, Any]] = []

    evaluate_random_split(
        all_rows,
        dataset="Llama-2-7B",
        df=llama_df,
        test_size=args.test_size,
        random_state=args.random_state,
        protocol="random_split",
    )
    evaluate_group_holdout(
        all_rows,
        dataset="Llama-2-7B",
        df=llama_df,
        group_col="source_tag",
        protocol="LOGO_source_group",
    )
    for group_col, protocol in [
        ("device_id", "leave_one_device"),
        ("quant_method", "leave_one_quantization"),
        ("length_bin", "leave_one_request_length_bin"),
        ("prompt_bin", "leave_one_prompt_length_bin"),
        ("concurrency_group", "leave_one_concurrency"),
    ]:
        evaluate_group_holdout(
            all_rows,
            dataset="Llama-2-7B",
            df=llama_df,
            group_col=group_col,
            protocol=protocol,
        )
    primary_device_df = llama_df[llama_df["device_id"].isin(["edge_3090", "edge_4090"])].reset_index(drop=True)
    evaluate_group_holdout(
        all_rows,
        dataset="Llama-2-7B",
        df=primary_device_df,
        group_col="device_id",
        protocol="leave_one_primary_device",
    )

    for trace_name, trace_csv in [
        ("WildChat_weighted", args.wildchat_trace),
        ("BurstGPT_weighted", args.burstgpt_trace),
    ]:
        weights, _ = train.build_trace_sample_weights(
            llama_df,
            trace_csv=str(ROOT / trace_csv),
            scope=args.trace_weight_scope,
            alpha=args.trace_weight_alpha,
            min_weight=args.trace_weight_min,
            max_weight=args.trace_weight_max,
        )
        evaluate_random_split(
            all_rows,
            dataset="Llama-2-7B",
            df=llama_df,
            test_size=args.test_size,
            random_state=args.random_state,
            sample_weight=weights,
            protocol=trace_name,
        )

    evaluate_random_split(
        all_rows,
        dataset="Qwen2.5-7B",
        df=qwen_df,
        test_size=args.test_size,
        random_state=args.random_state,
        protocol="random_split_refreshed_profile",
    )
    evaluate_group_holdout(
        all_rows,
        dataset="Qwen2.5-7B",
        df=qwen_df,
        group_col="source_tag",
        protocol="LOGO_source_group_refreshed_profile",
    )

    zero_shot_preds = fit_predict(llama_df, qwen_df)
    append_metric_rows(
        all_rows,
        dataset="Qwen2.5-7B",
        protocol="llama_to_qwen_zero_shot_boundary",
        held_out="all_qwen",
        train_df=llama_df,
        test_df=qwen_df,
        preds=zero_shot_preds,
    )

    metrics = pd.DataFrame(all_rows)
    summary = aggregate_summary(metrics)
    metrics_csv = run_root / "predictor_generalization_metrics.csv"
    summary_csv = run_root / "predictor_generalization_summary.csv"
    metadata_json = run_root / "predictor_generalization_metadata.json"
    metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    metadata = {
        "run_root": str(run_root),
        "report_root": str(report_root),
        "llama_rows": int(len(llama_df)),
        "qwen_rows": int(len(qwen_df)),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "wildchat_trace": args.wildchat_trace,
        "burstgpt_trace": args.burstgpt_trace,
        "metrics_csv": str(metrics_csv),
        "summary_csv": str(summary_csv),
        "metadata_json": str(metadata_json),
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_root=report_root, metrics=metrics, summary=summary, metadata=metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
