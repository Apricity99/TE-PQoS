import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_feature_frame
from .training import (
    align_columns,
    build_trace_sample_weights,
    fit_p95_calibrator,
    make_regressor,
    prepare_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare prediction model baselines on the same profiling features and split.",
    )
    parser.add_argument("--outdir", default="final/prediction_model_baselines_v1")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trace-weight-csv", default="data/workload_traces/wildchat_turn_trace_short_bucketed.csv")
    parser.add_argument("--trace-weight-scope", default="prompt_gen", choices=["prompt_gen", "prompt_gen_concurrency"])
    parser.add_argument("--trace-weight-alpha", type=float, default=1.0)
    parser.add_argument("--trace-weight-min", type=float, default=0.25)
    parser.add_argument("--trace-weight-max", type=float, default=4.0)
    return parser.parse_args()


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]) -> Dict[str, float]:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    y_true = np.asarray(y_true, dtype=float)
    abs_pct = np.abs((y_true - y_pred) / y_true) * 100.0
    if sample_weight is None:
        mape = float(np.mean(abs_pct))
    else:
        mape = float(np.average(abs_pct, weights=sample_weight))
    return {
        "mse": float(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)),
        "mae": float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)),
        "r2": float(r2_score(y_true, y_pred, sample_weight=sample_weight)),
        "mape_pct": mape,
    }


def fit_log_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight_train: Optional[np.ndarray],
) -> Tuple[object, np.ndarray, np.ndarray]:
    fitted = clone(model)
    fit_kwargs = {}
    if sample_weight_train is not None:
        if hasattr(fitted, "steps") and fitted.steps:
            fit_kwargs[f"{fitted.steps[-1][0]}__sample_weight"] = sample_weight_train
        else:
            fit_kwargs["sample_weight"] = sample_weight_train
    try:
        fitted.fit(X_train, np.log1p(y_train), **fit_kwargs)
    except (TypeError, ValueError):
        fitted.fit(X_train, np.log1p(y_train))
    train_pred = np.expm1(fitted.predict(X_train))
    test_pred = np.expm1(fitted.predict(X_test))
    return fitted, np.clip(train_pred, 1e-9, None), np.clip(test_pred, 1e-9, None)


def baseline_models() -> Dict[str, object]:
    return {
        "Mean": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "HistGBDT(raw)": make_regressor(),
    }


def evaluate_dataset(
    *,
    dataset_name: str,
    df: pd.DataFrame,
    sample_weight: Optional[np.ndarray],
    test_size: float,
    random_state: int,
) -> List[Dict[str, object]]:
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
    p95_train = y_p95[train_idx]
    p95_test = y_p95[test_idx]
    tps_train = y_tps[train_idx]
    tps_test = y_tps[test_idx]
    weight_train = sample_weight[train_idx] if sample_weight is not None else None
    weight_test = sample_weight[test_idx] if sample_weight is not None else None

    rows: List[Dict[str, object]] = []
    for model_name, model in baseline_models().items():
        _, p95_train_pred, p95_test_pred = fit_log_model(model, X_train, X_test, p95_train, weight_train)
        _, _, tps_test_pred = fit_log_model(model, X_train, X_test, tps_train, weight_train)
        p95_metrics = metric_dict(p95_test, p95_test_pred, weight_test)
        tps_metrics = metric_dict(tps_test, tps_test_pred, weight_test)
        rows.append(
            {
                "dataset": dataset_name,
                "protocol": "Random split",
                "model": model_name,
                "target": "p95 latency",
                **p95_metrics,
            }
        )
        rows.append(
            {
                "dataset": dataset_name,
                "protocol": "Random split",
                "model": model_name,
                "target": "throughput",
                **tps_metrics,
            }
        )

        if model_name == "HistGBDT(raw)":
            calibration = fit_p95_calibrator(
                X_train_base=X_train,
                X_test_base=X_test,
                y_p95_train=p95_train,
                raw_p95_train=p95_train_pred,
                raw_p95_test=p95_test_pred,
                sample_weight_train=weight_train,
            )
            cal_metrics = metric_dict(p95_test, calibration["p95_pred"], weight_test)
            rows.append(
                {
                    "dataset": dataset_name,
                    "protocol": "Random split",
                    "model": "Ours (HistGBDT+cal)",
                    "target": "p95 latency",
                    **cal_metrics,
                }
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "protocol": "Random split",
                    "model": "Ours (HistGBDT+cal)",
                    "target": "throughput",
                    **tps_metrics,
                }
            )

    return rows


def evaluate_logo_dataset(
    *,
    dataset_name: str,
    df: pd.DataFrame,
    sample_weight: Optional[np.ndarray],
) -> List[Dict[str, object]]:
    X_full = build_feature_frame(df)
    y_p95 = df["lat_p95_s"].to_numpy()
    y_tps = df["tok_per_s"].to_numpy()
    rows: List[Dict[str, object]] = []

    for model_name, model in baseline_models().items():
        p95_true_all: List[float] = []
        p95_pred_all: List[float] = []
        p95_cal_pred_all: List[float] = []
        tps_true_all: List[float] = []
        tps_pred_all: List[float] = []
        weight_all: List[float] = []

        for held_out in sorted(df["source_tag"].unique().tolist()):
            test_mask = (df["source_tag"] == held_out).to_numpy()
            train_mask = ~test_mask
            X_train = X_full.loc[train_mask].reset_index(drop=True)
            X_test = align_columns(X_full.loc[test_mask].reset_index(drop=True), list(X_train.columns))
            weight_train = sample_weight[train_mask] if sample_weight is not None else None
            weight_test = sample_weight[test_mask] if sample_weight is not None else None

            _, p95_train_pred, p95_test_pred = fit_log_model(model, X_train, X_test, y_p95[train_mask], weight_train)
            _, _, tps_test_pred = fit_log_model(model, X_train, X_test, y_tps[train_mask], weight_train)

            p95_true_all.extend(y_p95[test_mask].tolist())
            p95_pred_all.extend(p95_test_pred.tolist())
            tps_true_all.extend(y_tps[test_mask].tolist())
            tps_pred_all.extend(tps_test_pred.tolist())
            if weight_test is None:
                weight_all.extend([1.0] * int(test_mask.sum()))
            else:
                weight_all.extend(weight_test.tolist())

            if model_name == "HistGBDT(raw)":
                calibration = fit_p95_calibrator(
                    X_train_base=X_train,
                    X_test_base=X_test,
                    y_p95_train=y_p95[train_mask],
                    raw_p95_train=p95_train_pred,
                    raw_p95_test=p95_test_pred,
                    sample_weight_train=weight_train,
                )
                p95_cal_pred_all.extend(np.asarray(calibration["p95_pred"], dtype=float).tolist())

        weights = np.asarray(weight_all, dtype=float)
        weight_arg = weights if sample_weight is not None else None
        p95_metrics = metric_dict(np.asarray(p95_true_all), np.asarray(p95_pred_all), weight_arg)
        tps_metrics = metric_dict(np.asarray(tps_true_all), np.asarray(tps_pred_all), weight_arg)
        rows.append(
            {
                "dataset": dataset_name,
                "protocol": "LOGO",
                "model": model_name,
                "target": "p95 latency",
                **p95_metrics,
            }
        )
        rows.append(
            {
                "dataset": dataset_name,
                "protocol": "LOGO",
                "model": model_name,
                "target": "throughput",
                **tps_metrics,
            }
        )

        if model_name == "HistGBDT(raw)":
            cal_metrics = metric_dict(np.asarray(p95_true_all), np.asarray(p95_cal_pred_all), weight_arg)
            rows.append(
                {
                    "dataset": dataset_name,
                    "protocol": "LOGO",
                    "model": "Ours (HistGBDT+cal)",
                    "target": "p95 latency",
                    **cal_metrics,
                }
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "protocol": "LOGO",
                    "model": "Ours (HistGBDT+cal)",
                    "target": "throughput",
                    **tps_metrics,
                }
            )

    return rows


def make_tex_table(summary: pd.DataFrame) -> str:
    selected = summary[summary["target"].isin(["p95 latency", "throughput"])].copy()
    pivot = selected.pivot_table(
        index=["dataset", "protocol", "model"],
        columns="target",
        values=["mae", "r2"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{target}" for metric, target in pivot.columns]
    pivot = pivot.reset_index()
    model_order = {
        "Mean": 0,
        "Ridge": 1,
        "RandomForest": 2,
        "ExtraTrees": 3,
        "HistGBDT(raw)": 4,
        "Ours (HistGBDT+cal)": 5,
    }
    dataset_order = {"Controlled": 0, "Real trace": 1}
    protocol_order = {"Random split": 0, "LOGO": 1}
    pivot["dataset_order"] = pivot["dataset"].map(dataset_order)
    pivot["protocol_order"] = pivot["protocol"].map(protocol_order)
    pivot["model_order"] = pivot["model"].map(model_order)
    pivot = pivot.sort_values(["dataset_order", "protocol_order", "model_order"]).drop(
        columns=["dataset_order", "protocol_order", "model_order"]
    )

    lines = [
        "\\begin{table}[t]",
        "\\caption{Prediction Model Baseline Comparison}",
        "\\label{tab:prediction_model_baselines}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{0.9\\columnwidth}{!}{%",
        "\\begin{tabular}{lllcccc}",
        "\\toprule",
        "Dataset & Protocol & Model & p95 MAE & p95 $R^2$ & TPS MAE & TPS $R^2$ \\\\",
        "\\midrule",
    ]
    for _, row in pivot.iterrows():
        lines.append(
            f"{row['dataset']} & {row['protocol']} & {row['model']} & "
            f"{row['mae_p95 latency']:.4f} & {row['r2_p95 latency']:.3f} & "
            f"{row['mae_throughput']:.2f} & {row['r2_throughput']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(keep_error_rows=False)
    controlled_rows = evaluate_dataset(
        dataset_name="Controlled",
        df=df,
        sample_weight=None,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    controlled_logo_rows = evaluate_logo_dataset(
        dataset_name="Controlled",
        df=df,
        sample_weight=None,
    )

    trace_weights, trace_meta = build_trace_sample_weights(
        df,
        trace_csv=args.trace_weight_csv,
        scope=args.trace_weight_scope,
        alpha=args.trace_weight_alpha,
        min_weight=args.trace_weight_min,
        max_weight=args.trace_weight_max,
    )
    real_rows = evaluate_dataset(
        dataset_name="Real trace",
        df=df,
        sample_weight=trace_weights,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    real_logo_rows = evaluate_logo_dataset(
        dataset_name="Real trace",
        df=df,
        sample_weight=trace_weights,
    )

    summary = pd.DataFrame(controlled_rows + controlled_logo_rows + real_rows + real_logo_rows)
    summary.to_csv(outdir / "prediction_model_baseline_metrics.csv", index=False)
    (outdir / "prediction_model_baseline_table.tex").write_text(make_tex_table(summary), encoding="utf-8")
    (outdir / "metadata.json").write_text(
        json.dumps(
            {
                "rows_after_clean": int(len(df)),
                "test_size": args.test_size,
                "random_state": args.random_state,
                "trace_weighting": trace_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    compact = summary.pivot_table(
        index=["dataset", "protocol", "model"],
        columns="target",
        values=["mae", "r2"],
        aggfunc="first",
    )
    print(compact.round(4).to_string())
    print(json.dumps({"outdir": str(outdir), "rows": int(len(summary))}, indent=2))


if __name__ == "__main__":
    main()
