import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from tepqos.features import build_feature_frame
from tepqos.model_baselines import baseline_models, fit_log_model, metric_dict
from tepqos.training import (
    align_columns,
    build_trace_sample_weights,
    fit_p95_calibrator,
    prepare_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune HistGBDT predictor variants under LOGO and compare with tree baselines.",
    )
    parser.add_argument("--outdir", default="final/prediction_model_tuning_v1")
    parser.add_argument("--trace-weight-csv", default="data/workload_traces/wildchat_turn_trace_short_bucketed.csv")
    parser.add_argument("--trace-weight-scope", default="prompt_gen", choices=["prompt_gen", "prompt_gen_concurrency"])
    parser.add_argument("--trace-weight-alpha", type=float, default=1.0)
    parser.add_argument("--trace-weight-min", type=float, default=0.25)
    parser.add_argument("--trace-weight-max", type=float, default=4.0)
    return parser.parse_args()


def hgbdt_variants() -> Dict[str, HistGradientBoostingRegressor]:
    return {
        "current": HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.05,
            max_iter=1200,
            min_samples_leaf=4,
            l2_regularization=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=40,
            random_state=42,
        ),
        "leaf2_l2_0": HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.05,
            max_iter=1200,
            min_samples_leaf=2,
            l2_regularization=0.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=40,
            random_state=42,
        ),
        "depth6_leaf2": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=1200,
            min_samples_leaf=2,
            l2_regularization=0.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=40,
            random_state=42,
        ),
        "depth4_leaf2": HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=1200,
            min_samples_leaf=2,
            l2_regularization=1e-4,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=40,
            random_state=42,
        ),
        "lr03_leaf2": HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.03,
            max_iter=1800,
            min_samples_leaf=2,
            l2_regularization=0.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=60,
            random_state=42,
        ),
        "noearly_300_leaf4": HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=4,
            l2_regularization=1e-3,
            early_stopping=False,
            random_state=42,
        ),
        "noearly_500_leaf2": HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.03,
            max_iter=500,
            min_samples_leaf=2,
            l2_regularization=1e-4,
            early_stopping=False,
            random_state=42,
        ),
    }


def evaluate_logo(
    *,
    df: pd.DataFrame,
    sample_weight: Optional[np.ndarray],
    model,
    model_name: str,
    dataset_name: str,
    calibrated: bool,
) -> Dict[str, object]:
    X_full = build_feature_frame(df)
    y_p95 = df["lat_p95_s"].to_numpy()
    y_tps = df["tok_per_s"].to_numpy()

    p95_true_all: List[float] = []
    p95_raw_all: List[float] = []
    p95_cal_all: List[float] = []
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
        p95_raw_all.extend(p95_test_pred.tolist())
        tps_true_all.extend(y_tps[test_mask].tolist())
        tps_pred_all.extend(tps_test_pred.tolist())
        if weight_test is None:
            weight_all.extend([1.0] * int(test_mask.sum()))
        else:
            weight_all.extend(weight_test.tolist())

        if calibrated:
            calibration = fit_p95_calibrator(
                X_train_base=X_train,
                X_test_base=X_test,
                y_p95_train=y_p95[train_mask],
                raw_p95_train=p95_train_pred,
                raw_p95_test=p95_test_pred,
                sample_weight_train=weight_train,
            )
            p95_cal_all.extend(np.asarray(calibration["p95_pred"], dtype=float).tolist())

    weight_arg = np.asarray(weight_all, dtype=float) if sample_weight is not None else None
    p95_true = np.asarray(p95_true_all)
    p95_raw = np.asarray(p95_raw_all)
    p95_for_report = np.asarray(p95_cal_all) if calibrated else p95_raw
    tps_true = np.asarray(tps_true_all)
    tps_pred = np.asarray(tps_pred_all)
    p95_raw_metrics = metric_dict(p95_true, p95_raw, weight_arg)
    p95_metrics = metric_dict(p95_true, p95_for_report, weight_arg)
    tps_metrics = metric_dict(tps_true, tps_pred, weight_arg)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "calibrated_p95": calibrated,
        "p95_mse": p95_metrics["mse"],
        "p95_raw_mae": p95_raw_metrics["mae"],
        "p95_mae": p95_metrics["mae"],
        "p95_r2": p95_metrics["r2"],
        "p95_mape_pct": p95_metrics["mape_pct"],
        "tps_mse": tps_metrics["mse"],
        "tps_mae": tps_metrics["mae"],
        "tps_r2": tps_metrics["r2"],
        "tps_mape_pct": tps_metrics["mape_pct"],
    }


def summarize_best(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in results["dataset"].unique():
        subset = results[results["dataset"] == dataset]
        best_p95 = subset.loc[subset["p95_mae"].idxmin()]
        best_tps = subset.loc[subset["tps_mae"].idxmin()]
        rows.append(
            {
                "dataset": dataset,
                "criterion": "best_p95_mae",
                "model": best_p95["model"],
                "p95_mse": best_p95["p95_mse"],
                "p95_mae": best_p95["p95_mae"],
                "p95_r2": best_p95["p95_r2"],
                "tps_mse": best_p95["tps_mse"],
                "tps_mae": best_p95["tps_mae"],
                "tps_r2": best_p95["tps_r2"],
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "criterion": "best_tps_mae",
                "model": best_tps["model"],
                "p95_mse": best_tps["p95_mse"],
                "p95_mae": best_tps["p95_mae"],
                "p95_r2": best_tps["p95_r2"],
                "tps_mse": best_tps["tps_mse"],
                "tps_mae": best_tps["tps_mae"],
                "tps_r2": best_tps["tps_r2"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(keep_error_rows=False)
    trace_weights, trace_meta = build_trace_sample_weights(
        df,
        trace_csv=args.trace_weight_csv,
        scope=args.trace_weight_scope,
        alpha=args.trace_weight_alpha,
        min_weight=args.trace_weight_min,
        max_weight=args.trace_weight_max,
    )

    rows: List[Dict[str, object]] = []
    datasets = [
        ("Controlled", None),
        ("Real trace", trace_weights),
    ]

    for dataset_name, weights in datasets:
        for name, model in hgbdt_variants().items():
            rows.append(
                evaluate_logo(
                    df=df,
                    sample_weight=weights,
                    model=model,
                    model_name=f"HistGBDT/{name}",
                    dataset_name=dataset_name,
                    calibrated=True,
                )
            )
        for name in ["RandomForest", "ExtraTrees"]:
            rows.append(
                evaluate_logo(
                    df=df,
                    sample_weight=weights,
                    model=baseline_models()[name],
                    model_name=name,
                    dataset_name=dataset_name,
                    calibrated=False,
                )
            )

    results = pd.DataFrame(rows)
    results = results.sort_values(["dataset", "p95_mae", "tps_mae"]).reset_index(drop=True)
    best = summarize_best(results)

    results.to_csv(outdir / "histgbdt_tuning_logo_metrics.csv", index=False)
    best.to_csv(outdir / "histgbdt_tuning_best_summary.csv", index=False)
    (outdir / "metadata.json").write_text(
        json.dumps(
            {
                "rows_after_clean": int(len(df)),
                "protocol": "leave-one-source-group-out",
                "trace_weighting": trace_meta,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(results.round(4).to_string(index=False))
    print()
    print(best.round(4).to_string(index=False))
    print(json.dumps({"outdir": str(outdir), "rows": int(len(results))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
