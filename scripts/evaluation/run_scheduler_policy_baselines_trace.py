import argparse
import copy
import json
import os
from math import isfinite
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from tepqos import batch_eval
from scripts.evaluation import run_scheduler_policy_baselines as baselines
from tepqos.workloads import load_trace_workload_cases


DEFAULT_OUTDIR = os.path.join("artifacts_gbdt_multi", "scheduler_policy_baselines_public_trace")
SUPPORTED_POLICIES = {
    "scheduler_pred",
    "edf_adapted",
    "mef_adapted",
    "pfs_adapted",
    "robust_mef_adapted",
    "slo_goodput_greedy",
    "cloud_relative_greedy",
    "llm_pq_adapted",
    "latency_slack_first",
    "pareto_greedy",
    "melange_adapted",
    "best_static_edge",
    "cloud_only",
    "oracle_cost_actual",
    "oracle_throughput_actual",
    "random_edge",
    "bc_policy",
    "dagger_policy",
    "rl_dqn",
}


def parse_policy_filter(raw_value: str) -> Optional[set[str]]:
    text = str(raw_value or "").strip()
    if not text or text.lower() == "all":
        return None
    selected = {item.strip() for item in text.split(",") if item.strip()}
    unknown = sorted(selected - SUPPORTED_POLICIES)
    if unknown:
        raise ValueError(f"unknown policies requested: {unknown}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scheduler baselines on a bucketed public workload trace without modifying the original baseline script.",
    )
    parser.add_argument("--artifacts-dir", default=baselines.DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--catalog", default=baselines.DEFAULT_CATALOG)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--devices", default=baselines.DEFAULT_DEVICES)
    parser.add_argument("--slo-list", default=baselines.DEFAULT_SLO_LIST)
    parser.add_argument("--edge-mem-list", default=baselines.DEFAULT_EDGE_MEM_LIST)
    parser.add_argument("--random-seeds", default=baselines.DEFAULT_RANDOM_SEEDS)
    parser.add_argument("--device-rates", default=baselines.DEFAULT_DEVICE_RATES)
    parser.add_argument("--static-policy-map", default=baselines.DEFAULT_STATIC_POLICY)
    parser.add_argument("--prediction-source", default=baselines.DEFAULT_PREDICTION_SOURCE, choices=["model", "table"])
    parser.add_argument("--prediction-table-csv", default=baselines.DEFAULT_PREDICTION_TABLE_CSV)
    parser.add_argument("--prediction-p95-col", default=baselines.DEFAULT_PREDICTION_P95_COL)
    parser.add_argument("--prediction-tps-col", default=baselines.DEFAULT_PREDICTION_TPS_COL)
    parser.add_argument("--edge-queue-delay-s", type=float, default=0.0)
    parser.add_argument("--cloud-extra-delay-s", type=float, default=0.0)
    parser.add_argument("--latency-weight", type=float, default=1.0)
    parser.add_argument("--memory-weight", type=float, default=0.3)
    parser.add_argument("--throughput-weight", type=float, default=0.2)
    parser.add_argument("--cloud-route-penalty", type=float, default=0.8)
    parser.add_argument("--min-throughput-tps", type=float, default=None)
    parser.add_argument("--throughput-floor-ratio", type=float, default=0.85)
    parser.add_argument("--throughput-tie-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_THROUGHPUT_TIE_RATIO)
    parser.add_argument("--trace-small-max-prompt-len", type=int, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_MAX_PROMPT_LEN)
    parser.add_argument("--trace-small-max-gen-len", type=int, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_MAX_GEN_LEN)
    parser.add_argument("--trace-medium-max-prompt-len", type=int, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_MAX_PROMPT_LEN)
    parser.add_argument("--trace-medium-max-gen-len", type=int, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_MAX_GEN_LEN)
    parser.add_argument("--trace-small-edge-min-tps-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MIN_TPS_RATIO)
    parser.add_argument("--trace-medium-edge-min-tps-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MIN_TPS_RATIO)
    parser.add_argument("--trace-large-edge-min-tps-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MIN_TPS_RATIO)
    parser.add_argument("--trace-small-edge-max-slo-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MAX_SLO_RATIO)
    parser.add_argument("--trace-medium-edge-max-slo-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MAX_SLO_RATIO)
    parser.add_argument("--trace-large-edge-max-slo-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MAX_SLO_RATIO)
    parser.add_argument("--trace-small-edge-max-cloud-lat-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument("--trace-medium-edge-max-cloud-lat-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument("--trace-large-edge-max-cloud-lat-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument("--trace-small-latency-rescue-min-cloud-pressure-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument("--trace-medium-latency-rescue-min-cloud-pressure-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument("--trace-large-latency-rescue-min-cloud-pressure-ratio", type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_LARGE_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument("--disable-trace-edge-rescue", action="store_true")
    parser.add_argument("--cloud-throughput-mode", choices=["decode_only", "service_effective"], default=batch_eval.scheduler_core.DEFAULT_CLOUD_THROUGHPUT_MODE)
    parser.add_argument("--optimize-for", choices=["balanced", "latency", "throughput", "throughput_constrained", "throughput_constrained_trace", "throughput_constrained_trace_distilled", "latency_constrained"], default="balanced")
    parser.add_argument("--slo-safety-margin", type=float, default=1.0)
    parser.add_argument("--tie-cost-tolerance-ratio", type=float, default=0.05)
    parser.add_argument("--violation-tolerance-ratio", type=float, default=0.03)
    parser.add_argument("--memory-penalty-threshold", type=float, default=0.85)
    parser.add_argument("--reject-extrapolation", action="store_true")
    parser.add_argument("--no-prefer-edge", action="store_true")
    parser.add_argument("--learned-model", default=baselines.DEFAULT_LEARNED_MODEL)
    parser.add_argument("--disable-learned-policy", action="store_true")
    parser.add_argument("--workload-csv", required=True)
    parser.add_argument("--max-trace-cases", type=int, default=None)
    parser.add_argument(
        "--policies",
        default="all",
        help="Comma-separated subset of policies to evaluate; use 'all' for the full set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    devices = batch_eval.parse_csv_list(args.devices, str)
    slo_list = batch_eval.parse_csv_list(args.slo_list, float)
    edge_mem_list = batch_eval.parse_csv_list(args.edge_mem_list, float)
    random_seeds = batch_eval.parse_csv_list(args.random_seeds, int)
    device_rates = baselines.parse_kv_mapping(args.device_rates, float)
    static_policy_map = baselines.parse_kv_mapping(args.static_policy_map, str)
    selected_policies = parse_policy_filter(args.policies)

    assets = batch_eval.scheduler_core.load_assets(args.artifacts_dir, prediction_source=args.prediction_source)
    catalog = batch_eval.load_json(args.catalog)
    manifest, truth_lookup = batch_eval.load_truth_tables(args.artifacts_dir)
    prediction_lookup: Dict[Any, Any] = {}
    if args.prediction_source == "table":
        prediction_lookup = batch_eval.load_prediction_lookup(args.prediction_table_csv)

    workload_cases = load_trace_workload_cases(
        trace_csv=args.workload_csv,
        devices=devices,
        truth_lookup=truth_lookup,
        duplicate_across_devices=True,
        drop_unsupported=True,
        max_cases=args.max_trace_cases,
    )

    learned_bundle = None
    if not bool(args.disable_learned_policy):
        learned_bundle = baselines.load_learned_policy(args.learned_model, args)

    all_rows: List[Dict[str, Any]] = []
    max_slo = max(float(x) for x in slo_list)
    max_edge_mem = max(float(x) for x in edge_mem_list)

    for workload in workload_cases:
        device_id = str(workload["device_id"])
        static_config_id = static_policy_map.get(device_id)
        if not static_config_id:
            raise ValueError(f"missing static baseline mapping for {device_id}")

        base_args = batch_eval.make_scheduler_args(
            device_id=device_id,
            prompt_len=int(workload["prompt_len"]),
            gen_len=int(workload["gen_len"]),
            concurrency=int(workload["concurrency"]),
            slo_p95_s=max_slo,
            edge_mem_available_mb=max_edge_mem,
            config=args,
        )
        base_predicted_candidates = batch_eval.build_predicted_candidates(
            catalog=catalog,
            assets=assets,
            args=base_args,
            prediction_source=args.prediction_source,
            prediction_lookup=prediction_lookup,
            prediction_p95_col=args.prediction_p95_col,
            prediction_tps_col=args.prediction_tps_col,
        )
        base_actual_candidates = [
            batch_eval.actualize_candidate(candidate, truth_lookup=truth_lookup, args=base_args)
            for candidate in base_predicted_candidates
        ]

        for slo_p95_s in slo_list:
            for edge_mem_available_mb in edge_mem_list:
                scheduler_args = batch_eval.make_scheduler_args(
                    device_id=device_id,
                    prompt_len=int(workload["prompt_len"]),
                    gen_len=int(workload["gen_len"]),
                    concurrency=int(workload["concurrency"]),
                    slo_p95_s=float(slo_p95_s),
                    edge_mem_available_mb=float(edge_mem_available_mb),
                    config=args,
                )

                predicted_candidates = copy.deepcopy(base_predicted_candidates)
                actual_candidates = copy.deepcopy(base_actual_candidates)
                for candidate in predicted_candidates:
                    candidate.update(batch_eval.scheduler_core.runtime_constraints(candidate, scheduler_args))
                    batch_eval.scheduler_core.annotate_candidate_risk(
                        candidate,
                        scheduler_args,
                        risk_buffer_s=float(candidate.get("risk_buffer_s", 0.0)),
                    )
                    candidate["within_slo"] = bool(float(candidate.get("predicted_p95_s", 1e9)) <= float(scheduler_args.slo_p95_s))
                    candidate["within_safe_slo"] = bool(
                        float(candidate.get("predicted_p95_s", 1e9)) <= float(batch_eval.scheduler_core.safe_slo_limit(scheduler_args))
                    )
                batch_eval.scheduler_core.score_candidates(predicted_candidates, scheduler_args)

                for candidate in actual_candidates:
                    candidate.update(batch_eval.scheduler_core.runtime_constraints(candidate, scheduler_args))
                    batch_eval.scheduler_core.annotate_candidate_risk(
                        candidate,
                        scheduler_args,
                        risk_buffer_s=float(candidate.get("risk_buffer_s", 0.0)),
                    )
                    actual_latency = float(candidate.get("actual_p95_s", np.nan))
                    if isfinite(actual_latency):
                        candidate["within_slo"] = bool(actual_latency <= float(scheduler_args.slo_p95_s))
                        candidate["within_safe_slo"] = bool(actual_latency <= float(batch_eval.scheduler_core.safe_slo_limit(scheduler_args)))
                    else:
                        candidate["within_slo"] = False
                        candidate["within_safe_slo"] = False
                batch_eval.scheduler_core.score_candidates(actual_candidates, scheduler_args)

                case_info = {
                    "device_id": device_id,
                    "prompt_len": int(workload["prompt_len"]),
                    "gen_len": int(workload["gen_len"]),
                    "concurrency": int(workload["concurrency"]),
                    "request_total_tokens": int(workload["prompt_len"]) + int(workload["gen_len"]),
                    "slo_p95_s": float(slo_p95_s),
                    "edge_mem_available_mb": float(edge_mem_available_mb),
                }

                if selected_policies is None or "scheduler_pred" in selected_policies:
                    sched_pred, sched_reason, _ = batch_eval.scheduler_core.choose_candidate(predicted_candidates, scheduler_args)
                    sched_actual = baselines.get_actual_choice(sched_pred, actual_candidates)
                    all_rows.append(
                        baselines.build_result_row(
                            policy="scheduler_pred",
                            policy_type="adaptive",
                            seed=None,
                            selection_reason=sched_reason,
                            case_info=case_info,
                            choice=sched_actual,
                            device_rates=device_rates,
                        )
                    )

                if learned_bundle is not None:
                    learned_pred, learned_actual, learned_reason = baselines.choose_learned_policy(
                        predicted_candidates,
                        actual_candidates,
                        prompt_len=int(workload["prompt_len"]),
                        gen_len=int(workload["gen_len"]),
                        concurrency=int(workload["concurrency"]),
                        slo_p95_s=float(slo_p95_s),
                        edge_mem_available_mb=float(edge_mem_available_mb),
                        edge_device_id=device_id,
                        learned_bundle=learned_bundle,
                    )
                    learned_policy_name = str(learned_bundle["policy_name"])
                    if selected_policies is None or learned_policy_name in selected_policies:
                        all_rows.append(
                            baselines.build_result_row(
                                policy=learned_policy_name,
                                policy_type="learned",
                                seed=None,
                                selection_reason=learned_reason,
                                case_info=case_info,
                                choice=learned_actual,
                                device_rates=device_rates,
                            )
                        )

                deterministic_policies = [
                    ("edf_adapted", baselines.choose_edf_adapted(predicted_candidates, actual_candidates)),
                    ("mef_adapted", baselines.choose_mef_adapted(predicted_candidates, actual_candidates)),
                    ("robust_mef_adapted", baselines.choose_robust_mef_adapted(predicted_candidates, actual_candidates)),
                    (
                        "pfs_adapted",
                        baselines.choose_pfs_adapted(
                            predicted_candidates,
                            actual_candidates,
                            slo_p95_s=float(slo_p95_s),
                        ),
                    ),
                    (
                        "slo_goodput_greedy",
                        baselines.choose_slo_goodput_greedy(
                            predicted_candidates,
                            actual_candidates,
                            slo_p95_s=float(slo_p95_s),
                        ),
                    ),
                    ("cloud_relative_greedy", baselines.choose_cloud_relative_greedy(predicted_candidates, actual_candidates)),
                    ("llm_pq_adapted", baselines.choose_llm_pq_adapted(predicted_candidates, actual_candidates)),
                    ("latency_slack_first", baselines.choose_latency_slack_first(predicted_candidates, actual_candidates)),
                    ("pareto_greedy", baselines.choose_pareto_greedy(predicted_candidates, actual_candidates)),
                    ("melange_adapted", baselines.choose_melange_adapted(predicted_candidates, actual_candidates, device_rates)),
                    ("best_static_edge", baselines.choose_best_static(predicted_candidates, actual_candidates, static_config_id)),
                ]
                for policy_name, (pred_choice, actual_choice, reason) in deterministic_policies:
                    if selected_policies is not None and policy_name not in selected_policies:
                        continue
                    all_rows.append(
                        baselines.build_result_row(
                            policy=policy_name,
                            policy_type="baseline",
                            seed=None,
                            selection_reason=reason,
                            case_info=case_info,
                            choice=actual_choice,
                            device_rates=device_rates,
                        )
                    )

                if selected_policies is None or "cloud_only" in selected_policies:
                    cloud_pred = baselines.get_cloud_candidate(predicted_candidates)
                    cloud_actual = baselines.get_actual_choice(cloud_pred, actual_candidates)
                    all_rows.append(
                        baselines.build_result_row(
                            policy="cloud_only",
                            policy_type="baseline",
                            seed=None,
                            selection_reason="fixed_cloud",
                            case_info=case_info,
                            choice=cloud_actual,
                            device_rates=device_rates,
                        )
                    )

                if selected_policies is None or "oracle_cost_actual" in selected_policies:
                    oracle_actual, oracle_reason = baselines.choose_oracle_cost(actual_candidates, device_rates)
                    all_rows.append(
                        baselines.build_result_row(
                            policy="oracle_cost_actual",
                            policy_type="oracle",
                            seed=None,
                            selection_reason=oracle_reason,
                            case_info=case_info,
                            choice=oracle_actual,
                            device_rates=device_rates,
                        )
                    )

                if selected_policies is None or "oracle_throughput_actual" in selected_policies:
                    oracle_tps_actual, oracle_tps_reason = baselines.choose_oracle_throughput(actual_candidates)
                    all_rows.append(
                        baselines.build_result_row(
                            policy="oracle_throughput_actual",
                            policy_type="oracle",
                            seed=None,
                            selection_reason=oracle_tps_reason,
                            case_info=case_info,
                            choice=oracle_tps_actual,
                            device_rates=device_rates,
                        )
                    )

                if selected_policies is None or "random_edge" in selected_policies:
                    for seed in random_seeds:
                        rng = batch_eval.np.random.default_rng(seed)
                        random_pred, random_actual, random_reason = baselines.choose_random_edge(predicted_candidates, actual_candidates, rng)
                        all_rows.append(
                            baselines.build_result_row(
                                policy="random_edge",
                                policy_type="baseline_random",
                                seed=int(seed),
                                selection_reason=random_reason,
                                case_info=case_info,
                                choice=random_actual,
                                device_rates=device_rates,
                            )
                        )

    frame = pd.DataFrame(all_rows)
    case_csv = os.path.join(args.outdir, "baseline_case_results.csv")
    frame.to_csv(case_csv, index=False, encoding="utf-8-sig")

    summary = baselines.summarize(frame, ["device_id", "policy", "policy_type"])
    summary_csv = os.path.join(args.outdir, "baseline_policy_summary.csv")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    random_summary = baselines.summarize(
        frame[frame["policy"] == "random_edge"],
        ["device_id", "policy", "seed"],
    )
    random_summary_csv = os.path.join(args.outdir, "baseline_random_seed_summary.csv")
    random_summary.to_csv(random_summary_csv, index=False, encoding="utf-8-sig")

    policy_slo_mem = baselines.summarize(
        frame[frame["seed"].isna()],
        ["device_id", "policy", "slo_p95_s", "edge_mem_available_mb"],
    )
    policy_slo_mem_csv = os.path.join(args.outdir, "baseline_policy_summary_by_slo_mem.csv")
    policy_slo_mem.to_csv(policy_slo_mem_csv, index=False, encoding="utf-8-sig")

    metadata = {
        "devices": devices,
        "workload_source": os.path.abspath(args.workload_csv),
        "workload_cases": len(workload_cases),
        "slo_list": slo_list,
        "edge_mem_list": edge_mem_list,
        "optimize_for": args.optimize_for,
        "max_trace_cases": args.max_trace_cases,
        "prediction_source": args.prediction_source,
        "prediction_table_csv": os.path.abspath(args.prediction_table_csv) if args.prediction_source == "table" else None,
        "cloud_throughput_mode": args.cloud_throughput_mode,
        "throughput_tie_ratio": args.throughput_tie_ratio,
        "throughput_floor_ratio": args.throughput_floor_ratio,
        "trace_small_edge_min_tps_ratio": args.trace_small_edge_min_tps_ratio,
        "trace_medium_edge_min_tps_ratio": args.trace_medium_edge_min_tps_ratio,
        "trace_large_edge_min_tps_ratio": args.trace_large_edge_min_tps_ratio,
        "trace_small_edge_max_slo_ratio": args.trace_small_edge_max_slo_ratio,
        "trace_medium_edge_max_slo_ratio": args.trace_medium_edge_max_slo_ratio,
        "trace_large_edge_max_slo_ratio": args.trace_large_edge_max_slo_ratio,
        "trace_small_edge_max_cloud_lat_ratio": args.trace_small_edge_max_cloud_lat_ratio,
        "trace_medium_edge_max_cloud_lat_ratio": args.trace_medium_edge_max_cloud_lat_ratio,
        "trace_large_edge_max_cloud_lat_ratio": args.trace_large_edge_max_cloud_lat_ratio,
        "trace_small_latency_rescue_min_cloud_pressure_ratio": args.trace_small_latency_rescue_min_cloud_pressure_ratio,
        "trace_medium_latency_rescue_min_cloud_pressure_ratio": args.trace_medium_latency_rescue_min_cloud_pressure_ratio,
        "trace_large_latency_rescue_min_cloud_pressure_ratio": args.trace_large_latency_rescue_min_cloud_pressure_ratio,
        "disable_trace_edge_rescue": bool(args.disable_trace_edge_rescue),
        "selected_policies": sorted(selected_policies) if selected_policies is not None else "all",
            "baselines": {
                "robust_mef_adapted": "Maximum predicted throughput after risk-adjusted SLO admission.",
                "pfs_adapted": "Past-Future Scheduler inspired memory- and SLA-headroom weighted throughput baseline.",
                "slo_goodput_greedy": "Maximum T_hat * min(1, SLO / risk-adjusted latency).",
            "cloud_relative_greedy": "Risk-safe edge only when predicted throughput is no lower than cloud.",
            "llm_pq_adapted": "Per-request quantized edge configuration and micro-batch approximation of LLM-PQ; no phase partition or quality model.",
            "oracle_throughput_actual": "Ground-truth throughput upper bound under the SLO.",
        },
        "random_seeds": random_seeds,
        "static_policy_map": static_policy_map,
        "device_rates": device_rates,
        "manifest_rows_after_clean": int(manifest.get("rows_after_clean", 0)),
    }
    with open(os.path.join(args.outdir, "baseline_eval_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"summary_csv": os.path.abspath(summary_csv), "cases": len(frame)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
