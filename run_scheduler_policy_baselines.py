import argparse
import copy
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import run_scheduler_batch_eval as batch_eval
from rl_scheduler.env import build_env_from_args

DEFAULT_ARTIFACTS_DIR = 'artifacts_gbdt_multi'
DEFAULT_CATALOG = 'scheduler_catalog.json'
DEFAULT_OUTDIR = os.path.join(DEFAULT_ARTIFACTS_DIR, 'scheduler_policy_baselines')
DEFAULT_DEVICES = 'edge_3090,edge_4090'
DEFAULT_SLO_LIST = '0.75,1.0,2.0,4.0'
DEFAULT_EDGE_MEM_LIST = '6500,7500,8500,10000'
DEFAULT_RANDOM_SEEDS = '0,1,2,3,4,5,6,7,8,9'
DEFAULT_DEVICE_RATES = 'edge_3090=1.0,edge_4090=1.3,cloud_a100=4.0'
DEFAULT_STATIC_POLICY = 'edge_3090=gptq4_bs8_tok2048_kvauto,edge_4090=gptq4_bs8_tok2048_kvauto'
DEFAULT_PREDICTION_SOURCE = 'model'
DEFAULT_PREDICTION_TABLE_CSV = os.path.join(DEFAULT_ARTIFACTS_DIR, 'combined_perf_with_pred.csv')
DEFAULT_PREDICTION_P95_COL = 'p95_calibrated_s'
DEFAULT_PREDICTION_TPS_COL = 'tps_hat'
DEFAULT_LEARNED_MODEL = os.path.join('artifacts_rl_scheduler', 'dagger_only', 'dagger_warmstart.pt')


def parse_kv_mapping(raw_value: str, value_cast=str) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for item in str(raw_value or '').split(','):
        text = item.strip()
        if not text or '=' not in text:
            continue
        key, value = text.split('=', 1)
        mapping[key.strip()] = value_cast(value.strip())
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate paper-style baselines (Random / EDF / MEF / Melange) on the offline scheduler replay set.',
    )
    parser.add_argument('--artifacts-dir', default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument('--catalog', default=DEFAULT_CATALOG)
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR)
    parser.add_argument('--devices', default=DEFAULT_DEVICES)
    parser.add_argument('--slo-list', default=DEFAULT_SLO_LIST)
    parser.add_argument('--edge-mem-list', default=DEFAULT_EDGE_MEM_LIST)
    parser.add_argument('--workload-mode', default='intersection', choices=['intersection', 'union'])
    parser.add_argument('--random-seeds', default=DEFAULT_RANDOM_SEEDS)
    parser.add_argument('--device-rates', default=DEFAULT_DEVICE_RATES)
    parser.add_argument('--static-policy-map', default=DEFAULT_STATIC_POLICY)
    parser.add_argument('--prediction-source', default=DEFAULT_PREDICTION_SOURCE, choices=['model', 'table'])
    parser.add_argument('--prediction-table-csv', default=DEFAULT_PREDICTION_TABLE_CSV)
    parser.add_argument('--prediction-p95-col', default=DEFAULT_PREDICTION_P95_COL)
    parser.add_argument('--prediction-tps-col', default=DEFAULT_PREDICTION_TPS_COL)
    parser.add_argument('--edge-queue-delay-s', type=float, default=0.0)
    parser.add_argument('--cloud-extra-delay-s', type=float, default=0.0)
    parser.add_argument('--latency-weight', type=float, default=1.0)
    parser.add_argument('--memory-weight', type=float, default=0.3)
    parser.add_argument('--throughput-weight', type=float, default=0.2)
    parser.add_argument('--cloud-route-penalty', type=float, default=0.8)
    parser.add_argument('--min-throughput-tps', type=float, default=None)
    parser.add_argument('--throughput-floor-ratio', type=float, default=0.85)
    parser.add_argument('--throughput-tie-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_THROUGHPUT_TIE_RATIO)
    parser.add_argument('--trace-small-max-prompt-len', type=int, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_MAX_PROMPT_LEN)
    parser.add_argument('--trace-small-max-gen-len', type=int, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_MAX_GEN_LEN)
    parser.add_argument('--trace-medium-max-prompt-len', type=int, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_MAX_PROMPT_LEN)
    parser.add_argument('--trace-medium-max-gen-len', type=int, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_MAX_GEN_LEN)
    parser.add_argument('--trace-small-edge-min-tps-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MIN_TPS_RATIO)
    parser.add_argument('--trace-medium-edge-min-tps-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MIN_TPS_RATIO)
    parser.add_argument('--trace-large-edge-min-tps-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MIN_TPS_RATIO)
    parser.add_argument('--trace-small-edge-max-slo-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MAX_SLO_RATIO)
    parser.add_argument('--trace-medium-edge-max-slo-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MAX_SLO_RATIO)
    parser.add_argument('--trace-large-edge-max-slo-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MAX_SLO_RATIO)
    parser.add_argument('--trace-small-edge-max-cloud-lat-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument('--trace-medium-edge-max-cloud-lat-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument('--trace-large-edge-max-cloud-lat-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_LARGE_EDGE_MAX_CLOUD_LAT_RATIO)
    parser.add_argument('--trace-small-latency-rescue-min-cloud-pressure-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_SMALL_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument('--trace-medium-latency-rescue-min-cloud-pressure-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_MEDIUM_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument('--trace-large-latency-rescue-min-cloud-pressure-ratio', type=float, default=batch_eval.scheduler_core.DEFAULT_TRACE_LARGE_LATENCY_RESCUE_MIN_CLOUD_PRESSURE_RATIO)
    parser.add_argument('--cloud-throughput-mode', choices=['decode_only', 'service_effective'], default=batch_eval.scheduler_core.DEFAULT_CLOUD_THROUGHPUT_MODE)
    parser.add_argument('--optimize-for', choices=['balanced', 'latency', 'throughput', 'throughput_constrained', 'throughput_constrained_trace', 'throughput_constrained_trace_distilled', 'latency_constrained'], default='balanced')
    parser.add_argument('--slo-safety-margin', type=float, default=1.0)
    parser.add_argument('--tie-cost-tolerance-ratio', type=float, default=0.05)
    parser.add_argument('--violation-tolerance-ratio', type=float, default=0.03)
    parser.add_argument('--memory-penalty-threshold', type=float, default=0.85)
    parser.add_argument('--reject-extrapolation', action='store_true')
    parser.add_argument('--no-prefer-edge', action='store_true')
    parser.add_argument('--learned-model', default=DEFAULT_LEARNED_MODEL)
    parser.add_argument('--disable-learned-policy', action='store_true')
    return parser.parse_args()


def runtime_feasible_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if bool(candidate.get('within_memory', False))
        and bool(candidate.get('within_token_budget', False))
        and bool(candidate.get('within_batch_budget', False))
    ]


def edge_feasible_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [candidate for candidate in runtime_feasible_candidates(candidates) if str(candidate.get('target')) == 'edge']


def slo_feasible_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [candidate for candidate in runtime_feasible_candidates(candidates) if bool(candidate.get('within_slo', False))]


def get_cloud_candidate(candidates: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return batch_eval.find_candidate(candidates, target='cloud', config_id='fp16_bs16_tok2048')


def get_actual_choice(predicted_choice: Optional[Dict[str, Any]], actual_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if predicted_choice is None:
        return None
    return batch_eval.find_candidate(
        actual_candidates,
        target=str(predicted_choice['target']),
        config_id=str(predicted_choice['config_id']),
    )


def load_learned_policy(model_path: str, env_args: argparse.Namespace):
    if not model_path or not os.path.exists(model_path):
        return None
    import torch

    from evaluate_rl_scheduler import load_checkpoint
    from train_rl_scheduler import QNetwork

    checkpoint = load_checkpoint(model_path)
    rl_args = argparse.Namespace(**vars(env_args))
    if not hasattr(rl_args, 'edge_memory_mb'):
        rl_args.edge_memory_mb = 'edge_3090=10000,edge_4090=10000'
    if not hasattr(rl_args, 'episode_length'):
        rl_args.episode_length = 64
    if not hasattr(rl_args, 'arrival_gap_s'):
        rl_args.arrival_gap_s = 0.2
    if not hasattr(rl_args, 'switch_overhead_s'):
        rl_args.switch_overhead_s = 0.02
    if not hasattr(rl_args, 'cloud_extra_delay_s'):
        rl_args.cloud_extra_delay_s = 0.0
    if not hasattr(rl_args, 'reward_throughput_weight'):
        rl_args.reward_throughput_weight = 1.0
        rl_args.reward_latency_weight = 0.75
        rl_args.reward_violation_weight = 3.5
        rl_args.reward_cloud_penalty = 0.03
        rl_args.reward_switch_penalty = 0.12
        rl_args.reward_queue_penalty = 0.45
        rl_args.reward_success_bonus = 0.08
    env = build_env_from_args(rl_args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QNetwork(checkpoint['state_dim'], checkpoint['action_dim'], checkpoint['hidden_dim']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    policy_name = {'behavior_cloning': 'bc_policy', 'dagger': 'dagger_policy', 'rl': 'rl_dqn'}.get(checkpoint.get('stage'), 'rl_dqn')
    return {
        'checkpoint': checkpoint,
        'env': env,
        'device': device,
        'model': model,
        'policy_name': policy_name,
    }


def choose_learned_policy(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
    *,
    prompt_len: int,
    gen_len: int,
    concurrency: int,
    slo_p95_s: float,
    edge_mem_available_mb: float,
    edge_device_id: str,
    learned_bundle: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    if learned_bundle is None:
        return None, None, 'learned_policy_disabled'
    from train_rl_scheduler import select_action

    env = learned_bundle['env']
    obs, mask, candidates_by_action = env.build_single_request_input(
        prompt_len=prompt_len,
        gen_len=gen_len,
        concurrency=concurrency,
        slo_p95_s=slo_p95_s,
        predicted_candidates=predicted_candidates,
        edge_memory_override={edge_device_id: edge_mem_available_mb},
    )
    if float(mask.sum()) <= 0.0:
        return None, None, 'no_action_available'
    action_idx = select_action(learned_bundle['model'], obs, mask, epsilon=0.0, device=learned_bundle['device'])
    predicted_choice = candidates_by_action.get(int(action_idx))
    actual_choice = get_actual_choice(predicted_choice, actual_candidates)
    return predicted_choice, actual_choice, f"learned_policy:{learned_bundle['policy_name']}"


def choose_random_edge(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    edge_candidates = edge_feasible_candidates(predicted_candidates)
    if edge_candidates:
        predicted_choice = edge_candidates[int(rng.integers(len(edge_candidates)))]
        return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), 'random_edge_candidate'

    cloud_choice = get_cloud_candidate(predicted_candidates)
    return cloud_choice, get_actual_choice(cloud_choice, actual_candidates), 'random_fallback_cloud'


def choose_edf_adapted(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    feasible = runtime_feasible_candidates(predicted_candidates)
    if not feasible:
        return None, None, 'no_runtime_feasible_candidate'

    on_time = [candidate for candidate in feasible if bool(candidate.get('within_slo', False))]
    if on_time:
        predicted_choice = min(
            on_time,
            key=lambda item: (float(item['predicted_p95_s']), float(item.get('predicted_tok_per_s', 0.0)) * -1.0),
        )
        return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), 'min_predicted_latency_within_slo'

    predicted_choice = min(feasible, key=lambda item: float(item['predicted_p95_s']))
    return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), 'best_effort_min_predicted_latency'


def choose_mef_adapted(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    feasible = runtime_feasible_candidates(predicted_candidates)
    if not feasible:
        return None, None, 'no_runtime_feasible_candidate'

    on_time = [candidate for candidate in feasible if bool(candidate.get('within_slo', False))]
    if on_time:
        predicted_choice = max(
            on_time,
            key=lambda item: (float(item.get('predicted_tok_per_s', 0.0)), -float(item['predicted_p95_s'])),
        )
        return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), 'max_predicted_throughput_within_slo'

    predicted_choice = min(feasible, key=lambda item: float(item['predicted_p95_s']))
    return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), 'best_effort_min_predicted_latency'


def candidate_risk_latency(candidate: Dict[str, Any]) -> float:
    return float(candidate.get('risk_adjusted_p95_s', candidate.get('predicted_p95_s', 1e9)))


def choose_robust_mef_adapted(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """Maximize predicted throughput after risk-adjusted SLO admission."""
    feasible = runtime_feasible_candidates(predicted_candidates)
    if not feasible:
        return None, None, 'no_runtime_feasible_candidate'

    risk_on_time = [candidate for candidate in feasible if bool(candidate.get('within_risk_slo', False))]
    if risk_on_time:
        predicted_choice = max(
            risk_on_time,
            key=lambda item: (
                float(item.get('predicted_tok_per_s', 0.0)),
                -candidate_risk_latency(item),
            ),
        )
        return (
            predicted_choice,
            get_actual_choice(predicted_choice, actual_candidates),
            'max_predicted_throughput_within_risk_adjusted_slo',
        )

    predicted_choice = min(feasible, key=candidate_risk_latency)
    return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), 'best_effort_min_risk_adjusted_latency'


def choose_pfs_adapted(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
    *,
    slo_p95_s: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """SLA- and memory-aware adapted Past-Future Scheduler baseline."""
    feasible = runtime_feasible_candidates(predicted_candidates)
    if not feasible:
        return None, None, 'no_runtime_feasible_candidate'

    slo = max(float(slo_p95_s), 1e-9)
    risk_on_time = [candidate for candidate in feasible if bool(candidate.get('within_risk_slo', False))]
    if risk_on_time:
        def pfs_key(item: Dict[str, Any]) -> Tuple[float, float, float, float]:
            risk_latency = max(candidate_risk_latency(item), 1e-9)
            predicted_tps = max(float(item.get('predicted_tok_per_s', 0.0)), 0.0)
            latency_headroom = min(1.0, slo / risk_latency)
            memory_headroom = max(0.0, 1.0 - float(item.get('memory_pressure', 1.0)))
            future_safe_score = predicted_tps * latency_headroom * (0.5 + 0.5 * memory_headroom)
            return future_safe_score, predicted_tps, memory_headroom, -risk_latency

        predicted_choice = max(risk_on_time, key=pfs_key)
        return (
            predicted_choice,
            get_actual_choice(predicted_choice, actual_candidates),
            'pfs_sla_memory_aware_future_safe_score',
        )

    predicted_choice = min(feasible, key=candidate_risk_latency)
    return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), 'pfs_best_effort_min_risk_adjusted_latency'


def choose_slo_goodput_greedy(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
    *,
    slo_p95_s: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """Maximize a parameter-free SLO-conditioned goodput proxy.

    The proxy discounts predicted throughput in direct proportion to the
    risk-adjusted latency overshoot: T_hat * min(1, SLO / L_risk).
    """
    feasible = runtime_feasible_candidates(predicted_candidates)
    if not feasible:
        return None, None, 'no_runtime_feasible_candidate'

    slo = max(float(slo_p95_s), 1e-9)

    def goodput_key(item: Dict[str, Any]) -> Tuple[float, float, float]:
        risk_latency = max(candidate_risk_latency(item), 1e-9)
        predicted_tps = max(float(item.get('predicted_tok_per_s', 0.0)), 0.0)
        attainment_proxy = min(1.0, slo / risk_latency)
        predicted_goodput = predicted_tps * attainment_proxy
        return predicted_goodput, attainment_proxy, -risk_latency

    predicted_choice = max(feasible, key=goodput_key)
    return (
        predicted_choice,
        get_actual_choice(predicted_choice, actual_candidates),
        'max_slo_conditioned_predicted_goodput',
    )


def choose_cloud_relative_greedy(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """Use edge only when a risk-safe edge candidate beats cloud throughput."""
    feasible = runtime_feasible_candidates(predicted_candidates)
    if not feasible:
        return None, None, 'no_runtime_feasible_candidate'

    cloud_choice = get_cloud_candidate(feasible)
    risk_on_time_edges = [
        candidate
        for candidate in feasible
        if str(candidate.get('target')) == 'edge'
        and bool(candidate.get('within_risk_slo', False))
    ]
    if not risk_on_time_edges:
        return cloud_choice, get_actual_choice(cloud_choice, actual_candidates), 'no_risk_safe_edge_fallback_cloud'

    best_edge = max(
        risk_on_time_edges,
        key=lambda item: (
            float(item.get('predicted_tok_per_s', 0.0)),
            -candidate_risk_latency(item),
        ),
    )
    if cloud_choice is None:
        return best_edge, get_actual_choice(best_edge, actual_candidates), 'risk_safe_edge_without_cloud_reference'

    edge_tps = float(best_edge.get('predicted_tok_per_s', 0.0))
    cloud_tps = float(cloud_choice.get('predicted_tok_per_s', 0.0))
    if edge_tps + 1e-9 >= cloud_tps:
        return best_edge, get_actual_choice(best_edge, actual_candidates), 'risk_safe_edge_beats_cloud_throughput'
    return cloud_choice, get_actual_choice(cloud_choice, actual_candidates), 'cloud_beats_risk_safe_edge_throughput'


def choose_llm_pq_adapted(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """Per-request approximation of quantization-aware LLM-PQ selection.

    The current simulator cannot reproduce phase partitioning or a model-quality
    constraint, so this adapter retains the common quantized-configuration and
    micro-batch selection objective: maximize edge throughput under the latency
    and runtime constraints, with cloud fallback when no edge option is valid.
    """
    feasible_edges = edge_feasible_candidates(predicted_candidates)
    on_time_edges = [candidate for candidate in feasible_edges if bool(candidate.get('within_slo', False))]
    if on_time_edges:
        predicted_choice = max(
            on_time_edges,
            key=lambda item: (
                float(item.get('predicted_tok_per_s', 0.0)),
                -float(item.get('memory_pressure', 0.0)),
                -float(item.get('predicted_p95_s', 1e9)),
            ),
        )
        return (
            predicted_choice,
            get_actual_choice(predicted_choice, actual_candidates),
            'llm_pq_quantized_edge_max_throughput_within_slo',
        )

    cloud_choice = get_cloud_candidate(predicted_candidates)
    return cloud_choice, get_actual_choice(cloud_choice, actual_candidates), 'llm_pq_no_slo_feasible_edge_fallback_cloud'


def choose_latency_slack_first(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
    *,
    edge_bias: float = 0.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    feasible = runtime_feasible_candidates(predicted_candidates)
    if not feasible:
        return None, None, 'no_runtime_feasible_candidate'

    on_time = [candidate for candidate in feasible if bool(candidate.get('within_slo', False))]
    if on_time:
        predicted_choice = max(
            on_time,
            key=lambda item: (
                float(item.get('safe_slack_ratio', 0.0)),
                1 if str(item.get('target')) == 'edge' and edge_bias > 0 else 0,
                float(item.get('predicted_tok_per_s', 0.0)),
                -float(item['predicted_p95_s']),
            ),
        )
        return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), 'max_latency_slack_then_throughput'

    predicted_choice = min(feasible, key=lambda item: float(item['predicted_p95_s']))
    return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), 'best_effort_min_predicted_latency'


def pareto_front_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    front: List[Dict[str, Any]] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_latency = float(other['predicted_p95_s'])
            other_tps = float(other.get('predicted_tok_per_s', 0.0))
            cand_latency = float(candidate['predicted_p95_s'])
            cand_tps = float(candidate.get('predicted_tok_per_s', 0.0))
            if (
                other_latency <= cand_latency
                and other_tps >= cand_tps
                and (other_latency < cand_latency or other_tps > cand_tps)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def choose_pareto_greedy(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    feasible = runtime_feasible_candidates(predicted_candidates)
    if not feasible:
        return None, None, 'no_runtime_feasible_candidate'

    on_time = [candidate for candidate in feasible if bool(candidate.get('within_slo', False))]
    pool = on_time if on_time else feasible
    reason = 'pareto_front_ideal_point_within_slo' if on_time else 'pareto_front_best_effort'
    front = pareto_front_candidates(pool)
    if not front:
        front = pool

    latencies = [float(item['predicted_p95_s']) for item in front]
    throughputs = [float(item.get('predicted_tok_per_s', 0.0)) for item in front]
    min_latency = min(latencies)
    max_latency = max(latencies)
    min_tps = min(throughputs)
    max_tps = max(throughputs)

    def ideal_distance(item: Dict[str, Any]) -> tuple[float, float, float]:
        latency = float(item['predicted_p95_s'])
        tps = float(item.get('predicted_tok_per_s', 0.0))
        latency_norm = 0.0 if max_latency <= min_latency else (latency - min_latency) / (max_latency - min_latency)
        tps_norm = 1.0 if max_tps <= min_tps else (tps - min_tps) / (max_tps - min_tps)
        distance = (latency_norm ** 2 + (1.0 - tps_norm) ** 2) ** 0.5
        return (distance, latency, -tps)

    predicted_choice = min(front, key=ideal_distance)
    return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), reason


def choose_melange_adapted(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
    device_rates: Dict[str, float],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    feasible = runtime_feasible_candidates(predicted_candidates)
    if not feasible:
        return None, None, 'no_runtime_feasible_candidate'

    def predicted_cost(item: Dict[str, Any]) -> float:
        rate = float(device_rates.get(str(item['device_id']), 1.0))
        return rate * float(item['predicted_p95_s'])

    on_time = [candidate for candidate in feasible if bool(candidate.get('within_slo', False))]
    if on_time:
        predicted_choice = min(on_time, key=lambda item: (predicted_cost(item), float(item['predicted_p95_s'])))
        return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), 'min_predicted_rental_cost_within_slo'

    predicted_choice = min(feasible, key=lambda item: float(item['predicted_p95_s']))
    return predicted_choice, get_actual_choice(predicted_choice, actual_candidates), 'best_effort_min_predicted_latency'


def choose_best_static(
    predicted_candidates: List[Dict[str, Any]],
    actual_candidates: List[Dict[str, Any]],
    config_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    return batch_eval.choose_fixed_policy(
        config_id=config_id,
        predicted_candidates=predicted_candidates,
        actual_candidates=actual_candidates,
    )


def choose_oracle_cost(
    actual_candidates: List[Dict[str, Any]],
    device_rates: Dict[str, float],
) -> Tuple[Optional[Dict[str, Any]], str]:
    feasible = runtime_feasible_candidates(actual_candidates)
    if not feasible:
        return None, 'no_runtime_feasible_candidate'

    def actual_cost(item: Dict[str, Any]) -> float:
        rate = float(device_rates.get(str(item['device_id']), 1.0))
        return rate * float(item['actual_p95_s'])

    on_time = [candidate for candidate in feasible if bool(candidate.get('within_slo', False))]
    pool = on_time if on_time else feasible
    reason = 'min_actual_rental_cost_within_slo' if on_time else 'best_effort_min_actual_cost'
    choice = min(pool, key=lambda item: (actual_cost(item), float(item['actual_p95_s'])))
    return choice, reason


def choose_oracle_throughput(
    actual_candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    feasible = runtime_feasible_candidates(actual_candidates)
    if not feasible:
        return None, 'no_runtime_feasible_candidate'

    on_time = [candidate for candidate in feasible if bool(candidate.get('within_slo', False))]
    if on_time:
        choice = max(
            on_time,
            key=lambda item: (
                float(item.get('actual_tok_per_s', 0.0)),
                -float(item.get('actual_p95_s', 1e9)),
            ),
        )
        return choice, 'max_actual_throughput_within_slo'

    choice = min(feasible, key=lambda item: float(item.get('actual_p95_s', 1e9)))
    return choice, 'best_effort_min_actual_latency'


def build_result_row(
    *,
    policy: str,
    policy_type: str,
    seed: Optional[int],
    selection_reason: str,
    case_info: Dict[str, Any],
    choice: Optional[Dict[str, Any]],
    device_rates: Dict[str, float],
) -> Dict[str, Any]:
    row = dict(case_info)
    row['policy'] = policy
    row['policy_type'] = policy_type
    row['seed'] = seed
    row['selection_reason'] = selection_reason

    if choice is None:
        row.update(
            {
                'selected_target': None,
                'selected_device_id': None,
                'selected_config_id': None,
                'selected_quant_method': None,
                'selected_batch_size': None,
                'selected_kv_strategy': None,
                'actual_status': 'no_choice',
                'actual_p95_s': np.nan,
                'actual_tok_per_s': np.nan,
                'actual_within_slo': False,
                'tcr': 0.0,
                'edge_cost': np.nan,
                'total_cost': np.nan,
                'unit_rental_rate': np.nan,
            }
        )
        return row

    device_id = str(choice['device_id'])
    rate = float(device_rates.get(device_id, 1.0))
    target = str(choice['target'])
    actual_p95_s = float(choice.get('actual_p95_s', np.nan))
    edge_cost = rate * actual_p95_s if target == 'edge' else 0.0
    total_cost = rate * actual_p95_s

    row.update(
        {
            'selected_target': target,
            'selected_device_id': device_id,
            'selected_config_id': choice.get('config_id'),
            'selected_quant_method': choice.get('quant_method'),
            'selected_batch_size': choice.get('batch_size'),
            'selected_kv_strategy': choice.get('kv_strategy'),
            'actual_status': choice.get('actual_status', 'unknown'),
            'actual_p95_s': actual_p95_s,
            'actual_tok_per_s': float(choice.get('actual_tok_per_s', np.nan)),
            'actual_within_slo': bool(choice.get('within_slo', False)),
            'tcr': 1.0 if target == 'edge' else 0.0,
            'edge_cost': edge_cost,
            'total_cost': total_cost,
            'unit_rental_rate': rate,
        }
    )
    return row


def summarize(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group_key, part in frame.groupby(list(group_cols), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(group_cols, group_key)}
        row.update(
            {
                'cases': int(len(part)),
                'mean_edge_cost': float(part['edge_cost'].mean()),
                'mean_total_cost': float(part['total_cost'].mean()),
                'median_total_cost': float(part['total_cost'].median()),
                'tcr': float(part['tcr'].mean()),
                'slo_hit_rate': float(part['actual_within_slo'].fillna(False).mean()),
                'mean_actual_p95_s': float(part['actual_p95_s'].mean()),
                'mean_actual_tok_per_s': float(part['actual_tok_per_s'].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    devices = batch_eval.parse_csv_list(args.devices, str)
    slo_list = batch_eval.parse_csv_list(args.slo_list, float)
    edge_mem_list = batch_eval.parse_csv_list(args.edge_mem_list, float)
    random_seeds = batch_eval.parse_csv_list(args.random_seeds, int)
    device_rates = parse_kv_mapping(args.device_rates, float)
    static_policy_map = parse_kv_mapping(args.static_policy_map, str)

    assets = batch_eval.scheduler_core.load_assets(args.artifacts_dir, prediction_source=args.prediction_source)
    catalog = batch_eval.load_json(args.catalog)
    manifest, truth_lookup = batch_eval.load_truth_tables(args.artifacts_dir)
    prediction_lookup = {}
    if args.prediction_source == 'table':
        prediction_lookup = batch_eval.load_prediction_lookup(args.prediction_table_csv)
    workload_cases = batch_eval.build_workload_cases(
        catalog=catalog,
        truth_lookup=truth_lookup,
        devices=devices,
        workload_mode=args.workload_mode,
    )

    learned_bundle = None
    if not bool(args.disable_learned_policy):
        learned_bundle = load_learned_policy(args.learned_model, args)

    all_rows: List[Dict[str, Any]] = []

    for workload in workload_cases:
        device_id = str(workload['device_id'])
        static_config_id = static_policy_map.get(device_id)
        if not static_config_id:
            raise ValueError(f'missing static baseline mapping for {device_id}')

        for slo_p95_s in slo_list:
            for edge_mem_available_mb in edge_mem_list:
                scheduler_args = batch_eval.make_scheduler_args(
                    device_id=device_id,
                    prompt_len=int(workload['prompt_len']),
                    gen_len=int(workload['gen_len']),
                    concurrency=int(workload['concurrency']),
                    slo_p95_s=float(slo_p95_s),
                    edge_mem_available_mb=float(edge_mem_available_mb),
                    config=args,
                )

                predicted_candidates = batch_eval.build_predicted_candidates(
                    catalog=catalog,
                    assets=assets,
                    args=scheduler_args,
                    prediction_source=args.prediction_source,
                    prediction_lookup=prediction_lookup,
                    prediction_p95_col=args.prediction_p95_col,
                    prediction_tps_col=args.prediction_tps_col,
                )
                actual_candidates = [
                    batch_eval.actualize_candidate(candidate, truth_lookup=truth_lookup, args=scheduler_args)
                    for candidate in predicted_candidates
                ]
                batch_eval.scheduler_core.score_candidates(actual_candidates, scheduler_args)

                case_info = {
                    'device_id': device_id,
                    'prompt_len': int(workload['prompt_len']),
                    'gen_len': int(workload['gen_len']),
                    'concurrency': int(workload['concurrency']),
                    'request_total_tokens': int(workload['prompt_len']) + int(workload['gen_len']),
                    'slo_p95_s': float(slo_p95_s),
                    'edge_mem_available_mb': float(edge_mem_available_mb),
                }

                sched_pred, sched_reason, _ = batch_eval.scheduler_core.choose_candidate(predicted_candidates, scheduler_args)
                sched_actual = get_actual_choice(sched_pred, actual_candidates)
                all_rows.append(
                    build_result_row(
                        policy='scheduler_pred',
                        policy_type='adaptive',
                        seed=None,
                        selection_reason=sched_reason,
                        case_info=case_info,
                        choice=sched_actual,
                        device_rates=device_rates,
                    )
                )

                if learned_bundle is not None:
                    learned_pred, learned_actual, learned_reason = choose_learned_policy(
                        predicted_candidates,
                        actual_candidates,
                        prompt_len=int(workload['prompt_len']),
                        gen_len=int(workload['gen_len']),
                        concurrency=int(workload['concurrency']),
                        slo_p95_s=float(slo_p95_s),
                        edge_mem_available_mb=float(edge_mem_available_mb),
                        edge_device_id=device_id,
                        learned_bundle=learned_bundle,
                    )
                    all_rows.append(
                        build_result_row(
                            policy=str(learned_bundle['policy_name']),
                            policy_type='learned',
                            seed=None,
                            selection_reason=learned_reason,
                            case_info=case_info,
                            choice=learned_actual,
                            device_rates=device_rates,
                        )
                    )

                edf_pred, edf_actual, edf_reason = choose_edf_adapted(predicted_candidates, actual_candidates)
                all_rows.append(
                    build_result_row(
                        policy='edf_adapted',
                        policy_type='baseline',
                        seed=None,
                        selection_reason=edf_reason,
                        case_info=case_info,
                        choice=edf_actual,
                        device_rates=device_rates,
                    )
                )

                slack_pred, slack_actual, slack_reason = choose_latency_slack_first(predicted_candidates, actual_candidates)
                all_rows.append(
                    build_result_row(
                        policy='latency_slack_first',
                        policy_type='baseline',
                        seed=None,
                        selection_reason=slack_reason,
                        case_info=case_info,
                        choice=slack_actual,
                        device_rates=device_rates,
                    )
                )

                mef_pred, mef_actual, mef_reason = choose_mef_adapted(predicted_candidates, actual_candidates)
                all_rows.append(
                    build_result_row(
                        policy='mef_adapted',
                        policy_type='baseline',
                        seed=None,
                        selection_reason=mef_reason,
                        case_info=case_info,
                        choice=mef_actual,
                        device_rates=device_rates,
                    )
                )

                robust_mef_pred, robust_mef_actual, robust_mef_reason = choose_robust_mef_adapted(
                    predicted_candidates,
                    actual_candidates,
                )
                all_rows.append(
                    build_result_row(
                        policy='robust_mef_adapted',
                        policy_type='baseline',
                        seed=None,
                        selection_reason=robust_mef_reason,
                        case_info=case_info,
                        choice=robust_mef_actual,
                        device_rates=device_rates,
                    )
                )

                pfs_pred, pfs_actual, pfs_reason = choose_pfs_adapted(
                    predicted_candidates,
                    actual_candidates,
                    slo_p95_s=float(slo_p95_s),
                )
                all_rows.append(
                    build_result_row(
                        policy='pfs_adapted',
                        policy_type='baseline',
                        seed=None,
                        selection_reason=pfs_reason,
                        case_info=case_info,
                        choice=pfs_actual,
                        device_rates=device_rates,
                    )
                )

                goodput_pred, goodput_actual, goodput_reason = choose_slo_goodput_greedy(
                    predicted_candidates,
                    actual_candidates,
                    slo_p95_s=float(slo_p95_s),
                )
                all_rows.append(
                    build_result_row(
                        policy='slo_goodput_greedy',
                        policy_type='baseline',
                        seed=None,
                        selection_reason=goodput_reason,
                        case_info=case_info,
                        choice=goodput_actual,
                        device_rates=device_rates,
                    )
                )

                cloud_relative_pred, cloud_relative_actual, cloud_relative_reason = choose_cloud_relative_greedy(
                    predicted_candidates,
                    actual_candidates,
                )
                all_rows.append(
                    build_result_row(
                        policy='cloud_relative_greedy',
                        policy_type='baseline',
                        seed=None,
                        selection_reason=cloud_relative_reason,
                        case_info=case_info,
                        choice=cloud_relative_actual,
                        device_rates=device_rates,
                    )
                )

                llm_pq_pred, llm_pq_actual, llm_pq_reason = choose_llm_pq_adapted(
                    predicted_candidates,
                    actual_candidates,
                )
                all_rows.append(
                    build_result_row(
                        policy='llm_pq_adapted',
                        policy_type='baseline',
                        seed=None,
                        selection_reason=llm_pq_reason,
                        case_info=case_info,
                        choice=llm_pq_actual,
                        device_rates=device_rates,
                    )
                )

                pareto_pred, pareto_actual, pareto_reason = choose_pareto_greedy(predicted_candidates, actual_candidates)
                all_rows.append(
                    build_result_row(
                        policy='pareto_greedy',
                        policy_type='baseline',
                        seed=None,
                        selection_reason=pareto_reason,
                        case_info=case_info,
                        choice=pareto_actual,
                        device_rates=device_rates,
                    )
                )

                melange_pred, melange_actual, melange_reason = choose_melange_adapted(predicted_candidates, actual_candidates, device_rates)
                all_rows.append(
                    build_result_row(
                        policy='melange_adapted',
                        policy_type='baseline',
                        seed=None,
                        selection_reason=melange_reason,
                        case_info=case_info,
                        choice=melange_actual,
                        device_rates=device_rates,
                    )
                )

                static_pred, static_actual, static_reason = choose_best_static(predicted_candidates, actual_candidates, static_config_id)
                all_rows.append(
                    build_result_row(
                        policy='best_static_edge',
                        policy_type='baseline',
                        seed=None,
                        selection_reason=f'{static_reason}:{static_config_id}',
                        case_info=case_info,
                        choice=static_actual,
                        device_rates=device_rates,
                    )
                )

                cloud_actual = batch_eval.find_candidate(actual_candidates, target='cloud', config_id='fp16_bs16_tok2048')
                all_rows.append(
                    build_result_row(
                        policy='cloud_only',
                        policy_type='baseline',
                        seed=None,
                        selection_reason='always_cloud',
                        case_info=case_info,
                        choice=cloud_actual,
                        device_rates=device_rates,
                    )
                )

                oracle_choice, oracle_reason = choose_oracle_cost(actual_candidates, device_rates)
                all_rows.append(
                    build_result_row(
                        policy='oracle_cost_actual',
                        policy_type='oracle',
                        seed=None,
                        selection_reason=oracle_reason,
                        case_info=case_info,
                        choice=oracle_choice,
                        device_rates=device_rates,
                    )
                )

                oracle_tps_choice, oracle_tps_reason = choose_oracle_throughput(actual_candidates)
                all_rows.append(
                    build_result_row(
                        policy='oracle_throughput_actual',
                        policy_type='oracle',
                        seed=None,
                        selection_reason=oracle_tps_reason,
                        case_info=case_info,
                        choice=oracle_tps_choice,
                        device_rates=device_rates,
                    )
                )

                for seed in random_seeds:
                    rng = np.random.default_rng(seed)
                    rand_pred, rand_actual, rand_reason = choose_random_edge(predicted_candidates, actual_candidates, rng)
                    all_rows.append(
                        build_result_row(
                            policy='random_edge',
                            policy_type='baseline',
                            seed=seed,
                            selection_reason=rand_reason,
                            case_info=case_info,
                            choice=rand_actual,
                            device_rates=device_rates,
                        )
                    )

    case_df = pd.DataFrame(all_rows)
    summary_policy = summarize(case_df, ['device_id', 'policy', 'policy_type']).sort_values(['device_id', 'mean_total_cost', 'policy'])
    summary_slo_mem = summarize(case_df, ['device_id', 'policy', 'policy_type', 'slo_p95_s', 'edge_mem_available_mb']).sort_values(['device_id', 'policy', 'slo_p95_s', 'edge_mem_available_mb'])
    random_seed_summary = summarize(
        case_df[case_df['policy'] == 'random_edge'].copy(),
        ['device_id', 'policy', 'seed'],
    ).sort_values(['device_id', 'seed'])

    case_csv = os.path.join(args.outdir, 'baseline_case_results.csv')
    summary_csv = os.path.join(args.outdir, 'baseline_policy_summary.csv')
    summary_detail_csv = os.path.join(args.outdir, 'baseline_policy_summary_by_slo_mem.csv')
    random_csv = os.path.join(args.outdir, 'baseline_random_seed_summary.csv')
    metadata_json = os.path.join(args.outdir, 'baseline_eval_metadata.json')

    case_df.to_csv(case_csv, index=False, encoding='utf-8-sig')
    summary_policy.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    summary_slo_mem.to_csv(summary_detail_csv, index=False, encoding='utf-8-sig')
    random_seed_summary.to_csv(random_csv, index=False, encoding='utf-8-sig')

    with open(metadata_json, 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'devices': devices,
                'slo_list': slo_list,
                'edge_mem_list': edge_mem_list,
                'random_seeds': random_seeds,
                'workload_mode': args.workload_mode,
                'workload_cases': len(workload_cases),
                'static_policy_map': static_policy_map,
                'device_rental_rates': device_rates,
                'prediction_source': args.prediction_source,
                'prediction_table_csv': args.prediction_table_csv,
                'prediction_p95_col': args.prediction_p95_col,
                'prediction_tps_col': args.prediction_tps_col,
                'metrics': {
                    'edge_cost': 'rate(device) * actual_p95_s if routed to edge else 0',
                    'total_cost': 'rate(device) * actual_p95_s for both edge and cloud',
                    'tcr': '1 if routed to edge else 0',
                    'slo_hit_rate': '1 if actual_p95_s <= slo_p95_s else 0',
                },
                'baselines': {
                    'random_edge': 'Randomly choose among runtime-feasible edge candidates; fallback to cloud if no feasible edge exists.',
                    'edf_adapted': 'Choose the minimum predicted latency candidate among SLO-feasible candidates; otherwise best-effort minimum latency.',
                    'mef_adapted': 'Choose the maximum predicted throughput candidate among SLO-feasible candidates; otherwise best-effort minimum latency.',
                    'robust_mef_adapted': 'Choose maximum predicted throughput after risk-adjusted SLO admission; otherwise minimum risk-adjusted latency.',
                    'pfs_adapted': 'Past-Future Scheduler inspired baseline: maximize a memory- and SLA-headroom weighted future-safe throughput score.',
                    'slo_goodput_greedy': 'Maximize T_hat * min(1, SLO / risk-adjusted latency) over runtime-feasible candidates.',
                    'cloud_relative_greedy': 'Use a risk-safe edge candidate only when its predicted throughput is no lower than the cloud reference.',
                    'llm_pq_adapted': 'Per-request LLM-PQ approximation: maximize throughput over SLO-feasible quantized edge configurations with cloud fallback; phase partition and quality constraints are outside this simulator.',
                    'melange_adapted': 'Choose the minimum predicted rental cost candidate among SLO-feasible candidates; otherwise best-effort minimum latency.',
                    'best_static_edge': 'Fixed strong static edge deployment selected a priori per device.',
                    'scheduler_pred': 'Current predictor-driven multi-objective policy.',
                    'dagger_policy': 'Learned policy distilled from the balanced heuristic with DAgger.',
                    'bc_policy': 'Behavior-cloned policy distilled from the balanced heuristic.',
                    'oracle_cost_actual': 'Cost-optimal selection using ground-truth measured performance.',
                    'oracle_throughput_actual': 'Throughput-optimal SLO-feasible selection using ground-truth measured performance.',
                },
                'manifest_rows_after_clean': manifest.get('rows_after_clean'),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps({'case_csv': case_csv, 'rows': int(len(case_df))}, ensure_ascii=False))
    print(json.dumps({'summary_csv': summary_csv, 'summary_rows': int(len(summary_policy))}, ensure_ascii=False))
    print(json.dumps({'random_csv': random_csv, 'random_rows': int(len(random_seed_summary))}, ensure_ascii=False))


if __name__ == '__main__':
    main()



