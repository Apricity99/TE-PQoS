import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import run_scheduler_batch_eval as batch_eval
import scheduler as scheduler_core


DEFAULT_DEVICES = ("edge_3090", "edge_4090")
DEFAULT_EDGE_MEMORY_MB = {
    "edge_3090": 10000.0,
    "edge_4090": 10000.0,
}
DEFAULT_SLO_LEVELS = (0.75, 1.0, 2.0, 4.0)


@dataclass
class ActionSpec:
    action_idx: int
    target: str
    device_id: str
    config_id: str
    quant_method: str
    batch_size: int
    kv_strategy: str
    memory_mb: Optional[float]


class RLSchedulingEnv:
    def __init__(
        self,
        *,
        artifacts_dir: str = 'artifacts_gbdt_multi',
        catalog_path: str = 'scheduler_catalog.json',
        prediction_table_csv: str = 'artifacts_gbdt_multi/combined_perf_with_pred.csv',
        devices: Sequence[str] = DEFAULT_DEVICES,
        edge_memory_mb: Optional[Dict[str, float]] = None,
        slo_levels: Sequence[float] = DEFAULT_SLO_LEVELS,
        episode_length: int = 64,
        arrival_gap_s: float = 0.20,
        cloud_extra_delay_s: float = 0.0,
        switch_overhead_s: float = 0.02,
        reward_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.catalog_path = catalog_path
        self.prediction_table_csv = prediction_table_csv
        self.devices = tuple(devices)
        self.edge_memory_mb = dict(edge_memory_mb or DEFAULT_EDGE_MEMORY_MB)
        self.slo_levels = tuple(float(value) for value in slo_levels)
        self.episode_length = int(episode_length)
        self.arrival_gap_s = float(arrival_gap_s)
        self.cloud_extra_delay_s = float(cloud_extra_delay_s)
        self.switch_overhead_s = float(switch_overhead_s)
        self.reward_weights = {
            'throughput': 1.0,
            'latency': 0.75,
            'violation': 3.5,
            'cloud': 0.03,
            'switch': 0.12,
            'queue': 0.45,
            'success_bonus': 0.08,
        }
        if reward_weights:
            self.reward_weights.update({k: float(v) for k, v in reward_weights.items()})

        self.assets = scheduler_core.load_assets(self.artifacts_dir)
        self.catalog = batch_eval.load_json(self.catalog_path)
        _, self.truth_lookup = batch_eval.load_truth_tables(self.artifacts_dir)
        self.prediction_lookup = batch_eval.load_prediction_lookup(self.prediction_table_csv)

        self.edge_candidates = [
            candidate
            for candidate in self.catalog.get('edge_candidates', [])
            if bool(candidate.get('enabled', True))
            and bool(candidate.get('edge_deployable', True))
            and str(candidate.get('device_id')) in self.devices
        ]
        self.cloud_candidates = [
            candidate for candidate in self.catalog.get('cloud_candidates', []) if bool(candidate.get('enabled', True))
        ]
        if not self.cloud_candidates:
            raise ValueError('cloud candidate is required for RL scheduling env')

        self.action_specs = self._build_action_specs()
        self.action_index = {(spec.target, spec.device_id, spec.config_id): spec.action_idx for spec in self.action_specs}
        self.request_catalog = self._build_request_catalog()
        if not self.request_catalog:
            raise ValueError('request catalog is empty; cannot train RL scheduler')

        self.device_config_order: Dict[str, List[str]] = {}
        for device_id in self.devices:
            configs = [spec.config_id for spec in self.action_specs if spec.target == 'edge' and spec.device_id == device_id]
            self.device_config_order[device_id] = configs

        self.max_prompt_len = max(int(item['prompt_len']) for item in self.request_catalog)
        self.max_gen_len = max(int(item['gen_len']) for item in self.request_catalog)
        self.max_concurrency = max(int(item['concurrency']) for item in self.request_catalog)
        self.max_total_tokens = max(int(item['prompt_len']) + int(item['gen_len']) for item in self.request_catalog)
        self.max_latency_norm = max(self.slo_levels) * 2.0
        self.max_tps_norm = max(
            float(candidate['predicted_tok_per_s'])
            for request in self.request_catalog
            for candidate in request['candidates_by_action'].values()
        )
        self.max_tps_norm = max(self.max_tps_norm, 1.0)

        self.current_time = 0.0
        self.current_step = 0
        self.rng = np.random.default_rng(0)
        self.episode_requests: List[Dict[str, Any]] = []
        self.edge_state: Dict[str, Dict[str, Any]] = {}
        self.last_info: Dict[str, Any] = {}

    def _build_action_specs(self) -> List[ActionSpec]:
        specs: List[ActionSpec] = []
        action_idx = 0
        for candidate in self.edge_candidates:
            specs.append(
                ActionSpec(
                    action_idx=action_idx,
                    target='edge',
                    device_id=str(candidate['device_id']),
                    config_id=str(candidate['config_id']),
                    quant_method=str(candidate.get('quant_method', 'unknown')),
                    batch_size=int(candidate.get('batch_size') or 0),
                    kv_strategy=str(candidate.get('kv_strategy', 'unknown')),
                    memory_mb=float(candidate['memory_mb']) if candidate.get('memory_mb') is not None else None,
                )
            )
            action_idx += 1
        for candidate in self.cloud_candidates:
            specs.append(
                ActionSpec(
                    action_idx=action_idx,
                    target='cloud',
                    device_id=str(candidate['device_id']),
                    config_id=str(candidate['config_id']),
                    quant_method=str(candidate.get('quant_method', 'unknown')),
                    batch_size=int(candidate.get('batch_size') or 0),
                    kv_strategy=str(candidate.get('kv_strategy', 'unknown')),
                    memory_mb=None,
                )
            )
            action_idx += 1
        return specs

    def _make_args(self, *, prompt_len: int, gen_len: int, concurrency: int, slo_p95_s: float, edge_device_id: str) -> argparse.Namespace:
        return SimpleNamespace(
            edge_device_id=edge_device_id,
            prompt_len=int(prompt_len),
            gen_len=int(gen_len),
            concurrency=int(concurrency),
            slo_p95_s=float(slo_p95_s),
            edge_mem_available_mb=float(self.edge_memory_mb.get(edge_device_id, 10000.0)),
            edge_queue_delay_s=0.0,
            cloud_extra_delay_s=float(self.cloud_extra_delay_s),
            latency_weight=1.0,
            memory_weight=0.3,
            throughput_weight=0.2,
            cloud_route_penalty=0.8,
            slo_safety_margin=1.0,
            soft_slo_tolerance_ratio=0.05,
            tie_cost_tolerance_ratio=0.05,
            violation_tolerance_ratio=0.03,
            memory_penalty_threshold=0.85,
            reject_extrapolation=False,
            no_prefer_edge=False,
            optimize_for='balanced',
        )

    def _cloud_candidate_for_request(self, *, prompt_len: int, gen_len: int, concurrency: int, slo_p95_s: float) -> Dict[str, Any]:
        cloud_cfg = copy.deepcopy(self.cloud_candidates[0])
        args = self._make_args(
            prompt_len=prompt_len,
            gen_len=gen_len,
            concurrency=concurrency,
            slo_p95_s=slo_p95_s,
            edge_device_id=self.devices[0],
        )
        candidate = scheduler_core.predict_cloud_candidate(cloud_cfg, args)
        candidate['actual_p95_s'] = float(candidate['predicted_p95_s'])
        candidate['actual_tok_per_s'] = float(candidate['predicted_tok_per_s'])
        candidate['actual_status'] = 'ok'
        return candidate

    def _build_request_catalog(self) -> List[Dict[str, Any]]:
        frame = pd.read_csv(self.prediction_table_csv, encoding='utf-8-sig')
        frame = frame[frame['device_id'].isin(self.devices)].copy()
        request_catalog: List[Dict[str, Any]] = []

        workload_rows = frame[['prompt_len', 'gen_len', 'concurrency']].drop_duplicates().sort_values(['prompt_len', 'gen_len', 'concurrency'])
        for row in workload_rows.to_dict(orient='records'):
            prompt_len = int(row['prompt_len'])
            gen_len = int(row['gen_len'])
            concurrency = int(row['concurrency'])
            for slo_p95_s in self.slo_levels:
                candidates_by_action: Dict[int, Dict[str, Any]] = {}
                for spec in self.action_specs:
                    if spec.target == 'cloud':
                        cloud_candidate = self._cloud_candidate_for_request(
                            prompt_len=prompt_len,
                            gen_len=gen_len,
                            concurrency=concurrency,
                            slo_p95_s=float(slo_p95_s),
                        )
                        candidates_by_action[spec.action_idx] = cloud_candidate
                        continue

                    key = (spec.device_id, spec.config_id, prompt_len, gen_len, concurrency)
                    lookup_row = self.prediction_lookup.get(key)
                    truth_row = self.truth_lookup.get(key)
                    if lookup_row is None or truth_row is None or not bool(truth_row.get('success', False)):
                        continue

                    args = self._make_args(
                        prompt_len=prompt_len,
                        gen_len=gen_len,
                        concurrency=concurrency,
                        slo_p95_s=float(slo_p95_s),
                        edge_device_id=spec.device_id,
                    )
                    candidate_cfg = next(
                        item for item in self.edge_candidates
                        if str(item['device_id']) == spec.device_id and str(item['config_id']) == spec.config_id
                    )
                    candidate = batch_eval.predict_edge_candidate_from_table(
                        candidate=candidate_cfg,
                        args=args,
                        assets=self.assets,
                        lookup_row=lookup_row,
                        p95_col='p95_calibrated_s',
                        tps_col='tps_hat',
                    )
                    candidate['actual_p95_s'] = float(truth_row['lat_p95_s'])
                    candidate['actual_tok_per_s'] = float(truth_row['tok_per_s'])
                    candidate['actual_status'] = 'ok'
                    candidates_by_action[spec.action_idx] = candidate

                if candidates_by_action:
                    request_catalog.append(
                        {
                            'prompt_len': prompt_len,
                            'gen_len': gen_len,
                            'concurrency': concurrency,
                            'slo_p95_s': float(slo_p95_s),
                            'candidates_by_action': candidates_by_action,
                        }
                    )
        return request_catalog

    @property
    def observation_dim(self) -> int:
        request_dims = 5
        device_dims = len(self.devices) * (2 + max(len(configs) for configs in self.device_config_order.values()))
        action_dims = len(self.action_specs) * 5
        return request_dims + device_dims + action_dims

    @property
    def action_dim(self) -> int:
        return len(self.action_specs)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_time = 0.0
        self.current_step = 0
        self.edge_state = {
            device_id: {
                'available_at': 0.0,
                'active_config': None,
                'accepted': 0,
            }
            for device_id in self.devices
        }
        indices = self.rng.choice(len(self.request_catalog), size=self.episode_length, replace=True)
        self.episode_requests = [copy.deepcopy(self.request_catalog[int(idx)]) for idx in indices]
        self.last_info = {}
        return self._build_observation(), {'action_mask': self.action_mask()}

    def current_request(self) -> Dict[str, Any]:
        return self.episode_requests[self.current_step]

    def action_mask(self) -> np.ndarray:
        request = self.current_request()
        mask = np.zeros(self.action_dim, dtype=np.float32)
        for spec in self.action_specs:
            candidate = request['candidates_by_action'].get(spec.action_idx)
            if candidate is None:
                continue
            if spec.target == 'edge':
                capacity = float(self.edge_memory_mb.get(spec.device_id, 0.0))
                memory_mb = float(spec.memory_mb or 0.0)
                if memory_mb > capacity + 1e-9:
                    continue
            mask[spec.action_idx] = 1.0
        return mask

    def _device_queue_delay(self, device_id: str) -> float:
        return max(float(self.edge_state[device_id]['available_at']) - self.current_time, 0.0)

    def _switch_flag(self, device_id: str, config_id: str) -> float:
        active = self.edge_state[device_id]['active_config']
        return 1.0 if active is not None and active != config_id else 0.0

    def _build_observation(self) -> np.ndarray:
        request = self.current_request()
        prompt_len = float(request['prompt_len'])
        gen_len = float(request['gen_len'])
        concurrency = float(request['concurrency'])
        slo_p95_s = float(request['slo_p95_s'])
        request_features = [
            prompt_len / max(self.max_prompt_len, 1.0),
            gen_len / max(self.max_gen_len, 1.0),
            concurrency / max(self.max_concurrency, 1.0),
            (prompt_len + gen_len) / max(self.max_total_tokens, 1.0),
            slo_p95_s / max(self.slo_levels),
        ]

        device_features: List[float] = []
        max_configs = max(len(configs) for configs in self.device_config_order.values())
        for device_id in self.devices:
            queue_delay = self._device_queue_delay(device_id)
            device_features.append(min(queue_delay / self.max_latency_norm, 2.0))
            device_features.append(1.0 if queue_delay > 0.0 else 0.0)
            active_config = self.edge_state[device_id]['active_config']
            configs = self.device_config_order[device_id]
            one_hot = [0.0] * max_configs
            if active_config in configs:
                one_hot[configs.index(active_config)] = 1.0
            device_features.extend(one_hot)

        action_features: List[float] = []
        mask = self.action_mask()
        for spec in self.action_specs:
            candidate = request['candidates_by_action'].get(spec.action_idx)
            if candidate is None or mask[spec.action_idx] <= 0.0:
                action_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            queue_delay = self._device_queue_delay(spec.device_id) if spec.target == 'edge' else 0.0
            predicted_completion = queue_delay + float(candidate['predicted_p95_s'])
            switch_flag = self._switch_flag(spec.device_id, spec.config_id) if spec.target == 'edge' else 0.0
            action_features.extend(
                [
                    1.0,
                    min(predicted_completion / self.max_latency_norm, 2.0),
                    float(candidate['predicted_tok_per_s']) / self.max_tps_norm,
                    (float(spec.memory_mb or 0.0) / max(self.edge_memory_mb.get(spec.device_id, 1.0), 1.0)) if spec.target == 'edge' else 0.0,
                    switch_flag,
                ]
            )

        return np.asarray(request_features + device_features + action_features, dtype=np.float32)

    def build_single_request_input(
        self,
        *,
        prompt_len: int,
        gen_len: int,
        concurrency: int,
        slo_p95_s: float,
        predicted_candidates: Sequence[Dict[str, Any]],
        edge_memory_override: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, Any]]]:
        request = {
            'prompt_len': int(prompt_len),
            'gen_len': int(gen_len),
            'concurrency': int(concurrency),
            'slo_p95_s': float(slo_p95_s),
            'candidates_by_action': {},
        }
        by_key = {
            (str(candidate.get('target')), str(candidate.get('device_id')), str(candidate.get('config_id'))): candidate
            for candidate in predicted_candidates
        }
        for spec in self.action_specs:
            candidate = by_key.get((spec.target, spec.device_id, spec.config_id))
            if candidate is not None:
                request['candidates_by_action'][spec.action_idx] = candidate

        old_memory = dict(self.edge_memory_mb)
        try:
            if edge_memory_override:
                self.edge_memory_mb.update({str(k): float(v) for k, v in edge_memory_override.items()})
            self.current_time = 0.0
            self.current_step = 0
            self.episode_requests = [request]
            self.edge_state = {
                device_id: {
                    'available_at': 0.0,
                    'active_config': None,
                    'accepted': 0,
                }
                for device_id in self.devices
            }
            obs = self._build_observation()
            mask = self.action_mask()
            return obs, mask, dict(request['candidates_by_action'])
        finally:
            self.edge_memory_mb = old_memory

    def _fallback_action(self, mask: np.ndarray) -> int:
        cloud_actions = [spec.action_idx for spec in self.action_specs if spec.target == 'cloud' and mask[spec.action_idx] > 0.0]
        if cloud_actions:
            return cloud_actions[0]
        valid_actions = np.flatnonzero(mask > 0.0)
        if len(valid_actions) == 0:
            return 0
        return int(valid_actions[0])

    def _reward(self, *, target: str, completion_latency_s: float, actual_tps: float, slo_p95_s: float, switched: bool, queue_delay_s: float) -> Tuple[float, Dict[str, float]]:
        violation_s = max(completion_latency_s - slo_p95_s, 0.0)
        throughput_term = self.reward_weights['throughput'] * (actual_tps / self.max_tps_norm)
        latency_term = self.reward_weights['latency'] * (completion_latency_s / max(slo_p95_s, 1e-6))
        violation_ratio = violation_s / max(slo_p95_s, 1e-6)
        violation_term = self.reward_weights['violation'] * (violation_ratio ** 2)
        queue_term = self.reward_weights['queue'] * (queue_delay_s / max(slo_p95_s, 1e-6))
        cloud_term = self.reward_weights['cloud'] if target == 'cloud' else 0.0
        switch_term = self.reward_weights['switch'] if switched else 0.0
        success_bonus = self.reward_weights['success_bonus'] if violation_s <= 0.0 else 0.0
        reward = throughput_term - latency_term - violation_term - queue_term - cloud_term - switch_term + success_bonus
        return reward, {
            'throughput_term': throughput_term,
            'latency_term': latency_term,
            'violation_term': violation_term,
            'queue_term': queue_term,
            'cloud_term': cloud_term,
            'switch_term': switch_term,
            'success_bonus': success_bonus,
            'violation_s': violation_s,
        }

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        request = self.current_request()
        mask = self.action_mask()
        invalid_action = bool(action_idx < 0 or action_idx >= self.action_dim or mask[action_idx] <= 0.0)
        if invalid_action:
            action_idx = self._fallback_action(mask)

        spec = self.action_specs[action_idx]
        candidate = request['candidates_by_action'][action_idx]
        target = spec.target
        queue_delay_s = self._device_queue_delay(spec.device_id) if target == 'edge' else 0.0
        switched = bool(target == 'edge' and self._switch_flag(spec.device_id, spec.config_id) > 0.0)
        switch_overhead_s = self.switch_overhead_s if switched else 0.0
        actual_service_s = float(candidate['actual_p95_s'])
        completion_latency_s = queue_delay_s + switch_overhead_s + actual_service_s
        actual_tps = float(candidate['actual_tok_per_s'])

        if target == 'edge':
            available_at = max(self.current_time, float(self.edge_state[spec.device_id]['available_at']))
            self.edge_state[spec.device_id]['available_at'] = available_at + switch_overhead_s + actual_service_s
            self.edge_state[spec.device_id]['active_config'] = spec.config_id
            self.edge_state[spec.device_id]['accepted'] += 1

        reward, reward_terms = self._reward(
            target=target,
            completion_latency_s=completion_latency_s,
            actual_tps=actual_tps,
            slo_p95_s=float(request['slo_p95_s']),
            switched=switched,
            queue_delay_s=queue_delay_s,
        )

        info = {
            'request': {
                'prompt_len': int(request['prompt_len']),
                'gen_len': int(request['gen_len']),
                'concurrency': int(request['concurrency']),
                'slo_p95_s': float(request['slo_p95_s']),
            },
            'action_idx': int(action_idx),
            'action_name': f'{spec.device_id}:{spec.config_id}',
            'target': target,
            'invalid_action': invalid_action,
            'queue_delay_s': queue_delay_s,
            'switch_overhead_s': switch_overhead_s,
            'completion_latency_s': completion_latency_s,
            'actual_tok_per_s': actual_tps,
            'slo_hit': bool(completion_latency_s <= float(request['slo_p95_s'])),
            'reward_terms': reward_terms,
            'current_time_s': self.current_time,
        }

        self.current_time += self.arrival_gap_s
        self.current_step += 1
        done = self.current_step >= self.episode_length
        self.last_info = info
        if done:
            next_obs = np.zeros(self.observation_dim, dtype=np.float32)
        else:
            next_obs = self._build_observation()
            info['next_action_mask'] = self.action_mask()
        return next_obs, float(reward), done, info

    def heuristic_action(self, mode: str) -> int:
        mode = str(mode)
        request = self.current_request()
        mask = self.action_mask()
        valid_actions = [spec.action_idx for spec in self.action_specs if mask[spec.action_idx] > 0.0]
        if not valid_actions:
            return self._fallback_action(mask)

        candidates = []
        for action_idx in valid_actions:
            spec = self.action_specs[action_idx]
            candidate = request['candidates_by_action'][action_idx]
            queue_delay = self._device_queue_delay(spec.device_id) if spec.target == 'edge' else 0.0
            predicted_completion = queue_delay + float(candidate['predicted_p95_s'])
            candidates.append((action_idx, spec, candidate, predicted_completion))

        slo = float(request['slo_p95_s'])
        on_time = [item for item in candidates if item[3] <= slo + 1e-9]
        pool = on_time if on_time else candidates

        if mode == 'cloud_only':
            return self._fallback_action(mask)
        if mode == 'latency':
            return min(pool, key=lambda item: (item[3], -float(item[2]['predicted_tok_per_s'])))[0]
        if mode == 'throughput':
            return max(pool, key=lambda item: (float(item[2]['predicted_tok_per_s']), -item[3]))[0]
        if mode == 'random':
            return int(self.rng.choice([item[0] for item in pool]))

        def balanced_key(item: Tuple[int, ActionSpec, Dict[str, Any], float]) -> Tuple[float, float, float]:
            _, spec, candidate, predicted_completion = item
            tps = float(candidate['predicted_tok_per_s']) / self.max_tps_norm
            lat = predicted_completion / max(slo, 1e-6)
            score = 0.55 * tps - 0.45 * lat + (0.02 if spec.target == 'edge' else 0.0)
            return (score, float(candidate.get('pairwise_rank_score', 0.0)), -predicted_completion)

        return max(pool, key=balanced_key)[0]


def build_env_from_args(args: argparse.Namespace) -> RLSchedulingEnv:
    edge_memory_mb: Dict[str, float] = {}
    for item in str(args.edge_memory_mb or '').split(','):
        text = item.strip()
        if not text or '=' not in text:
            continue
        key, value = text.split('=', 1)
        edge_memory_mb[key.strip()] = float(value.strip())
    devices = tuple(text.strip() for text in str(args.devices).split(',') if text.strip())
    reward_weights = {
        'throughput': float(args.reward_throughput_weight),
        'latency': float(args.reward_latency_weight),
        'violation': float(args.reward_violation_weight),
        'cloud': float(args.reward_cloud_penalty),
        'switch': float(args.reward_switch_penalty),
        'queue': float(args.reward_queue_penalty),
        'success_bonus': float(args.reward_success_bonus),
    }
    return RLSchedulingEnv(
        artifacts_dir=args.artifacts_dir,
        catalog_path=args.catalog,
        prediction_table_csv=args.prediction_table_csv,
        devices=devices,
        edge_memory_mb=edge_memory_mb or None,
        episode_length=int(args.episode_length),
        arrival_gap_s=float(args.arrival_gap_s),
        cloud_extra_delay_s=float(args.cloud_extra_delay_s),
        switch_overhead_s=float(args.switch_overhead_s),
        reward_weights=reward_weights,
    )
