import unittest

from scripts.evaluation import run_scheduler_policy_baselines as baselines


def candidate(
    *,
    target: str,
    config_id: str,
    latency: float,
    throughput: float,
    risk_latency: float,
    within_slo: bool,
    within_risk_slo: bool,
):
    return {
        "target": target,
        "device_id": "cloud_a100" if target == "cloud" else "edge_3090",
        "config_id": config_id,
        "predicted_p95_s": latency,
        "predicted_tok_per_s": throughput,
        "risk_adjusted_p95_s": risk_latency,
        "within_slo": within_slo,
        "within_risk_slo": within_risk_slo,
        "within_memory": True,
        "within_token_budget": True,
        "within_batch_budget": True,
        "memory_pressure": 0.5,
    }


class SchedulerBaselineTests(unittest.TestCase):
    def setUp(self):
        self.edge_fast = candidate(
            target="edge",
            config_id="edge_fast",
            latency=0.9,
            throughput=200.0,
            risk_latency=1.1,
            within_slo=True,
            within_risk_slo=False,
        )
        self.edge_safe = candidate(
            target="edge",
            config_id="edge_safe",
            latency=0.8,
            throughput=150.0,
            risk_latency=0.9,
            within_slo=True,
            within_risk_slo=True,
        )
        self.cloud = candidate(
            target="cloud",
            config_id="fp16_bs16_tok2048",
            latency=0.7,
            throughput=140.0,
            risk_latency=0.7,
            within_slo=True,
            within_risk_slo=True,
        )
        self.predicted = [self.edge_fast, self.edge_safe, self.cloud]
        self.actual = [dict(item) for item in self.predicted]

    def test_robust_mef_filters_risk_unsafe_edge(self):
        predicted, _, _ = baselines.choose_robust_mef_adapted(self.predicted, self.actual)
        self.assertEqual(predicted["config_id"], "edge_safe")

    def test_cloud_relative_uses_safe_edge_when_it_beats_cloud(self):
        predicted, _, _ = baselines.choose_cloud_relative_greedy(self.predicted, self.actual)
        self.assertEqual(predicted["config_id"], "edge_safe")

    def test_llm_pq_uses_fastest_raw_slo_feasible_edge(self):
        predicted, _, _ = baselines.choose_llm_pq_adapted(self.predicted, self.actual)
        self.assertEqual(predicted["config_id"], "edge_fast")

    def test_goodput_proxy_is_deterministic(self):
        predicted, _, _ = baselines.choose_slo_goodput_greedy(
            self.predicted,
            self.actual,
            slo_p95_s=1.0,
        )
        self.assertEqual(predicted["config_id"], "edge_fast")


if __name__ == "__main__":
    unittest.main()
