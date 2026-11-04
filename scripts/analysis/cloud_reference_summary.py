from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "scheduler_catalog.json"
DEFAULT_REPORT_DIR = ROOT / "outputs/reports/20260714_reviewer2_reproducibility_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the analytic cloud reference.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    cloud = catalog["cloud_candidates"][0]
    model = cloud["perf_model"]
    upload_rate = 1.0 / model["prompt_upload_per_token_s"]
    download_rate = 1.0 / model["response_download_per_token_s"]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cloud Reference Settings", "",
        "The analytic cloud candidate is an A100-style FP16 reference with batch size 16 and a 2048-token budget.",
        "Its latency is base + RTT + prompt upload + prefill + concurrency-adjusted decode + response download.", "",
        "| Setting | Value |",
        "|---|---:|",
        f"| Base latency | {model['base_latency_s']:.3f} s |",
        f"| Network RTT | {model['network_rtt_s']:.3f} s |",
        f"| Prompt upload coefficient | {model['prompt_upload_per_token_s']:.5f} s/token |",
        f"| Effective prompt upload rate | {upload_rate:.1f} token/s |",
        f"| Response download coefficient | {model['response_download_per_token_s']:.5f} s/token |",
        f"| Effective response download rate | {download_rate:.1f} token/s |",
        f"| Prefill coefficient | {model['prefill_per_token_s']:.5f} s/token |",
        f"| Base decode throughput | {model['decode_tokens_per_s']:.1f} token/s |",
        f"| Per-additional-request concurrency penalty | {model['concurrency_penalty']:.2f} |",
        f"| Relative rental rate | {cloud['rental_rate']:.1f} |", "",
        "The coefficients are analytic effective transfer rates, not measurements of a named provider or physical link in Mbps.",
        "Service-effective throughput is generation length divided by total end-to-end latency.",
        "Cost is relative rental rate multiplied by realized service time.",
    ]
    report = args.report_dir / "CLOUD_REFERENCE_SETTINGS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
