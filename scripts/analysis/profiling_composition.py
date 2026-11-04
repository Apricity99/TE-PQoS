from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "final/unified_scheduler_dual_experiments_v1/models/main_controlled/combined_perf_dataset.csv"
DEFAULT_REPORT_DIR = ROOT / "outputs/reports/20260714_reviewer2_reproducibility_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the cleaned profiling corpus.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def values(frame: pd.DataFrame, column: str) -> str:
    return ", ".join(str(int(v)) for v in sorted(frame[column].dropna().unique()))


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    group_cols = [
        "source_tag", "device_id", "quant_method", "quant_bits",
        "batch_size", "max_tokens", "kv_strategy",
    ]
    groups = (
        frame.groupby(group_cols, dropna=False).size().rename("successful_samples")
        .reset_index().sort_values(["device_id", "quant_method", "batch_size"])
    )
    request_shapes = frame[["prompt_len", "gen_len", "concurrency"]].drop_duplicates()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    groups.to_csv(args.report_dir / "PROFILING_DATA_COMPOSITION.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Profiling Data Composition", "",
        f"Input: `{args.input_csv.relative_to(ROOT)}`", "",
        f"The cleaned corpus contains {len(frame)} successful samples from {len(groups)} source groups and {len(request_shapes)} distinct request shapes.",
        "The predictor corpus includes one legacy RTX 4060 AWQ group; the main scheduler comparison is restricted to RTX 3090 and RTX 4090.", "",
        f"Prompt lengths: {values(frame, 'prompt_len')} tokens.",
        f"Generation lengths: {values(frame, 'gen_len')} tokens.",
        f"Concurrency levels: {values(frame, 'concurrency')}.", "",
        "| Source group | Device | Quant. | Bits | Batch | Token budget | KV | Samples |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for row in groups.to_dict(orient="records"):
        lines.append(
            "| {source_tag} | {device_id} | {quant_method} | {quant_bits:.0f} | "
            "{batch_size:.0f} | {max_tokens:.0f} | {kv_strategy} | {successful_samples} |".format(**row)
        )
    lines += ["", "Invalid targets, non-positive measurements, error rows, and timeouts are excluded before this summary."]
    report = args.report_dir / "PROFILING_DATA_COMPOSITION.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
