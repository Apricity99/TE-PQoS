from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PP_ROOT / "data" / "workload_traces" / "burstgpt_without_fails_3_bucketed.csv"
DEFAULT_OUTPUT = PP_ROOT / "data" / "workload_traces" / "burstgpt_stratified500_bucketed.csv"
DEFAULT_MANIFEST = PP_ROOT / "data" / "workload_traces" / "burstgpt_stratified500_manifest.json"


def parse_columns(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def row_key(row: Dict[str, str], columns: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row[column]) for column in columns)


def count_strata(input_csv: Path, strata: Sequence[str]) -> Tuple[List[str], Counter[Tuple[str, ...]]]:
    counts: Counter[Tuple[str, ...]] = Counter()
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [column for column in strata if column not in fieldnames]
        if missing:
            raise ValueError(f"missing strata columns: {missing}")
        for row in reader:
            counts[row_key(row, strata)] += 1
    return fieldnames, counts


def allocate_samples(
    counts: Counter[Tuple[str, ...]],
    sample_size: int,
) -> Dict[Tuple[str, ...], int]:
    total_rows = sum(counts.values())
    target_size = min(max(int(sample_size), 0), total_rows)
    if target_size == 0:
        return {key: 0 for key in counts}

    ideal = {key: target_size * count / total_rows for key, count in counts.items()}
    allocation = {key: min(int(math.floor(value)), counts[key]) for key, value in ideal.items()}
    remaining = target_size - sum(allocation.values())

    # Largest-remainder allocation preserves the global stratum distribution.
    order = sorted(counts, key=lambda key: (-(ideal[key] - allocation[key]), key))
    while remaining > 0:
        progressed = False
        for key in order:
            if allocation[key] >= counts[key]:
                continue
            allocation[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("unable to allocate requested sample size")
    return allocation


def evenly_spaced_positions(count: int, target: int) -> set[int]:
    if target <= 0:
        return set()
    if target >= count:
        return set(range(count))
    return {min(int((index + 0.5) * count / target), count - 1) for index in range(target)}


def build_sample(
    *,
    input_csv: Path,
    output_csv: Path,
    fieldnames: Sequence[str],
    strata: Sequence[str],
    counts: Counter[Tuple[str, ...]],
    allocation: Dict[Tuple[str, ...], int],
) -> Counter[Tuple[str, ...]]:
    selected_positions = {
        key: evenly_spaced_positions(counts[key], allocation.get(key, 0)) for key in counts
    }
    seen: Dict[Tuple[str, ...], int] = defaultdict(int)
    sampled: Counter[Tuple[str, ...]] = Counter()
    output_rows = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        with output_csv.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                key = row_key(row, strata)
                position = seen[key]
                seen[key] += 1
                if position not in selected_positions[key]:
                    continue
                if "trace_id" in row:
                    row["trace_id"] = str(output_rows)
                writer.writerow(row)
                sampled[key] += 1
                output_rows += 1
    return sampled


def serialize_counts(counts: Iterable[Tuple[Tuple[str, ...], int]], strata: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, count in sorted(counts):
        rows.append({**dict(zip(strata, key)), "rows": int(count)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, proportionally stratified workload-trace sample.",
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--strata", default="prompt_len,gen_len")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    strata = parse_columns(args.strata)
    if not strata:
        raise ValueError("at least one stratum column is required")
    if not input_csv.exists():
        raise FileNotFoundError(f"missing input CSV: {input_csv}")

    fieldnames, counts = count_strata(input_csv, strata)
    allocation = allocate_samples(counts, args.sample_size)
    sampled = build_sample(
        input_csv=input_csv,
        output_csv=output_csv,
        fieldnames=fieldnames,
        strata=strata,
        counts=counts,
        allocation=allocation,
    )
    manifest = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "method": "proportional_largest_remainder_within_stratum_even_spacing",
        "strata": strata,
        "source_rows": int(sum(counts.values())),
        "sample_rows": int(sum(sampled.values())),
        "source_strata": serialize_counts(counts.items(), strata),
        "sample_strata": serialize_counts(sampled.items(), strata),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
