import argparse
import csv
import json
import os
from typing import Dict, List, Optional

from public_workload_trace import DEFAULT_LENGTH_BUCKETS, build_bucketed_trace, parse_int_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a public workload trace into a short-text subset and build a bucketed replay trace.",
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--raw-output-csv", required=True)
    parser.add_argument("--bucketed-output-csv", required=True)
    parser.add_argument("--prompt-col", default="prompt_len_raw")
    parser.add_argument("--gen-col", default="gen_len_raw")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--dataset-name", default="short_trace")
    parser.add_argument("--length-buckets", default="16,48,96,192")
    parser.add_argument("--default-concurrency", type=int, default=1)
    parser.add_argument("--max-prompt-len", type=int, default=96)
    parser.add_argument("--max-gen-len", type=int, default=96)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(csv_path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    outdir = os.path.dirname(csv_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _passes_filter(
    row: Dict[str, str],
    *,
    prompt_col: str,
    gen_col: str,
    max_prompt_len: int,
    max_gen_len: int,
    max_total_tokens: Optional[int],
) -> bool:
    try:
        prompt_len = int(float(row[prompt_col]))
        gen_len = int(float(row[gen_col]))
    except (KeyError, TypeError, ValueError):
        return False

    if prompt_len <= 0 or gen_len <= 0:
        return False
    if prompt_len > max_prompt_len or gen_len > max_gen_len:
        return False
    if max_total_tokens is not None and prompt_len + gen_len > max_total_tokens:
        return False
    return True


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input_csv)
    buckets = parse_int_list(args.length_buckets) or list(DEFAULT_LENGTH_BUCKETS)

    filtered_rows: List[Dict[str, str]] = []
    for row in rows:
        if not _passes_filter(
            row,
            prompt_col=args.prompt_col,
            gen_col=args.gen_col,
            max_prompt_len=int(args.max_prompt_len),
            max_gen_len=int(args.max_gen_len),
            max_total_tokens=args.max_total_tokens,
        ):
            continue
        filtered_rows.append(row)
        if args.max_rows is not None and len(filtered_rows) >= int(args.max_rows):
            break

    fieldnames = list(filtered_rows[0].keys()) if filtered_rows else list(rows[0].keys()) if rows else []
    _write_rows(args.raw_output_csv, filtered_rows, fieldnames)

    bucket_summary = build_bucketed_trace(
        input_csv=args.raw_output_csv,
        output_csv=args.bucketed_output_csv,
        prompt_col=args.prompt_col,
        gen_col=args.gen_col,
        concurrency_col=None,
        timestamp_col=args.timestamp_col if args.timestamp_col else None,
        dataset_name=args.dataset_name,
        length_buckets=buckets,
        default_concurrency=int(args.default_concurrency),
        max_rows=None,
    )

    summary = {
        "input_csv": os.path.abspath(args.input_csv),
        "raw_output_csv": os.path.abspath(args.raw_output_csv),
        "bucketed_output_csv": os.path.abspath(args.bucketed_output_csv),
        "input_rows": len(rows),
        "filtered_rows": len(filtered_rows),
        "max_prompt_len": int(args.max_prompt_len),
        "max_gen_len": int(args.max_gen_len),
        "max_total_tokens": args.max_total_tokens,
        "bucket_input_rows": int(bucket_summary.get("input_rows", 0)),
        "bucket_output_rows": int(bucket_summary.get("output_rows", 0)),
        "distinct_prompt_buckets": int(bucket_summary.get("distinct_prompt_buckets", 0)),
        "distinct_gen_buckets": int(bucket_summary.get("distinct_gen_buckets", 0)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
