from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_3.csv"
DEFAULT_RAW_CSV = PP_ROOT / "data" / "public_workloads" / "burstgpt" / "BurstGPT_without_fails_3.csv"
DEFAULT_TRACE_CSV = PP_ROOT / "data" / "workload_traces" / "burstgpt_without_fails_3_bucketed.csv"
DEFAULT_MANIFEST = PP_ROOT / "data" / "public_workloads" / "burstgpt" / "manifest.json"
DEFAULT_BUCKETS = (16, 48, 96, 192)


def parse_int_list(raw_value: str) -> List[int]:
    values: List[int] = []
    for item in str(raw_value or "").split(","):
        text = item.strip()
        if text:
            values.append(int(text))
    return values


def snap_to_bucket(value: int, buckets: Sequence[int]) -> int:
    numeric = max(int(value), 1)
    return min((int(bucket) for bucket in buckets), key=lambda bucket: abs(bucket - numeric))


def download_file(url: str, output_path: Path, *, overwrite: bool = False) -> bool:
    if output_path.exists() and not overwrite:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "tepqos-burstgpt-prep"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with output_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    return True


def build_bucketed_trace(
    *,
    input_csv: Path,
    output_csv: Path,
    length_buckets: Sequence[int],
    max_rows: Optional[int],
    dataset_name: str,
) -> Dict[str, object]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trace_id",
        "source_dataset",
        "prompt_len_raw",
        "gen_len_raw",
        "prompt_len",
        "gen_len",
        "concurrency",
        "arrival_ts",
    ]
    prompt_buckets: set[int] = set()
    gen_buckets: set[int] = set()
    input_rows = 0
    output_rows = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        with output_csv.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                input_rows += 1
                if max_rows is not None and output_rows >= int(max_rows):
                    break
                try:
                    prompt_len_raw = int(float(row["Request tokens"]))
                    gen_len_raw = int(float(row["Response tokens"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if prompt_len_raw <= 0 or gen_len_raw <= 0:
                    continue

                prompt_len = snap_to_bucket(prompt_len_raw, length_buckets)
                gen_len = snap_to_bucket(gen_len_raw, length_buckets)
                prompt_buckets.add(prompt_len)
                gen_buckets.add(gen_len)
                writer.writerow(
                    {
                        "trace_id": output_rows,
                        "source_dataset": dataset_name,
                        "prompt_len_raw": prompt_len_raw,
                        "gen_len_raw": gen_len_raw,
                        "prompt_len": prompt_len,
                        "gen_len": gen_len,
                        "concurrency": 1,
                        "arrival_ts": row.get("Timestamp", ""),
                    }
                )
                output_rows += 1

    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "input_rows_scanned": input_rows,
        "output_rows": output_rows,
        "length_buckets": list(length_buckets),
        "distinct_prompt_buckets": sorted(prompt_buckets),
        "distinct_gen_buckets": sorted(gen_buckets),
    }


def write_manifest(path: Path, summary: Dict[str, object], *, source_url: str) -> None:
    manifest = {
        "dataset": "BurstGPT",
        "variant": "BurstGPT_without_fails_3.csv",
        "source_url": source_url,
        "paper": "https://arxiv.org/abs/2401.17644",
        "repository": "https://github.com/HPMLL/BurstGPT",
        "license": "CC-BY-4.0",
        "raw_csv": str(DEFAULT_RAW_CSV),
        "trace_csv": str(DEFAULT_TRACE_CSV),
        "field_mapping": {
            "Timestamp": "arrival_ts",
            "Request tokens": "prompt_len_raw",
            "Response tokens": "gen_len_raw",
            "bucketed Request tokens": "prompt_len",
            "bucketed Response tokens": "gen_len",
            "fixed value 1": "concurrency",
        },
        "summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BurstGPT without-fails trace and build a bucketed workload trace for TE-PQoS evaluation.",
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-csv", default=str(DEFAULT_RAW_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_TRACE_CSV))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--length-buckets", default="16,48,96,192")
    parser.add_argument("--dataset-name", default="BurstGPT")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_csv = Path(args.raw_csv)
    output_csv = Path(args.output_csv)
    buckets = parse_int_list(args.length_buckets) or list(DEFAULT_BUCKETS)
    downloaded = False
    if args.download:
        downloaded = download_file(args.url, raw_csv, overwrite=bool(args.overwrite))
    if not raw_csv.exists():
        raise FileNotFoundError(f"missing raw CSV: {raw_csv}")

    summary = build_bucketed_trace(
        input_csv=raw_csv,
        output_csv=output_csv,
        length_buckets=buckets,
        max_rows=args.max_rows,
        dataset_name=str(args.dataset_name),
    )
    summary["downloaded"] = downloaded
    summary["raw_size_bytes"] = os.path.getsize(raw_csv)
    write_manifest(Path(args.manifest), summary, source_url=str(args.url))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
