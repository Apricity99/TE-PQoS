import csv
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_LENGTH_BUCKETS = (16, 48, 96, 192)


def parse_int_list(raw_value: str) -> List[int]:
    values: List[int] = []
    for item in str(raw_value or "").split(","):
        text = item.strip()
        if not text:
            continue
        values.append(int(text))
    return values


def snap_to_bucket(value: int, buckets: Sequence[int]) -> int:
    if not buckets:
        raise ValueError("buckets cannot be empty")
    numeric = max(int(value), 1)
    return min((int(bucket) for bucket in buckets), key=lambda bucket: abs(bucket - numeric))


def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_bucketed_trace(
    *,
    input_csv: str,
    output_csv: str,
    prompt_col: str,
    gen_col: str,
    concurrency_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    dataset_name: str = "",
    length_buckets: Sequence[int] = DEFAULT_LENGTH_BUCKETS,
    default_concurrency: int = 1,
    max_rows: Optional[int] = None,
) -> Dict[str, int]:
    rows = _read_rows(input_csv)
    bucketed_rows: List[Dict[str, object]] = []

    for index, row in enumerate(rows):
        if max_rows is not None and len(bucketed_rows) >= int(max_rows):
            break

        try:
            prompt_len_raw = int(float(row[prompt_col]))
            gen_len_raw = int(float(row[gen_col]))
        except (KeyError, TypeError, ValueError):
            continue

        if prompt_len_raw <= 0 or gen_len_raw <= 0:
            continue

        concurrency = default_concurrency
        if concurrency_col:
            try:
                concurrency = int(float(row.get(concurrency_col, default_concurrency)))
            except (TypeError, ValueError):
                concurrency = default_concurrency
        concurrency = max(concurrency, 1)

        bucketed_rows.append(
            {
                "trace_id": len(bucketed_rows),
                "source_dataset": dataset_name or os.path.basename(input_csv),
                "prompt_len_raw": prompt_len_raw,
                "gen_len_raw": gen_len_raw,
                "prompt_len": snap_to_bucket(prompt_len_raw, length_buckets),
                "gen_len": snap_to_bucket(gen_len_raw, length_buckets),
                "concurrency": concurrency,
                "arrival_ts": row.get(timestamp_col, "") if timestamp_col else "",
            }
        )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
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
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bucketed_rows)

    return {
        "input_rows": len(rows),
        "output_rows": len(bucketed_rows),
        "distinct_prompt_buckets": len({int(row["prompt_len"]) for row in bucketed_rows}),
        "distinct_gen_buckets": len({int(row["gen_len"]) for row in bucketed_rows}),
    }


def supported_workload_set(
    truth_lookup: Dict[Tuple[str, str, int, int, int], Dict[str, object]],
    device_id: str,
) -> set[Tuple[int, int, int]]:
    tuples: set[Tuple[int, int, int]] = set()
    for (dev, _cfg, prompt_len, gen_len, concurrency), row in truth_lookup.items():
        if dev != device_id:
            continue
        if bool(row.get("success", False)):
            tuples.add((int(prompt_len), int(gen_len), int(concurrency)))
    return tuples


def load_trace_workload_cases(
    *,
    trace_csv: str,
    devices: Sequence[str],
    truth_lookup: Dict[Tuple[str, str, int, int, int], Dict[str, object]],
    duplicate_across_devices: bool = True,
    drop_unsupported: bool = True,
    max_cases: Optional[int] = None,
) -> List[Dict[str, int]]:
    rows = _read_rows(trace_csv)
    supported_by_device = {
        str(device_id): supported_workload_set(truth_lookup, str(device_id))
        for device_id in devices
    }

    cases: List[Dict[str, int]] = []
    for row in rows:
        try:
            prompt_len = int(float(row["prompt_len"]))
            gen_len = int(float(row["gen_len"]))
            concurrency = max(int(float(row.get("concurrency", 1))), 1)
        except (KeyError, TypeError, ValueError):
            continue

        target_devices: Iterable[str]
        if duplicate_across_devices:
            target_devices = devices
        else:
            raw_device = str(row.get("device_id", "")).strip()
            if not raw_device:
                continue
            target_devices = [raw_device]

        for device_id in target_devices:
            key = (prompt_len, gen_len, concurrency)
            if drop_unsupported and key not in supported_by_device.get(str(device_id), set()):
                continue
            cases.append(
                {
                    "device_id": str(device_id),
                    "prompt_len": prompt_len,
                    "gen_len": gen_len,
                    "concurrency": concurrency,
                }
            )
            if max_cases is not None and len(cases) >= int(max_cases):
                return cases
    return cases
