import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, Iterable, List, Optional

import polars as pl
import tiktoken

from public_workload_trace import build_bucketed_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract assistant-turn workload traces from WildChat parquet shards.",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-raw-csv", required=True)
    parser.add_argument("--output-bucketed-csv", required=True)
    parser.add_argument("--max-records", type=int, default=50000)
    parser.add_argument("--length-buckets", default="16,48,96,192")
    parser.add_argument("--encoding-name", default="cl100k_base")
    return parser.parse_args()


def approximate_token_count(text: str) -> int:
    if not text:
        return 0
    pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(len(pieces), max(len(text) // 4, 1))


def count_tokens(encoding, text: str) -> int:
    if not text:
        return 0
    if encoding is None:
        return approximate_token_count(text)
    return len(encoding.encode(text))


def normalize_message(message: object) -> Optional[Dict[str, str]]:
    if isinstance(message, dict):
        role = str(message.get("role", "") or "").strip()
        content = str(message.get("content", "") or "")
        if not role or not content:
            return None
        return {"role": role, "content": content}
    return None


def iter_conversations(input_dir: str) -> Iterable[Dict[str, object]]:
    pattern = os.path.join(input_dir, "data", "*.parquet")
    for parquet_path in sorted(glob.glob(pattern)):
        frame = pl.read_parquet(parquet_path).select(["conversation_id", "timestamp", "conversation"])
        for row in frame.iter_rows(named=True):
            yield row


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_raw_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_bucketed_csv), exist_ok=True)

    encoding = None
    tokenizer_mode = "approximate"
    try:
        encoding = tiktoken.get_encoding(args.encoding_name)
        tokenizer_mode = args.encoding_name
    except Exception:
        encoding = None
    max_records = int(args.max_records) if args.max_records is not None else None
    length_buckets = [int(item.strip()) for item in str(args.length_buckets).split(",") if item.strip()]

    raw_fieldnames = [
        "trace_id",
        "conversation_id",
        "assistant_turn_index",
        "timestamp",
        "prompt_len_raw",
        "gen_len_raw",
        "source_dataset",
    ]

    written = 0
    with open(args.output_raw_csv, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fieldnames)
        writer.writeheader()

        for row in iter_conversations(args.input_dir):
            conversation = row.get("conversation")
            if conversation is None:
                continue

            history_parts: List[str] = []
            assistant_turn_index = 0

            for message in conversation:
                normalized = normalize_message(message)
                if normalized is None:
                    continue

                role = normalized["role"].lower()
                content = normalized["content"]

                if role == "assistant":
                    prompt_text = "\n".join(history_parts).strip()
                    prompt_len_raw = count_tokens(encoding, prompt_text)
                    gen_len_raw = count_tokens(encoding, content)
                    if prompt_len_raw > 0 and gen_len_raw > 0:
                        writer.writerow(
                            {
                                "trace_id": written,
                                "conversation_id": row.get("conversation_id", ""),
                                "assistant_turn_index": assistant_turn_index,
                                "timestamp": row.get("timestamp", ""),
                                "prompt_len_raw": prompt_len_raw,
                                "gen_len_raw": gen_len_raw,
                                "source_dataset": "WildChat",
                            }
                        )
                        written += 1
                        if max_records is not None and written >= max_records:
                            break
                    assistant_turn_index += 1

                history_parts.append(f"{role}: {content}")

            if max_records is not None and written >= max_records:
                break

    summary = build_bucketed_trace(
        input_csv=args.output_raw_csv,
        output_csv=args.output_bucketed_csv,
        prompt_col="prompt_len_raw",
        gen_col="gen_len_raw",
        concurrency_col=None,
        timestamp_col="timestamp",
        dataset_name="WildChat",
        length_buckets=length_buckets,
        default_concurrency=1,
        max_rows=None,
    )

    print(
        json.dumps(
            {
                "output_raw_csv": os.path.abspath(args.output_raw_csv),
                "output_bucketed_csv": os.path.abspath(args.output_bucketed_csv),
                "assistant_turn_records": written,
                "tokenizer_mode": tokenizer_mode,
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
