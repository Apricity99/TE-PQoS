import argparse
import json
import os

from tepqos.workloads import DEFAULT_LENGTH_BUCKETS, build_bucketed_trace, parse_int_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a public workload CSV into a bucketed trace aligned with the profiled prompt/gen grid.",
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--prompt-col", default="prompt_len_raw")
    parser.add_argument("--gen-col", default="gen_len_raw")
    parser.add_argument("--concurrency-col", default=None)
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--length-buckets", default="16,48,96,192")
    parser.add_argument("--default-concurrency", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    buckets = parse_int_list(args.length_buckets) or list(DEFAULT_LENGTH_BUCKETS)
    summary = build_bucketed_trace(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        prompt_col=args.prompt_col,
        gen_col=args.gen_col,
        concurrency_col=args.concurrency_col,
        timestamp_col=args.timestamp_col,
        dataset_name=args.dataset_name,
        length_buckets=buckets,
        default_concurrency=args.default_concurrency,
        max_rows=args.max_rows,
    )
    print(json.dumps({"output_csv": os.path.abspath(args.output_csv), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
