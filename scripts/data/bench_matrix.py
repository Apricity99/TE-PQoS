import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional


DEFAULT_COLUMNS = [
    "timestamp",
    "model_id",
    "device_id",
    "config_id",
    "prompt_len",
    "gen_len",
    "concurrency",
    "warmup",
    "rounds",
    "total_reqs",
    "success_reqs",
    "error_reqs",
    "timeout_reqs",
    "oom_flag",
    "lat_avg_s",
    "lat_p50_s",
    "lat_p95_s",
    "tok_per_s",
    "prompt_tokens_total",
    "completion_tokens_total",
    "peak_mem_mb",
]


def parse_int_list(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_json_dict(raw: str) -> Dict[str, object]:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("value must decode to a JSON object")
    return data


def percentile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (pos - lower)


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def make_prompt(target_len: int) -> str:
    words = [f"tok{i % 1024}" for i in range(target_len)]
    return " ".join(words)


def read_response_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return str(exc)
    return body.decode("utf-8", errors="replace")


def detect_timeout(message: str) -> bool:
    message = message.lower()
    return "timeout" in message or "timed out" in message


def detect_oom(message: str) -> bool:
    message = message.lower()
    candidates = [
        "out of memory",
        "cuda out of memory",
        "cuda error: out of memory",
        "oom",
    ]
    return any(token in message for token in candidates)


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    timeout: bool
    oom: bool
    error: str = ""


class GPUMemorySampler:
    def __init__(self, gpu_index: int, interval_s: float) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.peak_mem_mb: Optional[int] = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            current = self._read_memory_used()
            if current is not None:
                if self.peak_mem_mb is None or current > self.peak_mem_mb:
                    self.peak_mem_mb = current
            self._stop_event.wait(self.interval_s)

    def _read_memory_used(self) -> Optional[int]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
        except Exception:
            return None

        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if self.gpu_index >= len(lines):
            return None
        try:
            return int(float(lines[self.gpu_index]))
        except ValueError:
            return None


def build_request_body(
    endpoint_mode: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    extra_body: Dict[str, object],
) -> Dict[str, object]:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if endpoint_mode == "chat":
        body["messages"] = [{"role": "user", "content": prompt_text}]
    else:
        body["prompt"] = prompt_text
    body.update(extra_body)
    return body


def call_inference_api(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, object],
    timeout_s: float,
    fallback_prompt_tokens: int,
    fallback_completion_tokens: int,
) -> RequestResult:
    started = time.perf_counter()
    payload_bytes = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=payload_bytes,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
        latency_s = time.perf_counter() - started
        payload = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        latency_s = time.perf_counter() - started
        message = read_response_body(exc)
        return RequestResult(
            ok=False,
            latency_s=latency_s,
            prompt_tokens=fallback_prompt_tokens,
            completion_tokens=0,
            timeout=detect_timeout(message),
            oom=detect_oom(message),
            error=message,
        )
    except Exception as exc:
        latency_s = time.perf_counter() - started
        message = str(exc)
        return RequestResult(
            ok=False,
            latency_s=latency_s,
            prompt_tokens=fallback_prompt_tokens,
            completion_tokens=0,
            timeout=detect_timeout(message),
            oom=detect_oom(message),
            error=message,
        )

    usage = payload.get("usage") if isinstance(payload, dict) else None
    prompt_tokens = fallback_prompt_tokens
    completion_tokens = fallback_completion_tokens
    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
        completion_tokens = int(usage.get("completion_tokens", completion_tokens))

    return RequestResult(
        ok=True,
        latency_s=latency_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        timeout=False,
        oom=False,
    )

def resolve_output_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def write_rows(path: str, rows: List[Dict[str, object]]) -> None:
    resolved_path = resolve_output_path(path)
    parent_dir = os.path.dirname(resolved_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    file_exists = os.path.exists(resolved_path)
    with open(resolved_path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_COLUMNS)
        if not file_exists or os.path.getsize(resolved_path) == 0:
            writer.writeheader()
        writer.writerows(rows)


def run_case(
    *,
    url: str,
    headers: Dict[str, str],
    endpoint_mode: str,
    model: str,
    device_id: str,
    config_id: str,
    prompt_len: int,
    gen_len: int,
    concurrency: int,
    warmup: int,
    rounds: int,
    timeout_s: float,
    temperature: float,
    extra_body: Dict[str, object],
    memory_sampler: Optional[GPUMemorySampler],
) -> Dict[str, object]:
    prompt_text = make_prompt(prompt_len)
    request_body = build_request_body(
        endpoint_mode=endpoint_mode,
        model=model,
        prompt_text=prompt_text,
        max_tokens=gen_len,
        temperature=temperature,
        extra_body=extra_body,
    )

    for _ in range(warmup):
        call_inference_api(
            url=url,
            headers=headers,
            body=request_body,
            timeout_s=timeout_s,
            fallback_prompt_tokens=prompt_len,
            fallback_completion_tokens=gen_len,
        )

    latencies: List[float] = []
    success_reqs = 0
    error_reqs = 0
    timeout_reqs = 0
    oom_flag = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    elapsed_batches_s = 0.0

    for _ in range(rounds):
        batch_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    call_inference_api,
                    url,
                    headers,
                    request_body,
                    timeout_s,
                    prompt_len,
                    gen_len,
                )
                for _ in range(concurrency)
            ]
            results = [future.result() for future in futures]
        elapsed_batches_s += time.perf_counter() - batch_started

        for result in results:
            if result.ok:
                latencies.append(result.latency_s)
                success_reqs += 1
                prompt_tokens_total += result.prompt_tokens
                completion_tokens_total += result.completion_tokens
            else:
                error_reqs += 1
                timeout_reqs += int(result.timeout)
                oom_flag = max(oom_flag, int(result.oom))

    total_reqs = rounds * concurrency
    tok_per_s = (
        completion_tokens_total / elapsed_batches_s
        if elapsed_batches_s > 0 and completion_tokens_total > 0
        else math.nan
    )

    row = {
        "timestamp": iso_now(),
        "model_id": model,
        "device_id": device_id,
        "config_id": config_id,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "concurrency": concurrency,
        "warmup": warmup,
        "rounds": rounds,
        "total_reqs": total_reqs,
        "success_reqs": success_reqs,
        "error_reqs": error_reqs,
        "timeout_reqs": timeout_reqs,
        "oom_flag": oom_flag,
        "lat_avg_s": statistics.fmean(latencies) if latencies else math.nan,
        "lat_p50_s": percentile(latencies, 0.50),
        "lat_p95_s": percentile(latencies, 0.95),
        "tok_per_s": tok_per_s,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "peak_mem_mb": memory_sampler.peak_mem_mb if memory_sampler else "",
    }
    return row

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an OpenAI-compatible inference endpoint over a workload grid.",
    )
    parser.add_argument("--base-url", required=True, help="Server base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--endpoint-mode", choices=["chat", "completions"], default="chat")
    parser.add_argument("--model", required=True, help="Model name exposed by the server")
    parser.add_argument("--device-id", required=True, help="Tag for the current device, e.g. edge_4060")
    parser.add_argument("--config-id", required=True, help="Tag for the current compression config")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    parser.add_argument("--prompt-lens", type=parse_int_list, default=[16, 48, 96, 192])
    parser.add_argument("--gen-lens", type=parse_int_list, default=[16, 48, 96, 192])
    parser.add_argument("--concurrency-list", type=parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--extra-body-json", type=parse_json_dict, default={})
    parser.add_argument("--extra-headers-json", type=parse_json_dict, default={})
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--mem-poll-interval-s", type=float, default=0.2)
    parser.add_argument("--disable-mem-sampling", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    endpoint_suffix = "/v1/chat/completions" if args.endpoint_mode == "chat" else "/v1/completions"
    url = args.base_url.rstrip("/") + endpoint_suffix
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    headers.update({str(k): str(v) for k, v in args.extra_headers_json.items()})

    rows: List[Dict[str, object]] = []
    for prompt_len in args.prompt_lens:
        for gen_len in args.gen_lens:
            for concurrency in args.concurrency_list:
                sampler = None
                if not args.disable_mem_sampling:
                    sampler = GPUMemorySampler(
                        gpu_index=args.gpu_index,
                        interval_s=args.mem_poll_interval_s,
                    )
                    sampler.start()
                try:
                    row = run_case(
                        url=url,
                        headers=headers,
                        endpoint_mode=args.endpoint_mode,
                        model=args.model,
                        device_id=args.device_id,
                        config_id=args.config_id,
                        prompt_len=prompt_len,
                        gen_len=gen_len,
                        concurrency=concurrency,
                        warmup=args.warmup,
                        rounds=args.rounds,
                        timeout_s=args.timeout_s,
                        temperature=args.temperature,
                        extra_body=args.extra_body_json,
                        memory_sampler=sampler,
                    )
                finally:
                    if sampler is not None:
                        sampler.stop()

                rows.append(row)
                write_rows(args.out_csv, [row])
                print(
                    json.dumps(
                        {
                            "device_id": row["device_id"],
                            "config_id": row["config_id"],
                            "prompt_len": row["prompt_len"],
                            "gen_len": row["gen_len"],
                            "concurrency": row["concurrency"],
                            "lat_p95_s": row["lat_p95_s"],
                            "tok_per_s": row["tok_per_s"],
                            "success_reqs": row["success_reqs"],
                            "error_reqs": row["error_reqs"],
                            "peak_mem_mb": row["peak_mem_mb"],
                        },
                        ensure_ascii=True,
                    )
                )

    print(f"saved {len(rows)} rows to {resolve_output_path(args.out_csv)}")


if __name__ == "__main__":
    main()


