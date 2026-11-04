# Reproducibility notes

## Required local inputs

The source release does not redistribute large or restricted inputs. Prepare the following local files before running the paper-scale experiments:

```text
data/benchmarks/<device>/<quantization>/*.csv
data/workload_traces/wildchat_turn_trace_short_bucketed.csv
data/workload_traces/burstgpt_stratified500_bucketed.csv
```

The benchmark CSVs must contain the columns expected by `perf_feature_utils.py`, including request shape, device/configuration identity, measured p95 latency, throughput, success counts, and peak memory. The exact source-group layout used by the paper is encoded in `train_gbdt_multi.py` and `scripts/evaluation/run_qwen_scale_v1.py`.

## Preparing public traces

BurstGPT can be downloaded and bucketed from its upstream release:

```bash
python scripts/data/prepare_burstgpt_workload.py --download
python scripts/data/sample_workload_trace.py
```

WildChat extraction requires a locally obtained copy under the upstream dataset terms. Use the extraction utilities in the repository and write only derived traces to `data/workload_traces/`.

## Training and evaluation

Show the exact commands first:

```bash
python experiments/run_unified_dual_experiments.py show --strict-same-branch
```

Run the unified pipeline after the required inputs are available:

```bash
python experiments/run_unified_dual_experiments.py all --strict-same-branch
```

For a direct controlled run, the essential sequence is:

```bash
python train_gbdt_multi.py --outdir outputs/runs/controlled/models
python run_scheduler_policy_baselines.py \
  --artifacts-dir outputs/runs/controlled/models \
  --prediction-table-csv outputs/runs/controlled/models/combined_perf_with_pred.csv \
  --outdir outputs/runs/controlled/results \
  --catalog scheduler_catalog.json \
  --prediction-source table \
  --disable-learned-policy \
  --optimize-for throughput_constrained \
  --cloud-throughput-mode decode_only \
  --throughput-tie-ratio 0.03
```

Every new run should use a new output directory containing the command, configuration, metadata, metrics, and generated figures. Do not overwrite a paper result directory.

## Validation

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q .
```

The reported paper numbers depend on the exact profiling tables, trace versions, software versions, and experiment settings. A source checkout alone is not a claim that the full results can be regenerated without those inputs.
