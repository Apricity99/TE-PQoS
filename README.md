# TE-PQoS

Source release for **TE-PQoS: Throughput-Efficient Predictive QoS-Aware Scheduling for Private Edge-Cloud LLM Serving**.

This repository contains the prediction, calibration, ranking, cloud-reference, scheduling, baseline, workload-preparation, and evaluation code used by the paper. It is intentionally source-only: raw GPU profiling data, large public datasets, serialized models, experiment outputs, and LaTeX materials are not committed here.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py'
```

To inspect the reproducibility commands without running an experiment:

```bash
python experiments/run_unified_dual_experiments.py show --strict-same-branch
```

The full experiment pipeline requires the profiling tables and workload traces described in `docs/REPRODUCIBILITY.md`.

## Main components

- `src/tepqos/features.py`: workload, configuration, calibration, and ranking features.
- `src/tepqos/training.py`: p95-latency/throughput predictors, residual calibration, and pairwise ranking.
- `src/tepqos/scheduler.py`: risk-aware candidate filtering, tiered SLO admission, edge rescue, and cloud fallback.
- `scripts/training/`: predictor training, prediction, and model-baseline commands.
- `scripts/evaluation/`: controlled-workload, trace-replay, batch, Qwen, and sensitivity commands.
- `scripts/data/`: profiling and public-workload preparation commands.
- `scripts/analysis/` and `scripts/plotting/`: paper analysis and visualization utilities.
- `experiments/run_unified_dual_experiments.py`: unified launcher for the paper's two principal evaluation regimes.

## Data and artifacts

Do not commit private profiling data, raw conversation datasets, model files, or generated outputs. Put them in the local paths documented in `docs/REPRODUCIBILITY.md`; these paths are ignored by `.gitignore`.

Public workload provenance is documented in:

- `data/public_workloads/burstgpt/README.md`
- `data/public_workloads/wildchat/README.md`

The scripts can rebuild derived traces from locally obtained data. Dataset licenses and access conditions remain those of the upstream providers.

## License

No software license has been selected in this source snapshot. Add the authors' chosen license before making the GitHub repository public.
