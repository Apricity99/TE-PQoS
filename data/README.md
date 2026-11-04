# Data layout

Large or restricted data are intentionally excluded from the GitHub source release.

Use these local directories when running experiments:

- `benchmarks/<device>/<quantization>/`: measured profiling tables;
- `public_workloads/<dataset>/`: locally obtained public data and provenance notes;
- `workload_traces/`: derived, bucketed traces used by the evaluator.

See `docs/REPRODUCIBILITY.md` and the dataset-specific README files for provenance and preparation commands.
