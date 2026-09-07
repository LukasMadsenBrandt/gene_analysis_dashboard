# ZEB2 sample-level regression experiment

This experiment is independent of the Granger-causality pipeline. It does not
change the pipeline default or its outputs.

Run it from the repository root:

```bash
python scripts/experiments/zeb2_regression.py \
  --config configs/experiments/zeb2_regression.kutsche.yml
```

The default output is
`results/experiments/zeb2_regression/zeb2_regression_results.csv`; a run manifest
is written beside it as `zeb2_regression_results.manifest.json`.

## Data and time representation

The default input is `Data/Kutsche/genes_all.txt`. Its 35 WT sample columns are
retained without replicate aggregation: seven samples at each of the labels
`d0`, `d1`, `d2`, `d3`, and `d4`. The explicit numeric mapping is `d0 -> 0.0`,
`d1 -> 1.0`, `d2 -> 2.0`, `d3 -> 3.0`, and `d4 -> 4.0`. Predictor and target
values are selected from the same ordered sample columns, and the code verifies
their sample indexes before fitting.

The shared Kutsche loader removes genes that are zero in all WT samples. With
the repository's current `genes_all.txt`, this reduces 2,321 genes to 2,311,
including ZEB2, and produces 4,620 ordered result rows.

## Configuration

`quadratic_p_threshold` controls selection between the target-only linear and
quadratic time models. It is a methodological parameter, not a finalized
scientific choice. Normalization and transformation are also explicit in the
experiment YAML. No observations are silently removed for missing or non-finite
values; validation fails and reports the affected data instead.

The CSV contains all successful tests regardless of predictor p-value. It does
not apply an edge threshold or FDR correction and does not assign causal meaning
to an ordered relationship.
