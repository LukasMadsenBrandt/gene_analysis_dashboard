#!/usr/bin/env python3
"""Run the isolated sample-level Kutsche/ZEB2 regression experiment."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.experiments.zeb2_regression import (  # noqa: E402
    load_experiment_config,
    run_zeb2_regression_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit sample-level ordered regression models between ZEB2 and every "
            "other gene in the configured Kutsche expression matrix."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file.")
    parser.add_argument("--gene", help="Override the configured anchor gene (default: ZEB2).")
    parser.add_argument(
        "--quadratic-p-threshold",
        type=float,
        help="Override the methodological quadratic-term p-value threshold.",
    )
    parser.add_argument("--output", help="Override the output CSV path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_experiment_config(args.config)
    overrides = {}
    if args.gene:
        overrides["anchor_gene"] = args.gene
    if args.quadratic_p_threshold is not None:
        overrides["quadratic_p_threshold"] = args.quadratic_p_threshold
    if args.output:
        overrides["output_file"] = Path(args.output)
    if overrides:
        config = replace(config, **overrides)
    manifest = run_zeb2_regression_experiment(config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
