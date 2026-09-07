"""Sample-level Kutsche regression experiment anchored on ZEB2.

This module is intentionally separate from the Granger-causality pipeline.  It
retains all seven WT samples at each Kutsche time point and estimates ordered
predictor-target associations without interpreting them as causal effects.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from gene_analysis.datasets.kutsche import filter_data_wt, load_and_preprocess_data
from gene_analysis.io.paths import data_path, results_path


LOGGER = logging.getLogger(__name__)

# Kutsche sample labels are d0, d1, d2, d3, and d4.  Use those observed day
# numbers directly as the deterministic numeric regression variable.
KUTSCHE_DAY_TO_TIME: dict[int, float] = {
    0: 0.0,
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 4.0,
}
EXPECTED_KUTSCHE_OBSERVATIONS = 35
EXPECTED_REPLICATES_PER_TIME = 7

RESULT_COLUMNS = [
    "target_gene",
    "predictor_gene",
    "n_observations",
    "selected_time_model",
    "linear_time_r_squared",
    "linear_time_sse",
    "quadratic_time_r_squared",
    "quadratic_time_sse",
    "quadratic_term_coefficient",
    "quadratic_term_p_value",
    "final_model_r_squared",
    "final_model_sse",
    "predictor_coefficient",
    "predictor_p_value",
    "predictor_t_statistic",
    "adjusted_r_squared",
    "residual_degrees_of_freedom",
    "status",
    "error",
]


@dataclass(frozen=True)
class Zeb2RegressionConfig:
    """Configuration for the isolated Kutsche/ZEB2 experiment.

    ``quadratic_p_threshold`` is a methodological parameter exposed for
    investigation; its default is not a finalized scientific choice.
    """

    dataset: str = "kutsche"
    expression_file: Path = data_path("Kutsche", "genes_all.txt")
    anchor_gene: str = "ZEB2"
    quadratic_p_threshold: float = 0.05
    normalize: str = "none"
    transform: str = "none"
    output_file: Path = results_path(
        "experiments", "zeb2_regression", "zeb2_regression_results.csv"
    )

    def validate(self) -> None:
        if self.dataset.lower() != "kutsche":
            raise ValueError("This experiment currently supports only the Kutsche dataset.")
        if not self.anchor_gene:
            raise ValueError("anchor_gene is required.")
        if not 0 < float(self.quadratic_p_threshold) <= 1:
            raise ValueError("quadratic_p_threshold must be in (0, 1].")
        if str(self.normalize).lower() not in {
            "none", "deseq", "deseq2", "size_factors", "zscore", "z-score"
        }:
            raise ValueError("normalize must be one of: none, deseq, zscore.")
        if str(self.transform).lower() not in {"none", "log1p", "log+1", "sqrt"}:
            raise ValueError("transform must be one of: none, log1p, sqrt.")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Zeb2RegressionConfig":
        preprocessing = raw.get("preprocessing") or {}
        if not isinstance(preprocessing, Mapping):
            raise ValueError("preprocessing must be a mapping.")
        return cls(
            dataset=str(raw.get("dataset", "kutsche")),
            expression_file=Path(raw.get("expression_file", data_path("Kutsche", "genes_all.txt"))),
            anchor_gene=str(raw.get("anchor_gene", "ZEB2")),
            quadratic_p_threshold=float(raw.get("quadratic_p_threshold", 0.05)),
            normalize=str(preprocessing.get("normalize", raw.get("normalize", "none"))),
            transform=str(preprocessing.get("transform", raw.get("transform", "none"))),
            output_file=Path(
                raw.get(
                    "output_file",
                    results_path("experiments", "zeb2_regression", "zeb2_regression_results.csv"),
                )
            ),
        )


@dataclass(frozen=True)
class KutscheSampleData:
    """Aligned genes-by-samples expression and numeric time observations."""

    expression: pd.DataFrame
    time: pd.Series
    day_by_sample: pd.Series


@dataclass(frozen=True)
class _TimeModelFits:
    linear: Any
    quadratic: Any
    selected_time_model: str


def load_experiment_config(path: str | Path) -> Zeb2RegressionConfig:
    """Read a YAML experiment configuration."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Experiment configuration must be a YAML mapping.")
    config = Zeb2RegressionConfig.from_mapping(raw)
    config.validate()
    return config


def load_kutsche_sample_data(
    expression_file: str | Path,
    *,
    normalize: str = "none",
    transform: str = "none",
) -> KutscheSampleData:
    """Load and validate the 35 unaggregated Kutsche WT observations.

    The shared Kutsche filtering and expression preprocessing are reused, but
    replicate aggregation is deliberately not called.
    """
    raw = load_and_preprocess_data(expression_file)
    if raw.index.has_duplicates:
        duplicates = sorted(set(raw.index[raw.index.duplicated()].astype(str)))
        raise ValueError(f"Duplicate gene rows prevent unambiguous alignment: {duplicates[:10]}")

    expression, day_map, sample_columns = filter_data_wt(
        raw,
        transformed=transform,
        normalize=normalize,
    )
    unknown_days = sorted(set(day_map.values()) - set(KUTSCHE_DAY_TO_TIME))
    missing_days = sorted(set(KUTSCHE_DAY_TO_TIME) - set(day_map.values()))
    if unknown_days or missing_days:
        raise ValueError(
            "Kutsche time labels do not match the explicit d0-d4 mapping: "
            f"unknown={unknown_days}, missing={missing_days}."
        )
    if len(sample_columns) != EXPECTED_KUTSCHE_OBSERVATIONS:
        raise ValueError(
            "Expected 35 unaggregated Kutsche WT sample columns, "
            f"found {len(sample_columns)}."
        )
    if len(set(sample_columns)) != len(sample_columns):
        raise ValueError("Kutsche WT sample identifiers must be unique.")

    counts = pd.Series([day_map[column] for column in sample_columns]).value_counts()
    bad_counts = {
        day: int(counts.get(day, 0))
        for day in KUTSCHE_DAY_TO_TIME
        if int(counts.get(day, 0)) != EXPECTED_REPLICATES_PER_TIME
    }
    if bad_counts:
        raise ValueError(
            "Expected seven unaggregated WT samples at every Kutsche time point; "
            f"observed mismatches={bad_counts}."
        )

    expression = expression.loc[:, sample_columns]
    numeric = expression.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        bad_rows = np.flatnonzero(~np.isfinite(numeric).all(axis=1))
        bad_genes = expression.index[bad_rows].astype(str).tolist()
        raise ValueError(
            "Missing or non-finite Kutsche expression values were found; "
            f"no observations were dropped. Affected genes include {bad_genes[:10]}."
        )

    day_by_sample = pd.Series(
        [day_map[column] for column in sample_columns],
        index=sample_columns,
        name="kutsche_day",
        dtype=int,
    )
    time = day_by_sample.map(KUTSCHE_DAY_TO_TIME).rename("time").astype(float)
    return KutscheSampleData(expression=expression, time=time, day_by_sample=day_by_sample)


def validate_aligned_observations(
    predictor: Sequence[float] | pd.Series,
    target: Sequence[float] | pd.Series,
    time: Sequence[float] | pd.Series,
    *,
    expected_observations: int | None = None,
    expected_replicates_per_time: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return predictor, target, and time without dropping rows."""
    lengths = (len(predictor), len(target), len(time))
    if len(set(lengths)) != 1:
        raise ValueError(
            "Predictor, target, and time must have the same observation count; "
            f"found {lengths}."
        )
    if expected_observations is not None and lengths[0] != expected_observations:
        raise ValueError(
            f"Expected {expected_observations} unaggregated observations, found {lengths[0]}."
        )

    labelled = [value for value in (predictor, target, time) if isinstance(value, pd.Series)]
    if labelled:
        if len(labelled) != 3:
            raise ValueError(
                "Predictor, target, and time must all carry sample indexes when any one does."
            )
        if not predictor.index.equals(target.index) or not predictor.index.equals(time.index):
            raise ValueError(
                "Predictor, target, and time sample indexes are not identically aligned."
            )
        if predictor.index.has_duplicates:
            raise ValueError("Observation/sample identifiers must be unique.")

    predictor_values = np.asarray(predictor, dtype=float)
    target_values = np.asarray(target, dtype=float)
    time_values = np.asarray(time, dtype=float)
    for name, values in (
        ("predictor", predictor_values),
        ("target", target_values),
        ("time", time_values),
    ):
        if values.ndim != 1:
            raise ValueError(f"{name} observations must be one-dimensional.")
        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} contains missing or non-finite values; no observations were dropped."
            )

    if expected_replicates_per_time is not None:
        _, counts = np.unique(time_values, return_counts=True)
        if len(counts) == 0 or np.any(counts != expected_replicates_per_time):
            raise ValueError(
                f"Expected {expected_replicates_per_time} observations per numeric time value; "
                f"found counts {counts.tolist()}."
            )
    return predictor_values, target_values, time_values


def _fit_time_models(target: np.ndarray, time: np.ndarray, threshold: float) -> _TimeModelFits:
    linear_design = pd.DataFrame({"const": 1.0, "time": time})
    quadratic_design = pd.DataFrame(
        {"const": 1.0, "time": time, "time_squared": np.square(time)}
    )
    linear = sm.OLS(target, linear_design).fit()
    quadratic = sm.OLS(target, quadratic_design).fit()
    quadratic_p_value = float(quadratic.pvalues["time_squared"])
    selected = (
        "quadratic"
        if np.isfinite(quadratic_p_value) and quadratic_p_value < threshold
        else "linear"
    )
    return _TimeModelFits(linear=linear, quadratic=quadratic, selected_time_model=selected)


def _result_from_fits(
    *,
    predictor: np.ndarray,
    target: np.ndarray,
    time: np.ndarray,
    predictor_gene: str,
    target_gene: str,
    time_fits: _TimeModelFits,
) -> dict[str, Any]:
    design_values: dict[str, np.ndarray | float] = {"const": 1.0, "time": time}
    if time_fits.selected_time_model == "quadratic":
        design_values["time_squared"] = np.square(time)
    design_values["predictor_gene"] = predictor
    final = sm.OLS(target, pd.DataFrame(design_values)).fit()

    return {
        "target_gene": target_gene,
        "predictor_gene": predictor_gene,
        "n_observations": int(final.nobs),
        "selected_time_model": time_fits.selected_time_model,
        "linear_time_r_squared": float(time_fits.linear.rsquared),
        "linear_time_sse": float(time_fits.linear.ssr),
        "quadratic_time_r_squared": float(time_fits.quadratic.rsquared),
        "quadratic_time_sse": float(time_fits.quadratic.ssr),
        "quadratic_term_coefficient": float(time_fits.quadratic.params["time_squared"]),
        "quadratic_term_p_value": float(time_fits.quadratic.pvalues["time_squared"]),
        "final_model_r_squared": float(final.rsquared),
        "final_model_sse": float(final.ssr),
        "predictor_coefficient": float(final.params["predictor_gene"]),
        "predictor_p_value": float(final.pvalues["predictor_gene"]),
        "predictor_t_statistic": float(final.tvalues["predictor_gene"]),
        "adjusted_r_squared": float(final.rsquared_adj),
        "residual_degrees_of_freedom": float(final.df_resid),
        "status": "ok",
        "error": "",
    }


def fit_ordered_relationship(
    predictor: Sequence[float] | pd.Series,
    target: Sequence[float] | pd.Series,
    time: Sequence[float] | pd.Series,
    *,
    predictor_gene: str,
    target_gene: str,
    quadratic_p_threshold: float = 0.05,
    expected_observations: int | None = None,
    expected_replicates_per_time: int | None = None,
) -> dict[str, Any]:
    """Fit the specified time models and one ordered predictor-target model."""
    if not 0 < float(quadratic_p_threshold) <= 1:
        raise ValueError("quadratic_p_threshold must be in (0, 1].")
    predictor_values, target_values, time_values = validate_aligned_observations(
        predictor,
        target,
        time,
        expected_observations=expected_observations,
        expected_replicates_per_time=expected_replicates_per_time,
    )
    time_fits = _fit_time_models(target_values, time_values, quadratic_p_threshold)
    return _result_from_fits(
        predictor=predictor_values,
        target=target_values,
        time=time_values,
        predictor_gene=predictor_gene,
        target_gene=target_gene,
        time_fits=time_fits,
    )


def _failed_result(target_gene: str, predictor_gene: str, error: Exception) -> dict[str, Any]:
    row = {column: np.nan for column in RESULT_COLUMNS}
    row.update(
        {
            "target_gene": target_gene,
            "predictor_gene": predictor_gene,
            "n_observations": 0,
            "selected_time_model": "",
            "status": "failed",
            "error": str(error),
        }
    )
    return row


def run_zeb2_regression_experiment(config: Zeb2RegressionConfig) -> dict[str, Any]:
    """Run both ordered directions between the anchor and every other gene."""
    config.validate()
    LOGGER.info(
        "Starting regression experiment: dataset=%s anchor=%s threshold=%s input=%s",
        config.dataset,
        config.anchor_gene,
        config.quadratic_p_threshold,
        config.expression_file,
    )
    sample_data = load_kutsche_sample_data(
        config.expression_file,
        normalize=config.normalize,
        transform=config.transform,
    )
    expression = sample_data.expression
    if config.anchor_gene not in expression.index:
        raise ValueError(f"Anchor gene {config.anchor_gene!r} is absent after Kutsche filtering.")

    genes = [str(gene) for gene in expression.index if str(gene) != config.anchor_gene]
    expected_rows = 2 * len(genes)
    target_fit_cache: dict[str, _TimeModelFits] = {}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def fit_one(predictor_gene: str, target_gene: str) -> dict[str, Any]:
        try:
            predictor = expression.loc[predictor_gene]
            target = expression.loc[target_gene]
            predictor_values, target_values, aligned_time = validate_aligned_observations(
                predictor,
                target,
                sample_data.time,
                expected_observations=EXPECTED_KUTSCHE_OBSERVATIONS,
                expected_replicates_per_time=EXPECTED_REPLICATES_PER_TIME,
            )
            if target_gene not in target_fit_cache:
                target_fit_cache[target_gene] = _fit_time_models(
                    target_values, aligned_time, config.quadratic_p_threshold
                )
            return _result_from_fits(
                predictor=predictor_values,
                target=target_values,
                time=aligned_time,
                predictor_gene=predictor_gene,
                target_gene=target_gene,
                time_fits=target_fit_cache[target_gene],
            )
        except Exception as exc:  # retain an auditable row for a failed ordered fit
            LOGGER.warning("Regression failed for %s -> %s: %s", predictor_gene, target_gene, exc)
            failures.append(
                {"predictor_gene": predictor_gene, "target_gene": target_gene, "error": str(exc)}
            )
            return _failed_result(target_gene, predictor_gene, exc)

    for gene in genes:
        rows.append(fit_one(gene, config.anchor_gene))
        rows.append(fit_one(config.anchor_gene, gene))

    output_file = Path(config.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    result_frame.to_csv(output_file, index=False)

    ok_rows = int((result_frame["status"] == "ok").sum())
    manifest = {
        "experiment": "zeb2_sample_level_multivariate_regression",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config.dataset,
        "anchor_gene": config.anchor_gene,
        "expression_file": str(config.expression_file),
        "output_file": str(output_file),
        "quadratic_p_threshold": config.quadratic_p_threshold,
        "quadratic_p_threshold_note": "Methodological parameter; not a finalized scientific choice.",
        "preprocessing": {"normalize": config.normalize, "transform": config.transform},
        "time_mapping": {f"d{day}": value for day, value in KUTSCHE_DAY_TO_TIME.items()},
        "replicate_aggregation": "none",
        "gene_count_including_anchor": int(len(expression)),
        "other_gene_count": len(genes),
        "observations_per_gene": int(expression.shape[1]),
        "expected_result_rows": expected_rows,
        "result_rows": int(len(result_frame)),
        "successful_rows": ok_rows,
        "failed_rows": len(failures),
        "failures": failures,
    }
    manifest_file = output_file.with_suffix(".manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    manifest["manifest_file"] = str(manifest_file)

    LOGGER.info(
        "Completed regression experiment: genes=%d observations=%d rows=%d failures=%d output=%s",
        len(expression),
        expression.shape[1],
        len(result_frame),
        len(failures),
        output_file,
    )
    return manifest


__all__ = [
    "EXPECTED_KUTSCHE_OBSERVATIONS",
    "EXPECTED_REPLICATES_PER_TIME",
    "KUTSCHE_DAY_TO_TIME",
    "KutscheSampleData",
    "RESULT_COLUMNS",
    "Zeb2RegressionConfig",
    "fit_ordered_relationship",
    "load_experiment_config",
    "load_kutsche_sample_data",
    "run_zeb2_regression_experiment",
    "validate_aligned_observations",
]
