import numpy as np
import pandas as pd
import pytest

from gene_analysis.experiments.zeb2_regression import (
    EXPECTED_KUTSCHE_OBSERVATIONS,
    KUTSCHE_DAY_TO_TIME,
    Zeb2RegressionConfig,
    fit_ordered_relationship,
    load_kutsche_sample_data,
    run_zeb2_regression_experiment,
    validate_aligned_observations,
)


pytestmark = pytest.mark.unit


def _time() -> np.ndarray:
    return np.repeat(np.arange(5, dtype=float), 7)


def test_linear_synthetic_target_selects_linear_time_model():
    time = _time()
    within_time = np.tile(np.linspace(-1.0, 1.0, 7), 5)
    target = 2.0 + 3.0 * time + 0.03 * within_time
    predictor = np.sin(np.arange(35))

    result = fit_ordered_relationship(
        predictor,
        target,
        time,
        predictor_gene="X",
        target_gene="ZEB2",
        expected_observations=35,
        expected_replicates_per_time=7,
    )

    assert result["selected_time_model"] == "linear"
    assert result["n_observations"] == 35


def test_quadratic_synthetic_target_selects_quadratic_time_model():
    time = _time()
    within_time = np.tile(np.linspace(-1.0, 1.0, 7), 5)
    target = 1.0 + 0.5 * time + 4.0 * np.square(time) + 0.03 * within_time
    predictor = np.cos(np.arange(35))

    result = fit_ordered_relationship(
        predictor,
        target,
        time,
        predictor_gene="X",
        target_gene="ZEB2",
    )

    assert result["selected_time_model"] == "quadratic"
    assert result["quadratic_term_coefficient"] == pytest.approx(4.0, abs=0.02)
    assert result["quadratic_term_p_value"] < 0.05


def test_predictor_effect_is_detected_after_time_adjustment():
    rng = np.random.default_rng(401)
    time = _time()
    predictor = rng.normal(size=35)
    target = 5.0 + 1.5 * time + 2.75 * predictor + rng.normal(scale=0.05, size=35)

    result = fit_ordered_relationship(
        predictor,
        target,
        time,
        predictor_gene="X",
        target_gene="ZEB2",
    )

    assert result["predictor_coefficient"] == pytest.approx(2.75, abs=0.05)
    assert result["predictor_p_value"] < 1e-20


def test_no_predictor_effect_reports_lack_of_evidence():
    rng = np.random.default_rng(90210)
    time = _time()
    predictor = rng.normal(size=35)
    target = 3.0 + 2.0 * time + rng.normal(scale=0.5, size=35)

    result = fit_ordered_relationship(
        predictor,
        target,
        time,
        predictor_gene="X",
        target_gene="ZEB2",
    )

    assert result["predictor_p_value"] > 0.05


def test_alignment_and_length_validation_rejects_bad_observations():
    index = pd.Index(["sample-1", "sample-2", "sample-3"])
    predictor = pd.Series([1.0, 2.0, 3.0], index=index)
    target = pd.Series([1.0, 2.0], index=index[:2])
    time = pd.Series([0.0, 1.0, 2.0], index=index)

    with pytest.raises(ValueError, match="same observation count"):
        validate_aligned_observations(predictor, target, time)

    misaligned_target = pd.Series([1.0, 2.0, 3.0], index=index[::-1])
    with pytest.raises(ValueError, match="not identically aligned"):
        validate_aligned_observations(predictor, misaligned_target, time)

    with pytest.raises(ValueError, match="Expected 35 unaggregated observations"):
        validate_aligned_observations(predictor, predictor, time, expected_observations=35)

    nonfinite = pd.Series([1.0, np.nan, 3.0], index=index)
    with pytest.raises(ValueError, match="no observations were dropped"):
        validate_aligned_observations(nonfinite, predictor, time)


def _write_mini_kutsche(path):
    columns = []
    for day in range(5):
        for replicate in range(1, 8):
            columns.append(f"S{day}{replicate}_WT_d{day}_{replicate}_R1.count")
    frame = pd.DataFrame(
        [
            np.arange(1.0, 36.0),
            2.0 * np.arange(1.0, 36.0) + np.tile(np.arange(7), 5),
            np.zeros(35),
        ],
        index=["ZEB2", "GENEX", "ALL_ZERO"],
        columns=columns,
    )
    frame.index.name = "Gene"
    frame.to_csv(path, sep="\t")


def test_kutsche_loader_retains_35_aligned_samples_without_aggregation(tmp_path):
    expression_file = tmp_path / "mini_kutsche.txt"
    _write_mini_kutsche(expression_file)

    data = load_kutsche_sample_data(expression_file)

    assert data.expression.shape == (2, EXPECTED_KUTSCHE_OBSERVATIONS)
    assert data.expression.index.tolist() == ["ZEB2", "GENEX"]
    assert data.expression.columns.equals(data.time.index)
    assert data.time.tolist() == [KUTSCHE_DAY_TO_TIME[day] for day in range(5) for _ in range(7)]


def test_experiment_writes_both_ordered_directions_and_manifest(tmp_path):
    expression_file = tmp_path / "mini_kutsche.txt"
    output_file = tmp_path / "result.csv"
    _write_mini_kutsche(expression_file)
    config = Zeb2RegressionConfig(expression_file=expression_file, output_file=output_file)

    manifest = run_zeb2_regression_experiment(config)
    result = pd.read_csv(output_file)

    assert list(zip(result["predictor_gene"], result["target_gene"])) == [
        ("GENEX", "ZEB2"),
        ("ZEB2", "GENEX"),
    ]
    assert (result["n_observations"] == 35).all()
    assert manifest["result_rows"] == 2
    assert manifest["failed_rows"] == 0
    assert output_file.with_suffix(".manifest.json").exists()
