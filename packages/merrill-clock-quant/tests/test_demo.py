import numpy as np

from examples_quant_demo import build_synthetic_dataset, run_demo


def test_offline_pipeline_runs_without_network_and_is_reproducible():
    first = run_demo(seed=7, periods=180)
    second = run_demo(seed=7, periods=180)
    assert first == second
    assert first["synthetic"] is True
    assert first["evaluated_months"] > 100
    assert first["warmup_months_excluded"] > 0
    assert all(np.isfinite(value) for value in first["strategy"].values())
    assert all(np.isfinite(value) for value in first["equal_weight_benchmark_no_costs"].values())


def test_synthetic_dataset_shapes_and_feature_columns():
    ip, cpi, returns, features = build_synthetic_dataset(seed=1, periods=120)
    assert len(ip) == len(cpi) == 144
    assert len(returns) == 120
    assert set(features.columns) == {"growth_accel", "inflation_accel"}
    assert list(returns.columns) == ["EQ", "BOND", "GOLD", "COMM", "CASH"]
