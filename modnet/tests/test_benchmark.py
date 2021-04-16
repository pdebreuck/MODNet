#!/usr/bin/env python


def test_train_small_model_benchmark(subset_moddata, tf_session):
    """Tests the `matbench_benchmark()` method with optional arguments."""
    from modnet.matbench.benchmark import matbench_benchmark

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    results = matbench_benchmark(
        data,
        [[["eform"]]],
        {"eform": 1},
        inner_feat_selection=False,
        fast=True,
        nested=2,
        n_jobs=1,
    )

    expected_keys = (
        "nested_losses",
        "nested_learning_curves",
        "best_learning_curves",
        "predictions",
        "targets",
        "errors",
        "scores",
        "best_presets",
    )

    assert all(key in results for key in expected_keys)
    assert all(len(results[key]) == 5 for key in expected_keys)


def test_train_small_model_benchmark_with_extra_args(subset_moddata):
    """Tests the `matbench_benchmark()` method with some extra settings,
    parallelised over 2 jobs.

    """
    from modnet.matbench.benchmark import matbench_benchmark

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    # Check that other settings don't break the model creation,
    # but that they do get used in fitting
    other_fit_settings = {
        "epochs": 10,
        "increase_bs": False,
        "callbacks": [],
    }

    results = matbench_benchmark(
        data,
        [[["eform"]]],
        {"eform": 1},
        fit_settings=other_fit_settings,
        inner_feat_selection=False,
        fast=True,
        nested=2,
        n_jobs=2,
    )

    expected_keys = (
        "nested_losses",
        "nested_learning_curves",
        "best_learning_curves",
        "predictions",
        "targets",
        "errors",
        "scores",
        "best_presets",
    )

    assert all(key in results for key in expected_keys)
    assert all(len(results[key]) == 5 for key in expected_keys)
