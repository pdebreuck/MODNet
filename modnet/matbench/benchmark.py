import os
from collections import defaultdict
from traceback import print_exc
from typing import List, Dict, Any, Optional, Tuple, Type

import numpy as np

from modnet.preprocessing import MODData
from modnet.models import MODNetModel
from modnet.utils import LOG
from modnet.hyper_opt import FitGenetic

MATBENCH_SEED = 18012019


def matbench_kfold_splits(data: MODData, n_splits=5, classification=False):
    """Return the pre-defined k-fold splits to use when reporting matbench results.

    Arguments:
        data: The featurized MODData.
    """

    if classification:
        from sklearn.model_selection import StratifiedKFold as KFold
    else:
        from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=MATBENCH_SEED)
    kf_splits = kf.split(data.df_featurized, y=data.df_targets)
    return kf_splits


def matbench_benchmark(
    data: MODData,
    target: List[str],
    target_weights: Dict[str, float],
    fit_settings: Optional[Dict[str, Any]] = None,
    ga_settings: Optional[Dict[str, float]] = None,
    classification: bool = False,
    model_type: Type[MODNetModel] = MODNetModel,
    save_folds: bool = False,
    save_models: bool = False,
    hp_optimization: bool = True,
    inner_feat_selection: bool = True,
    use_precomputed_cross_nmi: bool = True,
    use_fit_preset: bool = False,
    use_ga: bool = False,
    presets: Optional[List[dict]] = None,
    fast: bool = False,
    n_jobs: Optional[int] = None,
    nested: bool = False,
    **model_init_kwargs,
) -> dict:
    """Train and cross-validate a model against Matbench data splits, optionally
    performing hyperparameter optimisation.

    Arguments:
        data: The entire dataset as a `MODData`.
        target: The list of target names to train on.
        target_weights: The target weights to use for the `MODNetModel`.
        fit_settings: Any settings to pass to `model.fit(...)` directly
            (typically when not performing hyperparameter optimisation).
        classification: Whether all tasks are classification rather than regression.
        model_type: The type of the model to create and benchmark.
        save_folds: Whether to save dataframes with pre-processed fold
            data (e.g. feature selection).
        save_models: Whether to pickle all trained models according to
            their fold index and performance.
        hp_optimization: Whether to perform hyperparameter optimisation.
        inner_feat_selection: Whether to perform split-level feature
            selection or try to use pre-computed values.
        use_precomputed_cross_nmi: Whether to use the precmputed cross NMI
            from the Materials Project dataset, or recompute per fold.
        presets: Override the built-in hyperparameter grid with these presets.
        fast: Whether to perform debug training, i.e. reduced presets and epochs.
        n_jobs: Try to parallelize the inner fit_preset over this number of
            processes. Maxes out at number_of_presets*nested_folds
        nested: Whether to perform nested CV for hyperparameter optimisation.
        **model_init_kwargs: Additional arguments to pass to the model on creation.

    Returns:
        A dictionary containing all the results from the training, broken
            down by model and by fold.

    """

    if use_fit_preset and use_ga:
        raise RuntimeError(
            "Both use_fit_preset and use_ga are set to True. Please choose one."
        )

    if fit_settings is None:
        fit_settings = {}

    if not fit_settings.get("n_feat"):
        nf = len(data.df_featurized.columns)
        fit_settings["n_feat"] = nf
    if not fit_settings.get("num_neurons"):
        # Pass dummy network
        fit_settings["num_neurons"] = [[4], [4], [4], [4]]

    if ga_settings is None:
        ga_settings = {
            "size_pop": 20,
            "num_generations": 10,
            "early_stopping": 4,
            "refit": False,
        }

    fold_data = []
    results = defaultdict(list)

    for ind, (train, test) in enumerate(
        matbench_kfold_splits(data, classification=classification)
    ):
        train_data, test_data = data.split((train, test))
        if inner_feat_selection:
            path = "folds/train_moddata_f{}".format(ind + 1)
            if os.path.isfile(path):
                train_data = MODData.load(path)
            else:
                train_data.feature_selection(
                    n=-1, use_precomputed_cross_nmi=use_precomputed_cross_nmi
                )
            os.makedirs("folds", exist_ok=True)
            train_data.save(path)

        fold_data.append((train_data, test_data))

    args = (target, target_weights, fit_settings, ga_settings)

    model_kwargs = {
        "model_type": model_type,
        "hp_optimization": hp_optimization,
        "fast": fast,
        "classification": classification,
        "save_folds": save_folds,
        "presets": presets,
        "use_fit_preset": use_fit_preset,
        "use_ga": use_ga,
        "save_models": save_models,
        "nested": nested,
        "n_jobs": n_jobs,
    }

    model_kwargs.update(model_init_kwargs)

    fold_results = []
    for fold in enumerate(fold_data):
        fold_results.append(train_fold(fold, *args, **model_kwargs))

    for fold in fold_results:
        for key in fold:
            results[key].append(fold[key])

    return results


def train_fold(
    fold: Tuple[int, Tuple[MODData, MODData]],
    target: List[str],
    target_weights: Dict[str, float],
    fit_settings: Dict[str, Any],
    ga_settings: Dict[str, float],
    model_type: Type[MODNetModel] = MODNetModel,
    presets=None,
    hp_optimization=True,
    use_fit_preset=False,
    use_ga=False,
    classification=False,
    save_folds=False,
    fast=False,
    save_models=False,
    nested=False,
    n_jobs=None,
    **model_kwargs,
) -> dict:
    """Train one fold of a CV.
    Unless stated, all arguments have the same meaning as in `matbench_benchmark(...)`.

    Arguments:
        fold: A tuple containing the fold index, and another tuple of the
            training MODData and test MODData.

    Returns:
        A dictionary summarising the fold results.

    """

    fold_ind, (train_data, test_data) = fold

    results = {}
    multi_target = bool(len(target) - 1)

    # If not performing hp_optimization, load model init settings from fit_settings
    model_settings = {}
    if not hp_optimization:
        model_settings = {
            "num_neurons": fit_settings["num_neurons"],
            "num_classes": fit_settings.get("num_classes"),
            "act": fit_settings.get("act"),
            "out_act": fit_settings.get("out_act", "linear"),
            "n_feat": fit_settings["n_feat"],
        }

    model_settings.update(model_kwargs)

    if classification:
        model_settings["num_classes"] = {t: 2 for t in target_weights}

    model = model_type(target, target_weights, **model_settings)

    if hp_optimization:
        if use_fit_preset:
            (
                models,
                val_losses,
                best_learning_curve,
                learning_curves,
                best_presets,
            ) = model.fit_preset(
                train_data,
                presets=presets,
                fast=fast,
                classification=classification,
                nested=nested,
                n_jobs=n_jobs,
            )
            results["nested_losses"] = val_losses
            results["nested_learning_curves"] = learning_curves
            results["best_learning_curves"] = best_learning_curve
            results["best_presets"] = best_presets
        elif use_ga:
            ga = FitGenetic(train_data)
            model = ga.run(
                size_pop=ga_settings["size_pop"],
                num_generations=ga_settings["num_generations"],
                n_jobs=n_jobs,
                early_stopping=ga_settings["early_stopping"],
                refit=ga_settings["refit"],
            )

        if save_models:
            for ind, nested_model in enumerate(models):
                score = val_losses[ind]
                nested_model.save(f"results/nested_model_{fold_ind}_{ind}_{score:3.3f}")

            model.save(f"results/best_model_{fold_ind}_{score:3.3f}")

    else:
        if fit_settings["increase_bs"]:
            model.fit(
                train_data,
                lr=fit_settings["lr"],
                epochs=fit_settings["epochs"],
                batch_size=fit_settings["batch_size"],
                loss=fit_settings["loss"],
            )
            model.fit(
                train_data,
                lr=fit_settings["lr"] / 7,
                epochs=fit_settings["epochs"] // 2,
                batch_size=fit_settings["batch_size"] * 2,
                loss=fit_settings["loss"],
            )
        else:
            model.fit(train_data, **fit_settings)

    try:
        predict_kwargs = {}
        if classification:
            predict_kwargs["return_prob"] = True
        if model.can_return_uncertainty:
            predict_kwargs["return_unc"] = True

        pred_results = model.predict(test_data, **predict_kwargs)
        if isinstance(pred_results, tuple):
            predictions, stds = pred_results
        else:
            predictions = pred_results
            stds = None

        targets = test_data.df_targets

        if classification:
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import OneHotEncoder

            y_true = OneHotEncoder().fit_transform(targets.values).toarray()
            score = roc_auc_score(y_true, predictions.values)
            pred_bool = model.predict(test_data, return_prob=False)
            LOG.info(f"ROC-AUC: {score}")
            errors = targets - pred_bool
        elif multi_target:
            errors = targets - predictions
            score = np.mean(np.abs(errors.values), axis=0)
        else:
            errors = targets - predictions
            score = np.mean(np.abs(errors.values))
    except Exception:
        print_exc()
        print("Something went wrong benchmarking this model.")
        predictions = None
        errors = None
        score = None

    if save_folds:
        opt_feat = train_data.optimal_features[: fit_settings["n_feat"]]
        df_train = train_data.df_featurized
        df_train = df_train[opt_feat]
        df_train.to_csv("folds/train_f{}.csv".format(ind + 1))
        df_test = test_data.df_featurized
        df_test = df_test[opt_feat]
        errors.columns = [x + "_error" for x in errors.columns]
        df_test = df_test.join(errors)
        df_test.to_csv("folds/test_f{}.csv".format(ind + 1))

    results["predictions"] = predictions
    if stds is not None:
        results["stds"] = stds
    results["targets"] = targets
    results["errors"] = errors
    results["scores"] = score
    results["model"] = model

    return results
