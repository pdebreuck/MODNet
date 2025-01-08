"""This submodule implements the `EnsembleMODNetModel`, an
extension of the vanilla model that bootstraps uncertainties
from multiple MODNet models, trained in parallel.

"""

from typing import List, Tuple, Dict, Optional, Any
import multiprocessing

import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf

from sklearn.utils import resample

from modnet.models.vanilla import (
    MODNetModel,
)
from modnet import __version__
from modnet.utils import LOG, generate_shuffled_and_stratified_val_split
from modnet.preprocessing import MODData

__all__ = ("EnsembleMODNetModel",)


class EnsembleMODNetModel(MODNetModel):
    """Container class for n_model (Bootstrap) MODNetModels, that handles
    setting up the architecture, activations, training and learning curve.

    Attributes:
        n_feat: The number of features used in the model.
        weights: The relative loss weights for each target.
        optimal_descriptors: The list of column names used
            in training the model.
        model: The `keras.model.Model` of the network itself.
        target_names: The list of targets names that the model
            was trained for.

    """

    can_return_uncertainty = True

    def __init__(
        self,
        *args,
        n_models=100,
        bootstrap=True,
        models=None,
        modnet_models=None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            *args: See MODNetModel
            n_models: number of inner MODNetModels, each model has the same architecture defined by the args nd kwargs.
            bootstrap: whether to bootstrap the samples for each inner MODNet fit.
            models: List of user provided MODNetModels. Enables to have different architectures. n_models is discarded in this case.
            random_state: fix a random state for use with this model.
            modnet_model: Deprecated. Same argument as models. For backward compatibility only.
            **kwargs: See MODNetModel
        """
        self.__modnet_version__ = __version__
        self.bootstrap = bootstrap
        self.random_state = random_state
        if modnet_models is not None and models is None:
            models = modnet_models
        if models is None:
            self.models = []
            self.n_models = n_models
            for i in range(self.n_models):
                self.models.append(MODNetModel(*args, **kwargs))
        else:
            self.models = models
            self.n_models = len(models)

        self.targets = self.models[0].targets
        self.weights = self.models[0].weights
        self.num_classes = self.models[0].num_classes
        self.targets_groups = self.models[0].targets_groups
        self.out_act = self.models[0].out_act

    def fit(
        self,
        training_data: MODData,
        n_jobs=1,
        **kwargs,
    ) -> None:
        """Train the model on the passed training `MODData` object.

        Parameters match those of `MODNetModel.fit`.
        """

        if self.bootstrap:
            LOG.info("Generating bootstrap data...")
            if self.random_state is None:
                random_state = self.n_models * [None]
            else:
                random_state = np.arange(self.n_models) + self.random_state

            # Loop over all targets and check if any involve classification, if so, stratify
            stratify = None
            for prop in self.targets_groups:
                if self.num_classes[prop[0]] >= 2:  # Classification
                    stratify = training_data.df_targets[prop[0]].array
                    break
            train_indices = [
                resample(
                    np.arange(len(training_data.df_targets)),
                    replace=True,
                    n_samples=len(training_data.df_targets),
                    random_state=random_state[i],
                    stratify=stratify,
                )
                for i in range(self.n_models)
            ]

            train_datas = [
                training_data.split((train_indices[i], []))[0]
                for i in range(self.n_models)
            ]
        else:
            train_datas = [training_data for _ in range(self.n_models)]

        if n_jobs <= 1:
            for i in range(self.n_models):
                LOG.info(f"Bootstrap fitting model #{i + 1}/{self.n_models}")
                self.models[i].fit(train_datas[i], **kwargs)
                model_summary = ""
                for k in self.models[i].history.keys():
                    model_summary += "{}: {:.4f}\t".format(
                        k, self.models[i].history[k][-1]
                    )
                LOG.info(model_summary)
        else:
            ctx = multiprocessing.get_context("spawn")
            pool = ctx.Pool(processes=n_jobs)
            tasks = []
            for i, m in enumerate(self.models):
                m._make_picklable()
                tasks.append(
                    {
                        "model": m,
                        "training_data": train_datas[i],
                        "model_id": i,
                        **kwargs,
                    }
                )
            for res in tqdm.tqdm(
                pool.imap_unordered(_map_fit_MODNet, tasks, chunksize=1),
                total=self.n_models,
            ):
                model, model_id = res
                model._restore_model()
                self.models[model_id] = model
                model_summary = f"Model #{model_id}\t"
                for k in model.history.keys():
                    model_summary += "{}: {:.4f}\t".format(k, model.history[k][-1])
                LOG.info(model_summary)
            pool.close()
            pool.join()

    def predict(
        self,
        test_data: MODData,
        return_unc: bool = False,
        return_prob: bool = False,
        remap_out_of_bounds: bool = True,
        voting_type: str = "soft",
    ) -> pd.DataFrame:
        """Predict the target values for the passed MODData.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.
            return_prob: For a classification task only: whether to return the probability of each
                class OR only return the most probable class.
            return_unc: whether to return a second dataframe containing the uncertainties
            remap_out_of_bounds: whether to remap out-of-bounds values to the nearest bound.
            voting_type: If classification task and return_prob is False, determines
                if soft or hard ensemble voting is performed.

        Returns:
            A `pandas.DataFrame` containing the predicted values of the targets.


        """
        return_prob_comput = return_prob
        if (
            not return_prob
            and max(self.num_classes.values()) >= 2
            and voting_type == "soft"
        ):
            return_prob_comput = True

        all_predictions = []
        for i in range(self.n_models):
            p = self.models[i].predict(
                test_data,
                return_prob=return_prob_comput,
                remap_out_of_bounds=remap_out_of_bounds,
            )
            all_predictions.append(p.values)

        p_columns = p.columns
        if max(self.num_classes.values()) == 0 or return_prob:
            p_mean = np.array(all_predictions).mean(axis=0)
        elif voting_type == "soft":
            p_columns, p_mean = [], []
            for prop in set(["_".join(s.split("_")[:-2]) for s in p.columns]):
                prop_ids = [
                    idx for idx, col in enumerate(p.columns) if col.startswith(prop)
                ]
                a = np.array(all_predictions)[:, :, prop_ids]
                p_mean.append(np.argmax(a.sum(axis=0), axis=1))
                p_columns.append(prop)
            p_mean = np.array(p_mean).transpose()
        else:
            p_mean = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)),
                axis=0,
                arr=np.array(all_predictions),
            )

        p_std = np.array(all_predictions).std(axis=0)
        df_mean = pd.DataFrame(p_mean, index=p.index, columns=p_columns)
        df_std = pd.DataFrame(p_std, index=p.index, columns=p.columns)

        if return_unc:
            return df_mean, df_std
        else:
            return df_mean

    def evaluate(self, test_data: MODData) -> pd.DataFrame:
        """Evaluates the target values for the passed MODData by returning the corresponding loss.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.


        Returns:
            Loss score
        """
        all_losses = np.zeros(self.n_models)
        for i, m in enumerate(self.models):
            all_losses[i] = m.evaluate(test_data)

        return all_losses.mean()

    def fit_preset(
        self,
        data: MODData,
        presets: List[Dict[str, Any]] = None,
        val_fraction: float = 0.15,
        verbose: int = 0,
        classification: bool = False,
        refit: bool = False,
        fast: bool = False,
        nested: int = 5,
        callbacks: List[Any] = None,
        n_jobs: int = 1,
    ) -> Tuple[
        List[List[Any]],
        np.ndarray,
        Optional[List[float]],
        List[List[float]],
        Dict[str, Any],
    ]:
        """Chooses an optimal hyper-parametered MODNet model from different presets.

        This function implements the "inner loop" of a cross-validation workflow. By
        modifying the `nested` argument, it can be run in full nested mode (i.e.
        train n_fold * n_preset models) or just with a simple random hold-out set.

        The data is first fitted on several well working MODNet presets
        with a validation set (10% of the furnished data by default).

        Sets the `self.models` attribute to the model with the lowest mean validation loss across
        all folds.

        Note: Inner models (presets) are 5-model bootstraps. The final (refit) model will be a self.n_model bootstrap.

        Args:
            data: MODData object contain training and validation samples.
            presets: A list of dictionaries containing custom presets.
            verbose: The verbosity level to pass to tf.keras
            val_fraction: The fraction of the data to use for validation.
            classification: Whether or not we are performing classification.
            refit: Whether or not to refit the final model for each fold with
                the best-performing settings.
            fast: Used for debugging. If `True`, only fit the first 2 presets, use 1-model ensembles and
                reduce the number of epochs.
            nested: integer specifying whether or not to perform a full nested CV. If 0,
                a simple validation split is performed based on val_fraction argument.
                If an integer, use this number of inner CV folds, ignoring the `val_fraction` argument.
                Note: If set to 1, the value will be overwritten to a default of 5 folds.
            n_jobs: number of concurrent processes to use when multiprocessing

        Returns:
            - A list of length num_outer_folds containing lists of MODNet models of length num_inner_folds.
            - A list of validation losses achieved by the best model for each fold during validation (excluding refit).
            - The learning curve of the final (refitted) model (or `None` if `refit` is `False`)
            - A nested list of learning curves for each trained model of lengths (num_outer_folds,  num_inner folds).
            - The settings of the best-performing preset.

        """

        from modnet.matbench.benchmark import matbench_kfold_splits
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
            "2"  # many models will be fitted => reduce output
        )

        if callbacks is None:
            es = tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                min_delta=0.001,
                patience=100,
                verbose=verbose,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            )
            callbacks = [es]

        if presets is None:
            from modnet.model_presets import gen_presets

            presets = gen_presets(
                len(data.optimal_features),
                len(data.df_targets),
                classification=classification,
            )

        if fast and len(presets) >= 2:
            presets = presets[:2]
            for k, _ in enumerate(presets):
                presets[k]["epochs"] = 5

        num_nested_folds = 5
        if nested:
            num_nested_folds = nested
        if num_nested_folds <= 1:
            num_nested_folds = 5

        # create tasks
        splits = matbench_kfold_splits(
            data, n_splits=num_nested_folds, classification=classification
        )
        if not nested:
            splits = [
                generate_shuffled_and_stratified_val_split(
                    y=data.df_targets.values,
                    val_fraction=val_fraction,
                    classification=classification,
                )
            ]
            n_splits = 1
        else:
            n_splits = num_nested_folds
        train_val_datas = []
        for train, val in splits:
            train_val_datas.append(data.split((train, val)))

        tasks = []
        for i, params in enumerate(presets):
            n_feat = min(len(data.get_optimal_descriptors()), params["n_feat"])

            for ind in range(n_splits):
                val_params = {}
                train_data, val_data = train_val_datas[ind]
                val_params["val_data"] = val_data

                tasks += [
                    {
                        "train_data": train_data,
                        "targets": self.targets,
                        "weights": self.weights,
                        "n_models": 1 if fast else 5,
                        "num_classes": self.num_classes,
                        "n_feat": n_feat,
                        "num_neurons": params["num_neurons"],
                        "lr": params["lr"],
                        "batch_size": params["batch_size"],
                        "epochs": params["epochs"],
                        "loss": params["loss"],
                        "act": params["act"],
                        "out_act": self.out_act,
                        "callbacks": callbacks,
                        "preset_id": i,
                        "fold_id": ind,
                        "verbose": verbose,
                        **val_params,
                    }
                ]

        val_losses = np.zeros((len(presets), n_splits))
        learning_curves = [[None for _ in range(n_splits)] for _ in range(len(presets))]
        models = [[None for _ in range(n_splits)] for _ in range(len(presets))]

        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=n_jobs)
        LOG.info(
            f"Multiprocessing on {n_jobs} cores. Total of {multiprocessing.cpu_count()} cores available."
        )

        for res in tqdm.tqdm(
            pool.imap_unordered(_map_validate_ensemble_model, tasks, chunksize=1),
            total=len(tasks),
        ):
            val_loss, learning_curve, model, preset_id, fold_id = res
            LOG.info(f"Preset #{preset_id} fitting finished, loss: {val_loss}")

            # reload model
            model._restore_model()

            val_losses[preset_id, fold_id] = val_loss
            learning_curves[preset_id][fold_id] = learning_curve
            models[preset_id][fold_id] = model
        pool.close()
        pool.join()

        val_loss_per_preset = np.mean(val_losses, axis=1)
        best_preset_idx = int(np.argmin(val_loss_per_preset))
        best_model_idx = int(np.argmin(val_losses[best_preset_idx, :]))
        best_preset = presets[best_preset_idx]
        best_learning_curve = learning_curves[best_preset_idx][best_model_idx]

        LOG.info(
            "Preset #{} resulted in lowest validation loss with params {}".format(
                best_preset_idx + 1, tasks[n_splits * best_preset_idx + best_model_idx]
            )
        )

        if refit:
            LOG.info(
                "Refitting with all data and parameters: {} models, {}".format(
                    100, best_preset
                )
            )
            # Building final model

            n_feat = min(len(data.get_optimal_descriptors()), best_preset["n_feat"])
            self.__init__(
                self.targets,
                self.weights,
                n_models=100,
                num_neurons=best_preset["num_neurons"],
                n_feat=n_feat,
                act=best_preset["act"],
                out_act=self.out_act,
                num_classes=self.num_classes,
            )
            self.fit(
                data,
                val_fraction=0,
                lr=best_preset["lr"],
                epochs=best_preset["epochs"],
                batch_size=best_preset["batch_size"],
                loss=best_preset["loss"],
                callbacks=callbacks,
                verbose=verbose,
                n_jobs=n_jobs,
            )
        else:
            # take 5 best 5-models on all inner folds = 125-bootstrap model for a 5  nested CV fold
            final_models = []
            for i in range(n_splits):
                best_5_idx = np.argsort(val_losses[:, i])[:5]
                for idx in best_5_idx:
                    final_models.extend(models[idx][i].models)
            self.__init__(modnet_models=final_models)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # reset

        return models, val_losses, best_learning_curve, learning_curves, best_preset

    def _make_picklable(self):
        """
        transforms inner keras model to jsons so that the MODNet object becomes picklable.
        """

        for m in self.models:
            m._make_picklable()

    def _restore_model(self):
        """
        restore inner keras model after running make_picklable
        """

        # backward compatibility for loading models saved <v0.3
        if not hasattr(self, "models") and hasattr(self, "model"):
            self.models = self.model

        for m in self.models:
            m._restore_model()

    def _get_param_names(self):
        possible_params = [
            "n_models",
            "bootstrap",
            "models",
        ]
        return possible_params


def _validate_ensemble_model(
    train_data=None,
    val_data=None,
    targets=None,
    weights=None,
    n_models=5,
    num_classes=None,
    n_feat=100,
    num_neurons=[[8], [8], [8], [8]],
    lr=0.1,
    batch_size=64,
    epochs=100,
    loss="mse",
    act="relu",
    out_act="linear",
    xscale="minmax",
    impute_missing=-1,
    callbacks=[],
    preset_id=None,
    fold_id=None,
    verbose=0,
):

    model = EnsembleMODNetModel(
        targets,
        weights,
        n_models=n_models,
        num_neurons=num_neurons,
        n_feat=n_feat,
        act=act,
        out_act=out_act,
        num_classes=num_classes,
    )

    model.fit(
        train_data,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        loss=loss,
        xscale=xscale,
        impute_missing=impute_missing,
        callbacks=callbacks,
        verbose=verbose,
        val_fraction=0,
        val_data=val_data,
    )

    learning_curves = [m.history["val_loss"] for m in model.models]

    val_loss = model.evaluate(val_data)

    model._make_picklable()

    return val_loss, learning_curves, model, preset_id, fold_id


def _map_validate_ensemble_model(kwargs):
    return _validate_ensemble_model(**kwargs)


def _fit_MODNet(model: MODNetModel, training_data: MODData, model_id=None, **kwargs):
    model._restore_model()
    model.fit(training_data, **kwargs)
    model._make_picklable()

    return model, model_id


def _map_fit_MODNet(kwargs):
    return _fit_MODNet(**kwargs)
