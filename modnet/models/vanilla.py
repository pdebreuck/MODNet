"""This submodule defines the "vanilla" `MODNetModel`, i.e. a single
model with deterministic weights and outputs.

"""

from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Callable, Any, Union

from pathlib import Path
import multiprocessing

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import tensorflow as tf

from modnet.preprocessing import MODData
from modnet.utils import LOG, generate_shuffled_and_stratified_val_split
from modnet import __version__

import tqdm

__all__ = "MODNetModel"


class MODNetModel:
    """Container class for the underlying tf.keras `Model`, that handles
    setting up the architecture, activations, training and learning curve.

    Attributes:
        n_feat: The number of features used in the model.
        weights: The relative loss weights for each target.
        optimal_descriptors: The list of column names used
            in training the model.
        model: The `tf.keras.model.Model` of the network itself.
        target_names: The list of targets names that the model
            was trained for.

    """

    can_return_uncertainty = False

    def __init__(
        self,
        targets: List,
        weights: Dict[str, float],
        num_neurons=([64], [32], [16], [16]),
        num_classes: Optional[Dict[str, int]] = None,
        multi_label: Optional[bool] = False,
        n_feat: Optional[int] = 64,
        act: str = "relu",
        out_act: str = "linear",
    ):
        """Initialise the model on the passed targets with the desired
        architecture, feature count and loss functions and activation functions.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            weights: The relative loss weights to apply for each target.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                 with n=0 for regression and n>=2 for classification with n the number of classes.
            multi_label: Whether the problem (if classification) is multi-label.
                In this case the softmax output-activation is replaced by a sigmoid.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `tf.keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            n_feat: The number of features to use as model inputs.
            act: A string defining a tf.keras activation function to pass to use
                in the `tf.keras.layers.Dense` layers.
            out_act: A string defining a tf.keras activation function to pass to use
                for the last output layer (regression only)

        """

        self.__modnet_version__ = __version__

        if n_feat is None:
            n_feat = 64
        self.n_feat = n_feat
        self.weights = weights
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.num_neurons = num_neurons
        self.act = act
        self.out_act = out_act

        self.xscale = None
        self._scaler = None
        self._imputer = None
        self.impute_missing = None
        self._scale_impute = None
        self.optimal_descriptors = None
        self.target_names = None
        self.targets = targets
        self.model = None

        self.targets_groups = [x for subl in targets for x in subl]
        self.targets_flatten = [x for subl in self.targets_groups for x in subl]
        self.num_classes = {name: 0 for name in self.targets_flatten}
        if num_classes is not None:
            self.num_classes.update(num_classes)
        self._multi_target = len(self.targets_flatten) > 1

        self.model = self.build_model(
            targets,
            n_feat,
            num_neurons,
            act=act,
            out_act=out_act,
            num_classes=self.num_classes,
            multi_label=multi_label,
        )

    def build_model(
        self,
        targets: List,
        n_feat: int,
        num_neurons: Tuple[List[int], List[int], List[int], List[int]],
        num_classes: Optional[Dict[str, int]] = None,
        multi_label: Optional[bool] = False,
        act: str = "relu",
        out_act: str = "linear",
    ):
        """Builds the tf.keras model and sets the `self.model` attribute.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            n_feat: The number of features to use as model inputs.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `tf.keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                with n=0 for regression and n>=2 for classification with n the number of classes.
            multi_label: Whether the problem (if classification) is multi-label.
                In this case the softmax output-activation is replaced by a sigmoid.
            act: A string defining a tf.keras activation function to pass to use
                in the `tf.keras.layers.Dense` layers.
            out_act: A string defining a tf.keras activation function to pass to use
                for the last output layer (regression only)

        """

        num_layers = [len(x) for x in num_neurons]

        # Build first common block
        f_input = tf.keras.layers.Input(shape=(n_feat,))
        previous_layer = f_input
        for i in range(num_layers[0]):
            previous_layer = tf.keras.layers.Dense(num_neurons[0][i], activation=act)(
                previous_layer
            )
            if self._multi_target:
                previous_layer = tf.keras.layers.BatchNormalization()(previous_layer)
        common_out = previous_layer

        # Build intermediate representations
        intermediate_models_out = []
        for _ in range(len(targets)):
            previous_layer = common_out
            for j in range(num_layers[1]):
                previous_layer = tf.keras.layers.Dense(
                    num_neurons[1][j], activation=act
                )(previous_layer)
                if self._multi_target:
                    previous_layer = tf.keras.layers.BatchNormalization()(
                        previous_layer
                    )
            intermediate_models_out.append(previous_layer)

        # Build outputs
        final_out = []
        output_names = []
        for group_idx, group in enumerate(targets):
            for prop_idx in range(len(group)):
                previous_layer = intermediate_models_out[group_idx]
                for k in range(num_layers[2]):
                    previous_layer = tf.keras.layers.Dense(
                        num_neurons[2][k], activation=act
                    )(previous_layer)
                    if self._multi_target:
                        previous_layer = tf.keras.layers.BatchNormalization()(
                            previous_layer
                        )

                n = num_classes[group[prop_idx][0]]
                name = group[prop_idx][0]
                if n >= 2:
                    out = tf.keras.layers.Dense(
                        n,
                        activation="sigmoid" if multi_label else "softmax",
                        name=name,
                    )(previous_layer)
                else:
                    out = tf.keras.layers.Dense(
                        len(group[prop_idx]),
                        activation=out_act,
                        name=name,
                    )(previous_layer)
                final_out.append(out)
                output_names.append(name)

        new_weights = dict()
        for n in output_names:
            w = self.weights.get(n, 1)
            new_weights[n] = w
        self.weights = new_weights

        return tf.keras.models.Model(inputs=f_input, outputs=final_out)

    def _set_scale_impute(
        self, impute_missing, xscale_before_impute, scaler=None, imputer=None
    ):
        """
        Sets the inner scaling and imputer mechanism.
        impute_missing: Determines how the NaN features are treated.
                If str, defines the strategy used in the scikit-learn SimpleImputer,
                e.g., "mean" sets the NaNs to the mean of their feature column.
                If a float is provided, this float is used to replace NaNs in the original dataset.
        xscale_before_impute: whether to first scale the input and then impute values, or
                first impute values and then scale the inputs.
        scaler: optional sklearn scaler to use
        imputer: optional sklearn imputer to use
        """
        # Define the scaler
        if scaler is not None:
            self._scaler = scaler
        elif self.xscale == "minmax":
            self._scaler = MinMaxScaler(feature_range=(-0.5, 0.5))

        elif self.xscale == "standard":
            self._scaler = StandardScaler()

        # Define the imputer
        if imputer is not None:
            self._imputer = imputer
        elif isinstance(impute_missing, str):
            self._imputer = SimpleImputer(
                missing_values=np.nan, strategy=impute_missing
            )
        else:
            self._imputer = SimpleImputer(
                missing_values=np.nan, strategy="constant", fill_value=impute_missing
            )

        # Scale and impute input features in the desired order
        if xscale_before_impute:
            self._scale_impute = Pipeline(
                [("scaler", self._scaler), ("imputer", self._imputer)]
            )
        else:
            self._scale_impute = Pipeline(
                [("imputer", self._imputer), ("scaler", self._scaler)]
            )

    def fit(
        self,
        training_data: MODData,
        custom_data: Optional[np.ndarray] = None,
        val_fraction: float = 0.0,
        val_key: Optional[str] = None,
        val_data: Optional[MODData] = None,
        lr: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        xscale: Optional[str] = "minmax",
        impute_missing: Optional[Union[float, str]] = 0,
        xscale_before_impute: bool = True,
        metrics: List[str] = ["mae"],
        callbacks: List[Callable] = None,
        verbose: int = 0,
        loss: str = None,
        **fit_params,
    ) -> None:
        """Train the model on the passed training `MODData` object.

        Parameters:
            training_data: A `MODData` that has been featurized and
                feature selected. The first `self.n_feat` entries in
                `training_data.get_optimal_descriptors()` will be used
                for training.
            custom_data (np.ndarray): Optional array of shape (n_sampels, n_custom_props) that will be appended to the targets (columns wise).
                This can be useful for defining custom loss functions.
            val_fraction: The fraction of the training data to use as a
                validation set for tracking model performance during
                training.
            val_key: The target name to track on the validation set
                during training, if performing multi-target learning.
            lr: The learning rate.
            epochs: The maximum number of epochs to train for.
            batch_size: The batch size to use for training.
            xscale: The feature scaler to use, either `None`,
                `'minmax'` or `'standard'`.
            impute_missing: Determines how the NaN features are treated.
                If str, defines the strategy used in the scikit-learn SimpleImputer,
                e.g., "mean" sets the NaNs to the mean of their feature column.
                If a float is provided, and if xscale_before_impute is False, this
                float is used to replace NaNs in the original dataset.
                If a float is provided but xscale_before_impute is True, the float
                is not used and standard values are used.
                If you want to do something more sophisticated, make your own
                modifications to MODData.df_featurized before fitting the model.
            xscale_before_impute: whether to first scale the input and then impute values, or
                first impute values and then scale the inputs.
            metrics: A list of tf.keras metrics to pass to `compile(...)`.
            loss: The built-in tf.keras loss to pass to `compile(...)`.
            fit_params: Any additional parameters to pass to `fit(...)`,
                these will be overwritten by the explicit keyword
                arguments above.

        """

        if self.n_feat > len(training_data.get_optimal_descriptors()):
            raise RuntimeError(
                "The model requires more features than computed in data. "
                f"Please reduce n_feat below or equal to {len(training_data.get_optimal_descriptors())}"
            )

        self.xscale = xscale
        self.impute_missing = impute_missing
        self.target_names = list(self.weights.keys())
        self.optimal_descriptors = training_data.get_optimal_descriptors()

        x = training_data.get_featurized_df()[
            self.optimal_descriptors[: self.n_feat]
        ].values

        # For compatibility with MODNet 0.1.7; if there is only one target in the training data,
        # use that for the name of the target too.
        if (
            len(self.targets_flatten) == 1
            and len(training_data.df_targets.columns) == 1
        ):
            self.targets_flatten = list(training_data.df_targets.columns)

        y = []
        for prop in self.targets_groups:
            if self.num_classes[prop[0]] >= 2:  # Classification
                targ = prop[0]
                if self.multi_label:
                    y_inner = np.stack(training_data.df_targets[targ].values)
                    if loss is None:
                        loss = "binary_crossentropy"
                else:
                    y_inner = tf.keras.utils.to_categorical(
                        training_data.df_targets[targ].values,
                        num_classes=self.num_classes[targ],
                    )
                    if loss is None:
                        loss = "categorical_crossentropy"
            else:
                y_inner = training_data.df_targets[prop].values.astype(
                    np.float64, copy=False
                )
            if custom_data is not None:
                val_data = None
                val_fraction = 0
                metrics = []
                y_inner = np.hstack(
                    (
                        np.reshape(y_inner, (len(y_inner), -1)),
                        custom_data.reshape((len(custom_data), -1)),
                    )
                )
            y.append(y_inner)

        # set scaler and imputer
        if self.xscale == "minmax":
            impute_missing = -1 if xscale_before_impute else impute_missing
        elif self.xscale == "standard":
            impute_missing = (
                10 * np.max(np.nan_to_num(StandardScaler().fit_transform(x)))
                if xscale_before_impute
                else impute_missing
            )
        self.impute_missing = impute_missing
        self._set_scale_impute(
            impute_missing=impute_missing, xscale_before_impute=xscale_before_impute
        )

        x = self._scale_impute.fit_transform(x)

        if val_data is not None:
            val_x = val_data.get_featurized_df()[
                self.optimal_descriptors[: self.n_feat]
            ].values
            val_x = self._scale_impute.transform(val_x)
            val_y = []
            for prop in self.targets_groups:
                if self.num_classes[prop[0]] >= 2:  # Classification
                    targ = prop[0]
                    if self.multi_label:
                        y_inner = np.stack(val_data.df_targets[targ].values)
                    else:
                        y_inner = tf.keras.utils.to_categorical(
                            val_data.df_targets[targ].values,
                            num_classes=self.num_classes[targ],
                        )
                else:
                    y_inner = val_data.df_targets[prop].values.astype(
                        np.float64, copy=False
                    )
                val_y.append(y_inner)
            validation_data = (val_x, val_y)
        elif val_fraction > 0:
            x, y, validation_data = generate_shuffled_and_stratified_val_data(
                x=x,
                y=y,
                val_fraction=val_fraction,
                classification=max(self.num_classes.values()) >= 2,
            )
        else:
            validation_data = None

        # set up bounds for postprocessing
        self.min_y = []
        self.max_y = []
        for prop in self.targets_groups:
            self.min_y.append(training_data.df_targets[prop].values.min(axis=0))
            self.max_y.append(training_data.df_targets[prop].values.max(axis=0))

        # Optionally set up print callback
        if verbose:
            if validation_data:
                if self._multi_target and val_key is not None:
                    val_metric_key = f"val_{val_key}_mae"
                else:
                    val_metric_key = "val_mae"
                print_callback = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: print(
                        f"epoch {epoch}: loss: {logs['loss']:.3f}, "
                        f"val_loss:{logs['val_loss']:.3f} {val_metric_key}:{logs[val_metric_key]:.3f}"
                    )
                )

            else:
                print_callback = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: print(
                        f"epoch {epoch}: loss: {logs['loss']:.3f}"
                    )
                )

            if callbacks is None:
                callbacks = [print_callback]
            else:
                callbacks.append(print_callback)

        fit_params_kw = {
            "x": x,
            "y": y,
            "epochs": epochs,
            "batch_size": batch_size,
            "verbose": 0,
            "validation_split": val_fraction,
            "validation_data": validation_data,
            "callbacks": callbacks,
        }

        fit_params.update(fit_params_kw)

        if loss is None:
            loss = "mse"
        self.model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
            metrics=metrics,
            loss_weights=self.weights,
        )
        history = self.model.fit(**fit_params)
        self.history = history.history

    def fit_preset(
        self,
        data: MODData,
        presets: List[Dict[str, Any]] = None,
        val_fraction: float = 0.15,
        verbose: int = 0,
        classification: bool = False,
        refit: bool = True,
        fast: bool = False,
        nested: int = 5,
        callbacks: List[Any] = None,
        n_jobs=None,
        **fit_params,
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

        Sets the `self.model` attribute to the model with the lowest mean validation loss across
        all folds.

        Args:
            data: MODData object contain training and validation samples.
            presets: A list of dictionaries containing custom presets.
            verbose: The verbosity level to pass to tf.keras
            val_fraction: The fraction of the data to use for validation.
            classification: Whether or not we are performing classification.
            refit: Whether or not to refit the final model for each fold with
                the best-performing settings.
            fast: Used for debugging. If `True`, only fit the first 2 presets and
                reduce the number of epochs.
            nested: integer specifying whether or not to perform a full nested CV. If 0,
                a simple validation split is performed based on val_fraction argument.
                If an integer, use this number of inner CV folds, ignoring the `val_fraction` argument.
                Note: If set to 1, the value will be overwritten to a default of 5 folds.
            n_jobs: number of jobs for multiprocessing

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
                presets[k]["epochs"] = 100

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

        val_losses = 1e20 * np.ones((len(presets), n_splits))
        learning_curves = [[None for _ in range(n_splits)] for _ in range(len(presets))]
        models = [[None for _ in range(n_splits)] for _ in range(len(presets))]

        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=n_jobs)
        LOG.info(
            f"Multiprocessing on {n_jobs} cores. Total of {multiprocessing.cpu_count()} cores available."
        )

        for res in tqdm.tqdm(
            pool.imap_unordered(map_validate_model, tasks, chunksize=1),
            total=len(tasks),
        ):
            val_loss, learning_curve, model, preset_id, fold_id = res
            LOG.info(f"Preset #{preset_id} fitting finished, loss: {val_loss}")
            # reload the model object after serialization
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
        best_model = models[best_preset_idx][best_model_idx]

        LOG.info(
            "Preset #{} resulted in lowest validation loss with params {}".format(
                best_preset_idx + 1, tasks[n_splits * best_preset_idx + best_model_idx]
            )
        )

        if refit:
            LOG.info("Refitting with all data and parameters: {}".format(best_preset))
            # Building final model

            n_feat = min(len(data.get_optimal_descriptors()), best_preset["n_feat"])
            self.model = MODNetModel(
                self.targets,
                self.weights,
                num_neurons=best_preset["num_neurons"],
                n_feat=n_feat,
                act=best_preset["act"],
                out_act=self.out_act,
                num_classes=self.num_classes,
            ).model
            self.n_feat = n_feat
            self.fit(
                data,
                val_fraction=0,
                lr=best_preset["lr"],
                epochs=best_preset["epochs"],
                batch_size=best_preset["batch_size"],
                loss=best_preset["loss"],
                callbacks=callbacks,
                verbose=verbose,
                **fit_params,
            )
        else:
            self.n_feat = best_model.n_feat
            self.model = best_model.model
            self._scaler = best_model._scaler
            self._imputer = best_model._imputer
            self._scale_impute = best_model._scale_impute

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # reset

        return models, val_losses, best_learning_curve, learning_curves, best_preset

    def predict(
        self,
        test_data: MODData,
        return_prob: bool = False,
        remap_out_of_bounds: bool = True,
    ) -> pd.DataFrame:
        """Predict the target values for the passed MODData.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.
            return_prob: For a classification tasks only: whether to return the probability of each
                class OR only return the most probable class.
            remap_out_of_bounds: Whether to remap out-of-bounds predictions to the training data distribution.

        Returns:
            A `pandas.DataFrame` containing the predicted values of the targets.


        """
        # prevents Nan predictions if some features are inf
        x = (
            test_data.get_featurized_df()
            .replace([np.inf, -np.inf], np.nan)[self.optimal_descriptors[: self.n_feat]]
            .values
        )

        # Scale and impute input features:
        if self._scale_impute is not None:
            x = self._scale_impute.transform(x)

        p = self.model.predict(x)

        if len(self.targets_groups) == 1:
            p = [p]

        # post-process based on training data
        if remap_out_of_bounds:
            if max(self.num_classes.values()) <= 2:  # regression
                for i, vals in enumerate(p):
                    yrange = self.max_y[i] - self.min_y[i]
                    upper_bound = self.max_y[i] + 0.25 * yrange
                    lower_bound = self.min_y[i] - 0.25 * yrange
                    for j in range(len(self.targets_groups[i])):
                        out_of_range_idxs = np.where(
                            (vals[:, j] < lower_bound[j])
                            | (vals[:, j] > upper_bound[j])
                        )
                        vals[out_of_range_idxs, j] = (
                            np.random.uniform(0, 1, size=len(out_of_range_idxs[0]))
                            * (yrange[j])
                            + self.min_y[i][j]
                        )

        p_dic = {}

        for i, props in enumerate(self.targets_groups):
            name = props[0]
            if self.num_classes[name] >= 2:
                if return_prob:
                    temp = p[i]
                    for j in range(temp.shape[-1]):
                        p_dic["{}_prob_{}".format(name, j)] = temp[:, j]
                else:
                    p_dic[name] = np.argmax(p[i], axis=1)
            else:
                for j, name in enumerate(props):
                    p_dic[name] = p[i][:, j]
        predictions = pd.DataFrame(p_dic, index=pd.Index(test_data.structure_ids))

        return predictions

    def evaluate(
        self,
        test_data: MODData,
        loss: Union[str, Callable] = "mae",
    ) -> pd.DataFrame:
        """Evaluates predictions on the passed MODData by returning the corresponding score:
            - for regression: loss function provided in loss argument. Defaults to mae.
            - for classification: negative ROC AUC.
            averaged over the targets when multi-target.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.


        Returns:
            Score defined hereabove.
        """
        # prevents Nan predictions if some features are inf
        x = (
            test_data.get_featurized_df()
            .replace([np.inf, -np.inf], np.nan)[self.optimal_descriptors[: self.n_feat]]
            .values
        )

        # Scale and impute input features:
        if self._scale_impute is not None:
            x = self._scale_impute.transform(x)

        y_pred = self.model.predict(x)
        if len(self.targets_groups) == 1:
            y_pred = [y_pred]

        score = []
        for i, prop in enumerate(self.targets_groups):
            if self.num_classes[prop[0]] >= 2:  # Classification
                targ = prop[0]
                if self.multi_label:
                    y_true = np.stack(test_data.df_targets[targ].values)
                else:
                    y_true = tf.keras.utils.to_categorical(
                        test_data.df_targets[targ].values,
                        num_classes=self.num_classes[targ],
                    )
                try:
                    score.append(-roc_auc_score(y_true, y_pred[i], multi_class="ovr"))
                except ValueError:
                    scores = []
                    for j in range(y_true.shape[1]):
                        try:
                            scores.append(-roc_auc_score(y_true[:, j], y_pred[i][:, j]))
                        except ValueError:
                            scores.append(float("nan"))
                    score.append(np.nanmean(scores))
            else:
                y_true = test_data.df_targets[prop].values.astype(
                    np.float64, copy=False
                )
                if loss == "mae":
                    loss = mean_absolute_error
                elif loss == "mse":
                    loss = mean_squared_error
                elif isinstance(loss, str):
                    raise RuntimeError(
                        f"Loss {loss} not recognized. Use mae, mse or a callable."
                    )
                else:
                    pass

                score.append(loss(y_true, y_pred[i]))

        return np.mean(score)

    def _make_picklable(self):
        """
        transforms inner keras model to jsons so that th MODNet object becomes picklable.
        """

        model_json = self.model.to_json()
        model_weights = self.model.get_weights()

        self.model = (model_json, model_weights)

    def _restore_model(self):
        """
        restore inner keras model after running make_picklable
        """

        model_json, model_weights = self.model
        self.model = tf.keras.models.model_from_json(model_json)
        self.model.set_weights(model_weights)
        if not hasattr(self, "_scale_impute"):
            self.xscale = "minmax"
            self._set_scale_impute(
                impute_missing=-1,
                xscale_before_impute=True,
                scaler=self._scaler,
                imputer=SimpleImputer(
                    missing_values=np.nan,
                    strategy="constant",
                    fill_value=-1,
                ).fit(np.zeros((1, self.n_feat))),
            )

    def save(self, filename: str) -> None:
        """Save the `MODNetModel` to filename:

        If the filename ends in "tgz", "bz2" or "zip", the pickle
        will be compressed accordingly by :meth:`pandas.DataFrame.to_pickle`.

        Parameters:
            filename: The base filename to save to.


        """
        self._make_picklable()
        pd.to_pickle(self, filename)
        self._restore_model()
        LOG.info(f"Model successfully saved as {filename}!")

    @staticmethod
    def load(filename: str) -> "MODNetModel":
        """Load `MODNetModel` object pickled by the :meth:`MODNetModel.save` method.

        If the filename ends in "tgz", "bz2" or "zip", the pickle
        will be decompressed accordingly by :func:`pandas.read_pickle`.

        Returns:
            The loaded `MODNetModel` object.
        """
        pickled_data = None

        if isinstance(filename, Path):
            filename = str(filename)

        # handle .zip files explicitly for OS X/macOS compatibility
        if filename.endswith(".zip"):
            from zipfile import ZipFile

            with ZipFile(filename, "r") as zf:
                namelist = zf.namelist()
                _files = [
                    _
                    for _ in namelist
                    if not _.startswith("__MACOSX/") or _.startswith(".DS_STORE")
                ]
                if len(_files) == 1:
                    with zf.open(_files.pop()) as f:
                        pickled_data = pd.read_pickle(f)

        if pickled_data is None:
            pickled_data = pd.read_pickle(filename)

        if isinstance(pickled_data, MODNetModel):
            if not hasattr(pickled_data, "__modnet_version__"):
                pickled_data.__modnet_version__ = "unknown"
            pickled_data._restore_model()
            LOG.info(
                f"Loaded {pickled_data} object, created with modnet version {pickled_data.__modnet_version__}"
            )
            if hasattr(pickled_data, "models"):
                for i, m in enumerate(pickled_data.models):  # ensemble
                    if not hasattr(m, "targets_groups"):
                        LOG.warning(
                            "Loaded model is old (v<0.4.0) and will not be supported in the future (v1.0.0 onward). Please consider retraining your model!\nLoaded with DepractedMODNetModel."
                        )
                        recovered_data = DeprecatedMODNetModel(targets=[], weights={})
                        recovered_data.__dict__ = m.__dict__.copy()
                        pickled_data.models[i] = recovered_data
            else:
                if not hasattr(pickled_data, "targets_groups"):  # single model
                    LOG.warning(
                        "Loaded model is old (v<0.4.0) and will not be supported in the future (v1.0.0 onward). Please consider retraining your model!\nLoaded with DepractedMODNetModel."
                    )
                    recovered_data = DeprecatedMODNetModel(targets=[], weights={})
                    recovered_data.__dict__ = pickled_data.__dict__.copy()
                    pickled_data.model = recovered_data
            return pickled_data

        raise ValueError(
            f"File {filename} did not contain compatible data to create a MODNetModel object, "
            f"instead found {pickled_data.__class__.__name__}."
        )

    def _get_param_names(self):
        possible_params = [
            "targets",
            "weights",
            "num_neurons",
            "num_classes",
            "multi_label",
            "n_feat",
            "act",
            "out_act",
        ]
        return possible_params

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Taken from sklearn.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()

        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Taken from sklearn.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            # TODO(1.4): remove specific handling of "base_estimator".
            # The "base_estimator" key is special. It was deprecated and
            # renamed to "estimator" for several estimators. This means we
            # need to translate it here and set sub-parameters on "estimator",
            # but only if the user did not explicitly set a value for
            # "base_estimator".
            if (
                key == "base_estimator"
                and valid_params[key] == "deprecated"
                and self.__module__.startswith("sklearn.")
            ):
                warnings.warn(
                    f"Parameter 'base_estimator' of {self.__class__.__name__} is"
                    " deprecated in favor of 'estimator'. See"
                    f" {self.__class__.__name__}'s docstring for more details.",
                    FutureWarning,
                    stacklevel=2,
                )
                key = "estimator"
            valid_params[key].set_params(**sub_params)

        return self


class DeprecatedMODNetModel(MODNetModel):
    def build_model(
        self,
        targets: List,
        n_feat: int,
        num_neurons: Tuple[List[int], List[int], List[int], List[int]],
        num_classes: Optional[Dict[str, int]] = None,
        multi_label: Optional[bool] = False,
        act: str = "relu",
        out_act: str = "linear",
    ):
        """Builds the tf.keras model and sets the `self.model` attribute.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            n_feat: The number of features to use as model inputs.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `tf.keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                with n=0 for regression and n>=2 for classification with n the number of classes.
            multi_label: Whether the problem (if classification) is multi-label.
                In this case the softmax output-activation is replaced by a sigmoid.
            act: A string defining a tf.keras activation function to pass to use
                in the `tf.keras.layers.Dense` layers.
            out_act: A string defining a tf.keras activation function to pass to use
                for the last output layer (regression only)

        """

        num_layers = [len(x) for x in num_neurons]

        # Build first common block
        f_input = tf.keras.layers.Input(shape=(n_feat,))
        previous_layer = f_input
        for i in range(num_layers[0]):
            previous_layer = tf.keras.layers.Dense(num_neurons[0][i], activation=act)(
                previous_layer
            )
            if self._multi_target:
                previous_layer = tf.keras.layers.BatchNormalization()(previous_layer)
        common_out = previous_layer

        # Build intermediate representations
        intermediate_models_out = []
        for _ in range(len(targets)):
            previous_layer = common_out
            for j in range(num_layers[1]):
                previous_layer = tf.keras.layers.Dense(
                    num_neurons[1][j], activation=act
                )(previous_layer)
                if self._multi_target:
                    previous_layer = tf.keras.layers.BatchNormalization()(
                        previous_layer
                    )
            intermediate_models_out.append(previous_layer)

        # Build outputs
        final_out = []
        for group_idx, group in enumerate(targets):
            for prop_idx in range(len(group)):
                previous_layer = intermediate_models_out[group_idx]
                for k in range(num_layers[2]):
                    previous_layer = tf.keras.layers.Dense(
                        num_neurons[2][k], activation=act
                    )(previous_layer)
                    if self._multi_target:
                        previous_layer = tf.keras.layers.BatchNormalization()(
                            previous_layer
                        )
                clayer = previous_layer
                for pi in range(len(group[prop_idx])):
                    previous_layer = clayer
                    for li in range(num_layers[3]):
                        previous_layer = tf.keras.layers.Dense(num_neurons[3][li])(
                            previous_layer
                        )
                    n = num_classes[group[prop_idx][pi]]
                    if n >= 2:
                        out = tf.keras.layers.Dense(
                            n,
                            activation="sigmoid" if multi_label else "softmax",
                            name=group[prop_idx][pi],
                        )(previous_layer)
                    else:
                        out = tf.keras.layers.Dense(
                            1, activation=out_act, name=group[prop_idx][pi]
                        )(previous_layer)
                    final_out.append(out)

        return tf.keras.models.Model(inputs=f_input, outputs=final_out)

    def fit(
        self,
        training_data: MODData,
        custom_data: Optional[np.ndarray] = None,
        val_fraction: float = 0.0,
        val_key: Optional[str] = None,
        val_data: Optional[MODData] = None,
        lr: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        xscale: Optional[str] = "minmax",
        impute_missing: Optional[Union[float, str]] = 0,
        xscale_before_impute: bool = True,
        metrics: List[str] = ["mae"],
        callbacks: List[Callable] = None,
        verbose: int = 0,
        loss: str = None,
        **fit_params,
    ) -> None:
        """Train the model on the passed training `MODData` object.

        Parameters:
            training_data: A `MODData` that has been featurized and
                feature selected. The first `self.n_feat` entries in
                `training_data.get_optimal_descriptors()` will be used
                for training.
            custom_data (np.ndarray): Optional array of shape (n_sampels, n_custom_props) that will be appended to the targets (columns wise).
                This can be useful for defining custom loss functions.
            val_fraction: The fraction of the training data to use as a
                validation set for tracking model performance during
                training.
            val_key: The target name to track on the validation set
                during training, if performing multi-target learning.
            lr: The learning rate.
            epochs: The maximum number of epochs to train for.
            batch_size: The batch size to use for training.
            xscale: The feature scaler to use, either `None`,
                `'minmax'` or `'standard'`.
            impute_missing: Determines how the NaN features are treated.
                If str, defines the strategy used in the scikit-learn SimpleImputer,
                e.g., "mean" sets the NaNs to the mean of their feature column.
                If a float is provided, and if xscale_before_impute is False, this
                float is used to replace NaNs in the original dataset.
                If a float is provided but xscale_before_impute is True, the float
                is not used and standard values are used.
                If you want to do something more sophisticated, make your own
                modifications to MODData.df_featurized before fitting the model.
            xscale_before_impute: whether to first scale the input and then impute values, or
                first impute values and then scale the inputs.
            metrics: A list of tf.keras metrics to pass to `compile(...)`.
            loss: The built-in tf.keras loss to pass to `compile(...)`.
            fit_params: Any additional parameters to pass to `fit(...)`,
                these will be overwritten by the explicit keyword
                arguments above.

        """

        if self.n_feat > len(training_data.get_optimal_descriptors()):
            raise RuntimeError(
                "The model requires more features than computed in data. "
                f"Please reduce n_feat below or equal to {len(training_data.get_optimal_descriptors())}"
            )

        self.xscale = xscale
        self.impute_missing = impute_missing
        self.target_names = list(self.weights.keys())
        self.optimal_descriptors = training_data.get_optimal_descriptors()

        x = training_data.get_featurized_df()[
            self.optimal_descriptors[: self.n_feat]
        ].values

        # For compatibility with MODNet 0.1.7; if there is only one target in the training data,
        # use that for the name of the target too.
        if (
            len(self.targets_flatten) == 1
            and len(training_data.df_targets.columns) == 1
        ):
            self.targets_flatten = list(training_data.df_targets.columns)

        y = []
        for targ in self.targets_flatten:
            if self.num_classes[targ] >= 2:  # Classification
                if self.multi_label:
                    y_inner = np.stack(training_data.df_targets[targ].values)
                    if loss is None:
                        loss = "binary_crossentropy"
                else:
                    y_inner = tf.keras.utils.to_categorical(
                        training_data.df_targets[targ].values,
                        num_classes=self.num_classes[targ],
                    )
                    if loss is None:
                        loss = "categorical_crossentropy"
            else:
                y_inner = training_data.df_targets[targ].values.astype(
                    np.float64, copy=False
                )
            if custom_data is not None:
                val_data = None
                val_fraction = 0
                metrics = []
                y_inner = np.hstack(
                    (
                        np.reshape(y_inner, (len(y_inner), -1)),
                        custom_data.reshape((len(custom_data), -1)),
                    )
                )
            y.append(y_inner)

        # set scaler and imputer
        if self.xscale == "minmax":
            impute_missing = -1 if xscale_before_impute else impute_missing
        elif self.xscale == "standard":
            impute_missing = (
                10 * np.max(np.nan_to_num(StandardScaler().fit_transform(x)))
                if xscale_before_impute
                else impute_missing
            )
        self.impute_missing = impute_missing
        self._set_scale_impute(
            impute_missing=impute_missing, xscale_before_impute=xscale_before_impute
        )

        x = self._scale_impute.fit_transform(x)

        if val_data is not None:
            val_x = val_data.get_featurized_df()[
                self.optimal_descriptors[: self.n_feat]
            ].values
            val_x = self._scale_impute.transform(val_x)
            val_y = []
            for targ in self.targets_flatten:
                if self.num_classes[targ] >= 2:  # Classification
                    if self.multi_label:
                        y_inner = np.stack(val_data.df_targets[targ].values)
                        if loss is None:
                            loss = "binary_crossentropy"
                    else:
                        y_inner = tf.keras.utils.to_categorical(
                            val_data.df_targets[targ].values,
                            num_classes=self.num_classes[targ],
                        )
                else:
                    y_inner = val_data.df_targets[targ].values.astype(
                        np.float64, copy=False
                    )
                val_y.append(y_inner)
            validation_data = (val_x, val_y)
        elif val_fraction > 0:
            x, y, validation_data = generate_shuffled_and_stratified_val_data(
                x=x,
                y=y,
                val_fraction=val_fraction,
                classification=max(self.num_classes.values()) >= 2,
            )
        else:
            validation_data = None

        # set up bounds for postprocessing
        if max(self.num_classes.values()) <= 2:  # regression
            self.min_y = training_data.df_targets.values.min(axis=0)
            self.max_y = training_data.df_targets.values.max(axis=0)

        # Optionally set up print callback
        if verbose:
            if validation_data:
                if self._multi_target and val_key is not None:
                    val_metric_key = f"val_{val_key}_mae"
                else:
                    val_metric_key = "val_mae"
                print_callback = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: print(
                        f"epoch {epoch}: loss: {logs['loss']:.3f}, "
                        f"val_loss:{logs['val_loss']:.3f} {val_metric_key}:{logs[val_metric_key]:.3f}"
                    )
                )

            else:
                print_callback = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: print(
                        f"epoch {epoch}: loss: {logs['loss']:.3f}"
                    )
                )

            if callbacks is None:
                callbacks = [print_callback]
            else:
                callbacks.append(print_callback)

        fit_params_kw = {
            "x": x,
            "y": y,
            "epochs": epochs,
            "batch_size": batch_size,
            "verbose": 0,
            "validation_split": val_fraction,
            "validation_data": validation_data,
            "callbacks": callbacks,
        }

        fit_params.update(fit_params_kw)
        if "learning_rate" in fit_params:
            fit_params.pop("learning_rate")
            warnings.warn("learning_rate is deprecated, use lr instead.")

        if loss is None:
            loss = "mse"
        self.model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
            metrics=metrics,
            loss_weights=self.weights,
        )
        history = self.model.fit(**fit_params)
        self.history = history.history

    def predict(self, test_data: MODData, return_prob=False) -> pd.DataFrame:
        """Predict the target values for the passed MODData.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.
            return_prob: For a classification tasks only: whether to return the probability of each
                class OR only return the most probable class.

        Returns:
            A `pandas.DataFrame` containing the predicted values of the targets.


        """
        # prevents Nan predictions if some features are inf
        x = (
            test_data.get_featurized_df()
            .replace([np.inf, -np.inf], np.nan)[self.optimal_descriptors[: self.n_feat]]
            .values
        )

        # Scale and impute input features:
        if self._scale_impute is not None:
            x = self._scale_impute.transform(x)

        p = np.array(self.model.predict(x))

        if len(p.shape) == 2:
            p = np.array([p])

        # post-process based on training data
        if max(self.num_classes.values()) <= 2:  # regression
            yrange = self.max_y - self.min_y
            upper_bound = self.max_y + 0.25 * yrange
            lower_bound = self.min_y - 0.25 * yrange
            for i, vals in enumerate(p):
                out_of_range_idxs = np.where(
                    (vals < lower_bound[i]) | (vals > upper_bound[i])
                )
                vals[out_of_range_idxs] = (
                    np.random.uniform(0, 1, size=len(out_of_range_idxs[0]))
                    * (self.max_y[i] - self.min_y[i])
                    + self.min_y[i]
                )

        p_dic = {}
        for i, name in enumerate(self.targets_flatten):
            if self.num_classes[name] >= 2:
                if return_prob:
                    # temp = p[i, :, :] / (p[i, :, :].sum(axis=1)).reshape((-1, 1))
                    temp = p[i, :, :]
                    for j in range(temp.shape[-1]):
                        p_dic["{}_prob_{}".format(name, j)] = temp[:, j]
                else:
                    p_dic[name] = np.argmax(p[i, :, :], axis=1)
            else:
                p_dic[name] = p[i, :, 0]
        predictions = pd.DataFrame(p_dic)
        predictions.index = test_data.structure_ids

        return predictions

    def evaluate(self, test_data: MODData) -> pd.DataFrame:
        """Evaluates predictions on the passed MODData by returning the corresponding score:
            - for regression: MAE
            - for classification: negative ROC AUC.
            averaged over the targets when multi-target.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.


        Returns:
            Score defined hereabove.
        """
        # prevents Nan predictions if some features are inf
        x = (
            test_data.get_featurized_df()
            .replace([np.inf, -np.inf], np.nan)[self.optimal_descriptors[: self.n_feat]]
            .values
        )

        # Scale and impute input features:
        if self._scale_impute is not None:
            x = self._scale_impute.transform(x)

        y_pred = np.array(self.model.predict(x))
        if len(y_pred.shape) == 2:
            y_pred = np.array([y_pred])
        score = []
        for i, targ in enumerate(self.targets_flatten):
            if self.num_classes[targ] >= 2:  # Classification
                if self.multi_label:
                    y_true = np.stack(test_data.df_targets[targ].values)
                else:
                    y_true = tf.keras.utils.to_categorical(
                        test_data.df_targets[targ].values,
                        num_classes=self.num_classes[targ],
                    )
                try:
                    score.append(-roc_auc_score(y_true, y_pred[i], multi_class="ovr"))
                except ValueError:
                    scores = []
                    for j in range(y_true.shape[1]):
                        try:
                            scores.append(-roc_auc_score(y_true[:, j], y_pred[i][:, j]))
                        except ValueError:
                            scores.append(float("nan"))
                    score.append(np.nanmean(scores))
            else:
                y_true = test_data.df_targets[targ].values.astype(
                    np.float64, copy=False
                )
                score.append(mean_absolute_error(y_true, y_pred[i]))

        return np.mean(score)


def validate_model(
    train_data=None,
    val_data=None,
    targets=None,
    weights=None,
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
    callbacks=[],
    preset_id=None,
    fold_id=None,
    verbose=0,
):
    """For a given set of parameters, create a new model and train it on the passed training data,
    validating it against the passed validation data and returning some relevant metrics.

    """

    model = MODNetModel(
        targets,
        weights,
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
        callbacks=callbacks,
        verbose=verbose,
        val_fraction=0,
        val_data=val_data,
    )

    learning_curve = model.history["val_loss"]

    val_loss = model.evaluate(val_data)

    # save model
    model._make_picklable()

    return val_loss, learning_curve, model, preset_id, fold_id


def map_validate_model(kwargs):
    return validate_model(**kwargs)


def generate_shuffled_and_stratified_val_data(
    x: np.ndarray,
    y: list[np.ndarray],
    val_fraction: float,
    classification: bool,
):
    """
    Generate train and validation data that is shuffled, reproducible and, if classification, stratified.
    Please note for classification tasks that stratification is performed on first target.
    y: list of 2D array with combined dimensions of (n_targets, n_samples, 1 or n_classes)
    """
    y_split = np.array([y[0]]).swapaxes(1, 0)
    train_idx, val_idx = generate_shuffled_and_stratified_val_split(
        y=y_split, val_fraction=val_fraction, classification=classification
    )
    return (
        x[train_idx],
        [t[train_idx] for t in y],
        (x[val_idx], [t[val_idx] for t in y]),
    )
