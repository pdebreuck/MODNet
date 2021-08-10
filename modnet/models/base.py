"""This submodule defines the interface to MODNet models of
various types.

"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Callable, Any
from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd

from modnet import __version__
from modnet.utils import LOG
from modnet.preprocessing import MODData


class BaseMODNetModel(ABC):
    """Base class for models implemented in the MODNet package.

    These models expect to be provided ``MODData`` objects
    as training input, and are typically defined via the
    Keras interface within TensorFlow.

    This class defines interfaces for setting up architectures,
    training and hyperparameter optimization.

    Attributes:
        n_feat: The number of features used in the model.
        weights: The relative loss weights for each target.
        optimal_descriptors: The list of column names used
            in training the model.
        model: The `tf.keras.model.Model` of the network itself.
        target_names: The list of targets names that the model
            was trained for.

    """

    can_return_uncertainty: bool = False
    """Whether or not this model supports the ``return_unc`` parameter
    in its `predict` method, which will enable returning uncertainties.
    """

    def __init__(
        self,
        targets: List,
        weights: Dict[str, float],
        num_neurons: Tuple[List[int], List[int], List[int], List[int]] = (
            [64],
            [32],
            [16],
            [16],
        ),
        num_classes: Optional[Dict[str, int]] = None,
        n_feat: int = 64,
        act: str = "relu",
        out_act: str = "linear",
        **model_kwargs,
    ):
        """Initialise the model on the passed targets with the desired
        architecture, feature count and loss functions and activation functions.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            weights: The relative loss weights to apply for each target.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `tf.keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                 with n=0 for regression and n>=2 for classification with n the number of classes.
            n_feat: The number of features to use as model inputs.
            act: A string defining a tf.keras activation function to pass to use
                in the `tf.keras.layers.Dense` layers.
            out_act: A string defining a tf.keras activation function to pass to use
                for the last output layer (regression only)
            model_kwargs: Any additional keyword arguments to pass to ``self.build_model(...)``.

        """

        self.__modnet_version__ = __version__

        if n_feat is None:
            n_feat = 64
        self.n_feat = n_feat
        self.weights = weights

        self.num_classes = num_classes
        self.num_neurons = num_neurons
        self.act = act
        self.out_act = out_act

        self._scaler = None
        self.optimal_descriptors = None
        self.target_names = None
        self.targets = targets
        self.model = None

        f_temp = [x for subl in targets for x in subl]
        self.targets_flatten = [x for subl in f_temp for x in subl]
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
            **model_kwargs,
        )

    @abstractmethod
    def build_model(
        self,
        targets: List,
        n_feat: int,
        num_neurons: Tuple[List[int], List[int], List[int], List[int]],
        num_classes: Optional[Dict[str, int]] = None,
        act: str = "relu",
        out_act: str = "linear",
        **model_kwargs,
    ) -> tf.keras.Model:
        """This method takes the class initialization arguments and builds the underlying
        ``tf.keras.Model``. Arguments take their meaning from the the base class ``__init__()``.

        """

    @abstractmethod
    def fit(
        self,
        training_data: MODData,
        val_fraction: float = 0.0,
        val_key: Optional[str] = None,
        val_data: Optional[MODData] = None,
        lr: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        xscale: Optional[str] = "minmax",
        metrics: List[str] = ["mae"],
        callbacks: List[Callable] = None,
        verbose: int = 0,
        loss: str = "mse",
        **fit_params,
    ) -> None:
        """Train the model on the passed training data."""

    @abstractmethod
    def predict(self, test_data, return_prob=False, return_unc=False):
        """Make predictions on the test data."""
        pass

    @abstractmethod
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
        n_jobs: Optional[int] = None,
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

    @abstractmethod
    def evaluate(self, test_data: MODData) -> pd.DataFrame:
        """Evaluates the target values for the passed `MODData` and returns the corresponding loss.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.

        Returns:
            An array containing the defined losses for the model on the passed test data.

        """

    def save(self, filename: str):
        """Save the `MODNetModel` to filename:

        Parameters:
            filename: The base filename to save to.

        If the filename ends in "tgz", "bz2" or "zip", the pickle
        will be compressed accordingly by `pandas.to_pickle(...)`.

        """
        self._make_picklable()
        pd.to_pickle(self, filename)
        self._restore_model()
        LOG.info(f"Model successfully saved as {filename}!")

    def _make_picklable(self):
        """Transforms inner Keras model to JSON (serialization) such that model becomes picklable."""
        model_json = self.model.to_json()
        model_weights = self.model.get_weights()
        self.model = (model_json, model_weights)

    def _restore_model(self):
        """Restore the inner keras model from JSON (deserialization) to a full Keras model."""

        model_json, model_weights = self.model
        self.model = tf.keras.models.model_from_json(model_json)
        self.model.set_weights(model_weights)

    @staticmethod
    def load(filename: str) -> "BaseMODNetModel":
        """Load `MODNetModel` object pickled by the `.save(...)` method.

        If the filename ends in "tgz", "bz2" or "zip", the pickle
        will be decompressed accordingly by `pandas.read_pickle(...)`.

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

        if isinstance(pickled_data, BaseMODNetModel):
            if not hasattr(pickled_data, "__modnet_version__"):
                pickled_data.__modnet_version__ = "unknown"
            pickled_data._restore_model()
            LOG.info(
                f"Loaded {pickled_data} object, created with modnet version {pickled_data.__modnet_version__}"
            )
            return pickled_data

        raise ValueError(
            f"File {filename} did not contain compatible data to create a `BaseMODNetModel` object, "
            f"instead found {pickled_data.__class__.__name__}."
        )
