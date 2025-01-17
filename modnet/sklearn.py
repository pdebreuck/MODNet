"""
sklearn API of modnet

This version implements the RR class of the sklearn API

The general pipeline will be:
from sklearn.pipeline import Pipeline
modnet_featurizer = MODNetFeaturizer(...arguments here...)
rr_analysis = RR(...arguments here...)
modnet_model = MODNet(...arguments here...)
p = Pipeline([('featurizer', modnet_featurizer), ('rr', rr_analysis), ('modnet', modnet_model)])

* One note about scikit learn's steps when performing cross-validation:
  A given transformer (e.g. PCA, or RR) will not be executed at each step of the cross validation if you cache it, i.e.
  scikit-learn detects that the inputs are the same (if they are indeed the same) and does not do it several times
  (see https://scikit-learn.org/stable/modules/compose.html#caching-transformers-avoid-repeated-computation).
* Another note about the fit method of a Pipeline object:
  It is possible to pass fit parameters to each step of a Pipeline object
  (see https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit)
* Another note about the __init__ method of sklearn derived estimators:
  The method should ONLY set the instance variables. In particular, "every keyword argument accepted by __init__
  should correspond to an attribute on the instance" (as stated in sklearn's developer documentation)
  (see https://scikit-learn.org/stable/developers/develop.html#instantiation).
"""

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.base import TransformerMixin
from typing import Union, Dict, Optional, List, Callable
from sklearn.utils.multiclass import type_of_target
from modnet.preprocessing import (
    MODData,
    get_features_relevance_redundancy,
    get_cross_nmi,
    nmi_target,
)
from modnet.utils import LOG
from modnet.featurizers import MODFeaturizer
from modnet.models import MODNetModel
import numpy as np
import pandas as pd


class MODNetFeaturizer(TransformerMixin, BaseEstimator):
    """Featurization class"""

    def __init__(
        self,
        featurizer: Optional[Union[MODFeaturizer, str]] = None,
    ):
        """Constructor for MODNetFeaturizer

        Parameters:
            featurizer: Optional modnet.featurizers.MODfeaturizer instance for specific feauriation
        """
        self.featurizer = featurizer

    def fit(self, X, y=None):
        """Ignored"""
        return self

    def transform(self, X, y=None, n_jobs: int = None):
        """Transform the input data (i.e. containing the composition and/or structure to the features
        Parameters:
            X: list of pymatgen structures or compositions
            y: ignored

        Returns
            ndarray of size (n_samples, n_features)
        """
        data = MODData(X, featurizer=self.featurizer)
        data.featurize(n_jobs=n_jobs)

        return data.df_featurized


class RR(TransformerMixin, BaseEstimator):
    """Relevance-Redundancy (RR) feature selection.
    Features are ranked and selected following a relevance-redundancy ratio as developed
    by De Breuck et al. (2020), see https://arxiv.org/abs/2004.14766.

    Use the fit method for computing the most important features.
    Then use the transform method to truncate the input data to those features.

    Parameters:
        n_feat: int, number of features to keep
        optimal_descriptors: list of length (n_feat)ordered list of best descriptors

    """

    def __init__(
        self,
        n_feat: Union[None, int] = None,
        rr_parameters: Union[None, Dict] = None,
        use_precomputed_cross_nmi: bool = False,
        drop_thr: float = 0.2,
    ):
        """Constructor for RR transformer.

        Parameters:
            n_feat: Number of features to keep and reorder using the RR procedure (default: None, i.e. all features).
            rr_parameters: Allows tuning of p and c parameters. Currently allows fixing of p and c
                to constant values instead of using the dynamical evaluation. Expects to find keys `"p"` and `"c"`,
                containing either a callable that takes `n` as an argument and returns the desired `p` or `c`,
                or another dictionary containing the key `"value"` that stores a constant value of `p` or `c`.
        """
        self.n_feat = n_feat
        self.rr_parameters = rr_parameters
        self.optimal_descriptors = []
        self.use_precomputed_cross_nmi = use_precomputed_cross_nmi
        self.drop_thr = drop_thr

    def fit(
        self, X, y, n_jobs: int = None, nmi_feats_target=None, cross_nmi_feats=None
    ):
        """Ranking of the features. This is based on relevance and redundancy provided as NMI dataframes.
        If not provided (i.e set to None), the NMIs are computed here.
        Nevertheless, it is strongly recommended to compute them separately and store them locally.

        Parameters:
            X: Array or pandas dataframe of shape (n_samples, n_features)
            y: array or dataframe of shape (n_samples,)
            nmi_feats_target: NMI between features and targets, pandas dataframe
            cross_nmi_feats: NMI between features, pandas dataframe

        Returns:
            self : object
            Fitted RR transformer
        """

        if cross_nmi_feats is None:
            if self.use_precomputed_cross_nmi:
                LOG.info("Loading cross NMI from 'Features_cross' file.")
                from modnet.ext_data import load_ext_dataset

                cnmi_path = load_ext_dataset("MP_2018.6_CROSS_NMI", "cross_nmi")
                cross_nmi_feats = pd.read_pickle(cnmi_path)
            else:
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(
                        X, columns=["f" + str(i) for i in range(X.shape[1])]
                    )
                cross_nmi_feats = get_cross_nmi(
                    X, drop_thr=self.drop_thr, n_jobs=n_jobs
                )
        if nmi_feats_target is None:
            if isinstance(y, np.ndarray) or isinstance(y, list):
                y = pd.DataFrame({"p0": y})
            if type_of_target(y.values) != "continuous":
                task_type = "classification"
            else:
                task_type = "regression"
            nmi_feats_target = nmi_target(X, y, task_type=task_type)

        rr_results = get_features_relevance_redundancy(
            nmi_feats_target,
            cross_nmi_feats,
            n_feat=self.n_feat,
            rr_parameters=self.rr_parameters,
        )
        self.optimal_descriptors = [x["feature"] for x in rr_results]

        return self

    def transform(self, X, y=None):
        """Transform the inputs X based on a fitted RR analysis. The best n_feat features are kept and returned.

        Parameters:
            X: array or pandas dataframe of shape (n_samples,n_features)
            y: ignored

        Returns:
            X data containing n_feat rows (best features) as a pandas dataframe or array (same as input type)
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=["f" + str(i) for i in range(X.shape[1])])
        print(self.optimal_descriptors)
        print(self.n_feat)
        return X[self.optimal_descriptors[: self.n_feat]]


class MODNetRegressor(RegressorMixin, BaseEstimator):
    """MODNet model."""

    def __init__(
        self,
        targets=None,
        weights=None,
        num_neurons=[[64], [32], [16], [16]],
        act: str = "relu",
        out_act: str = "linear",
        lr: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        xscale: Optional[str] = "minmax",
        metrics: List[str] = ["mae"],
        callbacks: List[Callable] = None,
        verbose: int = 0,
        loss: str = "mse",
    ):

        """Constructor for MODNet model.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            weights: The relative loss weights to apply for each target.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `tf.keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            act: A string defining a tf.keras activation function to pass to use
                in the `tf.keras.layers.Dense` layers.
            out_act: A string defining a tf.keras activation function to pass to use
                for the last output layer (regression only)
            lr: The learning rate.
            epochs: The maximum number of epochs to train for.
            batch_size: The batch size to use for training.
            xscale: The feature scaler to use, either `None`,
                `'minmax'` or `'standard'`.
            metrics: A list of tf.keras metrics to pass to `compile(...)`.
            callbacks: A list of tf.keras callbacks that will be provided to the model.
            verbose: integer handling verbosity.
            loss: The built-in tf.keras loss to pass to `compile(...)`.
        """
        self.targets = targets
        self.weights = weights
        self.num_neurons = num_neurons
        self.act = act
        self.out_act = out_act
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.xscale = xscale
        self.metrics = metrics
        self.callbacks = callbacks
        self.verbose = verbose
        self.loss = loss

    def fit(self, X, y):
        """
        Fit a MODNet regression model.

        Parameters:
            X: array or dataframe of shape (n_samples, n_features)
            y: array-like of shape (n_samples,) or (n_samples, n_properties)

        Returns:
            Fitted estimator
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=["f" + str(i) for i in range(X.shape[1])])

        if isinstance(y, list):
            y = np.array(y)

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        data_train = MODData(
            [None for _ in range(len(X))],
            df_featurized=X,
            targets=y,
            target_names=["p" + str(i) for i in range(y.shape[1])],
        )
        data_train.optimal_features = list(data_train.df_featurized.columns)
        if self.weights is None:
            self.weights = {p: 1 for p in data_train.df_targets.columns}

        if self.targets is None:
            self.targets = [[list(data_train.df_targets.columns)]]

        self.model = MODNetModel(
            self.targets,
            self.weights,
            n_feat=len(data_train.optimal_features),
            num_neurons=self.num_neurons,
            act=self.act,
            out_act=self.out_act,
        )

        self.model.fit(
            data_train,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            xscale=self.xscale,
            metrics=self.metrics,
            callbacks=self.callbacks,
            verbose=self.verbose,
            loss=self.loss,
        )

    def predict(self, X):
        """
        Predict based on a fitted MODNet regression model
        Fit a MODNet regression model.

        Parameters:
            X: array or dataframe of shape (n_samples, n_features)

        Returns:
           ndarray of shape (n_samples,) or (n_samples, n_properties)
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=["f" + str(i) for i in range(X.shape[1])])

        data_test = MODData([None for _ in range(len(X))], df_featurized=X)
        preds = self.model.predict(data_test)
        return preds.values


class MODNetClassifier(ClassifierMixin, BaseEstimator):
    """MODNet model."""

    def __init__(
        self,
        targets=None,
        weights=None,
        num_neurons=[[64], [32], [16], [16]],
        act: str = "relu",
        out_act: str = "linear",
        lr: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        xscale: Optional[str] = "minmax",
        metrics: List[str] = ["mae"],
        callbacks: List[Callable] = None,
        verbose: int = 0,
        loss: str = "mse",
    ):

        """Constructor for MODNet model.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            weights: The relative loss weights to apply for each target.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `tf.keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            act: A string defining a tf.keras activation function to pass to use
                in the `tf.keras.layers.Dense` layers.
            out_act: A string defining a tf.keras activation function to pass to use
                for the last output layer (regression only)
            lr: The learning rate.
            epochs: The maximum number of epochs to train for.
            batch_size: The batch size to use for training.
            xscale: The feature scaler to use, either `None`,
                `'minmax'` or `'standard'`.
            metrics: A list of tf.keras metrics to pass to `compile(...)`.
            callbacks: A list of tf.keras callbacks that will be provided to the model.
            verbose: integer handling verbosity.
            loss: The built-in tf.keras loss to pass to `compile(...)`.
        """
        self.targets = targets
        self.weights = weights
        self.num_neurons = num_neurons
        self.act = act
        self.out_act = out_act
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.xscale = xscale
        self.metrics = metrics
        self.callbacks = callbacks
        self.verbose = verbose
        self.loss = loss

    def fit(self, X, y):
        """
        Fit a MODNet classifier model.

        Parameters:
            X: array or dataframe of shape (n_samples, n_features)
            y: array-like of shape (n_samples,). The n classes should be encoded from 0 to n-1

        Returns:
            Fitted estimator
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=["f" + str(i) for i in range(X.shape[1])])

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        data_train = MODData(
            [None for _ in range(len(X))],
            df_featurized=X,
            targets=y,
            target_names=["p0"],
        )
        data_train.optimal_features = list(data_train.df_featurized.columns)
        data_train.num_classes = {"p0": y.max() + 1}
        if self.weights is None:
            self.weights = {"p0": 1}

        if self.targets is None:
            self.targets = [[list(data_train.df_targets.columns)]]

        self.model = MODNetModel(
            self.targets,
            self.weights,
            n_feat=len(data_train.optimal_features),
            num_classes=data_train.num_classes,
            num_neurons=self.num_neurons,
            act=self.act,
            out_act=self.out_act,
        )

        self.model.fit(
            data_train,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            xscale=self.xscale,
            metrics=self.metrics,
            callbacks=self.callbacks,
            verbose=self.verbose,
            loss=self.loss,
        )

    def predict(self, X):
        """
        Predict based on a fitted MODNet regression model
        Fit a MODNet regression model.

        Parameters:
            X: array or dataframe of shape (n_samples, n_features)

        Returns:
           ndarray of shape (n_samples,)
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=["f" + str(i) for i in range(X.shape[1])])

        data_test = MODData([None for _ in range(len(X))], df_featurized=X)
        preds = self.model.predict(data_test)
        return preds.values[:, 0]
