"""This submodule defines the `BayesianMODNetModel`, an extension to the vanilla
model that incorporates probabilistic `DenseVariational` layers from TensorFlow
Probability.

"""

from typing import Optional, Dict, List, Tuple
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from modnet.models.vanilla import MODNetModel
from modnet.preprocessing import MODData

__all__ = ("BayesianMODNetModel",)


class BayesianMODNetModel(MODNetModel):
    """Container class for the underlying Probabilistic Bayesian Neural Network, that handles
    setting up the architecture, activations, training and learning curve. Only epistemic uncertainty is taken into account.

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
        targets: List,
        weights: Dict[str, float],
        num_neurons=([64], [32], [16], [16]),
        num_classes: Optional[Dict[str, int]] = None,
        n_feat: Optional[int] = 64,
        act: str = "relu",
        out_act: str = "linear",
        bayesian_layers=None,
        prior=None,
        posterior=None,
        kl_weight=None,
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
                for the last output layer
            bayesian_layers: Same shape as num_neurons, with True for a Bayesian DenseVariational layer,
                False for a normal Dense layer. Default is None and will only set last layer as Bayesian.
            prior: Prior to use for the DenseVariational layers, default is independent normal with learnable mean.
            posterior: Posterior to use for the DenseVariational layers, default is indepent normal with learnable mean and variance.
            kl_weight: Amount by which to scale the KL divergence loss between prior and posterior.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                 with n=0 for regression and n>=2 for classification with n the number of classes.
            n_feat: The number of features to use as model inputs.

        """

        return super().__init__(
            targets=targets,
            weights=weights,
            num_neurons=num_neurons,
            num_classes=num_classes,
            n_feat=n_feat,
            act=act,
            out_act=out_act,
            bayesian_layers=bayesian_layers,
            prior=prior,
            posterior=posterior,
            kl_weight=kl_weight,
        )

    def build_model(
        self,
        targets: List,
        n_feat: int,
        num_neurons: Tuple[List[int], List[int], List[int], List[int]],
        num_classes: Optional[Dict[str, int]] = None,
        act: str = "relu",
        out_act: str = "relu",
        bayesian_layers=None,
        prior=None,
        posterior=None,
        kl_weight=None,
    ) -> tf.keras.Model:
        """Builds the Bayesian Neural Network and sets the `self.model` attribute.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            n_feat: The number of features to use as model inputs.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                with n=0 for regression and n>=2 for classification with n the number of classes.
            act: A string defining a Keras activation function to pass to use
                in the `keras.layers.Dense` layers.
            out_act: A string defining a tf.keras activation function to pass to use
                for the last output layer

        """

        num_layers = [len(x) for x in num_neurons]

        # define probabilistic layers
        tfd = tfp.distributions

        if bayesian_layers is None:
            bayesian_layers = [[False] * nl for nl in num_layers]

        if posterior is None:

            def posterior(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                c = np.log(np.expm1(1.0))
                return tf.keras.Sequential(
                    [
                        tfp.layers.VariableLayer(2 * n, dtype=dtype),
                        tfp.layers.DistributionLambda(
                            lambda t: tfd.Independent(
                                tfd.Normal(
                                    loc=t[..., :n],
                                    scale=1e-5 + tf.nn.softplus(c + t[..., n:]),
                                ),
                                reinterpreted_batch_ndims=1,
                            )
                        ),
                    ]
                )

        if prior is None:

            def prior(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                return tf.keras.Sequential(
                    [
                        tfp.layers.VariableLayer(n, dtype=dtype),
                        tfp.layers.DistributionLambda(
                            lambda t: tfd.Independent(
                                tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1
                            )
                        ),
                    ]
                )

        bayesian_layer = partial(
            tfp.layers.DenseVariational,
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1 / 3619,
            activation=act,
        )
        dense_layer = partial(tf.keras.layers.Dense, activation=act)

        # Build first common block
        f_input = tf.keras.layers.Input(shape=(n_feat,))
        previous_layer = f_input
        for i in range(num_layers[0]):
            if bayesian_layers[0][i]:
                previous_layer = bayesian_layer(num_neurons[0][i])(previous_layer)
            else:
                previous_layer = dense_layer(num_neurons[0][i])(previous_layer)
            if self._multi_target:
                previous_layer = tf.keras.layers.BatchNormalization()(previous_layer)
        common_out = previous_layer

        # Build intermediate representations
        intermediate_models_out = []
        for _ in range(len(targets)):
            previous_layer = common_out
            for j in range(num_layers[1]):
                if bayesian_layers[1][j]:
                    previous_layer = bayesian_layer(num_neurons[1][j])(previous_layer)
                else:
                    previous_layer = dense_layer(num_neurons[1][j])(previous_layer)
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
                    if bayesian_layers[2][k]:
                        previous_layer = bayesian_layer(num_neurons[2][k])(
                            previous_layer
                        )
                    else:
                        previous_layer = dense_layer(num_neurons[2][k])(previous_layer)
                    if self._multi_target:
                        previous_layer = tf.keras.layers.BatchNormalization()(
                            previous_layer
                        )
                clayer = previous_layer
                for pi in range(len(group[prop_idx])):
                    previous_layer = clayer
                    for li in range(num_layers[3]):
                        if bayesian_layers[3][li]:
                            previous_layer = bayesian_layer(num_neurons[3][li])(
                                previous_layer
                            )
                        else:
                            previous_layer = dense_layer(num_neurons[3][li])(
                                previous_layer
                            )
                    n = num_classes[group[prop_idx][pi]]
                    if n >= 2:
                        out = tfp.layers.DenseVariational(
                            n,
                            make_posterior_fn=posterior,
                            make_prior_fn=prior,
                            kl_weight=kl_weight,
                            activation="softmax",
                            name=group[prop_idx][pi],
                        )(previous_layer)
                    else:
                        out = tfp.layers.DenseVariational(
                            1,
                            make_posterior_fn=posterior,
                            make_prior_fn=prior,
                            kl_weight=kl_weight,
                            activation=out_act,
                            name=group[prop_idx][pi],
                        )(previous_layer)
                    final_out.append(out)

        return tf.keras.models.Model(inputs=f_input, outputs=final_out)

    def predict(
        self, test_data: MODData, return_prob=False, return_unc=False
    ) -> pd.DataFrame:
        """Predict the target values for the passed MODData.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.
            return_prob: For a classification tasks only: whether to return the probability of each
                class OR only return the most probable class.
            return_unc: whether to return the standard deviation as a second dataframe

        Returns:
            A `pandas.DataFrame` containing the predicted values of the targets.
            If return_unc=True, two `pandas.DataFrame` : (predictions,std) containing the predicted values of the targets and
             the standard deviations of the epistemic uncertainty.


        """
        # prevents Nan predictions if some features are inf
        x = (
            test_data.get_featurized_df()
            .replace([np.inf, -np.inf, np.nan], 0)[
                self.optimal_descriptors[: self.n_feat]
            ]
            .values
        )

        # Scale the input features:
        x = np.nan_to_num(x)
        if self._scaler is not None:
            x = self._scaler.transform(x)
            x = np.nan_to_num(x)

        all_predictions = []

        for i in range(1000):
            p = self.model.predict(x)
            if len(self.targets_flatten) == 1:
                p = np.array([p])
            all_predictions.append(p)

        p_dic = {}
        unc_dic = {}
        for i, name in enumerate(self.targets_flatten):
            if self.num_classes[name] >= 2:
                if return_prob:
                    preds = np.array([pred[i] for pred in all_predictions])
                    probs = preds / (preds.sum(axis=-1)).reshape((-1, 1))
                    mean_prob = probs.mean()
                    std_prob = probs.std()
                    for j in range(mean_prob.shape[-1]):
                        p_dic["{}_prob_{}".format(name, j)] = mean_prob[:, j]
                        unc_dic["{}_prob_{}".format(name, j)] = std_prob[:, j]
                else:
                    p_dic[name] = np.argmax(
                        np.array([pred[i] for pred in all_predictions]).mean(axis=0),
                        axis=1,
                    )
                    unc_dic[name] = np.max(
                        np.array([pred[i] for pred in all_predictions]).mean(axis=0),
                        axis=1,
                    )
            else:
                mean_p = np.array([pred[i] for pred in all_predictions]).mean(axis=0)
                std_p = np.array([pred[i] for pred in all_predictions]).std(axis=0)
                p_dic[name] = mean_p[:, 0]
                unc_dic[name] = std_p[:, 0]

        predictions = pd.DataFrame(p_dic)
        unc = pd.DataFrame(unc_dic)

        predictions.index = test_data.structure_ids
        unc.index = test_data.structure_ids

        if return_unc:
            return predictions, unc
        else:
            return predictions
