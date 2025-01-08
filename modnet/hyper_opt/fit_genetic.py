from __future__ import annotations
import math
import random
from typing import List, Optional, Dict, Union, Callable
import numpy as np
import tensorflow as tf
from modnet.preprocessing import MODData
from modnet.models import (
    MODNetModel,
    EnsembleMODNetModel,
)
from modnet.utils import LOG, generate_shuffled_and_stratified_val_split
import multiprocessing
import tqdm


class Individual:
    """Class representing a set of hyperparameters for the genetic algorithm."""

    def __init__(
        self,
        max_feat: int,
        num_classes: dict,
        multi_label: bool,
        loss: Union[str, Callable] = "mae",
        targets: List = None,
        weights: Dict[str, float] = None,
        **fit_params,
    ) -> Individual:
        """
        Args:
            max_feat (int): Maximum number of features
            num_classes (dict): MODData num_classes parameter.Used for distinguishing between regression and classification.
            multi_label (bool): whether the task is a classification multi-label problem.
            loss: The built-in tf.keras loss to pass to `compile(...)`.
            targets (List): Optional (for joint learning only). A nested list of targets names that defines the hierarchy
                of the output layers.
            weights (Dict[str, float]): Optional (for joint learning only). The relative loss weights to apply for each target.
            fit_params: Any additional parameters to pass to `MODNetModel.fit(...)`,
        """

        self.act = "elu"
        self.loss = loss
        self.n_neurons_first_layer = 32 * random.randint(1, 10)
        self.max_feat = max_feat
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.targets = targets
        self.weights = weights
        self.fit_params = fit_params

        self.xscale_list = ["minmax", "standard"]
        self.impute_missing_list = [0, "mean"]
        self.xscale_before_impute = True
        self.lr_list = [0.1, 0.01, 0.005, 0.001]
        self.batch_size_list = [32, 64, 128, 256]
        self.fraction_list = [1, 1, 0.75, 0.5, 0.25]
        # add 1 to balance the chance of having an architecture with the same num_neurons on each layer

        if fit_params:
            self.__dict__.update(fit_params)

        self.genes = {
            "act": self.act,
            "loss": self.loss,
            "n_neurons_first_layer": self.n_neurons_first_layer,
            "fraction1": random.choice(self.fraction_list),
            "fraction2": random.choice(self.fraction_list),
            "fraction3": random.choice(self.fraction_list),
            "xscale": random.choice(self.xscale_list),
            "impute_missing": random.choice(self.impute_missing_list),
            "xscale_before_impute": self.xscale_before_impute,
            "lr": random.choice(self.lr_list),
            "batch_size": random.choice(self.batch_size_list),
            "n_feat": 0,
        }

        if max_feat <= 100:
            b = int(max_feat / 2)
            self.genes["n_feat"] = random.randint(1, b) + b
        elif max_feat > 100 and max_feat < 2000:
            max = max_feat
            self.genes["n_feat"] = 10 * random.randint(1, int(max / 10))
        else:
            max = np.sqrt(max_feat)
            self.genes["n_feat"] = random.randint(1, max) ** 2

    def crossover(self, partner: Individual) -> Individual:
        """Does the crossover of two parents and returns a 'child' which has a mix of the parents hyperparams.

        Args:
            partner (Individual): Partner individual.
        Returns:
            Individual: Child.
        """
        # creates indices to take randomly half the genes from one parent, and half the genes from the other
        mother_genes = random.sample(self.genes.keys(), k=len(self.genes) // 2)

        child_genes = {
            gene: self.genes[gene] if gene in mother_genes else partner.genes[gene]
            for gene in self.genes
        }

        child = Individual(
            max_feat=self.max_feat,
            num_classes=self.num_classes,
            multi_label=self.multi_label,
            loss=self.genes["loss"],
            targets=self.targets,
            weights=self.weights,
        )
        child.genes = child_genes
        return child

    def mutation(self, prob_mut: float) -> None:
        """Performs mutation in the hyper parameters in order to maintain diversity in the population.

        Args:
            prob_mut (float): Probability [0,1] of mutation.

        Returns: None (inplace operator).
        """

        if np.random.rand() < prob_mut:
            individual = Individual(
                max_feat=self.max_feat,
                num_classes=self.num_classes,
                multi_label=self.multi_label,
                loss=self.genes["loss"],
                targets=self.targets,
                weights=self.weights,
            )
            # modification of the number of features in a [-10%, +10%] range
            self.genes["n_feat"] = np.absolute(
                int(
                    self.genes["n_feat"]
                    + random.randint(
                        -int(0.1 * self.max_feat),
                        int(0.1 * self.max_feat),
                    )
                )
            )
            if self.genes["n_feat"] <= 0:
                self.genes["n_feat"] = 1
            elif self.genes["n_feat"] > self.max_feat:
                self.genes["n_feat"] = self.max_feat
            # modification of the number of neurons in the first layer of [-64, -32, 0, 32, 64]
            self.genes["n_neurons_first_layer"] = np.absolute(
                self.genes["n_neurons_first_layer"] + 32 * random.randint(-2, 2)
            )
            if self.genes["n_neurons_first_layer"] == 0:
                self.genes["n_neurons_first_layer"] = 32
            # modification of the 1st, 2nd or 3rd fraction
            i = random.choices([1, 2, 3])
            if i == 1:
                self.genes["fraction1"] = individual.genes["fraction1"]
            elif i == 2:
                self.genes["fraction2"] = individual.genes["fraction2"]
            else:
                self.genes["fraction3"] = individual.genes["fraction3"]
            # multiplication of the initial batch size by a factor of [1/2, 1, 2]
            self.genes["batch_size"] = int(
                self.genes["batch_size"] * 2 ** random.randint(-1, 1)
            )
        else:
            pass
        return None

    def evaluate(
        self,
        train_data: MODData,
        val_data: MODData,
        fast: bool = False,
    ):
        """Internally evaluates the validation loss by setting self.val_loss

        Args:
            train_data (MODData): Training MODData
            val_data (MODData): Validation MODData
            fast (bool, optional): Limited epoch for testing or debugging only. Defaults to False.
        """

        es = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.001,
            patience=30,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        )
        callbacks = [es]
        model = MODNetModel(
            targets=self.targets,
            weights=self.weights,
            n_feat=self.genes["n_feat"],
            num_neurons=[
                [int(self.genes["n_neurons_first_layer"])],
                [int(self.genes["n_neurons_first_layer"] * self.genes["fraction1"])],
                [
                    int(
                        self.genes["n_neurons_first_layer"]
                        * self.genes["fraction1"]
                        * self.genes["fraction2"]
                    )
                ],
                [
                    int(
                        self.genes["n_neurons_first_layer"]
                        * self.genes["fraction1"]
                        * self.genes["fraction2"]
                        * self.genes["fraction3"]
                    )
                ],
            ],
            act=self.genes["act"],
            num_classes=self.num_classes,
            multi_label=self.multi_label,
        )
        if "custom_data" in train_data.df_targets.columns:
            custom_data = np.array(list(train_data.df_targets["custom_data"].values))
        else:
            custom_data = None
        model.fit(
            train_data,
            custom_data=custom_data,
            val_data=val_data,
            loss=self.genes["loss"],
            lr=self.genes["lr"],
            epochs=800 if not fast else 1,
            batch_size=self.genes["batch_size"],
            xscale=self.genes["xscale"],
            impute_missing=self.genes["impute_missing"],
            xscale_before_impute=self.genes["xscale_before_impute"],
            callbacks=callbacks,
            verbose=0,
            **self.fit_params,
        )

        self.val_loss = model.evaluate(
            val_data,
            loss=self.genes["loss"],
        )
        self.model = model

    def refit_model(self, data: MODData, n_models=10, n_jobs=1, fast: bool = False):
        """Refit inner model on specified data.
        Args:
            data (MODData): Training data
            fast (bool, optional): Limited epoch for testing or debugging only. Defaults to False.
        """

        es = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.001,
            patience=30,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        )
        callbacks = [es]
        model = EnsembleMODNetModel(
            targets=self.targets,
            weights=self.weights,
            n_models=n_models,
            n_feat=self.genes["n_feat"],
            num_neurons=[
                [int(self.genes["n_neurons_first_layer"])],
                [int(self.genes["n_neurons_first_layer"] * self.genes["fraction1"])],
                [
                    int(
                        self.genes["n_neurons_first_layer"]
                        * self.genes["fraction1"]
                        * self.genes["fraction2"]
                    )
                ],
                [
                    int(
                        self.genes["n_neurons_first_layer"]
                        * self.genes["fraction1"]
                        * self.genes["fraction2"]
                        * self.genes["fraction3"]
                    )
                ],
            ],
            act=self.genes["act"],
            num_classes=self.num_classes,
            multi_label=self.multi_label,
        )
        if "custom_data" in data.df_targets.columns:
            custom_data = np.array(list(data.df_targets["custom_data"].values))
        else:
            custom_data = None
        model.fit(
            data,
            custom_data=custom_data,
            n_jobs=n_jobs,
            val_fraction=0,
            loss=self.genes["loss"],
            lr=self.genes["lr"],
            epochs=800 if not fast else 1,
            batch_size=self.genes["batch_size"],
            xscale=self.genes["xscale"],
            impute_missing=self.genes["impute_missing"],
            xscale_before_impute=self.genes["xscale_before_impute"],
            callbacks=callbacks,
            verbose=0,
            **self.fit_params,
        )

        self.model = model
        return self.model


class FitGenetic:
    """Class optimizing the model parameters using a genetic algorithm."""

    def __init__(
        self,
        data: MODData,
        custom_data: Optional[np.ndarray] = None,
        targets: List = None,
        weights: Dict[str, float] = None,
        sample_threshold: int = 5000,
        ignore_names: Optional[List] = [],
    ):
        """Genetic algorithm hyperparameter optimization for MODNet.

        Args:
            data (MODData): Training MODData
            custom_data (np.ndarray): Optional array of shape (n_sampels, n_custom_props) that will be appended to the targets (columns wise).
                This can be useful for defining custom loss functions.
            targets (List): Optional (for joint learning only). A nested list of targets names that defines the hierarchy
                of the output layers.
            weights (Dict[str, float]): Optional (for joint learning only). The relative loss weights to apply for each target.
            sample_threshold (int, optional): If the dataset size exceeds this threshold, individuals are
                trained on sampled subsets of this size. Defaults to 5000.
            ignore_names (List): Optional list of property names to ignore during feature selection.
                Feature selection will be performed w.r.t. all properties except the ones in ignore_names.

        """
        for n in ignore_names:
            if n not in data.names:
                raise RuntimeError(
                    f"Names provided in ignore_names should be part of {data.names}. {n} was not found."
                )

        self.data = data
        if custom_data is not None:
            self.data.df_targets["custom_data"] = [list(x) for x in custom_data]
        subset_ids = np.random.permutation(len(data.df_featurized))[:sample_threshold]
        self.train_data, _ = data.split((subset_ids, []))
        self.num_classes = data.num_classes
        t_names = list(set(data.names).difference(set(ignore_names)))
        if targets is None:
            targets = [[t_names]]
        if weights is None:
            weights = {n: 1 for n in t_names}
        self.targets = targets
        self.weights = weights

        LOG.info("Targets:")
        for i, (k, v) in enumerate(self.num_classes.items()):
            if v >= 2:
                type = "classification"
            else:
                type = "regression"
            LOG.info(f"{i+1}){k}: {type}")

    def _init_run(self, n_jobs: Optional[int] = None):
        if n_jobs is None:
            n_jobs = 4
        ctx = multiprocessing.get_context("spawn")
        self.pool = ctx.Pool(processes=n_jobs)
        LOG.info(
            "Multiprocessing on {} cores. Total of {} cores available.".format(
                n_jobs, multiprocessing.cpu_count()
            )
        )

    def _end_run(self):
        self.pool.close()
        self.pool.join()

    def initialization_population(
        self,
        size_pop: int,
        multi_label: bool,
        loss: Union[str, Callable] = "mae",
        **fit_params,
    ) -> None:
        """Initializes the initial population (Generation 0).

        Args:
            size_pop (int): Size of population.
            multi_label: Whether the problem (if classification) is multi-label.
                In this case the softmax output-activation is replaced by a sigmoid.
            loss: The built-in tf.keras loss to pass to `compile(...)`.
            fit_params: Any additional parameters to pass to `MODNetModel.fit(...)`,
        """

        self.pop = [
            Individual(
                max_feat=len(self.train_data.get_optimal_descriptors()),
                num_classes=self.train_data.num_classes,
                multi_label=multi_label,
                loss=loss,
                targets=self.targets,
                weights=self.weights,
                **fit_params,
            )
            for _ in range(size_pop)
        ]

    def function_fitness(
        self,
        pop: List[Individual],
        n_jobs: int,
        nested=5,
        val_fraction=0.1,
        multi_label: Optional[bool] = False,
        fast=False,
    ) -> None:
        """Calculates the fitness of each model, which has the parameters contained in the pop argument.
        The function returns a list containing respectively the MAE calculated on the validation set, the model, and the parameters of that model.

        Args:
            pop (List[Individual]): List of individuals
            n_jobs (int): number of jobs to parallelize on.
            nested (int, optional): CV fold size. Defaults to 5. Use <=0 for hold-out validation.
            val_fraction (float, optional): Validation fraction if no CV is used. Defaults to 0.1.
            multi_label: Whether the problem (if classification) is multi-label.
                In this case the softmax output-activation is replaced by a sigmoid.
            fast (bool, optional): Limited epochs for testing and debugging only. Defaults to False.

        Returns:
            val_losses, models, individuals
        """

        from modnet.matbench.benchmark import matbench_kfold_splits
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
            "2"  # many models will be fitted => reduce output
        )

        num_nested_folds = 5
        if nested:
            num_nested_folds = nested
        if num_nested_folds <= 1:
            num_nested_folds = 5

        # create tasks
        splits = matbench_kfold_splits(
            self.train_data,
            n_splits=num_nested_folds,
            classification=max(self.num_classes.values()) >= 2,
        )
        if not nested:
            splits = [
                generate_shuffled_and_stratified_val_split(
                    y=self.train_data.df_targets.values,
                    val_fraction=val_fraction,
                    classification=max(self.num_classes.values()) >= 2,
                )
            ]
            n_splits = 1
        else:
            n_splits = num_nested_folds
        train_val_datas = []
        for train, val in splits:
            train_val_datas.append(self.train_data.split((train, val)))

        tasks = []
        for i, individual in enumerate(pop):
            for j in range(n_splits):
                train_data, val_data = train_val_datas[j]
                tasks += [
                    {
                        "individual": individual,
                        "train_data": train_data,
                        "val_data": val_data,
                        "individual_id": i,
                        "fold_id": j,
                        "fast": fast,
                    }
                ]

        val_losses = 1e20 * np.ones((len(pop), n_splits))
        models = [[None for _ in range(n_splits)] for _ in range(len(pop))]
        individuals = [None for _ in range(len(pop))]

        for res in tqdm.tqdm(
            self.pool.imap_unordered(_map_evaluate_individual, tasks, chunksize=1),
            total=len(tasks),
        ):
            individual, individual_id, fold_id = res
            individual.model._restore_model()
            val_losses[individual_id, fold_id] = individual.val_loss
            individuals[individual_id] = individual
            models[individual_id][fold_id] = individual.model

        models = [EnsembleMODNetModel(models=inner_models) for inner_models in models]
        val_loss_per_individual = np.mean(val_losses, axis=1)
        res_str = "Loss per individual: "
        for ind, vl in enumerate(val_loss_per_individual):
            res_str += "ind {}: {:.3f} \t".format(ind, vl)
        LOG.info(res_str)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # reset

        return val_loss_per_individual, np.array(models), np.array(individuals)

    def run(
        self,
        size_pop: int = 20,
        num_generations: int = 10,
        prob_mut: Optional[int] = None,
        nested: Optional[int] = 5,
        multi_label: bool = False,
        loss: Union[str, Callable] = "mae",
        n_jobs: Optional[int] = None,
        early_stopping: Optional[int] = 4,
        refit: Optional[int] = 5,
        fast=False,
        **fit_params,
    ) -> EnsembleMODNetModel:
        """Run the GA and return best model.

        Args:
            size_pop (int, optional): Size of the population per generation.. Defaults to 20.
            num_generations (int, optional): Size of the population per generation. Defaults to 10.
            prob_mut (Optional[int], optional): Probability of mutation. Defaults to None.
            nested (Optional[int], optional): CV fold size. Use 0 for hold-out validation (fraction of 0.1). Negative values and a value of 1 are equivalent to the default (5).
            multi_label: Whether the problem (if classification) is multi-label.
                In this case the softmax output-activation is replaced by a sigmoid.
            loss: The built-in tf.keras loss to pass to `compile(...)`.
            n_jobs (Optional[int], optional): Number of jobs to parallelize on. Defaults to None.
            early_stopping (Optional[int], optional): Number of successive generations without improvement before stopping. Defaults to 4.
            refit (Optional[int], optional): Whether to refit (>0) the best hyperparameters on the whole dataset or use the best Individual instead (=0).
                The amount corresponds to the number of models used in the ensemble. Defaults to 5.
            fast (bool, optional): Use only for debugging and testing. A fast GA run with small number of epochs, generations, individuals and folds.
                Overrides the size_pop, num_generation and nested arguments.. Defaults to False.
            fit_params: Any additional parameters to pass to `MODNetModel.fit(...)`,

        Returns:
            EnsembleMODNetModel: Fitted model with best hyperparameters
        """
        self._init_run(n_jobs=n_jobs)

        if fast:
            size_pop, num_generations, nested = 2, 2, 2

        LOG.info("Generation number 0")
        self.initialization_population(
            size_pop,
            multi_label=multi_label,
            loss=loss,
            **fit_params,
        )  # initialization of the population
        val_loss, models, individuals = self.function_fitness(
            pop=self.pop,
            nested=nested,
            n_jobs=n_jobs,
            multi_label=multi_label,
            fast=fast,
        )
        ranking = val_loss.argsort()
        best_model_per_gen = [None for _ in range(num_generations)]
        self.best_model = models[ranking[0]]
        best_model_per_gen[0] = self.best_model

        for j in range(1, num_generations):
            LOG.info("Generation number {}".format(j))

            # select parents
            weights = [
                1 / lw**5 for lw in val_loss[ranking]
            ]  # **5 in order to give relatively more importance to the best individuals
            weights = [1e-5 if math.isnan(weight) else weight for weight in weights]
            weights = [w / sum(weights) for w in weights]
            # selection: weighted choice of the parents -> parents with a low MAE have more chance to be selected
            parents_1 = random.choices(
                individuals[ranking], weights=weights, k=size_pop
            )
            parents_2 = random.choices(
                individuals[ranking], weights=weights, k=size_pop
            )

            # crossover
            children = [parents_1[i].crossover(parents_2[i]) for i in range(size_pop)]
            if prob_mut is None:
                prob_mut = 1 / size_pop
            for c in children:
                c.mutation(prob_mut)

            # calculates children's fitness to choose who will pass to the next generation
            (
                val_loss_children,
                models_children,
                individuals_children,
            ) = self.function_fitness(
                pop=children,
                nested=nested,
                n_jobs=n_jobs,
                multi_label=multi_label,
                fast=fast,
            )
            val_loss = np.concatenate([val_loss, val_loss_children])
            models = np.concatenate([models, models_children])
            individuals = np.concatenate([individuals, individuals_children])

            ranking = val_loss.argsort()

            self.best_model = models[ranking[0]]
            self.best_individual = individuals[ranking[0]]
            best_model_per_gen[j] = self.best_model

            # early stopping if we have the same best_individual for early_stopping generations
            if (
                j >= early_stopping - 1
                and best_model_per_gen[j - (early_stopping - 1)]
                == best_model_per_gen[j]
            ):
                LOG.info(
                    "Early stopping: same best model for {} consecutive generations".format(
                        early_stopping
                    )
                )
                LOG.info("Early stopping at generation number {}".format(j))
                break

        self._end_run()
        if refit:
            LOG.info("Refit...")
            """
            self.best_individual.model = None
            ensemble = []
            tasks = []
            for i in range(refit):
                tasks += [
                    {
                        "individual": copy.deepcopy(self.best_individual),
                        "data": self.data,
                        "fast": fast,
                    }
                ]

            for res in tqdm.tqdm(
                self.pool.imap_unordered(_map_refit_individual, tasks, chunksize=1),
                total=len(tasks),
            ):
                model = res
                model._restore_model()
                ensemble.append(model)

            self.best_model = EnsembleMODNetModel(models=ensemble)
            """
            self.best_model = self.best_individual.refit_model(
                self.data, n_models=refit, n_jobs=n_jobs or 1, fast=fast
            )

        else:
            ensemble = []
            for m in models[ranking[:10]]:
                ensemble += m.models
            self.best_model = EnsembleMODNetModel(models=ensemble)

        self.results = self.best_individual.genes

        return self.best_model


def _map_evaluate_individual(kwargs):
    return _evaluate_individual(**kwargs)


def _evaluate_individual(
    individual: Individual,
    train_data: MODData,
    val_data: MODData,
    individual_id: int,
    fold_id: int,
    fast: bool = False,
):
    """Evaluate individual

    Args:
        individual (Individual): Individual to be evaluated
        train_data (MODData): Training MODData
        val_data (MODData): Validation MODData
        individual_id (int): Individual ID
        fold_id (int): Fold ID
        fast (bool, optional): Limited number of epochs for debugging and testing purposes only. Defaults to False.

    Returns:
        individual, individual_id, fold_id
        individual.val_loss contains the corresponding validation score.
    """
    individual.evaluate(train_data, val_data, fast=fast)
    individual.model._make_picklable()
    return individual, individual_id, fold_id


def _map_refit_individual(kwargs):
    return _refit_individual(**kwargs)


def _refit_individual(
    individual: Individual,
    data: MODData,
    fast: bool = False,
):
    """Evaluate individual

    Args:
        individual (Individual): Individual to be evaluated
        train_data (MODData): Training MODData
        val_data (MODData): Validation MODData
        individual_id (int): Individual ID
        fold_id (int): Fold ID
        fast (bool, optional): Limited number of epochs for debugging and testing purposes only. Defaults to False.

    Returns:
        individual, individual_id, fold_id
        individual.val_loss contains the corresponding validation score.
    """
    model = individual.refit_model(data, fast=fast)
    model._make_picklable()
    return model
