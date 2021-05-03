from __future__ import annotations
import random
from typing import List
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from modnet.preprocessing import MODData
from modnet.models import MODNetModel, EnsembleMODNetModel
from modnet.utils import LOG
import multiprocessing
import tqdm


class Individual:
    
    """Class containing each of the tuned hyperparameters for the genetic algorithm.
    """
    
    def __init__(self, data:MODData):

        self.data = data

        self.xscale_list = ['minmax', 'standard']
        self.lr_list = [0.01, 0.005, 0.001]
        self.initial_batch_size_list = [8, 16, 32, 64, 128]
        self.fraction_list = [1, 0.75, 0.5, 0.25]

        self.genes = {"act": 'elu',
                      "loss": 'mae',
                      "n_neurons_first_layer": 32 * random.randint(1, 10),
                      "fraction1": random.choice(self.fraction_list),
                      "fraction2": random.choice(self.fraction_list),
                      "fraction3": random.choice(self.fraction_list),
                      "xscale": random.choice(self.xscale_list),
                      "lr": random.choice(self.lr_list),
                      "initial_batch_size": random.choice(self.initial_batch_size_list),
                      "n_feat": 0,
                      }

        if len(data.get_optimal_descriptors()) <= 100:
            b = int(len(data.get_optimal_descriptors())/2)
            self.genes["n_feat"] = random.randint(1, b) + b
        elif len(data.get_optimal_descriptors()) > 100 and len(data.get_optimal_descriptors()) < 2000:
            max = len(data.get_optimal_descriptors())
            self.genes["n_feat"] = 10*random.randint(1,int(max/10))
        else:
            max = np.sqrt(len(data.get_optimal_descriptors()))
            self.genes["n_feat"]= random.randint(1,max)**2


    def crossover(
            self,
            partner: Individual
    ) -> Individual:

        """Does the crossover of two parents and returns a 'child' which have the combined genetic information of both parents.
        Parameters:
            mother: List containing the gentic information of the first parent.
            father: List containing the gentic information of the second parent.
        """

        genes_from_mother = random.sample(range(10),
                                          k=5)  # creates indices to take randomly 5 genes from one parent, and 5 genes from the other

        child_genes = {
            list(self.genes.keys())[i]: list(self.genes.values())[i] if i in genes_from_mother else list(partner.genes.values())[i] for
            i in range(10)}
        
        child = Individual(self.data)
        child.genes = child_genes
        return child

    def mutation(
            self,
            prob_mut: int
    ) -> None:

        """Performs mutation in the genetic information in order to maintain diversity in the population.
        Paramters:
            children: List containing the genetic information of the 'children'.
        """

        if np.random.rand() > prob_mut:
            individual = Individual(self.data)
            # modification of the number of features in a [-10%, +10%] range
            self.genes['n_feat'] = np.absolute(int(
                self.genes['n_feat'] + random.randint(-int(0.1 * len(self.data.get_optimal_descriptors())),
                                                int(0.1 * len(self.data.get_optimal_descriptors())))))
            # modification of the number of neurons in the first layer of [-64, -32, 0, 32, 64]
            self.genes['n_neurons_first_layer'] = np.absolute(
                self.genes['n_neurons_first_layer'] + 32 * random.randint(-2, 2))
            if self.genes['n_neurons_first_layer'] == 0:
                self.genes['n_neurons_first_layer'] = 32
            # modification of the 1st, 2nd or 3rd fraction
            i = random.choices([1, 2, 3])
            if i == 1:
                self.genes['fraction1'] = individual.genes['fraction1']
            elif i == 2:
                self.genes['fraction2'] = individual.genes['fraction2']
            else:
                self.genes['fraction3'] = individual.genes['fraction3']
            # multiplication of the initial batch size by a factor of [1/2, 1, 2]
            self.genes['initial_batch_size'] = int(self.genes['initial_batch_size'] * 2 ** random.randint(-1, 1))
        else:
            pass
        return None

    def evaluate(
            self,
            train_data,
            val_data,
    ):

        """Returns the MODNet model given some parameters stored in ind and given the dataset to train the model on.
        Paramters:
            ind: An individual of the population, which is a list wherein the parameters are stored.
            md: MODData where the model is trained on.
        """

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=100,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        callbacks = [es]
        model = MODNetModel(targets = [[train_data.target_names]],
            weights = {n: 1 for n in train_data.target_names},
            n_feat=self.genes['n_feat'],
            num_neurons=[
                [int(self.genes['n_neurons_first_layer'])],
                [int(self.genes['n_neurons_first_layer'] * self.genes['fraction1'])],
                [int(self.genes['n_neurons_first_layer'] * self.genes['fraction1'] * self.genes['fraction2'])],
                [int(self.genes['n_neurons_first_layer'] * self.genes['fraction1'] * self.genes['fraction2'] * self.genes['fraction3'])]],
            act=self.genes['act'],
        )

        model.fit(
            train_data,
            val_data = val_data,
            loss=self.genes['loss'],
            lr=self.genes['lr'],
            epochs=1000,
            batch_size=self.genes['initial_batch_size'],
            xscale=self.genes['xscale'],
            callbacks=callbacks,
            verbose=0
        )

        self.val_loss = model.evaluate(val_data)
        self.model = model

class FitGenetic:
    """Class optimizing the model parameters using a genitic algorithm.
    """

    def __init__(
            self,
            data: MODData,
    ):

        """Initializes the MODData used in this class.
        Parameters:
            data: A 'MODData' that has been featurized and feature selected.
        """
        self.data = data

    def initialization_population(
            self,
            size_pop: int
    ) -> None:

        """Inintializes the initial population (Generation 0).
        Paramter:
            size_pop: Size of the population.
        """

        self.pop = [Individual(self.data) for _ in range(size_pop)]

    def function_fitness(
            self,
            pop: List,
            n_jobs=None,
            nested = 5,
            val_fraction=0.1,
    ) -> None:

        """Calculates the fitness of each model, which has the parameters contained in the pop argument. The function returns a list containing respectively the MAE calculated on the validation set, the model, and the parameters of that model.
        Parameters:
            pop: List containing the genetic information (i.e., the parameters) of the model.
            md_train: Input MODData.
            n_jobs: Number of jobs for multiprocessing
        """
        from modnet.matbench.benchmark import matbench_kfold_splits

        num_nested_folds = 5
        if nested:
            num_nested_folds = nested
        if num_nested_folds <= 1:
            num_nested_folds = 5

        # create tasks
        splits = matbench_kfold_splits(self.data, n_splits=num_nested_folds)
        if not nested:
            splits = [train_test_split(range(len(self.data.df_featurized)), test_size=val_fraction)]
            n_splits = 1
        else:
            n_splits = num_nested_folds
        train_val_datas = []
        for train, val in splits:
            train_val_datas.append(self.data.split((train, val)))

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
                        "fold_id": j
                    }
                ]

        val_losses = 1e20 * np.ones((len(pop), n_splits))
        models = [[None for _ in range(n_splits)] for _ in range(len(pop))]
        individuals = [None for _ in range(len(pop))]

        if n_jobs == None:
            n_jobs = 4
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=n_jobs)
        LOG.info(
            f"Multiprocessing on {n_jobs} cores. Total of {multiprocessing.cpu_count()} cores available."
        )

        for res in tqdm.tqdm(
                pool.imap_unordered(_map_evaluate_individual, tasks, chunksize=1),
                total=len(tasks)
        ):
            individual, individual_id, fold_id = res
            individual.model._restore_model()
            val_losses[individual_id, fold_id] = individual.val_loss
            individuals[individual_id] = individual
            models[individual_id][fold_id] = individual.model

        models = [EnsembleMODNetModel(modnet_models=inner_models) for inner_models in models]
        val_loss_per_individual = np.mean(val_losses, axis=1)
        res_str = "Loss per individual: "
        for ind,vl in enumerate(val_loss_per_individual):
            res_str += "ind {}: {:.3f} \t".format(ind,vl)
        LOG.info(res_str)

        pool.close()
        pool.join()

        return val_loss_per_individual, np.array(models), np.array(individuals)

    def run(
            self,
            size_pop: int = 10,
            num_generations: int = 20,
            prob_mut: int = 0.5,
    ) -> None:

        """Selects the best individual (the model with the best parameters) for the next generation. The selection is based on a minimisation of the MAE on the validation set.
        Parameters:
            md: A 'MODData' that has been featurized and feature selected.
            size_pop: Size of the population per generation.
            num_generations: Number of generations.
        """


        LOG.info('Generation number 0')
        self.initialization_population(size_pop)  # initialization of the population
        val_loss, models, individuals = self.function_fitness(self.pop)
        ranking = val_loss.argsort()
        best_model_per_gen = [None for _ in range(num_generations)]
        self.best_model = models[ranking[0]]
        best_model_per_gen[0] = self.best_model

        for j in range(1, num_generations):
            LOG.info("Generation number {}".format(j))

            # select parents
            weights = [1 / l ** 10 for l in
                     val_loss[ranking]]  # **10 in order to give relatively more importance to the best individuals
            weights = [w / sum(weights) for w in weights]
            # selection: weighted choice of the parents -> parents with a low MAE have more chance to be selected
            parents_1 = random.choices(individuals[ranking], weights=weights, k=size_pop)
            parents_2 = random.choices(individuals[ranking], weights=weights, k=size_pop)

            # crossover
            children = [parents_1[i].crossover(parents_2[i]) for i in range(size_pop)]
            for c in children:
                c.mutation(prob_mut)

            # calculates children's fitness to choose who will pass to the next generation
            val_loss_children, models_children, individuals_children = self.function_fitness(children)
            val_loss = np.concatenate([val_loss,val_loss_children])
            models = np.concatenate([models,models_children])
            individuals = np.concatenate([individuals,individuals_children])

            ranking = val_loss.argsort()

            self.best_model = models[ranking[0]]
            best_model_per_gen[j] = self.best_model

            # early stopping if we have the same best_individual for 3 generations
            if j >= 2 and best_model_per_gen[j - 2] == best_model_per_gen[j]:
                LOG.info("Early stopping: same best model for 3 consecutive generations")
                break

        return self.best_model


def _map_evaluate_individual(kwargs):
    return _evaluate_individual(**kwargs)


def _evaluate_individual(
        individual: Individual,
        train_data: MODData,
        val_data: MODData,
        individual_id: int,
        fold_id: int
):
    """Returns the MAE of a modnet model given some parameters stored in ind and given the training and validation sets sorted in fold.
    Paramters:
        individual: An individual of the population, which is a list wherein the parameters are stored.
        fold: Tuple giving the training and validation MODData.
    """
    individual.evaluate(train_data,val_data)
    individual.model._make_picklable()
    return individual, individual_id, fold_id
