#!/usr/bin/env python


def test_ga(small_moddata, tf_session):
    """Tests the modnet.hyper_opt.FitGenetic algorithm."""
    from modnet.hyper_opt import FitGenetic

    data = small_moddata
    # set regression problem
    data.num_classes = {"eform": 0}
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    ga = FitGenetic(data)
    model = ga.run(
        size_pop=2,
        num_generations=2,
        prob_mut=0.5,
        nested=2,
        n_jobs=2,
        early_stopping=2,
        refit=1,
        fast=True,
    )

    from modnet.models import EnsembleMODNetModel

    assert type(model) is EnsembleMODNetModel
    assert len(model.models) == 1


def test_crossover():
    from modnet.hyper_opt.fit_genetic import Individual

    indiv1 = Individual(max_feat=50, num_classes={}, multi_label=False)
    indiv2 = Individual(max_feat=50, num_classes={}, multi_label=False)
    indiv1.crossover(indiv2)
