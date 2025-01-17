import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from modnet.sklearn import MODNetFeaturizer, RR, MODNetRegressor, MODNetClassifier
from sklearn.metrics import accuracy_score
from pymatgen.core import Composition


def test_featurize():
    comps = [Composition("Li"), Composition("Si")]
    f = MODNetFeaturizer()
    f.fit(comps)
    df = f.transform(comps)
    assert df.shape[1] > 0


def test_rr_fit():
    np.random.seed(11)
    npoints = 1000

    x = np.linspace(0, 50, npoints)
    x2 = np.random.rand(npoints) * 50
    y = x + x2
    x3 = x + np.random.rand(npoints) * 5  # x with noise
    x4 = np.random.rand(npoints) * 10
    # x and x2 are the two most important variables. x4 should be last (random noise). x3 is redundant wrt. x

    df_x = pd.DataFrame({"x": x, "x2": x2, "x3": x3, "x4": x4})
    df_y = pd.DataFrame({"y": y})

    # A.
    # test n_feat size is correct
    rr = RR(n_feat=2)
    rr.fit(df_x, df_y)
    assert rr.n_feat == 2
    assert len(rr.optimal_descriptors) == 2

    # B.
    # test when no NMIS are not provided
    rr = RR()
    rr.fit(df_x, df_y)
    assert rr.optimal_descriptors == ["x", "x2", "x3", "x4"]  # see previous explanation

    # C.
    # test when NMis are provided
    df_target_nmis = pd.DataFrame(
        [[0], [1], [0.4], [0.2]], index=["x", "x2", "x3", "x4"], columns=["y"]
    )
    df_cross_nmis = pd.DataFrame(
        [[1, 0, 0, 0], [0, 1, 0.9, 0], [0, 0.9, 1, 0], [0, 0, 0, 1]],
        index=["x", "x2", "x3", "x4"],
        columns=["x", "x2", "x3", "x4"],
    )
    # In this example x2 is the most important, then x3 and x4. x has zero relevance.
    # Moreover x2 and x3 are very redundant.
    # Therefore the order of importance should be x2,x4,x3,x.
    rr = RR()
    rr.fit(df_x, df_y, nmi_feats_target=df_target_nmis, cross_nmi_feats=df_cross_nmis)
    assert rr.optimal_descriptors == ["x2", "x4", "x3", "x"]

    # D.
    # test rr_params
    # simple constant values
    rr = RR(
        rr_parameters={
            "p": {"function": "constant", "value": 1},
            "c": {"function": "constant", "value": 0.01},
        }
    )
    rr.fit(df_x, df_y, nmi_feats_target=df_target_nmis, cross_nmi_feats=df_cross_nmis)
    assert rr.optimal_descriptors == ["x2", "x4", "x3", "x"]

    # dynamical values, for later once matthew PR is merged
    # rr = RR(rr_parameters={'p': lambda n: max(0.1, 4.5 - (n ** 0.4) * 0.4), 'c': lambda n: min(100000, 0.000001 * n ** 3))
    # rr.fit(df_x, df_y, nmi_feats_target=df_target_nmis, cross_nmi_feats=df_cross_nmis)
    # assert rr.optimal_descriptors == ['x2', 'x4', 'x3', 'x']

    # neglect redundancy
    rr = RR(
        rr_parameters={
            "p": {"function": "constant", "value": 1},
            "c": {"function": "constant", "value": 100},
        }
    )
    rr.fit(df_x, df_y, nmi_feats_target=df_target_nmis, cross_nmi_feats=df_cross_nmis)
    assert rr.optimal_descriptors == ["x2", "x3", "x4", "x"]


def test_rr_transform():
    # simple test where only two features should be kept
    df_x = pd.DataFrame(
        {"a": [1, 2, 2], "b": [5, 5, 9], "c": [0, 1, 0], "d": [114, 21, 68]},
        index=["id1", "id2", "id3"],
    )
    rr = RR()
    rr.optimal_descriptors = ["c", "a"]  # bypass fit method
    df = rr.transform(df_x)
    assert list(df.index) == ["id1", "id2", "id3"]
    assert list(df.columns) == ["c", "a"]
    assert (df.values == np.array([[0, 1], [1, 2], [0, 2]])).all()


def test_regressor():
    X = np.random.rand(100, 20)
    y = X.min(axis=1)
    regr = MODNetRegressor()
    regr.fit(X, y)
    preds = regr.predict(X)
    assert np.absolute(preds - y).max() < 1


def test_classifier():
    X = np.random.rand(100, 50)
    y = np.random.randint(0, 2, 100)
    X[:, 0] = y
    regr = MODNetClassifier()
    regr.fit(X, y)
    preds = regr.predict(X)
    assert accuracy_score(y, preds) > 0.95


def test_pipe_transform():
    pipe = make_pipeline(MODNetFeaturizer(), RR(drop_thr=0, n_feat=10))
    X = [Composition("Si"), Composition("Cu"), Composition("Al"), Composition("Ti")]
    y = pd.DataFrame({"p": [1.1, 1.6, 2.6, 0.5]})
    X_new = pipe.fit_transform(X, y)
    assert X_new.shape == (4, 10)


def test_pipe_fit():
    pipe = make_pipeline(
        MODNetFeaturizer(), RR(drop_thr=0, n_feat=10), MODNetRegressor()
    )
    X = [Composition("Si"), Composition("Cu"), Composition("Al"), Composition("Ti")]
    y = pd.DataFrame({"p": [1.1, 1.6, 2.6, 0.5]})
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (4, 1)
