# coding: utf-8
# Distributed under the terms of the MIT License.

""" This module defines the :class:`MODData` class, featurizer functions
and functions to compute normalized mutual information (NMI) and relevance redundancy
(RR) between descriptors.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union, Optional, Callable, Hashable, Iterable, Tuple
from functools import partial

from pymatgen import Structure, Composition

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import tqdm
from multiprocessing import Pool

from modnet.featurizers import MODFeaturizer
from modnet import __version__
from modnet.utils import LOG

DATABASE = pd.DataFrame([])


class CompositionContainer:
    """A simple compatbility wrapper class for structure-less pymatgen `Structure`s."""

    def __init__(self, composition):
        self.composition = composition


EPS = 1e-16


def compute_mi(
    x: np.ndarray = None,
    y: np.ndarray = None,
    x_name: str = None,
    y_name: str = None,
    random_state=None,
    n_neighbors=3,
):

    mi = mutual_info_regression(
        x.reshape(-1, 1),
        y,
        random_state=random_state,
        n_neighbors=n_neighbors,
    )[0]

    return mi, x_name, y_name


def map_mi(kwargs):
    return compute_mi(**kwargs)


def nmi_target(
    df_feat: pd.DataFrame,
    df_target: pd.DataFrame,
    task_type: str = "regression",
    drop_constant_features: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Computes the Normalized Mutual Information (NMI) between a list of
    input features and a target variable.

    Args:
        df_feat (pandas.DataFrame): Dataframe containing the input features for
            which the NMI with the target variable is to be computed.
        df_target (pandas.DataFrame): Dataframe containing the target variable.
            This DataFrame should contain only one column and have the same
            size as `df_feat`.
        task_type (integer): 0 for regression, 1 for classification
        drop_constant_features (bool): If True, the features that are constant
            across the entire data set will be dropped.
        **kwargs: Keyword arguments to be passed down to the
            :py:func:`mutual_info_regression` function from scikit-learn. This
            can be useful e.g. for testing purposes.

    Returns:
        pandas.DataFrame: Dataframe containing the NMI between each of
            the input features and the target variable.

    """
    # Initial checks
    if df_target.shape[1] != 1:
        raise ValueError("The target DataFrame should have exactly one column.")

    if len(df_feat) != len(df_target):
        raise ValueError(
            "The input features DataFrame and the target variable DataFrame "
            "should contain the same number of data points."
        )

    # Drop features which have the same value for the entire data set
    if drop_constant_features:
        frange = df_feat.max(axis=0) - df_feat.min(axis=0)
        to_drop = frange[frange == 0].index
        df_feat = df_feat.drop(to_drop, axis=1)

    # preprocess the input matrix
    if (
        df_feat.isna().any().any()
    ):  # only preprocess if nans are present to preserve past behaviour
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        x = df_feat.values
        x = scaler.fit_transform(x)
        x = np.nan_to_num(x, nan=-1)
        df_feat = pd.DataFrame(x, index=df_feat.index, columns=df_feat.columns)

    # Take right MI fun depending on regression / classification
    if task_type == "regression":
        _mifun = mutual_info_regression
        _self_mifun = mutual_info_regression
    elif task_type == "classification":
        _mifun = mutual_info_classif
        _self_mifun = partial(mutual_info_classif, discrete_features=True)

    # Prepare the output DataFrame and compute the mutual information
    target_name = df_target.columns[0]
    mutual_info = pd.DataFrame([], columns=[target_name], index=df_feat.columns)

    mutual_info.loc[:, target_name] = _mifun(df_feat, df_target[target_name], **kwargs)

    # Compute the "self" mutual information (i.e. information entropy) of the target variable and of the input features
    target_mi = _self_mifun(
        df_target[target_name].values.reshape(-1, 1), df_target[target_name], **kwargs
    )[0]
    diag = {}
    for x in df_feat.columns:
        diag[x] = (
            mutual_info_regression(
                df_feat[x].values.reshape(-1, 1), df_feat[x], **kwargs
            )
        )[0]

    # Normalize the mutual information
    for x in mutual_info.index:
        mutual_info.loc[x, target_name] = mutual_info.loc[x, target_name] / (
            (target_mi + diag[x]) / 2
        )

    mutual_info.fillna(0, inplace=True)  # if na => no relation => set to zero
    return mutual_info


def get_cross_nmi(
    df_feat: pd.DataFrame,
    drop_thr: float = 0.2,
    return_entropy=False,
    n_jobs: int = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Computes the Normalized Mutual Information (NMI) between input features.

    Args:
        df_feat (pandas.DataFrame): Dataframe containing the input features for
            which the NMI with the target variable is to be computed.
        drop_thr: Features having an information entropy (or self mutual information) threshold below this value will be dropped.
        return_entropy: If set to True, the information entropy of each feature is also returned
        **kwargs: Keyword arguments to be passed down to the
            :py:func:`mutual_info_regression` function from scikit-learn. This
            can be useful e.g. for testing purposes.

    Returns:
        mutual_info: pandas.DataFrame containing the Normalized Mutual Information between features.
        if return_entropy=True : (mutual_info, diag): With diag a dictionary with all features as keys and information entropy as values.
    """

    if kwargs.get("random_state"):
        seed = kwargs.pop("random_state")
    else:
        seed = np.random.RandomState()

    if kwargs.get("n_neighbors"):
        n_neighbors = kwargs.pop("n_neighbors")
    else:
        n_neighbors = 3

    # preprocess the input matrix
    if (
        df_feat.isna().any().any()
    ):  # only preprocess if nans are present to preserve past behaviour
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        x = df_feat.values
        x = scaler.fit_transform(x)
        x = np.nan_to_num(x, nan=-1)
        df_feat = pd.DataFrame(x, index=df_feat.index, columns=df_feat.columns)

    # Prepare the output DataFrame and compute the mutual information
    mutual_info = pd.DataFrame([], columns=df_feat.columns, index=df_feat.columns)

    # create pool of workers
    if n_jobs is None:
        n_jobs = 1
    pool = Pool(processes=n_jobs)

    LOG.info(f"Multiprocessing on {n_jobs} workers.")

    # Compute the "self" mutual information (i.e. information entropy) of the features
    LOG.info('Computing "self" MI (i.e. information entropy) of features')
    diag = {}
    tasks = []
    for x_feat in df_feat.columns:
        tasks += [
            {
                "x": df_feat[x_feat].values,
                "y": df_feat[x_feat].values,
                "x_name": x_feat,
                "y_name": x_feat,
                "random_state": seed,
                "n_neighbors": n_neighbors,
            }
        ]

    for res in tqdm.tqdm(
        pool.imap_unordered(map_mi, tasks, chunksize=100), total=len(tasks)
    ):
        feat_name = res[1]
        diag[feat_name] = res[0]
        if (
            diag[feat_name] < drop_thr
            or abs(df_feat[feat_name].max() - df_feat[feat_name].min()) < EPS
        ):
            mutual_info.drop(feat_name, axis=0, inplace=True)
            mutual_info.drop(feat_name, axis=1, inplace=True)
        else:
            mutual_info.loc[feat_name, feat_name] = 1.0

    tasks = []
    LOG.info("Computing cross NMI between all features...")
    for idx, x_feat in enumerate(mutual_info.columns):
        for y_feat in mutual_info.columns[idx + 1 :]:
            tasks += [
                {
                    "x": df_feat[x_feat].values,
                    "y": df_feat[y_feat].values,
                    "x_name": x_feat,
                    "y_name": y_feat,
                    "random_state": seed,
                    "n_neighbors": n_neighbors,
                }
            ]

    for res in tqdm.tqdm(
        pool.imap_unordered(map_mi, tasks, chunksize=100), total=len(tasks)
    ):
        mutual_info.loc[res[1], res[2]] = mutual_info.loc[res[2], res[1]] = res[0] / (
            0.5 * (diag[res[1]] + diag[res[2]])
        )
    pool.close()
    pool.join()

    mutual_info.fillna(0, inplace=True)  # if na => no relation => set to zero

    if return_entropy:
        return (
            mutual_info,
            diag,
        )  # diag can be useful for future elimination based on entropy without the need of recomputing the cross NMI
    else:
        return mutual_info


def get_rr_p_parameter_default(nn: int) -> float:
    """
    Returns p for the default expression outlined in arXiv:2004:14766.

    Args:
        nn (int): number of features currently in chosen subset.

    Returns:
        float: the value for p.

    """
    return max(0.1, 4.5 - 0.4 * nn ** 0.4)


def get_rr_c_parameter_default(nn: int) -> float:
    """
    Returns c for the default expression outlined in arXiv:2004:14766.

    Args:
        nn (int): number of features currently in chosen subset.

    Returns:
        float: the value for p.

    """
    return min(1e5, 1e-6 * nn ** 3)


def get_features_relevance_redundancy(
    target_nmi: pd.DataFrame,
    cross_nmi: pd.DataFrame,
    n_feat: Optional[int] = None,
    rr_parameters: Optional[Dict[str, Union[float, Callable[[int], float]]]] = None,
    return_pc: bool = False,
) -> List:
    """
    Select features from the Relevance Redundancy (RR) score between the input
    features and the target output.

    The RR is defined following Equation 2 of De Breuck et al, arXiv:2004:14766,
    with default values,

    ..math:: p = \\max{0.1, 4.5 -  n^{0.4}},

    and

    ..math:: c = 10^{-6} n^3,

    where :math:`n` is the number of features in the "chosen" subset for that iteration.
    These values can be overriden with the `rr_parameters` dictionary argument.

    Args:
        target_nmi (pandas.DataFrame): dataframe  containing the Normalized
            Mutual Information (NMI) between a list of input features and a
            target variable, as computed from :py:func:`nmi_target`.
        cross_nmi (pandas.DataFrame): dataframe containing the NMI between the
            input features, as computed from :py:func:`get_cross_nmi`.
        n_feat (int): Number of features for which the RR score needs to be computed (default: all features).
        rr_parameters (dict): Allows tuning of p and c parameters. Currently
            allows fixing of p and c to constant values instead of using the
            dynamical evaluation. Expects to find keys `"p"` and `"c"`, containing
            either a callable that takes `n` as an argument and returns the
            desired `p` or `c`, or another dictionary containing the key `"value"`
            that stores a constant value of `p` or `c`.
        return_pc: Whether to return p and c values in the output dictionaries.

    Returns:
        list: List of dictionaries containing the results of the relevance-redundancy selection algorithm.

    """
    # Initial checks
    if set(cross_nmi.index) != set(cross_nmi.columns):
        raise ValueError(
            "The cross_nmi DataFrame should have its indices and columns identical."
        )
    if not set(target_nmi.index).issubset(set(cross_nmi.index)):
        raise ValueError(
            "The indices of the target DataFrame should be included in the cross_nmi DataFrame indices."
        )

    # Define the functions for the parameters
    if rr_parameters is None:
        get_p = get_rr_p_parameter_default
        get_c = get_rr_c_parameter_default
    else:
        if "p" not in rr_parameters or "c" not in rr_parameters:
            raise ValueError(
                "When tuning p and c with rr_parameters in get_features_relevance_redundancy, "
                "both parameters should be tuned"
            )
        # Set up p
        if callable(rr_parameters["p"]):
            get_p = rr_parameters["p"]
        elif rr_parameters["p"].get("function") == "constant":

            def get_p(_):
                return rr_parameters["p"]["value"]

        else:
            raise ValueError(
                'If not passing a callable, "p" dict must contain keys "function" and "value".'
            )
        # Set up c
        if callable(rr_parameters["c"]):
            get_c = rr_parameters["c"]
        elif rr_parameters["c"].get("function") == "constant":

            def get_c(_):
                return rr_parameters["c"]["value"]

        else:
            raise ValueError(
                'If not passing a callable, "c" dict must contain keys "function" and "value".'
            )

    # Set up the output list
    out = []

    # The first feature is the one with the largest target NMI
    target_column = target_nmi.columns[0]
    first_feature = target_nmi.nlargest(1, columns=target_column).index[0]
    feature_set = [first_feature]
    feat_out = {
        "feature": first_feature,
        "RR_score": None,
        "NMI_target": target_nmi[target_column][first_feature],
    }
    if return_pc:
        feat_out["RR_p"] = None
        feat_out["RR_c"] = None
    out.append(feat_out)

    # Default is to get the RR score for all features
    if n_feat is None:
        n_feat = len(target_nmi.index)

    missing = [x for x in cross_nmi.index if x not in target_nmi.index]
    cross_nmi = cross_nmi.drop(missing, axis=0).drop(missing, axis=1)
    # Loop on the number of features
    for n in range(1, n_feat):
        LOG.debug("In selection of feature {}/{} features...".format(n + 1, n_feat))
        if (n + 1) % 50 == 0:
            LOG.info("Selected {}/{} features...".format(n, n_feat))
        p = get_p(n)
        c = get_c(n)

        # Compute the RR score
        score = cross_nmi.copy()
        # Remove features already selected for the index
        score = score.drop(feature_set, axis=0)
        # Use features already selected to compute the maximum NMI between
        # the remaining features and those already selected
        score = score[feature_set]

        # Get the scores of the remaining features
        for i in score.index:
            row = score.loc[i, :]
            score.loc[i, :] = target_nmi.loc[i, target_column] / (row ** p + c)

        # Get the next feature (the one with the highest score)
        scores_remaining_features = score.min(axis=1)
        next_feature = scores_remaining_features.idxmax(axis=0)
        feature_set.append(next_feature)

        # Add the results for the next feature to the list
        feat_out = {
            "feature": next_feature,
            "RR_score": scores_remaining_features[next_feature],
            "NMI_target": target_nmi[target_column][next_feature],
        }
        if return_pc:
            feat_out["RR_p"] = p
            feat_out["RR_c"] = c

        out.append(feat_out)

    return out


def get_features_dyn(n_feat, cross_nmi, target_nmi):
    missing = [x for x in cross_nmi.index if x not in target_nmi.index]
    cross_nmi = cross_nmi.drop(missing, axis=0).drop(missing, axis=1)

    missing = [x for x in target_nmi.index if x not in cross_nmi.index]
    target_nmi = target_nmi.drop(missing, axis=0)
    target_nmi = target_nmi.replace([np.inf, -np.inf, np.nan], 0)

    first_feature = target_nmi.nlargest(1).index[0]
    feature_set = [first_feature]
    get_p = get_rr_p_parameter_default
    get_c = get_rr_c_parameter_default

    if n_feat == -1:
        n_feat = len(cross_nmi.index)
    else:
        n_feat = min(len(cross_nmi.index), n_feat)

    for n in range(n_feat - 1):
        if (n + 1) % 50 == 0:
            LOG.info("Selected {}/{} features...".format(n + 1, n_feat))

        p = get_p(n)
        c = get_c(n)

        score = cross_nmi.copy()
        # score = score.loc[target_mi.index, target_mi.index]
        score = score.drop(feature_set, axis=0)
        score = score[feature_set]

        for i in score.index:
            row = score.loc[i, :]
            score.loc[i, :] = target_nmi[i] / (row ** p + c)

        next_feature = score.min(axis=1).idxmax(axis=0)
        feature_set.append(next_feature)

    return feature_set


def merge_ranked(lists: List[List[Hashable]]) -> List[Hashable]:
    """For multiple lists of ranked feature names/IDs (e.g. for different
    targets), work through the lists and merge them such that each
    feature is included once according to its highest rank across each
    list.

    Args:
        lists (List[List[Hashable]]): the list of lists to merge.

    Returns:
        List[Hashable]: list of merged and ranked feature names/IDs.

    """
    if not all(len(lists[0]) == len(sublist) for sublist in lists):
        # pad all lists to same length
        max_len = max(len(sublist) for sublist in lists)
        for ind, sublist in enumerate(lists):
            if len(sublist) < max_len:
                lists[ind].extend((max_len - len(sublist)) * [None])

    total_set = set()
    ranked_list = []
    for subrank in zip(*lists):
        for feature in subrank:
            if feature not in total_set and feature is not None:
                ranked_list.append(feature)
                total_set.add(feature)

    return ranked_list


class MODData:
    """The MODData class takes takes a list of `pymatgen.Structure`
    objects and creates a `pandas.DataFrame` that contains many matminer
    features per structure. It then uses mutual information between
    features and targets, and between the features themselves, to
    perform feature selection using relevance-redundancy indices.

    Attributes:
        df_structure (pd.DataFrame): dataframe storing the `pymatgen.Structure`
            representations for each structured, indexed by ID.
        df_targets (pd.Dataframe): dataframe storing the prediction targets
            per structure, indexed by ID.
        df_featurized (pd.DataFrame): dataframe with columns storing all
            computed features per structure, indexed by ID.
        optimal_features (List[str]): if feature selection has been performed
            this attribute stores a list of the selected features.
        optimal_features_by_target (Dict[str, List[str]]): If feature selection has been performed
            this attribute stores a list of the selected features, broken down by target property.
        featurizer (MODFeaturizer): the class used to featurize the data.
        __modnet_version__ (str): The MODNet version number used to create the object
        cross_nmi (pd.DataFrame): If feature selection has been performed, this attribute
            stores the normalized mutual information between all features.
        feature_entropy (Dictionary): Information entropy of all features. Only computed after a call to compute cross_nmi.
        num_classes (Dictionary): Defining the target types (classification or regression).
            Should be constructed as follows: key: string giving the target name; value: integer n,
            with n=0 for regression and n>=2 for classification with n the number of classes.
    """

    def __init__(
        self,
        materials: Optional[List[Union[Structure, Composition]]] = None,
        targets: Optional[Union[List[float], np.ndarray]] = None,
        target_names: Optional[Iterable] = None,
        structure_ids: Optional[Iterable] = None,
        num_classes: Optional[Dict[str, int]] = None,
        df_featurized: Optional[pd.DataFrame] = None,
        featurizer: Optional[Union[MODFeaturizer, str]] = None,
        structures: Optional[List[Union[Structure, Composition]]] = None,
    ):
        """Initialise the MODData object either from a list of structures
        or from an already featurized dataframe. Prediction targets per
        structure can be specified as lists or an array alongside their
        target names. A list of unique IDs can be provided to label the
        structures.

        Args:
            materials: list of structures or compositions to featurize and predict.
            targets: optional List of targets corresponding to each structure. When learning on multiple targets this
             is a ndarray where each column corresponds to a target, i.e. of shape (n_materials,n_targets).
            target_names: optional Iterable (e.g. list) of names of target properties to use in the dataframe.
            structure_ids: optional Iterable of unique IDs to use instead of generated integers.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                 with n=0 for regression and n>=2 for classification with n the number of classes.
            df_featurized: optional featurized dataframe to use instead of
                featurizing a new one. Should be passed without structures.
            featurizer: optional MODFeaturizer object to use for featurization, or string
                preset to look up in presets dictionary.
            structures: deprecated (alias to materials for backward compatibility) do not use this.

        """

        from modnet.featurizers.presets import FEATURIZER_PRESETS

        self.__modnet_version__ = __version__
        self.df_featurized = df_featurized
        self.featurizer = featurizer
        self.cross_nmi = None

        if structures is not None:  # overwrite materials for backward compatibility
            materials = structures

        if materials is not None and self.df_featurized is not None:
            if len(materials) != len(self.df_featurized):
                raise RuntimeError(
                    "Mismatched shape of structures and passed df_featurized"
                )

        if materials is None and self.df_featurized is None:
            raise RuntimeError(
                "At least one of `structures` or `df_featurized` should be passed to `MODData`."
            )

        if targets is not None:
            targets = np.array(targets).reshape((len(targets), -1))

        if materials is not None and targets is not None:
            if np.shape(targets)[0] != len(materials):
                raise ValueError(
                    f"Targets must have same length as structures: {np.shape(targets)} vs {len(materials)}"
                )

        if materials is not None and isinstance(materials[0], Composition):
            materials = [CompositionContainer(s) for s in materials]
            self._composition_only = True

        if isinstance(featurizer, str):
            self.featurizer = FEATURIZER_PRESETS.get(featurizer)()
            if self.featurizer is None:
                raise RuntimeError(
                    "Requested preset {featurizer} not found in available presets: {FEATURIZER_PRESETS.keys()}"
                )
        elif isinstance(featurizer, MODFeaturizer):
            self.featurizer = featurizer
        elif featurizer is None and self.df_featurized is None:
            if getattr(self, "_composition_only", False):
                self.featurizer = FEATURIZER_PRESETS["CompositionOnly"]()
            else:
                self.featurizer = FEATURIZER_PRESETS["DeBreuck2020"]()

        if self.featurizer is not None:
            LOG.info(f"Loaded {self.featurizer.__class__.__name__} featurizer.")

        if target_names is not None:
            if np.shape(targets)[-1] != len(target_names):
                raise ValueError("Target names must be supplied for every target.")
        elif targets is not None:
            target_names = ["prop" + str(i) for i in range(len(targets))]

        if structure_ids is not None:
            # for backwards compat, always store the *passed* list of
            # IDs, so they can be used when loading from a database file
            # check ids are unique
            if len(set(structure_ids)) != len(structure_ids):
                raise ValueError(
                    "List of IDs (`structure_ids`) provided must be unique."
                )

            if len(structure_ids) != len(materials):
                raise ValueError(
                    "List of IDs (`structure_ids`) must have same length as list of structure."
                )

        else:
            num_entries = (
                len(materials) if materials is not None else len(df_featurized)
            )
            structure_ids = [f"id{i}" for i in range(num_entries)]

        if targets is not None:
            # set up dataframe for targets with columns (id, property_1, ..., property_n)
            self.df_targets = pd.DataFrame(
                targets, index=structure_ids, columns=target_names
            )
            # set up number of classes
            self.num_classes = {name: 0 for name in self.target_names}
            if num_classes is not None:
                self.num_classes.update(num_classes)

        # set up dataframe for structures with columns (id, structure)
        self.df_structure = pd.DataFrame({"id": structure_ids, "structure": materials})
        self.df_structure.set_index("id", inplace=True)

    def featurize(
        self, fast: bool = False, db_file: str = "feature_database.pkl", n_jobs=None
    ):
        """For the input structures, construct many matminer features
        and save a featurized dataframe. If `db_file` is specified, this
        method will try to load previous feature calculations for each
        structure ID instead of recomputing.

        Sets the `self.df_featurized` attribute.

        Args:
            fast (bool): whether or not to try to load from a backup.
            db_file (str): filename of a pickled dataframe containing
                with the same ID index as this `MODData` object.

        """

        LOG.info("Computing features, this can take time...")

        df_done = None
        df_todo = None

        if n_jobs is not None:
            self.featurizer.set_n_jobs(n_jobs)

        if self.df_featurized is not None:
            raise RuntimeError("Not overwriting existing featurized dataframe.")

        if fast:
            LOG.info("Fast featurization on, retrieving from database...")

            global DATABASE
            if DATABASE.empty:
                DATABASE = pd.read_pickle(db_file)

            ids_done = [x for x in self.structure_ids if x in DATABASE.index]

            LOG.info(
                f"Retrieved features for {len(ids_done)} out of {len(self.structure_ids)} materials"
            )
            df_done = DATABASE.loc[ids_done]
            df_todo = self.df_structure.drop(ids_done, axis=0)

        # if any structures were already loaded
        if fast and not df_done.empty:
            # if any are left to compute, do them
            if len(df_todo) > 0:
                df_finished = self.featurizer.featurize(df_todo)
                df_final = df_done.append(df_finished)
                df_final = df_final.reindex(self.structure_ids)

            # otherwise, all structures were successfully loaded
            else:
                df_final = df_done

        # otherwise, no structures were loaded, so we need to compute all
        else:
            df_final = self.featurizer.featurize(self.df_structure)

        df_final = df_final.replace([np.inf, -np.inf, np.nan], 0)

        self.df_featurized = df_final
        LOG.info("Data has successfully been featurized!")

    def feature_selection(
        self,
        n: int = 1500,
        cross_nmi: Optional[pd.DataFrame] = None,
        use_precomputed_cross_nmi: bool = False,
        n_samples=6000,
        n_jobs: int = None,
    ):
        """Compute the mutual information between features and targets,
        then apply relevance-redundancy rankings to choose the top `n`
        features.

        Sets the `self.optimal_features` attribute to a list of feature
        names.

        Args:
            n: number of desired features.
            cross_nmi: specify the cross NMI between features as a
                dataframe.
            use_precomputed_cross_nmi: Whether or not to use the cross NMI
                that was computed on Materials Project features, instead of
                precomputing.
            n_jobs: max. number of processes to use when calculating cross NMI.

        """
        if getattr(self, "df_featurized", None) is None:
            raise RuntimeError(
                "Mutual information feature selection requiresd featurized data, please call `.featurize()`"
            )
        if getattr(self, "df_targets", None) is None:
            raise RuntimeError(
                "Mutual information feature selection requires target properties"
            )

        ranked_lists = []
        optimal_features_by_target = {}

        if cross_nmi is not None:
            self.cross_nmi = cross_nmi
        elif getattr(self, "cross_nmi", None) is None:
            self.cross_nmi = None

        # Loading mutual information between features
        if use_precomputed_cross_nmi:
            LOG.info("Loading cross NMI from 'Features_cross' file.")
            from modnet.ext_data import load_ext_dataset

            cnmi_path = load_ext_dataset("MP_2018.6_CROSS_NMI", "cross_nmi")
            self.cross_nmi = pd.read_pickle(cnmi_path)
            precomputed_cols = set(self.cross_nmi.columns)
            featurized_cols = set(self.df_featurized.columns)
            if len(precomputed_cols | featurized_cols) > len(precomputed_cols):
                LOG.warning(
                    "Feature mismatch between precomputed `Features_cross` and `df_featurized`. "
                    f"Missing columns: {featurized_cols - precomputed_cols}"
                )

        if self.cross_nmi is None:
            if len(self.df_featurized) > n_samples:
                df = self.df_featurized.sample(n=n_samples, random_state=12)
            else:
                df = self.df_featurized.copy()
            self.cross_nmi, self.feature_entropy = get_cross_nmi(
                df, return_entropy=True, n_jobs=n_jobs
            )

        if self.cross_nmi.isna().sum().sum() > 0:
            raise RuntimeError(
                "Cross NMI (`moddata.cross_nmi`) contains NaN values, consider setting them to zero."
            )

        for i, name in enumerate(self.names):
            LOG.info(f"Starting target {i + 1}/{len(self.names)}: {self.names[i]} ...")

            # Computing mutual information with target
            LOG.info("Computing mutual information between features and target...")
            if getattr(self, "num_classes", None) and self.num_classes[name] >= 2:
                task_type = "classification"
            else:
                task_type = "regression"

            if len(self.df_featurized) > n_samples:
                subset_ids = np.random.permutation(len(self.df_featurized))[:n_samples]
                df = self.df_featurized.iloc[subset_ids]
                df_target = self.df_targets.iloc[subset_ids][[name]]
            else:
                df = self.df_featurized.copy()
                df_target = self.df_targets[[name]]
            self.target_nmi = nmi_target(
                df,
                df_target,
                task_type,
            )[name]

            LOG.info("Computing optimal features...")
            optimal_features_by_target[name] = get_features_dyn(
                n, self.cross_nmi, self.target_nmi
            )
            ranked_lists.append(optimal_features_by_target[name])

            LOG.info("Done with target {}/{}: {}.".format(i + 1, len(self.names), name))

        LOG.info("Merging all features...")
        self.optimal_features = merge_ranked(ranked_lists)
        self.optimal_features_by_target = optimal_features_by_target
        LOG.info("Done.")

    def shuffle(self):
        # caution, not fully implemented
        raise NotImplementedError("shuffle function not yet finished.")
        self.df_featurized = self.df_featurized.sample(frac=1)
        self.df_targets = self.df_targets.loc[self.df_featurized.index]

    @property
    def structures(self) -> List[Union[Structure, CompositionContainer]]:
        """Returns the list of `pymatgen.Structure` objects. """
        return list(self.df_structure["structure"])

    @property
    def compositions(self) -> List[Union[Structure, CompositionContainer]]:
        """Returns the list of materials as`pymatgen.Composition` objects. """
        return [s.composition for s in self.df_structure["structure"]]

    @property
    def targets(self) -> np.ndarray:
        """ Returns a ndarray of prediction targets. """
        return self.df_targets.values

    @property
    def names(self) -> List[str]:
        """ Returns the list of prediction target field names. """
        return list(self.df_targets)

    @property
    def target_names(self) -> List[str]:
        """ Returns the list of prediction target field names. """
        return list(self.df_targets)

    @property
    def structure_ids(self) -> List[str]:
        """ Returns the list of prediction target field names. """
        return list(self.df_structure.index)

    def save(self, filename: str):
        """Pickle the contents of the `MODData` object
        so that it can be loaded in  with `MODData.load()`.

        If the filename ends in "tgz", "bz2" or "zip", the pickle
        will be compressed accordingly by `pandas.to_pickle(...)`.

        """
        pd.to_pickle(self, filename)
        LOG.info(f"Data successfully saved as {filename}!")

    @staticmethod
    def load(filename: Union[str, Path]) -> MODData:
        """Load `MODData` object pickled by the `.save(...)` method.

        If the filename ends in "tgz", "bz2" or "zip", the pickle
        will be decompressed accordingly by `pandas.read_pickle(...)`.

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

        if isinstance(pickled_data, MODData):
            if not hasattr(pickled_data, "__modnet_version__"):
                pickled_data.__modnet_version__ = "<=0.1.7"
            LOG.info(
                f"Loaded {pickled_data} object, created with modnet version {pickled_data.__modnet_version__}"
            )
            return pickled_data

        raise ValueError(
            f"File {filename} did not contain compatible data to create a MODData object, "
            f"instead found {pickled_data.__class__.__name__}."
        )

    @classmethod
    def load_precomputed(cls, dataset_name: str):
        """Load a `MODData` object from a pre-computed dataset.

        Note:
            Datasets may require significant (~10 GB) amounts of memory
            to load.

        Arguments:
            dataset: the name of the precomputed dataset to load.
                Currently available: 'MP_2018.6'.

        Returns:
            MODData: the precomputed dataset.

        """
        from modnet.ext_data import load_ext_dataset

        model_path = load_ext_dataset(dataset_name, "MODData")
        return cls.load(str(model_path))

    def get_structure_df(self):
        return self.df_structure

    def get_target_df(self):
        return self.df_targets

    def get_featurized_df(self):
        return self.df_featurized

    def get_optimal_descriptors(self):
        return self.optimal_features

    def get_optimal_df(self):
        return self.df_featurized[self.optimal_features].join(self.get_target_df())

    def split(
        self, train_test_split: Tuple[List[int], List[int]]
    ) -> Tuple[MODData, MODData]:
        """Create two new MODData's that contain only the data corresponding
        to the indices passed in the `train_test_split` tuple.

        Arguments:
            train_test_split: A tuple containing two lists of integers: the
                indices of the training data and test data respectively.

        Returns:
            The training MODData and the test MODData as a tuple.

        """

        train, test = train_test_split
        train_moddata = self.from_indices(train)
        test_moddata = self.from_indices(test)
        return train_moddata, test_moddata

    def from_indices(self, indices: List[int]) -> MODData:
        """Create a new MODData that contains only the data at the given
        rows indices provided.

        Arguments:
            indices: The list of integers corresponding to the rows.

        Returns:
            A `MODData` containing only the rows passed.

        """
        split_data = MODData.__new__(MODData)
        extensive_dataframes = ("df_structure", "df_targets", "df_featurized")
        for attr in extensive_dataframes:
            setattr(split_data, attr, getattr(self, attr).iloc[indices])

        for attr in [_ for _ in dir(self) if _ not in extensive_dataframes]:
            try:
                if not callable(getattr(self, attr)) and not attr.startswith("__"):
                    setattr(split_data, attr, getattr(self, attr))
            except AttributeError:
                pass

        split_data.__modnet_version__ = __version__

        return split_data
