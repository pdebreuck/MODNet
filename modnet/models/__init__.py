import warnings

from .vanilla import MODNetModel

try:
    from .bayesian import BayesianMODNetModel
except ImportError:
    warnings.warn(
        "BayesianMODNetModel is deprecated and may be removed in the future.",
        DeprecationWarning,
    )

    BayesianMODNetModel = None

from .ensemble import EnsembleMODNetModel

__all__ = (
    "MODNetModel",
    "BayesianMODNetModel",
    "EnsembleMODNetModel",
)
