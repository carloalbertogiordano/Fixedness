from .base import Anonymizer
from .syntactic import NoAnonymization, Suppression, MondrianK, LDiversity, TCloseness
from .perturbative import NoiseAddition, MicroAggregation, RandomizedResponse
from .formal import LaplaceDP, LocalDP

_REGISTRY = {
    'no_anonymization':   NoAnonymization,
    'suppression':        Suppression,
    'mondrian_k':         MondrianK,
    'l_diversity':        LDiversity,
    't_closeness':        TCloseness,
    'noise_addition':     NoiseAddition,
    'microaggregation':   MicroAggregation,
    'randomized_response': RandomizedResponse,
    'laplace_dp':         LaplaceDP,
    'local_dp':           LocalDP,
}


def get_anonymizer(method: str) -> Anonymizer:
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown anonymization method '{method}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[method]()


def available_methods():
    return sorted(_REGISTRY.keys())
