from typing import Optional
from typing import List
from typing import Union
from typing import Any
from loguru import logger
import numpy as np

import skcriteria
from skcriteria.preprocessing.scalers import VectorScaler
from skcriteria.preprocessing.scalers import MinMaxScaler
from skcriteria.preprocessing.scalers import StandarScaler
from skcriteria.preprocessing.scalers import SumScaler
from skcriteria.preprocessing.scalers import MaxScaler
from skcriteria.madm.moora import ReferencePointMOORA
from skcriteria.madm.moora import RatioMOORA
from skcriteria.madm.moora import MultiMOORA

MOORA_METHODS = {
    "ReferencePointMOORA": ReferencePointMOORA,
    "RatioMOORA": RatioMOORA,
    "MultiMOORA": MultiMOORA,
}

MOORA_TRANSFORMERS = {
    "VectorScaler": VectorScaler,
    "MinMaxScaler": MinMaxScaler,
    "StandarScaler": StandarScaler,
    "SumScaler": SumScaler,
    "MaxScaler": MaxScaler,
}

OBJECTIVE_MAPPER = {
    "max": max,
    "min": min,
    "nanmin": np.nanmin,
    "nanmax": np.nanmax,
}


def ranking_moora(
    matrix: np.ndarray,
    objectives: Union[Any, List[Any]],
    weights: Union[float, List[float]] = None,
    method: str = "RatioMOORA",
    transformer: Optional[str] = "MinMaxScaler",
):
    """Rank data points using Multi-Objective Optimization Method by Ratio Analysis (MOORA).

    Under the hood the lib scikit-criteria is used. More details at https://github.com/quatrope/scikit-criteria

    Args:
        matrix: The values to be ranked.
        objectives: The objectives to use for ranking. Either `min` or `max` for each objective.
        weights: The weight of the criterias.
        method: The ranking method to use from `ReferencePointMOORA`, `RatioMOORA` or `MultiMOORA`.
        transformer: The transformer to use. It can be `None` or choose from `VectorScaler`, `MinMaxScaler`,
            `StandarScaler`, `SumScaler` or `MaxScaler`.
    """

    if matrix.ndim != 2:
        raise ValueError("The matrix should be 2-dimensional.")

    if not isinstance(objectives, list):
        objectives = [objectives] * matrix.shape[1]

    if isinstance(objectives[0], str):
        objectives = [OBJECTIVE_MAPPER[k] for k in objectives]

    if len(objectives) != matrix.shape[1]:
        raise ValueError(
            "The number of objectives should be equal to the number of columns in the matrix."
        )

    if weights is None:
        weights = [1] * len(objectives)

    if isinstance(weights, list) and len(weights) != len(objectives):
        raise ValueError(
            "The number of weights should be equal to the number of objectives."
        )

    if any(not callable(x) for x in set(objectives)):
        raise ValueError(
            "The objectives should be a callable. Preferably `min` or `max`."
        )

    # Get the MOORA method class
    ranker_class = MOORA_METHODS.get(method)

    if ranker_class is None:
        raise ValueError(
            f"The MOORA method '{method}' is not allowed. Supported methods are"
            f" {list(MOORA_METHODS.keys())}"
        )

    # Build the decision matrix object
    decision_matrix = skcriteria.mkdm(
        matrix=matrix, objectives=objectives, weights=weights
    )

    if transformer is not None:

        # Get the transformer class
        transformer_class = MOORA_TRANSFORMERS.get(transformer)

        if transformer_class is None:
            raise ValueError(
                f"The MOORA transformer '{transformer}' is not allowed. Supported methods are"
                f" {list(MOORA_TRANSFORMERS.keys())}"
            )

        # Scale values
        transformer_obj = transformer_class(target="matrix")
        decision_matrix = transformer_obj.transform(decision_matrix)

    # Rank
    ranker = ranker_class()
    result = ranker.evaluate(decision_matrix)

    return result.rank_
