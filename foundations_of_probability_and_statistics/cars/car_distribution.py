from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# the word distribution here stands for probability distribution

# integer codes for the categorical values
# the codes exist because scipy.stats.rv_discrete requires numeric support values
# it sorts the support and applies arithmetic such as max which fails on strings
# the names themselves stay fixed
# so the mappings BRAND_NAMES and COLOR_NAMES are the single source of truth

VW = 0
PORSCHE = 1
FERRARI = 2
BRAND_NAMES: Mapping[int, str] = {VW: "VW", PORSCHE: "Porsche", FERRARI: "Ferrari"}

RED = 1
GREEN = 2
BLUE = 3
GRAY = 4
BLACK = 5
YELLOW = 6
COLOR_NAMES: Mapping[int, str] = {BLUE: "blue", GRAY: "gray", BLACK: "black", GREEN: "green", RED: "red", YELLOW: "yellow"}


@dataclass(frozen=True)
class CarDistribution:
    """Joint car distribution p(brand, horsepower, color) = p(brand) * p(horsepower | brand) * p(color | brand).

    Horsepower and color are conditionally independent given the brand.

    Attributes:
        brand_rv: Marginal distribution of the brand over the integer brand codes.
        horsepower_rvs: Mapping from brand code to the conditional distribution
            p(horsepower | brand) over the horsepower values.
        color_rvs: Mapping from brand code to the conditional distribution
            p(color | brand) over the integer color codes.
        brand_names: Mapping from integer brand code to brand name, e.g. VW to "VW".
        color_names: Mapping from integer color code to color name, e.g. RED to "red".
    """

    brand_rv: stats.rv_discrete
    horsepower_rvs: Mapping[int, stats.rv_discrete]
    color_rvs: Mapping[int, stats.rv_discrete]
    brand_names: Mapping[int, str]
    color_names: Mapping[int, str]


def _create_rv_discrete(name: str, xk: Sequence[int], pk: Sequence[float]) -> stats.rv_discrete:
    # p(X = x_k) = p_k for the support values x_k
    return stats.rv_discrete(name=name, values=(np.asarray(xk), np.asarray(pk)))


def create_car_distribution() -> CarDistribution:
    """Create the joint distribution of cars as SciPy discrete random variables.

    The joint distribution factorizes as
    p(brand, horsepower, color) = p(brand) * p(horsepower | brand) * p(color | brand),
    with horsepower and color conditionally independent given the brand.
    Categories are encoded as integer codes (see BRAND_NAMES and COLOR_NAMES)
    because scipy.stats.rv_discrete only supports numeric support values.

    Returns:
        CarDistribution: The brand marginal plus the per-brand conditional
            horsepower and color random variables, together with the mappings
            from integer codes to category names.
    """
    # brand marginals: P(VW) = 0.5, P(Porsche) = 0.3, P(Ferrari) = 0.2
    brand_rv = _create_rv_discrete("brand", [VW, PORSCHE, FERRARI], [0.5, 0.3, 0.2])

    # conditional horsepower distributions p(horsepower | brand)
    horsepower_rvs: Mapping[int, stats.rv_discrete] = {
        VW: _create_rv_discrete("vw_horsepower", [100, 200, 300], [0.6, 0.3, 0.1]),
        PORSCHE: _create_rv_discrete("porsche_horsepower", [300, 400, 500, 600, 700], [0.4, 0.4, 0.1, 0.05, 0.05]),
        FERRARI: _create_rv_discrete("ferrari_horsepower", [400, 500, 600], [0.3, 0.4, 0.3]),
    }

    # conditional color distributions p(color | brand)
    color_rvs: Mapping[int, stats.rv_discrete] = {
        VW: _create_rv_discrete("vw_color", [BLUE, GRAY, BLACK], [0.2, 0.5, 0.3]),
        PORSCHE: _create_rv_discrete("porsche_color", [GREEN, BLUE, GRAY, BLACK], [0.1, 0.2, 0.3, 0.4]),
        FERRARI: _create_rv_discrete("ferrari_color", [RED, BLACK, YELLOW], [0.6, 0.2, 0.2]),
    }

    return CarDistribution(
        brand_rv=brand_rv,
        horsepower_rvs=horsepower_rvs,
        color_rvs=color_rvs,
        brand_names=BRAND_NAMES,
        color_names=COLOR_NAMES,
    )


def create_joint_car_distribution() -> pd.Series:
    """Create the full joint distribution of cars as a pandas Series.

    Enumerates all 38 support states and computes each probability via the
    product rule p(brand, horsepower, color) = p(brand) * p(horsepower | brand) * p(color | brand),
    so the joint is exactly equivalent to the factorized representation.

    Returns:
        pd.Series: Probability of each joint state, indexed by the MultiIndex
            (brand, horsepower, color) with the category names, and summing
            to one.
    """
    car_distribution = create_car_distribution()
    # full joint via the product rule p(brand, horsepower, color) = p(brand) * p(horsepower | brand) * p(color | brand)
    joint = pd.Series(
        {
            (car_distribution.brand_names[brand], int(horsepower), car_distribution.color_names[color]): car_distribution.brand_rv.pmf(brand)
            * car_distribution.horsepower_rvs[brand].pmf(horsepower)
            * car_distribution.color_rvs[brand].pmf(color)
            for brand in car_distribution.horsepower_rvs
            for horsepower in car_distribution.horsepower_rvs[brand].xk
            for color in car_distribution.color_rvs[brand].xk
        },
        name="p",
    )
    # named index levels for readable selection, e.g. joint.xs("Porsche", level="brand")
    joint.index.names = ["brand", "horsepower", "color"]
    return joint
