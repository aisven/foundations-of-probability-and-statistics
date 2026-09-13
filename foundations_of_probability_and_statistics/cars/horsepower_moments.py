import numpy as np

from foundations_of_probability_and_statistics.cars.car_distribution import create_car_distribution


def _brand_code(brand: str) -> int:
    # map the brand name back to its integer code, failing readably on unknown brands
    car_distribution = create_car_distribution()
    codes = {name: code for code, name in car_distribution.brand_names.items()}
    if brand not in codes:
        raise ValueError(f"Unknown brand '{brand}', must be one of {sorted(codes)}.")
    return codes[brand]


def conditional_horsepower_mean(brand: str) -> float:
    """Compute the exact conditional mean of horsepower given a brand.

    Evaluates E[H | B = b] = sum_h h * p(H = h | B = b).

    Args:
        brand: Brand to condition on, one of VW, Porsche or Ferrari.

    Returns:
        float: The exact conditional mean horsepower for the brand.
    """
    car_distribution = create_car_distribution()
    code = _brand_code(brand)
    # E[H | B = b] = sum_h h * p(H = h | B = b)
    rv = car_distribution.horsepower_rvs[code]
    return float(np.sum(np.asarray(rv.xk, dtype=float) * np.asarray(rv.pmf(rv.xk), dtype=float)))


def conditional_horsepower_variance(brand: str) -> float:
    """Compute the exact conditional variance of horsepower given a brand.

    Evaluates Var(H | B = b) = sum_h (h - E[H | B = b])^2 * p(H = h | B = b).

    Args:
        brand: Brand to condition on, one of VW, Porsche or Ferrari.

    Returns:
        float: The exact conditional variance of horsepower for the brand.
    """
    car_distribution = create_car_distribution()
    code = _brand_code(brand)
    # Var(H | B = b) = sum_h (h - E[H | B = b])^2 * p(H = h | B = b)
    rv = car_distribution.horsepower_rvs[code]
    xk = np.asarray(rv.xk, dtype=float)
    pk = np.asarray(rv.pmf(rv.xk), dtype=float)
    mean = float(np.sum(xk * pk))
    return float(np.sum(((xk - mean) ** 2) * pk))


def conditional_horsepower_moments(brand: str) -> tuple[float, float]:
    """Compute the exact conditional mean and variance of horsepower given a brand.

    Args:
        brand: Brand to condition on, one of VW, Porsche or Ferrari.

    Returns:
        tuple[float, float]: The pair (E[H | B = b], Var(H | B = b)).
    """
    return conditional_horsepower_mean(brand), conditional_horsepower_variance(brand)
