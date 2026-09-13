import numpy as np
import pandas as pd

from foundations_of_probability_and_statistics.cars.car_distribution import create_car_distribution


def sample_from_car_distribution(n_cars: int, random_state: int | np.random.Generator | None = None) -> pd.DataFrame:
    """Sample cars from the joint car distribution.

    Draws are taken via the factorization
    p(brand, horsepower, color) = p(brand) * p(horsepower | brand) * p(color | brand)
    from the random variables created by create_car_distribution.

    Args:
        n_cars: Number of cars to sample.
        random_state: Seed or generator controlling the pseudo-random number
            generator, e.g. None for fresh entropy, an int for a reproducible
            sample, or a numpy.random.Generator for caller-controlled streams.

    Returns:
        pd.DataFrame: One row per sampled car with columns brand, horsepower
            and color, holding the category names and the horsepower values.
    """
    # sample the joint via p(brand, horsepower, color) = p(brand) * p(horsepower | brand) * p(color | brand)
    car_distribution = create_car_distribution()
    # single generator so all draws share one reproducible stream
    rng = np.random.default_rng(random_state)
    brand_codes = car_distribution.brand_rv.rvs(size=n_cars, random_state=rng).astype(int)

    horsepower_values = np.empty(n_cars, dtype=int)
    color_codes = np.empty(n_cars, dtype=int)
    for brand, horsepower_rv in car_distribution.horsepower_rvs.items():
        # sample the conditionals per brand block, vectorized within each brand
        idx = np.flatnonzero(brand_codes == brand)
        horsepower_values[idx] = horsepower_rv.rvs(size=idx.size, random_state=rng)
        color_codes[idx] = car_distribution.color_rvs[brand].rvs(size=idx.size, random_state=rng)

    return pd.DataFrame(
        {
            "brand": [car_distribution.brand_names[code] for code in brand_codes],
            "horsepower": horsepower_values,
            "color": [car_distribution.color_names[code] for code in color_codes],
        }
    )
