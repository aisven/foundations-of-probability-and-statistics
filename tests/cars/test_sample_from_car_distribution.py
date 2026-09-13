import numpy as np
import pandas as pd
import pytest

from foundations_of_probability_and_statistics.cars.car_distribution import FERRARI, PORSCHE, VW, create_car_distribution
from foundations_of_probability_and_statistics.cars.sample_from_car_distribution import sample_from_car_distribution


def test_sample_from_car_distribution():
    n_cars = 10_000
    cars = sample_from_car_distribution(n_cars=n_cars, random_state=42)

    # the sample is a data frame with the expected columns and rows
    assert list(cars.columns) == ["brand", "horsepower", "color"]
    assert len(cars) == n_cars

    # values stay within the supported domains
    assert set(cars["brand"]) <= {"VW", "Porsche", "Ferrari"}
    assert set(cars["horsepower"]) <= {100, 200, 300, 400, 500, 600, 700}
    assert set(cars["color"]) <= {"blue", "gray", "black", "green", "red", "yellow"}

    # sampling is reproducible for a fixed random_state
    assert cars.equals(sample_from_car_distribution(n_cars=n_cars, random_state=42))

    # empirical frequencies approximate the theoretical probabilities
    empirical = cars["brand"].value_counts(normalize=True)
    assert np.allclose([empirical["VW"], empirical["Porsche"], empirical["Ferrari"]], [0.5, 0.3, 0.2], atol=0.01)

    # empirical joint frequency approximates p(horsepower = 700, color = black) = 0.006
    joint = ((cars["horsepower"] == 700) & (cars["color"] == "black")).mean()
    assert joint == pytest.approx(0.006, abs=0.005)


def test_sample_with_int_seed_is_reproducible():
    cars_a = sample_from_car_distribution(n_cars=50, random_state=7)
    cars_b = sample_from_car_distribution(n_cars=50, random_state=7)

    # the same integer seed reproduces the same sample
    pd.testing.assert_frame_equal(cars_a, cars_b)


def test_sample_without_seed_varies():
    cars_a = sample_from_car_distribution(n_cars=50)
    cars_b = sample_from_car_distribution(n_cars=50)

    # fresh entropy leads to different samples
    assert not cars_a.equals(cars_b)


def test_sample_with_same_generator_seed_is_reproducible():
    cars_a = sample_from_car_distribution(n_cars=50, random_state=np.random.default_rng(7))
    cars_b = sample_from_car_distribution(n_cars=50, random_state=np.random.default_rng(7))

    # two generators in the same state produce the same sample
    pd.testing.assert_frame_equal(cars_a, cars_b)


def test_sample_with_shared_generator_continues_stream():
    rng = np.random.default_rng(7)
    cars_a = sample_from_car_distribution(n_cars=50, random_state=rng)
    cars_b = sample_from_car_distribution(n_cars=50, random_state=rng)

    # a shared generator continues its stream, so consecutive samples differ
    assert not cars_a.equals(cars_b)


def test_sample_zero_cars():
    cars = sample_from_car_distribution(n_cars=0, random_state=42)

    assert len(cars) == 0
    assert list(cars.columns) == ["brand", "horsepower", "color"]


def test_sample_negative_cars_raises():
    with pytest.raises(ValueError):
        sample_from_car_distribution(n_cars=-1)


@pytest.mark.parametrize(
    ("brand_name", "expected"),
    [
        ("VW", {"blue": 0.2, "gray": 0.5, "black": 0.3}),
        ("Porsche", {"green": 0.1, "blue": 0.2, "gray": 0.3, "black": 0.4}),
        ("Ferrari", {"red": 0.6, "black": 0.2, "yellow": 0.2}),
    ],
)
def test_empirical_color_frequencies_match_conditionals(brand_name: str, expected: dict[str, float]):
    cars = sample_from_car_distribution(n_cars=60_000, random_state=42)
    empirical = cars[cars["brand"] == brand_name]["color"].value_counts(normalize=True)

    for color, pk in expected.items():
        assert empirical.get(color, 0.0) == pytest.approx(pk, abs=0.02)


def test_empirical_joint_within_brand_matches_conditional_product():
    car_distribution = create_car_distribution()
    cars = sample_from_car_distribution(n_cars=200_000, random_state=42)

    # within a brand, horsepower and color are independent: p(h, c | b) = p(h | b) p(c | b)
    for brand in [VW, PORSCHE, FERRARI]:
        sub = cars[cars["brand"] == car_distribution.brand_names[brand]]
        for h, p_h in zip(
            car_distribution.horsepower_rvs[brand].xk, car_distribution.horsepower_rvs[brand].pmf(car_distribution.horsepower_rvs[brand].xk), strict=True
        ):
            for c, p_c in zip(car_distribution.color_rvs[brand].xk, car_distribution.color_rvs[brand].pmf(car_distribution.color_rvs[brand].xk), strict=True):
                color = car_distribution.color_names[c]
                empirical = ((sub["horsepower"] == h) & (sub["color"] == color)).mean()
                assert empirical == pytest.approx(p_h * p_c, abs=0.01)
