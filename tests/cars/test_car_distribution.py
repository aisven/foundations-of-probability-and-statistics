import dataclasses

import numpy as np
import pandas as pd
import pytest

from foundations_of_probability_and_statistics.cars.car_distribution import (
    BLACK,
    BLUE,
    FERRARI,
    GRAY,
    GREEN,
    PORSCHE,
    RED,
    VW,
    YELLOW,
    create_car_distribution,
    create_joint_car_distribution,
    sample_from_car_distribution,
)


def test_create_car_distribution():
    car_distribution = create_car_distribution()

    # brand marginals: p(VW) = 0.5, p(Porsche) = 0.3, p(Ferrari) = 0.2
    assert car_distribution.brand_rv.pmf(PORSCHE) == pytest.approx(0.3)

    # conditionals: p(horsepower = 600 | Porsche) = 0.05 and p(color = black | Porsche) = 0.4
    assert car_distribution.horsepower_rvs[PORSCHE].pmf(600) == pytest.approx(0.05)
    assert car_distribution.color_rvs[PORSCHE].pmf(BLACK) == pytest.approx(0.4)

    # joint factorization: p(Porsche, horsepower = 600, color = black) = 0.3 * 0.05 * 0.4 = 0.006
    joint = car_distribution.brand_rv.pmf(PORSCHE) * car_distribution.horsepower_rvs[PORSCHE].pmf(600) * car_distribution.color_rvs[PORSCHE].pmf(BLACK)
    assert joint == pytest.approx(0.006)

    # all pmfs sum to one: sum_k p(X = x_k) = 1
    for rv in [car_distribution.brand_rv, *car_distribution.horsepower_rvs.values(), *car_distribution.color_rvs.values()]:
        assert rv.pmf(rv.xk).sum() == pytest.approx(1.0)


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


@pytest.mark.parametrize(
    ("brand", "expected_pk"),
    [(VW, 0.5), (PORSCHE, 0.3), (FERRARI, 0.2)],
)
def test_brand_marginals(brand: int, expected_pk: float):
    car_distribution = create_car_distribution()

    assert car_distribution.brand_rv.pmf(brand) == pytest.approx(expected_pk)


@pytest.mark.parametrize(
    ("brand", "expected_xk", "expected_pk"),
    [
        (VW, [100, 200, 300], [0.6, 0.3, 0.1]),
        (PORSCHE, [300, 400, 500, 600, 700], [0.4, 0.4, 0.1, 0.05, 0.05]),
        (FERRARI, [400, 500, 600], [0.3, 0.4, 0.3]),
    ],
)
def test_conditional_horsepower_distributions(brand: int, expected_xk: list[int], expected_pk: list[float]):
    car_distribution = create_car_distribution()

    rv = car_distribution.horsepower_rvs[brand]
    assert rv.xk.tolist() == expected_xk
    assert rv.pmf(expected_xk).tolist() == pytest.approx(expected_pk)


@pytest.mark.parametrize(
    ("brand", "expected_xk", "expected_pk"),
    [
        (VW, [BLUE, GRAY, BLACK], [0.2, 0.5, 0.3]),
        (PORSCHE, [GREEN, BLUE, GRAY, BLACK], [0.1, 0.2, 0.3, 0.4]),
        (FERRARI, [RED, BLACK, YELLOW], [0.6, 0.2, 0.2]),
    ],
)
def test_conditional_color_distributions(brand: int, expected_xk: list[int], expected_pk: list[float]):
    car_distribution = create_car_distribution()

    rv = car_distribution.color_rvs[brand]
    assert rv.xk.tolist() == expected_xk
    assert rv.pmf(expected_xk).tolist() == pytest.approx(expected_pk)


def test_joint_pmf_sums_to_one():
    car_distribution = create_car_distribution()

    # sum over the full support grid: sum_b sum_h sum_c p(b) p(h | b) p(c | b) = 1
    total = sum(
        car_distribution.brand_rv.pmf(b) * car_distribution.horsepower_rvs[b].pmf(h) * car_distribution.color_rvs[b].pmf(c)
        for b in [VW, PORSCHE, FERRARI]
        for h in car_distribution.horsepower_rvs[b].xk
        for c in car_distribution.color_rvs[b].xk
    )
    assert total == pytest.approx(1.0)


def test_code_mappings_are_consistent():
    car_distribution = create_car_distribution()

    # codes and names are unique and cover every support value
    assert set(car_distribution.brand_names) == {VW, PORSCHE, FERRARI}
    assert set(car_distribution.brand_names.values()) == {"VW", "Porsche", "Ferrari"}
    assert set(car_distribution.color_names) == {RED, GREEN, BLUE, GRAY, BLACK, YELLOW}
    assert set(car_distribution.color_names.values()) == {"blue", "gray", "black", "green", "red", "yellow"}
    assert set(car_distribution.brand_rv.xk) == set(car_distribution.brand_names)
    for brand in car_distribution.horsepower_rvs:
        assert set(car_distribution.horsepower_rvs[brand].xk) <= {100, 200, 300, 400, 500, 600, 700}
        assert set(car_distribution.color_rvs[brand].xk) <= set(car_distribution.color_names)


def test_car_distribution_is_immutable():
    car_distribution = create_car_distribution()

    # the frozen dataclass rejects attribute reassignment
    with pytest.raises(dataclasses.FrozenInstanceError):
        car_distribution.brand_rv = None


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


def test_joint_car_distribution_support_and_normalization():
    joint = create_joint_car_distribution()

    # 9 + 20 + 9 support states, non-negative, normalized, with the expected index
    assert len(joint) == 38
    assert (joint >= 0).all()
    assert joint.sum() == pytest.approx(1.0)
    assert joint.index.names == ["brand", "horsepower", "color"]
    assert set(joint.index.get_level_values("brand")) == {"VW", "Porsche", "Ferrari"}


def test_joint_car_distribution_matches_factorization():
    car_distribution = create_car_distribution()
    joint = create_joint_car_distribution()

    # spot checks of p(b, h, c) = p(b) p(h | b) p(c | b)
    expected_vw = car_distribution.brand_rv.pmf(VW) * car_distribution.horsepower_rvs[VW].pmf(100) * car_distribution.color_rvs[VW].pmf(BLUE)
    expected_porsche = (
        car_distribution.brand_rv.pmf(PORSCHE) * car_distribution.horsepower_rvs[PORSCHE].pmf(600) * car_distribution.color_rvs[PORSCHE].pmf(BLACK)
    )
    expected_ferrari = car_distribution.brand_rv.pmf(FERRARI) * car_distribution.horsepower_rvs[FERRARI].pmf(500) * car_distribution.color_rvs[FERRARI].pmf(RED)
    assert joint.loc[("VW", 100, "blue")] == pytest.approx(expected_vw)
    assert joint.loc[("Porsche", 600, "black")] == pytest.approx(expected_porsche)
    assert joint.loc[("Ferrari", 500, "red")] == pytest.approx(expected_ferrari)


def test_marginals_and_conditionals_recovered_from_joint():
    joint = create_joint_car_distribution()

    # marginalizing the joint over (horsepower, color) recovers the brand marginals
    brand_marginal = joint.groupby(level=0).sum()
    assert brand_marginal["VW"] == pytest.approx(0.5)
    assert brand_marginal["Porsche"] == pytest.approx(0.3)
    assert brand_marginal["Ferrari"] == pytest.approx(0.2)

    # normalizing within a brand recovers the conditional p(horsepower | brand)
    conditional = joint.xs("Porsche", level="brand").groupby(level=0).sum()
    conditional = conditional / conditional.sum()
    assert conditional.loc[600] == pytest.approx(0.05)
    assert conditional.loc[700] == pytest.approx(0.05)
