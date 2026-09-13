import pytest

from foundations_of_probability_and_statistics.cars.horsepower_moments import (
    conditional_horsepower_mean,
    conditional_horsepower_moments,
    conditional_horsepower_variance,
)


@pytest.mark.parametrize(
    ("brand", "expected_mean", "expected_variance"),
    [
        # E[H | B = VW] = 100 * 0.6 + 200 * 0.3 + 300 * 0.1 = 150
        # Var(H | B = VW) = 0.6 * 2500 + 0.3 * 2500 + 0.1 * 22500 = 4500
        ("VW", 150.0, 4500.0),
        # E[H | B = Porsche] = 300 * 0.4 + 400 * 0.4 + 500 * 0.1 + 600 * 0.05 + 700 * 0.05 = 395
        # Var(H | B = Porsche) = 167500 - 395^2 = 11475
        ("Porsche", 395.0, 11475.0),
        # E[H | B = Ferrari] = 400 * 0.3 + 500 * 0.4 + 600 * 0.3 = 500
        # Var(H | B = Ferrari) = 256000 - 500^2 = 6000
        ("Ferrari", 500.0, 6000.0),
    ],
)
def test_conditional_horsepower_moments(brand: str, expected_mean: float, expected_variance: float):
    assert conditional_horsepower_mean(brand) == pytest.approx(expected_mean)
    assert conditional_horsepower_variance(brand) == pytest.approx(expected_variance)
    assert conditional_horsepower_moments(brand) == pytest.approx((expected_mean, expected_variance))


def test_conditional_horsepower_moments_unknown_brand():
    # unknown brands fail with a readable error instead of a KeyError later
    with pytest.raises(ValueError, match="Unknown brand"):
        conditional_horsepower_mean("Fiat")
    with pytest.raises(ValueError, match="Unknown brand"):
        conditional_horsepower_variance("Fiat")
