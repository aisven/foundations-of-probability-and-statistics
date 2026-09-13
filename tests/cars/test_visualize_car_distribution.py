import matplotlib
import matplotlib.pyplot as plt
import pytest

from foundations_of_probability_and_statistics.cars.visualize_car_distribution import visualize_joint_car_distribution_in_histogram

# non-interactive backend for testing
matplotlib.use("Agg")


def test_visualize_joint_car_distribution_in_histogram():
    figure = visualize_joint_car_distribution_in_histogram()

    # the figure contains exactly one subplot with one bar per joint state
    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) == 1
    ax = figure.axes[0]
    assert len(ax.patches) == 38

    # the bar heights are the joint probabilities and sum to one
    heights = [patch.get_height() for patch in ax.patches]
    assert sum(heights) == pytest.approx(1.0)

    # the labels cover all brands and all horsepower values
    labels = [label.get_text() for label in ax.get_xticklabels()]
    assert {label.split(",")[0] for label in labels} == {"VW", "Porsche", "Ferrari"}
    assert {label.split(" ")[1] for label in labels} == {"100", "200", "300", "400", "500", "600", "700"}

    plt.close(figure)
