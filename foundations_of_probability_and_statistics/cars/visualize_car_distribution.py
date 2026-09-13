import matplotlib.pyplot as plt

from foundations_of_probability_and_statistics.cars.car_distribution import create_joint_car_distribution


def visualize_joint_car_distribution_in_histogram() -> plt.Figure:
    """Visualize the full joint car distribution as a histogram.

    Shows the probability of each of the 38 joint states
    p(brand, horsepower, color) = p(brand) * p(horsepower | brand) * p(color | brand)
    as one bar per state, with one color per brand.

    Returns:
        plt.Figure: The figure containing the histogram of the joint states.
    """
    joint = create_joint_car_distribution()

    # one color per brand to make the blocks visually separable
    brand_colors = {"VW": "tab:blue", "Porsche": "tab:orange", "Ferrari": "tab:red"}
    # state labels in reading order, e.g. "VW, 100 hp, blue"
    labels = [f"{brand}, {horsepower} hp, {color}" for brand, horsepower, color in joint.index]

    # modern matplotlib usage style: subplots even for a single axis
    fig, ax = plt.subplots(figsize=(10.0, 10.0), dpi=100, layout="constrained")
    ax.bar(range(len(joint)), joint.values, color=[brand_colors[brand] for brand, _, _ in joint.index])
    ax.set_xticks(range(len(joint)), labels, rotation=90)
    ax.set_ylabel("probability p(brand, horsepower, color)")
    ax.set_title("Full joint car distribution over all 38 states")
    return fig
