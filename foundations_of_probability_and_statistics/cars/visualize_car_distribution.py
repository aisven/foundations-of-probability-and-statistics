import matplotlib.pyplot as plt

from foundations_of_probability_and_statistics.cars.car_distribution import create_joint_car_distribution

# one colormap per brand to keep the brand hue while shading with horsepower
BRAND_COLORMAPS = {"VW": "Blues", "Porsche": "Oranges", "Ferrari": "Reds"}


def visualize_joint_car_distribution_in_histogram() -> plt.Figure:
    """Visualize the full joint car distribution as a histogram.

    Shows the probability of each of the 38 joint states
    p(brand, horsepower, color) = p(brand) * p(horsepower | brand) * p(color | brand)
    as one bar per state, with one hue per brand, shading from lighter
    to darker with horsepower.

    Returns:
        plt.Figure: The figure containing the histogram of the joint states.
    """
    joint = create_joint_car_distribution()

    # horsepower range per brand, for the lighter to darker shading
    brand_hp_ranges = {
        brand: (min(horsepower for b, horsepower, _ in joint.index if b == brand), max(horsepower for b, horsepower, _ in joint.index if b == brand))
        for brand in BRAND_COLORMAPS
    }

    def bar_color(brand: str, horsepower: int) -> tuple[float, float, float, float]:
        # fraction in [0.35, 0.85] so bars stay visible while getting darker with horsepower
        hp_min, hp_max = brand_hp_ranges[brand]
        fraction = 0.35 + 0.5 * (horsepower - hp_min) / (hp_max - hp_min)
        return plt.get_cmap(BRAND_COLORMAPS[brand])(fraction)

    # state labels in reading order, e.g. "VW, 100 hp, blue"
    labels = [f"{brand}, {horsepower} hp, {color}" for brand, horsepower, color in joint.index]

    # modern matplotlib usage style: subplots even for a single axis
    fig, ax = plt.subplots(figsize=(10.0, 10.0), dpi=100, layout="constrained")
    ax.bar(range(len(joint)), joint.values, color=[bar_color(brand, horsepower) for brand, horsepower, _ in joint.index])
    ax.set_xticks(range(len(joint)), labels, rotation=90)
    ax.set_ylabel("probability p(brand, horsepower, color)")
    ax.set_title("Full joint car distribution over all 38 states")
    return fig
