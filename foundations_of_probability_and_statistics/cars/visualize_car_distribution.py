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


def visualize_pmf_given_brand(brand: str) -> plt.Figure:
    """Visualize the conditional PMF of horsepower and color given a brand as a tile heatmap.

    Conditions the joint on the brand via
    p(horsepower, color | brand) = p(brand, horsepower, color) / p(brand),
    showing one tile per (horsepower, color) combination with the probability annotated.

    Args:
        brand: Brand to condition on, one of VW, Porsche or Ferrari.

    Returns:
        plt.Figure: The figure containing the tile heatmap of the conditional PMF.
    """
    # unknown brands fail here with a readable error instead of a KeyError later
    if brand not in BRAND_COLORMAPS:
        raise ValueError(f"Unknown brand '{brand}', must be one of {sorted(BRAND_COLORMAPS)}.")

    joint = create_joint_car_distribution()
    # condition on the brand: p(h, c | b) = p(b, h, c) / p(b)
    conditional = joint.xs(brand, level="brand")
    conditional = (conditional / conditional.sum()).unstack("color", fill_value=0.0)
    # sort the horsepower rows in descending order so more hp tiles sit higher up
    conditional = conditional.sort_index(ascending=False)

    # modern matplotlib usage style: subplots even for a single axis
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=100, layout="constrained")
    # sequential colormap per brand, scaled per brand so the probabilities remain readable
    image = ax.imshow(conditional.values, cmap=BRAND_COLORMAPS[brand], vmin=0.0, vmax=conditional.values.max())
    ax.set_xticks(range(len(conditional.columns)), conditional.columns)
    ax.set_yticks(range(len(conditional.index)), conditional.index)
    ax.set_xlabel("color")
    ax.set_ylabel("horsepower")
    ax.set_title(f"Conditional PMF p(horsepower, color | {brand})")
    # annotate each tile with its probability, white text on dark tiles
    for i, horsepower in enumerate(conditional.index):
        for j, color in enumerate(conditional.columns):
            value = conditional.loc[horsepower, color]
            text_color = "white" if value > 0.5 * conditional.values.max() else "black"
            ax.text(j, i, f"{value:.4f}", ha="center", va="center", color=text_color)
    fig.colorbar(image, ax=ax, label="p(horsepower, color | brand)")
    return fig
