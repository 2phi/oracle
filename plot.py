# Third-party imports
import numpy as np
import matplotlib.pyplot as plt


def density_vs_hand_hardness(grainform_df):
    """
    Plots the density of different grain types versus hand hardness and returns the figure and axis objects.

    Parameters:
    grainform_df (pd.DataFrame): A DataFrame containing the following columns:
        - 'id': Numeric identifier for each grain type.
        - 'type': Name of the grain type.
        - 'abbreviation': Abbreviation for the grain type.
        - 'a': Coefficient 'a' for the density formula.
        - 'b': Coefficient 'b' for the density formula.

    The density is calculated using the following formulas:
    - For most grain types: density = a + b * hand_hardness
    - Except 'Rounded grains' (abbreviation = 'RG'): density = a + b * (hand_hardness ^ 3.15)

    The plot will display:
    - X-axis: Hand hardness ranging from 1 to 5.
    - Y-axis: Density ranging from 50 to 450 kg/m³.
    - Each grain type is plotted with a different color and marker.

    Returns:
    fig (matplotlib.figure.Figure): The Figure object containing the plot.
    ax (matplotlib.axes.Axes): The Axes object containing the plot.
    """
    # Define hand hardness values
    hand_hardness = np.arange(1, 6, 0.1)

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 5))

    # Iterate over each grain type to compute and plot densities
    for index, row in grainform_df.iterrows():
        a = row["a"]
        b = row["b"]
        if row["abbreviation"] == "RG":  # exponential case for Rounded grains
            densities = a + b * (hand_hardness**3.15)
        else:
            densities = a + b * hand_hardness

        ax.plot(hand_hardness, densities, label=row["abbreviation"])

    # Set plot limits and labels
    ax.set_ylim(50, 450)
    ax.set_xlim(1, 5)
    ax.set_xlabel("Hand Hardness")
    ax.set_ylabel("Density (kg/m³)")
    ax.set_title("Density vs Hand Hardness for Different Grain Types")
    ax.legend(loc="best")

    # Add grid for better readability
    ax.grid(True)

    # Return the figure and axis objects
    return fig, ax


def snow_stratification(weaklayer_thickness, layers, grain_list):
    """
    Plots snow stratification with weak layer highlighted and grain types annotated.

    Parameters:
    weaklayer_thickness (int): Thickness of the weak layer in mm.
    layers (list of list): 2D list where each sublist contains [density, thickness] for each snow layer.
    grain_list (list of str): List of grain types corresponding to each layer.

    Returns:
    fig (matplotlib.figure.Figure): The Figure object containing the plot.
    ax (matplotlib.axes.Axes): The Axes object containing the plot.
    """
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Define substratum thickness and position
    substratum_thickness = 100
    substratum_bottom = -substratum_thickness
    substratum_top = 0

    # Calculate total height of all layers
    total_height = weaklayer_thickness + sum(
        thickness for _, thickness in layers
    )

    # Plot the substratum
    ax.fill_betweenx(
        [substratum_bottom, substratum_top], 0, 1, color="lightgrey", alpha=0.6
    )
    ax.text(
        0.5,
        (substratum_bottom + substratum_top) / 2 - 15,
        "substratum",
        ha="center",
        va="center",
        color="black",
        fontsize=10,
    )
    return fig, ax