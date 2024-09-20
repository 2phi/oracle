# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


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
        [substratum_bottom, substratum_top], 0, 1, color='lightgrey', alpha=0.6
    )
    ax.text(
        0.5,
        (substratum_bottom + substratum_top) / 2 - 15,
        'substratum',
        ha='center',
        va='center',
        color='black',
        fontsize=10,
    )

    # Plot the weak layer at the bottom
    current_height = weaklayer_thickness
    weak_layer_top = weaklayer_thickness
    ax.axhline(0, color='grey', linestyle='-', linewidth=1)
    ax.axhline(weak_layer_top, color='grey', linestyle='-', linewidth=1)
    ax.fill_betweenx(
        [0, weak_layer_top], 0, 1, color='coral', alpha=0.3, hatch='x'
    )
    ax.text(
        0.5,
        -15,
        'weak layer',
        ha='center',
        va='center',
        color='coral',
        fontsize=10,
    )

    # Plot each layer from bottom to top
    for (density, thickness), grain in zip(
        reversed(layers), reversed(grain_list)
    ):
        layer_bottom = current_height
        layer_top = current_height + thickness

        # Determine color and hatch pattern based on grain type
        color = plt.cm.viridis(1 - density / 450)
        hatch = '//' if grain == 'mfc' else None

        # Fill the layer with color and optional hatch pattern
        ax.fill_betweenx(
            [layer_bottom, layer_top],
            0,
            1,
            color=color,
            alpha=0.6,
            hatch=hatch,
        )
        ax.axhline(layer_top, color='grey', linestyle='-', linewidth=1)

        # Annotate density in the middle of the layer
        ax.text(
            0.5,
            (layer_bottom + layer_top) / 2,
            f'{int(density)} kg/m³',
            ha='center',
            va='center',
            color='black',
            fontsize=10,
        )

        # Annotate grain type on the right side in the middle of the layer
        ax.text(
            1.1,
            (layer_bottom + layer_top) / 2,
            grain,
            ha='left',
            va='center',
            color='black',
            fontsize=10,
        )

        # Update the current height
        current_height = layer_top

    # Set axis limits and labels
    ax.set_ylim(
        substratum_bottom, max(total_height, 500)
    )  # Ensure y-axis starts at 500 mm or lower
    ax.set_xlim(
        0, 1.2
    )  # Adjust x-axis limit to make space for grain annotations
    ax.set_xticks([])  # No numbers on the x-axis
    ax.set_ylabel('mm')
    ax.set_title('Snow Stratification', fontsize=10)

    # Add grid for better readability
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

    # Return the figure and axis objects
    return fig, ax


def lognorm_pdf(
    data: pd.Series,
    bins: int = 75,
    range: tuple[float, float] = (0, 25),
    density: bool = True,
    histogram: bool = True,
    function: bool = True,
    zorder: int = 2,
):
    """
    Fit and plot the probability distribution function using a lognormal distribution.

    Parameters
    ----------
    data : pd.Series
        Dataset.
    bins : int, optional
        Number of bins. Default is 75.
    range : tuple[float, float], optional
        Range to plot. Default is (0, 25).
    density : bool, optional
        Wether to plot the density. Default is True.
    histogram : bool, optional
        Wether to plot the histogram. Default is True.
    function : bool, optional
        Wether to plot the probability distribution function. Default is True.
    zorder : int, optional
        Z-order for plotting. Default is 2.
    """

    # Get histogram data and bins
    hist_data, hist_bins = np.histogram(
        data, bins=bins, range=range, density=density
    )
    hist_x = (hist_bins[:-1] + hist_bins[1:]) / 2

    # Calculate the bin width and reduce it slightly to create a gap between bars
    bin_width = hist_bins[1] - hist_bins[0]
    bar_width = 0.75 * bin_width

    # Calc the probability density function
    x = np.linspace(min(data), min(max(data), range[1]), 1000)
    shape, loc, scale = stats.lognorm.fit(data)
    pdf_data = stats.lognorm.pdf(x, shape, loc, scale)

    # Plot
    if histogram:
        plt.bar(hist_x, hist_data, width=bar_width, color='w', zorder=zorder)
        plt.bar(hist_x, hist_data, width=bar_width, alpha=0.5, zorder=zorder)
    if function and density:
        plt.plot(x, pdf_data, color='w', lw=3, zorder=zorder)
        plt.plot(x, pdf_data, zorder=zorder)

    return plt.gca()


def lognorm_cdf(
    data: pd.Series,
    bins: int = 75,
    range: tuple[float, float] = (0, 25),
    density: bool = True,
    histogram: bool = True,
    function: bool = True,
    zorder: int = 1,
):
    """
    Fit and plot the cumulative distribution function using a lognormal distribution.

    Parameters
    ----------
    data : pd.Series
        Dataset.
    bins : int, optional
        Number of bins. Default is 75.
    range : tuple[float, float], optional
        Range to plot. Default is (0, 25).
    density : bool, optional
        Wether to plot the histogram. Default is True.
    histogram : bool, optional
        Wether to plot the histogram. Default is True.
    function : bool, optional
        Wether to plot the cumulative distribution function. Default is True.
    zorder : int, optional
        Z-order for plotting. Default is 1.
    """

    # Get histogram data and bins
    hist_data, hist_bins = np.histogram(
        data, bins=bins, range=range, density=density
    )
    hist_data = np.cumsum(hist_data) * np.diff(hist_bins)
    hist_x = (hist_bins[:-1] + hist_bins[1:]) / 2

    # Calculate the bin width and reduce it slightly to create a gap between bars
    bin_width = hist_bins[1] - hist_bins[0]
    bar_width = 0.75 * bin_width

    # Calc the cumulative distribution function
    x = np.linspace(min(data), min(max(data), range[1]), 1000)
    shape, loc, scale = stats.lognorm.fit(data)
    cdf_data = stats.lognorm.cdf(x, shape, loc, scale)

    # Plot
    if histogram:
        plt.bar(hist_x, hist_data, width=bar_width, color='w', zorder=zorder)
        plt.bar(hist_x, hist_data, width=bar_width, alpha=0.5, zorder=zorder)
    if function and density:
        plt.plot(x, cdf_data, color='w', lw=3, zorder=zorder)
        plt.plot(x, cdf_data, zorder=zorder)


def lognorm_distribution(
    data: pd.Series,
    kind: str = 'pdf',
    bins: int = 75,
    range: tuple[float, float] = (0, 25),
    density: bool = True,
    histogram: bool = True,
    function: bool = True,
    zorder: int | None = None,
):
    """
    Fit and plot the lognormal distribution (PDF or CDF) for the given data.

    Parameters
    ----------
    data : pd.Series
        Dataset to be analyzed.
    kind : str, optional
        Type of distribution to plot: 'pdf' or 'cdf'. Default is 'pdf'.
    bins : int, optional
        Number of bins for the histogram. Default is 75.
    range : tuple[float, float], optional
        Range for the histogram and plot. Default is (0, 25).
    density : bool, optional
        If True, the histogram is normalized to form a probability density.
        Default is True.
    histogram : bool, optional
        Whether to plot the histogram. Default is True.
    function : bool, optional
        Whether to plot the fitted distribution function (PDF or CDF).
        Default is True.
    zorder : int or None, optional
        The drawing order of plot elements. If None, defaults to 2 for 'pdf' and
        1 for 'cdf'. If provided, uses the given value.

    Raises
    ------
    ValueError
        If the 'kind' parameter is not 'pdf' or 'cdf'.

    Examples
    --------
    >>> data = pd.Series(np.random.lognormal(mean=1, sigma=0.5, size=1000))
    >>> lognorm_distribution(data, kind='pdf')
    >>> lognorm_distribution(data, kind='cdf')
    """

    # Set default zorder based on 'kind' if zorder is None
    if zorder is None:
        if kind == 'pdf':
            zorder = 2
        elif kind == 'cdf':
            zorder = 1
        else:
            raise ValueError(
                "Invalid 'kind' parameter. Must be 'pdf' or 'cdf'."
            )
    else:
        # Ensure zorder is an integer
        if not isinstance(zorder, int):
            raise TypeError("zorder must be an integer or None.")

    # Fit the lognormal distribution to the data
    shape, loc, scale = stats.lognorm.fit(data, floc=0)

    # Generate x values for plotting the function
    x = np.linspace(max(min(data), range[0]), min(max(data), range[1]), 1000)

    # Calculate the PDF or CDF based on the 'kind' parameter
    if kind == 'pdf':
        y_data = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
    elif kind == 'cdf':
        y_data = stats.lognorm.cdf(x, shape, loc=loc, scale=scale)
    else:
        raise ValueError("Invalid 'kind' parameter. Must be 'pdf' or 'cdf'.")

    # Get histogram data and bins
    hist_data, hist_bins = np.histogram(
        data, bins=bins, range=range, density=density
    )
    hist_x = (hist_bins[:-1] + hist_bins[1:]) / 2

    # For CDF, compute the cumulative sum of histogram data
    if kind == 'cdf':
        hist_data = np.cumsum(hist_data)
        if density:
            hist_data = (
                hist_data / hist_data[-1]
            )  # Normalize to 1 if density is True

    # Calculate the bin width and reduce it slightly to create a gap between bars
    bin_width = hist_bins[1] - hist_bins[0]
    bar_width = 0.75 * bin_width

    # Plot the histogram
    if histogram:
        plt.bar(hist_x, hist_data, width=bar_width, color='w', zorder=zorder)
        plt.bar(hist_x, hist_data, width=bar_width, alpha=0.5, zorder=zorder)

    # Plot the fitted distribution function
    if function and density:
        plt.plot(x, y_data, color='w', lw=3, zorder=zorder)
        plt.plot(x, y_data, zorder=zorder)

    # Label the axes
    plt.xlabel('Value')
    if kind == 'pdf':
        plt.ylabel('Probability Density')
    elif kind == 'cdf':
        plt.ylabel('Cumulative Probability')
