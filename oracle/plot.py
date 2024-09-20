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
    layers (list of list): 3D list where each sublist contains [density, thickness, hand hardness] for each snow layer.
    grain_list (list of str): List of grain types corresponding to each layer.

    Returns:
    fig (matplotlib.figure.Figure): The Figure object containing the plot.
    ax (matplotlib.axes.Axes): The Axes object containing the plot.
    """

    # Initialize figure and axis and color used for density-plots
    fig, ax = plt.subplots(figsize=(10, 5))
    x_max = 550  # Defining max of x-axis (density)
    medium_blue = plt.cm.Blues(0.5)
    dark_blue = plt.cm.Blues(0.99)
    previous_density = 0  # Help variable to plot outline of density-plot

    # Initialize hardness map to translate input values to traditional index
    hardness_mapping = {1: "F", 2: "4F", 3: "1F", 4: "P", 5: "K"}

    # Defining help variables used for table-plot on RHS of graph
    current_table = weaklayer_thickness
    first_column_start = (-0.7) * 100
    second_column_start = (-1.9) * 100
    third_column_start = (-2.4) * 100
    third_column_end = (-2.8) * 100

    # Midpoints of vertical column borders defined above
    first_column_midpoint = (first_column_start + second_column_start) / 2
    second_column_midpoint = (third_column_start + second_column_start) / 2
    third_column_midpoint = (third_column_end + third_column_start) / 2

    # Calculate total height of all layers
    total_height = weaklayer_thickness + sum(
        thickness for _, thickness, _ in layers
    )

    # Defining y_max and column header height
    y_max = max(total_height, 500) * 1.15
    column_header = y_max / 1.1

    # Average height of layers used for plot of columns
    avg_height = (column_header - weaklayer_thickness) / max(1, (len(layers)))

    # Define substratum thickness and position
    substratum_thickness = 40
    substratum_bottom = -substratum_thickness
    substratum_top = 0

    # Plot the substratum and annotate text
    ax.fill_betweenx(
        [substratum_bottom, substratum_top], 0, x_max, color=dark_blue, alpha=1
    )
    ax.text(
        250,
        (substratum_bottom + substratum_top) / 2,
        'substratum',
        ha='center',
        va='center',
        color='white',
        fontsize=8,
    )

    # Plot the weak layer at the bottom
    current_height = weaklayer_thickness
    weak_layer_top = weaklayer_thickness

    if len(layers) > 0:
        ax.fill_betweenx(
            [0, weak_layer_top],
            0,
            (layers[0][0]) / 2,
            color='coral',
            alpha=0.3,
            hatch='x',
        )
        ax.text(
            layers[0][0],
            weaklayer_thickness / 2,
            'weak layer',
            ha='right',
            va='center',
            color='coral',
            fontsize=8,
        )
    else:
        ax.fill_betweenx(
            [0, weak_layer_top], 0, x_max, color='coral', alpha=0.3, hatch='x'
        )
        ax.text(
            250,
            weaklayer_thickness / 2,
            'weak layer',
            ha='center',
            va='center',
            color='coral',
            fontsize=8,
        )

    # Loop to plot each layer from bottom to top
    for (density, thickness, hand_hardness), grain in zip(layers, grain_list):

        # Plot of layers in hand_hardness graph
        layer_bottom = current_height
        layer_top = current_height + thickness

        # Plot of table (adding set height of 50 for each column)
        table_bottom = current_table
        table_top = current_table + min(avg_height, 50)

        # Determine color and hatch pattern based on grain type
        color = plt.cm.Blues(0.25)
        hatch = '//' if grain == 'mfc' else None

        # Plotting density on x-axis
        ax.fill_betweenx(
            [layer_bottom + 1, layer_top],
            0,
            density,
            color=color,
            alpha=0.8,
            hatch=hatch,
            zorder=1,
        )

        # Plotting outline of density plot
        ax.plot(
            [density, density],
            [layer_bottom + 1, layer_top],
            color=dark_blue,
            linestyle='-',
            linewidth=1,
        )
        ax.plot(
            [previous_density, density],
            [layer_bottom, layer_bottom],
            color=dark_blue,
            linestyle='-',
            linewidth=1,
        )
        previous_density = density

        # Manually plotting y-axis ticks
        ax.plot(
            [0, -10],
            [layer_bottom, layer_bottom],
            color='black',
            linestyle='-',
            linewidth=0.5,
        )
        ax.text(
            (-12),
            layer_bottom,
            round(layer_bottom / 10),
            ha='left',
            va='center',
            color='black',
            fontsize=7,
        )

        # Plotting data legend columns
        ax.plot(
            [first_column_start, third_column_end],
            [table_bottom, table_bottom],
            color='grey',
            linestyle='dotted',
            linewidth=0.5,
        )

        # Annotate density in the 1st column
        ax.text(
            first_column_midpoint,
            (table_bottom + table_top) / 2,
            round(density),
            ha='center',
            va='center',
            color='black',
            fontsize=8,
        )

        # Annotate grain type in the 2nd column
        ax.text(
            second_column_midpoint,
            (table_bottom + table_top) / 2,
            grain,
            ha='center',
            va='center',
            color='black',
            fontsize=8,
        )

        # Annotate hand_hardness in 3rd column
        ax.text(
            third_column_midpoint,
            (table_bottom + table_top) / 2,
            hardness_mapping.get(hand_hardness, "Unknown hardness"),
            ha='center',
            va='center',
            color='black',
            fontsize=8,
        )

        # Linking hand_hardness layers to table
        ax.plot(
            [0, first_column_start],
            [layer_bottom, table_bottom],
            color='grey',
            linestyle='dotted',
            linewidth=0.25,
        )
        ax.plot(
            [0, first_column_start],
            [layer_top, table_top],
            color='grey',
            linestyle='dotted',
            linewidth=0.25,
        )

        # Update the current height and table
        current_height = layer_top
        current_table = table_top

    ### Loop over layers is finished ###

    # Plotting final tick at max height
    ax.plot(
        [0, -10],
        [total_height, total_height],
        color='black',
        linestyle='-',
        linewidth=0.5,
    )
    ax.text(
        (-12),
        total_height,
        round(total_height / 10),
        ha='left',
        va='center',
        color='black',
        fontsize=7,
    )

    # Drawing final contour-line of density plot
    ax.plot(
        [previous_density, 0],
        [total_height, total_height],
        color=dark_blue,
        linestyle='-',
        linewidth=1,
    )

    # Y-axis adjustments
    # Manually plotting grid-lines
    ax.set_ylim(substratum_bottom, y_max)
    y_grid = np.arange(0, column_header, 100)
    for y in y_grid:
        ax.plot(
            [0, x_max],
            [y, y],
            color='grey',
            linestyle='--',
            linewidth=0.5,
            zorder=0,
        )
    y_tick_positions = y_grid
    y_tick_labels = [
        pos // 10 for pos in y_tick_positions
    ]  # adjusting labels to cm
    plt.yticks(ticks=y_tick_positions, labels=y_tick_labels)
    ax.set_ylabel('Height (cm)')

    # X-axis adjustments
    # Inverting, aligning at top and coloring in blue
    ax.set_xlim(third_column_end, x_max)
    ax.invert_xaxis()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    x_ticks = [100, 200, 300, 400, 500]
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', colors=medium_blue, direction='in', pad=-15)
    title_position = 0.35  # Normalized position on positive axis
    ax.set_xlabel('Density (kg/m³)', x=title_position, color=medium_blue)

    # Data table-adjustments
    # Plotting table columns and annotating titles
    ax.plot(
        [0, 0],
        [substratum_bottom, y_max],
        color='black',
        linestyle='-',
        linewidth=1,
    )
    ax.plot(
        [first_column_start, first_column_start],
        [weaklayer_thickness, y_max],
        color='grey',
        linestyle='dotted',
        linewidth=0.5,
    )
    ax.plot(
        [second_column_start, second_column_start],
        [weaklayer_thickness, y_max],
        color='grey',
        linestyle='dotted',
        linewidth=0.5,
    )
    ax.plot(
        [third_column_start, third_column_start],
        [weaklayer_thickness, y_max],
        color='grey',
        linestyle='dotted',
        linewidth=0.5,
    )
    ax.plot(
        [0, third_column_end],
        [column_header, column_header],
        color='grey',
        linestyle='dotted',
        linewidth=0.5,
    )
    ax.text(
        first_column_start / 2,
        (y_max + column_header) / 2,
        "H (cm)",
        ha='center',
        va='center',
        color='black',
        fontsize=9,
    )
    ax.text(
        first_column_midpoint,
        (y_max + column_header) / 2,
        "Density (kg/m³)",
        ha='center',
        va='center',
        color='black',
        fontsize=9,
    )
    ax.text(
        second_column_midpoint,
        (y_max + column_header) / 2,
        "GF",
        ha='center',
        va='center',
        color='black',
        fontsize=9,
    )
    ax.text(
        third_column_midpoint,
        (y_max + column_header) / 2,
        "R",
        ha='center',
        va='center',
        color='black',
        fontsize=9,
    )

    # Title of plot
    ax.set_title('Snow Stratification', fontsize=14)

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
