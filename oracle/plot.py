# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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


# Function to plot snow stratification
def snow_stratification(weaklayer_thickness, layers, grain_list):
    fig, ax = plt.subplots(figsize=(10, 5))
    x_max = 550
    medium_blue = plt.cm.Blues(0.5)
    dark_blue = plt.cm.Blues(0.99)
    previous_density = 0
    hardness_mapping = {1: "F", 2: "4F", 3: "1F", 4: "P", 5: "K"}

    current_table = weaklayer_thickness
    first_column_start = -0.7 * 100
    second_column_start = -1.9 * 100
    third_column_start = -2.4 * 100
    third_column_end = -2.8 * 100

    first_column_midpoint = (first_column_start + second_column_start) / 2
    second_column_midpoint = (third_column_start + second_column_start) / 2
    third_column_midpoint = (third_column_end + third_column_start) / 2

    total_height = weaklayer_thickness + sum(
        thickness for _, thickness, _ in layers
    )
    y_max = max(total_height, 500) * 1.15
    column_header = y_max / 1.1
    avg_height = (column_header - weaklayer_thickness) / max(1, (len(layers)))
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
        "substratum",
        ha="center",
        va="center",
        color="white",
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
            color="coral",
            alpha=0.3,
            hatch="x",
        )
        ax.text(
            layers[0][0],
            weaklayer_thickness / 2,
            "weak layer",
            ha="right",
            va="center",
            color="coral",
            fontsize=8,
        )
    else:
        ax.fill_betweenx(
            [0, weak_layer_top], 0, x_max, color="coral", alpha=0.3, hatch="x"
        )
        ax.text(
            250,
            weaklayer_thickness / 2,
            "weak layer",
            ha="center",
            va="center",
            color="coral",
            fontsize=8,
        )

    # Loop to plot each layer from bottom to top
    for (density, thickness, hand_hardness), grain in zip(layers, grain_list):
        layer_bottom = current_height
        layer_top = current_height + thickness
        table_bottom = current_table
        table_top = current_table + min(avg_height, 50)
        color = plt.cm.Blues(0.25)
        hatch = "//" if grain == "mfc" else None

        ax.fill_betweenx(
            [layer_bottom + 1, layer_top],
            0,
            density,
            color=color,
            alpha=0.8,
            hatch=hatch,
            zorder=1,
        )
        ax.plot(
            [density, density],
            [layer_bottom + 1, layer_top],
            color=dark_blue,
            linestyle="-",
            linewidth=1,
        )
        ax.plot(
            [previous_density, density],
            [layer_bottom, layer_bottom],
            color=dark_blue,
            linestyle="-",
            linewidth=1,
        )
        previous_density = density

        ax.plot(
            [0, -10],
            [layer_bottom, layer_bottom],
            color="black",
            linestyle="-",
            linewidth=0.5,
        )
        ax.text(
            -12,
            layer_bottom,
            round(layer_bottom / 10),
            ha="left",
            va="center",
            color="black",
            fontsize=7,
        )

        ax.plot(
            [first_column_start, third_column_end],
            [table_bottom, table_bottom],
            color="grey",
            linestyle="dotted",
            linewidth=0.5,
        )

        ax.text(
            first_column_midpoint,
            (table_bottom + table_top) / 2,
            round(density),
            ha="center",
            va="center",
            color="black",
            fontsize=8,
        )

        ax.text(
            second_column_midpoint,
            (table_bottom + table_top) / 2,
            grain,
            ha="center",
            va="center",
            color="black",
            fontsize=8,
        )

        ax.text(
            third_column_midpoint,
            (table_bottom + table_top) / 2,
            hardness_mapping.get(hand_hardness, "Unknown hardness"),
            ha="center",
            va="center",
            color="black",
            fontsize=8,
        )

        ax.plot(
            [0, first_column_start],
            [layer_bottom, table_bottom],
            color="grey",
            linestyle="dotted",
            linewidth=0.25,
        )
        ax.plot(
            [0, first_column_start],
            [layer_top, table_top],
            color="grey",
            linestyle="dotted",
            linewidth=0.25,
        )

        current_height = layer_top
        current_table = table_top

    ax.plot(
        [0, -10],
        [total_height, total_height],
        color="black",
        linestyle="-",
        linewidth=0.5,
    )
    ax.text(
        -12,
        total_height,
        round(total_height / 10),
        ha="left",
        va="center",
        color="black",
        fontsize=7,
    )
    ax.plot(
        [previous_density, 0],
        [total_height, total_height],
        color=dark_blue,
        linestyle="-",
        linewidth=1,
    )

    ax.set_ylim(substratum_bottom, y_max)
    y_grid = np.arange(0, column_header, 100)
    for y in y_grid:
        ax.plot(
            [0, x_max],
            [y, y],
            color="grey",
            linestyle="--",
            linewidth=0.5,
            zorder=0,
        )
    y_tick_positions = y_grid
    y_tick_labels = [pos // 10 for pos in y_tick_positions]
    plt.yticks(ticks=y_tick_positions, labels=y_tick_labels)
    ax.set_ylabel("Height (cm)")

    ax.set_xlim(third_column_end, x_max)
    ax.invert_xaxis()
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    x_ticks = [100, 200, 300, 400, 500]
    ax.set_xticks(x_ticks)
    ax.tick_params(axis="x", colors=medium_blue, direction="in", pad=-15)
    title_position = 0.35
    ax.set_xlabel("Density (kg/m³)", x=title_position, color=medium_blue)

    ax.plot(
        [0, 0],
        [substratum_bottom, y_max],
        color="black",
        linestyle="-",
        linewidth=1,
    )
    ax.plot(
        [first_column_start, first_column_start],
        [weaklayer_thickness, y_max],
        color="grey",
        linestyle="dotted",
        linewidth=0.5,
    )
    ax.plot(
        [second_column_start, second_column_start],
        [weaklayer_thickness, y_max],
        color="grey",
        linestyle="dotted",
        linewidth=0.5,
    )
    ax.plot(
        [third_column_start, third_column_start],
        [weaklayer_thickness, y_max],
        color="grey",
        linestyle="dotted",
        linewidth=0.5,
    )
    ax.plot(
        [0, third_column_end],
        [column_header, column_header],
        color="grey",
        linestyle="dotted",
        linewidth=0.5,
    )
    ax.text(
        first_column_start / 2,
        (y_max + column_header) / 2,
        "H (cm)",
        ha="center",
        va="center",
        color="black",
        fontsize=9,
    )
    ax.text(
        first_column_midpoint,
        (y_max + column_header) / 2,
        "Density (kg/m³)",
        ha="center",
        va="center",
        color="black",
        fontsize=9,
    )
    ax.text(
        second_column_midpoint,
        (y_max + column_header) / 2,
        "GF",
        ha="center",
        va="center",
        color="black",
        fontsize=9,
    )
    ax.text(
        third_column_midpoint,
        (y_max + column_header) / 2,
        "R",
        ha="center",
        va="center",
        color="black",
        fontsize=9,
    )

    return fig


def distribution(
    data: pd.Series,
    kind: str = 'pdf',
    bins: int = 75,
    range: tuple[float, float] = (0, 25),
    fit_to_range: bool = False,
    density: bool = True,
    histogram: bool = True,
    function: bool = True,
    zorder: int | None = None,
    log: bool = False,
    dist_type: str = 'lognorm',
):
    """
    Fit and plot the specified distribution (PDF or CDF) for the given data.

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
    fit_to_range : bool, optional
        If True, filters data to be within the specified range. Default is False.
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
    log : bool, optional
        If True, plots with logarithmically spaced x-axes. Default is False.
    dist_type : str, optional
        Type of distribution to fit and plot: 'lognorm', 'cauchy', 'chi2', or 'expon'.
        Default is 'lognorm'.

    Raises
    ------
    ValueError
        If the 'kind' parameter is not 'pdf' or 'cdf'.
    ValueError
        If the 'dist_type' parameter is not 'lognorm', 'cauchy', 'chi2', or 'expon'.
    TypeError
        If zorder is not an integer or None.

    Examples
    --------
    >>> data = pd.Series(np.random.lognormal(mean=1, sigma=0.5, size=1000))
    >>> lognorm_distribution(data, kind='pdf', log=True, dist_type='lognorm')
    >>> lognorm_distribution(data, kind='cdf', log=True, dist_type='cauchy')
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

    # Unpack range
    x_min = 1e-3 if log and range[0] <= 0 else range[0]
    x_max = range[1]

    # Filter data if necessary
    if fit_to_range:
        data = data[(data >= x_min) & (data <= x_max)]

    # Fit the specified distribution to the data
    if dist_type == 'lognorm':
        dist = stats.lognorm
        params = dist.fit(data)
        shape, loc, scale = params
        args = (shape,)
        kwargs = {'loc': loc, 'scale': scale}
    elif dist_type == 'cauchy':
        dist = stats.cauchy
        params = dist.fit(data)
        loc, scale = params
        args = ()
        kwargs = {'loc': loc, 'scale': scale}
    elif dist_type == 'expon':
        dist = stats.expon
        params = dist.fit(data)
        loc, scale = params
        args = ()
        kwargs = {'loc': loc, 'scale': scale}
    elif dist_type == 'chi2':
        dist = stats.chi2
        params = dist.fit(data)
        df, loc, scale = params
        args = (df,)
        kwargs = {'loc': loc, 'scale': scale}
    else:
        raise ValueError(
            "Invalid 'dist_type' parameter. Must be 'lognorm', 'cauchy', 'chi2', or 'expon'."
        )

    # Generate bin edges
    if log:
        bins_edges = np.logspace(np.log10(x_min), np.log10(x_max), bins + 1)
    else:
        bins_edges = np.linspace(x_min, x_max, bins + 1)

    # Compute the histogram
    hist_data, hist_bins = np.histogram(data, bins=bins_edges, density=density)

    # For CDF, compute the cumulative sum of histogram data
    if kind == 'cdf':
        # Multiply by bin widths to get probability masses
        hist_data = np.cumsum(hist_data * np.diff(hist_bins))

    # Calculate bin widths
    bar_widths = 0.7 * np.diff(hist_bins)

    # Plot the histogram
    if histogram:
        plt.bar(
            hist_bins[:-1],
            hist_data,
            width=bar_widths,
            color='w',
            zorder=zorder,
            align='center',
        )
        plt.bar(
            hist_bins[:-1],
            hist_data,
            width=bar_widths,
            alpha=0.5,
            zorder=zorder,
            align='center',
        )

    # Generate x values for plotting the function
    if log:
        x = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
    else:
        x = np.linspace(x_min, x_max, 1000)

    # Calculate the PDF or CDF based on the 'kind' parameter
    if kind == 'pdf':
        y_data = dist.pdf(x, *args, **kwargs)
    elif kind == 'cdf':
        y_data = dist.cdf(x, *args, **kwargs)
    else:
        raise ValueError("Invalid 'kind' parameter. Must be 'pdf' or 'cdf'.")

    # Plot the fitted distribution function
    if function and density:
        if not histogram:
            plt.fill_between(x, y_data, zorder=zorder, alpha=0.8, color='w')
            plt.fill_between(x, y_data, zorder=zorder, alpha=0.2)
        plt.plot(x, y_data, color='w', lw=3, zorder=zorder)
        plt.plot(x, y_data, zorder=zorder)

    # Set the x-axis to logarithmic scale if log=True
    if log:
        plt.xscale('log')


def snow_stratification_plotly(weaklayer_thickness, layers, grain_list):

    medium_blue = "rgba(115, 170, 220, .4)"
    dark_blue = "rgba(8, 48, 107, 1.0)"

    hardness_mapping = {1: "F", 2: "4F", 3: "1F", 4: "P", 5: "K"}
    total_height = weaklayer_thickness + sum(
        thickness for _, thickness, _ in layers
    )
    y_max = max(total_height, 500) * 1.15
    substratum_thickness = 40
    substratum_bottom = -substratum_thickness
    substratum_top = 0
    x_max = 550  # Maximum density value for x-axis

    # Initialize figure
    fig = go.Figure()

    # Plot the substratum (background layer)
    fig.add_shape(
        type="rect",
        x0=-x_max,
        x1=0,
        y0=substratum_bottom,
        y1=substratum_top,
        fillcolor="rgba(8, 48, 107, .9)",
        line=dict(width=0, color="rgba(8, 48, 107, .9)"),
        # layer='below',  # Ensure it's in the background
    )

    # Add substratum text
    fig.add_annotation(
        x=-x_max / 2,
        y=(substratum_bottom + substratum_top) / 2,
        text="substratum",
        showarrow=False,
        font=dict(color='white', size=12),
        xanchor='center',
        yanchor='middle',
    )

    # Plot the weak layer
    if len(layers) > 0:
        weak_density = layers[0][0] / 2
    else:
        weak_density = x_max

    # Plot weak layer from x=-weak_density to x=0
    fig.add_shape(
        type="rect",
        x0=-weak_density,  # Negative x-values
        x1=0,
        y0=0,
        y1=weaklayer_thickness,
        fillcolor="rgba(255, 127, 80, .7)",
        line=dict(width=1, color="rgba(255, 127, 80, 1)"),
        layer='below',  # Ensure it's above gridlines
    )

    # Add weak layer text
    fig.add_annotation(
        x=-(weak_density if len(layers) > 0 else x_max / 2),
        y=weaklayer_thickness / 2,
        text=" weak layer",
        showarrow=False,
        font=dict(color='white', size=12),
        xanchor='left',
        yanchor='middle',
    )

    # Initialize variables for layers
    current_height = weaklayer_thickness
    previous_density = 0  # Start from zero density

    # Positions for annotations (positive x-values for table area)
    first_column_start = 0.8 * 100
    second_column_start = 2.2 * 100
    third_column_start = 2.8 * 100
    third_column_end = 3.2 * 100

    first_column_midpoint = (first_column_start + second_column_start) / 2
    second_column_midpoint = (second_column_start + third_column_start) / 2
    third_column_midpoint = (third_column_start + third_column_end) / 2

    column_header = y_max / 1.07
    avg_height = (column_header - weaklayer_thickness) / max(1, len(layers))
    current_table = weaklayer_thickness

    # Loop through each layer
    for i, ((density, thickness, hand_hardness), grain) in enumerate(
        zip(layers, grain_list)
    ):
        layer_bottom = current_height
        layer_top = current_height + thickness
        table_bottom = current_table
        table_top = current_table + min(avg_height, 50)

        # Plot the layer from x=-density to x=0
        fig.add_shape(
            type="rect",
            x0=-density,
            x1=0,
            y0=layer_bottom + 1,
            y1=layer_top,
            fillcolor=medium_blue,
            line=dict(width=0.4, color=medium_blue),
            layer='above',  # Ensure bars are above gridlines
        )

        # Line from previous_density to current density at layer_bottom
        fig.add_shape(
            type="line",
            x0=-previous_density,
            y0=layer_bottom,
            x1=-density,
            y1=layer_bottom,
            line=dict(color=dark_blue, width=1.2),
        )

        # Vertical line at current density
        fig.add_shape(
            type="line",
            x0=-density,
            y0=layer_bottom + 1,
            x1=-density,
            y1=layer_top,
            line=dict(color=dark_blue, width=1.2),
        )

        # Horizontal line at layer_bottom (height markers on the left)
        fig.add_shape(
            type="line",
            x0=0,
            y0=layer_bottom,
            x1=10,
            y1=layer_bottom,
            line=dict(color="black", width=0.5),
        )

        # Text for height at layer_bottom
        fig.add_annotation(
            x=12,
            y=layer_bottom,
            text=str(round(layer_bottom / 10)),
            showarrow=False,
            font=dict(color='black', size=10),
            xanchor='left',
            yanchor='middle',
        )

        # Line across the table columns
        fig.add_shape(
            type="line",
            x0=first_column_start,
            y0=table_bottom,
            x1=third_column_end,
            y1=table_bottom,
            line=dict(color="lightgrey", width=0.5),
        )

        # Annotations for density, grain form, and hardness
        fig.add_annotation(
            x=first_column_midpoint,
            y=(table_bottom + table_top) / 2,
            text=str(round(density)),
            showarrow=False,
            font=dict(color='black', size=10),
            xanchor='center',
            yanchor='middle',
        )

        fig.add_annotation(
            x=second_column_midpoint,
            y=(table_bottom + table_top) / 2,
            text=grain,
            showarrow=False,
            font=dict(color='black', size=10),
            xanchor='center',
            yanchor='middle',
        )

        fig.add_annotation(
            x=third_column_midpoint,
            y=(table_bottom + table_top) / 2,
            text=hardness_mapping.get(hand_hardness, "?"),
            showarrow=False,
            font=dict(color='black', size=10),
            xanchor='center',
            yanchor='middle',
        )

        # Dotted lines from layer edges to table
        fig.add_shape(
            type="line",
            x0=0,
            y0=layer_bottom,
            x1=first_column_start,
            y1=table_bottom,
            line=dict(color="lightgrey", width=0.25),
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=layer_top,
            x1=first_column_start,
            y1=table_top,
            line=dict(color="lightgrey", width=0.25),
        )

        previous_density = density
        current_height = layer_top
        current_table = table_top

    # Top layer horizontal line and height annotation
    fig.add_shape(
        type="line",
        x0=0,
        y0=total_height,
        x1=10,
        y1=total_height,
        line=dict(color="black", width=0.5),
    )
    fig.add_annotation(
        x=12,
        y=total_height,
        text=str(round(total_height / 10)),
        showarrow=False,
        font=dict(color='black', size=10),
        xanchor='left',
        yanchor='middle',
    )

    # Line from previous_density to x=0 at total_height
    fig.add_shape(
        type="line",
        x0=-previous_density,
        y0=total_height,
        x1=0,
        y1=total_height,
        line=dict(color=dark_blue, width=1),
    )

    # Set axes properties
    fig.update_layout(
        yaxis=dict(range=[substratum_bottom, y_max]),
        xaxis=dict(
            range=[-x_max - 50, third_column_end + 50], autorange=False
        ),
        plot_bgcolor='white',
        width=800,
        height=600,
        # title=dict(text="Snow Stratification", x=0.5, xanchor='center'),
    )

    # Adjust x and y axis titles
    fig.update_xaxes(title_text="Density (kg/m³)", side='top')
    fig.update_yaxes(title_text="Height (cm)")

    # Adjust y-axis ticks
    y_grid = np.arange(0, column_header, 100)
    fig.update_yaxes(
        tickvals=y_grid,
        ticktext=[str(int(pos // 10)) for pos in y_grid],
        ticks="outside",
        tickwidth=1,
        tickcolor='black',
        ticklen=5,
    )

    # Add horizontal grid lines only in the table area (from x=0 to x=third_column_end)
    for y in y_grid:
        fig.add_shape(
            type="line",
            x0=0,
            y0=y,
            x1=-10*x_max,  # make sure the line is long enough
            y1=y,
            line=dict(color='lightgrey', width=0.5),
            layer='below',  # Ensure grid lines are behind elements
        )

    # Vertical line at x=0 (y-axis)
    fig.add_shape(
        type="line",
        x0=0,
        y0=substratum_bottom,
        x1=0,
        y1=y_max,
        line=dict(color='black', width=1),
    )

    # Vertical dotted lines for table columns (positive x-values)
    for x in [first_column_start, second_column_start, third_column_start]:
        fig.add_shape(
            type="line",
            x0=x,
            y0=weaklayer_thickness,
            x1=x,
            y1=y_max,
            line=dict(color="lightgrey", width=0.5),
        )

    # Horizontal line at column header
    fig.add_shape(
        type="line",
        x0=0,
        y0=column_header,
        x1=third_column_end,
        y1=column_header,
        line=dict(color='lightgrey', width=0.5),
    )

    # Annotations for table headers
    fig.add_annotation(
        x=(0 + first_column_start) / 2,
        y=(y_max + column_header) / 2,
        text="H (cm)",
        showarrow=False,
        font=dict(color='black', size=12),
        xanchor='center',
        yanchor='middle',
    )

    fig.add_annotation(
        x=first_column_midpoint,
        y=(y_max + column_header) / 2,
        text="Density (kg/m³)",
        showarrow=False,
        font=dict(color='black', size=12),
        xanchor='center',
        yanchor='middle',
    )

    fig.add_annotation(
        x=second_column_midpoint,
        y=(y_max + column_header) / 2,
        text="GF",
        showarrow=False,
        font=dict(color='black', size=12),
        xanchor='center',
        yanchor='middle',
    )

    fig.add_annotation(
        x=third_column_midpoint,
        y=(y_max + column_header) / 2,
        text="R",
        showarrow=False,
        font=dict(color='black', size=12),
        xanchor='center',
        yanchor='middle',
    )

    # Hide x-axis ticks and labels
    fig.update_xaxes(
        showticklabels=False,
        ticks='',
    )
    
    # Remove default hlines
    fig.update_yaxes(
        showgrid=False,
    )

    # Adjust the plot margins
    fig.update_layout(
        # margin=dict(l=80, r=80, t=80, b=80)
    )

    return fig
