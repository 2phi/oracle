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


def snow_profile(weaklayer_thickness, layers, grain_list):
    """
    Generates a snow stratification profile plot using Plotly.

    Parameters:
    - weaklayer_thickness (float): Thickness of the weak layer in the snowpack.
    - layers (list of tuples): Each tuple contains (density, thickness, hand_hardness) of a layer.
    - grain_list (list): List of grain forms corresponding to each layer.

    Returns:
    - fig (go.Figure): A Plotly figure object representing the snow profile.
    """

    # Define colors
    COLORS = {
        'slab_fill': "#D1E2F2",
        'slab_line': "#3C658B",
        'weak_layer_fill': "#FFCDD2",
        'weak_layer_line': "#E57373",
        'weak_layer_text': "#C62828",
        'substratum_fill': "#ECEFF1",
        'substratum_line': "#607D8B",
        'substratum_text': "#607D8B",
    }

    # Compute total height and set y-axis maximum
    total_height = weaklayer_thickness + sum(
        thickness for _, thickness, _ in layers
    )
    y_max = max(total_height * 1.1, 450)  # Ensure y_max is at least 500

    # Define substratum properties
    substratum_thickness = 50

    # Compute x-axis maximum based on layer densities
    max_density = max((density for density, _, _ in layers), default=400)
    x_max = max(1.05 * max_density, 400)  # Ensure x_max is at least 400

    # Initialize the Plotly figure
    fig = go.Figure()

    # Plot the substratum (base layer)
    fig.add_shape(
        type="rect",
        x0=-x_max,
        x1=x_max,
        y0=-substratum_thickness,
        y1=0,
        fillcolor=COLORS['substratum_fill'],
        line=dict(width=2, color=COLORS['substratum_fill']),
        # layer='below',
    )

    fig.add_shape(
        type="line",
        x0=-x_max,
        y0=0,
        x1=x_max,
        y1=0,
        line=dict(color=COLORS['substratum_line'], width=1.2),
    )

    # Determine weak layer density
    weak_density = 100

    # Add substratum label
    fig.add_annotation(
        x=-weak_density / 2,  # -x_max / 2,
        y=-substratum_thickness / 2,
        text="substratum ",
        showarrow=False,
        font=dict(color=COLORS['substratum_text'], size=10),
        xanchor='center',
        yanchor='middle',
    )

    # Plot the weak layer
    fig.add_shape(
        type="rect",
        x0=-weak_density,
        x1=0,
        y0=0,
        y1=weaklayer_thickness,
        fillcolor=COLORS['weak_layer_fill'],
        line=dict(width=1, color=COLORS['weak_layer_line']),
        layer='below',
    )

    # Add weak layer label
    fig.add_annotation(
        x=-weak_density / 2,
        y=weaklayer_thickness / 2,
        text="weak layer",
        showarrow=False,
        font=dict(color=COLORS['weak_layer_text'], size=10),
        xanchor='center',
        yanchor='middle',
    )

    # Initialize variables for plotting layers
    current_height = weaklayer_thickness
    previous_density = 0  # Start from zero density

    # Define positions for annotations (table columns)
    col_width = 0.08
    x_pos = {
        'col1_start': 1 * col_width * x_max,
        'col2_start': 2 * col_width * x_max,
        'col3_start': 3 * col_width * x_max,
        'col3_end': 4 * col_width * x_max,
    }

    # Compute midpoints for annotation placement
    first_column_mid = (x_pos['col1_start'] + x_pos['col2_start']) / 2
    second_column_mid = (x_pos['col2_start'] + x_pos['col3_start']) / 2
    third_column_mid = (x_pos['col3_start'] + x_pos['col3_end']) / 2

    # Set the position for the table header
    column_header_y = y_max / 1.1
    max_table_row_height = 85  # Maximum height for table rows

    # Calculate average height per table row
    num_layers = max(len(layers), 1)
    avg_row_height = (column_header_y - weaklayer_thickness) / num_layers
    avg_row_height = min(avg_row_height, max_table_row_height)

    # Initialize current table height
    current_table_y = weaklayer_thickness

    # Loop through each layer and plot
    for (density, thickness, hand_hardness), grain in zip(layers, grain_list):
        # Define layer boundaries
        layer_bottom = current_height
        layer_top = current_height + thickness

        # Plot the layer
        fig.add_shape(
            type="rect",
            x0=-density,
            x1=0,
            y0=layer_bottom,
            y1=layer_top,
            fillcolor=COLORS['slab_fill'],
            line=dict(width=0.4, color=COLORS['slab_fill']),
            layer='above',
        )

        # Plot lines connecting previous and current densities
        fig.add_shape(
            type="line",
            x0=-previous_density,
            y0=layer_bottom,
            x1=-density,
            y1=layer_bottom,
            line=dict(color=COLORS['slab_line'], width=1.2),
        )
        fig.add_shape(
            type="line",
            x0=-density,
            y0=layer_bottom,
            x1=-density,
            y1=layer_top,
            line=dict(color=COLORS['slab_line'], width=1.2),
        )

        # Add height markers on the left
        fig.add_shape(
            type="line",
            x0=0,
            y0=layer_bottom,
            x1=10,
            y1=layer_bottom,
            line=dict(width=0.5),
        )
        fig.add_annotation(
            x=12,
            y=layer_bottom,
            text=str(round(layer_bottom / 10)),
            showarrow=False,
            font=dict(size=10),
            xanchor='left',
            yanchor='middle',
            bgcolor='white',
        )

        # Define table row boundaries
        table_bottom = current_table_y
        table_top = current_table_y + avg_row_height

        # Add table grid lines
        fig.add_shape(
            type="line",
            x0=x_pos['col1_start'],
            y0=table_bottom,
            x1=x_pos['col3_end'],
            y1=table_bottom,
            line=dict(color="lightgrey", width=0.5),
        )

        # Add annotations for density, grain form, and hand hardness
        fig.add_annotation(
            x=first_column_mid,
            y=(table_bottom + table_top) / 2,
            text=str(round(density)),
            showarrow=False,
            font=dict(size=10),
            xanchor='center',
            yanchor='middle',
        )
        fig.add_annotation(
            x=second_column_mid,
            y=(table_bottom + table_top) / 2,
            text=grain,
            showarrow=False,
            font=dict(size=10),
            xanchor='center',
            yanchor='middle',
        )
        fig.add_annotation(
            x=third_column_mid,
            y=(table_bottom + table_top) / 2,
            text=hand_hardness,
            showarrow=False,
            font=dict(size=10),
            xanchor='center',
            yanchor='middle',
        )

        # Lines from layer edges to table
        fig.add_shape(
            type="line",
            x0=0,
            y0=layer_bottom,
            x1=x_pos['col1_start'],
            y1=table_bottom,
            line=dict(color="lightgrey", width=0.5),
        )

        # Update variables for next iteration
        previous_density = density
        current_height = layer_top
        current_table_y = table_top

    # Add top layer height marker
    fig.add_shape(
        type="line",
        x0=0,
        y0=total_height,
        x1=10,
        y1=total_height,
        line=dict(width=0.5),
    )
    fig.add_annotation(
        x=12,
        y=total_height,
        text=str(round(total_height / 10)),
        showarrow=False,
        font=dict(size=10),
        xanchor='left',
        yanchor='middle',
    )

    # Final line connecting last density to x=0 at total_height
    fig.add_shape(
        type="line",
        x0=-previous_density,
        y0=total_height,
        x1=0,
        y1=total_height,
        line=dict(color=COLORS['slab_line'], width=1),
    )

    # Set axes properties
    fig.update_layout(
        yaxis=dict(range=[-1.05 * substratum_thickness, y_max]),
        xaxis=dict(
            range=[-1.05 * x_max, x_pos['col3_end']],
            autorange=False,
        ),
        height=max(500, 0.5 * y_max),
    )

    # Add horizontal grid lines
    y_tick_spacing = 100 if total_height < 800 else 200
    y_grid = np.arange(0, total_height, y_tick_spacing)
    for y in y_grid:
        fig.add_shape(
            type="line",
            x0=0,
            y0=y,
            x1=-x_max,  # Extend grid line to the left
            y1=y,
            line=dict(color='lightgrey', width=0.5),
            layer='below',
        )

    # Adjust axes labels and ticks
    fig.update_xaxes(
        tickvals=[],  # np.arange(-1000, 0, 100),
        # ticks='inside',
        # ticklabelposition='inside',
        # title_text="Density (kg/m³)",
        # side='top',
        # showticklabels=True,
        # linewidth=.5,
        # mirror=True,
        # linecolor='lightgray',
    )

    fig.update_yaxes(
        zeroline=False,
        tickvals=[],  # y_grid,
        # ticktext=[str(int(y // 10)) for y in y_grid],
        # title_text="Height (cm)",
        showgrid=False,
        # linewidth=.5,
        # showline=True,
        # mirror=True,
        # linecolor='lightgray',
    )

    # Vertical line at x=0 (y-axis)
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,  # -substratum_thickness,
        x1=0,
        y1=y_max,
        line=dict(width=1),
    )

    # Vertical lines for table columns
    for x in [
        x_pos['col1_start'],
        x_pos['col2_start'],
        x_pos['col3_start'],
    ]:
        fig.add_shape(
            type="line",
            x0=x,
            y0=weaklayer_thickness,
            x1=x,
            y1=y_max,
            line=dict(color="lightgrey", width=0.5),
        )

    # Horizontal line at table header
    fig.add_shape(
        type="line",
        x0=0,
        y0=column_header_y,
        x1=x_pos['col3_end'],
        y1=column_header_y,
        line=dict(color='lightgrey', width=0.5),
    )

    # Annotations for table headers
    header_y_position = (y_max + column_header_y) / 2
    fig.add_annotation(
        x=(0 + x_pos['col1_start']) / 2,
        y=header_y_position,
        text="H",  # "H<br>cm",  # "H (cm)",
        showarrow=False,
        font=dict(size=10),
        xanchor='center',
        yanchor='middle',
    )
    fig.add_annotation(
        x=first_column_mid,
        y=header_y_position,
        text="D",  # 'D<br>kg/m³',  # "Density (kg/m³)",
        showarrow=False,
        font=dict(size=10),
        xanchor='center',
        yanchor='middle',
    )
    fig.add_annotation(
        x=second_column_mid,
        y=header_y_position,
        text='F',  # "GF",
        showarrow=False,
        font=dict(size=10),
        xanchor='center',
        yanchor='middle',
    )
    fig.add_annotation(
        x=third_column_mid,
        y=header_y_position,
        text="R",
        showarrow=False,
        font=dict(size=10),
        xanchor='center',
        yanchor='middle',
    )

    fig.add_annotation(
        x=-x_max,
        y=-substratum_thickness - 2,
        text="H – Height (cm)           D – Density (kg/m³)           F – Grain Form           R – Hand Hardness",
        showarrow=False,
        xanchor='left',
        yanchor='top',
        align='left',
    )

    # Adjust the plot margins (optional)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=40))

    return fig
