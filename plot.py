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
    layers (list of list): 3D list where each sublist contains [density, thickness, hand hardness] for each snow layer.
    grain_list (list of str): List of grain types corresponding to each layer.

    Returns:
    fig (matplotlib.figure.Figure): The Figure object containing the plot.
    ax (matplotlib.axes.Axes): The Axes object containing the plot.
    """
  
    # Initialize figure and axis and color used for density-plots
    fig, ax = plt.subplots(figsize=(10, 5))
    x_max = 550 # Defining max of x-axis (density)
    medium_blue = plt.cm.Blues(0.5)
    dark_blue = plt.cm.Blues(0.99)
    previous_density = 0   # Help variable to plot outline of density-plot

    #Initialize hardness map to translate input values to traditional index
    hardness_mapping = {
    1: "F",
    2: "4F",
    3: "1F",
    4: "P",
    5: "K"
}
  
     # Defining help variables used for table-plot on RHS of graph
    current_table = weaklayer_thickness  
    first_column_start = (-0.5)*100 
    second_column_start = (-1.7)*100
    third_column_start = (-2.1)*100
    third_column_end = (-2.4)*100

    # Midpoints of vertical column borders defined above
    first_column_midpoint = (first_column_start + second_column_start) / 2
    second_column_midpoint = (third_column_start + second_column_start) / 2
    third_column_midpoint = (third_column_end+third_column_start )/2

    # Calculate total height of all layers
    total_height = weaklayer_thickness + sum(thickness for _, thickness, _ in layers)
  
    # Define substratum thickness and position
    substratum_thickness = 100
    substratum_bottom = -substratum_thickness
    substratum_top = 0

    # Plot the substratum and annotate text
    ax.fill_betweenx([substratum_bottom, substratum_top], third_column_end, x_max, color=dark_blue, alpha=1)
    ax.text(250, (substratum_bottom + substratum_top) / 2 - 15, 'substratum', ha='center', va='center', color='white', fontsize=10)

    # Plot the weak layer at the bottom
    current_height = weaklayer_thickness
    weak_layer_top = weaklayer_thickness
    ax.axhline(0, color='grey', linestyle='-', linewidth=1)
    ax.fill_betweenx([0, weak_layer_top], 0, x_max, color='coral', alpha=0.3, hatch = 'x')
    ax.text(250, -15, 'weak layer', ha='center', va='center', color='coral', fontsize=10)

    # Loop to plot each layer from bottom to top
    for (density, thickness, hand_hardness), grain in zip(reversed(layers), reversed(grain_list)):
        # Plot of layers in hand_hardness graph
        layer_bottom = current_height
        layer_top = current_height + thickness

        # Plot of table (adding set height of 50 for each column)
        table_bottom = current_table
        table_top = current_table+50 
      
        # Determine color and hatch pattern based on grain type
        color = plt.cm.Blues(density / 450)
        hatch = '//' if grain == 'mfc' else None

        # Plotting density on x-axis
        ax.fill_betweenx([layer_bottom+1, layer_top], 0, density, color=color, alpha=0.8, hatch=hatch,zorder=1)

        # Plotting outline of density plot
        ax.plot([density, density], [layer_bottom+1, layer_top], color=dark_blue, linestyle='-', linewidth=1)
        ax.plot([previous_density, density], [layer_bottom, layer_bottom], color=dark_blue, linestyle='-', linewidth=1)
        previous_density = density

        # Manually plotting y-axis ticks
        ax.plot([0, -10], [layer_bottom, layer_bottom], color='black', linestyle='-', linewidth=0.5)
        ax.text((-12), layer_bottom, round(layer_bottom/10), ha='left', va='center', color='black', fontsize=7)
      
        # Plotting data legend columns
        ax.plot([first_column_start, third_column_end], [table_bottom, table_bottom], color='grey', linestyle='dotted', linewidth=0.5) 

        # Annotate density in the 1st column
        ax.text(first_column_midpoint, (table_bottom + table_top) / 2, round(density), ha='center', va='center', color='black', fontsize=10)

        # Annotate grain type in the 2nd column
        ax.text(second_column_midpoint, (table_bottom + table_top) / 2, grain, ha='center', va='center', color='black', fontsize=10)

        # Annotate hand_hardness in 3rd column
        ax.text(third_column_midpoint, (table_bottom + table_top) / 2, hardness_mapping.get(hand_hardness, "Unknown hardness"), ha='center', va='center', color='black', fontsize=10)

        # Linking hand_hardness layers to table 
        ax.plot([0, first_column_start], [layer_bottom, table_bottom], color='grey', linestyle='dotted', linewidth=0.25) 
        ax.plot([0, first_column_start], [layer_top, table_top], color='grey', linestyle='dotted', linewidth=0.25) 

        # Update the current height and table
        current_height = layer_top
        current_table = table_top

### Loop over layers is finished ###
  
    # Plotting final tick at max height
    ax.plot([0, -10], [total_height, total_height], color='black', linestyle='-', linewidth=0.5)
    ax.text((-12), total_height, round(total_height/10), ha='left', va='center', color='black', fontsize=7)
  
    # Drawing final contour-line of density plot
    ax.plot([previous_density, 0], [total_height, total_height], color=dark_blue, linestyle='-', linewidth=1)

    # Y-axis adjustments
    y_max = max(total_height, 500)+50 # Adding 50 to ensure space for data table headers
    # Manually plotting grid-lines
    ax.set_ylim(substratum_bottom, y_max)  
    y_grid = np.arange(0, y_max, 100)  
    for y in y_grid:
        ax.plot([0, x_max], [y, y], color='grey', linestyle='--', linewidth=0.5, zorder=0)
    y_tick_positions = y_grid
    y_tick_labels = [pos // 10 for pos in y_tick_positions] # adjusting labels to cm
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
    ax.tick_params(axis='x', colors=medium_blue, direction='in',pad=-15)
    title_position = 0.35  # Normalized position on positive axis
    ax.set_xlabel('Density (kg/m³)', x=title_position, color=medium_blue)
  
    # Data table-adjustments
    # Plotting table columns and annotating titles
    ax.plot([0, 0], [substratum_bottom, y_max], color='black', linestyle='-', linewidth=1) 
    ax.plot([first_column_start, first_column_start], [0, y_max], color='grey', linestyle='dotted', linewidth=0.5) 
    ax.plot([second_column_start, second_column_start], [0, y_max], color='grey', linestyle='dotted', linewidth=0.5) 
    ax.plot([third_column_start, third_column_start], [0, y_max], color='grey', linestyle='dotted', linewidth=0.5) 
    ax.plot([0, third_column_end], [y_max-50, y_max-50], color='grey', linestyle='dotted', linewidth=0.5) 
    ax.text(first_column_start/2, (y_max + y_max-50) / 2, "H (cm)", ha='center', va='center', color='black', fontsize=9)
    ax.text(first_column_midpoint, (y_max + y_max-50) / 2, "Density (kg/m³)", ha='center', va='center', color='black', fontsize=9)
    ax.text(second_column_midpoint, (y_max + y_max-50) / 2, "GF", ha='center', va='center', color='black', fontsize=9)
    ax.text(third_column_midpoint, (y_max + y_max-50) / 2, "R", ha='center', va='center', color='black', fontsize=9)

    # Title of plot
    ax.set_title('Snow Stratification', fontsize=14)

    # Return the figure and axis objects
    return fig, ax