import marimo

__generated_with = "0.7.17"
app = marimo.App(
    width="medium",
    app_title="ORACLE",
    layout_file="layouts/dashboard.grid.json",
)


@app.cell
def __():
    import marimo as mo

    mo.md("# ORACLE")
    return mo,


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## 1. PREAMBLE
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import weac
    from dataclasses import dataclass
    return dataclass, np, os, pd, plt, weac


@app.cell
def __(os):
    # current working directory 
    cwd = os.getcwd()
    # Define run variable (helper for buttons)
    run = True
    return cwd, run


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## TODO
        """
    )
    return


@app.cell
def __(dataclass, mo):
    @dataclass
    class Task:
        name: str
        done: bool = False


    get_tasks, set_tasks = mo.state([])
    task_added, set_task_added = mo.state(False)
    return Task, get_tasks, set_task_added, set_tasks, task_added


@app.cell
def __(mo, task_added):
    # Refresh the text box whenever a task is added
    task_added

    task_entry_box = mo.ui.text(placeholder="a task ...")
    return task_entry_box,


@app.cell
def __(Task, mo, set_task_added, set_tasks, task_entry_box):
    def add_task():
        if task_entry_box.value:
            set_tasks(lambda v: v + [Task(task_entry_box.value)])
            set_task_added(True)

    def clear_tasks():
        set_tasks(lambda v: [task for task in v if not task.done])

    add_task_button = mo.ui.button(
        label="add task",
        on_change=lambda _: add_task(),
    )

    clear_tasks_button = mo.ui.button(
        label="clear completed tasks",
        on_change=lambda _: clear_tasks()
    )
    return add_task, add_task_button, clear_tasks, clear_tasks_button


@app.cell
def __(Task, get_tasks, mo, set_tasks):
    task_list = mo.ui.array(
        [mo.ui.checkbox(value=task.done, label=task.name) for task in get_tasks()],
        label="tasks",
        on_change=lambda v: set_tasks(
            lambda tasks: [Task(task.name, done=v[i]) for i, task in enumerate(tasks)]
        ),
    )
    return task_list,


@app.cell
def __(add_task_button, clear_tasks_button, mo, task_entry_box):
    mo.hstack(
        [task_entry_box, add_task_button, clear_tasks_button], justify="start"
    )
    return


@app.cell
def __(task_list):
    task_list
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## 1. USER INPUT

        - user input: inclination
        - user input: cutting direction
        - user input: slab faces (normal, vertical)
        - user input: column length
        - user input: cut length
        - change style sheet?
        - Run WEAC to compute ERR
        - Dataset: NCOMMS data
        - Determine distribution function for ERRs and use it as a metric for the probability of propagation

        ### FOR THE PAPER
        - Determine distribution function for ERRs and use it as a metric for the probability of propagation
        - use example layering (e.g. from NCOMMS paper)
        - plot: distribution function of ERRs
        """
    )
    return


@app.cell
def __(mo):
    mo.md("""### STRATIFICATION""")
    return


@app.cell
def __(mo):
    mo.accordion(
        {
            "**Dry snow density from garin form and hand hardness**": mo.md(
                rf"""
            Parameters for density parametrisations based on snow form and hand hardness according to Geldsetzer and Jamiesnon         (See paper below).

            *Geldsetzer, Torsten & Jamieson, Bruce. (2000). ESTIMATING DRY SNOW DENSITY FROM GRAIN FORM AND HAND HARDNESS*
            """
            )
        }
    )
    return


@app.cell
def __(mo, np, pd):
    grainform_df = pd.DataFrame()
    grainform_df['number']=np.arange(0,9,1)
    grainform_df['type']=['Precipitation particles', 'Graupel', 'Decomposing and Fragmented precipitation particles'\
                   , 'Rounded grains', 'Rounded mixed forms', 'Faceted crystals', 'Faceted mixed forms', 'Depth hoar' , 'melt-freeze crusts']
    grainform_df['type_abbr']=['PP', 'PPgp', 'DF', 'RG', 'RGmx', 'FC', 'FCmx', 'DH' , 'mfc'] # RG with non-linear regression
    # for mfc the everage value of table 1 is taken (see below)
    sig_mfc = np.mean([332, 284, 278, 286, 282, 304, 296, 276])
    grainform_df['a']=[45 ,83 ,65 ,154 ,91 ,112 ,56 ,185 ,sig_mfc] 
    grainform_df['b']=[36 ,37 ,36 ,1.51 ,42 ,46 ,64 ,25 ,0]
    #grainform_df['included'] = [True, False, True, True, False, True, False, False, True] # included in slab generator 

    grainform_table = mo.ui.table(
        data=grainform_df,
        # use pagination when your table has many rows
        pagination=True,
        label="Grain form and hand hardness to density",
    )
    grainform_table
    return grainform_df, grainform_table, sig_mfc


@app.cell
def __(mo):
    mo.pdf(
        src="https://arc.lib.montana.edu/snow-science/objects/issw-2000-121-127.pdf",
        width="100%",
        height="50vh",
    )
    return


@app.cell
def __(grainform_df, mo, plot_density_vs_hand_hardness):
    _fig,_axis = plot_density_vs_hand_hardness(grainform_df)
    mo.mpl.interactive(_fig)
    return


@app.cell
def __(mo):
    mo.md("""**User Interface**""")
    return


@app.cell
def __(b_resetlayers, run):
    if run or b_resetlayers:
        layers = []
        # 2D list of top-to-bottom layer densities and thicknesses. Columns are density (kg/m^3) and thickness (mm).
        grain_list = []
    return grain_list, layers


@app.cell
def __(mo):
    wl_thickness = mo.ui.number(1,100,1, label='Insert weak layer thickness in mm')
    wl_thickness
    return wl_thickness,


@app.cell
def __(grainform_df, mo):
    num_layer_thickness = mo.ui.number(start=1, stop=1000, step=1, label='layer thickness in mm')
    opt_grainform = grainform_df['type_abbr']
    drop_grainform = mo.ui.dropdown(options=opt_grainform, label='grain form')
    num_hardness = mo.ui.number(start=1, stop=5, step=1, label='hand hardness')
    b_addlayer = mo.ui.run_button(label='Add layer')
    b_resetlayers = mo.ui.run_button(label='Reset all')
    return (
        b_addlayer,
        b_resetlayers,
        drop_grainform,
        num_hardness,
        num_layer_thickness,
        opt_grainform,
    )


@app.cell
def __(
    b_addlayer,
    b_resetlayers,
    drop_grainform,
    mo,
    num_hardness,
    num_layer_thickness,
):
    mo.vstack(['Add layers from bottom to top:', mo.hstack([num_layer_thickness, drop_grainform, num_hardness, b_addlayer, b_resetlayers], justify="center")], align='start', justify='space-between')
    return


@app.cell
def __(
    b_addlayer,
    drop_grainform,
    grain_list,
    grainform_df,
    layers,
    num_hardness,
    num_layer_thickness,
):
    if b_addlayer.value: 
        grainform_row = grainform_df.loc[grainform_df['type_abbr'] == drop_grainform.value]
        _a = grainform_row['a'].values[0]
        _b = grainform_row['b'].values[0]
        if drop_grainform.value == 'RG':  # exponential case for Rounded grains
            _density = _a + _b * (num_hardness.value ** 3.15)
        else:
            _density = _a + _b * num_hardness.value

        layers.insert(0, [_density,num_layer_thickness.value])
        grain_list.insert(0, drop_grainform.value)
    return grainform_row,


@app.cell
def __(layers):
    print(layers)
    return


@app.cell
def __(
    b_addlayer,
    grain_list,
    layers,
    plot_snow_stratification,
    run,
    wl_thickness,
):
    if run or b_addlayer:
        fig, ax = plot_snow_stratification(wl_thickness.value, layers, grain_list)
    return ax, fig


@app.cell
def __(fig):
    fig
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## FUNCTIONS
        """
    )
    return


@app.cell
def __(np, plt):
    def plot_density_vs_hand_hardness(grainform_df):
        """
        Plots the density of different grain types versus hand hardness and returns the figure and axis objects.

        Parameters:
        grainform_df (pd.DataFrame): A DataFrame containing the following columns:
            - 'number': Numeric identifier for each grain type.
            - 'type': Name of the grain type.
            - 'type_abbr': Abbreviation for the grain type.
            - 'a': Coefficient 'a' for the density formula.
            - 'b': Coefficient 'b' for the density formula.

        The density is calculated using the following formulas:
        - For most grain types: density = a + b * hand_hardness
        - Except 'Rounded grains' (type_abbr = 'RG'): density = a + b * (hand_hardness ^ 3.15)

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
            a = row['a']
            b = row['b']
            if row['type_abbr'] == 'RG':  # exponential case for Rounded grains
                densities = a + b * (hand_hardness ** 3.15)
            else:
                densities = a + b * hand_hardness

            ax.plot(hand_hardness, densities, label=row['type_abbr'])

        # Set plot limits and labels
        ax.set_ylim(50, 450)
        ax.set_xlim(1, 5)
        ax.set_xlabel('Hand Hardness')
        ax.set_ylabel('Density (kg/m³)')
        ax.set_title('Density vs Hand Hardness for Different Grain Types')
        ax.legend(loc='best')

        # Add grid for better readability
        ax.grid(True)

        # Return the figure and axis objects
        return fig, ax


    def plot_snow_stratification(weaklayer_thickness, layers, grain_list):
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
        total_height = weaklayer_thickness + sum(thickness for _, thickness in layers)

        # Plot the substratum
        ax.fill_betweenx([substratum_bottom, substratum_top], 0, 1, color='lightgrey', alpha=0.6)
        ax.text(0.5, (substratum_bottom + substratum_top) / 2 - 15, 'substratum', ha='center', va='center', color='black', fontsize=10)

        # Plot the weak layer at the bottom
        current_height = weaklayer_thickness
        weak_layer_top = weaklayer_thickness
        ax.axhline(0, color='grey', linestyle='-', linewidth=1)
        ax.axhline(weak_layer_top, color='grey', linestyle='-', linewidth=1)
        ax.fill_betweenx([0, weak_layer_top], 0, 1, color='coral', alpha=0.3, hatch = 'x')
        ax.text(0.5, -15, 'weak layer', ha='center', va='center', color='coral', fontsize=10)

        # Plot each layer from bottom to top
        for (density, thickness), grain in zip(reversed(layers), reversed(grain_list)):
            layer_bottom = current_height
            layer_top = current_height + thickness

            # Determine color and hatch pattern based on grain type
            color = plt.cm.viridis(1 - density / 450)
            hatch = '//' if grain == 'mfc' else None

            # Fill the layer with color and optional hatch pattern
            ax.fill_betweenx([layer_bottom, layer_top], 0, 1, color=color, alpha=0.6, hatch=hatch)
            ax.axhline(layer_top, color='grey', linestyle='-', linewidth=1)

            # Annotate density in the middle of the layer
            ax.text(0.5, (layer_bottom + layer_top) / 2, f'{int(density)} kg/m³', ha='center', va='center', color='black', fontsize=10)

            # Annotate grain type on the right side in the middle of the layer
            ax.text(1.1, (layer_bottom + layer_top) / 2, grain, ha='left', va='center', color='black', fontsize=10)

            # Update the current height
            current_height = layer_top

        # Set axis limits and labels
        ax.set_ylim(substratum_bottom, max(total_height, 500))  # Ensure y-axis starts at 500 mm or lower
        ax.set_xlim(0, 1.2)  # Adjust x-axis limit to make space for grain annotations
        ax.set_xticks([])  # No numbers on the x-axis
        ax.set_ylabel('mm')
        ax.set_title('Snow Stratification', fontsize=10)

        # Add grid for better readability
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

        # Return the figure and axis objects
        return fig, ax
    return plot_density_vs_hand_hardness, plot_snow_stratification


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
