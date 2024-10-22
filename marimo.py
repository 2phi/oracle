import marimo

__generated_with = "0.8.19"
app = marimo.App(
    width="medium",
    app_title="ORACLE",
    layout_file="layouts/app.grid.json",
)


@app.cell
def __():
    import marimo as mo

    mo.md(
        '<h1 style="font-family: Gill Sans, Tahoma;">🔮 ORACLE</h1>'
        '<p align="center"><b>Observation, Research, and Analysis of Collapse and Loading Experiments</b></p>'
        # '<p align="center">Implementation of closed-form analytical models for the analysis of anticracks in the avalanche release process.</p>'
    )
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        <h1 align="center">
          <img src="https://github.com/2phi/oracle/raw/main/img/steampunk-v1.png" alt="ORACLE" width="200">
          <b>ORACLE</b>
          <br>
          <sub><sup>Observation, Research, and Analysis of Collapse and Loading Experiments</sup></sub>
          <br>
        </h1>
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        <div style="margin-bottom: 2px;">
            <h2 style="font-family: Gill Sans, Tahoma;">⏳ TODO</h2><hr><br>
        </div>

        <h3 style="font-family: Gill Sans, Tahoma;">🎮 App</h3>
        <ul>
          <li><input type="checkbox"> user input: inclination</li>
          <li><input type="checkbox"> user input: cutting direction</li>
          <li><input type="checkbox"> user input: slab faces (normal, vertical)</li>
          <li><input type="checkbox"> user input: column length</li>
          <li><input type="checkbox"> user input: cut length</li>
          <li><input type="checkbox"> Run WEAC to compute ERR</li>
        </ul>
        """
    )
    return


@app.cell
def __(mo):
    # Standard library imports
    import os

    # Third-party imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import weac

    # Local application-specific imports
    from oracle import plot

    # Dataclasses
    from dataclasses import dataclass

    mo.md('<h2 style="font-family: Gill Sans, Tahoma;">⚙️ PREAMBLE</h2>' "---")
    return dataclass, np, os, pd, plot, plt, weac


@app.cell
def __(__file__, os):
    # Change the CWD to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Define run variable (helper for buttons)
    run = True
    return run, script_dir


@app.cell
def __(mo, pd):
    # Hand hardness density parametrization according to Geldsetzer & Jamieson (2000) [1]
    # [1] https://arc.lib.montana.edu/snow-science/objects/issw-2000-121-127.pdf
    grainforms = [
        # ID, abbrv, symbol, a, b, description
        (0, "PP", 45, 36, "Precipitation particles"),
        (1, "PPgp", 83, 37, "Graupel"),
        (2, "DF", 65, 36, "Decomposing and fragmented precipitation particles"),
        (3, "RG", 154, 1.51, "Rounded grains"),
        (4, "RGmx", 91, 42, "Rounded mixed forms"),
        (5, "FC", 112, 46, "Faceted crystals"),
        (6, "FCmx", 56, 64, "Faceted mixed forms"),
        (7, "DH", 185, 25, "Depth hoar"),
        # MFCr density is constant and takes as mean of Table 1 in [1]
        (8, "MFCr", 292.25, 0, "Melt-freeze crusts"),
    ]

    # Collect grainforms info in a dataframe
    grainform_df = pd.DataFrame(
        grainforms, columns=["id", "abbreviation", "a", "b", "type"]
    )

    # Provide a table view of the dataframe
    grainform_table_view = mo.ui.table(
        data=grainform_df,
        show_column_summaries=False,
        selection=None,
        label="Hand-hardness-to-density parametrization of depending on grain type",
    )

    mo.md(
        '<h2 style="font-family: Gill Sans, Tahoma;">⚖️ DENSITY PARAMETRIZATION</h2><hr>'
    )
    return grainform_df, grainform_table_view, grainforms


@app.cell
def __(grainform_table_view):
    grainform_table_view
    return


@app.cell
def __(grainform_df, mo, plot):
    _fig, _axis = plot.density_vs_hand_hardness(grainform_df)
    mo.mpl.interactive(_fig)
    return


@app.cell
def __(mo):
    mo.md("""<h2 style="font-family: Gill Sans, Tahoma;">🖥️ USER INTERFACE</h2><hr>""")
    return


@app.cell
def __(mo, run):
    if run:
        # 3D list of layer densities, thicknesses and hand hardness. Columns are density (kg/m^3), thickness (mm) and hand_hardness (N/A).
        layers = []

        grain_list = []

        # Global button ensuring updating of dynamic table and plot of snow stratification
        b_update_table_plot = mo.ui.run_button(label="Update table and plot", full_width=True)
    return b_update_table_plot, grain_list, layers


@app.cell
def __(mo):
    wl_thickness = mo.ui.number(
        1, 100, 1, label="Set weak-layer thickness (mm)", value=30, full_width=True
    )
    wl_thickness
    return (wl_thickness,)


@app.cell
def __(add_layer, grainform_df, mo, reset_all_layers):
    # Dropdown lists
    num_layer_thickness = mo.ui.number(
        value=100, start=1, stop=1000, step=1, label="with thickness (mm)", full_width=True
    )
    opt_grainform = grainform_df["abbreviation"]
    drop_grainform = mo.ui.dropdown(value=opt_grainform[1], options=opt_grainform, label="of grain type", full_width=True)
    num_hardness = mo.ui.number(start=1, stop=5, step=1, label="and hand hardness", full_width=True)

    # Buttons
    b_addlayer = mo.ui.run_button(label="Add layer", on_change=lambda value: add_layer(num_layer_thickness.value, drop_grainform.value, num_hardness.value), full_width=True)
    b_resetlayers = mo.ui.run_button(label="Reset all", on_change=lambda value: reset_all_layers(), full_width=True)
    return (
        b_addlayer,
        b_resetlayers,
        drop_grainform,
        num_hardness,
        num_layer_thickness,
        opt_grainform,
    )


@app.cell
def __(mo):
    mo.md("""<h2 style="font-family: Gill Sans, Tahoma;">✍🏼 Add layers bottom to top</h2><hr>""")
    return


@app.cell
def __(num_layer_thickness):
    num_layer_thickness
    return


@app.cell
def __(drop_grainform):
    drop_grainform
    return


@app.cell
def __(num_hardness):
    num_hardness
    return


@app.cell
def __(b_addlayer):
    b_addlayer
    return


@app.cell
def __(b_resetlayers):
    b_resetlayers
    return


@app.cell
def __(
    b_addlayer,
    b_resetlayers,
    drop_grainform,
    mo,
    num_hardness,
    num_layer_thickness,
):
    mo.vstack(
        [
            "Add layers from bottom to top:",
            mo.hstack(
                [
                    num_layer_thickness,
                    drop_grainform,
                    num_hardness,
                    b_addlayer,
                    b_resetlayers,
                ],
                justify="center",
            ),
        ],
        align="start",
        justify="space-between",
    )
    return


@app.cell
def __(
    b_addlayer,
    b_resetlayers,
    b_update_table_plot,
    deleting_layers,
    grain_list,
    layers,
    mo,
    opt_grainform,
    update_grainform,
    update_hand_hardness,
    update_thickness,
):
    if b_addlayer.value or b_resetlayers.value or b_update_table_plot.value:

        # Creating the table
        updated_thickness = mo.ui.array(
                mo.ui.number(value=int(layer[1]), start=1, stop=1000, step=1,
                            on_change=lambda value, i=i: update_thickness(i, value))
                for i, layer in enumerate(layers)
        )

        updated_grainform = mo.ui.array(
                mo.ui.dropdown(options=opt_grainform, value=grainform,
                              on_change=lambda value, i=i: update_grainform(i, value))
                for i, grainform in enumerate(grain_list)
        )

        updated_hand_hardness = mo.ui.array(
                mo.ui.number(value=int(layer[2]), start=1, stop=5, step=1,
                            on_change=lambda value, i=i: update_hand_hardness(i, value)
                            )
                for i, layer in enumerate(layers)
        )

        remove_buttons = mo.ui.array(
                        mo.ui.run_button(label=f"Remove layer {i+1}",
                                        on_change=lambda value, i=i: deleting_layers(i))
                        for i, layer in enumerate(layers)
        )

        table = mo.hstack(
            [
                mo.vstack( ["Thickness (mm)", updated_thickness] ),
                mo.vstack( ["Grain form", updated_grainform] ), 
                mo.vstack( ["Hand hardness", updated_hand_hardness] ),
                mo.vstack( ["Remove layers", remove_buttons] )
            ]
        )
    return (
        remove_buttons,
        table,
        updated_grainform,
        updated_hand_hardness,
        updated_thickness,
    )


@app.cell
def __(table):
    table
    return


@app.cell
def __(b_update_table_plot):
    # Only possible to remove one layer at a time, after which update table and plot must be pressed
    b_update_table_plot
    return


@app.cell
def __(
    b_addlayer,
    b_update_table_plot,
    grain_list,
    layers,
    plot,
    run,
    wl_thickness,
):
    if run or b_addlayer or b_update_table_plot:
        fig_2, ax_2 = plot.snow_stratification(wl_thickness.value, layers, grain_list)
    return ax_2, fig_2


@app.cell
def __(fig_2):
    fig_2
    return


@app.cell
def __(grain_list, grainform_df, layers):
    def add_layer(_num_layer_thickness, _drop_grainform, _num_hardness):
        global grain_list
        global layers

        _grainform_row = grainform_df.loc[
            grainform_df["abbreviation"] == _drop_grainform
        ]
        _a = _grainform_row["a"].values[0]
        _b = _grainform_row["b"].values[0]
        if _drop_grainform == "RG":  # exponential case for Rounded grains
            _density = _a + _b * (_num_hardness**3.15)
        else:
            _density = _a + _b * _num_hardness

        layers.append([_density, _num_layer_thickness, _num_hardness])
        grain_list.append(_drop_grainform)
    return (add_layer,)


@app.cell
def __(grain_list, layers):
    def reset_all_layers():
        global grain_list
        global layers

        layers.clear()
        grain_list.clear()
    return (reset_all_layers,)


@app.cell
def __(layers):
    def update_thickness(index, new_thickness):
        global layers

        layers[index][1] = new_thickness
    return (update_thickness,)


@app.cell
def __(drop_grainform, grain_list, grainform_df, layers):
    def update_grainform(index, new_grainform):
        global grain_list
        global layers

        current_hardness = layers[index][2] 

        grainform_row = grainform_df.loc[
            grainform_df["abbreviation"] == new_grainform
        ]
        _a = grainform_row["a"].values[0]
        _b = grainform_row["b"].values[0]
        if drop_grainform.value == "RG":  # exponential case for Rounded grains
            _density = _a + _b * (current_hardness**3.15)
        else:
            _density = _a + _b * current_hardness

        grain_list[index] = new_grainform
        layers[index][0] = _density
    return (update_grainform,)


@app.cell
def __(drop_grainform, grain_list, grainform_df, layers):
    def update_hand_hardness(index, new_hardness):
        global layers
        global grain_list

        current_grainform = grain_list[index]

        grainform_row = grainform_df.loc[
            grainform_df["abbreviation"] == current_grainform
        ]
        _a = grainform_row["a"].values[0]
        _b = grainform_row["b"].values[0]
        if drop_grainform.value == "RG":  # exponential case for Rounded grains
            _density = _a + _b * (new_hardness**3.15)
        else:
            _density = _a + _b * new_hardness

        layers[index][2] = new_hardness
        layers[index][0] = _density
    return (update_hand_hardness,)


@app.cell
def __(grain_list, layers):
    def deleting_layers(index):
        global layers
        global grain_list

        layers.pop(index)
        grain_list.pop(index)
    return (deleting_layers,)


if __name__ == "__main__":
    app.run()
