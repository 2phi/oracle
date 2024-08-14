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

    mo.md(
        '<h1 style="font-family: Gill Sans, Tahoma;">🔮 ORACLE</h1>'
        '<p align="center"><b>Observation, Research, and Analysis of Collapse and Loading Experiments</b></p>'
        '<p align="center">Implementation of closed-form analytical models for the analysis of anticracks in the avalanche release process.</p>'
    )
    return mo,


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
          <li><input type="checkbox"> Dataset: NCOMMS data</li>
          <li><input type="checkbox"> Determine distribution function for ERRs and use it as a metric for the probability of propagation</li>
        </ul>

        <h3 style="font-family: Gill Sans, Tahoma;">🔬 Paper</h3>
        <ul>
          <li><input type="checkbox"> Determine distribution function for ERRs and use it as a metric for the probability of propagation</li>
          <li><input type="checkbox"> use example layering (e.g., from NCOMMS paper)</li>
          <li><input type="checkbox"> plot: distribution function of</li>
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

    # Local application-specific imports
    import weac
    import plot

    # Dataclasses
    from dataclasses import dataclass

    mo.md('<h2 style="font-family: Gill Sans, Tahoma;">⚙️ PREAMBLE</h2>' "---")
    return dataclass, np, os, pd, plot, plt, weac


@app.cell
def __(os):
    # current working directory
    cwd = os.getcwd()
    # Define run variable (helper for buttons)
    run = True
    return cwd, run


@app.cell
def __(mo, pd):
    # Hand hardness density parametrization according to Geldsetzer & Jamieson (2000) [1]
    # [1] https://arc.lib.montana.edu/snow-science/objects/issw-2000-121-127.pdf
    grainforms = [
        # ID, abbrv, a, b, description
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
    grainform_df = pd.DataFrame(grainforms, columns=["id", "abbreviation", "a", "b", "type"])

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
    wl_thickness = mo.ui.number(
        1, 100, 1, label="Insert weak layer thickness in mm"
    )
    wl_thickness
    return wl_thickness,


@app.cell
def __(grainform_df, mo):
    num_layer_thickness = mo.ui.number(
        start=1, stop=1000, step=1, label="layer thickness in mm"
    )
    opt_grainform = grainform_df["abbreviation"]
    drop_grainform = mo.ui.dropdown(options=opt_grainform, label="grain form")
    num_hardness = mo.ui.number(start=1, stop=5, step=1, label="hand hardness")
    b_addlayer = mo.ui.run_button(label="Add layer")
    b_resetlayers = mo.ui.run_button(label="Reset all")
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
    drop_grainform,
    grain_list,
    grainform_df,
    layers,
    num_hardness,
    num_layer_thickness,
):
    if b_addlayer.value:
        grainform_row = grainform_df.loc[
            grainform_df["abbreviation"] == drop_grainform.value
        ]
        _a = grainform_row["a"].values[0]
        _b = grainform_row["b"].values[0]
        if drop_grainform.value == "RG":  # exponential case for Rounded grains
            _density = _a + _b * (num_hardness.value**3.15)
        else:
            _density = _a + _b * num_hardness.value

        layers.insert(0, [_density, num_layer_thickness.value])
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
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
