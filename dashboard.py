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
def __(b_addlayer, grain_list, layers, plot, run, wl_thickness):
    if run or b_addlayer:
        fig, ax = plot.snow_stratification(wl_thickness.value, layers, grain_list)
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
def __(pd):
    df_adam2014 = pd.read_pickle("data/adam2024.pkl")
    df_vanherwijnen2016 = pd.read_pickle("data/vanherwijnen2016.pkl")
    return df_adam2014, df_vanherwijnen2016


@app.cell
def __(df_vanherwijnen2016):
    df_vanherwijnen2016.L.describe()
    return


@app.cell
def __(df_adam2014, df_vanherwijnen2016, pd):
    df_err = pd.concat(
        [
            df_adam2014[["Gc", "GIc", "GIIc", "slope_incl"]],
            df_vanherwijnen2016[["Gc", "GIc", "GIIc", "slope_incl"]],
        ],
        axis=0,
    )
    return df_err,


@app.cell
def __(df_err):
    from uncertainties import unumpy
    df_err['Gc_mean'] = df_err['Gc'].apply(unumpy.nominal_values)
    df_err['Gc_std'] = df_err['Gc'].apply(unumpy.std_devs)
    return unumpy,


@app.cell
def __(df_err):
    df_err.describe()
    return


@app.cell
def __(df_err):
    import seaborn as sns
    sns.displot(data=df_err, x="Gc_mean", kind="hist", bins = 100, aspect = 1.5)
    return sns,


@app.cell
def __():
    from fitter import Fitter, get_common_distributions, get_distributions
    return Fitter, get_common_distributions, get_distributions


@app.cell
def __(get_common_distributions):
    get_common_distributions()
    return


@app.cell
def __(Fitter, df_err, get_common_distributions):
    # Create a Fitter instance, specify the data and the distributions you want to fit
    f = Fitter(df_err['Gc_mean'], distributions=get_common_distributions())
    # f = Fitter(df_err['Gc_mean'], distributions=get_distributions())

    # Fit the data
    f.fit()
    return f,


@app.cell
def __(f):
    # Display a summary of the results
    f.summary(Nbest=5)
    return


@app.cell
def __(f, mo, plt):
    f.plot_pdf(Nbest=1)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(f):
    # Get the best fitting distribution and its parameters
    best_fit = f.get_best(method='sumsquare_error')
    print(best_fit)
    return best_fit,


@app.cell
def __(df_err, np, plt, sns):
    import scipy.stats as stats

    shape, loc, scale = stats.lognorm.fit(df_err['Gc_mean'])

    x = np.linspace(min(df_err['Gc_mean']), max(df_err['Gc_mean']), 1000)
    pdf = stats.lognorm.pdf(x, shape, loc, scale)
    cdf = stats.lognorm.cdf(x, shape, loc, scale)

    plt.plot(x, pdf, lw=2, label='Probability density function', color='darkblue')
    plt.plot(x, cdf, lw=2, label='Cumulative density function', color='darkred')
    sns.histplot(df_err['Gc_mean'], bins=75, kde=False, stat="density", color="red",cumulative=True, label="Cumulative histogram")
    sns.histplot(df_err['Gc_mean'], bins=75, kde=False, stat="density", color="skyblue", label="Histogram")
    return cdf, loc, pdf, scale, shape, stats, x


@app.cell
def __(cdf, df_err, np, pd, pdf, x):
    pdf_data, hist_bins = np.histogram(df_err['Gc_mean'], bins=75, density=True)
    cdf_data = np.cumsum(pdf_data) * np.diff(hist_bins)
    hist_x = (hist_bins[:-1] + hist_bins[1:]) / 2

    hist_df = pd.DataFrame({'Gc(J/m^2)': hist_x, 'probability_density': pdf_data})
    cum_df = pd.DataFrame({'Gc(J/m^2)': hist_x, 'probability': cdf_data})
    pdf_df = pd.DataFrame({'Gc(J/m^2)': x, 'probability_density': pdf})
    cdf_df = pd.DataFrame({'Gc(J/m^2)': x, 'probability': cdf})
    probability_df = pd.DataFrame({'Gc(J/m^2)': x, 'P': 1 - cdf})

    hist_df.to_csv('data/histogram.txt', index=False, sep='\t')
    cum_df.to_csv('data/cumulative.txt', index=False, sep='\t')
    pdf_df.to_csv('data/pdf.txt', index=False, sep='\t')
    cdf_df.to_csv('data/cdf.txt', index=False, sep='\t')
    probability_df.to_csv('data/probability.txt', index=False, sep='\t')
    return (
        cdf_data,
        cdf_df,
        cum_df,
        hist_bins,
        hist_df,
        hist_x,
        pdf_data,
        pdf_df,
        probability_df,
    )


@app.cell
def __(hist_x, pdf, pdf_data, plt, x):
    plt.bar(hist_x, pdf_data, width=.2, alpha=.3)
    plt.plot(x, pdf)
    return


@app.cell
def __(cdf, cdf_data, hist_x, plt, x):
    plt.bar(hist_x, cdf_data, width=.2, alpha=.3)
    plt.plot(x, cdf)
    return


@app.cell
def __(df_err):
    df_err['Gc_mean'].describe()
    return


@app.cell
def __(loc, scale, shape, stats):
    propability_function = lambda x: 1 - stats.lognorm.cdf(x, shape, loc, scale)
    return propability_function,


@app.cell
def __(propability_function):
    propability_function(1.2)
    return


@app.cell
def __(np, plt, propability_function, weac):
    # Parameters
    phi_ = 45  # positive is downslope, negative is upslope
    crack_lengths_ = np.arange(0, 500, 25)

    plt.figure()

    for profile_ in ['A', 'B', 'C', 'D', 'E', 'F', 'H']:

        P_ = []

        for a_ in crack_lengths_:

            # Initialize PST object
            pst_ = weac.Layered(system='-pst', layers=profile_)

            # Set weak-layer properties
            pst_.set_foundation_properties(t=20, E=0.15, update=True)

            # Calculate segmentation
            segments_ = pst_.calc_segments(phi=phi_, L=1000, a=a_)['crack']

            # Assemble model for segments and solve for free constants
            C_ = pst_.assemble_and_solve(phi=phi_, **segments_)

            # Calculate ERR and probability
            Gdif_ = pst_.gdif(C_, phi_, **segments_, unit='J/m^2')
            P_.append(propability_function(Gdif_[0]))

        plt.plot(crack_lengths_, P_, label=profile_)

    plt.legend()
    plt.show()
    # Calculate probability
    # propability_function(Gdif_[0])
    return (
        C_,
        Gdif_,
        P_,
        a_,
        crack_lengths_,
        phi_,
        profile_,
        pst_,
        segments_,
    )


@app.cell
def __(propability_function, weac):
    # Parameters
    phi = -45  # positive is downslope, negative is upslope
    t = 20
    E = 0.15
    a = 500

    # Initialize PST object
    pst = weac.Layered(system='-pst', layers='H')

    # Set weak-layer properties
    pst.set_foundation_properties(t=20, E=0.15, update=True)

    # Calculate segmentation
    segments = pst.calc_segments(phi=phi, L=1000, a=a)['crack']

    # Assemble model for segments and solve for free constants
    C = pst.assemble_and_solve(phi=phi, **segments)

    # Calculate ERR and probability
    Gdif = pst.gdif(C, phi, **segments, unit='J/m^2')

    # Calculate probability
    propability_function(Gdif[0])
    return C, E, Gdif, a, phi, pst, segments, t


if __name__ == "__main__":
    app.run()
