import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="ORACLE", layout="centered")

# Hide Streamlit's default styles to allow custom HTML/CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.html(
    body="""
        <style>
            /* hide hyperlink anchors generated next to headers */
            h1 > div > a {
                display: none !important;
            }
            h2 > div > a {
                display: none !important;
            }
            h3 > div > a {
                display: none !important;
            }
            h4 > div > a {
                display: none !important;
            }
            h5 > div > a {
                display: none !important;
            }
            h6 > div > a {
                display: none !important;
            }
        </style>
    """,
)

# Display the title and image
st.html(
    """
    <div style="text-align: center;">
        <img src="https://github.com/2phi/oracle/raw/main/img/steampunk-v1.png" alt="ORACLE" width="200">
        <h1>ORACLE</h1>
        <p><b>Observation, Research, and Analysis of<br>Collapse and Loading Experiments</b></p>
    </div>
    """
)

# Display the TODO list (Optional)
st.html(
    """
    <h2 style= >⏳ tudu</h2>
    <ul style="list-style-type: none;">
      <li><input type="checkbox"> User input: inclination</li>
      <li><input type="checkbox"> User input: cutting direction</li>
      <li><input type="checkbox"> User input: slab faces (normal, vertical)</li>
      <li><input type="checkbox"> User input: column length</li>
      <li><input type="checkbox"> User input: cut length</li>
      <li><input type="checkbox"> Run WEAC to compute ERR</li>
    </ul>
    """
)

# Grainforms data
grainforms = [
    (0, "PP", 45, 36, "Precipitation particles"),
    (1, "PPgp", 83, 37, "Graupel"),
    (2, "DF", 65, 36, "Decomposing and fragmented precipitation particles"),
    (3, "RG", 154, 1.51, "Rounded grains"),
    (4, "RGmx", 91, 42, "Rounded mixed forms"),
    (5, "FC", 112, 46, "Faceted crystals"),
    (6, "FCmx", 56, 64, "Faceted mixed forms"),
    (7, "DH", 185, 25, "Depth hoar"),
    (8, "MFCr", 292.25, 0, "Melt-freeze crusts"),
]

grainform_df = pd.DataFrame(
    grainforms, columns=["id", "abbreviation", "a", "b", "description"]
)

# Display the grainform table
# st.table(grainform_df.drop(columns=["id"]))


# Function to plot density vs hand hardness
def density_vs_hand_hardness(grainform_df):
    hand_hardness = np.arange(1, 6, 0.1)
    fig, ax = plt.subplots(figsize=(8, 5))

    for index, row in grainform_df.iterrows():
        a = row["a"]
        b = row["b"]
        if row["abbreviation"] == "RG":
            densities = a + b * (hand_hardness**3.15)
        else:
            densities = a + b * hand_hardness
        ax.plot(hand_hardness, densities, label=row["abbreviation"])

    ax.set_ylim(50, 450)
    ax.set_xlim(1, 5)
    ax.set_xlabel("Hand Hardness")
    ax.set_ylabel("Density (kg/m³)")
    ax.set_title("Density vs Hand Hardness for Different Grain Types")
    ax.legend(loc="best")
    ax.grid(True)
    return fig


# Plot and display the density vs hand hardness plot
fig = density_vs_hand_hardness(grainform_df)
# st.pyplot(fig)


# Initialize session state variables
if "layers" not in st.session_state:
    st.session_state.layers = []

if "grain_list" not in st.session_state:
    st.session_state.grain_list = []


# Functions to modify session state
def add_layer():
    num_layer_thickness = st.session_state["layer_thickness"]
    drop_grainform = st.session_state["grainform"]
    num_hardness = st.session_state["hand_hardness"]

    grainform_row = grainform_df.loc[
        grainform_df["abbreviation"] == drop_grainform
    ]
    a = grainform_row["a"].values[0]
    b = grainform_row["b"].values[0]
    if drop_grainform == "RG":
        density = a + b * (num_hardness**3.15)
    else:
        density = a + b * num_hardness

    st.session_state.layers.append(
        [density, num_layer_thickness, num_hardness]
    )
    st.session_state.grain_list.append(drop_grainform)


def reset_all_layers():
    st.session_state.layers = []
    st.session_state.grain_list = []


def update_thickness(index, new_thickness):
    st.session_state.layers[index][1] = new_thickness


def update_grainform(index, new_grainform):
    current_hardness = st.session_state.layers[index][2]
    grainform_row = grainform_df.loc[
        grainform_df["abbreviation"] == new_grainform
    ]
    a = grainform_row["a"].values[0]
    b = grainform_row["b"].values[0]
    if new_grainform == "RG":
        density = a + b * (current_hardness**3.15)
    else:
        density = a + b * current_hardness

    st.session_state.grain_list[index] = new_grainform
    st.session_state.layers[index][0] = density


def update_hand_hardness(index, new_hardness):
    current_grainform = st.session_state.grain_list[index]
    grainform_row = grainform_df.loc[
        grainform_df["abbreviation"] == current_grainform
    ]
    a = grainform_row["a"].values[0]
    b = grainform_row["b"].values[0]
    if current_grainform == "RG":
        density = a + b * (new_hardness**3.15)
    else:
        density = a + b * new_hardness

    st.session_state.layers[index][2] = new_hardness
    st.session_state.layers[index][0] = density


def delete_layer(index):
    st.session_state.layers.pop(index)
    st.session_state.grain_list.pop(index)
    # No need for st.experimental_rerun()


# UI for adding layers
st.markdown(
    '<h2>✍🏼 Add layers bottom to top</h2>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.number_input(
        "Layer thickness (mm)",
        min_value=1,
        max_value=1000,
        value=100,
        step=1,
        key="layer_thickness",
    )
with col2:
    grain_options = grainform_df["abbreviation"].tolist()
    st.selectbox("Grain type", options=grain_options, index=1, key="grainform")
with col3:
    st.number_input(
        "Hand hardness",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        key="hand_hardness",
    )

col4, col5 = st.columns(2)

with col4:
    if st.button("Add layer"):
        add_layer()
with col5:
    if st.button("Reset all layers"):
        reset_all_layers()

# Display the table of layers
if len(st.session_state.layers) > 0:
    st.markdown("### Layers")
    for i, (layer, grain) in enumerate(
        zip(st.session_state.layers, st.session_state.grain_list)
    ):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            new_thickness = st.number_input(
                f"Thickness (mm) for layer {i+1}",
                min_value=1,
                max_value=1000,
                value=int(layer[1]),
                step=1,
                key=f"thickness_{i}",
            )
            if new_thickness != layer[1]:
                update_thickness(i, new_thickness)
        with col2:
            new_grainform = st.selectbox(
                f"Grain form for layer {i+1}",
                options=grain_options,
                index=grain_options.index(grain),
                key=f"grainform_{i}",
            )
            if new_grainform != grain:
                update_grainform(i, new_grainform)
        with col3:
            new_hardness = st.number_input(
                f"Hand hardness for layer {i+1}",
                min_value=1,
                max_value=5,
                value=int(layer[2]),
                step=1,
                key=f"hardness_{i}",
            )
            if new_hardness != layer[2]:
                update_hand_hardness(i, new_hardness)
        with col4:
            if st.button(f"Remove layer {i+1}", key=f"remove_{i}"):
                delete_layer(i)
                # No need for st.experimental_rerun()
                # The app will automatically rerun on state change

# Set weak-layer thickness
wl_thickness = st.number_input(
    "Set weak-layer thickness (mm)",
    min_value=1,
    max_value=100,
    value=30,
    step=1,
)


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

    ax.set_title("Snow Stratification", fontsize=14)
    return fig


# Plot snow stratification
if len(st.session_state.layers) > 0:
    fig2 = snow_stratification(
        wl_thickness, st.session_state.layers, st.session_state.grain_list
    )
    st.pyplot(fig2)
else:
    st.write("No layers to display.")
