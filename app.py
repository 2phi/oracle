import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from oracle.config import DENSITY_PARAMETERS
from oracle import plot

# Set page configuration
st.set_page_config(page_title="ORACLE", layout="centered")

# Display ORACLE logo and title
st.html(
    """
    <div style="text-align: center;">
        <img src="https://github.com/2phi/oracle/raw/main/img/steampunk-v1.png" alt="ORACLE" width="200">
        <h1>ORACLE</h1>
        <p><b>Observation, Research, and Analysis of<br>Collapse and Loading Experiments</b></p>
    </div>
    """
)

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

    a, b = DENSITY_PARAMETERS[drop_grainform]

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
    a, b = DENSITY_PARAMETERS[new_grainform]
    if new_grainform == "RG":
        density = a + b * (current_hardness**3.15)
    else:
        density = a + b * current_hardness

    st.session_state.grain_list[index] = new_grainform
    st.session_state.layers[index][0] = density


def update_hand_hardness(index, new_hardness):
    current_grainform = st.session_state.grain_list[index]
    (
        a,
        b,
    ) = DENSITY_PARAMETERS[current_grainform]
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
st.markdown('#### Layers')


col1, col2, col3, col4 = st.columns(4, vertical_alignment='bottom')

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
    grain_options = list(DENSITY_PARAMETERS.keys())
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
with col4:
    if st.button("Add", use_container_width=True):
        add_layer()

# Set weak-layer thickness
wl_thickness = st.number_input(
    "Weak-layer thickness (mm)",
    min_value=1,
    max_value=100,
    value=30,
    step=1,
)

st.markdown('---')

# Display the table of layers
if len(st.session_state.layers) > 0:
    # st.markdown("#### Layers")
    col1, col2, col3, col4 = st.columns(
        [3.5, 3.5, 3.5, 1], vertical_alignment='bottom'
    )
    with col1:
        st.markdown('Layer thickness (mm)')
    with col2:
        st.markdown('Grain form')
    with col3:
        st.markdown('Hand hardness')
    with col4:
        st.markdown('Delete')
    for i, (layer, grain) in reversed(
        list(
            enumerate(
                zip(st.session_state.layers, st.session_state.grain_list)
            )
        )
    ):
        with col1:
            new_thickness = st.number_input(
                f"Thickness (mm) for layer {i+1}",
                label_visibility='collapsed',
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
                label_visibility='collapsed',
                options=grain_options,
                index=grain_options.index(grain),
                key=f"grainform_{i}",
            )
            if new_grainform != grain:
                update_grainform(i, new_grainform)
        with col3:
            new_hardness = st.number_input(
                f"Hand hardness for layer {i+1}",
                label_visibility='collapsed',
                min_value=1,
                max_value=5,
                value=int(layer[2]),
                step=1,
                key=f"hardness_{i}",
            )
            if new_hardness != layer[2]:
                update_hand_hardness(i, new_hardness)
        with col4:
            if st.button("🗑️", key=f"remove_{i}", use_container_width=True):
                delete_layer(i)

# _, col1 = st.columns([.7, .3])
# with col1:
if st.button("Reset all layers", use_container_width=True):
    reset_all_layers()



# Plot snow stratification using Plotly
# if len(st.session_state.layers) > 0:
fig = plot.snow_stratification_plotly(
    wl_thickness, st.session_state.layers, st.session_state.grain_list
)
st.plotly_chart(fig, use_container_width=True)
# else:
#     st.write("No layers to display.")


# Plot snow stratification
if len(st.session_state.layers) > 0:
    fig2 = plot.snow_stratification(
        wl_thickness, st.session_state.layers, st.session_state.grain_list
    )
    st.pyplot(fig2)
else:
    st.write("No layers to display.")

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
