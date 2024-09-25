# Third-party imports
import streamlit as st

# Local imports
from oracle.config import DENSITY_PARAMETERS, HAND_HARDNESS
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

if "layer_id_counter" not in st.session_state:
    st.session_state.layer_id_counter = 0

# Functions to compute density
def compute_density(grainform, hardness):
    a, b = DENSITY_PARAMETERS[grainform]
    hardness = HAND_HARDNESS[hardness]
    if grainform == "RG":
        return a + b * (hardness ** 3.15)
    else:
        return a + b * hardness

grain_options = list(DENSITY_PARAMETERS.keys())
hardness_options = list(HAND_HARDNESS.keys())[1:]

st.markdown('#### Layers')
col1, col2, col3, col4 = st.columns([3.5, 3.5, 3.5, 1], vertical_alignment='bottom')
with col1:
    st.markdown('Layer thickness (mm)')
with col2:
    st.markdown('Grain form')
with col3:
    st.markdown('Hand hardness')
with col4:
    st.markdown('Delete')

# --- Create a placeholder for the layer table ---
layer_table_placeholder = st.empty()

# --- Place the buttons below the layer table ---
add_col, reset_col = st.columns([0.8, 0.2])
with add_col:
    add_layer_clicked = st.button("Add layer", use_container_width=True, type='primary')
with reset_col:
    reset_layers_clicked = st.button("Reset all layers", use_container_width=True)

# --- Handle button clicks before rendering the table ---
if reset_layers_clicked:
    st.session_state.layers = []
    st.session_state.layer_id_counter = 0

if add_layer_clicked:
    num_layer_thickness = 100  # Default thickness in mm
    drop_grainform = 'RG'      # Default grain form
    num_hardness = '4F'        # Default hand hardness

    density = compute_density(drop_grainform, num_hardness)

    layer_id = st.session_state.layer_id_counter
    st.session_state.layer_id_counter += 1

    layer = {
        'id': layer_id,
        'density': density,
        'thickness': num_layer_thickness,
        'hardness': num_hardness,
        'grain': drop_grainform,
    }

    st.session_state.layers.insert(0, layer)


# --- Render the layer table in the placeholder ---
# Define a function to remove a layer
def remove_layer(layer_id):
    st.session_state['layer_to_remove'] = layer_id
    
# Define functions to update layer properties
def update_thickness(layer_id):
    # Find the layer with the given layer_id and update its thickness
    for layer in st.session_state.layers:
        if layer['id'] == layer_id:
            layer['thickness'] = st.session_state[f"thickness_{layer_id}"]
            break

def update_grainform(layer_id):
    # Find the layer with the given layer_id and update its grain form and density
    for layer in st.session_state.layers:
        if layer['id'] == layer_id:
            layer['grain'] = st.session_state[f"grainform_{layer_id}"]
            layer['density'] = compute_density(layer['grain'], layer['hardness'])
            break

def update_hardness(layer_id):
    # Find the layer with the given layer_id and update its hardness and density
    for layer in st.session_state.layers:
        if layer['id'] == layer_id:
            layer['hardness'] = st.session_state[f"hardness_{layer_id}"]
            layer['density'] = compute_density(layer['grain'], layer['hardness'])
            break

# Initialize the 'layer_to_remove' in session_state if not already set
if 'layer_to_remove' not in st.session_state:
    st.session_state['layer_to_remove'] = None

# Check if a layer needs to be removed and update the session state
if st.session_state['layer_to_remove'] is not None:
    layer_id_to_remove = st.session_state['layer_to_remove']
    st.session_state.layers = [l for l in st.session_state.layers if l['id'] != layer_id_to_remove]
    st.session_state['layer_to_remove'] = None  # Reset after removing

with layer_table_placeholder.container():
    if len(st.session_state.layers) > 0:
        col1, col2, col3, col4 = st.columns([3.5, 3.5, 3.5, 1], vertical_alignment='bottom')
        
        # Display layers
        for layer in reversed(st.session_state.layers):
            layer_id = layer['id']
            with col1:
                st.number_input(
                    f"Thickness (mm) of layer {layer_id}",
                    label_visibility='collapsed',
                    min_value=1,
                    max_value=1000,
                    value=int(layer['thickness']),
                    step=10,
                    key=f"thickness_{layer_id}",
                    on_change=update_thickness,
                    args=(layer_id,),
                )
            with col2:
                st.selectbox(
                    f"Grain form of layer {layer_id}",
                    label_visibility='collapsed',
                    options=grain_options,
                    index=grain_options.index(layer['grain']),
                    key=f"grainform_{layer_id}",
                    on_change=update_grainform,
                    args=(layer_id,),
                )
            with col3:
                st.selectbox(
                    f"Hand hardness of layer {layer_id}",
                    label_visibility='collapsed',
                    options=hardness_options,
                    index=hardness_options.index(layer['hardness']),
                    key=f"hardness_{layer_id}",
                    on_change=update_hardness,
                    args=(layer_id,),
                )
            with col4:
                st.button(
                    "🗑️",
                    key=f"remove_{layer_id}",
                    use_container_width=True,
                    on_click=remove_layer,
                    args=(layer_id,),
                )

# --- Continue with the rest of your app ---

# Set weak-layer thickness
wl_thickness = st.number_input(
    "Weak-layer thickness (mm)",
    min_value=1,
    max_value=100,
    value=30,
    step=1,
)

# Plot snow stratification using Plotly
if len(st.session_state.layers) > 0:
    fig = plot.snow_stratification_plotly(
        wl_thickness,
        [(layer['density'], layer['thickness'], layer['hardness']) for layer in st.session_state.layers],
        [layer['grain'] for layer in st.session_state.layers],
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No layers to display.")

# Display the TODO list (Optional)
st.markdown(
    """
    <h2>⏳ TODO</h2>
    <ul style="list-style-type: none;">
      <li><input type="checkbox"> User input: inclination</li>
      <li><input type="checkbox"> User input: cutting direction</li>
      <li><input type="checkbox"> User input: slab faces (normal, vertical)</li>
      <li><input type="checkbox"> User input: column length</li>
      <li><input type="checkbox"> User input: cut length</li>
      <li><input type="checkbox"> Run WEAC to compute ERR</li>
    </ul>
    """,
    unsafe_allow_html=True,
)