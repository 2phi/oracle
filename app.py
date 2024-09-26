# Third-party imports
import streamlit as st

# Local imports
from oracle.config import DENSITY_PARAMETERS, HAND_HARDNESS
from oracle import plot


def main():
    # Set page configuration
    st.set_page_config(page_title="ORACLE", layout="centered")

    # Display ORACLE logo and title
    display_header()

    # Initialize session state variables
    initialize_session_state()

    # Get options for grain forms and hand hardness
    grain_options = list(DENSITY_PARAMETERS.keys())
    hardness_options = list(HAND_HARDNESS.keys())[1:]

    # Display layer headers
    display_layer_headers()

    # Create a placeholder for the layer table
    layer_table_placeholder = st.empty()

    # Handle 'Add layer' and 'Reset all layers' buttons
    handle_layer_buttons()

    # Handle layer removal if needed
    handle_layer_removal()

    # Render the layer table
    render_layer_table(
        grain_options, hardness_options, layer_table_placeholder
    )

    # Display the plot
    display_plot()


def display_header():
    """Displays the ORACLE logo and title."""
    st.html(
        """
        <div style="text-align: center;">
            <img src="https://github.com/2phi/oracle/raw/main/img/steampunk-v1.png" alt="ORACLE" width="200">
            <h1>ORACLE</h1>
            <p><b>Observation, Research, and Analysis of<br>Collapse and Loading Experiments</b></p>
        </div>
        """
    )


def initialize_session_state():
    """Initializes session state variables."""
    if "layers" not in st.session_state:
        st.session_state.layers = []

    if "layer_id_counter" not in st.session_state:
        st.session_state.layer_id_counter = 0

    if 'layer_to_remove' not in st.session_state:
        st.session_state['layer_to_remove'] = None


def display_layer_headers():
    """Displays the headers for the layer table."""
    st.markdown('#### Layers')


def handle_layer_buttons():
    """Handles the 'Add layer' and 'Reset all layers' buttons."""
    add_col, reset_col = st.columns([0.8, 0.2])
    with add_col:
        add_layer_clicked = st.button(
            "Add layer", use_container_width=True, type='primary'
        )
    with reset_col:
        reset_layers_clicked = st.button(
            "Reset all layers", use_container_width=True
        )

    if reset_layers_clicked:
        st.session_state.layers = []
        st.session_state.layer_id_counter = 0

    if add_layer_clicked:
        add_new_layer()


def add_new_layer():
    """Adds a new layer with default values."""
    default_thickness = 100  # Default thickness in mm
    default_grainform = 'RG'  # Default grain form
    default_hardness = '4F'  # Default hand hardness

    density = compute_density(default_grainform, default_hardness)

    layer_id = st.session_state.layer_id_counter
    st.session_state.layer_id_counter += 1

    layer = {
        'id': layer_id,
        'density': density,
        'thickness': default_thickness,
        'hardness': default_hardness,
        'grain': default_grainform,
    }

    st.session_state.layers.insert(0, layer)


def handle_layer_removal():
    """Removes a layer if the remove button was clicked."""
    if st.session_state['layer_to_remove'] is not None:
        layer_id_to_remove = st.session_state['layer_to_remove']
        st.session_state.layers = [
            l for l in st.session_state.layers if l['id'] != layer_id_to_remove
        ]
        st.session_state['layer_to_remove'] = None  # Reset after removing


def render_layer_table(grain_options, hardness_options, placeholder):
    """Renders the layer table with interactive widgets."""
    with placeholder.container():
        if len(st.session_state.layers) > 0:

            col0, col1, col2, col3, col4 = st.columns(
                [1.5, 4, 3, 3, 1.4], vertical_alignment='center'
            )
            with col0:
                st.markdown('Order')
            with col1:
                st.markdown('Layer thickness (mm)')
            with col2:
                st.markdown('Grain form')
            with col3:
                st.markdown('Hand hardness')
            with col4:
                st.markdown('Delete')

            # Display each layer
            for i, layer in enumerate(reversed(st.session_state.layers)):
                layer_id = layer['id']
                with st.container():
                    col0, col1, col2, col3, col4 = st.columns(
                        [1.5, 4, 3, 3, 1.4], vertical_alignment='center'
                    )
                    with col0:
                        st.markdown(f"Layer {layer_id + 1}")
                    with col1:
                        st.number_input(
                            f"Thickness (mm) of layer {layer_id}",
                            label_visibility='collapsed',
                            format='%d',
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
        else:
            st.write("_Please add a first layer to be displayed here..._")


def compute_density(grainform, hardness):
    """Computes the density based on grain form and hand hardness."""
    a, b = DENSITY_PARAMETERS[grainform]
    hardness_value = HAND_HARDNESS[hardness]
    if grainform == "RG":
        return a + b * (hardness_value**3.15)
    else:
        return a + b * hardness_value


def update_thickness(layer_id):
    """Updates the thickness of a layer."""
    for layer in st.session_state.layers:
        if layer['id'] == layer_id:
            layer['thickness'] = st.session_state[f"thickness_{layer_id}"]
            break


def update_grainform(layer_id):
    """Updates the grain form and density of a layer."""
    for layer in st.session_state.layers:
        if layer['id'] == layer_id:
            layer['grain'] = st.session_state[f"grainform_{layer_id}"]
            layer['density'] = compute_density(
                layer['grain'], layer['hardness']
            )
            break


def update_hardness(layer_id):
    """Updates the hand hardness and density of a layer."""
    for layer in st.session_state.layers:
        if layer['id'] == layer_id:
            layer['hardness'] = st.session_state[f"hardness_{layer_id}"]
            layer['density'] = compute_density(
                layer['grain'], layer['hardness']
            )
            break


def remove_layer(layer_id):
    """Triggers the removal of a layer."""
    st.session_state['layer_to_remove'] = layer_id


def display_plot():
    """Displays the snow stratification plot."""
    # Set weak-layer thickness
    weak_layer_thickness = st.number_input(
        "Weak-layer thickness (mm)",
        min_value=1,
        max_value=100,
        value=30,
        step=5,
    )

    # Plot snow stratification using Plotly
    layers_data = [
        (layer['density'], layer['thickness'], layer['hardness'])
        for layer in st.session_state.layers
    ]
    grains = [layer['grain'] for layer in st.session_state.layers]

    fig = plot.snow_profile(weak_layer_thickness, layers_data, grains)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            'displayModeBar': False,
            'scrollZoom': False,
            'staticPlot': True,
        },
    )


if __name__ == "__main__":
    main()

# tudu
# Abbreviations for H, D, F, R (hardness, density, grain form, hand hardness) with legends below plot
# User input: enter layers top to bottom or bottom to top (tabs)
# User input: inclination
# User input: cutting direction
# User input: slab faces (normal, vertical)
# User input: column length
# User input: cut length
# Run WEAC to compute ERR
