# Third-party imports
import random
import numpy as np
import streamlit as st
from st_screen_stats import ScreenData

# Local imports
from oracle.config import DENSITY_PARAMETERS, HAND_HARDNESS
from oracle import plot


def main():
    # Set page configuration
    st.set_page_config(page_title='ORACLE', layout='centered')

    # Display ORACLE logo and title
    display_header()

    # Initialize session state variables
    initialize_session_state()

    # Get and monitor the current screen width
    watch_screen_width()

    # Display layer headers
    display_snowprofile_header()

    # Handle 'Add layer' and 'Reset all layers' buttons
    handle_layer_buttons()

    # Handle layer removal if needed
    handle_layer_removal()

    # Handle layer movement if needed
    handle_layer_movement()

    # Create a placeholder for the layer table
    layer_table_placeholder = st.empty()

    # Render the layer table
    render_layer_table(layer_table_placeholder)

    # Display the weak-layer input
    show_weaklayer_input()

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

    if 'weaklayer_thickness' not in st.session_state:
        st.session_state['weaklayer_thickness'] = 30

    if 'layer_to_move_up' not in st.session_state:
        st.session_state['layer_to_move_up'] = None

    if 'layer_to_move_down' not in st.session_state:
        st.session_state['layer_to_move_down'] = None

    if "grain_options" not in st.session_state:
        st.session_state.grain_options = list(DENSITY_PARAMETERS.keys())

    if "hardness_options" not in st.session_state:
        st.session_state.hardness_options = list(HAND_HARDNESS.keys())[1:]


def watch_screen_width():
    # Write the screen data dict to the session state "screen_stats"
    screenD = ScreenData(setTimeout=1000)
    screenD.st_screen_data(key="screen_stats")


def display_snowprofile_header():
    """Displays the headers for the layer table."""
    st.markdown('#### Snow profile')


def handle_layer_buttons():
    """Handles the 'Add layer' and 'Reset all layers' buttons."""
    add_col, reset_col = st.columns([0.75, 0.25])
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

    def weighted_choice(options, weights):
        """Returns a weighted random choice from the options list."""
        return random.choices(options, weights=weights, k=1)[0]

    def generate_random_layer(
        layer_id, grain_options, hardness_options, max=10
    ):
        """Generates randomized values for a new layer, with layer_id biasing the selection."""

        # Adjust weights based on layer_id: lower layer_id means more bias towards the end of the list
        id = min(layer_id, max)
        n_grains = len(grain_options)
        n_hardness = len(hardness_options)

        # Stronger bias factor - increase the multiplier for stronger bias
        bias = 5

        # Linear interpolation for weights (stronger bias at layer_id = 0, uniform at layer_id = 10)
        grain_weights = np.linspace(1, bias * (max - id) / max, n_grains)
        hardness_weights = np.linspace(1, bias * (max - id) / max, n_hardness)

        # Normalize weights so they sum to 1 (random.choices needs normalized weights)
        grain_weights /= np.sum(grain_weights)
        hardness_weights /= np.sum(hardness_weights)

        # Randomize thickness: normal distribution with mean 100 and standard deviation 50,
        # round to the nearest 20, and clip to the range 20-220
        thickness = round(np.random.normal(100, 50) / 20) * 20
        thickness = np.clip(thickness, 20, 220)

        # Select grainform and hardness with weighted bias towards the end
        grainform = weighted_choice(grain_options, grain_weights)
        hardness = weighted_choice(hardness_options, hardness_weights)

        return thickness, grainform, hardness

    layer_id = st.session_state.layer_id_counter
    st.session_state.layer_id_counter += 1

    thickness, grainform, hardness = generate_random_layer(
        layer_id,
        st.session_state.grain_options[3:-2],
        st.session_state.hardness_options[:-4],
    )

    density = compute_density(grainform, hardness)

    layer = {
        'id': layer_id,
        'density': density,
        'thickness': thickness,
        'hardness': hardness,
        'grain': grainform,
    }

    # Append the new layer at the end of the list
    st.session_state.layers.append(layer)


def handle_layer_removal():
    """Removes a layer if the remove button was clicked, accounting for reversed display order."""
    if st.session_state['layer_to_remove'] is not None:
        layer_id_to_remove = st.session_state['layer_to_remove']
        # Since layers are displayed in reversed order, find the layer index accordingly
        for idx, layer in enumerate(reversed(st.session_state.layers)):
            if layer['id'] == layer_id_to_remove:
                # Calculate the actual index in st.session_state.layers
                actual_idx = len(st.session_state.layers) - 1 - idx
                # Remove the layer at the actual index
                del st.session_state.layers[actual_idx]
                break
        st.session_state['layer_to_remove'] = None  # Reset after removing


def handle_layer_movement():
    """Handles moving layers up or down, accounting for reversed display order."""
    layers = st.session_state.layers
    total_layers = len(layers)

    if st.session_state['layer_to_move_up'] is not None:
        layer_id = st.session_state['layer_to_move_up']
        # Since layers are displayed in reversed order, moving up increases the index
        for idx, layer in enumerate(reversed(layers)):
            if layer['id'] == layer_id:
                actual_idx = total_layers - 1 - idx
                if actual_idx < total_layers - 1:
                    # Swap with the next layer in the list to move up in display
                    layers[actual_idx], layers[actual_idx + 1] = (
                        layers[actual_idx + 1],
                        layers[actual_idx],
                    )
                break
        st.session_state['layer_to_move_up'] = None

    if st.session_state['layer_to_move_down'] is not None:
        layer_id = st.session_state['layer_to_move_down']
        # Since layers are displayed in reversed order, moving down decreases the index
        for idx, layer in enumerate(reversed(layers)):
            if layer['id'] == layer_id:
                actual_idx = total_layers - 1 - idx
                if actual_idx > 0:
                    # Swap with the previous layer in the list to move down in display
                    layers[actual_idx], layers[actual_idx - 1] = (
                        layers[actual_idx - 1],
                        layers[actual_idx],
                    )
                break
        st.session_state['layer_to_move_down'] = None


def render_layer_table(placeholder):
    """Renders the layer table with interactive widgets."""
    grain_options = st.session_state.grain_options
    hardness_options = st.session_state.hardness_options

    with placeholder.container():
        if len(st.session_state.layers) > 0:

            # Adjusted column layout to include move buttons
            if len(st.session_state.layers) > 1:
                col0, col1, col2, col3, col4, col5, col6 = st.columns(
                    [1.6, 4, 3, 3, 1.2, 1.2, 1.2], vertical_alignment='center'
                )
            else:
                col0, col1, col2, col3, col6 = st.columns(
                    [1.6, 4, 3, 3, 1.4], vertical_alignment='center'
                )
            if st.session_state["screen_stats"]['innerWidth'] > 640:
                with col1:
                    st.markdown('Layer thickness (mm)')
                with col2:
                    st.markdown('Grain form')
                with col3:
                    st.markdown('Hand hardness')

            # Reverse the layers to display the newest at the top
            layers = list(reversed(st.session_state.layers))

            # Display each layer
            for i, layer in enumerate(layers):
                layer_id = layer['id']
                with st.container():
                    if (
                        st.session_state["screen_stats"]['innerWidth'] > 640
                        and len(st.session_state.layers) > 1
                    ):
                        col0, col1, col2, col3, col4, col5, col6 = st.columns(
                            [1.6, 4, 3, 3, 1.2, 1.2, 1.2],
                            vertical_alignment='center',
                        )
                    else:
                        col4 = None
                        col5 = None
                        col0, col1, col2, col3, col6 = st.columns(
                            [1.6, 4, 3, 3, 1.4], vertical_alignment='center'
                        )

                    if st.session_state["screen_stats"]['innerWidth'] < 640:
                        label_visibility = 'visible'
                    else:
                        label_visibility = 'collapsed'

                    # Calculate the layer number
                    layer_number = i + 1

                    with col0:
                        st.markdown(f"Layer {layer_number}")
                    with col1:
                        st.number_input(
                            "Thickness (mm)",
                            label_visibility=label_visibility,
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
                            "Grain Form",
                            label_visibility=label_visibility,
                            options=grain_options,
                            index=grain_options.index(layer['grain']),
                            key=f"grainform_{layer_id}",
                            on_change=update_grainform,
                            args=(layer_id,),
                        )
                    with col3:
                        st.selectbox(
                            "Hand Hardness",
                            label_visibility=label_visibility,
                            options=hardness_options,
                            index=hardness_options.index(layer['hardness']),
                            key=f"hardness_{layer_id}",
                            on_change=update_hardness,
                            args=(layer_id,),
                        )
                    if col4 and col5 and len(st.session_state.layers) > 1:
                        with col4:
                            # Move down button
                            disabled = (
                                i == len(st.session_state.layers) - 1
                            )  # Disable for the last layer
                            st.button(
                                "&#x2935;",
                                key=f"move_down_{layer_id}",
                                use_container_width=True,
                                on_click=move_layer_down,
                                args=(layer_id,),
                                disabled=disabled,
                                type='secondary',
                            )
                        with col5:
                            # Move up button
                            disabled = i == 0  # Disable for the first layer
                            st.button(
                                label="&#x2934;",
                                key=f"move_up_{layer_id}",
                                use_container_width=True,
                                on_click=move_layer_up,
                                args=(layer_id,),
                                disabled=disabled,
                                type='secondary',
                            )
                    with col6:
                        if label_visibility == 'visible':
                            st.markdown(
                                """<p style="font-size: 14px; margin-bottom: 0;">Remove</p>""",
                                unsafe_allow_html=True,
                            )
                        st.button(
                            "🗑️",
                            key=f"remove_{layer_id}",
                            use_container_width=True,
                            on_click=remove_layer,
                            args=(layer_id,),
                        )
        else:
            pass
            # st.write("_Please add a first layer to be displayed here..._")


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


def move_layer_up(layer_id):
    """Triggers moving a layer up."""
    st.session_state['layer_to_move_up'] = layer_id


def move_layer_down(layer_id):
    """Triggers moving a layer down."""
    st.session_state['layer_to_move_down'] = layer_id


def show_weaklayer_input():
    """Display the weak-layer thickness number input."""

    col1, col2 = st.columns([0.27, 0.73], vertical_alignment='center')

    with col1:
        st.write('Weak-layer thickness (mm)')
    with col2:
        st.session_state['weaklayer_thickness'] = st.number_input(
            label="Weak-layer thickness (mm)",
            label_visibility='collapsed',
            min_value=1,
            max_value=100,
            value=30,
            step=5,
        )


def display_plot():
    """Displays the snow stratification plot."""

    # Plot snow stratification using Plotly
    weak_layer_thickness = st.session_state['weaklayer_thickness']
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
# User input in inches
# User input: enter layers top to bottom or bottom to top (tabs)
# User input: inclination
# User input: cutting direction
# User input: slab faces (normal, vertical)
# User input: column length
# User input: cut length
# Run WEAC to compute ERR
