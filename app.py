# Third-party imports
import random
import numpy as np
import streamlit as st
from st_screen_stats import ScreenData

# Local imports
from oracle.config import DENSITY_PARAMETERS, HAND_HARDNESS
from oracle import plot


def main():
    """Main function to run the Streamlit app."""
    # Set page configuration
    st.set_page_config(page_title='ORACLE', layout='centered')

    # Display the ORACLE logo and title
    display_header()

    # Initialize session state variables
    initialize_session_state()

    # Monitor screen width for responsive design
    watch_screen_width()

    # Display header for the snow profile section
    st.markdown('#### Snow profile')

    # Handle 'Add layer' and 'Reset all layers' buttons
    handle_layer_buttons()

    # Handle layer removal and movement
    handle_layer_actions()

    # Render the layer table
    render_layer_table()

    # Display weak-layer thickness input
    show_weaklayer_input()

    # Display the snow stratification plot
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
        unsafe_allow_html=True,
    )


def initialize_session_state():
    """Initializes session state variables."""
    state = st.session_state
    # Set default values for session state variables if they don't exist
    state.setdefault('layers', [])
    state.setdefault('layer_id_counter', 0)
    state.setdefault('layer_to_remove', None)
    state.setdefault('weaklayer_thickness', 30)
    state.setdefault('layer_to_move_up', None)
    state.setdefault('layer_to_move_down', None)
    state.setdefault('grain_options', list(DENSITY_PARAMETERS.keys()))
    state.setdefault('hardness_options', list(HAND_HARDNESS.keys())[1:])


def watch_screen_width():
    """Monitors the screen width for responsive layout."""
    screenD = ScreenData(setTimeout=1000)
    # Store screen data in session state
    screenD.st_screen_data(key="screen_stats")


def handle_layer_buttons():
    """Handles 'Add layer' and 'Reset all layers' buttons."""
    # Create columns for buttons
    add_col, reset_col = st.columns([0.75, 0.25])

    with add_col:
        # 'Add layer' button
        add_layer_clicked = st.button(
            "Add layer", use_container_width=True, type='primary'
        )

    with reset_col:
        # 'Reset all layers' button
        reset_layers_clicked = st.button(
            "Reset all layers", use_container_width=True
        )

    if reset_layers_clicked:
        # Reset layers and layer ID counter
        st.session_state.layers = []
        st.session_state.layer_id_counter = 0

    if add_layer_clicked:
        # Add a new layer
        add_new_layer()


def add_new_layer():
    """Adds a new layer with randomized default values."""

    def weighted_choice(options, weights):
        """Selects an option based on provided weights."""
        return random.choices(options, weights=weights, k=1)[0]

    def generate_random_layer(
        layer_id, grain_options, hardness_options, max_layers=10
    ):
        """Generates random properties for a new layer."""
        # Adjust the bias based on the layer ID
        id = min(layer_id, max_layers)
        n_grains = len(grain_options)
        n_hardness = len(hardness_options)
        bias = 5  # Bias factor for weighted choices

        # Create weights for grain options
        grain_weights = np.linspace(
            1, bias * (max_layers - id) / max_layers, n_grains
        )
        grain_weights /= np.sum(grain_weights)

        # Create weights for hardness options
        hardness_weights = np.linspace(
            1, bias * (max_layers - id) / max_layers, n_hardness
        )
        hardness_weights /= np.sum(hardness_weights)

        # Generate random thickness
        thickness = (
            round(np.clip(np.random.normal(100, 50), 20, 220) / 20) * 20
        )
        thickness = int(thickness)

        # Select grain form and hardness using weighted choices
        grainform = weighted_choice(grain_options, grain_weights)
        hardness = weighted_choice(hardness_options, hardness_weights)

        return thickness, grainform, hardness

    # Get a new layer ID
    layer_id = st.session_state.layer_id_counter
    st.session_state.layer_id_counter += 1

    # Generate random properties for the new layer
    thickness, grainform, hardness = generate_random_layer(
        layer_id,
        st.session_state.grain_options[3:-2],
        st.session_state.hardness_options[:-4],
    )

    # Compute density based on grain form and hardness
    density = compute_density(grainform, hardness)

    # Create the new layer dictionary
    layer = {
        'id': layer_id,
        'density': density,
        'thickness': thickness,
        'hardness': hardness,
        'grain': grainform,
    }

    # Insert the new layer at the beginning of the list (top)
    st.session_state.layers.insert(0, layer)


def compute_density(grainform, hardness):
    """Computes the density based on grain form and hand hardness."""
    a, b = DENSITY_PARAMETERS[grainform]
    hardness_value = HAND_HARDNESS[hardness]

    if grainform == "RG":
        # Special computation for 'RG' grain form
        return a + b * (hardness_value**3.15)
    else:
        return a + b * hardness_value


def handle_layer_actions():
    """Handles layer removal and movement actions."""
    if st.session_state['layer_to_remove'] is not None:
        # Remove the specified layer
        layer_id = st.session_state['layer_to_remove']
        idx = get_layer_index(layer_id)
        if idx is not None:
            del st.session_state.layers[idx]
        st.session_state['layer_to_remove'] = None

    if st.session_state['layer_to_move_up'] is not None:
        # Move the specified layer up
        layer_id = st.session_state['layer_to_move_up']
        idx = get_layer_index(layer_id)
        if idx is not None and idx > 0:
            st.session_state.layers[idx], st.session_state.layers[idx - 1] = (
                st.session_state.layers[idx - 1],
                st.session_state.layers[idx],
            )
        st.session_state['layer_to_move_up'] = None

    if st.session_state['layer_to_move_down'] is not None:
        # Move the specified layer down
        layer_id = st.session_state['layer_to_move_down']
        idx = get_layer_index(layer_id)
        if idx is not None and idx < len(st.session_state.layers) - 1:
            st.session_state.layers[idx], st.session_state.layers[idx + 1] = (
                st.session_state.layers[idx + 1],
                st.session_state.layers[idx],
            )
        st.session_state['layer_to_move_down'] = None


def get_layer_index(layer_id):
    """Returns the index of the layer with the given ID."""
    for idx, layer in enumerate(st.session_state.layers):
        if layer['id'] == layer_id:
            return idx
    return None


def render_layer_table():
    """Renders the layer table with interactive widgets."""
    grain_options = st.session_state.grain_options
    hardness_options = st.session_state.hardness_options
    layers = st.session_state.layers
    num_layers = len(layers)
    screen_width = st.session_state["screen_stats"]['innerWidth']

    # Determine label visibility based on screen width
    label_visibility = 'visible' if screen_width < 640 else 'collapsed'

    if num_layers > 0:
        # Display table headers based on screen width
        if screen_width > 640:
            # Define column widths
            header_cols = [1.6, 4, 3, 3]
            if num_layers > 1:
                header_cols.extend(
                    [1.2, 1.2, 1.2]
                )  # Add columns for movement buttons
            else:
                header_cols.append(1.4)  # Only remove button

            # Create header columns
            cols = st.columns(header_cols)

            with cols[1]:
                st.markdown('Layer thickness (mm)')
            with cols[2]:
                st.markdown('Grain form')
            with cols[3]:
                st.markdown('Hand hardness')

        # Display each layer
        for i, layer in enumerate(layers):
            layer_id = layer['id']

            # Define column widths for layer rows
            row_cols = [1.6, 4, 3, 3]
            if num_layers > 1:
                row_cols.extend(
                    [1.2, 1.2, 1.2]
                )  # Add columns for movement buttons
            else:
                row_cols.append(1.4)  # Only remove button

            # Create columns for the layer
            cols = st.columns(row_cols)

            # Unpack columns
            col_label, col_thickness, col_grain, col_hardness = cols[:4]
            col_move_down = cols[4] if num_layers > 1 else None
            col_move_up = cols[5] if num_layers > 1 else None
            col_remove = cols[-1]

            with col_label:
                st.markdown(f"Layer {i + 1}")

            with col_thickness:
                # Thickness input
                st.number_input(
                    "Thickness (mm)",
                    label_visibility=label_visibility,
                    min_value=1,
                    max_value=1000,
                    value=int(layer['thickness']),
                    step=10,
                    key=f"thickness_{layer_id}",
                    on_change=update_thickness,
                    args=(layer_id,),
                )

            with col_grain:
                # Grain form selection
                st.selectbox(
                    "Grain Form",
                    label_visibility=label_visibility,
                    options=grain_options,
                    index=grain_options.index(layer['grain']),
                    key=f"grainform_{layer_id}",
                    on_change=update_grainform,
                    args=(layer_id,),
                )

            with col_hardness:
                # Hand hardness selection
                st.selectbox(
                    "Hand Hardness",
                    label_visibility=label_visibility,
                    options=hardness_options,
                    index=hardness_options.index(layer['hardness']),
                    key=f"hardness_{layer_id}",
                    on_change=update_hardness,
                    args=(layer_id,),
                )

            if num_layers > 1 and col_move_down and col_move_up:
                with col_move_down:
                    # Move layer down button
                    disabled = (
                        i == num_layers - 1
                    )  # Disable for the bottom layer
                    st.button(
                        "&#x2935;",
                        key=f"move_down_{layer_id}",
                        use_container_width=True,
                        on_click=move_layer_down,
                        args=(layer_id,),
                        disabled=disabled,
                        type='secondary',
                    )

                with col_move_up:
                    # Move layer up button
                    disabled = i == 0  # Disable for the top layer
                    st.button(
                        "&#x2934;",
                        key=f"move_up_{layer_id}",
                        use_container_width=True,
                        on_click=move_layer_up,
                        args=(layer_id,),
                        disabled=disabled,
                        type='secondary',
                    )

            with col_remove:
                if label_visibility == 'visible':
                    st.markdown("Remove")
                # Remove layer button
                st.button(
                    "🗑️",
                    key=f"remove_{layer_id}",
                    use_container_width=True,
                    on_click=remove_layer,
                    args=(layer_id,),
                )


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
            # Update grain form
            layer['grain'] = st.session_state[f"grainform_{layer_id}"]
            # Recompute density
            layer['density'] = compute_density(
                layer['grain'], layer['hardness']
            )
            break


def update_hardness(layer_id):
    """Updates the hand hardness and density of a layer."""
    for layer in st.session_state.layers:
        if layer['id'] == layer_id:
            # Update hand hardness
            layer['hardness'] = st.session_state[f"hardness_{layer_id}"]
            # Recompute density
            layer['density'] = compute_density(
                layer['grain'], layer['hardness']
            )
            break


def remove_layer(layer_id):
    """Sets the layer to be removed."""
    st.session_state['layer_to_remove'] = layer_id


def move_layer_up(layer_id):
    """Sets the layer to move up."""
    st.session_state['layer_to_move_up'] = layer_id


def move_layer_down(layer_id):
    """Sets the layer to move down."""
    st.session_state['layer_to_move_down'] = layer_id


def show_weaklayer_input():
    """Displays the weak-layer thickness input."""
    # Create columns for label and input
    col_label, col_input = st.columns(
        [0.27, 0.73], vertical_alignment='center'
    )

    with col_label:
        st.write('Weak-layer thickness (mm)')

    with col_input:
        # Weak-layer thickness input
        st.number_input(
            label="Weak-layer thickness (mm)",
            label_visibility='collapsed',
            min_value=1,
            max_value=100,
            value=st.session_state.get('weaklayer_thickness', 30),
            step=5,
            key='weaklayer_thickness',
        )


def display_plot():
    """Displays the snow stratification plot."""
    # Get weak-layer thickness
    weak_layer_thickness = st.session_state['weaklayer_thickness']

    # Prepare data for plotting
    layers_data = [
        (layer['density'], layer['thickness'], layer['hardness'])
        for layer in st.session_state.layers
    ]
    grains = [layer['grain'] for layer in st.session_state.layers]

    # Generate the plot
    fig = plot.snow_profile(weak_layer_thickness, layers_data, grains)

    # Display the plot
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

# TODO:
# - Allow user input in inches
# - Provide option to enter layers top-to-bottom or bottom-to-top
# - Add inputs for inclination, cutting direction, slab faces, column length, and cut length
# - Integrate WEAC to compute ERR
