# Third-party imports
import weac
import random
import base64
import json
import numpy as np
import streamlit as st
import scipy.stats as stats
from pathlib import Path
from st_screen_stats import ScreenData
from streamlit_theme import st_theme

# Local imports
from oracle.config import DENSITY_PARAMETERS, HAND_HARDNESS
from oracle import plot


def main():
    """Main function to run the Streamlit app."""
    # Set page configuration
    st.set_page_config(page_title='ORACLE', layout='centered', page_icon='🔮')

    # Display the ORACLE logo and title
    display_header()

    # Initialize session state variables
    initialize_session_state()

    # Monitor screen width for responsive design
    watch_screen_width()

    st.info(
        r"""
        **ORACLE** evaluates weak-layer conditions based on propagation saw tests (PSTs).
        
        1. Click **Add Layer** to add slab layers from the bottom to the top, starting above the weak layer.
        2. Enter your PST data and view the results in real time.
        3. Adjust parameters as needed and observe how they impact the results instantly.
        """
    )

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
    display_snow_profile()

    # Propagation saw test
    st.markdown('#### Propagation saw test')

    # Handle PST inputs
    handle_pst_inputs()

    # Show PST instructions
    show_pst_tooltip()

    # Run weac to compute ERR and weak-layer instability
    run_weac()

    # Show results
    st.markdown('#### Weak-layer condition')

    # ORACLE!
    display_result()


def display_header():
    """Displays the ORACLE logo and title."""

    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        return base64.b64encode(img_bytes).decode()

    def img_to_html(img_path, width=200, align='center'):
        img_html = f"""
            <div style="text-align: {align};">
                <img src='data:image/png;base64,{img_to_bytes(img_path)}' class='img-fluid' width={width}>
            </div>
            """
        return img_html

    # Display the ORACLE logo
    st.html(img_to_html('img/steampunk-v1.png'))

    # Display the ORACLE title
    st.html(
        """
        <div style="text-align: center;">
            <h1>ORACLE</h1>
            <p><b>Observation, Research, and Analysis of<br>Collapse and Loading Experiments</b></p>
        </div>
        """
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
    state.setdefault('weakness', None)
    st_theme(key='theme')


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
        layer_id, grain_options, hardness_options, max=10
    ):
        """Generates random properties for a new layer."""
        # Adjust the bias based on the layer ID
        id = min(layer_id, max)
        n_grains = len(grain_options)
        n_hardness = len(hardness_options)
        bias = 5  # Bias factor for weighted choices

        # Create weights for grain options
        grain_weights = np.linspace(1, bias * (max - id) / max, n_grains)
        grain_weights /= np.sum(grain_weights)

        # Create weights for hardness options
        hardness_weights = np.linspace(1, bias * (max - id) / max, n_hardness)
        hardness_weights /= np.sum(hardness_weights)

        # Generate random thickness (mean=100, std=50, rounded to nearest 20, clipped to 20-200)
        thickness = round(np.random.normal(100, 50) / 20) * 20
        thickness = np.clip(thickness, 20, 200)

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
    n_layers = len(layers)
    screen_width = st.session_state["screen_stats"]['innerWidth']

    # Determine label visibility based on screen width
    label_visibility = 'visible' if screen_width < 640 else 'collapsed'

    if n_layers > 0:

        # Define column widths
        col_widths = [1.6, 4, 3, 3]
        if n_layers > 1 and screen_width > 640:
            # Add columns for movement buttons
            col_widths.extend([1.2, 1.2, 1.2])
        else:
            # Only remove button
            col_widths.append(1.4)

        # Display table headers based on screen width
        if screen_width > 640:

            # Create header columns
            cols = st.columns(col_widths)

            with cols[1]:
                st.markdown('Layer thickness (mm)')
            with cols[2]:
                st.markdown('Grain form')
            with cols[3]:
                st.markdown('Hand hardness')

        # Display each layer
        for i, layer in enumerate(layers):
            layer_id = layer['id']

            # Create columns for the layer
            cols = st.columns(col_widths, vertical_alignment='center')

            # Unpack columns
            col_label, col_thickness, col_grain, col_hardness = cols[:4]
            if n_layers > 1 and screen_width > 640:
                col_move_down = cols[4]
                col_move_up = cols[5]
            else:
                col_move_down = None
                col_move_up = None
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

            if n_layers > 1 and col_move_down and col_move_up:
                with col_move_down:
                    # Disable for the bottom layer
                    disabled = i == n_layers - 1
                    # Move layer down button
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
                    # Disable for the top layer
                    disabled = i == 0
                    # Move layer up button
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
                    st.markdown(
                        """<p style="font-size: 14px; margin-bottom: 0;">Remove</p>""",
                        unsafe_allow_html=True,
                    )
                # Remove layer button
                st.button(
                    "🗑️",
                    key=f"remove_{layer_id}",
                    use_container_width=True,
                    on_click=remove_layer,
                    args=(layer_id,),
                )

    else:
        st.warning(
            """
        Click **Add layer** to add layers above the weak layer and
        to display the snow profile.
        """
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

    if st.session_state.layers:

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


def show_snowprofile_tooltip():
    """Displays additional info on snow profile data."""
    s = st.expander('💡 What is shown here?')
    with s:
        s.markdown(
            """
            The snow profile plot displays the densities and thicknesses of
            the weak layer and the overlying slab layers. Layer densities are
            derived based on their primary grain form and hand hardness. The
            table on the right provides detailed information on each layer,
            including height (H), density (D), grain form (F), and hand
            hardness (R). You can adjust the layer thickness, grain type,
            and hand hardness at any time. Additionally, you can rearrange
            layers by moving them up or down, or delete them as needed.
            """
        )


def display_snow_profile():
    """Displays the snow stratification plot."""

    if st.session_state['layers']:

        # Generate the plot
        fig = plot.snow_profile(
            st.session_state['weaklayer_thickness'],
            st.session_state['layers'][::-1],
            json.loads(st.session_state.theme),
        )

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

        # Show snow profile instructions
        show_snowprofile_tooltip()


def show_pst_tooltip():
    """Displays instructions for entering PST data."""
    s = st.expander('💡 Why are these inputs needed?')
    with s:
        s.markdown(
            r"""
            **ORACLE** calculates the weak-layer condition based on the energy release
            rate at the critical cut length. To perform this calculation, the model
            requires the PST geometry (column length, slab face geometry) and loading
            parameters (cut length, cutting direction, slope angle). In the future,
            we will add distinctions between crack arrest and full propagation.
            """
        )


def run_weac(E=0.2, s=1.435, loc=-0.0036, scale=1.143):

    # Vertical or slope-normale slab faces
    if st.session_state['slab_faces'] == 'Vertical':
        system = 'vpst'
    elif st.session_state['slab_faces'] == 'Slope-normal':
        system = 'pst'

    # Upslope or downslope cut
    if st.session_state['cutting_direction'] == 'Upslope':
        system = '-' + system
    elif st.session_state['cutting_direction'] == 'Downslope':
        system = system + '-'

    # Layer data top to bottom
    layers = [
        (layer['density'], layer['thickness'])
        for layer in st.session_state.layers
    ]

    # Parameters
    t = st.session_state['weaklayer_thickness']
    L = 10 * st.session_state['column_length']
    a = 10 * st.session_state['cut_length']
    phi = st.session_state['inclination']

    # Initialize PST object and set weak-layer properties
    pst = weac.Layered(system=system, layers=layers)
    pst.set_foundation_properties(t=t, E=E, update=True)

    # Calculate segmentation and solve for free constants
    segments = pst.calc_segments(phi=phi, L=L, a=a)['crack']
    C = pst.assemble_and_solve(phi=phi, **segments)

    # Calculate ERR and weak-layer instability
    Gdif = pst.gdif(C, phi, **segments, unit='J/m^2')[0]
    st.session_state['weakness'] = 1 - stats.lognorm.cdf(Gdif, s, loc, scale)


def handle_pst_inputs():

    cols = st.columns([3, 2], gap='large')

    st.slider(
        "Slope angle ( ° )",
        min_value=0,
        max_value=60,
        value=25,
        step=1,
        key='inclination',
    )
    with cols[0]:
        st.number_input(
            "Cut length (cm)",
            min_value=1,
            max_value=100,
            value=30,
            step=5,
            key='cut_length',
        )
    with cols[0]:
        st.number_input(
            "Column length (cm)",
            min_value=1,
            max_value=10000,
            value=100,
            step=50,
            key='column_length',
        )
    with cols[1]:
        st.radio(
            "Cutting direction",
            options=['Upslope', 'Downslope'],
            index=0,
            key='cutting_direction',
            horizontal=True,
        )
    with cols[1]:
        st.radio(
            "Slab faces",
            options=['Slope-normal', 'Vertical'],
            index=0,
            key='slab_faces',
            horizontal=True,
        )


def display_result():
    """Show final result."""

    if st.session_state['layers']:

        # Generate the plot
        fig = plot.weaklayer_instability(
            st.session_state['weakness'],
            json.loads(st.session_state.theme),
        )

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

        # Show result explanation
        show_result_tooltip()

    else:
        st.warning('Add snow-profile information to display a result.')


def show_result_tooltip():
    """Displays the explanation of the result."""
    s = st.expander('💡 How is this calculated?')
    with s:
        s.markdown(
            """
            ORACLE calculates the fracture toughness (fracture energy) of the
            weak layer based on the provided propagation saw test (PST) result.
            Fracture toughness refers to the critical energy release rate at which
            a crack, introduced by the saw, becomes unstable and propagates through
            the weak layer. The model takes into account every slab layer and
            considers boundary effects. We have analyzed over 2,300 PSTs to establish
            the probability distribution of expected weak-layer fracture toughness
            values. By comparing the fracture toughness of the entered weak layer to
            this distribution, we can estimate how weak the layer is relative to other
            weak layers in our database.
            """
        )


if __name__ == "__main__":
    main()

# TODO:
# - Link in Github repo
# - Allow user input in inches
# - Weak-layer graintype input (with E from Jakob's data)
# - Provide option to enter layers top-to-bottom or bottom-to-top
# - Color grain types in profile
# - Record input data for future reference
# - Add version number an manage relaeses
