import pandas as pd
import weac


class PropagationSawTestEngine:
    """
    Class to calculate weak-layer fracture toughness from propagation saw tests.

    Attributes
    ----------
    Ewl : float
        Young's modulus of the weak layer.
    twl : float
        Thickness of the weak layer.
    nu : float
        Poisson's ratio of the weak layer.
    gamma : float
        Exponent of Young-modulus parametrization.

    Methods
    -------
    __init__(Ewl: float = 0.2, twl: float = 20, nu: float = 0.25, gamma: float = 4.4) -> None
        Initializes the fracture toughness calculator with default weak-layer properties.
    calc_fracture_toughness(df: pd.DataFrame) -> pd.DataFrame
        Calculates weak-layer fracture toughness on a dataframe.
    """

    def __init__(
        self,
        Ewl: float = 0.2,
        twl: float = 20,
        nu: float = 0.25,
        gamma: float = 4.4,
    ) -> None:
        """
        Initialize the fracture toughness calculator with default weak-layer properties.

        Parameters
        ----------
        Ewl : float, optional
            Young's modulus of the weak layer. Default is 0.2 MPa.
        twl : float, optional
            Thickness of the weak layer. Default is 20 mm.
        nu : float, optional
            Poisson's ratio of the weak layer. Default is 0.25.
        gamma : float, optional
            Exponent of Young-modulus parametrization. Default is 4.4.
        """
        self.Ewl = Ewl
        self.twl = twl
        self.nu = nu
        self.gamma = gamma

    def _calc_gdif(
        self,
        pst_row: pd.Series,
        layers: list[list] | None = None,
        phi: float | None = None,
        a: float | None = None,
        L: float | None = None,
        t: float | None = None,
        E: float | None = None,
        nu: float | None = None,
    ) -> pd.Series:
        """
        Calculate weak-layer fracture toughness for a dataframe row.

        Parameters
        ----------
        pst_row : pd.Series, optional
            A row in the dataframe containing the following columns:
            - layers: ndarray
            - incline: float
            - lengthOfCut: float
            - lengthOfColumn: int
        layers : ndarray, optional
            Slab layering as list of densities (kg/m^3) and thicknesses (mm).
        phi : float, optional
            Slope angle (degrees).
        a : float, optional
            Cut length (mm).
        L : int, optional
            PST length (mm). Default is 1000 mm.
        t : float, optional
            Weak-layer thickness (mm). Default is `self.twl`.
        E : float, optional
            Weak-layer Young's modulus (MPa). Default is `self.Ewl`.
        nu : float, optional
            Weak-layer Poisson's ratio. Default is `self.nu`.

        Returns
        -------
        pd.Series or tuple
            A Series with 'Gc', 'GIc', and 'GIIc' values if `pst_row` is
            provided. Otherwise, a tuple of values.
        """
        # Extract values from the dataframe row
        layers = pst_row['layers']
        phi = pst_row['incline']
        a = pst_row['lengthOfCut']
        L = pst_row['lengthOfColumn']
        
        # Return NaN if layer info is missing
        if len(layers) == 0:
            return pd.NA

        # Use class defaults if specific values are not provided
        t = t if t is not None else self.twl
        E = E if E is not None else self.Ewl
        nu = nu if nu is not None else self.nu

        # Initialize the system
        model = weac.Layered(system='-pst')
        model.set_foundation_properties(t=t, E=E, nu=nu, update=True)
        model.set_beam_properties(layers=layers, C1=self.gamma, update=True)

        # Compute segmentation with crack as an unsupported segment
        segments = model.calc_segments(L=L, a=a)['crack']
        C = model.assemble_and_solve(phi=phi, **segments)

        # Calculate energy release rates and bending stiffness
        G, Gi, Gii = model.gdif(C=C, phi=phi, **segments, unit='J/m^2')
        D11 = 1e-8 * model.D11

        # Return the results in a pd.Series format for dataframe processing
        return pd.Series({'Gc': G, 'GIc': Gi, 'GIIc': Gii, 'D11': D11})

    def calc_fracture_toughness(
        self, df: pd.DataFrame, use_t: bool = False, use_E: bool = False
    ) -> pd.DataFrame:
        """
        Apply the fracture toughness calculation to an entire dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing slab layering, weak-layer properties, and cut lengths.

        Returns
        -------
        pd.DataFrame
            Updated dataframe with 'Gc', 'GIc', 'GIIc', and 'D11' columns.
        """
        df.loc[:, ['Gc', 'GIc', 'GIIc', 'D11']] = df.apply(
            lambda row: self._calc_gdif(
                pst_row=row,
                t=row['t'] if use_t else None,  # Pass 't' if use_t is True
                E=row['E'] if use_E else None,  # Pass 'E' if use_E is True
            ),
            axis=1,
        )
        return df
