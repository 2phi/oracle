import pandas as pd
from pathlib import Path


class SLFDataParser:

    def __init__(
        self,
        Ewl: float = 0.2,
        nu: float = 0.25,
        gamma: float = 4.4,
        data_path: str = 'data/slf/',
        file_name: str = 'psts.txt',
    ) -> None:
        self.Ewl = Ewl
        self.nu = nu
        self.gamma = gamma
        self.data_path = Path(data_path)
        self.file_path = self.data_path / file_name
        self.dtypes = {
            'id': int,
            'rc[cm]': float,
            'slope[degree]': float,
            'H[cm]': float,
            'rho[kg/m3]': float,
            'L[cm]': float,
            'dy[cm]': float,
            'wl_thick[cm]': float,
            'propagation': bool,
            'slab_fracture': bool,
            'slab_fracture_distance[cm]': float,
        }

    def parse(self) -> None:
        self._read_txt()
        self._add_layers_column()
        self._drop_nan()
        self._convert_to_mm()
        self._rename_columns()

    def get_dataframe(self) -> pd.DataFrame:
        return self._df

    def _read_txt(self) -> None:
        self._df = pd.read_csv(
            self.file_path,
            sep=r'\s+',
            names=self.dtypes.keys(),
            dtype=self.dtypes,
            comment='#',
            header=0,
        )

    def _read_2d_array_from_file(self, file_path: Path) -> list[list]:
        with open(file_path, 'r') as f:
            next(f)  # Skip the first line (header)
            data = []  # Read and parse the rest of the lines into a 2D list
            for line in f:
                # Split the line into values
                row = line.strip().split()
                # First column is 1 for slab, 2 for weak, and 3 for base
                if int(float(row[0])) == 1:
                    # density[kg/m3], thickness[cm]
                    data.append([float(row[2]), float(row[1])])
        return data

    def _add_layers_column(self):
        """Read 'layers' from files and add to the DataFrame."""
        layers_data = []

        # Iterate over each row in the DataFrame
        for idx, row in self._df.iterrows():
            # Construct the file name based on the row index (e.g., '001.txt', '002.txt')
            file_name = f"{row['id']:03d}.txt"
            file_path = self.data_path / file_name

            if file_path.exists():
                # Read the 2D array from the file (skipping the first line)
                layers = self._read_2d_array_from_file(file_path)
            else:
                # If the file does not exist, append None
                layers = None

            layers_data.append(layers)

        # Add the new column to the DataFrame
        self._df['layers'] = layers_data

    def _drop_nan(self) -> None:

        def has_nan(layers: list[list[float]]) -> bool:
            return any(-999 in layer for layer in layers)

        self._df = self._df[~self._df['layers'].apply(has_nan)]

    def _convert_to_mm(self) -> None:
        def layers_to_mm(layers_cm: list[list[float]]) -> list[list[float]]:
            return [[rho, 10 * hi] for rho, hi in layers_cm]

        self._df['layers'] = self._df['layers'].apply(layers_to_mm)
        self._df['rc[mm]'] = self._df['rc[cm]'] * 10
        self._df['dy[mm]'] = self._df['dy[cm]'] * 10
        self._df['L[mm]'] = self._df['L[cm]'] * 10

    def _rename_columns(self) -> None:
        self._df.rename(
            columns={
                'slope[degree]': 'incline',
                'rc[mm]': 'lengthOfCut',
                'L[mm]': 'lengthOfColumn',
                'dy[mm]': 't',
            },
            inplace=True,
        )