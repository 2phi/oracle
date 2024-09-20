# Standard library imports
import os
import shutil
import calendar
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

# Third-party imports
import requests
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from tqdm import tqdm
from time import sleep
from dotenv import load_dotenv

# Application imports
from oracle.config import GRAIN_TYPE, HAND_HARDNESS, DENSITY_PARAMETERS

# Load environment variables from .env
load_dotenv(override=True)


class SnowPilotQueryEngine:
    """
    A class to query the SnowPilot API.

    This class provides methods to query the SnowPilot API, download and process
    snow pit observations, and convert the data into a structured format for analysis.

    Parameters
    ----------
    data_path : str or Path, optional
        The path to the data directory. Default is 'data'.
    xml_path : str or Path, optional
        The path to the XML directory. Default is 'data/snowpilot/xml'.
    caaml_path : str or Path, optional
        The path to the CAAML directory. Default is 'data/snowpilot/caaml'.

    Attributes
    ----------
    data_path : Path
        The path to the data directory.
    xml_path : Path
        The path to the XML directory.
    caaml_path : Path
        The path to the CAAML directory.
    site_url : str
        The URL of the SnowPilot website.
    log_in_url : str
        The URL for the user login to the SnowPilot website.
    caaml_query_url : str
        The URL for querying CAAML data.
    xml_query_url : str
        The URL for querying XML data.
    data_url : str
        The URL for downloading data.
    credentials : dict
        The login credentials for the SnowPilot website.

    Methods
    -------
    run()
        Query the SnowPilot API and download and process snow pit observations.
    query_xml(total_obs=33568, batch_size=500)
        Query the SnowPilot API and download snow pit observations.
    query_caaml(years_back=10, per='week', pause=120)
        Query snowpilot.org for CAAML data.
    unzip_caaml()
        Find all .tar.gz files in `caaml_path` and extract them.
    merge_xml(output_file='all.xml')
        Merge all XML files into one.
    filter_psts(input_file='all.xml', output_file='psts.xml')
        Filter Pit_Observation elements containing PST shear test results.
    count_psts(xml_file='psts.xml')
        Count the number of Pit_Observation elements with PSTs in an XML file.
    """

    def __init__(
        self,
        data_path: str | Path = Path('data/snowpilot/'),
        xml_path: str | Path = Path('data/snowpilot/xml'),
        caaml_path: str | Path = Path('data/snowpilot/caaml'),
    ) -> None:
        # Directories
        self._data_path = Path(data_path)
        self._xml_path = Path(xml_path)
        self._caaml_path = Path(caaml_path)
        # URLs
        self._site_url = "https://snowpilot.org"
        self._log_in_url = self._site_url + '/user/login'
        self._caaml_query_url = self._site_url + '/avscience-query-caaml.xml?'
        self._xml_query_url = self._site_url + '/snowpilot-query-feed.xml?'
        self._data_url = 'https://snowpilot.org/sites/default/files/tmp/'
        # Login credentials
        self._credentials = {
            'name': os.environ.get('SNOWPILOT_USER'),
            'pass': os.environ.get('SNOWPILOT_PASSWORD'),
            'form_id': 'user_login',
            'op': 'Log in',
        }

    def run(self) -> None:
        """
        Run the pipeline to query the SnowPilot API and download and process snow pit observations.
        """
        print('Querying the complete SnowPilot database...')
        self.query_xml()

        print('Merging downloaded XML files into one...')
        self.merge_xml()

        print('Filtering for PSTs...')
        self.filter_psts()
        self.count_psts()

        print('Converting XML to dataframe...')
        self.xml_to_pkl()

        print('SnowPilot database query and processing complete.')

    def query_xml(self, total_obs: int = 33568, batch_size: int = 500) -> None:
        """
        Query the SnowPilot API and download snow pit observations.

        This function queries the SnowPilot API to retrieve snow pit observations in XML format.
        The observations are downloaded in batches and saved to the specified output directory.

        Parameters
        ----------
        total_obs : int, optional
            Total number of pit observations in the SnowPilot databse. Default is 33568.
        batch_size : int, optional
            The number of observations to retrieve in each batch. Default (and maximum) is 500.
        """

        # Total number of pages
        total_pages = total_obs // batch_size + 1
        page_numbers = range(total_pages)

        # Loop through each page and download the XML
        with tqdm(page_numbers, desc='Querying SnowPilot') as pbar:
            for i in page_numbers:
                # Download XML data and update status
                _, msg = self._download_xml(i, batch_size)
                pbar.set_postfix({'Status': msg})
                pbar.update(1)

        print('Download completed')

    def query_caaml(
        self, years_back: int = 10, per: str = 'week', pause: int = 120
    ) -> None:
        """
        Query snowpilot.org for CAAML data.

        Parameters
        ----------
        years_back : int, optional
            Number of years back to query. Default is 10.
        per : str, optional
            Query per week or month. Default is 'week'.
        pause : int, optional
            Sleep time between queries in seconds. Default is 120.
        """
        # Get date ranges for the past 10 years
        if per == 'month':
            date_ranges = self._get_monthly_date_ranges(years_back=years_back)
        elif per == 'week':
            date_ranges = self._get_weekly_date_ranges(years_back=years_back)
        else:
            raise ValueError(
                "Invalid parameter for 'per'. Choose 'month' or 'week'."
            )

        # Query CAAML data for each date range
        with tqdm(total=len(date_ranges), desc='Querying SnowPilot') as pbar:
            for start, end in date_ranges:
                # Update the progress bar with the current date range and status
                postfix = {
                    'From': start,
                    'To': end,
                    'Status': 'Query submitted...',
                }
                pbar.set_postfix(postfix)

                # Download CAAML data and update status
                _, postfix['Status'] = self._download_caaml(start, end)
                pbar.set_postfix(postfix)
                pbar.update(1)
                sleep(10)

                # Update the progress bar and pause for two minutes
                postfix['Status'] = 'Waiting for next query...'
                pbar.set_postfix(postfix)
                sleep(max(0, pause - 10))

        print('Download complete')

    def _download_xml(self, page: int, batch_size: int = 500) -> None:
        """
        Download XML data from SnowPilot API.

        Parameters
        ----------
        page : int
            Page number to download.
        batch_size : int, optional
            Number of entries per page. Default and maximum is 500.

        Returns
        -------
        _type_
            _description_
        """
        assert batch_size <= 500, 'Batch cannot be larger than 500.'

        # Construct the URL with page number and send the request
        q = f'caaml_feed=CAAML%20XML%20Output&per_page={batch_size}&page='
        url = f'{self._xml_query_url}{q}{page}'
        r = requests.get(url)

        # Check if the request was successful and save the XML file
        if r.status_code == 200:
            file_path = self._xml_path / f'snowpilot_page_{page}.xml'
            with open(file_path, 'wb') as file:
                file.write(r.content)
            return 1, 'Download successful.'
        else:
            return 2, f'Download failed with {r.status_code}'

    def _download_caaml(self, start_date: str, end_date: str) -> None:
        """
        Query snowpilot.org for a given date range.

        Parameters
        ----------
        start_date : str
            Format of YYYY-MM-DD.
        end_date : str
            Format of YYYY-MM-DD.

        Returns
        -------
        str
            Status message.
        """
        # Query
        q = f'OBS_DATE_MIN={start_date}&OBS_DATE_MAX={end_date}&per_page=1000'

        # Create session, authenticate, and query CAAML feed
        with requests.Session() as s:
            # Authenticate
            a = s.post(self._log_in_url, data=self._credentials)
            # Check if authentication was successful
            if a.status_code != 200:
                return 3, 'Authentication error.'
            # Query CAAML feed
            r = s.post(self._caaml_query_url + q)
            # Get content disposition
            disposition = r.headers.get('Content-Disposition', None)
            # Download the file
            if r.status_code == 200:
                if len(disposition) < 40:
                    return 0, 'No data found.'
                else:
                    # Extract gzip file name
                    f_name = disposition[22:-1].replace('_caaml', '')
                    f_path = self._data_url + f_name
                    # Get gzip file
                    data = s.get(f_path)
                    # Save gzip file to disk
                    save_f_name = f'{start_date}_to_{end_date}.tar.gz'
                    save_f_path = os.path.join(self._caaml_path, save_f_name)
                    with open(save_f_path, 'wb') as f:
                        f.write(data.content)
                    return 1, 'Download successful.'
            else:
                return 2, f'Download failed with {r.status_code}'

    def _get_monthly_date_ranges(self, years_back: int = 10) -> None:
        """
        Generate monthly date ranges for the past `years_back` years.

        Parameters:
        ----------
        years_back : int
            Number of years back to generate the date ranges.

        Returns:
        -------
        date_ranges : list of tuples
            List of (start_date, end_date) tuples for each month.
        """
        today = datetime.now()
        date_ranges = []

        for year in range(today.year - years_back, today.year + 1):
            for month in range(1, 13):
                start_date = datetime(year, month, 1)
                _, last_day = calendar.monthrange(year, month)
                end_date = datetime(year, month, last_day)

                if start_date > today:
                    break

                # Append formatted date strings
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                date_ranges.append((start_str, end_str))

        return date_ranges[::-1]

    def _get_weekly_date_ranges(self, years_back: int = 10) -> None:
        """
        Generate weekly date ranges for the past `years_back` years.

        Parameters:
        ----------
        years_back : int
            Number of years back to generate the date ranges.

        Returns:
        -------
        date_ranges : list of tuples
            List of (start_date, end_date) tuples for each week.
        """
        today = datetime.now()
        date_ranges = []

        # Calculate the start date, years_back years from today
        start_date = today.replace(year=today.year - years_back)

        while start_date <= today:
            # Calculate end date for the week (6 days after start date)
            end_date = start_date + timedelta(days=6)

            # If the end date is beyond today, set it to today
            if end_date > today:
                end_date = today

            # Append formatted date strings
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            date_ranges.append((start_str, end_str))

            # Move to the next week
            start_date = start_date + timedelta(days=7)

        return date_ranges[::-1]

    def unzip_caaml(self) -> None:
        """Find all .tar.gz files in `self._caaml_path` and extract them."""
        # Find all .tar.gz files in CAAML_PATH
        tar_files = glob(os.path.join(self._caaml_path, '*.tar.gz'))

        # Loop over all found .tar.gz files and extract them into a temporary directory
        for filename in tar_files:
            # Create a temporary directory for extraction
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Extract the tar.gz file into the temporary directory
                shutil.unpack_archive(filename, tmp_dir)

                # Find the extracted folder (usually there's only one top-level folder)
                extracted_dir = os.path.join(tmp_dir, os.listdir(tmp_dir)[0])

                # Move all extracted caaml.xml files to caaml_path
                for f in glob(
                    os.path.join(extracted_dir, '**/*caaml.xml'),
                    recursive=True,
                ):
                    dest = os.path.join(
                        self._caaml_path, os.path.basename(f)
                    )  # Use os.path.basename for the file name
                    shutil.copyfile(f, dest)

        print(
            'Extraction complete. All .tar.gz files in',
            self._caaml_path.resolve(),
            'have been processed.',
        )

    def merge_xml(self, output_file: str = 'all.xml') -> None:
        """
        Merge all XML files into a one.

        Parameters
        ----------
        output_file : str, optional
            Name of the merged output XML file.
        """
        # Create the root element for the combined XML file
        root = ET.Element('Pit_Data')

        # Iterate over all XML files in the specified directory
        for filename in os.listdir(self._xml_path):
            if filename.endswith(".xml"):
                file_path = os.path.join(self._xml_path, filename)
                try:
                    # Parse each XML file
                    tree = ET.parse(file_path)
                    file_root = tree.getroot()

                    # Append each Pit_Observation element to the root of the combined file
                    for pit_obs in file_root.findall('Pit_Observation'):
                        root.append(pit_obs)

                except ET.ParseError as e:
                    print(f'Error parsing file {file_path}: {e}')
                    continue  # Skip this file and move on to the next

        # Create the final tree and write it to the output file
        tree = ET.ElementTree(root)
        tree.write(
            self._data_path / output_file,
            encoding='utf-8',
            xml_declaration=True,
        )

        print('Merged XML saved to', self._data_path / output_file)

    def filter_psts(
        self, input_file: str = 'all.xml', output_file: str = 'psts.xml'
    ) -> None:
        """
        Filter Pit_Observation elements containing PST shear test results.

        This function parses an input XML file containing snow pit observations, filters out
        the Pit_Observation elements that contain a Shear_Test_Result with code="PST", and writes
        the filtered observations to an output XML file.

        Parameters
        ----------
        input_file : str, optional
            Name of the input XML file to parse. Default is 'all.xml'.
        output_file : str, optional
            Name of the output XML file to save the filtered data. Default is 'psts.xml'.
        """

        # Parse the input XML file
        tree = ET.parse(self._data_path / input_file)
        root = tree.getroot()

        # Create a new root for the filtered XML
        filtered_root = ET.Element('Pit_Data')

        # Iterate through all Pit_Observation elements
        for pit_obs in root.findall('Pit_Observation'):
            # Check if there is a Shear_Test_Result with code='PST'
            for shear_test in pit_obs.findall('Shear_Test_Result'):
                if shear_test.get('code') == 'PST':
                    # If found, append this Pit_Observation to the filtered root
                    filtered_root.append(pit_obs)
                    break  # No need to check further Shear_Test_Result elements

        # Write the filtered XML to the output file
        filtered_tree = ET.ElementTree(filtered_root)
        filtered_tree.write(
            self._data_path / output_file,
            encoding='utf-8',
            xml_declaration=True,
        )

        print('Filtered PSTs saved to', self._data_path / output_file)

    def count_psts(self, xml_file: str = 'psts.xml') -> None:
        """
        Count the number of Pit_Observation elements with PSTs in an XML file.

        Parameters
        ----------
        xml_file : str, optional
            Name of the XML file to parse. Default is 'psts.xml'.
        """

        # Parse the XML file
        try:
            tree = ET.parse(self._data_path / xml_file)
            root = tree.getroot()

            # Count the number of Pit_Observation elements
            pit_observation_count = len(root.findall('Pit_Observation'))
            return print(f'Found {pit_observation_count} pits with PSTs')
        except ET.ParseError as e:
            print(f'Error parsing XML file: {e}')
        except FileNotFoundError as e:
            print(f'File not found: {e}')

    def _process_shear_tests(self, shear_tests, keep_test_data):
        """
        Process shear tests to filter and extract relevant PST data.

        Parameters
        ----------
        shear_tests : list of dict
            A list of dictionaries, each representing a shear test with various attributes.
        keep_test_data : list of str
            A list of keys to keep from each shear test dictionary.

        Returns
        -------
        dict
            A dictionary containing the processed PST data. If no PST tests are found, an empty dictionary is returned.
        """
        # Filter for tests where 'code' == 'PST'
        psts = [t for t in shear_tests if t.attrib.get('code') == 'PST']

        # If no PST tests found, return an empty dict
        if not psts:
            return {}

        # Convert sdepth, lengthOfColumn, and lengthOfCut to float for comparison and calculation
        for pst in psts:
            pst.set('sdepth', float(pst.get('sdepth')))
            pst.set('lengthOfColumn', float(pst.get('lengthOfColumn')))
            pst.set('lengthOfCut', float(pst.get('lengthOfCut')))

        # Step 1: Keep only the smallest value of sdepth (can have multiple)
        min_sdepth = min(test.get('sdepth') for test in psts)
        psts = [t for t in psts if t.get('sdepth') == min_sdepth]

        # Step 2: Keep only the largest value of lengthOfColumn (can have multiple)
        max_lengthOfColumn = max(t.get('lengthOfColumn') for t in psts)
        psts = [
            pst
            for pst in psts
            if pst.get('lengthOfColumn') == max_lengthOfColumn
        ]

        # Step 3: If there are both 'END' and non-'END' dataCode entries, keep only those that are 'END'
        is_end = any(t.get('dataCode') == 'END' for t in psts)
        is_not_end = any(t.get('dataCode') != 'END' for t in psts)

        if is_end and is_not_end:
            psts = [t for t in psts if t.get('dataCode') == 'END']

        # Step 4: Average lengthOfCut
        avg_lengthOfCut = np.mean([t.get('lengthOfCut') for t in psts])

        # Prepare the final test data to update
        # Take one test as representative (since we filtered, they should be similar) and update the avg_lengthOfCut
        final = psts[0]
        final.set('lengthOfCut', avg_lengthOfCut)

        # Only keep attributes that are in keep_test_data
        test_data = {k: v for k, v in final.items() if k in keep_test_data}

        return test_data

    def xml_to_pkl(
        self, xml_file: str = 'psts.xml', pkl_file: str = 'psts.pkl'
    ) -> None:
        """
        Convert snowpilot XML files to a pickled dataframe.

        This function parses a snowpilot XML file containing snow pit observations,
        extracts relevant data, and saves it into a pickle file for later use.

        Parameters
        ----------
        xml_file : str, optional
            Name of the XML file to parse. Default is 'psts.xml'.
        pkl_file : str, optional
            Name of the pickle file to save the parsed data. Default is 'psts.pkl'.
        """
        # Parse the XML file
        tree = ET.parse(self._data_path / xml_file)
        root = tree.getroot()

        # List to hold data for each row
        data = []

        # Iterate through all Pit_Observation elements
        for pit_obs in root.findall('Pit_Observation'):
            # Extract attributes and sub-elements to create a row
            keep_pit_data = [
                'depthUnits',
                'heightOfSnowpack',
                'measureFrom',
                'nid',
                'incline',
            ]
            pit_data = {
                k: v for k, v in pit_obs.attrib.items() if k in keep_pit_data
            }

            # Extract user information
            user_info = pit_obs.find('User')
            keep_user_info = ['depthUnits', 'measureFrom']
            user_data = {
                k: v
                for k, v in user_info.attrib.items()
                if k in keep_user_info
            }
            pit_data.update(user_data)

            # Extract layer information
            layers = pit_obs.findall('Layer')
            keep_layer_data = [
                'layerNumber',  # layer ID
                'startDepth',  # top/bottom depth of layer
                'endDepth',  # top/bottom depth of layer
                'grainType',  # primary grain type?
                'grainType1',  # secondary grain type?
                'hardness1',  # primary hand hardness?
                'hardness2',  # secondary hand hardness?
            ]
            layer_data = []
            for layer in layers:
                layer_data.append(
                    {
                        k: v
                        for k, v in layer.attrib.items()
                        if k in keep_layer_data
                    }
                )
            pit_data.update({'layers': layer_data})

            # Get PST results
            shear_tests = pit_obs.findall('Shear_Test_Result')
            keep_test_data = [
                'dataCode',  # PST result (End, Arr, SF, X)
                'depthUnits',  # units of layer depth
                'lengthOfColumn',  # PST column length
                'lengthOfCut',  # critical cut length
                'sdepth',  # depth of layer tested in stability test
            ]
            # Usually only one PST in each pit, but applying logic to handle multiples
            test_data = self._process_shear_tests(shear_tests, keep_test_data)
            pit_data.update(test_data)

            # Append the row data to the list
            data.append(pit_data)

        # Convert list of dictionaries to DataFrame and save as pickle
        df = pd.DataFrame(data)

        # Make sure the data types are correct
        df['nid'] = pd.to_numeric(df['nid'], errors='coerce').astype(int)
        df['incline'] = pd.to_numeric(df['incline'], errors='coerce')
        df.dropna(subset=['incline'], inplace=True)

        # Save the DataFrame to a pickle file
        df.to_pickle(self._data_path / pkl_file)
        print('Pickle saved to', self._data_path / pkl_file)


class SnowPilotParser:
    """
    Parse and process SnowPilot data.

    Attributes
    ----------
    _data_path : Path
        The path to the data directory.
    _df_raw : pd.DataFrame
        The raw DataFrame loaded from the pickle file.
    _df : pd.DataFrame
        The processed DataFrame.

    Methods
    -------
    __init__(data_path: str | Path = Path('data')) -> None
        Initializes the Parser with the specified data path.
    parse(pkl_file: str = 'psts.pkl') -> None
        Parses the SnowPilot data from the specified pickle file.
    get_dataframe(which: str = 'parsed') -> pd.DataFrame
        Returns the DataFrame containing the SnowPilot data.
    """

    def __init__(self, data_path: str | Path = Path('data/snowpilot')) -> None:
        """
        Initialize the SnowPilot Parser class.

        Parameters
        ----------
        data_path : str | Path, optional
            The path to the data directory. Default is 'data'.
        """
        # Load configuration for grain types, hand hardness, and density parametrization
        self._load_config()
        # Set data path
        self._data_path = Path(data_path)

    def parse(self, pkl_file: str = 'psts.pkl') -> None:
        """
        Parse the snow pilot data from the pickle file.

        Parameters
        ----------
        pkl_file : str, optional
            The name of the pickle file to load. Default is 'psts.pkl'.
        """
        # Load SnowPilot data into a DataFrame
        self._df_raw = self._load_pkl(pkl_file)
        self._df = self._df_raw.copy(deep=True)
        # Parse layers from dictionary to list format
        self.layer_dict_to_list()
        self.calculate_layer_density()
        self.units_to_mm()
        self.calculate_wl_depth()
        self.remove_layers_outside_slab()

    def get_dataframe(self, which: str = 'parsed') -> pd.DataFrame:
        """
        Get the DataFrame containing the snow pilot data.

        Parameters
        ----------
        which : str, optional
            Specify whether to return the `raw` or `parsed` DataFrame. Default is 'parsed'.

        Returns
        -------
        pd.DataFrame
            The requested DataFrame.
        """
        if which == 'raw':
            return self._df_raw
        elif which == 'parsed':
            return self._df
        else:
            raise ValueError(
                "Invalid option for `how`. Choose 'raw' or 'parsed'."
            )

    def _load_config(self):
        """
        Load configuration for grain types, hand hardness, and density parameters.
        """
        self.grain_type = GRAIN_TYPE
        self.hand_hardness = HAND_HARDNESS
        self.density_parameters = DENSITY_PARAMETERS

    def _get_density_params(self, grain_type: str) -> tuple:
        """
        Get density parametriation parameters for a given grain type.

        Parameters
        ----------
        grain_type : str
            The grain type for which to retrieve density parameters.

        Returns
        -------
        tuple
            A tuple containing the density parameters (a, b).
        """
        return self.density_parameters.get(grain_type, (0, 0))

    def _get_grain_type(self, grain_type: str) -> str:
        """
        Get the grain type description for a given grain type code.

        Parameters
        ----------
        grain_type : str
            The grain type code to look up.

        Returns
        -------
        str
            The grain type description or '!skip' if not found.
        """
        return self.grain_type.get(grain_type, '!skip')

    def _get_hand_hardness(self, hardness_code: str) -> str | float:
        """
        Get the hand hardness value for a given hardness code.

        Parameters
        ----------
        hardness_code : str
            The hardness code to look up.

        Returns
        -------
        float
            The hand hardness value or '!skip' if not found.
        """
        return self.hand_hardness.get(hardness_code, '!skip')

    def _load_pkl(self, pkl_file: str = 'psts.pkl'):
        """
        Load a pickle file into a pandas DataFrame.

        Parameters
        ----------
        pkl_file : str, optional
            Name of the pickle file to load. Default is 'psts.pkl'.

        Returns
        -------
        pd.DataFrame
            The DataFrame loaded from the pickle file.
        """
        return pd.read_pickle(self._data_path / pkl_file)

    def layer_dict_to_list(self) -> None:
        """Convert layer information from dictionary format to list format."""

        def parse_layers(list_of_dicts: list[dict]) -> list[list]:
            """
            Parse layer information in dictionary format to list format.

            Parameters
            ----------
            list_of_dicts : List[Dict]
                List of layers with information of each layer in dictionary format.

            Returns
            -------
            list_of_lists : List[List]
                List of layers with information of each layer in list format, ordered as:
                [layerNumber, startDepth, endDepth, grainType, hardness1]
            """
            list_of_lists = []
            for layer in list_of_dicts:
                # layerNumber, startDepth, endDepth, grainType, hardness1
                layer_list = [
                    int(layer['layerNumber']),
                    float(layer['startDepth']),
                    float(layer['endDepth']),
                    str(layer['grainType']),
                    str(layer['hardness1']),
                ]
                # Calc layer thickness
                list_of_lists.append(layer_list)

            return list_of_lists

        self._df['layers'] = self._df['layers'].apply(parse_layers)

    def calculate_layer_density(self) -> None:
        """Calculate layer densities from grain type and hand hardness."""

        def calc_layer_thickness(layer, measure_from):
            """Calculate the layer thickness from start and end depths."""
            if measure_from == 'top':
                return layer[2] - layer[1]
            elif measure_from == 'bottom':
                return layer[1] - layer[2]
            else:
                raise ValueError('Unknown measureFrom value')

        def convert_to_mm(thickness, depth_units):
            """Convert thickness to mm based on depth units."""
            if depth_units == 'cm':
                return thickness * 10
            elif depth_units == 'in':
                return thickness * 25.4
            elif depth_units == 'mm':
                return thickness
            else:
                raise ValueError('Unknown depthUnits value')

        # List to keep track of rows with incomplete data
        incomplete = []

        # Iterate through all rows in the DataFrame
        for index, row in self._df.iterrows():
            layers = row['layers']
            # if len(layers) < 2:
            #     incomplete.append(index)
            #     continue
            new_layers = []
            # layers are ordered top to bottom
            for layer in layers:
                # Calculate layer thickness
                thickness = calc_layer_thickness(layer, row['measureFrom'])
                thickness = convert_to_mm(thickness, row['depthUnits'])
                # Get grain type and hand hardness
                grainType = self._get_grain_type(layer[3])
                hardness = self._get_hand_hardness(layer[4])
                if '!skip' in {grainType, hardness}:
                    incomplete.append(index)
                else:
                    # Calculate density
                    a, b = self._get_density_params(grainType)
                    if grainType == 'RG':
                        density = a + b * (hardness**3.15)
                    else:
                        density = a + b * hardness

                new_layers.append([density, thickness])

            # Update layers with density
            self._df.at[index, 'layers'] = new_layers

        # Remove duplicate indices
        incomplete = np.unique(incomplete)
        self._df.drop(incomplete)

    def units_to_mm(
        self,
        columns_to_convert: list[str] = [
            'heightOfSnowpack',
            'sdepth',
            'lengthOfCut',
            'lengthOfColumn',
        ],
    ) -> None:
        """
        Convert specified columns from cm or in to mm.

        Parameters
        ----------
        columns_to_convert : list[str], optional
            List of column names to convert. Default is ['heightOfSnowpack',
            'sdepth', 'lengthOfCut', 'lengthOfColumn'].
        """

        def convert_to_mm(row, col_name):
            if row['depthUnits'] == 'cm':
                return 10 * float(row[col_name])
            elif row['depthUnits'] == 'in':
                return 25.4 * float(row[col_name])
            else:
                return None  # Handle cases where unit is not recognized

        # Remove rows with empty values in the columns to convert
        self._df = self._df[(self._df[columns_to_convert] != '').all(axis=1)]

        # Iterate over the columns to convert them based on depthUnits
        for col in columns_to_convert:
            self._df[col] = self._df.apply(
                lambda row: convert_to_mm(row, col), axis=1
            )

    def calculate_wl_depth(self) -> None:
        """Calculate water layer depth based on measureFrom and sdepth."""

        def wl_depth(row: pd.Series) -> float:
            if row['measureFrom'] == 'top':
                return row['sdepth']
            elif row['measureFrom'] == 'bottom':
                return row['heightOfSnowpack'] - row['sdepth']
            else:
                return None  # Handle cases where measureFrom is not recognized

        self._df['wl_depth'] = self._df.apply(wl_depth, axis=1)

    def remove_layers_outside_slab(self) -> None:
        """Remove layers that are not part of the slab."""

        def trucate_layers(row):
            wl_depth = row['wl_depth']
            layers = np.array(row['layers'])

            # Get layer thicknesses and the end depths of each layer
            thicknesses = layers[:, 1]
            end_depths = np.cumsum(thicknesses)

            # Find the index lower coordiante of the layer exceeds the wl_depth
            idx = np.searchsorted(end_depths, wl_depth, side='right')

            # If the end depth exceeds wl_depth, adjust the last layer's thickness
            if idx < len(layers):
                # Adjust the thickness of the last layer if needed
                new_thickness = wl_depth - end_depths[idx - 1]
                # Add the last layer if the adjusted thickness is greater than 0
                if new_thickness > 0:
                    layers = np.vstack(
                        [layers[:idx], [layers[idx][0], new_thickness]]
                    )
                else:
                    layers = layers[:idx]

            return layers

        # Apply truncation to the entire dataframe
        self._df['layers'] = self._df.apply(trucate_layers, axis=1)

        # Remove rows where layers is an empty list
        self._df = self._df[self._df['layers'].apply(lambda x: len(x) > 0)]
