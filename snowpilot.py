# Standard library imports
import os
from xml.etree import ElementTree as ET

# Third-party imports
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm


def query_snowpilot(
    output_dir="data/snowpilot", total_observation=33566, batch_size=500
):
    """
    Query the SnowPilot API and download snow pit observations.

    This function queries the SnowPilot API to retrieve snow pit observations in XML format.
    The observations are downloaded in batches and saved to the specified output directory.

    Parameters
    ----------
    output_dir : str, optional
        The directory where the downloaded XML files will be saved. Default is 'data/snowpilot'.
    total_observation : int, optional
        The total number of observations to retrieve. Default is 33,566.
    batch_size : int, optional
        The number of observations to retrieve in each batch. Default (and maximum) is 500.
    """
    print('Querying SnowPilot...')
    # Base URL for SnowPilot query
    xml_feed_url = "https://snowpilot.org/snowpilot-query-feed.xml?"
    # Query parameters
    query_args = (
        # "PIT_NAME=&",
        # "recent_dates=0",
        # "OBS_DATE_MIN",
        # "OBS_DATE_MAX",
        # "USERNAME=",
        # "AFFIL=none",
        # "proximity_pit_nid=",
        # "proximity_radius=",
        # "proximity_search=Proximity%20Search",
        # "testpit=0",
        f"per_page={batch_size}",
        "caaml_feed=CAAML%20XML%20Output",
        "page=",
    )
    # Construct the query string
    query = "&".join(query_args)

    # Total number of pages (calculated by dividing 35000 by 500)
    total_pages = total_observation // 500 + 1

    # Directory to save the XML files
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each page and download the XML
    for i in tqdm(
        range(total_pages), desc="Downloading XML files", unit="page"
    ):
        # Construct the URL with page number
        url = f"{xml_feed_url}{query}{i}"

        # Send the request to download the XML
        response = requests.get(url)

        # Check if the response is successful
        if response.status_code == 200:
            # Save the file
            file_path = os.path.join(output_dir, f"snowpilot_page_{i}.xml")
            with open(file_path, "wb") as file:
                file.write(response.content)
        else:
            print(
                f"Failed to download page {i}, status code: {response.status_code}"
            )

    print("Download completed!")


def concatenate_xml_files(
    directory="data/snowpilot",
    output_file="data/snowpilot_all.xml",
):
    """
    Concatenate multiple XML files into a single XML file.

    This function iterates over all XML files in the specified directory, parses each file,
    and appends the Pit_Observation elements to a new combined XML file.

    Parameters
    ----------
    directory : str, optional
        The path to the directory containing the XML files to concatenate. Default is 'data/snowpilot'.
    output_file : str, optional
        The path to the output XML file to save the concatenated data. Default is 'data/snowpilot_all.xml'.
    """
    # Create the root element for the combined XML file
    root = ET.Element("Pit_Data")

    # Iterate over all XML files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            try:
                # Parse each XML file
                tree = ET.parse(file_path)
                file_root = tree.getroot()

                # Append each Pit_Observation element to the root of the combined file
                for pit_obs in file_root.findall("Pit_Observation"):
                    root.append(pit_obs)

            except ET.ParseError as e:
                print(f"Error parsing file: {file_path} - {e}")
                continue  # Skip this file and move on to the next

    # Create the final tree and write it to the output file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def filter_pst_observations(
    input_file="data/snowpilot_all.xml",
    output_file="data/snowpilot_psts.xml",
):
    """
    Filter Pit_Observation elements containing PST shear test results from an XML file.

    This function parses an input XML file containing snow pit observations, filters out
    the Pit_Observation elements that contain a Shear_Test_Result with code="PST", and writes
    the filtered observations to an output XML file.

    Parameters
    ----------
    input_file : str, optional
        The path to the input XML file to parse. Default is 'data/snowpilot_all.xml'.
    output_file : str, optional
        The path to the output XML file to save the filtered data. Default is 'data/snowpilot_psts.xml'.
    """

    # Parse the input XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Create a new root for the filtered XML
    filtered_root = ET.Element("Pit_Data")

    # Iterate through all Pit_Observation elements
    for pit_obs in root.findall("Pit_Observation"):
        # Check if there is a Shear_Test_Result with code="PST"
        for shear_test in pit_obs.findall("Shear_Test_Result"):
            if shear_test.get("code") == "PST":
                # If found, append this Pit_Observation to the filtered root
                filtered_root.append(pit_obs)
                break  # No need to check further Shear_Test_Result elements

    # Write the filtered XML to the output file
    filtered_tree = ET.ElementTree(filtered_root)
    filtered_tree.write(output_file, encoding="utf-8", xml_declaration=True)


def count_psts(xml_file="data/snowpilot_psts.xml"):
    """
    Count the number of Pit_Observation elements with PSTs in an XML file.

    Parameters
    ----------
    xml_file : str, optional
        The path to the XML file to parse. Default is 'data/snowpilot_psts.xml'.
    """

    # Parse the XML file
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Count the number of Pit_Observation elements
        pit_observation_count = len(root.findall("Pit_Observation"))
        return print(f'Found {pit_observation_count} pits with PSTs')
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")


def xml_to_pkl(
    xml_file='data/snowpilot_psts.xml',
    pkl_file='data/snowpilot_psts.pkl',
):
    """
    Convert snowpilot XML files to a pickled dataframe.

    This function parses a snowpilot XML file containing snow pit observations,
    extracts relevant data, and saves it into a pickle file for later use.

    Parameters
    ----------
    xml_file : str, optional
        The path to the XML file to parse. Default is 'data/snowpilot_psts.xml'.
    pkl_file : str, optional
        The path to the pickle file to save the parsed data. Default is 'data/snowpilot_psts.pkl'.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
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
            k: v for k, v in user_info.attrib.items() if k in keep_user_info
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
                {k: v for k, v in layer.attrib.items() if k in keep_layer_data}
            )
        pit_data.update({'layers': layer_data})

        def process_shear_tests(shear_tests, keep_test_data):
            """
            Process shear tests to filter and extract relevant PST data.

            This function filters the provided shear tests to include only those with a 'code' of 'PST'.
            It then processes these tests to keep only the most relevant data based on specific criteria.

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
            pst_tests = [
                test
                for test in shear_tests
                if test.attrib.get('code') == 'PST'
            ]

            # If no PST tests found, return an empty dict
            if not pst_tests:
                return {}

            # Convert sdepth, lengthOfColumn, and lengthOfCut to float for comparison and calculation
            for test in pst_tests:
                test['sdepth'] = float(test.get('sdepth'))
                test['lengthOfColumn'] = float(test.get('lengthOfColumn'))
                test['lengthOfCut'] = float(test.get('lengthOfCut'))

            # Step 1: Keep only the smallest value of sdepth (can have multiple)
            min_sdepth = min(test['sdepth'] for test in pst_tests)
            pst_tests = [t for t in pst_tests if t['sdepth'] == min_sdepth]

            # Step 2: Keep only the largest value of lengthOfColumn (can have multiple)
            max_lengthOfColumn = max(t['lengthOfColumn'] for t in pst_tests)
            pst_tests = [
                test
                for test in pst_tests
                if test['lengthOfColumn'] == max_lengthOfColumn
            ]

            # Step 3: If there are both 'END' and non-'END' dataCode entries, keep only those that are 'END'
            has_end = any(test.get('dataCode') == 'END' for test in pst_tests)
            has_non_end = any(t['dataCode'] != 'END' for t in pst_tests)

            if has_end and has_non_end:
                pst_tests = [t for t in pst_tests if t['dataCode'] == 'END']

            # Step 4: Average lengthOfCut
            avg_lengthOfCut = np.mean([t['lengthOfCut'] for t in pst_tests])

            # Prepare the final test data to update
            # Take one test as representative (since we filtered, they should be similar) and update the avg_lengthOfCut
            final_test = pst_tests[0]
            final_test['lengthOfCut'] = avg_lengthOfCut

            # Only keep attributes that are in keep_test_data
            test_data = {
                k: v for k, v in final_test.items() if k in keep_test_data
            }

            return test_data

        # Get PST results
        shear_tests = pit_obs.findall('Shear_Test_Result')
        keep_test_data = [
            'dataCode',  # PST result (End, Arr, SF)
            'depthUnits',  # units of layer depth
            'lengthOfColumn',  # PST column length
            'lengthOfCut',  # critical cut length
            'sdepth',  # depth of layer tested in stability test
        ]
        # Usually only one PST in each pit, but applying logic to handle multiples
        test_data = process_shear_tests(shear_tests, keep_test_data)
        pit_data.update(test_data)

        # Get density profile
        # density_profile = pit_obs.find('Density_Profile')
        # if density_profile.attrib["profile"] != "":
        #     print(density_profile.attrib["profile"])

        # Append the row data to the list
        data.append(pit_data)

    # Convert list of dictionaries to DataFrame and save as pickle
    df = pd.DataFrame(data)
    df.to_pickle(pkl_file)


def load_pkl(pkl_file='data/snowpilot_psts.pkl'):
    """
    Load a pickle file into a pandas DataFrame.

    Parameters
    ----------
    pkl_file : str, optional
        The path to the pickle file to load. Default is 'data/snowpilot_psts.pkl'.

    Returns
    -------
    pd.DataFrame
        The DataFrame loaded from the pickle file.
    """
    return pd.read_pickle(pkl_file)


def parse_layers(list_of_dicts):
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

    # Compute layer thickness
    # Translate hand hardness to density
    # Define a default value if grainType is not available

    return list_of_lists


if __name__ == "__main__":
    # query_snowpilot()
    # concatenate_xml_files()
    # filter_pst_observations()
    # xml_to_pkl()
    df = load_pkl()

    # Exeriment whit data
    df['layers'] = df['layers'].apply(parse_layers)

    print()
    print('index 4')
    for layer in df.loc[4, 'layers']:
        print(layer)

    print()
    print('index 3568')
    for layer in df.loc[3568, 'layers']:
        print(layer)

    print()
    print(df)
