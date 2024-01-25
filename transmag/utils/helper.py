"""
Module: helper
Description: Contains helper functions.
"""
import numpy as np
from ..utils.exceptions import RGBScaleError
from pathlib import Path


def checktype(X,dtype):
    """
    Check if the input is a NumPy array. Raise a ValueError if not.

    Parameters:
    - X: The input.

    Raises:
    - ValueError: If the input is not a NumPy array.
    """
    if not isinstance(X, dtype):
        raise TypeError(f"Expected {dtype}, got {type(X)}")
    return True

def checkoptions(param, valid_options, param_name):
    """
    Check if a variable is one of the valid options.

    Args:
    - param: The param to check.
    - valid_options (list): List of valid options.
    - param_name str: Name of the param to check

    Raises:
    - TypeError: If the param is not one of the valid options.
    """
    if param not in valid_options:
        raise TypeError(f"parameter `{param_name}` must be one of {' or '.join(valid_options)}")
    return True


def calculate_information_loss(pre, post):
    """
    Calculate information loss between pre and post images.

    Parameters:
    - pre (numpy array): The pre-transformation image.
    - post (numpy array): The post-transformation image.

    Returns:
    - float: Information loss.
    """
    # Check for zero division
    if np.sum(np.abs(post)) == 0:
        # Return a large value indicating significant information loss
        return float('inf')

    # Calculate information loss
    return abs(round(1 - (np.sum(np.abs(pre)) / np.sum(np.abs(post))), 2))


def _generate_rgb_array(colormap_file, input_array,):
    """
    Generate RGB array using a colormap and scaled integer data.

    Args:
    - colormap_file (str): Path to the CSV file containing the colormap data.
    - input_array (numpy array): Scaled integer data.
    - scale_indices (bool): Flag to indicate whether to scale the indices.

    Returns:
    - numpy array: RGB array.
    """

    try:
        # Load the colormap from the CSV file
        colormap_data = np.genfromtxt(colormap_file, delimiter=',')

        # Map the values using the colormap
        rgb_array = np.zeros((input_array.shape[0], input_array.shape[1], 3), dtype=np.uint8)

        for i in range(3):
            rgb_array[:, :, i] = colormap_data[input_array[:, :].astype(int), i]

    except IndexError as index_error:
        # Check if the input array scale exceeds the RGB scale
        if np.max(input_array) > np.max(colormap_data):
            raise RGBScaleError("Scale of input array exceeds RGB scale. Try setting `scale=255` or lower in transform method")      
        raise  index_error

    except Exception as e:
        raise e

    return rgb_array


def to_rgb(input_array):
    """
    Convert an input array to RGB representation using a colormap.

    Args:
    - input_array (numpy array): Input array to convert to RGB.

    Returns:
    - numpy array: RGB representation of the input array.
    """
    checktype(input_array, np.ndarray)

    colormap_file_path = Path(__file__).parent / 'hmi_mag.csv'
    if not colormap_file_path.exists():
        colormap_file_path = 'https://raw.githubusercontent.com/sunpy/sunpy/main/sunpy/visualization/colormaps/data/hmi_mag.csv'

    rgb_array = _generate_rgb_array(colormap_file=str(colormap_file_path), input_array=input_array)
    
    return rgb_array