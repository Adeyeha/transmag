"""
Module: byte_scaling_transformer
Description: Contains the ByteScalingTransformer class for applying byte scaling to an input array.
"""

import numpy as np
from ..utils.logger import VerboseLogger
from ..utils.helper import checktype, to_rgb

class ByteScalingTransformer:

    def __init__(self, min_value=None, max_value=None, scaler=255, verbose=False, **kwargs):
        """
        Byte Scaling Transformer

        ByteScalingTransformer performs scaling of values in an input array, excluding NaNs,
        to fit within a specified range.

        Parameters:
        - min_value (float, optional): The minimum value for scaling. If None, computed from the array.
        - max_value (float, optional): The maximum value for scaling. If None, computed from the array.
        - scaler (float, optional): The scaling factor. Default is 255.
        - verbose (bool, optional): If True, enable verbose logging. Default is False.
        - **kwargs: Additional keyword arguments.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.scaler = scaler
        self.verbose = verbose
        self.logger = VerboseLogger(verbose=self.verbose)
        self.kwargs = kwargs

    def _bytscal_with_nan(self, input_array, min_value, max_value):
        """
        Perform byte scaling on the input array while preserving NaN values.

        The function scales the values in the input array (excluding NaNs) to the specified range.
        If min_value or max_value are not provided, it computes them while ignoring NaNs.

        Parameters:
        - input_array (numpy array): The input array to be scaled.
        - min_value (float, optional): The minimum value for scaling. If None, computed from the array.
        - max_value (float, optional): The maximum value for scaling. If None, computed from the array.

        Returns:
        - numpy array: The byte-scaled array with NaN values preserved.
        """
        # Check if input is a NumPy array
        checktype(input_array, np.ndarray)

        # Compute min_value and max_value if they are not provided, ignoring NaNs
        input_array = np.nan_to_num(input_array)
        if min_value is None:
            min_value = np.nanmin(input_array)
        if max_value is None:
            max_value = np.nanmax(input_array)

        # Perform byte scaling while preserving NaN values
        # Scale non-NaN values to specified range
        scaled_array = np.where(np.isnan(input_array), np.nan,
                                ((input_array - min_value) / (max_value - min_value) * self.scaler).astype(np.uint8))

        return scaled_array

    def transform(self, X, rgb=False):
        """
        Transform the input array by applying byte scaling.

        Parameters:
        - X (numpy array): The input array to be transformed.
        - rgb (bool, optional): If True, generate RGB array. Default is False.

        Returns:
        - numpy array: The transformed (byte-scaled) array.
        """

        output_array = self._bytscal_with_nan(X, self.min_value, self.max_value)

        if rgb:
            output_array = to_rgb(output_array)

        return output_array
