"""
Module Name: histogram_equalization_transformer.py
Description: This module contains the HistogramEqualizationTransformer class, which is a transformer for applying histogram equalization to a magnetogram map.
"""

import numpy as np
from ..utils.logger import VerboseLogger
from ..utils.helper import checktype,to_rgb
from ._byte_scaling_transformer import ByteScalingTransformer

class HistogramEqualizationTransformer:

    def __init__(self, bins=256, range=[0, 256], verbose=False, **kwargs):
        """
        Histogram Equalization Transformer

        HistogramEqualizationTransformer enhances the contrast of a magnetogram map through the application of histogram equalization. 
        The module redistributes intensity values, promoting improved visual contrast in the magnetogram.

        Args:
        - bins (int): Number of bins in the histogram. Default is 256.
        - range (list): Range of intensity values for the histogram. Default is [0, 256].
        - verbose (bool): If True, display verbose output during transformation. Default is False.
        - **kwargs: Additional keyword arguments for future expansion.
        """
        self.bins = bins
        self.range = range
        self.verbose = verbose
        self.logger = VerboseLogger(verbose=self.verbose)
        self.kwargs = kwargs

    def _histogram_equalization(self, magnetogram, bins, range):
        """
        Apply histogram equalization to a magnetogram map using NumPy.

        Args:
        - magnetogram (numpy array): The magnetogram map to be enhanced.

        Returns:
        - numpy array: The contrast-enhanced magnetogram map.
        """

        # Check if input is a NumPy array
        checktype(magnetogram, np.ndarray)

        # Flatten the image to 1D array for histogram computation
        flat_magnetogram = magnetogram.flatten()

        # Calculate the histogram
        histogram, bins = np.histogram(flat_magnetogram, bins, range, density=True)

        # Compute the cumulative distribution function (CDF)
        cdf = histogram.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]  # Normalize the CDF

        # Use linear interpolation of CDF to find new pixel values
        equalized_magnetogram = np.interp(flat_magnetogram, bins[:-1], cdf_normalized)

        return equalized_magnetogram.reshape(magnetogram.shape)

    def transform(self, magnetogram, rgb=False, scale=None):
        """
        Apply histogram equalization transformation to a magnetogram.

        Args:
        - magnetogram (numpy array): The magnetogram map to be transformed.
        - rgb (bool, optional): If True, generate RGB array. Default is False.
        - scale (float, optional): Scaling factor applied to the output array.

        Returns:
        - numpy array: The transformed magnetogram map with enhanced contrast.
        """

        output_array = self._histogram_equalization(magnetogram, self.bins, self.range)

        if scale is not None:
            output_array = ByteScalingTransformer(scaler=scale).transform(output_array)

        if rgb:
            output_array = to_rgb(output_array)

        return output_array 