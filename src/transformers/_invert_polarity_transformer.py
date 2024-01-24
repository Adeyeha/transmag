"""
Module: invert_polarity_transformer
Description: Contains the InvertPolarityTransformer class for inverting the polarity of a magnetogram map.
"""

import numpy as np
from ..utils.helper import checktype,to_rgb
from ..utils.logger import VerboseLogger
from ._byte_scaling_transformer import ByteScalingTransformer

class InvertPolarityTransformer:

    def __init__(self, verbose=False, **kwargs):
        """
        Invert Polarity Transformer

        InvertPolarityTransformer reverses the polarity of a magnetogram map by multiplying all values in the magnetogram by -1.
        This transformation effectively flips the direction of the magnetic field lines.
        
        Parameters:
        - verbose (bool, optional): If True, enable verbose logging. Default is False.
        - **kwargs: Additional keyword arguments.
        """
        self.verbose = verbose
        self.logger = VerboseLogger(verbose=self.verbose)
        self.kwargs = kwargs

    def _invert_polarity(self, magnetogram):
        """
        Invert the polarity of a magnetogram map.

        This function multiplies all the values in the magnetogram map by -1,
        effectively reversing the direction of the magnetic field lines.

        Parameters:
        - magnetogram (numpy array): The magnetogram map whose polarity is to be inverted.

        Returns:
        - numpy array: The polarity-inverted magnetogram map.
        """
        # Check if input is a NumPy array
        checktype(magnetogram, np.ndarray)

        # Invert the polarity by multiplying by -1
        inverted_magnetogram = magnetogram * -1

        return inverted_magnetogram

    def transform(self, magnetogram, rgb=False, scale=None):
        """
        Transform the input magnetogram by inverting its polarity.

        Parameters:
        - magnetogram (numpy array): The magnetogram map to be transformed.
        - rgb (bool, optional): If True, generate RGB array. Default is False.
        - scale (float, optional): Scaling factor applied to the output array.

        Returns:
        - numpy array: The polarity-inverted magnetogram map.
        """

        output_array = self._invert_polarity(magnetogram)

        if scale is not None:
            output_array = ByteScalingTransformer(scaler=scale).transform(output_array)

        if rgb:
            output_array = to_rgb(output_array)

        return output_array

