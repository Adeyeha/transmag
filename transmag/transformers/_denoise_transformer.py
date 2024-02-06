"""
Module: denoise_transformer
Description: Contains the DenoiseTransformer class for denoising an active region patch.
"""

import numpy as np
from ..utils.logger import VerboseLogger
from ..utils.helper import checktype, to_rgb
from ._byte_scaling_transformer import ByteScalingTransformer

class DenoiseTransformer:

    def __init__(self, lower_bound=-50, upper_bound=50, maximum_range=256, verbose=False, **kwargs):
        """
        Denoise Transformer

        DenoiseTransformer enhances the quality of an active region patch by employing denoising techniques. 
        It constrains the values of the active region patch within a defined range and subsequently
        eliminates values falling below a specified noise threshold.
        
        Parameters:
        - lower_bound (int): The lower bound of the noise threshold. Default is -50.
        - upper_bound (int): The upper bound of the noise threshold. Default is 50.
        - maximum_range (int, optional): The maximum range for clipping values. Default is 256.
        - verbose (bool, optional): If True, enable verbose logging. Default is False.
        
        Basic Example:
        
        # Create an instance of the transformer
        transformer = DenoiseTransformer()
        # Load a sample magnetogram
        magnetogram, magnetogram_header, bitmap = load_fits_data()
        # Transform the magnetogram
        transformed_magnetogram = transformer.transform(magnetogram, scale=255, rgb=True)

        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.maximum_range = maximum_range
        self.verbose = verbose
        self.logger = VerboseLogger(verbose=self.verbose)
        self.requires_bitmap = False
        self.orient_changing = False
        self.kwargs = kwargs

    def _denoise(self, active_region_patch, lower_bound, upper_bound, maximum_range):
        """
        Denoise the active region patch.

        This function first limits the values of the active region patch to a default specified range (-256, 256).
        Then, it sets values within the noise threshold (lower_bound, upper_bound) to zero.

        Args:
        - active_region_patch (numpy array): The input Active Region (AR) patch.
        - lower_bound (int): The lower bound of the noise threshold.
        - upper_bound (int): The upper bound of the noise threshold.
        - maximum_range (int, optional): The maximum range for clipping values. Default is 256.
        Returns:
        - numpy array: The denoised Active Region patch.
        """

        # Check if input is a NumPy array
        checktype(active_region_patch,np.ndarray)

        # Clip the active region patch values to the range [-maximum_range, maximum_range]
        clipped_patch = np.clip(active_region_patch, -1 * maximum_range, maximum_range)

        # Identify values outside the noise threshold range
        outside_noise_threshold = np.logical_or(clipped_patch < lower_bound, clipped_patch > upper_bound)

        # Set the values within the noise threshold to zero
        denoised_patch = np.where(outside_noise_threshold, clipped_patch, float(0))

        return denoised_patch

    def transform(self, X, rgb=False, scale=None):
        """
        Transform the input active region patch by denoising.

        Parameters:
        - X (numpy array): The input Active Region (AR) patch.
        - rgb (bool, optional): If True, generate RGB array. Default is False.
        - scale (float, optional): Scaling factor applied to the output array.


        Returns:
        - numpy array: The denoised Active Region patch.
        """
        output_array =  self._denoise(X, self.lower_bound, self.upper_bound, self.maximum_range)

        if scale is not None:
            output_array = ByteScalingTransformer(scaler=scale).transform(output_array)

        if rgb:
            output_array = to_rgb(output_array)

        return output_array

