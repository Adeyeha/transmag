"""
Module: gaussian_blur_transformer
Description: Contains the GaussianBlurTransformer class for applying Gaussian blurring to a magnetogram map.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from ..utils.helper import checktype, to_rgb
from ..utils.logger import VerboseLogger
from ._byte_scaling_transformer import ByteScalingTransformer

class GaussianBlurTransformer:

    def __init__(self, sigma=10, verbose=False, **kwargs):
        """
        Gaussian Blur Transformer

        GaussianBlurTransformer applies Gaussian blurring to a magnetogram map. Utilizing a Gaussian kernel, 
        the extent of blurring is determined by the 'sigma' value. A higher sigma value leads to more pronounced blurring.
        
        Parameters:
        - sigma (float, optional): The standard deviation of the Gaussian kernel. Default is 10.
        - verbose (bool, optional): If True, enable verbose logging. Default is False.
        - **kwargs: Additional keyword arguments.
        """
        self.sigma = sigma
        self.verbose = verbose
        self.logger = VerboseLogger(verbose=self.verbose)
        self.kwargs = kwargs

    def _gaussian_blur(self, magnetogram, sigma):
        """
        Apply Gaussian blurring to a magnetogram map.

        This function uses a Gaussian kernel for blurring, where 'sigma' determines the extent of blurring.

        Parameters:
        - magnetogram (numpy array): The magnetogram map to be blurred.
        - sigma (float): The standard deviation of the Gaussian kernel.

        Returns:
        - numpy array: The blurred magnetogram map.
        """
        # Check if input is a NumPy array
        checktype(magnetogram, np.ndarray)

        # Apply Gaussian blurring using the specified sigma value
        blurred_magnetogram = gaussian_filter(magnetogram, sigma=sigma)
        return blurred_magnetogram

    def transform(self, magnetogram, rgb=False, scale=None):
        """
        Transform the input magnetogram by applying Gaussian blurring.

        Parameters:
        - magnetogram (numpy array): The magnetogram map to be transformed.
        - rgb (bool, optional): If True, generate RGB array. Default is False.
        - scale (float, optional): Scaling factor applied to the output array.

        Returns:
        - numpy array: The blurred magnetogram map.
        """
        output_array = self._gaussian_blur(magnetogram, self.sigma)

        if scale is not None:
            output_array = ByteScalingTransformer(scaler=scale).transform(output_array)

        if rgb:
            output_array = to_rgb(output_array)

        return output_array 
