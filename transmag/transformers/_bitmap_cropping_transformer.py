"""
Module: bitmap_cropping_transformer
Description: Contains the BitmapCroppingTransformer class for cropping an AR Patch based on bitmap data.
"""

import numpy as np
from scipy.ndimage import label

from ..utils.exceptions import NoBitmapError
from ..utils.logger import VerboseLogger
from ..utils.helper import checktype,to_rgb
from ._byte_scaling_transformer import ByteScalingTransformer

class BitmapCroppingTransformer:

    def __init__(self, verbose=False, **kwargs):
        """
        Bitmap Cropping Transformer

        BitmapCroppingTransformer performs cropping of an Active Region (AR) Patch based on bitmap data, 
        utilizing the largest connected component in the bitmap data to determine the cropping region.
        
        Parameters:
        - verbose (bool, optional): If True, enable verbose logging. Default is False.
        - **kwargs: Additional keyword arguments.

        Basic Example:
        
        # Create an instance of the transformer
        transformer = BitmapCroppingTransformer()
        # Load a sample magnetogram
        magnetogram, magnetogram_header, bitmap = load_fits_data()
        # Transform the magnetogram
        transformed_magnetogram = transformer.transform(magnetogram, bitmap, scale=255, rgb=True)
        
        """
        self.verbose = verbose
        self.logger = VerboseLogger(verbose=self.verbose)
        self.requires_bitmap = True
        self.orient_changing = False
        self.kwargs = kwargs

    def _bitmap_cropping(self, AR_patch, bitmap_data):
        """
        Crop the AR Patch based on the largest connected component in the bitmap data.

        The function thresholds the bitmap data to identify regions of interest,
        handles NaN values, finds the largest connected component, and crops the AR Patch accordingly.

        Parameters:
        - AR_Patch (numpy array): The input Active Region patch to be cropped.
        - bitmap_data (numpy array): The bitmap data used to identify the cropping region.

        Returns:
        - numpy array: The cropped region of the AR Patch.
        """
        # Check if input AR_patch and bitmap_data are NumPy arrays
        checktype(AR_patch, np.ndarray)
        checktype(bitmap_data, np.ndarray)

        # Check for NaN values in the bitmap data and create a mask
        nan_mask = np.isnan(bitmap_data)

        # Threshold the bitmap data to binary values (0s and 1s)
        bitmap_data = (bitmap_data > 2).astype(np.uint8)

        # Exclude regions with NaN values from the binary bitmap
        bitmap_data[nan_mask] = 0

        # Label the connected components in the updated bitmap
        labels, num_features = label(bitmap_data)

        # Calculate the sizes of each connected component
        component_sizes = np.bincount(labels.ravel())

        # Find the label of the largest component (excluding the background component)
        largest_component_label = np.argmax(component_sizes[1:]) + 1

        # Find the coordinates of the largest component
        largest_component_coords = np.argwhere(labels == largest_component_label)

        # Get the minimum and maximum coordinates of the largest component
        min_x, min_y = largest_component_coords.min(axis=0)
        max_x, max_y = largest_component_coords.max(axis=0)

        # Crop the AR Patch to the region defined by the largest component
        cropped_magnetogram = AR_patch[min_x:max_x + 1, min_y:max_y + 1]

        return cropped_magnetogram

    def transform(self, AR_patch, bitmap_data=None, rgb=False, scale=None):
        """
        Transform the input AR Patch by cropping based on bitmap data.

        Parameters:
        - AR_patch (numpy array): The input Active Region patch to be cropped.
        - bitmap_data (numpy array): The bitmap data used to identify the cropping region.
        - rgb (bool, optional): If True, generate RGB array. Default is False.
        - scale (float, optional): Scaling factor applied to the output array.


        Returns:
        - numpy array: The transformed (cropped) region of the AR Patch.
        """
        if bitmap_data is None:
            raise NoBitmapError("Bitmap data must be provided for transformations that require bitmap data.")

        output_array = self._bitmap_cropping(AR_patch, bitmap_data)

        if scale is not None:
            output_array = ByteScalingTransformer(scaler=scale).transform(output_array)

        if rgb:
            output_array = to_rgb(output_array)

        return output_array
