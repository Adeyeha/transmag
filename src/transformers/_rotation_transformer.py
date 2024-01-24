"""
Module: rotation_transformer
Description: Contains the RotationTransformer class for rotating a magnetogram map by a specified angle.
"""

import numpy as np
from scipy.ndimage import rotate
from ..utils.exceptions import InformationLossWarning
from ..utils.logger import VerboseLogger
from ..utils.helper import calculate_information_loss, checktype, to_rgb
from ._byte_scaling_transformer import ByteScalingTransformer

class RotationTransformer:

    def __init__(self, angle=0, order=3, loss_threshold=1.0, verbose=False, **kwargs):
        """
        Rotation Transformer

        RotationTransformer facilitates the rotation of a magnetogram map by a user-specified angle. 
        The transformation is accomplished through the rotation function, enabling rotation at any angle within the range of 0 to 360 degrees.
        
        Parameters:
        - angle (float, optional): The angle of rotation in degrees (0 to 360). Default is 0.
        - order (integer, optional): The rotation order. Default is 3.
        - loss_threshold (float, optional): The threshold for information loss. If the information loss exceeds this threshold, the input array will be returned. Default is 1.0.
        - verbose (bool, optional): If True, enable verbose logging. Default is False.
        - **kwargs: Additional keyword arguments.
        """
        self.angle = angle
        self.order = order
        self.loss_threshold = loss_threshold
        self.verbose = verbose
        self.logger = VerboseLogger(verbose=self.verbose)
        self.kwargs = kwargs

    def _rotate(self, input, order, angle):
        """
        Rotate the input map by a specified angle.

        Parameters:
        - input (numpy array): The input map to be rotated.
        - order (integer, optional): The rotation order. Default is 3.
        - angle (float, optional): The angle of rotation in degrees (0 to 360). Default is 0.

        Returns:
        - numpy array: The rotated input map.
        """
        # Check if input is a NumPy array
        checktype(input,np.ndarray)

        # Rotate the magnetogram and return the result
        return rotate(input, angle, order=order, reshape=False, mode='nearest')

    def transform(self, input_array, rgb=False, scale=None):
        """
        Transform the magnetogram/bitmap_data.

        Parameters:
        - input_array (numpy array): The magnetogram/bitmap to be transformed.
        - rgb (bool, optional): If True, generate RGB array. Default is False.
        - scale (float, optional): Scaling factor applied to the output array.

        Returns:
        - Tuple[numpy array]: The transformed magnetogram/bitmap_data.
        """
        if np.count_nonzero(np.isnan(input_array)):
            self.order = 0

        output_array = self._rotate(input_array, self.order, self.angle)

        information_loss = calculate_information_loss(input_array, output_array)

        if information_loss > self.loss_threshold:
            warning_msg = f"Skipping RotationTransformer due to information loss ({information_loss}) exceeding threshold ({self.loss_threshold})."
            self.logger.warn(warning_msg,InformationLossWarning)

            return input_array

        if scale is not None:
            output_array = ByteScalingTransformer(scaler=scale).transform(output_array)

        if rgb:
            output_array = to_rgb(output_array)

        return output_array

