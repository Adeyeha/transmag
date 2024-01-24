"""
Module: flip_transformer
Description: Contains the FlipTransformer class for flipping a magnetogram map either horizontally or vertically.
"""

import numpy as np
from ..utils.logger import VerboseLogger
from ..utils.exceptions import InformationLossWarning
from ..utils.helper import checktype,calculate_information_loss,to_rgb, checkoptions
from ._byte_scaling_transformer import ByteScalingTransformer

class FlipTransformer:

    def __init__(self, direction='horizontal', loss_threshold=1.0, verbose=False, **kwargs):
        """
        Flip Transformer

        FlipTransformer modifies a magnetogram map by flipping it either horizontally or vertically. 
        It encapsulates the logic for flipping the magnetogram based on a specified direction, 
        which can be either 'horizontal' or 'vertical'.
        
        Parameters:
        - direction (str, optional): The direction of the flip. Should be either 'horizontal' or 'vertical'. Default is 'horizontal'.
        - loss_threshold (float, optional): The threshold for information loss. If the information loss exceeds this threshold, the input array will be returned. Default is 1.0.
        - verbose (bool, optional): If True, enable verbose logging. Default is False.
        - **kwargs: Additional keyword arguments.
        """

        self.validoptions = ["horizontal", "vertical"]
        self.direction = direction if checkoptions(direction, self.validoptions, "direction") else None
        self.verbose = verbose
        self.loss_threshold = loss_threshold
        self.logger = VerboseLogger(verbose=self.verbose)
        self.kwargs = kwargs

    def _flip(self, input, direction):
        """
        Flip an input map either horizontally or vertically.

        Parameters:
        - input (numpy array): The input map to be flipped.
        - direction (str, optional): The direction of the flip. Should be either 'horizontal' or 'vertical'. Default is 'horizontal'.

        Returns:
        - numpy array: The flipped input map.
        """

        # Check if input is a NumPy array
        checktype(input,np.ndarray)

        if direction.lower() == 'horizontal':
            # Flip the magnetogram horizontally
            return np.fliplr(input)
        elif direction.lower() == 'vertical':
            # Flip the magnetogram vertically
            return np.flipud(input)


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
        output_array = self._flip(input_array, self.direction)

        information_loss = calculate_information_loss(input_array, output_array)

        if information_loss > self.loss_threshold:
            warning_msg = f"Skipping FlipTransformer due to information loss ({information_loss}) exceeding threshold ({self.loss_threshold})."
            self.logger.warn(warning_msg,InformationLossWarning)

            return input_array

        if scale is not None:
            output_array = ByteScalingTransformer(scaler=scale).transform(output_array)

        if rgb:
            output_array = to_rgb(output_array)

        return output_array


