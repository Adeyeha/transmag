"""
Module: pad_transformer
Description: Contains the PaddingTransformer class for adding padding to an image array.
"""

import numpy as np
from ..utils.logger import VerboseLogger
from ..utils.helper import checktype, to_rgb
from ..utils.exceptions import InvalidOutputSizeError
from ._byte_scaling_transformer import ByteScalingTransformer

class PadTransformer:
    """
    Transformer for adding padding to an image array.

    This class wraps the padding function and applies it to an input array.
    The padding is added to make the image of a specified size.
    """
    def __init__(self, output_size=1024, infer_output_size=False, constant_value=0, verbose=False, **kwargs):
        """
        Padding Transformer

        PadTransformer introduces padding to an image array, ensuring it reaches a designated size. 
        The transformation adds the necessary padding to achieve the desired dimensions.

        Parameters:
        - output_size (int, optional): The desired size of the output image. Default is 1024.
        - infer_output_size (bool, optional): If True, the output size is automatically inferred based on the maximum dimension of the original array. Default is False.
        - constant_value (int, optional): The padding value to use. Default is 0.
        - verbose (bool, optional): If True, enable verbose logging. Default is False.
        - **kwargs: Additional keyword arguments.
        """
        self.constant_value = constant_value
        self.output_size = output_size
        self.infer_output_size = infer_output_size
        self.verbose = verbose
        self.logger = VerboseLogger(verbose=self.verbose)
        self.kwargs = kwargs

    def _padding(self, X, constant_value, output_size):
        """
        Add padding to an image array to make it of size output_size x output_size.

        Parameters:
        - X (numpy array): The input image array.
        - constant_value (int, optional): The padding value to use. Default is 0.

        Returns:
        - numpy array: The padded image array.
        """


        # Check if input is a NumPy array
        checktype(X,np.ndarray)

        # Get the current height and width of the image
        h, w = X.shape

        # Calculate the total padding required for height and width
        pad_h = output_size - h
        pad_w = output_size - w

        # Calculate padding for top/bottom and left/right
        # Use floor division to handle odd and even dimensions uniformly
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding to the image and return the result
        return np.pad(X, ((pad_top, pad_bottom), (pad_left, pad_right)),
                      'constant', constant_values=constant_value)

    def transform(self, X, rgb=False, scale=None):
        """
        Transform the input image array by applying padding.

        Parameters:
        - X (numpy array): The input image array.
        - rgb (bool, optional): If True, generate RGB array. Default is False.
        - scale (float, optional): Scaling factor applied to the output array.

        Returns:
        - numpy array: The transformed (padded) image array.
        """
        try:

            output_array = self._padding(X, self.constant_value, self.output_size)
        except ValueError as e:
            if self.infer_output_size:
                self.logger.info("ValueError occurred during padding. Attempting the operation with the maximum dimension of the original array.")
                output_array = self._padding(X, self.constant_value, max(X.shape))
            else:
                error_message = "The `output_size` parameter cannot be set lower than the highest dimension of the original array. Try setting `infer_output_size=True` or providing a higher value for `output_size` in the transfromer definition"
                raise InvalidOutputSizeError(error_message) from e 
        except Exception as e:
            raise e

        if scale is not None:
            output_array = ByteScalingTransformer(scaler=scale).transform(output_array)

        if rgb:
            output_array = to_rgb(output_array)
        
        return output_array
        

