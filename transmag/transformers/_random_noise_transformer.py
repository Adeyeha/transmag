"""
Module: random_noise_transformer
Description: Contains the RandomNoiseTransformer class for adding random noise to a magnetogram map.
"""

import numpy as np
from ..utils.logger import VerboseLogger
from ..utils.helper import checktype, to_rgb
from ._byte_scaling_transformer import ByteScalingTransformer

class RandomNoiseTransformer:

    def __init__(self, gauss=300, verbose=False, **kwargs):
        """
        Random Noise Transformer

        RandomNoiseTransformer introduces random noise to a magnetogram map by applying noise to each pixel. 
        The noise is generated from a uniform distribution within the specified range of -gauss to gauss.

        Parameters:
        - gauss (int, optional): The maximum absolute value of the Gauss noise to be added. Default is 300.
        - verbose (bool, optional): If True, enable verbose logging. Default is False.
        - **kwargs: Additional keyword arguments.

        Basic Example:
        
        # Create an instance of the transformer
        transformer = RandomNoiseTransformer()
        # Load a sample magnetogram
        magnetogram, magnetogram_header, bitmap = load_fits_data()
        # Transform the magnetogram
        transformed_magnetogram = transformer.transform(magnetogram, scale=255, rgb=True)

        """
        self.gauss = gauss
        self.verbose = verbose
        self.logger = VerboseLogger(verbose=self.verbose)
        self.requires_bitmap = False
        self.orient_changing = False
        self.kwargs = kwargs

    def _random_noise(self, magnetogram, gauss):
        """
        Add random noise to a magnetogram map.

        For each pixel in the magnetogram, random noise within the range of -gauss to gauss
        is added from a uniform distribution. By default, gauss is set to 300,
        meaning the noise will range between -300 and 300 Gauss.

        Parameters:
        - magnetogram (numpy array): The magnetogram map to which noise will be added.
        - gauss (int, optional): The maximum absolute value of the Gauss noise to be added. Default is 300.

        Returns:
        - numpy array: The noisy magnetogram map.
        """
        # Check if input is a NumPy array
        checktype(magnetogram, np.ndarray)

        # Generate random noise from a uniform distribution within the specified range
        noise = np.random.uniform(-gauss, gauss, magnetogram.shape)

        # Add the noise to the magnetogram
        noisy_magnetogram = magnetogram + noise

        return noisy_magnetogram

    def transform(self, magnetogram, rgb=False, scale=None):
        """
        Transform the input magnetogram by adding random noise.

        Parameters:
        - magnetogram (numpy array): The magnetogram map to be transformed.
        - rgb (bool, optional): If True, generate RGB array. Default is False.
        - scale (float, optional): Scaling factor applied to the output array.

        Returns:
        - numpy array: The transformed (noisy) magnetogram map.
        """
        output_array = self._random_noise(magnetogram, self.gauss)

        if scale is not None:
            output_array = ByteScalingTransformer(scaler=scale).transform(output_array)


        if rgb:
            output_array = to_rgb(output_array)

        return output_array
