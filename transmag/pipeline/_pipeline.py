"""
Module: pipeline
Description: Contains the Pipeline class for applying a sequence of transformations to an input data array.
"""

import numpy as np
from ..transformers import RotationTransformer, FlipTransformer, BitmapCroppingTransformer, InformativePatchTransformer, ByteScalingTransformer
from ..utils.logger import VerboseLogger
from ..utils.exceptions import NoBitmapError, NoHeaderError, PipelineExecutionError, NullOutputError, NullOutputWarning, PipelineExecutionWarning
from ..utils.helper import to_rgb,checkoptions,checktype

class Pipeline:
    """
    A pipeline to apply a sequence of transformations to an input data array.

    The pipeline takes a list of transformer instances and applies them sequentially.
    The pipeline also performs validation checks for required additional data.
    """
    def __init__(self, steps, on_error="ignore", on_null_output="ignore", verbose=None):
        """
        Initialize the PreprocessingPipeline.

        Parameters:
        - steps (list): List of transformer instances to be applied.
        - on_error (str, optional): Define the behavior when an error occurs during transformation.
            - "ignore": Continue with the next transformer and log a warning (default).
            - "raise": Raise an error immediately when an error occurs.
        - on_null_output (str, optional): Define the behavior when a transformer returns None.
            - "ignore": Continue with the next transformer and log a warning (default).
            - "raise": Raise an error if a transformer returns None.
        - verbose (bool, optional): If True, enable verbose logging. Default is None.

        Basic Example:
        
        from transmag.pipeline import Pipeline

        # Create a pipeline with multiple transformations
        pipe = Pipeline([
        GaussianBlurTransformer(sigma=20),
        RandomNoiseTransformer(gauss=200),
        InvertPolarityTransformer(),
        ])

        # Load a sample magnetogram
        magnetogram, magnetogram_header, bitmap = load_fits_data()

        # Transform the magnetogram
        transformed_magnetogram = pipe.transform(magnetogram, scale=255, rgb=True)
        """
        self.steps = steps
        self.transformerSequence = ((BitmapCroppingTransformer, InformativePatchTransformer),)

        # Check if any step requires bitmap (BitmapCroppingTransformer)
        self.requires_bitmap = any(step.requires_bitmap for step in steps)
        self.verbose = verbose
        self.logger = VerboseLogger(verbose=self.verbose)
        self.validoptions = ["ignore", "raise"]
        self.on_error = on_error if checkoptions(on_error, self.validoptions, "on_error") else None
        self.on_null_output = on_null_output if checkoptions(on_null_output, self.validoptions, "on_null_output") else None
   
        
    def _background(self, img, header, lb1=-256, ub1=255, lb2=-50, ub2=50):
        """
        Apply background processing to the image.

        Args:
        - img (numpy array): Input image array.
        - header (dict): Header information for the image.
        - lb1, ub1, lb2, ub2 (float, optional): Threshold values for image clipping. Defaults are specified values.

        Returns:
        - numpy array: Processed image array.
        """
        # Clip image values to the specified range
        img = np.clip(img, lb1, ub1)

        # Create coordinate arrays
        x = np.arange(0, 4096)
        y = np.arange(0, 4096)

        # Extract header information
        cx = header['CRPIX1']
        cy = header['CRPIX2']
        r = header['RSUN_OBS'] / header['CDELT1']

        # Circular Mask
        mask1 = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 > r ** 2

        # Masks based on intensity range
        mask2 = np.logical_and(img >= lb2, img <= ub2)
        mask3 = np.where(img > ub2)
        mask4 = np.where(img < lb2)

        # Set values outside the intensity range to 0
        img[mask2] = 0.

        # Set pixels outside the circular region to NaN
        img[mask1] = np.nan

        return img


    def _reorganize_pipeline(self):
        """
        Organize the pipeline to follow the specified sequence.

        This function rearranges the pipeline steps to match the specified transformer sequence.
        """
        # Check if all transformers in the group are present in the pipeline
        for transformer_group in self.transformerSequence:
            if all(any(isinstance(step, transformer) for step in self.steps) for transformer in transformer_group):
                indices = [self.steps.index(instance) for transformer in transformer_group for instance in self.steps if isinstance(instance, transformer)]

                # Check if the indices are in sequence
                if not indices == sorted(indices):
                    # Rearrange the steps within the subgroup to follow the specified sequence
                    min_index = min(indices)
                    shift_size = len(indices) - 1

                    # Remove other values from the list before shifting
                    subgroup_values = [self.steps[idx] for idx in indices]

                    # Set values at indices to None
                    for idx in indices:
                        self.steps[idx] = None

                    # Extend the list with None elements
                    self.steps.extend([None] * shift_size)

                    # Shift values after the smallest index by the size of indices - 1
                    for i in range(len(self.steps) - shift_size - 1, min_index, -1):
                        self.steps[i + shift_size] = self.steps[i]

                    # Insert the values from the subgroup at the correct positions
                    for i, step in enumerate(subgroup_values):
                        self.steps[min_index + i] = step

                    # Remove None values
                    self.steps = [x for x in self.steps if x]


    def transform(self, input_array, bitmap_data=None, header=None, fulldisk=False, rgb=False, scale=None):
        """
        Transform the input array using the specified pipeline.

        Args:
        - input_array (numpy array): The input data array.
        - bitmap_data (numpy array, optional): Bitmap data for transformers that require it.
        - header (dict, optional): Header data for full disk type.
        - fulldisk (bool, optional): If True, perform transformations for fulldisk. Default is False.
        - rgb (bool, optional): If True, generate RGB array. Default is False.
        - scale (float, optional): Scaling factor applied to the output array.

        Returns:
        - numpy array: Transformed array.
        """

        # Check if input is a NumPy array
        checktype(input_array,np.ndarray)

        # Validation for bitmap_data if BitmapCroppingTransformer is in the pipeline
        if self.requires_bitmap and bitmap_data is None:
            raise NoBitmapError(f"Bitmap data must be provided for transformations that require bitmap data.")

        if fulldisk:
            if header is None:
                raise NoHeaderError("Header data must be provided for fulldisk fits")
            input_array = self._background(input_array, header)

        # Set verbose
        if self.verbose is not None:
            for step in self.steps:
                step.verbose = self.verbose

        # Organize Pipeline to be sequential with transformer sequence
        self._reorganize_pipeline()

        try:

            self.input_array = input_array
            self.bitmap_data = bitmap_data
            
            for step in self.steps:

                # Set on_null_output attribute
                if hasattr(step, 'on_null_output'):
                    setattr(step, 'on_null_output', self.on_null_output)

                try:
                    # Both bitmap and magnetogram are passed for transformers that require bitmap
                    if step.requires_bitmap:
                        input_array = step.transform(self.input_array, self.bitmap_data )
                    else:
                        input_array = step.transform(self.input_array)

                    # Transform both bitmap and magnetogram when applying orientation changing transformers and requires_bitmap is True
                    if step.orient_changing and self.requires_bitmap:
                        bitmap_data = step.transform(self.bitmap_data )

                    if input_array is None:
                        # Skip Transformation if on_null_output is ignore
                        if self.on_null_output == "ignore":
                            self.logger.warn(f"Skipping {type(step).__name__} due to None output", NullOutputWarning)
                        # Raise Exception if on_null_output is raise
                        elif self.on_null_output == "raise":
                            error_message = f"Transformer returned None. Check input data and parameters"
                            raise NullOutputError(error_message)
                    else:
                        self.input_array = input_array
                        self.bitmap_data = bitmap_data

                except Exception as e:
                    # Skip Transformation if on_error is ignore
                    if self.on_error == 'ignore':
                        self.logger.warn(f"Skipping {type(step).__name__} due to error: {str(e)}", PipelineExecutionWarning)

                    # Raise Exception if on_error is raise
                    elif self.on_error == 'raise':
                        raise e

        except Exception as e:
            # Constructing an error message with the transformer class name
            error_message = f"Error occurred in transformer {type(step).__name__}: {str(e)}"
            raise PipelineExecutionError(error_message) from e

        # Apply scaling to avoid RGBScaleError
        if scale is not None:
            input_array = ByteScalingTransformer(scaler=scale).transform(input_array)

        # Apply RGB to output array
        if rgb:
            input_array = to_rgb(input_array)

        return input_array
