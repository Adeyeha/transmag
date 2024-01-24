## Overview of the `transmag` Module

The `transmag` module is a comprehensive package designed for transforming magnetogram maps using various techniques to reduce information loss and augment magnetograms for predictive modeling. It provides a collection of transformers, each tailored for a specific transformation method. These transformers can be easily integrated into your data preprocessing pipeline.

## Helpers

### Helper Function: `display_magnetogram`
To simplify the process of visualizing original and transformed magnetograms, we provide a helper function named `display_magnetogram`. This function takes the magnetogram, bitmap and transformed maps and displays them side by side for easy comparison.

```python
import matplotlib.pyplot as plt

def display_magnetogram(original, bitmap, transformed):
    """
    Display original and transformed magnetograms side by side.

    Args:
    - original (numpy array): Original magnetogram map.
    - bitmap (numpy array): Bitmap data
    - transformed (numpy array): Transformed magnetogram map.
    """

    # List of data to display
    data_list = [original, bitmap, transformed]
    titles = ['Original Image', 'Bitmap Data', 'Processed Image']

    # Visualize the images
    plt.figure(figsize=(15, 5))
    for i, (data, title) in enumerate(zip(data_list, titles), 1):
        plt.subplot(1, 4, i)
        plt.imshow(data, cmap='gray')
        plt.title(title)

    plt.show()
```

### Helper Function: `broilerplate`

For brevity reasons, we have included a broilerplate code that can be used throughout the guide. Replace `<TRANSFORMER_TYPE>` with the transformer of choice.

```python 
# Boilerplate: Replace TRANSFORMER_TYPE with the desired transformer type (e.g., HistogramEqualizationTransformer)
from transmag.transformers import `<TRANSFORMER_TYPE>`
import matplotlib.pyplot as plt
from transmag.datasets import load_fits_data

magnetogram, magnetogram_header, bitmap_data = load_fits_data()

transformer = `<TRANSFORMER_TYPE()>`

# Apply the transform 
transformed = transformer.transform(dataHMI)

# OR

# Apply the transform with rgb
transformed = transformer.transform(dataHMI, scale=255, rgb=True)

# Visualize
display_magnetogram(magnetogram, bitmap_data, transformed)
```

### Sample Output
![output](https://github.com/Adeyeha/transmag/blob/master/docs/images/transformer_output.png)

## Transformers

### 1. `DenoiseTransformer`

The `DenoiseTransformer` enhances the quality of an active region patch by employing denoising techniques. It constrains the values of the active region patch within a defined range and subsequently eliminates values falling below a specified noise threshold.

#### Example Usage:

```python
from transmag.transformers import DenoiseTransformer

# Instantiate the transformer
transformer = DenoiseTransformer()

# Replace with transformer in broilerplate code
```

### 2. `FlipTransformer`

The `FlipTransformer` modifies a magnetogram map by flipping it either horizontally or vertically. It encapsulates the logic for flipping the magnetogram based on a specified direction, which can be either 'horizontal' or 'vertical'.

#### Example Usage:

```python
from transmag.transformers import FlipTransformer

# Instantiate the transformer
transformer = FlipTransformer(direction='horizontal')

# Replace with transformer in broilerplate code
```

### 3. `GaussianBlurTransformer`

The `GaussianBlurTransformer` applies Gaussian blurring to a magnetogram map. Utilizing a Gaussian kernel, the extent of blurring is determined by the 'sigma' value. A higher sigma value leads to more pronounced blurring.

#### Example Usage:

```python
from transmag.transformers import GaussianBlurTransformer

# Instantiate the transformer
transformer = GaussianBlurTransformer(sigma=100)

# Replace with transformer in broilerplate code
```

### 4. `InvertPolarityTransformer`

The `InvertPolarityTransformer` reverses the polarity of a magnetogram map by multiplying all values in the magnetogram by -1. This transformation effectively flips the direction of the magnetic field lines.

#### Example Usage:

```python
from transmag.transformers import InvertPolarityTransformer

# Instantiate the transformer
transformer = InvertPolarityTransformer()

# Replace with transformer in broilerplate code
```

### 5. `PadTransformer`

The `PadTransformer` introduces padding to an image array, ensuring it reaches a designated size. The transformation adds the necessary padding to achieve the desired dimensions.

#### Example Usage:

```python
from transmag.transformers import PadTransformer

# Instantiate the transformer
pad_transformer = PadTransformer()

# Replace with transformer in broilerplate code
```

### 6. `InformativePatchTransformer`

The `InformativePatchTransformer` identifies the optimal patch within an image employing a stride-based approach. This transformer is designed to locate the most informative patch within the input image.

#### Example Usage:

```python
from transmag.transformers import PadTransformer

# Instantiate the transformer
pad_transformer = InformativePatchTransformer(stride=10)

# Replace with transformer in broilerplate code
```


### 7. `RandomNoiseTransformer`

The `RandomNoiseTransformer` introduces random noise to a magnetogram map by applying noise to each pixel. The noise is generated from a uniform distribution within the specified range of -gauss to gauss.

#### Example Usage:

```python
from transmag.transformers import RandomNoiseTransformer

# Instantiate the transformer
random_noise_transformer = RandomNoiseTransformer(gauss=500)

# Replace with transformer in broilerplate code
```

### 8. `ResizeByHalfTransformer`

The `ResizeByHalfTransformer` modifies a 2D numpy array by reducing its size to half of the original dimensions. The transformation achieves the resizing by employing average pooling on the image.

#### Example Usage:

```python
from transmag.transformers import ResizeByHalfTransformer

# Instantiate the transformer
resize_by_half_transformer = ResizeByHalfTransformer()

# Replace with transformer in broilerplate code
```

### 9. `RotationTransformer`

The `RotationTransformer` facilitates the rotation of a magnetogram map by a user-specified angle. The transformation is accomplished through the rotation function, enabling rotation at any angle within the range of 0 to 360 degrees.

#### Example Usage:

```python
from transmag.transformers import RotationTransformer

# Instantiate the transformer
rotation_transformer = RotationTransformer(angle=45)

# Replace with transformer in broilerplate code
```

### 10. `ByteScalingTransformer`

The `ByteScalingTransformer` performs scaling of values in an input array, excluding NaNs, to fit within a specified range.

#### Example Usage:

```python
from transmag.transformers import ByteScalingTransformer

# Instantiate the transformer
byte_scaling_transformer = ByteScalingTransformer()

# Replace with transformer in broilerplate code
```

### 11. `BitmapCroppingTransformer`

The `BitmapCroppingTransformer` performs cropping of an Active Region (AR) Patch based on bitmap data, utilizing the largest connected component in the bitmap data to determine the cropping region.

#### Example Usage:

```python
from transmag.transformers import BitmapCroppingTransformer

# Instantiate the transformer
bitmap_cropping_transformer = BitmapCroppingTransformer()

# Replace with transformer in broilerplate code
```

### 12. `HistogramEqualizationTransformer`

The `HistogramEqualizationTransformer` enhances the contrast of a magnetogram map by redistributing intensity values. This can be particularly useful for improving visual contrast in the magnetogram.

#### Example Usage:
```python
from transmag.transformers import BitmapCroppingTransformer

# Instantiate the transformer
bitmap_cropping_transformer = HistogramEqualizationTransformer()

# Replace with transformer in broilerplate code
```

## Next Steps

Feel free to explore more transformers within the `transmag` module or integrate them into your magnetogram preprocessing pipeline. For detailed guidance on efficiently organizing and applying these transformers, check out the [pipeline](./PIPELINE.md) documentation.


Happy transforming!
