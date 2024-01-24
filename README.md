# TransMag: Magnetogram Transformer Module

<!-- ![TransMag Logo](transmag_logo.png) -->

TransMag is a Python module designed for transforming magnetogram maps using various techniques that reduce information loss. This module is particularly useful for augmenting magnetograms, addressing data imbalance for predictive modeling.

## Installation

TransMag requires Python (>= 3.9) and the following dependencies:
- numpy (>=1.17.0)
- scipy (>=1.3.0)
- astropy (>=5.0.6,!=5.1.0)

Install TransMag using pip:
```bash
pip install transmag
```

or using conda:
```bash
conda install -c conda-forge transmag
```

## Usage Example

```python

from transmag import FlipTransformer
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# Create an instance of the transformer
transformer = FlipTransformer(direction='vertical')

# Load a sample magnetogram
magnetogram, magnetogram_header, bitmap  = load_fits_data()

# Transform the magnetogram
transformed_magnetogram = transformer.transform(magnetogram)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(dataHMI, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(bitmap_data, cmap='gray')
plt.title('Flipped Image')
```

## Documentation

For detailed documentation and examples, visit the [TransMag Documentation](.docs\examples\TRANSFORMER.md).

## Changelog

See the [Changelog](https://github.com/Adeyeha/transmag/blob/master/CHANGELOG.md) for information about notable changes.

<!-- ## Contributing

We welcome contributions from the community. Refer to the [Contributing Guide](https://github.com/transmag/transmag/blob/main/CONTRIBUTING.md) for more details. -->

## License

TransMag is distributed under the GNU License. See [GNU General Public License, Version 3](https://github.com/Adeyeha/transmag/blob/master/LICENSE.txt)   for details.


## Citation

If you use TransMag in a scientific publication, please consider citing our work. Refer to the [Citation Guide](https://transmag.example.com/citation) for details.

## Acknowledgment

This work was supported in part by NASA Grant Award No. NNH14ZDA001N, NASA/SRAG Direct Contract and two NSF Grant
Awards: No. AC1443061 and AC1931555.

***

This software is distributed using the [GNU General Public License, Version 3](https://github.com/Adeyeha/transmag/blob/master/LICENSE.txt)  
![alt text](https://github.com/Adeyeha/transmag/blob/master/docs/images/gplv3-88x31.png)

***

© 2024 Temitope Adeyeha, Chetraj Pandey, Berkay Aydin

[Data Mining Lab](http://dmlab.cs.gsu.edu/)

[Georgia State University](http://www.gsu.edu/)
