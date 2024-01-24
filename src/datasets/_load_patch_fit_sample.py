from astropy.io import fits
from ..utils.helper import checkoptions
import warnings

def load_fits_data(fits_type="patch", sample_no=1):
    """
    Load FITS data for solar flare analysis.

    Args:
    - fits_type (str): Type of FITS data to load. Either `patch` or `fulldisk`. Default is `patch`.
    - sample_no (int): Sample number. Default is 1.

    Returns:
    - tuple: Tuple containing magnetogram data, magnetogram header, and bitmap data.
    """

    checkoptions(fits_type,["patch","fulldisk"],'fits_type')
    
    # Read AR_PATCH
    HMI_fits_path = f'solarflarepy/datasets/fit_samples/hmi.sharp_{fits_type}_sample_{sample_no}_TAI.magnetogram.fits'
    HMI_fits = fits.open(HMI_fits_path, cache=False)
    HMI_fits.verify('fix')
    dataHMI = HMI_fits[1].data
    dataHMI_header = HMI_fits[1].header

    # Read Bitmap
    bitmap_path = f'solarflarepy/datasets/fit_samples/hmi.sharp_{fits_type}_sample_{sample_no}_TAI.bitmap.fits'
    bitmap_hdul = fits.open(bitmap_path, cache=False)
    bitmap_hdul.verify('fix')
    bitmap_data = bitmap_hdul[0].data

    return dataHMI, dataHMI_header, bitmap_data
