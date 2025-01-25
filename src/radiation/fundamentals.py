import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Define Constants
h = 6.62607015e-34  # Planck's constant in J·s
c = 3e8          # speed of light in m/s
k_B = 1.380649e-23  # Boltzmann constant in J/K

def planck_function ( self , wavelength , temperature ) :
    """ Calculate spectral radiance from Planck ’s law .

    Args : 
    wavelength ( float or array ) : Wavelength in meters
    temperature ( float ) : Temperature in Kelvin
    
    Returns :
    float or array : Spectral radiance in W .m -2. sr -1. um -1
    """
    
    # Calculate Spectral Radiance in W .m -2. sr -1. um -1
    exponent = (h * c) / (wavelength * k_B * temperature)
    spectral_radiance = (2 * h * c**2) / (wavelength**5) * (1 / (np.exp(exponent) - 1))
    
    return spectral_radiance


def planck_inv ( self, wavelength , spectral_radiance ) :
    """ Calculate temperature from spectral radiance using Planck ’s law .

    Args :
    wavelength ( float ) : Wavelength in meters
    spectral_radiance ( float ) : Spectral radiance in W .m -2. sr -1. um -1

    Returns :
    float : Temperature in Kelvin
    """

    # Calculate Temperature
    term = (2 * h * c**2) / (wavelength**5 * spectral_radiance)
    #if term <= 1:
        #raise ValueError("Spectral radiance is too high for the given wavelength.")

    log_term = np.log(term + 1)
    temperature = (h * c) / (wavelength * k_B * log_term)
    
    return temperature


def brightness_temperature ( self , radiance , band_center , band_width ) :
    """ Calculate brightness temperature for a given rectangular bandpass .
    
    Implements numerical integration over a rectangular bandpass defined
    by its center wavelength and width . The bandpass is assumed to have
    unity transmission within its bounds and zero outside .

    Args :
    radiance ( float ) : Observed radiance in W .m -2. sr -1. um -1
    band_center ( float ) : Center wavelength of bandpass in meters
    band_width ( float ) : Width of bandpass in meters

    Returns :
    float : Brightness temperature in Kelvin

    Raises :
    ValueError : If band_width <= 0 or band_center <= band_width /2
    """
    
    # Check Input Validity
    if band_width <= 0:
        raise ValueError("band_width must be greater than 0")
    if band_center <= band_width / 2:
        raise ValueError("band_center must be greater than half of band_width")
    
    # Define Wavelength Range for the Bandpass
    lambda_min = band_center - band_width / 2
    lambda_max = band_center + band_width / 2
    
    # Calculate Radiance over the Bandpass
    def planck_radiance(wavelength):
        """Planck's function to calculate radiance at each wavelength"""
        exponent = (h * c) / (wavelength * k_B * 1)  # T = 1 to invert later
        return (2 * h * c**2) / (wavelength**5) * (1 / (np.exp(exponent) - 1))
    
    # Calculate Brightness Temperature
    def brightness_temp_integrand(wavelength):
        """Integrand function to compute the temperature corresponding to the radiance"""
        return radiance / planck_radiance(wavelength)
    
    # Integrate over the bandpass range
    integral, _ = quad(brightness_temp_integrand, lambda_min, lambda_max)
    
    # Compute the effective brightness temperature using the average
    brightness_temp = 1 / integral
    
    return brightness_temp
    
    
    
    
def radiance ( self , temperature , band_center , band_width ) :
    """ Calculate band - integrated radiance for a given temperature and
    rectangular bandpass .

    Integrates Planck function over a rectangular bandpass defined
    by its center wavelength and width . The bandpass is assumed to
    have unity transmission within its bounds and zero outside .

    Args :
    temperature ( float ) : Temperature in Kelvin
    band_center ( float ) : Center wavelength of bandpass in meters

    band_width ( float ) : Width of bandpass in meters
    
    Returns :
    float : Band - integrated radiance in W .m -2. sr -1

    Raises :
    ValueError : If temperature <= 0 , band_width <= 0 , or
    band_center <= band_width /2
    """
    
    # Check Input Validity
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")
    if band_width <= 0:
        raise ValueError("Band width must be greater than 0")
    if band_center <= band_width / 2:
        raise ValueError("Band center must be greater than half of band_width")

    # Define Wavelength Range for the Bandpass
    lambda_min = band_center - band_width / 2
    lambda_max = band_center + band_width / 2
    
    # Perform Numerical Integration over the Bandpass Range using Planck's Function
    radiance_int, _ = quad(planck_function, lambda_min, lambda_max, args=(temperature,))
    
    return radiance_int
    
    
def calculate_NEDT ( self , temperature , NER , band_center , band_width ) :
    """ Calculate the noise - equivalent differential temperature ( NEDT )
    for given scene temperature and noise - equivalent radiance ( NER ) .

    Uses numerical derivative of band - integrated radiance with respect
    to temperature to determine the temperature uncertainty corresponding
    to the NER .

    Args :
    temperature ( float ) : Scene temperature in Kelvin
    NER ( float ) : Noise - equivalent radiance in W .m -2. sr -1
    band_center ( float ) : Center wavelength of bandpass in meters
    band_width ( float ) : Width of bandpass in meters

    Returns :
    float : NEDT in Kelvin

    Raises :
    ValueError : If temperature <= 0 , NER <= 0 , band_width <= 0 ,
    or band_center <= band_width /2
    """
    
    # Check Input Validity
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")
    if NER <= 0:
        raise ValueError("NER must be greater than 0")
    if band_width <= 0:
        raise ValueError("Band width must be greater than 0")
    if band_center <= band_width / 2:
        raise ValueError("Band center must be greater than band_width / 2")

    # Calculate Band-Integrated Radiance for the Given Temperature
    radiance_int = radiance(temperature, band_center, band_width)
    
    # Perturb the temperature slightly to compute the numerical derivative
    delta_T = 0.01  # Small temperature perturbation (in Kelvin)
    radiance_at_T_plus_delta = band_integrated_radiance(temperature + delta_T, band_center, band_width)
    
    # Numerical derivative of band-integrated radiance with respect to temperature
    derivative_B = (radiance_at_T_plus_delta - radiance_at_T) / delta_T
    
    # Compute the NEDT (Noise-Equivalent Differential Temperature)
    nedt_value = NER / np.abs(derivative_B)
    
    return nedt_value
