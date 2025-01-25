import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Define Constants
h = 6.62607015e-34  # Planck's constant in J·s
c = 3e8          # speed of light in m/s
k_B = 1.380649e-23  # Boltzmann constant in J/K


def planck_function (wavelength, temperature) :
    """ Calculate spectral radiance from Planck ’s law .

    Args : 
    wavelength ( float or array ) : Wavelength in meters
    temperature ( float ) : Temperature in Kelvin
    
    Returns :
    float or array : Spectral radiance in W .m -2. sr -1. um -1
    """
    
    # Calculate Spectral Radiance in W/m^2/sr/um from Planck's
    exponent = (h * c) / (wavelength * k_B * temperature)
    spectral_radiance = (2 * h * c**2) / (wavelength**5) * (1 / (np.exp(exponent) - 1)) * 1e-6
    
    return spectral_radiance



def planck_inv (wavelength, spectral_radiance) :
    """ Calculate temperature from spectral radiance using Planck ’s law .

    Args :
    wavelength ( float ) : Wavelength in meters
    spectral_radiance ( float ) : Spectral radiance in W .m -2. sr -1. um -1

    Returns :
    float : Temperature in Kelvin
    """

    # Calculate Temperature in Kelvin from Inverse Planck's
    term = (2 * h * c**2) / (wavelength**5 * spectral_radiance)
    log_term = np.log(term + 1)
    temperature = (h * c) / (wavelength * k_B * log_term)
    
    return temperature



def brightness_temperature (radiance, band_center, band_width) :
    """ Calculate brightness temperature for a given rectangular bandpass .
    
    Implements numerical integration over a rectangular bandpass defined
    by its center wavelength and width. The bandpass is assumed to have
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
    
    # Function to integrate (difference between Planck's law and observed radiance)
    def integrand(wavelength, temperature):
        return planck_function(wavelength, temperature) - radiance

    # Function to calculate the integral of the Planck function over the bandpass
    def bandpass_integral(temperature):
        integral, _ = quad(integrand, lambda_min, lambda_max, args=(temperature,))
        return integral

    # Solve for temperature by numerical methods
    from scipy.optimize import fsolve

    # Use fsolve to find the temperature that gives a zero value for the integral
    temperature_guess = 300  # Initial guess for the temperature in K
    brightnessTemp = fsolve(bandpass_integral, temperature_guess)
    
    return brightnessTemp
    
    
    
def radiance (temperature, band_center, band_width) :
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
    if np.isscalar(temperature):
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")
    else:    
        if any(temperature <= 0):
            raise ValueError("Temperature must be greater than 0")
    if band_width <= 0:
        raise ValueError("Band width must be greater than 0")
    if band_center <= band_width / 2:
        raise ValueError("Band center must be greater than half of band_width")

    # Define Wavelength Range for the Bandpass
    lambda_min = band_center - band_width / 2
    lambda_max = band_center + band_width / 2
    wavelengths = np.linspace(lambda_min, lambda_max, 1000)
    
    # Perform Numerical Integration over the Bandpass Range using Planck's Function
    #radiance_int, _ = quad(planck_function, lambda_min, lambda_max, args=(temperature,))
    
    if np.isscalar(temperature):
        rads = planck_function(wavelengths, temperature)
        return scipy.integrate.simpson(rads, x=wavelengths)
    else:
        rads = np.array([planck_function(wavelengths, T) for T in temperature])
        return np.array([scipy.integrate.simpson(R, x=wavelengths) for R in rads])
    
    
    
def calculate_NEDT (temperature, NER, band_center, band_width) :
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
    radInt = radiance(temperature, band_center, band_width)
    
    # Perturb the temperature slightly to compute the numerical derivative
    delta_T = 0.01  # Small temperature perturbation (in Kelvin)
    radiance_at_T_plus_delta = band_integrated_radiance(temperature + delta_T, band_center, band_width)
    
    # Numerical derivative of band-integrated radiance with respect to temperature
    derivative_B = (radiance_at_T_plus_delta - radiance_at_T) / delta_T
    
    # Compute the NEDT (Noise-Equivalent Differential Temperature)
    nedt_value = NER / np.abs(derivative_B)
    
    return nedt_value
