# These relations might be wrong. please check them with ETC

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Constants
h = 6.62607015 * 10 ** (-34)  # Planck's constant
c = 299792458  # Speed of light
k = 1.380649 * 10 ** (-23)  # Boltzmann constant
pix_scale = 0.25  # Pixel scale in arcseconds per pixel
pix_size = 1.5 * 10 ** (-5)  # Pixel size in meters

# Telescope data. These are dummy datas. Please put INO's datas.
diameter = 8.2
surface = math.pi * diameter ** 2 / 4
ccd_seeing = 0.25
max_count = 100000
ron = 3.15
dark = 0.000583

efficiencies = {
    'u': 0.15,
    'g': 0.5,
    'r': 0.6,
    'i': 0.5,
    'z': 0.4
}

ab_vega = {
    'u': 0.91,
    'g': -0.08,
    'r': 0.16,
    'i': 0.37,
    'z': 0.54
}

SDSS_FILTERS = {
    'u': {'center': 354.3, 'lambda_min': 324, 'lambda_max': 395},
    'g': {'center': 477, 'lambda_min': 405, 'lambda_max': 552},
    'r': {'center': 623.1, 'lambda_min': 552, 'lambda_max': 691},
    'i': {'center': 762.5, 'lambda_min': 695, 'lambda_max': 844},
    'z': {'center': 913.4, 'lambda_min': 826, 'lambda_max': 1000}
}

def mag_to_photon_flux(syst, mag, filter_band):
    if syst == 'AB':
        photon_flux = 1 / h * (np.log(c / (SDSS_FILTERS[filter_band]['lambda_min'] * 10 ** (-9))) - np.log(c / (SDSS_FILTERS[filter_band]['lambda_max']* 10 ** (-9)))) * 10 ** (-0.4 * (mag + 56.1))
    elif syst == 'Vega':
        ab_mag = mag + ab_vega[filter_band]
        photon_flux = 1 / h * (np.log(c / (SDSS_FILTERS[filter_band]['lambda_min'] * 10 ** (-9))) - np.log(c / (SDSS_FILTERS[filter_band]['lambda_max']* 10 ** (-9)))) * 10 ** (-0.4 * (ab_mag + 56.1))
    return photon_flux




def calculate_nsky(pwv, fli, airmass, fwhm, filter_band):  # Dummy function. Exact function needs to be obtained from experimental data
    solid_angle = math.pi * fwhm ** 2
    return 2390 / 2.41 * solid_angle

def time_for_snr_point(snr_target, photon_flux, efficiency, nsky, bin, npix, ron, dark, surface):
    nbin = bin ** 2
    
    a = snr_target
    b = photon_flux * efficiency * surface
    c = photon_flux * efficiency * surface + nbin * nsky + npix * dark
    d = nbin * ron ** 2

    time = (a ** 2 * c + np.sqrt(a ** 4 * c ** 2 + 4 * a ** 2 * b ** 2 * d)) / (2 * b ** 2)
    
    return time


def time_for_snr_ext(snr_target, photon_flux, efficiency, solid_angle, nsky, bin, npix, ron, dark, surface):
    nbin = bin ** 2
    a = snr_target
    b = photon_flux * efficiency * surface * solid_angle
    c = photon_flux * efficiency * surface * solid_angle + nbin * nsky + npix * dark
    d = nbin * ron ** 2

    time = (a ** 2 * c + np.sqrt(a ** 4 * c ** 2 + 4 * a ** 2 * b ** 2 * d)) / (2 * b ** 2)
    
    return time
