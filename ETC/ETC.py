# Written by Bardiya Alian
# Disclaimer: All data used in this code, including constants, telescope specifications, and sky parameters,
# are dummy data and may differ significantly from real-world values.
# The `calculate_nsky` function is a placeholder and should be replaced with calculations based on INO's experimental sky model.

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from saturation import saturation_chart

# Function to convert Right Ascension from hh:mm:ss format to degrees
def ra_to_degrees(ra_str):
    """
    Convert Right Ascension from hh:mm:ss format to degrees.
    
    Args:
        ra_str (str): Right Ascension in hh:mm:ss format.
        
    Returns:
        float: Right Ascension in degrees.
    """
    hours, minutes, seconds = map(float, ra_str.split(':'))
    degrees = (hours + minutes / 60 + seconds / 3600) * 15
    return degrees

# Function to convert Declination from dd:mm:ss format to degrees
def dec_to_degrees(dec_str):
    """
    Convert Declination from dd:mm:ss format to degrees.
    
    Args:
        dec_str (str): Declination in dd:mm:ss format.
        
    Returns:
        float: Declination in degrees.
    """
    parts = dec_str.split(':')
    degrees = abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600
    if float(parts[0]) < 0:
        degrees = -degrees
    return degrees

# Constants
h = 6.62607015 * 10 ** (-34)  # Planck's constant in J.s
c = 299792458  # Speed of light in m/s
k = 1.380649 * 10 ** (-23)  # Boltzmann constant in J/K
pix_scale = 0.25  # Pixel scale in arcseconds per pixel
pix_size = 1.5 * 10 ** (-5)  # Pixel size in meters

# Telescope data
diameter = 8.2  # Diameter of the telescope in meters
surface = math.pi * diameter ** 2 / 4  # Effective surface area of the telescope in m^2
ccd_seeing = 0.25  # CCD seeing in arcseconds
max_count = 100000  # Maximum count of the CCD
ron = 3.15  # Readout noise in e-
dark = 0.000583  # Dark current in e-/s/pixel

# Efficiencies and magnitude conversions for different filters
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

# SDSS filter specifications: center wavelength, minimum wavelength, maximum wavelength (in nm)
SDSS_FILTERS = {
    'u': {'center': 354.3, 'lambda_min': 324, 'lambda_max': 395},
    'g': {'center': 477, 'lambda_min': 405, 'lambda_max': 552},
    'r': {'center': 623.1, 'lambda_min': 552, 'lambda_max': 691},
    'i': {'center': 762.5, 'lambda_min': 695, 'lambda_max': 844},
    'z': {'center': 913.4, 'lambda_min': 826, 'lambda_max': 1000}
}

# Function to convert magnitude to photon flux
def mag_to_photon_flux(syst, mag, filter_band):
    """
    Convert magnitude to photon flux based on the filter band and magnitude system.
    
    Args:
        syst (str): Magnitude system ('AB' or 'Vega').
        mag (float): Magnitude of the object.
        filter_band (str): The filter band (e.g., 'u', 'g', 'r', 'i', 'z').
        
    Returns:
        float: Photon flux in #/s/m^2.
    """
    if syst == 'AB':
        photon_flux = 1 / h * (np.log(c / (SDSS_FILTERS[filter_band]['lambda_min'] * 10 ** (-9))) -
                               np.log(c / (SDSS_FILTERS[filter_band]['lambda_max']* 10 ** (-9)))) * 10 ** (-0.4 * (mag + 56.1))
    elif syst == 'Vega':
        ab_mag = mag + ab_vega[filter_band]
        photon_flux = 1 / h * (np.log(c / (SDSS_FILTERS[filter_band]['lambda_min'] * 10 ** (-9))) -
                               np.log(c / (SDSS_FILTERS[filter_band]['lambda_max']* 10 ** (-9)))) * 10 ** (-0.4 * (ab_mag + 56.1))
    return photon_flux

# Function to convert photon flux to magnitude
def photon_flux_to_mag(syst, photon_flux, filter_band):
    """
    Convert photon flux to magnitude based on the filter band and magnitude system.
    
    Args:
        syst (str): Magnitude system ('AB' or 'Vega').
        photon_flux (float): Photon flux in #/s/m^2.
        filter_band (str): The filter band (e.g., 'u', 'g', 'r', 'i', 'z').
        
    Returns:
        float: Magnitude of the object.
    """
    if syst == 'AB':
        mag = -2.5 * np.log10(photon_flux * h /
                              (np.log(c / (SDSS_FILTERS[filter_band]['lambda_min'] * 10**(-9))) -
                               np.log(c / (SDSS_FILTERS[filter_band]['lambda_max'] * 10**(-9))))) - 56.1
    elif syst == 'Vega':
        ab_mag = -2.5 * np.log10(photon_flux * h /
                                 (np.log(c / (SDSS_FILTERS[filter_band]['lambda_min'] * 10**(-9))) -
                                  np.log(c / (SDSS_FILTERS[filter_band]['lambda_max'] * 10**(-9))))) - 56.1
        mag = ab_mag - ab_vega[filter_band]
    return mag

# Function to calculate atmospheric seeing based on turbulence, airmass, and filter band
def atm_seeing(turbulance, airmass, filter_band):
    """
    Calculate atmospheric seeing based on turbulence, airmass, and filter band.
    
    Args:
        turbulance (str): Turbulence level (e.g., '10%', '20%', etc.).
        airmass (float): Airmass value.
        filter_band (str): The filter band (e.g., 'u', 'g', 'r', 'i', 'z').
        
    Returns:
        float: Atmospheric seeing in arcseconds.
    """
    seeing0 = {'10%': 0.374, '20%': 0.467, '30%': 0.559 , '50%': 0.650, '70%': 0.832, '85%': 1.106, '100%': 1.746}
    wavelength0 = 557  # Reference wavelength in nm
    atm_seeing = seeing0[turbulance] * airmass ** (3 / 5) * (SDSS_FILTERS[filter_band]['center'] / wavelength0) ** (-1 / 5)
    return atm_seeing

# Function to calculate the Full Width at Half Maximum (FWHM)
def fwhm(ccd_seeing, filter_band, turbulance, airmass, bin):
    """
    Calculate the Full Width at Half Maximum (FWHM) considering contributions from atmospheric, CCD, and telescope optics.
    
    Args:
        ccd_seeing (float): CCD seeing in arcseconds.
        filter_band (str): The filter band (e.g., 'u', 'g', 'r', 'i', 'z').
        turbulance (str): Turbulence level (e.g., '10%', '20%', etc.).
        airmass (float): Airmass value.
        bin (int): Binning factor (e.g., 1 for 1x1, 2 for 2x2).
        
    Returns:
        float: Total FWHM in arcseconds.
    """
    fwhm_atm = atm_seeing(turbulance, airmass, filter_band)
    fwhm_ccd = bin * ccd_seeing
    fwhm_tel = 0.000212 * SDSS_FILTERS[filter_band]['center'] / diameter  # Telescope FWHM
    return np.sqrt(fwhm_atm ** (1/2) +  fwhm_ccd ** (1/2) + fwhm_tel ** (1/2))

# Dummy function to calculate Nsky
def calculate_nsky(pwv, fli, airmass, solid_angle, filter_band):
    """
    Dummy function to calculate sky background noise (Nsky).
    This function should be replaced with calculations based on INO's experimental sky model.
    
    Args:
        pwv (float): Percipitable Water Vapour.
        fli (float): Fractional Lunar Illumination.
        airmass (float): Airmass value.
        solid_angle (float): Solid angle in arcsec^2.
        filter_band (str): The filter band (e.g., 'u', 'g', 'r', 'i', 'z').
        
    Returns:
        float: Nsky value.
    """
    return 2390 / 2.41 * solid_angle  # Placeholder calculation

# Function to calculate the signal for point source imaging
def signal_point_imaging(photon_flux, time, efficiency):
    """
    Calculate the signal for point source imaging.
    
    Args:
        photon_flux (float): Photon flux in #/s/m^2.
        time (float): Exposure time in seconds.
        efficiency (float): Efficiency of the filter.
        
    Returns:
        float: Signal in electrons.
    """
    return photon_flux * time * efficiency * surface

# Function to calculate the signal for extended source imaging
def signal_ext_imaging(photon_flux, time, efficiency, solid_angle):
    """
    Calculate the signal for extended source imaging.
    
    Args:
        photon_flux (float): Photon flux in #/s/m^2.
        time (float): Exposure time in seconds.
        efficiency (float): Efficiency of the filter.
        solid_angle (float): Solid angle in arcsec^2.
        
    Returns:
        float: Signal in electrons.
    """
    return photon_flux * time * efficiency * surface * solid_angle

# Function to calculate the Signal-to-Noise Ratio (SNR)
def snr(signal, nsky, bin, npix, ron, dark, time):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for a given signal and noise parameters.
    
    Args:
        signal (float): Signal in electrons.
        nsky (float): Sky noise in electrons/second.
        bin (int): Binning factor (e.g., 1 for 1x1, 2 for 2x2).
        npix (float): Number of pixels.
        ron (float): Readout noise in electrons.
        dark (float): Dark current in electrons/second/pixel.
        time (float): Exposure time in seconds.
        
    Returns:
        float: Signal-to-Noise Ratio (SNR).
    """
    nbin = bin ** 2
    return signal / np.sqrt(signal + nbin * nsky * time + nbin * ron ** 2 + npix * dark * time)

# Function to calculate the required exposure time for a target SNR (point source)
def time_for_snr_point(snr_target, photon_flux, efficiency, nsky, bin, npix, ron, dark, surface):
    """
    Calculate the exposure time required to achieve a target SNR for point sources.
    
    Args:
        snr_target (float): Target SNR.
        photon_flux (float): Photon flux in #/s/m^2.
        efficiency (float): Efficiency of the filter.
        nsky (float): Sky noise in electrons/second.
        bin (int): Binning factor (e.g., 1 for 1x1, 2 for 2x2).
        npix (float): Number of pixels.
        ron (float): Readout noise in electrons.
        dark (float): Dark current in electrons/second/pixel.
        surface (float): Effective surface area of the telescope in m^2.
        
    Returns:
        float: Required exposure time in seconds.
    """
    nbin = bin ** 2
    
    a = snr_target
    b = photon_flux * efficiency * surface
    c = photon_flux * efficiency * surface + nbin * nsky + npix * dark
    d = nbin * ron ** 2

    time = (a ** 2 * c + np.sqrt(a ** 4 * c ** 2 + 4 * a ** 2 * b ** 2 * d)) / (2 * b ** 2)
    
    return time

# Function to calculate the required exposure time for a target SNR (extended source)
def time_for_snr_ext(snr_target, photon_flux, efficiency, solid_angle, nsky, bin, npix, ron, dark, surface):
    """
    Calculate the exposure time required to achieve a target SNR for extended sources.
    
    Args:
        snr_target (float): Target SNR.
        photon_flux (float): Photon flux in #/s/m^2.
        efficiency (float): Efficiency of the filter.
        solid_angle (float): Solid angle in arcsec^2.
        nsky (float): Sky noise in electrons/second.
        bin (int): Binning factor (e.g., 1 for 1x1, 2 for 2x2).
        npix (float): Number of pixels.
        ron (float): Readout noise in electrons.
        dark (float): Dark current in electrons/second/pixel.
        surface (float): Effective surface area of the telescope in m^2.
        
    Returns:
        float: Required exposure time in seconds.
    """
    nbin = bin ** 2
    
    a = snr_target
    b = photon_flux * efficiency * surface * solid_angle
    c = photon_flux * efficiency * surface * solid_angle + nbin * nsky + npix * dark
    d = nbin * ron ** 2

    time = (a ** 2 * c + np.sqrt(a ** 4 * c ** 2 + 4 * a ** 2 * b ** 2 * d)) / (2 * b ** 2)
    
    return time
 
# Function to calculate the saturation magnitude for point sources
def calculate_saturation_mag_point(time, filter_band, syst, efficiency, sigma, nsky, solid_angle, pix_scale, max_count):
    """
    Calculate the saturation magnitude for point sources.
    
    Args:
        time (float): Exposure time in seconds.
        filter_band (str): The filter band (e.g., 'u', 'g', 'r', 'i', 'z').
        syst (str): Magnitude system ('AB' or 'Vega').
        efficiency (float): Efficiency of the filter.
        sigma (float): Standard deviation of the PSF in arcseconds.
        nsky (float): Sky noise in electrons/second.
        solid_angle (float): Solid angle in arcsec^2.
        pix_scale (float): Pixel scale in arcseconds/pixel.
        max_count (float): Maximum count of the CCD.
        
    Returns:
        float: Saturation magnitude in the given magnitude system.
    """
    F_sky_pix = nsky / solid_angle * pix_scale ** 2  # Sky flux per pixel
    
    photon_flux = (max_count - F_sky_pix * time) / (
        math.sqrt(2 * math.pi) * sigma * math.erf(pix_scale / (math.sqrt(2) * 3 * sigma)) ** 2)
    
    mag = photon_flux_to_mag(syst, photon_flux, filter_band)
    
    return mag

# Function to calculate the saturation magnitude for extended sources
def calculate_saturation_mag_extended(time, filter_band, syst, nsky, solid_angle, pix_scale, max_count):
    """
    Calculate the saturation magnitude for extended sources.
    
    Args:
        time (float): Exposure time in seconds.
        filter_band (str): The filter band (e.g., 'u', 'g', 'r', 'i', 'z').
        syst (str): Magnitude system ('AB' or 'Vega').
        nsky (float): Sky noise in electrons/second.
        solid_angle (float): Solid angle in arcsec^2.
        pix_scale (float): Pixel scale in arcseconds/pixel.
        max_count (float): Maximum count of the CCD.
        
    Returns:
        float: Saturation magnitude in the given magnitude system (per arcsec^2).
    """
    F_sky_pix = nsky / solid_angle * pix_scale ** 2  # Sky flux per pixel
    
    photon_flux_pix = (max_count - F_sky_pix * time) / time
    
    photon_flux_arcsec = photon_flux_pix / pix_scale ** 2
    
    mag = photon_flux_to_mag(syst, photon_flux_arcsec, filter_band)
    
    return mag

# Function to calculate the saturation time for point sources
def calculate_saturation_time_point(mag, filter_band, syst, efficiency, sigma, nsky, solid_angle, pix_scale, max_count):
    """
    Calculate the time to reach saturation for point sources.
    
    Args:
        mag (float): Magnitude of the object.
        filter_band (str): The filter band (e.g., 'u', 'g', 'r', 'i', 'z').
        syst (str): Magnitude system ('AB' or 'Vega').
        efficiency (float): Efficiency of the filter.
        sigma (float): Standard deviation of the PSF in arcseconds.
        nsky (float): Sky noise in electrons/second.
        solid_angle (float): Solid angle in arcsec^2.
        pix_scale (float): Pixel scale in arcseconds/pixel.
        max_count (float): Maximum count of the CCD.
        
    Returns:
        float: Saturation time in seconds.
    """
    photon_flux = mag_to_photon_flux(syst, mag, filter_band)
    
    F_sky_pix = nsky / solid_angle * pix_scale ** 2  # Sky flux per pixel
    
    term = (max_count / ((pix_scale / (2 * math.sqrt(2) * sigma)) ** 2 * photon_flux + F_sky_pix))
    
    time = math.sqrt(term)
    
    return time

# Function to calculate the saturation time for extended sources
def calculate_saturation_time_extended(target_mag, filter_band, syst, nsky, solid_angle, max_count, pix_scale):
    """
    Calculate the time to reach saturation for extended sources.
    
    Args:
        target_mag (float): Target magnitude per arcsec^2.
        filter_band (str): The filter band (e.g., 'u', 'g', 'r', 'i', 'z').
        syst (str): Magnitude system ('AB' or 'Vega').
        nsky (float): Sky noise in electrons/second.
        solid_angle (float): Solid angle in arcsec^2.
        max_count (float): Maximum count of the CCD.
        pix_scale (float): Pixel scale in arcseconds/pixel.
        
    Returns:
        float: Saturation time in seconds.
    """
    target_photon_flux_arcsec = mag_to_photon_flux(syst, target_mag, filter_band)
    
    target_photon_flux_pix = target_photon_flux_arcsec * pix_scale ** 2
    
    F_sky_pix = nsky / solid_angle * pix_scale ** 2  # Sky flux per pixel
    
    time = max_count / (target_photon_flux_pix + F_sky_pix)
    
    return time

# User interaction and options
options = []
while True:
    layer0 = input("""1. Target
2. Sky
3. Seeing
4. Exp time and Sig/Noise
5. give results
6. Saturation Chart
7. Quit
""")

    if layer0 == '1':
        obj_type = input("Object type ((p) Point or (e) Extended): ")
        obj_ra = input("Object Right Ascension (hh:mm:ss): ")
        obj_dec = input("Object Declination (dd:mm:ss): ")
        pos_ang = float(input("CCD position Angle (degree): "))
        ra = ra_to_degrees(obj_ra)
        dec = dec_to_degrees(obj_dec)
        
        if obj_type == 'p':
            filter_band = input("Give your filter band (Filter bands are SDSS u g r i): ")
            mag = float(input("Give your filter magnitude: "))
            syst = input("Give your magnitude system (AB or Vega): ")
            bin = int(input("Give your binning(1 for 1x1 and 2 for 2x2): "))
            options.append('1')
        
        elif obj_type == 'e':
            filter_band = input("Give your filter band (Filter bands are SDSS u g r i): ")
            mag = float(input("Give your filter average magnitude (mag/arcsec^2): "))
            min_mag = float(input("Give your filter minimum magnitude (mag/arcsec^2): "))
            syst = input("Give your magnitude system (AB or Vega): ")
            solid_angle = float(input("Give your object solid angle in space (arcsec^2): "))
            bin = int(input("Give your binning: "))
            options.append('1')
        
        if '3' in options:
            fwhm_tot = fwhm(ccd_seeing, filter_band, turbulance, airmass, bin)
            print(f"IQ to use: {fwhm_tot} arcsec")

    elif layer0 == '2':
        airmass = float(input("Give your airmass (1 - 2.9): "))
        fli = float(input("Give your Fractional Lunar Illumination (0 - 1): "))
        pwv = float(input("Give your Percipitable Water Vapour: "))
        options.append('2')
    
    elif layer0 == '3':
        while True:
            option = input("""turbulance mode:
1. 10%
2. 20%
3. 30%
4. 50%
5. 70%
6. 85%
7. 100%
""")
            if option == '1':
                turbulance = '10%'
                break
            elif option == '2':
                turbulance = '20%'
                break
            elif option == '3':
                turbulance = '30%'
                break
            elif option == '4':
                turbulance = '50%'
                break
            elif option == '5':
                turbulance = '70%'
                break
            elif option == '6':
                turbulance = '85%'
                break
            elif option == '7':
                turbulance = '100%'
                break
        if '1' in options:
            fwhm_tot = fwhm(ccd_seeing, filter_band, turbulance, airmass, bin)
            print(f"IQ to use: {fwhm_tot} arcsec")
        options.append('3')
    
    elif layer0 == '4':
        option2 = input("""1. Give Exp time
2. Give SNR
""")
        if option2 == '1':
            time = float(input("Time (sec): "))
            no = float(input("No. of exposures: "))
            options.append('4_time')

        elif option2 == '2':
            snr_input = float(input("Total SNR: "))
            no = float(input("No. of exposures: "))
            snr_input = snr_input / np.sqrt(no)
            options.append('4_snr')
    
    elif layer0 == '5':
        if '1' in options and '2' in options and '3' in options and ('4_time' in options or '4_snr' in options):
            options.append('5')
            efficiency = efficiencies[filter_band]
            photon_flux = mag_to_photon_flux(syst, mag, filter_band)
            fwhm_tot = fwhm(ccd_seeing, filter_band, turbulance, airmass, bin)
            sigma = fwhm_tot / 2.355
            
            if obj_type == 'p':
                if '4_time' in options:
                    signal = signal_point_imaging(photon_flux, time, efficiency)
                    npix = 2 * fwhm_tot / pix_scale
                    solid_angle = math.pi * fwhm_tot ** 2
                    nsky = calculate_nsky(pwv, fli, airmass, solid_angle, filter_band)

                    snrr = snr(signal, nsky, bin, npix, ron, dark, time)
                    saturation_time = calculate_saturation_time_point(mag, filter_band, syst, efficiency, sigma, nsky, solid_angle)
                    saturation_snr = snr(signal_point_imaging(photon_flux, saturation_time, efficiency), nsky, bin, npix, ron, dark, saturation_time)
                    saturation_limit = calculate_saturation_mag_point(time, filter_band, 'AB', efficiency, sigma, nsky, solid_angle)
                    saturation_mag = calculate_saturation_mag_point(time, filter_band, syst, efficiency, sigma, nsky, solid_angle)
                    if saturation_time < no * time:
                        times = np.linspace(0, 3 * no * time, 1000000)
                    
                    else:
                        times = np.linspace(0, 3 * saturation_time, 1000000)
                    
                    snr_over_time = snr(signal_point_imaging(photon_flux, times, efficiency), nsky, bin, npix, ron, dark, times)
                    print(f"Number of exposures: {no}")
                    print(f"Total time: {no * time}")
                    print(f"Apreture: {solid_angle} arcsec^2")
                    print(f"Number of detector pixels in aperture: {solid_angle / pix_scale ** 2}")
                    print("-" * 70)
                    print(f"Target Signal: {no * signal} e-")
                    print(f"Sky signal: {no * nsky * time} e-")
                    print(f"Detector dark current: {dark} e-/s/pixel")
                    print(f"Detector RON: {ron} e-/pixel")
                    print(f"Total SNR: {no * snrr}")
                    print(f"Single image SNR: {snrr}")
                    print("-" * 70)
                    print(f"Saturation magnitude for given time (In {syst} system): {saturation_mag} mag")
                    print(f"Satruration time: {saturation_time} s")
                    print(f"Saturation SNR: {saturation_snr}")
                    if time > saturation_time:
                        print("----SATURATED----")

                    plt.figure()
                    plt.plot(times, snr_over_time, label='SNR vs Time')
                    plt.axvline(x=saturation_time, color='r', linestyle='--', label=f'Saturation Time: {saturation_time:.2f} s')
                    plt.axhline(y=saturation_snr, color='g', linestyle='--', label=f'Saturation SNR: {saturation_snr:.2f}')
                    plt.scatter(time, snrr, color='b', zorder=5, label=f'Single Time: {time:.2f} s, Single SNR: {snrr:.2f}')
                    plt.scatter(no * time, np.sqrt(no) * snrr, color='y', zorder=5, label=f'Total Time: {no * time:.2f} s, Total SNR: {np.sqrt(no) * snrr:.2f}')
                    plt.xlabel('Exposure Time (s)')
                    plt.ylabel('Signal-to-Noise Ratio (SNR)')
                    plt.title('SNR as a function of Exposure Time')
                    plt.grid(True)
                    plt.legend()
                    plt.show()
                    

                elif '4_snr' in options:
                    npix = 2 * fwhm_tot / pix_scale
                    solid_angle = math.pi * fwhm_tot ** 2
                    nsky = calculate_nsky(pwv, fli, airmass, solid_angle, filter_band)
                    
                    # Calculate the required exposure time to reach the desired SNR
                    time_for_given_snr = time_for_snr_point(snr_input, photon_flux, efficiency, nsky, bin, npix, ron, dark, surface)
                    saturation_time = calculate_saturation_time_point(mag, filter_band, syst, efficiency, sigma, nsky, solid_angle)
                    saturation_snr = snr(signal_point_imaging(photon_flux, saturation_time, efficiency), nsky, bin, npix, ron, dark, saturation_time)
                    saturation_limit = calculate_saturation_mag_point(time_for_given_snr, filter_band, 'AB', efficiency, sigma, nsky, solid_angle)
                    saturation_mag = calculate_saturation_mag_point(time_for_given_snr, filter_band, syst, efficiency, sigma, nsky, solid_angle)
                    if saturation_time < no * time_for_given_snr:
                        times = np.linspace(0, 3 * no * time_for_given_snr, 1000000)
                    else:
                        times = np.linspace(0, 3 * saturation_time, 1000000)
                    
                    snr_over_time = snr(signal_point_imaging(photon_flux, times, efficiency), nsky, bin, npix, ron, dark, times)
                        
                    print(f"Number of exposures: {no}")
                    print(f"Total time: {no * time_for_given_snr}")
                    print(f"Aperture: {solid_angle} arcsec^2")
                    print(f"Number of detector pixels in aperture: {solid_angle / pix_scale ** 2}")
                    print("-" * 70)
                    print(f"Target Signal: {signal_point_imaging(photon_flux, time_for_given_snr, efficiency) * no} e-")
                    print(f"Sky signal: {nsky * time_for_given_snr * no} e-")
                    print(f"Detector dark current: {dark} e-/s/pixel")
                    print(f"Detector RON: {ron} e-/pixel")
                    print(f"Total SNR: {snr_input * no}")
                    print(f"Single image SNR: {snr_input}")
                    print("-" * 70)
                    print(f"Saturation magnitude for given time (In {syst} system): {saturation_mag} mag")
                    print(f"Satruration time: {saturation_time} s")
                    print(f"Saturation SNR: {saturation_snr}")
                    if time_for_given_snr > saturation_time:
                        print("----SATURATED----")

                    plt.figure()
                    plt.plot(times, snr_over_time, label='SNR vs Time')
                    plt.axvline(x=saturation_time, color='r', linestyle='--', label=f'Saturation Time: {saturation_time:.2f} s')
                    plt.axhline(y=saturation_snr, color='g', linestyle='--', label=f'Saturation SNR: {saturation_snr:.2f}')
                    plt.scatter(time_for_given_snr, snr_input, color='b', zorder=5, label=f'Single Time: {time_for_given_snr:.2f} s, Single SNR: {snr_input:.2f}')
                    plt.scatter(no * time_for_given_snr, np.sqrt(no) * snr_input, color='y', zorder=5, label=f'Total Time: {no * time_for_given_snr:.2f} s, Total SNR: {np.sqrt(no) * snr_input:.2f}')
                    plt.xlabel('Exposure Time (s)')
                    plt.ylabel('Signal-to-Noise Ratio (SNR)')
                    plt.title('SNR as a function of Exposure Time')
                    plt.grid(True)
                    plt.legend()
                    plt.show()
                    

            elif obj_type == 'e':
                if '4_time' in options:
                    signal = signal_ext_imaging(photon_flux, time, efficiency, solid_angle)
                    npix = solid_angle / pix_scale ** 2
                    nsky = calculate_nsky(pwv, fli, airmass, solid_angle, filter_band)
                    snrr = snr(signal, nsky, bin, npix, ron, dark, time)
                    saturation_time = calculate_saturation_time_extended(min_mag, filter_band, syst, nsky, solid_angle, max_count, pix_scale)
                    saturation_snr = snr(signal_ext_imaging(photon_flux, saturation_time, efficiency, solid_angle), nsky, bin, npix, ron, dark, saturation_time)
                    saturation_limit = calculate_saturation_mag_point(time, filter_band, 'AB', efficiency, sigma, nsky, solid_angle)
                    saturation_limit = calculate_saturation_mag_point(time, filter_band, syst, efficiency, sigma, nsky, solid_angle)
                    
                    if saturation_time < no * time:
                        times = np.linspace(0, 3 * no * time, 1000000)
                    else:
                        times = np.linspace(0, 3 * saturation_time, 1000000)
                    
                    snr_over_time = snr(signal_ext_imaging(photon_flux, times, efficiency, solid_angle), nsky, bin, npix, ron, dark, times)
                    print(f"Number of exposures: {no}")
                    print(f"Total time: {no * time}")
                    print(f"Aperture: {solid_angle} arcsec^2")
                    print(f"Number of detector pixels in aperture: {solid_angle / pix_scale ** 2}")
                    print("-" * 70)
                    print(f"Target Signal: {no * signal} e-")
                    print(f"Sky signal: {no * nsky * time} e-")
                    print(f"Detector dark current: {dark} e-/s/pixel")
                    print(f"Detector RON: {ron} e-/pixel")
                    print(f"Total SNR: {np.sqrt(no) * snrr}")
                    print(f"Single image SNR: {snrr}")
                    print("-" * 70)
                    print(f"Saturation magnitude for given time (In {syst} system): {saturation_mag} mag/arcsec^2")
                    print(f"Satruration time: {saturation_time} s")
                    print(f"Saturation SNR: {saturation_snr}")
                    if time > saturation_time:
                        print("----SATURATED----")

                    plt.figure()
                    plt.plot(times, snr_over_time, label='SNR vs Time')
                    plt.axvline(x=saturation_time, color='r', linestyle='--', label=f'Saturation Time: {saturation_time:.2f} s')
                    plt.axhline(y=saturation_snr, color='g', linestyle='--', label=f'Saturation SNR: {saturation_snr:.2f}')
                    plt.scatter(time, snrr, color='b', zorder=5, label=f'Single Time: {time:.2f} s, Single SNR: {snrr:.2f}')
                    plt.scatter(no * time, np.sqrt(no) * snrr, color='y', zorder=5, label=f'Total Time: {no * time:.2f} s, Total SNR: {np.sqrt(no) * snrr:.2f}')
                    plt.xlabel('Exposure Time (s)')
                    plt.ylabel('Signal-to-Noise Ratio (SNR)')
                    plt.title('SNR as a function of Exposure Time')
                    plt.grid(True)
                    plt.legend()
                    plt.show()
                    

                elif '4_snr' in options:
                    npix = solid_angle / pix_scale ** 2
                    nsky = calculate_nsky(pwv, fli, airmass, solid_angle, filter_band)
                    
                    # Calculate the required exposure time to reach the desired SNR
                    time_for_given_snr = time_for_snr_ext(snr_input, photon_flux, efficiency, solid_angle, nsky, bin, npix, ron, dark, surface)
                    saturation_time = calculate_saturation_time_extended(min_mag, filter_band, syst, nsky, solid_angle, max_count, pix_scale)
                    saturation_snr = snr(signal_ext_imaging(photon_flux, saturation_time, efficiency, solid_angle), nsky, bin, npix, ron, dark, saturation_time)
                    saturation_limit = calculate_saturation_mag_point(time_for_given_snr, filter_band, 'AB', efficiency, sigma, nsky, solid_angle)
                    saturation_limit = calculate_saturation_mag_point(time_for_given_snr, filter_band, syst, efficiency, sigma, nsky, solid_angle)
                    
                    if saturation_time < no * time_for_given_snr:
                        times = np.linspace(0, 3 * no * time_for_given_snr, 1000000)
                    else:
                        times = np.linspace(0, 3 * saturation_time, 1000000)
                    
                    snr_over_time = snr(signal_ext_imaging(photon_flux, times, efficiency, solid_angle), nsky, bin, npix, ron, dark, times)
                        
                    print(f"Number of exposures: {no}")
                    print(f"Total time: {no * time_for_given_snr}")
                    print(f"Aperture: {solid_angle} arcsec^2")
                    print(f"Number of detector pixels in aperture: {npix}")
                    print("-" * 70)
                    print(f"Target Signal: {signal_ext_imaging(photon_flux, time_for_given_snr, efficiency, solid_angle) * no} e-")
                    print(f"Sky signal: {nsky * time_for_given_snr * no} e-")
                    print(f"Detector dark current: {dark} e-/s/pixel")
                    print(f"Detector RON: {ron} e-/pixel")
                    print(f"Total SNR: {snr_input * np.sqrt(no)}")
                    print(f"Single image SNR: {snr_input}")
                    print("-" * 70)
                    print(f"Saturation magnitude for given time (In {syst} system): {saturation_mag} mag/arcsec^2")
                    print(f"Satruration time: {saturation_time} s")
                    print(f"Saturation SNR: {saturation_snr}")
                    if time_for_given_snr > saturation_time:
                        print("----SATURATED----")

                    plt.figure()
                    plt.plot(times, snr_over_time, label='SNR vs Time')
                    plt.axvline(x=saturation_time, color='r', linestyle='--', label=f'Saturation Time: {saturation_time:.2f} s')
                    plt.axhline(y=saturation_snr, color='g', linestyle='--', label=f'Saturation SNR: {saturation_snr:.2f}')
                    plt.scatter(time_for_given_snr, snr_input, color='b', zorder=5, label=f'Single Time: {time_for_given_snr:.2f} s, Single SNR: {snr_input:.2f}')
                    plt.scatter(no * time_for_given_snr, np.sqrt(no) * snr_input, color='y', zorder=5, label=f'Total Time: {no * time_for_given_snr:.2f} s, Total SNR: {np.sqrt(no) * snr_input:.2f}')
                    plt.xlabel('Exposure Time (s)')
                    plt.ylabel('Signal-to-Noise Ratio (SNR)')
                    plt.title('SNR as a function of Exposure Time')
                    plt.grid(True)
                    plt.legend()
                    plt.show()

    elif layer0 == '6':
        if '1' in options and '2' in options and '3' and '5' in options and ('4_time' in options or '4_snr' in options):
            saturation_chart(ra, dec, filter_band, saturation_limit, pos_ang)
    
    elif layer0 == '7':
        break
        
    else:
        print("Please complete the previous steps (1 to 4) first.")
