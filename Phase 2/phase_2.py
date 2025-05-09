# Written by Bardia Alian
'''
Install these libraries
numpy
matplotlib
csv
shutil
astroquery
astropy
requests
PIL
mplcursors
io
cv2
scipy
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import shutil
import finding_chart_ver2 as fc
from copy import deepcopy
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import snr_to_exp as se

# Define the OB class which contains all the observation block details
class OB:
    """
    A class to represent an Observation Block (OB).
    
    Attributes:
        obs_description (OB.Obs_Description): Details of the observation.
        target (OB.Target): Target information.
        cons_set (OB.Cons_Set): Constraints for the observation.
        time_intervals (OB.Time_Intervals): Preferred time intervals for observation.
        ephemeris (OB.Ephemeris): Ephemeris details for moving objects.
        obsprep (OB.Obsprep): Observation preparation details.
        finding_chart (OB.Finding_Chart): Finding chart information.
    """

    # Nested class to describe the observation details
    class Obs_Description:
        """
        A class to represent the details of an observation.

        Attributes:
            obs_name (str): Name of the observation.
            comments (list): List of comments for the observation.
            sci_temp_exp (OB.Obs_Description.Sci_Temp_Exp): Science template based on exposure time.
            sci_temp_sig_noise (OB.Obs_Description.Sci_Temp_Sig_Noise): Science template based on signal-to-noise ratio.
            fin_temp (OB.Obs_Description.Fin_Temp): Finding chart template.
            exp_time (float): Calculated exposure time.
            filt (str): Filter used in the observation.
            no_exp (int): Number of exposures.
            tar_mag (float): Target magnitude.
            binn (str): Binning used (1x1 or 2x2).
            typee (str): Object type (Point source or Extended).
            wavelength (str): Wavelength corresponding to the filter.
        """

        def __init__(self):
            # Initialize observation description attributes
            self.obs_name = ""  # Name of the observation
            self.comments = []  # List to store comments
            self.sci_temp_exp = None  # Science template based on exposure time
            self.sci_temp_sig_noise = None  # Science template based on signal to noise ratio
            self.fin_temp = None  # Finding chart template

            # Initialize exposure time, filter, number of exposures, and target magnitude
            self.exp_time = self.calculate_exp_time()  # Calculate exposure time
            self.filt = self.calculate_filter()  # Determine filter
            self.no_exp = self.calculate_no_exp()  # Calculate number of exposures
            self.tar_mag = self.calculate_tar_mag()  # Calculate target magnitude
            self.binn = self.calculate_bin()  # Calculate binning
            self.typee = self.calculate_type()  # Determine object type

            # Define wavelength based on filter
            self.wavelength = {
                'u': '354.3',
                'g': '477.0',
                'r': '623.1',
                'i': '762.5',
                'z': '913.4'
            }.get(self.filt, '')  # Get wavelength corresponding to filter

        # Nested class for the science template based on exposure time
        class Sci_Temp_Exp:
            """
            A class to represent the science template based on exposure time.
            
            Attributes:
                tar_mag (str): Target magnitude.
                exp_time (str): Exposure time in seconds.
                no_exp (str): Number of exposures.
                obs_filter (str): Observation filter.
                binn (str): Binning used (1x1 or 2x2).
                typee (str): Object type (Point source or Extended).
            """

            def __init__(self):
                self.tar_mag = ""  # Target magnitude
                self.exp_time = ""  # Exposure time in seconds
                self.no_exp = ""  # Number of exposures
                self.obs_filter = ""  # Observation filter
                self.binn = ""  # Binning (1x1 or 2x2)
                self.typee = ""  # Object type (Point source or Extended)

        # Nested class for the science template based on signal to noise ratio
        class Sci_Temp_Sig_Noise:
            """
            A class to represent the science template based on signal-to-noise ratio.
            
            Attributes:
                tar_mag (str): Target magnitude.
                sig_noise (str): Preferred signal-to-noise ratio.
                no_exp (str): Number of exposures.
                obs_filter (str): Observation filter.
                binn (str): Binning used (1x1 or 2x2).
                typee (str): Object type (Point source or Extended).
            """

            def __init__(self):
                self.tar_mag = ""  # Target magnitude
                self.sig_noise = ""  # Preferred signal to noise ratio
                self.no_exp = ""  # Number of exposures
                self.obs_filter = ""  # Observation filter
                self.binn = ""  # Binning (1x1 or 2x2)
                self.typee = ""  # Object type (Point source or Extended)
                self.exp_time = 0
                self.calculation  = False # Check if it can use calc_exp_time method because It also needs Constrants Set
                
            def calc_exp_time(self, cons_set, solid_angle):
                """
                Calculate the exposure time based on the signal-to-noise ratio.
                
                Args:
                    cons_set (OB.Cons_Set): The constraint set that provides necessary parameters.
                    surface (float): The effective surface area of the telescope.

                Returns:
                    float: The calculated exposure time, or None if cons_set is not available.
                """
                # Constants
                h = 6.62607015 * 10 ** (-34)  # Planck's constant
                c = 299792458  # Speed of light
                k = 1.380649 * 10 ** (-23)  # Boltzmann constant
                pix_scale = 0.25  # Pixel scale in arcseconds per pixel
                pix_size = 1.5 * 10 ** (-5)  # Pixel size in meters

                # Telescope data
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
                
                # Calculate the exposure time if cons_set is available
                fwhm = cons_set.img_quality  # Image quality is FWHM
                pwv = cons_set.pwv
                airmass = cons_set.airmass
                filt = self.obs_filter
                binn = int(self.binn.split('x')[0])  # Extract binning factor (e.g., '1x1' -> 1)
                lunar_ill = cons_set.lunar_ill

                photon_flux = se.mag_to_photon_flux('Vega', self.tar_mag, filt)
                efficiency = efficiencies[filt]
                nsky = se.calculate_nsky(pwv, lunar_ill, airmass, fwhm, filt)
                npix = 2 * fwhm / pix_scale
                
                if self.typee == 'Point':
                    self.exp_time = se.time_for_snr_point(self.sig_noise / np.sqrt(self.no_exp), photon_flux, efficiency, nsky, binn, npix, ron, dark, surface)
                    print(f"Single exposure time: {self.exp_time} s")
                    print(f"Total exposure time: {self.no_exp * self.exp_time} s")
                elif self.typee == 'Extended':
                    solid_angle = npix * (pix_size / pix_scale) ** 2
                    self.exp_time = se.time_for_snr_ext(np.sqrt(self.no_exp), photon_flux, efficiency, solid_angle, nsky, binn, npix, ron, dark, surface)
                    print(f"Single exposure time: {self.exp_time} s")
                    print(f"Total exposure time: {self.no_exp * self.exp_time} s")
                else:
                    raise ValueError("Invalid object type. Must be 'Point' or 'Extended'.")

                return self.exp_time

        # Nested class for the finding chart template
        class Fin_Temp:
            """
            A class to represent the finding chart template.
            
            Attributes:
                obs_filter (str): Observation filter.
                scale_length (str): Scale length in arcseconds.
                fov (str): Field of view in arcminutes.
                wavelength (str): Wavelength corresponding to the filter.
            """

            def __init__(self):
                self.obs_filter = ""  # Observation filter
                self.scale_length = ""  # Scale length in arcseconds
                self.fov = ""  # Field of view in arcminutes

                # Define wavelength based on observation filter
                self.wavelength = {
                    'u': '354.3',
                    'g': '477.0',
                    'r': '623.1',
                    'i': '762.5',
                    'z': '913.4'
                }.get(self.obs_filter, '')  # Get wavelength based on filter

        # Method to calculate the exposure time
        def calculate_exp_time(self):
            """Calculate the exposure time based on the selected science template."""
            if self.sci_temp_exp is None:
                if self.sci_temp_sig_noise:
                    exp_time = float(self.sci_temp_sig_noise.exp_time)
                else:
                    exp_time = 0.0  # Default to 0.0 if not set
            else:
                exp_time = float(self.sci_temp_exp.exp_time or 0.0)  # Default to 0.0 if not set
            return exp_time

        # Method to calculate the filter
        def calculate_filter(self):
            """Determine the filter to be used based on the selected science template."""
            if self.sci_temp_exp is None:
                if self.sci_temp_sig_noise:
                    filt = self.sci_temp_sig_noise.obs_filter
                else:
                    filt = None
            else:
                filt = self.sci_temp_exp.obs_filter
            return filt

        # Method to calculate the number of exposures
        def calculate_no_exp(self):
            """Calculate the number of exposures based on the selected science template."""
            if self.sci_temp_exp is None:
                if self.sci_temp_sig_noise:
                    no_exp = int(self.sci_temp_sig_noise.no_exp)
                else:
                    no_exp = 0  # Default to 0 if not set
            else:
                no_exp = int(self.sci_temp_exp.no_exp or 0)  # Default to 0 if not set
            return no_exp

        # Method to calculate the binning
        def calculate_bin(self):
            """Calculate the binning based on the selected science template."""
            if self.sci_temp_exp is None:
                if self.sci_temp_sig_noise:
                    binn = self.sci_temp_sig_noise.binn
                else:
                    binn = None
            else:
                binn = self.sci_temp_exp.binn
            return binn

        # Method to determine the object type
        def calculate_type(self):
            """Determine the object type based on the selected science template."""
            if self.sci_temp_exp is None:
                if self.sci_temp_sig_noise:
                    typee = self.sci_temp_sig_noise.typee
                else:
                    typee = None
            else:
                typee = self.sci_temp_exp.typee
            return typee

        # Method to calculate the target magnitude
        def calculate_tar_mag(self):
            """Calculate the target magnitude based on the selected science template."""
            if self.sci_temp_exp is None:
                if self.sci_temp_sig_noise:
                    tar_mag = float(self.sci_temp_sig_noise.tar_mag)
                else:
                    tar_mag = None
            else:
                tar_mag = float(self.sci_temp_exp.tar_mag)
            return tar_mag

    # Nested class for the target details
    class Target:
        """
        A class to represent the target details.
        
        Attributes:
            tar_name (str): Name of the target.
            dec (float): Declination in degrees.
            ra (float): Right ascension in degrees.
            equinox (str): Equinox of the coordinates.
            epoch (str): Epoch of the coordinates.
            pmra (float): Proper motion in RA (arcsec/year).
            pmd (float): Proper motion in Dec (arcsec/year).
            dra (float): Differential RA (arcsec/year).
            dd (float): Differential Dec (arcsec/year).
        """

        def __init__(self):
            self.tar_name = ""  # Target name
            self.dec = 0.0  # Declination in degrees
            self.ra = 0.0  # Right ascension in degrees
            self.equinox = ""  # Equinox of coordinates
            self.epoch = ""  # Epoch of coordinates
            self.pmra = 0.0  # Proper motion in RA (arcsec/year)
            self.pmd = 0.0  # Proper motion in Dec (arcsec/year)
            self.dra = 0.0  # Differential RA (arcsec/year)
            self.dd = 0.0  # Differential Dec (arcsec/year)

    # Nested class for the constraint set
    class Cons_Set:
        """
        A class to represent the constraints for the observation.
        
        Attributes:
            cons_name (str): Name of the constraint set.
            airmass (float): Airmass value.
            sky_trans (str): Sky transparency conditions.
            lunar_ill (float): Lunar illumination (fraction between 0 and 1).
            img_quality (float): Allowable maximum image quality (FWHM).
            mad (float): Moon angular distance in degrees.
            twilight (float): Twilight time in minutes.
            pwv (float): Precipitable water vapor in millimeters.
        """

        def __init__(self):
            self.cons_name = ""  # Constraint set name
            self.airmass = 0.0  # Airmass value
            self.sky_trans = ""  # Sky transparency
            self.lunar_ill = 0.0  # Lunar illumination
            self.img_quality = 0.0  # Allowable maximum image quality (FWHM)
            self.mad = 0.0  # Moon angular distance (degrees)
            self.twilight = 0.0  # Twilight time (minutes)
            self.pwv = 0.0  # Precipitable water vapor (mm)
            self.calculation  = False # Check if it can use calc_exp_time method in Science template exposure time

    # Nested class for the time intervals
    class Time_Intervals:
        """
        A class to represent the preferred time intervals for observation.
        
        Attributes:
            beginning (str): Start of the preferred time interval (d/m/y).
            end (str): End of the preferred time interval (d/m/y).
        """

        def __init__(self):
            self.beginning = ""  # Beginning of preferred time interval (d/m/y)
            self.end = ""  # End of preferred time interval (d/m/y)

    # Nested class for the ephemeris
    class Ephemeris:
        """
        A class to represent the ephemeris details for the observation.
        
        Attributes:
            ephemeris_ava (bool): Indicates if an ephemeris file is available.
        """

        def __init__(self):
            self.ephemeris_ava = False  # Availability of ephemeris file

    # Nested class for the observation preparation
    class Obsprep:
        """
        A class to represent the observation preparation details.
        
        Attributes:
            pointing (OB.Obsprep.Pointing): Pointing details.
            blind_offset (OB.Obsprep.Blind_Offset): Blind offset details.
            obs_offset (OB.Obsprep.Obs_Offset): Observing offset details.
            guide_star (OB.Obsprep.Guide_Star): Guide star details.
        """

        def __init__(self):
            self.pointing = None  # Pointing details
            self.blind_offset = None  # Blind offset details
            self.obs_offset = None  # Observing offset details
            self.guide_star = None  # Guide star details

        # Nested class for the pointing details
        class Pointing:
            """
            A class to represent the pointing details.
            
            Attributes:
                diff_dec (float): Differential declination.
                diff_ra (float): Differential right ascension.
                pos_angle (float): Position angle.
            """

            def __init__(self):
                self.diff_dec = 0.0  # Differential declination
                self.diff_ra = 0.0  # Differential right ascension
                self.pos_angle = 0.0  # Position angle

        # Nested class for the blind offset details
        class Blind_Offset:
            """
            A class to represent the blind offset details.
            
            Attributes:
                acq_dec (list): Acquisition declinations.
                acq_ra (list): Acquisition right ascensions.
                acq_mag (list): Acquisition magnitudes.
            """

            def __init__(self):
                self.acq_dec = []  # Acquisition declinations
                self.acq_ra = []  # Acquisition right ascensions
                self.acq_mag = []  # Acquisition magnitudes

        # Nested class for the observing offset details
        class Obs_Offset:
            """
            A class to represent the observing offset details.
            
            Attributes:
                off_dec (list): Offset declinations.
                off_ra (list): Offset right ascensions.
                obs_off (list): Observing offsets.
            """

            def __init__(self):
                self.off_dec = []  # Offset declinations
                self.off_ra = []  # Offset right ascensions
                self.obs_off = []  # Observing offsets

        # Nested class for the guide star details
        class Guide_Star:
            """
            A class to represent the guide star details.
            
            Attributes:
                guide_dec (list): Guide star declinations.
                guide_ra (list): Guide star right ascensions.
                guide_mag (list): Guide star magnitudes.
            """

            def __init__(self):
                self.guide_dec = []  # Guide star declinations
                self.guide_ra = []  # Guide star right ascensions
                self.guide_mag = []  # Guide star magnitudes

    # Nested class for the finding chart details
    class Finding_Chart:
        """
        A class to represent the finding chart details.
        
        Attributes:
            fin_ava (bool): Indicates if a finding chart is available.
        """

        def __init__(self):
            self.fin_ava = False  # Availability of finding chart

    # Initialize the OB class with its nested classes
    def __init__(self):
        self.obs_description = self.Obs_Description()  # Initialize observation description
        self.target = self.Target()  # Initialize target details
        self.cons_set = self.Cons_Set()  # Initialize constraint set
        self.time_intervals = self.Time_Intervals()  # Initialize time intervals
        self.ephemeris = self.Ephemeris() # Initialize ephemeris
        self.obsprep = self.Obsprep()  # Initialize observation preparation
        self.finding_chart = self.Finding_Chart()  # Initialize finding chart

# Define the CBs class which contains calibration block details
class CBs:
    """
    A class to represent the Calibration Blocks (CBs).
    
    Attributes:
        ob_list (list): List of OB objects.
        filters (list): List of unique filters used in the OBs.
        exp_times (list): List of unique exposure times used in the OBs.
        bias (CBs.Bias): Bias calibration block.
        flat (CBs.Flat): Flat calibration block.
        dark (CBs.Dark): Dark calibration block.
    """

    def __init__(self, ob_list):
        self.ob_list = ob_list  # List of Observation Blocks (OBs)
        self.filters = self.all_filters()  # Get all unique filters used in OBs
        self.exp_times = self.all_exp_times()  # Get all unique exposure times used in OBs
        self.bias = self.Bias(self.filters)  # Initialize Bias calibration block
        self.flat = self.Flat(self.filters)  # Initialize Flat calibration block
        self.dark = self.Dark(self.exp_times)  # Initialize Dark calibration block
    
    # Method to get all unique filters used in the OBs
    def all_filters(self):
        """Return a list of all unique filters used in the OBs."""
        seen = set()  # Set to track seen filters
        all_filters = []  # List to store all filters
        for ob in self.ob_list:
            filt = ob.obs_description.filt  # Get filter from OB
            if filt not in seen:
                seen.add(filt)  # Add filter to seen set
                all_filters.append(filt)  # Append to list
        return all_filters

    # Method to get all unique exposure times used in the OBs
    def all_exp_times(self):
        """Return a list of all unique exposure times used in the OBs."""
        seen = set()  # Set to track seen exposure times
        all_exp_times = []  # List to store all exposure times
        for ob in self.ob_list:
            exp_time = ob.obs_description.exp_time  # Get exposure time from OB
            if exp_time not in seen:
                seen.add(exp_time)  # Add exposure time to seen set
                all_exp_times.append(exp_time)  # Append to list
        return all_exp_times

    # Nested class for Bias calibration block
    class Bias:
        """
        A class to represent the Bias calibration block.
        
        Attributes:
            number (int): Number of bias frames.
            exp_times (None): No exposure times for bias frames.
            obs_filters (None): No filters for bias frames.
        """

        def __init__(self, filters):
            self.number = 5  # Number of bias frames
            self.exp_times = None  # No exposure times for bias frames
            self.obs_filters = None  # No filters for bias frames
    
    # Nested class for Flat calibration block
    class Flat:
        """
        A class to represent the Flat calibration block.
        
        Attributes:
            number (int): Number of flat frames.
            exp_time (int): Exposure time for flat frames (seconds).
            filters (list): Observation filters used in flat frames.
        """

        def __init__(self, filters):
            self.number = 5  # Number of flat frames
            self.exp_time = 3  # Exposure time for flat frames (seconds)
            self.filters = filters  # Observation filters used in flat frames
    
    # Nested class for Dark calibration block
    class Dark:
        """
        A class to represent the Dark calibration block.
        
        Attributes:
            number (int): Number of dark frames.
            exp_times (list): Exposure times for dark frames.
            filters (None): No filters for dark frames.
        """

        def __init__(self, exp_times):
            self.number = 5  # Number of dark frames
            self.exp_times = exp_times  # Exposure times for dark frames
            self.filters = None  # No filters for dark frames

# Function to write OBs and CBs to a CSV file
def write_csv(ob_list, cb, user):
    """
    Write the Observation Blocks (OBs) and Calibration Blocks (CBs) to a CSV file.
    
    Args:
        ob_list (list): List of OB objects.
        cb (CBs): CBs object containing calibration blocks.
        user (str): Username for output file.
    """
    with open(f'outputs/{user}/{user}_observations.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"User: {user}"])
        writer.writerow([])  # Empty row for separation
        writer.writerow([])  # Empty row for separation
        
        y = 1  # OB counter
        for ob in ob_list:
            writer.writerow([f"OB no. {y}"])  # Write OB number
            writer.writerow([
                'Obs Name', 'Comments', 'Exposure Time', 'Filter',
                'Number of Exposures', 'Target Magnitude', 'Binning', 'Object type',
                'Finding Chart Filter', 'Finding Chart Scale Length', 'Finding Chart FOV',
                'Target Name', 'Declination', 'Right Ascension',
                'Equinox', 'Epoch', 'Proper Motion RA',
                'Proper Motion Dec', 'Differential RA',
                'Differential Dec', 'Constraint Name',
                'Airmass', 'Sky Transparency', 'Lunar Illumination',
                'Moon Angular Distance', 'Twilight', 'PWV',
                'Time Interval Beginning', 'Time Interval End',
                'Ephemeris File', 'Pointing Differential Dec',
                'Pointing Differential RA', 'Pointing Position Angle',
                'Blind Offset Acquisition Dec', 'Blind Offset Acquisition RA',
                'Blind Offset Acquisition Mag', 'Obs Offset Dec',
                'Obs Offset RA', 'Obs Offset Observing Offset',
                'Guide Star Dec', 'Guide Star RA', 'Guide Star Mag',
                'Finding Chart File'
            ])
            writer.writerow([
                ob.obs_description.obs_name,  # Observation name
                '; '.join(ob.obs_description.comments),  # Comments joined with ';'
                f"{ob.obs_description.exp_time:.2f}",
                ob.obs_description.filt,
                ob.obs_description.no_exp,
                ob.obs_description.tar_mag,
                ob.obs_description.binn,  # Include binning here
                ob.obs_description.typee,  # Include object type here
                ob.obs_description.fin_temp.obs_filter if ob.obs_description.fin_temp else '',
                ob.obs_description.fin_temp.scale_length if ob.obs_description.fin_temp else '',
                ob.obs_description.fin_temp.fov if ob.obs_description.fin_temp else '',
                ob.target.tar_name,
                ob.target.dec,
                ob.target.ra,
                ob.target.equinox,
                ob.target.epoch,
                ob.target.pmra,
                ob.target.pmd,
                ob.target.dra,
                ob.target.dd,
                ob.cons_set.cons_name,
                ob.cons_set.airmass,
                ob.cons_set.sky_trans,
                ob.cons_set.lunar_ill,
                ob.cons_set.mad,
                ob.cons_set.twilight,
                ob.cons_set.pwv,
                ob.time_intervals.beginning,
                ob.time_intervals.end,
                ob.ephemeris.ephemeris_ava,
                ob.obsprep.pointing.diff_dec if ob.obsprep.pointing else '',
                ob.obsprep.pointing.diff_ra if ob.obsprep.pointing else '',
                ob.obsprep.pointing.pos_angle if ob.obsprep.pointing else '',
                '; '.join(map(str, ob.obsprep.blind_offset.acq_dec)) if ob.obsprep.blind_offset else '',
                '; '.join(map(str, ob.obsprep.blind_offset.acq_ra)) if ob.obsprep.blind_offset else '',
                '; '.join(map(str, ob.obsprep.blind_offset.acq_mag)) if ob.obsprep.blind_offset else '',
                '; '.join(map(str, ob.obsprep.obs_offset.off_dec)) if ob.obsprep.obs_offset else '',
                '; '.join(map(str, ob.obsprep.obs_offset.off_ra)) if ob.obsprep.obs_offset else '',
                '; '.join(map(str, ob.obsprep.obs_offset.obs_off)) if ob.obsprep.obs_offset else '',
                '; '.join(map(str, ob.obsprep.guide_star.guide_dec)) if ob.obsprep.guide_star else '',
                '; '.join(map(str, ob.obsprep.guide_star.guide_ra)) if ob.obsprep.guide_star else '',
                '; '.join(map(str, ob.obsprep.guide_star.guide_mag)) if ob.obsprep.guide_star else '',
                ob.finding_chart.fin_ava
            ])
            
            y += 1
            writer.writerow([])  # Empty row for separation
        z = 1  # CB counter
        # Writing Bias
        writer.writerow([f"CB no. {z}"])  # Write CB number
        # Writing the header for CB
        writer.writerow([
            'Calibration Block Type', 'Number', 'Filter', 'Exposure Time'
        ])
            
        writer.writerow(['Bias', cb.bias.number, 'None', 'None'])
        writer.writerow([])  # Empty row for separation
            
        # Writing Flats
        for filter in cb.flat.filters:
            writer.writerow([f"CB no. {z}"])  # Write CB number
            # Writing the header for CB
            writer.writerow([
                'Calibration Block Type', 'Number', 'Filter', 'Exposure Time'
            ])

            writer.writerow(['Flat', cb.flat.number, filter, cb.flat.exp_time])
            z += 1
            writer.writerow([])  # Empty row for separation

        # Writing Darks
        for exp_time in cb.dark.exp_times:
            writer.writerow([f"CB no. {z}"])  # Write CB number
            # Writing the header for CB
            writer.writerow([
               'Calibration Block Type', 'Number', 'Filter', 'Exposure Time'
            ])
        
            writer.writerow(['Dark', cb.dark.number, 'None', exp_time])
            z += 1
            writer.writerow([])  # Empty row for separation

# Helper function to safely get a float input
def get_float_input(prompt):
    """
    Prompt the user for a float input and ensure the input is valid.
    
    Args:
        prompt (str): The input prompt message.
        
    Returns:
        float: The valid float input.
    """
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print(f"Invalid input for \"{prompt}\". Please enter a numeric value.")

# Function to update and plot the finding chart
def update_plot(user, ob, new_ra, new_dec, angle):
    """
    Update and plot the finding chart.
    
    Args:
        user (str): Username for output file.
        ob (OB): The Observation Block object.
        new_ra (float): Updated right ascension in degrees.
        new_dec (float): Updated declination in degrees.
        angle (float): Position angle for the observation.
    """
    plt.close('all')  # Close any existing plots
    # Call the main function of finding_chart_ver2 to generate the plot
    fc.main(user, user, ob.obs_description.obs_name, ob.target.tar_name, new_ra, new_dec,
            ob.obs_description.fin_temp.wavelength, ob.obs_description.fin_temp.scale_length,
            ob.obs_description.fin_temp.fov, ob.obsprep.obs_offset.off_ra, ob.obsprep.obs_offset.off_dec, angle)

# Function to check for errors before closing an observation block
def closing_error(ob):
    """
    Check for errors before closing an observation block.
    
    Args:
        ob (OB): The Observation Block object.
    
    Returns:
        bool: True if no errors, False otherwise.
    """
    checker = True

    def error_individually(section, section_name):
        """Check if a given section is filled and return error status."""
        if section:
            return True
        else:
            print(f"You haven't filled {section_name} field.")
            return False

    dome_time = 60.0  # Time taken to rotate the dome (seconds). Dummy data please change it to the accurate data
    tel_time = 60.0  # Time taken to move the telescope (seconds) Dummy data please change it to the accurate data
    # Safely handle potential None values by converting to float if possible, otherwise default to 0
    no_exp = float(ob.obs_description.calculate_no_exp() or 0)
    exp_time = float(ob.obs_description.calculate_exp_time() or 0)

    # Calculate total observation time
    total_time = 30.0 * no_exp + dome_time + tel_time + no_exp * exp_time # 30 seconds is Dummy data please change it to the accurate data.

    # Check if any section is not properly filled
    section_checking = [
        error_individually(
            ob.obs_description.obs_name and
            (ob.obs_description.sci_temp_exp or ob.obs_description.sci_temp_sig_noise),
            "Obs. Description"
        ),
        error_individually(ob.target.tar_name, "Target"),  # Check target name instead of the object
        error_individually(ob.cons_set.cons_name, "Constraint Set"),  # Check constraint set name instead of the object
        error_individually(ob.time_intervals.beginning, "Time Intervals"),  # Check beginning interval
        error_individually(ob.obsprep.pointing, "Obsprep"),  # Check pointing instead of the object
        error_individually(ob.finding_chart.fin_ava, "Finding Chart")  # Check finding chart availability
    ]

    if total_time > 3600:
        # Check if the total time exceeds 1 hour
        print("Your observing block time is bigger than 1 hour")
        checker = False  # Indicate an issue

    # Only prompt the user if there is an issue with sections or total time
    if not all(section_checking) and checker:
        while True:  # Loop to prompt user decision
            layer4 = input('''0. Go out anyway
1. Go to the OB menu
''')
            if layer4 == '0':
                return True  # Indicate that the user chose to go out anyway
            elif layer4 == '1':
                return False  # Indicate that the user chose to return to the OB menu

    return checker  # Return true if no errors and total time is within limit

# Helper function to convert proper motion from mas/yr to arcsec/yr
def convert_to_arcsecyr(value):
    """
    Convert proper motion from milliarcseconds per year (mas/yr) to arcseconds per year (arcsec/yr).
    
    Args:
        value (str or None): Proper motion value in mas/yr.
        
    Returns:
        float: Proper motion in arcsec/yr.
    """
    if isinstance(value, (str, type(None))) and value == '--':
        return 0.0
    try:
        return float(value) / 1000.0
    except (ValueError, TypeError):
        return 0.0

# Function to retrieve object information from Simbad
def get_object_info(object_name):
    """
    Retrieve object information from Simbad.
    
    Args:
        object_name (str): Name of the object.
        
    Returns:
        dict or str: A dictionary containing object details, or an error message if not found.
    """
    customSimbad = Simbad()
    customSimbad.TIMEOUT = 600  # Set timeout for query
    customSimbad.add_votable_fields('pmra', 'pmdec')  # Add proper motion fields

    try:
        result = customSimbad.query_object(object_name)
        
        if result is None:
            return f"Object '{object_name}' not found in SIMBAD."

        ra_str = result['ra'][0]  # RA in hourangle format
        dec_str = result['dec'][0]  # Dec in degree format
        pmra_masyr = result['pmra'][0]  # Proper motion RA in mas/yr
        pmdec_masyr = result['pmdec'][0]  # Proper motion Dec in mas/yr
        
        # Convert RA and Dec to degrees using astropy
        sky_coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
        ra_deg = sky_coord.ra.deg  # Convert RA to degrees
        dec_deg = sky_coord.dec.deg  # Convert Dec to degrees

        # Handle '--' or missing values explicitly
        pmra_arcsecyr = convert_to_arcsecyr(pmra_masyr)
        pmdec_arcsecyr = convert_to_arcsecyr(pmdec_masyr)

        # Assume J2000.0 epoch and equinox
        epoch = 2000
        equinox = 2000.0

        return {
            "Object Name": object_name,
            "Coordinates (RA, DEC) in degrees": (ra_deg, dec_deg),
            "Proper Motion (PMRA, PMDEC) in arcsec/yr": (pmra_arcsecyr, pmdec_arcsecyr),
            "Epoch": epoch,
            "Equinox": equinox
        }

    except Exception as e:
        print(str(e))
        return False

# Additional helper functions for coordinate conversions
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

def degrees_to_ra(deg):
    """
    Convert degrees to Right Ascension in hh:mm:ss format.
    
    Args:
        deg (float): Right Ascension in degrees.
        
    Returns:
        str: Right Ascension in hh:mm:ss format.
    """
    hours = int(deg / 15)
    minutes = int((deg / 15 - hours) * 60)
    seconds = ((deg / 15 - hours) * 60 - minutes) * 60
    return f"{hours:02d}:{minutes:02d}:{seconds:.2f}"

def degrees_to_dec(deg):
    """
    Convert degrees to Declination in dd:mm:ss format.
    
    Args:
        deg (float): Declination in degrees.
        
    Returns:
        str: Declination in dd:mm:ss format.
    """
    sign = '-' if deg < 0 else '+'
    deg = abs(deg)
    degrees = int(deg)
    arcminutes = int((deg - degrees) * 60)
    arcseconds = ((deg - degrees) * 60 - arcminutes) * 60
    return f"{sign}{degrees:02d}:{arcminutes:02d}:{arcseconds:.2f}"

# Main menu function to interact with the user and create OBs and CBs
def main_menu():
    """
    Main menu function to interact with the user and create Observation Blocks (OBs) and Calibration Blocks (CBs).
    """
    ob_list = []  # List to store observation blocks
    while True:
        brking = False
        user = input("Please give your username: ")  # Prompt user for username
        user_dir = os.path.join('outputs', user)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            brking = True
        else:
            print("User already exists. Give another user")
        
        if brking:
            break

    while True:
        # Main menu options
        layer0 = input('''
1. Create a new OB
2. See OBs
0. End
''')
        if layer0 == '1':
            ob = OB()  # Create a new observation block
            solid_angle = None
            new_dec = None
            new_ra = None
            while True:
                # OB menu options
                layer1 = input('''
1. Obs. Description
2. Target
3. Constraint Set
4. Time Intervals
5. Ephemeris (Optional)
6. ObsPrep
7. Finding Chart
0. Go Back
''')
                if layer1 == '1':
                    ob.obs_description = ob.Obs_Description()  # Initialize observation description
                    option1 = print("Your last Obs. descriptions:")  # Getting last observations names
                    for i in range(len(ob_list)):
                        print(f"{i+1}.", ob_list[i].obs_description.obs_name)

                    while True:
                        option2 = input("""1. Input from your last observations
2. Give a new Obs. Description
""")
                        if option2 == '1' and len(ob_list) != 0:  # Check if you have any last obs or not
                            for i in range(len(ob_list)):
                                print(f"{i+1}.", ob_list[i].obs_description.obs_name)

                            option3 = int(input("Choose your observation: "))
                            while True:
                                if option3 in range(1, len(ob_list) + 1):  # Check if you've chosen a previous observation or not
                                    if ob_list[option3 - 1].obs_description.sci_temp_exp:  # Checking which template your previous ob has
                                        ob.obs_description.sci_temp_exp = deepcopy(ob_list[option3 - 1].obs_description.sci_temp_exp)
                                        while True:
                                            option4 = input("""1. Add a comment (Optional)
2. Quit
""")
                                            if option4 == '1':
                                                ob.obs_description.comments.append(input("Enter your comment: "))  # Getting your new observation comments
                                            elif option4 == '2':
                                                break
                                        break
                                    else:
                                        ob.obs_description.sci_temp_sig_noise = deepcopy(ob_list[option3 - 1].obs_description.sci_temp_sig_noise)
                                        ob.obs_description.sci_temp_sig_noise.exp_time = 0
                                        if ob.obs_description.sci_temp_sig_noise.typee == 'Extended':
                                            solid_angle = get_float_input("Object solid angle (arcsec^2): ")
                                
                                        
                                        if ob.cons_set.calculation and ob.obs_description.sci_temp_sig_noise.calculation:
                                            ob.obs_description.sci_temp_sig_noise.calc_exp_time(ob.cons_set, solid_angle)
                                        while True:
                                            option4 = input("""1. Add a comment (Optional)
2. Quit
""")
                                            if option4 == '1':
                                                ob.obs_description.comments.append(input("Enter your comment: "))  # Getting your new observation comments
                                            elif option4 == '2':
                                                break
                                        break
                        
                                    
                        elif option2 == '1' and len(ob_list) == 0:
                            print("No previous obs")
                            
                        elif option2 == '2':
                            while True:
                                obs_names = [i.obs_description.obs_name for i in ob_list]  # previous obs names
                                ob.obs_description.obs_name = input("Observation Name: ")  # Prompt for observation name
                                
                                if ob.obs_description.obs_name in obs_names:  # Check if the name already exists
                                    print("Name already exists. Give another name")
                                    
                                else:
                                    break

                            while True:
                                # Template creation menu
                                layer2 = input('''
Select one template to create. You should create only 1 science template, 1 finding chart template, and putting comments is optional.
1. Science Template (Exposure Time)
2. Science Template (Signal to Noise)
3. Add a Comment (Optional)
0. Go to OB Menu
''')
                                if layer2 == '1':
                                    # Science template based on exposure time
                                    print("Please ensure that each OB's total exposure time is approximately under 40 minutes (including intervals between shots). After submission, check your email for validation. If invalid, follow the correction instructions within an hour to avoid resubmission.\n")
                                    ob.obs_description.sci_temp_exp = ob.Obs_Description.Sci_Temp_Exp()
                                    ob.obs_description.sci_temp_exp.tar_mag = get_float_input("Target Magnitude (In Vega system): ")  # Input target magnitude
                                    ob.obs_description.sci_temp_exp.exp_time = get_float_input("Exposure Time (sec): ")  # Input exposure time
                                    ob.obs_description.sci_temp_exp.no_exp = int(get_float_input("Number of Exposures: "))  # Input number of exposures
                                    ob.obs_description.sci_temp_exp.obs_filter = input("Observation Filter: ")  # Input observation filter
                                    ob.obs_description.sci_temp_exp.binn = input("Preferred binning (1x1 or 2x2): ")  # Input binning
                                    ob.obs_description.sci_temp_exp.typee = input("Object type (Point or Extended): ")

                                elif layer2 == '2':
                                    # Science template based on signal to noise
                                    print("Please ensure that each OB's total exposure time is approximately under 40 minutes (including intervals between shots). After submission, check your email for validation. If invalid, follow the correction instructions within an hour to avoid resubmission.\n")
                                    ob.obs_description.sci_temp_sig_noise = ob.Obs_Description.Sci_Temp_Sig_Noise()
                                    ob.obs_description.sci_temp_sig_noise.tar_mag = get_float_input("Target Magnitude (In Vega system): ")  # Input target magnitude
                                    ob.obs_description.sci_temp_sig_noise.sig_noise = get_float_input("Preferred Signal to Noise: ")  # Input signal to noise
                                    ob.obs_description.sci_temp_sig_noise.no_exp = int(get_float_input("Number of Exposures: "))  # Input number of exposures
                                    ob.obs_description.sci_temp_sig_noise.obs_filter = input("Observation Filter: ")  # Input observation filter
                                    ob.obs_description.sci_temp_sig_noise.binn = input("Preferred binning (1x1 or 2x2): ")  # Input binning
                                    ob.obs_description.sci_temp_sig_noise.typee = input("Object type (Point or Extended): ")
                                    if ob.obs_description.sci_temp_sig_noise.typee == 'Extended':
                                        solid_angle = get_float_input("Object solid angle (arcsec^2): ")
                                        
                                    ob.obs_description.sci_temp_sig_noise.calculation = True  # Indicates that you've filled obs. description for exposure time claculation from snr
                                    
                                    if ob.cons_set.calculation and ob.obs_description.sci_temp_sig_noise.calculation:  # Check if both sections are filled
                                        ob.obs_description.sci_temp_sig_noise.calc_exp_time(ob.cons_set, solid_angle)

                                elif layer2 == '3':
                                    # Add a comment
                                    comment = input("Enter your comment: ")  # Input comment
                                    ob.obs_description.comments.append(comment)  # Append comment to list

                                elif layer2 == '0':
                                    # Check for template conflicts and completeness
                                    if ob.obs_description.sci_temp_sig_noise or ob.obs_description.sci_temp_exp:
                                        # If either science template is complete
                                        break

                            if ob.obs_description.sci_temp_sig_noise and ob.obs_description.sci_temp_exp:
                                # If both science templates are present
                                layer6 = input('''You have given two Types of science templates. You only can have one of them. Delete:
1. Science Template (Exposure Time)
2. Science Template (Signal to Noise)
''')
                                if layer6 == '1':
                                    ob.obs_description.sci_temp_exp = None  # Remove exposure time template
                                    break
                                elif layer6 == '2':
                                    ob.obs_description.sci_temp_sig_noise = None  # Remove signal to noise template
                                    break
                            
                            elif ob.obs_description.sci_temp_sig_noise:
                                break
                            
                            elif ob.obs_description.sci_temp_exp:
                                break
                            
                            else:
                                # Prompt if templates are incomplete
                                layer7 = input('''You haven't filled Templates completely. Are you sure you want to go to the main menu?
1. Yes
2. No
''')
                                if layer7 == '1':
                                    break
                                elif layer7 == '2':
                                    continue

                        if (option2 == '1' and len(ob_list) != 0) or option2 == '2':
                            break


                elif layer1 == '2':
                    breaking = False
                    # Target details menu
                    ob.target = ob.Target()  # Initialize target details
                    while True:
                        ob.target.tar_name = input("Target Name: ")  # Input target name
                        layer13 = input("""1. Get the data from Simbad
2. Give the data manually
""")
                        if layer13 == '1':
                            while True:
                                tar_info = get_object_info(ob.target.tar_name)
                                # Fetch data from Simbad
                                if not tar_info:
                                    break
                                else:
                                    # If target found, confirm or input data manually
                                    t_ra = tar_info["Coordinates (RA, DEC) in degrees"][0]
                                    t_dec = tar_info["Coordinates (RA, DEC) in degrees"][1]
                                    t_pmra = tar_info["Proper Motion (PMRA, PMDEC) in arcsec/yr"][0]
                                    t_pmdec = tar_info["Proper Motion (PMRA, PMDEC) in arcsec/yr"][1]
                                    t_epoch = tar_info["Epoch"]
                                    t_equinox = tar_info["Equinox"]
                                    input_string = f"""Data:
Target name: {ob.target.tar_name}
Declination (dd:mm:ss): {degrees_to_dec(t_dec)}
Right Ascension (hh:mm:ss): {degrees_to_ra(t_ra)}
Proper Motion Declination ("/yr): {t_pmdec}
Proper Motion Right Ascension ("/yr): {t_pmra}

1. Confirm
2. Input manually
"""
                                
                                    if 'nan' in input_string:  # If proper motions are 0 it gives them as nan. This section is for fixing it
                                        t_pmra = 0
                                        t_pmdec = 0
                                        input_string = f"""Data:
Target name: {ob.target.tar_name}
Declination (dd:mm:ss): {degrees_to_dec(t_dec)}
Right Ascension (hh:mm:ss): {degrees_to_ra(t_ra)}
Proper Motion Declination ("/yr): 0
Proper Motion Right Ascension ("/yr): 0

1. Confirm
2. Input manually
"""
                                layer14 = input(input_string)
                                
                                if layer14 == '2':
                                    # Manual data input
                                    ob.target.dec = dec_to_degrees(input("Declination (dd:mm:ss): "))
                                    ob.target.ra = ra_to_degrees(input("Right Ascension (hh:mm:ss): "))
                                    ob.target.equinox = input("Equinox: ")
                                    ob.target.epoch = input("Epoch: ")
                                    ob.target.pmra = get_float_input("Proper Motion Right Ascension (\"/yr): ")
                                    ob.target.pmd = get_float_input("Proper Motion Declination (\"/yr): ")
                                    ob.target.dra = get_float_input("Differential Right Ascension (\"/yr): ")
                                    ob.target.dd = get_float_input("Differential Declination (\"/yr): ")
                                    breaking = True
                                    break
                                    
                                elif layer14 == '1':
                                    # Confirm fetched data
                                    ob.target.dec = float(t_dec)
                                    ob.target.ra = float(t_ra)
                                    ob.target.equinox = t_equinox
                                    ob.target.epoch = t_epoch
                                    ob.target.pmra = float(t_pmra)
                                    ob.target.pmd = float(t_pmdec)
                                    ob.target.dra = get_float_input("Differential Right Ascension (\"/yr): ")
                                    ob.target.dd = get_float_input("Differential Declination (\"/yr): ")
                                    breaking = True
                                    break
                                
                        elif layer13 == '2':
                            # Manual data input
                            ob.target.dec = dec_to_degrees(input("Declination (dd:mm:ss): "))
                            ob.target.ra = ra_to_degrees(input("Right Ascension (hh:mm:ss): "))
                            ob.target.equinox = input("Equinox: ")
                            ob.target.epoch = input("Epoch: ")
                            ob.target.pmra = get_float_input("Proper Motion Right Ascension (\"/yr): ")
                            ob.target.pmd = get_float_input("Proper Motion Declination (\"/yr): ")
                            ob.target.dra = get_float_input("Differential Right Ascension (\"/yr): ")
                            ob.target.dd = get_float_input("Differential Declination (\"/yr): ")
                            breaking = True
                            break
                       
                        if breaking:
                            break


                elif layer1 == '3':
                    ob.cons_set = ob.Cons_Set()  # Initialize constraint set
                    if len(ob_list) > 0:  # Loading previous cons sets like obs. descrption
                        print("Your last Constraint Sets:")
                        for i in range(len(ob_list)):
                            print(f"{i+1}. {ob_list[i].cons_set.cons_name}")

                    while True:
                        option2 = input("""1. Input from your last Constraint Sets
2. Give a new Constraint Set
""")
                        if option2 == '1' and len(ob_list) != 0:
                            for i in range(len(ob_list)):
                                print(f"{i+1}. {ob_list[i].cons_set.cons_name}")

                            option3 = int(input("Choose your Constraint Set: "))
                            if option3 in range(1, len(ob_list) + 1):
                                ob.cons_set = deepcopy(ob_list[option3 - 1].cons_set)
                                if ob.cons_set.calculation and ob.obs_description.sci_temp_sig_noise.calculation:
                                    ob.obs_description.sci_temp_sig_noise.calc_exp_time(ob.cons_set, solid_angle)
                                break
                            else:
                                print("Invalid choice. Please select a valid Constraint Set.")

                        elif option2 == '2':
                            ob.cons_set.calculation = True  # Indicates that you've filled constrants set for exposure time claculation from snr
                            ob.cons_set.cons_name = input("Constraint Name: ")  # Input constraint name
                            ob.cons_set.airmass = get_float_input("Airmass: ")  # Input airmass

                            layer12 = input("""Sky Transparency:
1. Photometric
2. Clear
3. Variable, thin cirrus
4. Variable, thick cirrus
""")
                            # Map input to sky transparency
                            if layer12 == '1':
                                ob.cons_set.sky_trans = "Photometric"
                            elif layer12 == '2':
                                ob.cons_set.sky_trans = "Clear"
                            elif layer12 == '3':
                                ob.cons_set.sky_trans = "Variable, thin cirrus"
                            elif layer12 == '4':
                                ob.cons_set.sky_trans = "Variable, thick cirrus"
                                        
                            ob.cons_set.img_quality = get_float_input("Image Quality: ")  # Input image quality
                            ob.cons_set.lunar_ill = get_float_input("Lunar Illumination (Between 0 - 1): ")  # Input lunar illumination
                            ob.cons_set.mad = get_float_input("Moon Angular Distance (degree) (If you want it to be default type 30): ")  # Input moon angular distance
                            ob.cons_set.twilight = get_float_input("Twilight (min): ")  # Input twilight
                            ob.cons_set.pwv = get_float_input("PWV (mm): ")  # Input precipitable water vapor
                            if ob.cons_set.calculation and ob.obs_description.sci_temp_sig_noise.calculation:  # Check if both sections are filled
                                ob.obs_description.sci_temp_sig_noise.calc_exp_time(ob.cons_set, solid_angle)

                            break
                        elif option2 == '1' and len(ob_list) == 0:
                            print("No previous constraint sets found.")
                            continue


                elif layer1 == '4':
                    # Time intervals details
                    ob.time_intervals = ob.Time_Intervals()  # Initialize time intervals
                    ob.time_intervals.beginning = input("Beginning of your preferred time interval (d/m/y): ")  # Input beginning time
                    ob.time_intervals.end = input("End of your preferred time interval(d/m/y): ")  # Input end time

                elif layer1 == '5':
                    # Ephemeris details
                    ob.ephemeris = ob.Ephemeris()  # Initialize ephemeris
                    while True:
                        ephemeris_filepath = input("Upload an Ephemeris File: ")  # Upload ephemeris file
                        if os.path.isfile(ephemeris_filepath):
                            shutil.copy(ephemeris_filepath, f'output/{user}/{user}_OB_no_{len(ob_list)}_Ephemeris_file.xlxx')  # Copy ephemeris file to output
                            ob.ephemeris.ephemeris_ava = True  # Set ephemeris availability to True
                            break
                        else:
                            # Prompt for incorrect file path
                            print("Invalid file path.")
                            retry_choice = input('''
1. Give another path
0. Quit anyway
''')
                            if retry_choice == '0':
                                break

                elif layer1 == '6':
                    # Observation preparation details
                    if ob.obsprep is None:
                        ob.obsprep = ob.Obsprep()  # Initialize observation preparation
                    if ob.obsprep.pointing is None:
                        ob.obsprep.pointing = ob.Obsprep.Pointing()  # Initialize pointing
                    if ob.obsprep.blind_offset is None:
                        ob.obsprep.blind_offset = ob.Obsprep.Blind_Offset()  # Initialize blind offset
                    if ob.obsprep.obs_offset is None:
                        ob.obsprep.obs_offset = ob.Obsprep.Obs_Offset()  # Initialize observing offset
                    if ob.obsprep.guide_star is None:
                        ob.obsprep.guide_star = ob.Obsprep.Guide_Star()  # Initialize guide star

                    new_dec = deepcopy(ob.target.dec)  # Copy target declination
                    new_ra = deepcopy(ob.target.ra)  # Copy target right ascension
                    initial_plot = True  # Flag for initial plot
                    while True:
                        # Observation preparation menu
                        layer3 = input('''
Guide Star and Blind Offset is optional.
1. Pointing
2. Blind Offset (Optional)
3. Obs Offset
4. Guide Star (Optional)
0. Go Back
''')
                        if layer3 == '1':
                            # Pointing details
                            if initial_plot:
                                update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Initial plot
                                initial_plot = False
                            else:
                                update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Update plot
                            ob.obsprep.pointing.diff_dec = get_float_input("Differential Declination (arcsec): ") / 3600  # Input differential declination
                            ob.obsprep.pointing.diff_ra = get_float_input("Differential Right Ascension (sec): ") / 240  # Input differential right ascension
                            ob.obsprep.pointing.pos_angle = get_float_input("Position Angle: ")  # Input position angle
                            new_ra += ob.obsprep.pointing.diff_ra  # Update right ascension
                            new_dec += ob.obsprep.pointing.diff_dec  # Update declination
                            update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Update plot with new position

                        elif layer3 == '2':
                            # Blind offset details
                            if initial_plot:
                                update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Initial plot
                                initial_plot = False
                            else:
                                update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Update plot
                            while True:
                                # Blind offset menu
                                layer9 = input('''
1. Add a Blind Offset
0. Quit
''')
                                if layer9 == '1':
                                    # Input blind offset details
                                    ob.obsprep.blind_offset.acq_dec.append(input("Acquisition Declination (dd:mm:ss):  "))
                                    ob.obsprep.blind_offset.acq_ra.append(input("Acquisition Right Ascension (hh:mm:ss): "))
                                    ob.obsprep.blind_offset.acq_mag.append(get_float_input("Acquisition Magnitude: "))
                                    update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Update plot
                                elif layer9 == '0':
                                    break

                        elif layer3 == '3':
                            # Observing offset details
                            if initial_plot:
                                update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Initial plot
                                initial_plot = False
                            else:
                                update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Update plot
                            while True:
                                # Observing offset menu
                                layer10 = input('''
1. Add an Observing Offset
0. Quit
''')
                                if layer10 == '1':
                                    # Input observing offset details
                                    ob.obsprep.obs_offset.off_dec.append(get_float_input("Declination Offset (arcsec): "))
                                    ob.obsprep.obs_offset.off_ra.append(get_float_input("Right Ascension Offset (arcsec): "))
                                    ob.obsprep.obs_offset.obs_off.append(input("Observing Offset (‘O’ for Object or ‘S’ for sky): "))
                                    update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Update plot
                                elif layer10 == '0':
                                    break

                        elif layer3 == '4':
                            # Guide star details
                            if initial_plot:
                                update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Initial plot
                                initial_plot = False
                            else:
                                update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Update plot
                            while True:
                                # Guide star menu
                                layer11 = input('''
1. Add a Guiding Star
0. Quit
''')
                                if layer11 == '1':
                                    # Input guide star details
                                    ob.obsprep.guide_star.guide_dec.append(input("Guide Star Declination (dd:mm:ss): "))
                                    ob.obsprep.guide_star.guide_ra.append(input("Guide Star Right Ascension (hh:mm:ss): "))
                                    ob.obsprep.guide_star.guide_mag.append(get_float_input("Guide Star Magnitude: "))
                                    update_plot(user, ob, new_ra, new_dec, float(ob.obsprep.pointing.pos_angle))  # Update plot
                                elif layer11 == '0':
                                    break

                        elif layer3 == '0':
                            # Close plot and go back
                            plt.close('all')
                            break

                elif layer1 == '7':
                    # Finding chart details
                    ob.finding_chart = ob.Finding_Chart()  # Initialize finding chart
                    while True:
                        # Finding chart menu
                        layer8 = input('''
1. Upload a Finding Chart
2. Generate a Finding Chart
0. Go Back
''')
                        if layer8 == '1':
                            # Upload finding chart
                            while True:
                                fin_filepath = input("Upload a Finding Chart: ")  # Input file path
                                if os.path.isfile(fin_filepath):
                                    shutil.copy(fin_filepath, f'outputs/{user}/{user}_OB_no_{len(ob_list)}_Finding_Chart.png')  # Copy to output
                                    ob.finding_chart.fin_ava = True  # Set finding chart availability to True
                                    break
                                else:
                                    # Prompt for incorrect file path
                                    print("Invalid file path.")
                                    retry_choice = input('''
1. Give another path
0. Quit anyway
''')
                                    if retry_choice == '0':
                                        break

                        elif layer8 == '2':
                            # Generate finding chart
                            fc.main(user, user, ob.obs_description.obs_name, ob.target.tar_name, ob.target.ra, ob.target.dec, ob.obs_description.fin_temp.wavelength, ob.obs_description.fin_temp.scale_length, ob.obs_description.fin_temp.fov, ob.obsprep.obs_offset.off_ra, ob.obsprep.obs_offset.off_dec, ob.obsprep.pointing.pos_angle)  # Generate finding chart
                            plt.savefig(f'outputs/{user}/{user}_OB_no_{len(ob_list)}_Finding_Chart.png')  # Save finding chart to output
                            plt.close('all')
                            ob.finding_chart.fin_ava = True  # Set finding chart availability to True
                            break

                        elif layer8 == '0':
                            break

                elif layer1 == '0':
                    # Check if new_dec and new_ra were assigned values
                    if new_dec is not None and new_ra is not None:  # The program put coordinates as the updated coordinates from obsprep. This section is for that
                        ob.target.dec = degrees_to_dec(new_dec)
                        ob.target.ra = degrees_to_ra(new_ra)
                    else:
                        # Fall back to original target RA and DEC if ObsPrep wasn't set
                        ob.target.dec = degrees_to_dec(ob.target.dec) if float(ob.target.dec) else ob.target.dec
                        ob.target.ra = degrees_to_ra(ob.target.ra) if float(ob.target.ra) else ob.target.ra

                    # Check for errors before closing OB
                    if closing_error(ob):
                        if ob.obs_description.sci_temp_exp or ob.obs_description.sci_temp_sig_noise:
                            ob.obs_description.exp_time = ob.obs_description.calculate_exp_time()  # Calculate exposure time
                            ob.obs_description.filt = ob.obs_description.calculate_filter()  # Determine filter
                            ob.obs_description.no_exp = ob.obs_description.calculate_no_exp()  # Calculate number of exposures
                            ob.obs_description.tar_mag = ob.obs_description.calculate_tar_mag()  # Calculate target magnitude
                            ob.obs_description.binn = ob.obs_description.calculate_bin()
                            ob.obs_description.typee = ob.obs_description.calculate_type()
                            ob.obs_description.fin_temp = ob.Obs_Description.Fin_Temp()  # Initialize finding chart template
                            ob.obs_description.fin_temp.obs_filter = ob.obs_description.filt  # Set finding chart filter
                            ob.obs_description.fin_temp.scale_length = 10  # Set scale length
                            ob.obs_description.fin_temp.fov = 4  # Set field of view
                        ob_list.append(ob)  # Add the new OB to the list
                        break

        elif layer0 == '2':
            # Display all created OBs
            for i, ob in enumerate(ob_list):
                print(f"OB {i+1}:")
                print("  Obs. Description:")
                print(f"    Obs Name: {ob.obs_description.obs_name}")
                print(f"    Comments: {ob.obs_description.comments}")
                if ob.obs_description.sci_temp_exp:
                    print("    Science Template (Exposure Time):")
                    print(f"     Target Magnitude: {ob.obs_description.sci_temp_exp.tar_mag}")
                    print(f"     Exposure Time: {ob.obs_description.sci_temp_exp.exp_time}")
                    print(f"     Number of Exposures: {ob.obs_description.sci_temp_exp.no_exp}")
                    print(f"     Observation Filter: {ob.obs_description.sci_temp_exp.obs_filter}")
                if ob.obs_description.sci_temp_sig_noise:
                    print("    Science Template (Signal to Noise):")
                    print(f"     Target Magnitude: {ob.obs_description.sci_temp_sig_noise.tar_mag}")
                    print(f"     Preferred Signal to Noise: {ob.obs_description.sci_temp_sig_noise.sig_noise}")
                    print(f"     Number of Exposures: {ob.obs_description.sci_temp_sig_noise.no_exp}")
                    print(f"     Observation Filter: {ob.obs_description.sci_temp_sig_noise.obs_filter}")
                if ob.obs_description.fin_temp:
                    print("    Finding Chart Template:")
                    print(f"      Observation Filter: {ob.obs_description.fin_temp.obs_filter}")
                    print(f"      Scale Length: {ob.obs_description.fin_temp.scale_length}")
                    print(f"      FOV: {ob.obs_description.fin_temp.fov}")
                print("  Target:")
                print(f"      Target Name: {ob.target.tar_name}")
                print(f"      Declination: {ob.target.dec}")
                print(f"      Right Ascension: {ob.target.ra}")
                print(f"      Equinox: {ob.target.equinox}")
                print(f"      Epoch: {ob.target.epoch}")
                print(f"      Proper Motion Right Ascension: {ob.target.pmra}")
                print(f"      Proper Motion Declination: {ob.target.pmd}")
                print(f"      Differential Right Ascension: {ob.target.dra}")
                print(f"      Differential Declination: {ob.target.dd}")
                print("  Constraint Set:")
                print(f"      Constraint Name: {ob.cons_set.cons_name}")
                print(f"      Airmass: {ob.cons_set.airmass}")
                print(f"      Sky Transparency: {ob.cons_set.sky_trans}")
                print(f"      Lunar Illumination: {ob.cons_set.lunar_ill}")
                print(f"      Moon Angular Distance: {ob.cons_set.mad}")
                print(f"      Twilight (min): {ob.cons_set.twilight}")
                print(f"      PWV (mm): {ob.cons_set.pwv}")
                print("  Time Intervals:")
                print(f"      Beginning: {ob.time_intervals.beginning}")
                print(f"      End: {ob.time_intervals.end}")
                print("  Ephemeris:")
                print(f"      Ephemeris File: {ob.ephemeris.ephemeris_ava}")
                print("  ObsPrep:")
                if ob.obsprep.pointing:
                    print("    Pointing:")
                    print(f"      Differential Declination: {ob.obsprep.pointing.diff_dec}")
                    print(f"      Differential Right Ascension: {ob.obsprep.pointing.diff_ra}")
                    print(f"      Position Angle: {ob.obsprep.pointing.pos_angle}")
                if ob.obsprep.blind_offset:
                    print("    Blind Offset:")
                    print(f"      Acquisition Declination: {ob.obsprep.blind_offset.acq_dec}")
                    print(f"      Acquisition Right Ascension: {ob.obsprep.blind_offset.acq_ra}")
                    print(f"      Acquisition Magnitude: {ob.obsprep.blind_offset.acq_mag}")
                if ob.obsprep.obs_offset:
                    print("    Obs Offset:")
                    print(f"      Offset Declination: {ob.obsprep.obs_offset.off_dec}")
                    print(f"      Offset Right Ascension: {ob.obsprep.obs_offset.off_ra}")
                    print(f"      Observing Offset: {ob.obsprep.obs_offset.obs_off}")
                if ob.obsprep.guide_star:
                    print("    Guide Star:")
                    print(f"      Guide Star Declination: {ob.obsprep.guide_star.guide_dec}")
                    print(f"      Guide Star Right Ascension: {ob.obsprep.guide_star.guide_ra}")
                    print(f"      Guide Star Magnitude: {ob.obsprep.guide_star.guide_mag}")
                print("  Finding Chart:")
                print(f"      Finding Chart Available: {ob.finding_chart.fin_ava}")
                print()

        elif layer0 == '0':
            # Exit the main menu loop
            break

    # Create a CBs object using the list of OBs
    cb = CBs(ob_list)
    # Write OBs and CBs to a CSV file
    write_csv(ob_list, cb, user)

# Execute the main menu function if the script is run directly
if __name__ == "__main__":
    main_menu()

