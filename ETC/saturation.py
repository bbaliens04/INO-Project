from astroquery.mast import Catalogs, Observations
from astroquery.skyview import SkyView
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astroquery.vizier import Vizier
from astropy.wcs import WCS
from astroquery.sdss import SDSS

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

def saturation_chart(ra, dec, filter_band, magnitude_limit, position_angle_deg, fov = 4 / 60):
    """
    Generate a saturation chart for a given sky region and filter band.
    
    Args:
        ra (float): Right Ascension in degrees.
        dec (float): Declination in degrees.
        filter_band (str): The filter band (e.g., 'u', 'g', 'r', 'i', 'z').
        magnitude_limit (float): The magnitude limit for saturation.
        position_angle_deg (float): The position angle of the square in degrees.
        fov (float): The field of view (in degrees).
        
    Returns:
        None: Displays the saturation chart.
    """
    
    # Convert RA and Dec to a SkyCoord object
    coordinates = SkyCoord(ra, dec, unit=(u.deg, u.deg))

    # Initialize Vizier to query Gaia DR2 catalog for RA and Dec
    vizier = Vizier(columns=['RA_ICRS', 'DE_ICRS'])
    vizier.ROW_LIMIT = 200  # Set row limit to retrieve all stars within the FOV

    # Query the Gaia DR2 catalog for stars within the FOV
    result = vizier.query_region(coordinates, radius=fov * u.deg, catalog='I/345/gaia2')

    # Initialize list to hold RA and Dec of stars
    star_coords_list = []

    # Check if any stars were found
    if len(result) > 0:
        stars = result[0]
        # Extract RA and Dec for each star
        for star in stars:
            ra_deg = star['RA_ICRS']
            dec_deg = star['DE_ICRS']
            star_coords_list.append((ra_deg, dec_deg))
    else:
        print("No stars found in the specified region and catalog.")

    # Array to hold the results (RA, Dec, magnitude)
    results = []

    # Query SDSS for the specified filter band magnitude at each star's position
    for ra_deg, dec_deg in star_coords_list:
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')

        try:
            # Query SDSS for photometric data
            photometric_data = SDSS.query_region(
                coord, spectro=False, radius=2 * u.arcsec, photoobj_fields=[filter_band]
            )

            # Extract magnitude or set to None if not found
            if photometric_data and len(photometric_data) > 0:
                mag = photometric_data[filter_band][0]
                if mag == -9999.00:  # SDSS had a bug showing some magnitudes of some stars -9999. It's for filtering those datas
                    mag = None
            else:
                mag = None

        except Exception as e:
            print(f"Failed to retrieve data for {coord}: {e}")
            mag = None

        # Append RA, Dec, and magnitude to results list
        results.append((ra_deg, dec_deg, mag))

    # Convert results to a numpy array for easy processing
    results_array = np.array(results)

    # Filter the results to only include stars with magnitude below the limit
    filtered_results = results_array[(results_array[:, 2] != None) & (results_array[:, 2].astype(float) < magnitude_limit)]

    # Extract the RA and Dec of filtered stars
    star_coords = [[inner_array[0], inner_array[1]] for inner_array in filtered_results]

    # Use SkyView to fetch an image of the region with the specified FOV
    image_list = SkyView.get_images(position=coordinates, survey='DSS', width=2 * fov * u.deg, height=2 * fov * u.deg)

    # Display the image and overlay the saturation chart
    if image_list:
        # Extract the WCS information from the image
        wcs = WCS(image_list[0][0].header)

        plt.figure(figsize=(8, 8))
        plt.imshow(image_list[0][0].data, cmap='gray', origin='lower')

        # Calculate the center of the image
        center_x = image_list[0][0].data.shape[1] // 2
        center_y = image_list[0][0].data.shape[0] // 2

        # Draw a circle with a diameter equal to the FOV
        circle_radius_pixels = center_x / 2  # Assume FOV matches image size
        circle = plt.Circle((center_x, center_y),
                            radius=circle_radius_pixels,
                            edgecolor='red', facecolor='none', linewidth=1)

        # Add the circle to the plot
        plt.gca().add_patch(circle)

        # Draw a small plus sign in the middle to indicate the center
        plus_size = 10  # Size of the plus sign
        plt.plot([center_x, center_x], [center_y - plus_size, center_y + plus_size], color='red', linewidth=1)
        plt.plot([center_x - plus_size, center_x + plus_size], [center_y, center_y], color='red', linewidth=1)

        # Draw a square inside the circle rotated by the position angle
        position_angle_rad = np.deg2rad(position_angle_deg)

        # Calculate the vertices of the square
        square_side = circle_radius_pixels * np.sqrt(2)  # Length of square side
        half_side = square_side / 2

        # Define the four corners of the square before rotation
        corners = np.array([
            [-half_side, -half_side],
            [half_side, -half_side],
            [half_side, half_side],
            [-half_side, half_side]
        ])

        # Rotate the square by the position angle
        rotation_matrix = np.array([
            [np.cos(position_angle_rad), -np.sin(position_angle_rad)],
            [np.sin(position_angle_rad), np.cos(position_angle_rad)]
        ])
        rotated_corners = np.dot(corners, rotation_matrix)

        # Translate the square to the center of the circle
        rotated_corners[:, 0] += center_x
        rotated_corners[:, 1] += center_y

        # Create the square as a polygon and add it to the plot
        square = plt.Polygon(rotated_corners, edgecolor='red', facecolor='none', linewidth=1)
        plt.gca().add_patch(square)

        if star_coords:
            # Convert star coordinates to SkyCoord objects
            star_coords_skycoord = SkyCoord(ra=[coord[0] for coord in star_coords] * u.deg,
                                            dec=[coord[1] for coord in star_coords] * u.deg, frame='icrs')

            # Convert star coordinates to pixel positions in the image
            star_pixel_coords = wcs.world_to_pixel(star_coords_skycoord)

            # Plot the stars on the image
            plt.scatter(star_pixel_coords[0], star_pixel_coords[1], s=30, edgecolor='blue', facecolor='none', linewidth=1, label='Saturated Stars')

        # Add title and show the plot
        plt.title(f'Image at RA: {degrees_to_ra(ra)}, Dec: {degrees_to_dec(dec)}\nFOV (arcmin): {fov * 60} degrees (DSS Survey)')
        plt.legend()
        plt.show()
    else:
        print(f"No images found at RA: {ra}, Dec: {dec} in the DSS survey.")
