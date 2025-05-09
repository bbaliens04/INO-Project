from astroquery.jplhorizons import Horizons
from datetime import datetime, timedelta
import sys

# Function to generate ephemeris data
def generate_ephemeris(object_id, start_epoch, end_epoch, step_size_seconds, latitude, longitude, altitude):
    epochs = []
    current_time = start_epoch
    while current_time <= end_epoch:
        jd = (current_time - datetime(2000, 1, 1, 12)).total_seconds() / 86400 + 2451545.0
        epochs.append(jd)
        current_time += timedelta(seconds=step_size_seconds)
    
    obj = Horizons(id=object_id, location={'lon': longitude, 'lat': latitude, 'elevation': altitude}, epochs=epochs)
    ephemeris = obj.ephemerides()
    return ephemeris

# Function to convert ephemeris data to VLT (PAF) format
def convert_to_vlt_format(ephemeris, position_angle):
    vlt_ephemeris = []
    for entry in ephemeris:
        record = (
            f"INS.EPHEM.RECORD  \"{entry['datetime_str']}, "
            f"{entry['datetime_jd']}, "
            f"{entry['RA']}, "
            f"{entry['DEC']}, "
            f"{entry['delta']}, "
            f"{entry['delta_rate']}, "
            f"{entry['V']}, "
            f"{position_angle}, "
            f"*\""
        )
        vlt_ephemeris.append(record)
    return vlt_ephemeris

# Function to save ephemeris data to a file
def save_ephemeris_to_file(ephemeris, filename):
    with open(filename, "w") as file:
        for entry in ephemeris:
            file.write(f"{entry}\n")
    print(f"Ephemeris saved to {filename}")

# Function to print ephemeris data to the console
def print_ephemeris(ephemeris):
    for entry in ephemeris:
        print(f"{entry['datetime_str']}, {entry['RA']}, {entry['DEC']}")

if __name__ == "__main__":
    object_id = input("Enter the object ID (e.g., 499 for Mars): ")
    start_epoch = input("Enter the start epoch (YYYY-MM-DD HH:MM:SS): ")
    end_epoch = input("Enter the end epoch (YYYY-MM-DD HH:MM:SS): ")
    step_size = input("Enter the step size in seconds: ")
    position_angle = float(input("Enter the position angle in degrees: "))

    # INO coordinates
    latitude = 33.587  # INO Latitude in degrees
    longitude = 51.382  # INO Longitude in degrees
    altitude = 3.6  # INO Altitude in kilometers

    try:
        start_epoch = datetime.strptime(start_epoch, "%Y-%m-%d %H:%M:%S")
        end_epoch = datetime.strptime(end_epoch, "%Y-%m-%d %H:%M:%S")
        step_size_seconds = int(step_size)
    except ValueError as e:
        print(f"Invalid input: {e}")
        sys.exit(1)

    ephemeris = generate_ephemeris(object_id, start_epoch, end_epoch, step_size_seconds, latitude, longitude, altitude)
    vlt_ephemeris = convert_to_vlt_format(ephemeris, position_angle)
    save_ephemeris_to_file(vlt_ephemeris, f"{object_id}_ephemeris.paf")
    print_ephemeris(ephemeris)
