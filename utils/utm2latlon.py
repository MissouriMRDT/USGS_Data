#!/usr/bin/env python3

# ******************************************************************************
#  @file      utm2latlon.py
#  @brief     Parallelized UTM-to-Lat/Lon converter for CSV datasets.
# 
#  This script reads a CSV containing UTM coordinates and zone strings, converts
#  each point to geographic coordinates (latitude, longitude) using `pyproj`, and 
#  writes the results to one or more output CSV files. It supports chunked output
#  and uses multiprocessing to speed up conversion for large datasets.
#
#  Key features:
#    - Parses UTM zone strings (e.g., "15N") into projection metadata
#    - Converts (easting, northing, zone) to (latitude, longitude)
#    - Processes rows in parallel with `multiprocessing.Pool`
#    - Splits output into multiple CSV chunks (default 1000 rows each)
#    - Gracefully handles conversion errors and skips failed rows
# 
#  Example usage:
#      $ python utm2latlon.py  # Modify the call to process_csv at the bottom
# 
#  Dependencies:
#      - pyproj
#      - csv
#      - tqdm
#      - multiprocessing
# 
#  Author:     Eli Byrd (edbgkk@mst.edu)
#  Date:       2025-05-31
#  Copyright:  Copyright Mars Rover Design Team 2025 – All Rights Reserved
# ******************************************************************************

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import csv
import os
from pyproj import Proj, Transformer

def parse_zone(zone_str):
    """
    Parses a UTM zone string (e.g., '15N' or '33S') into zone number and hemisphere.

    Args:
        zone_str (str): UTM zone string consisting of a numeric zone followed by 
                        a hemisphere letter ('N' or 'S').

    Returns:
        tuple: A tuple (zone_num, hemisphere) where:
            - zone_num (int): The numeric UTM zone (1 through 60).
            - hemisphere (str): 'north' or 'south' depending on the letter.

    Raises:
        ValueError: If the zone number cannot be parsed from the string.
    """

    # Convert the expected format '15N' or '33S' into numeric zone and hemisphere
    zone_num = int(zone_str[:-1])
    hemisphere = 'north' if zone_str[-1].upper() == 'N' else 'south'

    # Return the parsed zone number and hemisphere
    return zone_num, hemisphere

def convert_utm_to_latlon(easting, northing, zone_str):
    """
    Converts UTM coordinates to geographic latitude and longitude (WGS84).

    Args:
        easting (float): Easting value in meters (UTM X coordinate).
        northing (float): Northing value in meters (UTM Y coordinate).
        zone_str (str): UTM zone string, e.g., '15N' or '33S'.

    Returns:
        tuple: A tuple (latitude, longitude) in decimal degrees.

    Notes:
        - The conversion uses the WGS84 datum.
        - Internally uses `pyproj` to handle the coordinate transformation.
    """

    # Parse the UTM zone string to get zone number and hemisphere
    zone_number, hemisphere = parse_zone(zone_str)

    # Create a UTM projection object using the parsed zone and hemisphere
    proj = Proj(proj='utm', zone=zone_number, hemisphere=hemisphere, datum='WGS84')

    # Create a transformer to convert from UTM to geographic coordinates (EPSG:4326)
    transformer = Transformer.from_proj(proj, 'epsg:4326', always_xy=True)

    # Transform the UTM coordinates to latitude and longitude
    lon, lat = transformer.transform(easting, northing)

    # Return the latitude and longitude as a tuple
    return lat, lon

def process_row(row):
    """
    Processes a single CSV row by converting UTM coordinates to latitude and longitude.

    Args:
        row (dict): A dictionary representing a row from a CSV file. Must contain:
            - 'easting': UTM easting value (string or float)
            - 'northing': UTM northing value (string or float)
            - 'zone': UTM zone string (e.g., '15N')

    Returns:
        dict or None: The updated row dictionary with added 'latitude' and 'longitude' 
        keys (as strings formatted to 6 decimal places), or None if an error occurs.

    Notes:
        - Invalid or missing values in the input row will result in the row being skipped.
        - Errors are printed to standard output for debugging.
    """

    try:
        # Convert easting and northing to float, and parse the zone string
        easting = float(row['easting'])
        northing = float(row['northing'])
        zone = row['zone']

        # Convert UTM to latitude and longitude
        lat, lon = convert_utm_to_latlon(easting, northing, zone)

        # Add latitude and longitude to the row, formatted to 6 decimal places
        row['latitude'] = f"{lat:.6f}"
        row['longitude'] = f"{lon:.6f}"

        # Return the updated row with new fields
        return row

    # Handle any exceptions that occur during processing
    except Exception as e:
        print(f"Error processing row {row}: {e}")
        return None

def write_chunk(chunk, fieldnames, output_prefix, chunk_idx):
    """
    Writes a chunk of processed rows to a CSV file.

    Args:
        chunk (list of dict): List of row dictionaries to write. Each row must contain 
                              keys matching the provided `fieldnames`.
        fieldnames (list of str): Ordered list of column names to write in the CSV header.
        output_prefix (str): Base name for the output file (excluding extension and chunk index).
        chunk_idx (int): Index of the chunk, used to distinguish file parts.

    Side Effects:
        Creates a CSV file named `{output_prefix}_part{chunk_idx}.csv` in the current directory.
        Only non-None rows in the chunk are written.

    Notes:
        - Ensures newline handling is consistent across platforms using `newline=''`.
        - Skips rows that are `None` (e.g., failed conversions).
    """

    # Create the output filename based on the prefix and chunk index
    filename = f"{output_prefix}_part{chunk_idx}.csv"

    with open(filename, 'w', newline='') as f:
        # Create a CSV writer with the specified fieldnames
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write the header row to the CSV file
        writer.writeheader()

        # Write each row in the chunk to the CSV file
        for row in chunk:
            if row:
                writer.writerow(row)

def process_csv(input_file, output_prefix, chunk_size=1000):
    """
    Reads a CSV file with UTM coordinates, converts them to latitude and longitude,
    and writes the results to chunked output CSV files.

    Args:
        input_file (str): Path to the input CSV file. Each row must include:
            - 'easting': UTM easting (X)
            - 'northing': UTM northing (Y)
            - 'zone': UTM zone string (e.g., '15N')
        output_prefix (str): Base filename prefix for output CSV chunks.
        chunk_size (int, optional): Number of rows per output file. Defaults to 1000.

    Side Effects:
        - Processes all rows in parallel using multiprocessing.
        - Appends 'latitude' and 'longitude' columns to each row.
        - Writes one or more CSV files to disk using the format: 
          `{output_prefix}_partN.csv`.

    Notes:
        - Invalid rows are skipped and not written to output files.
        - Uses `tqdm` to display a progress bar.
        - Optimized for large datasets with parallel processing and chunked writing.
    """

    # Open the input CSV file and read its contents
    with open(input_file, newline='') as infile:

        # Use DictReader to read rows as dictionaries with headers as keys
        reader = csv.DictReader(infile)

        # Prepare the fieldnames for output, adding 'latitude' and 'longitude'
        fieldnames = reader.fieldnames + ['latitude', 'longitude']

        # Read all rows into a list for processing
        rows = list(reader)

    # Create the results list to hold processed rows
    results = []

    # Use a multiprocessing pool to process rows in parallel
    with Pool(cpu_count()) as pool:
        for row in tqdm(pool.imap_unordered(process_row, rows), total=len(rows), desc="Processing"):
            results.append(row)

    # Split and write results
    for i in range(0, len(results), chunk_size):
        chunk = results[i:i+chunk_size]
        write_chunk(chunk, fieldnames, output_prefix, i // chunk_size)

if __name__ == '__main__':
    process_csv('trav_test.csv', 'output_latlon', chunk_size=100000)
