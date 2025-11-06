#!/usr/bin/env python3

# ******************************************************************************
#  @file      lidar_preprocess.py
#  @brief     End-to-end pipeline for loading, indexing, and analyzing LiDAR data.
# 
#  This script handles the full preprocessing of LAS 1.4 point cloud data for use in
#  Missouri S&T Mars Rover Design Team systems. It performs the following steps:
#    - Loads LAS files using `laspy` and extracts scaled coordinates + classification
#    - Converts geographic metadata (UTM zone, hemisphere) from VLR headers
#    - Inserts parsed points into a normalized SQLite database (`ProcessedLiDARPoints`)
#    - Constructs an R-Tree virtual index for spatial lookups using `easting` and `northing`
#    - Computes per-point metrics including:
#        - Surface normal vectors (x, y, z)
#        - Local terrain slope (degrees)
#        - Surface roughness (RMS plane fit error)
#        - Curvature
#        - Traversability score (based on slope & roughness)
#    - Uses multiprocessing for parallel computation and batched database updates
#    - Applies best practices for SQLite performance (WAL mode, deferred indexing)
# 
#  Intended for offline terrain preparation before real-time autonomous navigation.
# 
#  Example usage:
#      $ python lidar_preprocess.py --input path/to/*.las --output terrain.db
# 
#  Dependencies:
#      - laspy
#      - numpy
#      - sqlite3
#      - concurrent.futures
#      - argparse
# 
#  @author     Eli Byrd (edbgkk@mst.edu), ClayJay3 (claytonraycowen@gmail.com)
#  @date       2025-05-31
#  @copyright  Copyright Mars Rover Design Team 2025 – All Rights Reserved
# ******************************************************************************

import os
import sys
import time
import glob
import argparse
import sqlite3
import re
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import laspy
import numpy as np
from collections import deque

######################################
## Constants and Configuration
######################################

# Mapping of LAS classification codes to human-readable labels. Values 0-63: Reserved by LAS specification. 64-255: User definable.
CLASSIFICATION_MAP = {
    0: "Never Classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point Noise",
    8: "Reserved",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Reserved",
    13: "Wire Guard Shield",
    14: "Wire Conductor Phase",
    15: "Transmission Tower",
    16: "Wire Structure Connector",
    17: "Bridge Deck",
    18: "High Noise",
    19: "Overhead Structure",
    20: "Ignored Ground",
    21: "Snow",
    22: "Temporal Exclusion",
    23: "Reserved",
}

######################################
## Database Creation and Management
######################################

def create_db(db_path):
    """
    Initialize the SQLite database for storing and processing LiDAR point cloud data.

    This function creates the `ProcessedLiDARPoints` table, which is used to store
    point cloud data extracted from LAS files. Each point includes positional
    attributes (easting, northing, altitude), metadata (UTM zone, classification),
    and optional precomputed metrics (normal vector components, slope, roughness,
    curvature, and a traversal score used in terrain assessment).

    The function also ensures that the database directory exists and sets
    performance-optimized PRAGMA settings suitable for concurrent reads/writes.

    Table Schema:
    -------------
    - id             INTEGER PRIMARY KEY AUTOINCREMENT
    - easting        REAL: X-coordinate in UTM
    - northing       REAL: Y-coordinate in UTM
    - altitude       REAL: Elevation in meters
    - zone           TEXT: UTM zone identifier (e.g., "12N")
    - classification TEXT: Interpreted class label of the point (e.g., "Ground", "Vegetation")

    Precomputed Metric Fields:
    --------------------------
    - normal_x, normal_y, normal_z: Components of the surface normal vector
    - slope: Local terrain slope in degrees
    - rough: RMS plane fitting error representing roughness
    - curvature: Surface curvature estimate
    - trav_score: Normalized traversal cost score (0.0 to 1.0)

    Args:
    -----
    db_path : str
        Full file path to the SQLite database to be created or initialized.

    Returns:
    --------
    None
    """

    # Only create directory if a path component is provided.
    dir_name = os.path.dirname(db_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Open connection and enable WAL for better concurrency.
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode = WAL;")
    c.execute("PRAGMA synchronous = NORMAL;")

    # Create the table to store processed LiDAR points and computed metrics.
    c.execute("""
        CREATE TABLE IF NOT EXISTS ProcessedLiDARPoints (
            id             INTEGER PRIMARY KEY,
            easting        REAL    NOT NULL,
            northing       REAL    NOT NULL,
            altitude       REAL,
            zone           TEXT,
            classification TEXT,
              
            -- precomputed metrics
            normal_x       REAL,
            normal_y       REAL,
            normal_z       REAL,
            slope       REAL,
            rough       REAL,
            curvature   REAL,
            trav_score     REAL
        )
    """)

    # Commit the changes and close the connection.
    conn.commit()
    conn.close()

def create_rtree_index(db_path):
    """
    Create an R-Tree spatial index for fast range and neighbor queries on LiDAR points.

    This function creates a virtual table using SQLite's R-Tree module, which allows
    efficient 2D spatial indexing and querying. The index is built from the `easting`
    and `northing` coordinates of points in the `ProcessedLiDARPoints` table. This
    enables performant neighborhood searches during metric computations like
    slope, roughness, and curvature.

    Structure:
    ----------
    - Table Name: ProcessedLiDARPoints_idx
    - Columns:
        - id      : INTEGER, must match `ProcessedLiDARPoints.id`
        - min_x   : REAL, minimum easting (same as easting)
        - max_x   : REAL, maximum easting (same as easting)
        - min_y   : REAL, minimum northing (same as northing)
        - max_y   : REAL, maximum northing (same as northing)

    Notes:
    ------
    - The R-Tree is populated by copying existing entries from `ProcessedLiDARPoints`.
    - Each point is treated as a degenerate bounding box (min = max = coordinate).
    - PRAGMA `WAL` and `NORMAL` are used to improve concurrency and performance.

    Args:
    -----
    db_path : str
        Path to the SQLite database containing `ProcessedLiDARPoints`.

    Returns:
    --------
    None
    """

    # Open connection and enable WAL for better concurrency.
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode = WAL;")
    c.execute("PRAGMA synchronous = NORMAL;")

    # Create the R-tree virtual table for spatial indexing.
    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS ProcessedLiDARPoints_idx
        USING rtree(
            id,        -- matches ProcessedLiDARPoints.id
            min_x,     -- lower-bound easting
            max_x,     -- upper-bound easting
            min_y,     -- lower-bound northing
            max_y      -- upper-bound northing
        )
    """)

    # Populate the R-tree index from existing ProcessedLiDARPoints entries.
    c.execute("""
        INSERT INTO ProcessedLiDARPoints_idx(id, min_x, max_x, min_y, max_y)
        SELECT id, easting, easting, northing, northing FROM ProcessedLiDARPoints
    """)

    # Commit the changes and close the connection.
    conn.commit()
    conn.close()

def create_rtree_triggers(db_path):
    """
    Create SQLite triggers to keep the R-Tree index in sync with the main LiDAR table.

    This function sets up automatic triggers on the `ProcessedLiDARPoints` table so that
    any insert, delete, or update operation is mirrored in the `ProcessedLiDARPoints_idx`
    R-Tree spatial index. This ensures that the spatial index always reflects the current
    state of the main table without requiring manual updates.

    Triggers:
    ---------
    - rp_ai (AFTER INSERT): Adds the new point to the R-Tree using its easting/northing.
    - rp_ad (AFTER DELETE): Removes the corresponding R-Tree entry when a point is deleted.
    - rp_au (AFTER UPDATE): Updates the R-Tree entry if a point's coordinates change.

    Notes:
    ------
    - Uses PRAGMA `journal_mode = WAL` and `synchronous = NORMAL` for concurrency and speed.
    - The R-Tree entries are degenerate bounding boxes: (min = max = coordinate).

    Args:
    -----
    db_path : str
        Path to the SQLite database containing the LiDAR tables.

    Returns:
    --------
    None
    """

    # Open connection and enable WAL for better concurrency.
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode = WAL;")
    c.execute("PRAGMA synchronous = NORMAL;")

    # Create triggers to automatically maintain the R-Tree index in sync with changes to the ProcessedLiDARPoints table.
    c.executescript("""
        CREATE TRIGGER IF NOT EXISTS rp_ai
        AFTER INSERT ON ProcessedLiDARPoints
        BEGIN
            INSERT INTO ProcessedLiDARPoints_idx(id, min_x, max_x, min_y, max_y)
            VALUES (NEW.id, NEW.easting, NEW.easting, NEW.northing, NEW.northing);
        END;

        CREATE TRIGGER IF NOT EXISTS rp_ad
        AFTER DELETE ON ProcessedLiDARPoints
        BEGIN
            DELETE FROM ProcessedLiDARPoints_idx WHERE id = OLD.id;
        END;

        CREATE TRIGGER IF NOT EXISTS rp_au
        AFTER UPDATE ON ProcessedLiDARPoints
        BEGIN
            UPDATE ProcessedLiDARPoints_idx
            SET min_x = NEW.easting,
                max_x = NEW.easting,
                min_y = NEW.northing,
                max_y = NEW.northing
            WHERE id = OLD.id;
        END;
    """)

    # Commit the changes and close the connection.
    conn.commit()
    conn.close()

def create_indexes(db_path):
    """
    Create performance-enhancing indexes on the ProcessedLiDARPoints table.

    This function adds conventional B-tree indexes to frequently queried columns 
    in the ProcessedLiDARPoints table to speed up filtering, analysis, and route 
    planning operations. Indexes include common metrics such as slope, roughness, 
    and traversability, as well as spatial and classification data.

    Notes:
    ------
    - Indexes improve query speed at the cost of slightly slower inserts/updates.
    - Coordinates (easting, northing) are indexed together for spatial filtering,
      complementing the R-tree index used for radius-based spatial searches.

    Indexes Created:
    ----------------
    - classification
    - trav_score
    - rough
    - slope
    - curvature
    - altitude
    - normal_x
    - normal_y
    - normal_z
    - easting + northing (compound)

    Args:
    -----
    db_path : str
        Path to the SQLite database where indexes will be created.

    Returns:
    --------
    None
    """

    # Open connection and enable WAL for better concurrency.
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode = WAL;")
    c.execute("PRAGMA synchronous = NORMAL;")

    # Create indexes on the ProcessedLiDARPoints table to optimize query performance
    # for analysis, filtering, and path planning operations.
    c.executescript("""
        -- For analysis, filtering, or route planning
        CREATE INDEX IF NOT EXISTS idx_plp_classification ON ProcessedLiDARPoints(classification);
        CREATE INDEX IF NOT EXISTS idx_plp_trav_score     ON ProcessedLiDARPoints(trav_score);
        CREATE INDEX IF NOT EXISTS idx_plp_rough          ON ProcessedLiDARPoints(rough);
        CREATE INDEX IF NOT EXISTS idx_plp_slope          ON ProcessedLiDARPoints(slope);
        CREATE INDEX IF NOT EXISTS idx_plp_curvature      ON ProcessedLiDARPoints(curvature);
        CREATE INDEX IF NOT EXISTS idx_plp_altitude       ON ProcessedLiDARPoints(altitude);
        CREATE INDEX IF NOT EXISTS idx_plp_normal_x       ON ProcessedLiDARPoints(normal_x);
        CREATE INDEX IF NOT EXISTS idx_plp_normal_y       ON ProcessedLiDARPoints(normal_y);
        CREATE INDEX IF NOT EXISTS idx_plp_normal_z       ON ProcessedLiDARPoints(normal_z);
        CREATE INDEX IF NOT EXISTS idx_plp_coord          ON ProcessedLiDARPoints(easting, northing);
    """)

    # Commit the changes and close the connection.
    conn.commit()
    conn.close()

def create_spatial_views(db_path):
    """
    Create example spatial views using the R-tree index on LiDAR data.

    This function defines example SQL views that join the R-tree index 
    (`ProcessedLiDARPoints_idx`) with the main table (`ProcessedLiDARPoints`)
    to illustrate how bounding-box and attribute-based spatial queries can 
    be written using indexed lookups.

    These views use hardcoded bounding box coordinates for demonstration and 
    cannot accept parameters dynamically. They are meant as query templates 
    or educational references for building custom queries.

    Views Created:
    --------------
    - PointsNearby: All points within a bounding box.
    - NonGroundPointsNearby: All non-ground points in a bounding box.
    - PointsWithClassification: Points in a bounding box with class = 'Ground'.
    - PointsInRangeWithAltitude: Points in a bounding box within a specific altitude range.

    Note from Clayton:
    -----
    These views aren't actually helpful since views can't have parameters. I'm just leaving them here as a reference for different ways to query the data.

    Args:
    -----
    db_path : str
        Path to the SQLite database file.

    Returns:
    --------
    None
    """

    # Open connection and enable WAL for better concurrency.
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode = WAL;")
    c.execute("PRAGMA synchronous = NORMAL;")

    # Create useful spatial views using the R-tree index for rapid querying.
    c.executescript("""
        -- View: All points in a bounding box
        CREATE VIEW IF NOT EXISTS PointsNearby AS
        SELECT p.*
        FROM ProcessedLiDARPoints_idx AS idx
        JOIN ProcessedLiDARPoints AS p ON p.id = idx.id
        WHERE idx.min_x BETWEEN 519307 AND 519317
          AND idx.min_y BETWEEN 4252958 AND 4252973;

        -- View: All points except ground within the same bounding box
        CREATE VIEW IF NOT EXISTS NonGroundPointsNearby AS
        SELECT p.*
        FROM ProcessedLiDARPoints_idx AS idx
        JOIN ProcessedLiDARPoints AS p ON p.id = idx.id
        WHERE p.classification != 'Ground'
          AND idx.min_x BETWEEN 519307 AND 519317
          AND idx.min_y BETWEEN 4252958 AND 4252973;

        -- View: Points in a bounding box with a specific classification
        CREATE VIEW IF NOT EXISTS PointsWithClassification AS
        SELECT p.*
        FROM ProcessedLiDARPoints_idx AS idx
        JOIN ProcessedLiDARPoints AS p ON p.id = idx.id
        WHERE idx.min_x BETWEEN 519307 AND 519317
          AND idx.min_y BETWEEN 4252958 AND 4252973
          AND p.classification = 'Ground';

        -- View: Points in a bounding box and within a specific altitude range
        CREATE VIEW IF NOT EXISTS PointsInRangeWithAltitude AS
        SELECT p.*
        FROM ProcessedLiDARPoints_idx AS idx
        JOIN ProcessedLiDARPoints AS p ON p.id = idx.id
        WHERE idx.min_x BETWEEN 519307 AND 519317
          AND idx.min_y BETWEEN 4252958 AND 4252973
          AND p.altitude BETWEEN 100 AND 200;
    """)

    # Commit the changes and close the connection.
    conn.commit()
    conn.close()

######################################
## LAS File Processing
######################################

def extract_utm_zone(las):
    """
    Attempt to extract the UTM zone and hemisphere from a LAS file's metadata.

    This function inspects the Variable Length Records (VLRs) of a LAS 1.4 file
    to identify spatial reference information—such as Well-Known Text (WKT) strings—
    from which the UTM zone and hemisphere can be derived. The result is used to 
    georeference LiDAR points during preprocessing.

    Parameters
    ----------
    las : laspy.LasData
        A parsed LAS file object from the `laspy` library.

    Returns
    -------
    tuple of (int, str)
        A tuple containing:
        - `zone` (int): The UTM zone number (1-60). Returns -1 if not found.
        - `hemisphere` (str): 'N' for Northern Hemisphere, 'S' for Southern,
          or '?' if unknown.
    """
    
    # Attempt 1: Extract UTM zone from WKT string (most reliable).
    wkt_pattern = re.compile(r"UTM\s+zone\s+(\d+)([NS])", re.IGNORECASE)

    for vlr in las.header.vlrs:
        if isinstance(vlr, laspy.vlrs.known.WktCoordinateSystemVlr):
            wkt_string = vlr.string
            m = wkt_pattern.search(wkt_string)
            if m:
                zone = int(m.group(1))
                hemi = m.group(2).upper()
                return zone, hemi
    
    # Attempt 2: Search other VLR fields for UTM zone patterns.
    pattern = re.compile(r"UTM\s+zone\s*([0-9]+)([NS])?", re.IGNORECASE)
    for vlr in las.header.vlrs:
        # Try raw VLR payload data.
        payload = getattr(vlr, 'record_data', None) or getattr(vlr, 'record_data_bytes', None)
        if payload:
            text = payload.decode('utf-8', errors='ignore') if isinstance(payload, (bytes, bytearray)) else str(payload)
            m = pattern.search(text)
            if m:
                zone = int(m.group(1))
                hemi = m.group(2).upper() if m.group(2) else '?'
                return zone, hemi

        # Fallback to VLR description field.
        desc = getattr(vlr, 'description', None)
        if desc:
            text = desc.decode('utf-8', errors='ignore') if isinstance(desc, (bytes, bytearray)) else str(desc)
            m = pattern.search(text)
            if m:
                zone = int(m.group(1))
                hemi = m.group(2).upper() if m.group(2) else '?'
                return zone, hemi

    # If no UTM zone found, return default values.                
    return -1, '?'


def process_file(file_path, db_path, manual_zone=None):
    """
    Load a LAS file and insert point data into a SQLite database.

    This function reads a LAS 1.4 file using `laspy`, extracts spatial point
    data (easting, northing, altitude), classification codes, and the UTM
    zone and hemisphere. It maps classification codes to human-readable
    labels and performs batched inserts into the `ProcessedLiDARPoints` table
    within the specified SQLite database.

    Parameters
    ----------
    file_path : str
        Path to the `.las` file to process.
    db_path : str
        Path to the target SQLite database.
    manual_zone : str, optional
        Manually specified UTM zone (e.g., '15N').

    Notes
    -----
    - Assumes the LAS file is in UTM coordinates and contains a supported VLR
      for extracting the UTM zone (if present).
    - Classification labels are mapped using the `CLASSIFICATION_MAP` dictionary.
    - Uses batched inserts of 10,000 rows per commit to improve performance
      and reduce SQLite transaction overhead.
    - The database must already contain a table named `ProcessedLiDARPoints`.

    Returns
    -------
    None
    """

    # Load LAS file and extract UTM zone
    las = laspy.read(file_path)
    if manual_zone:
        zone_str = manual_zone
    else:
        zone, hemi = extract_utm_zone(las)
        zone_str = f"{zone}{hemi}" if zone > 0 else None

    # Stop and print a warning if UTM zone is not found
    if zone_str is None:
        print(f"[ERROR] Could not determine UTM zone for file: {file_path}", file=sys.stderr)
        print("        This LAS file does not contain UTM zone information in its VLRs.", file=sys.stderr)
        print("        Please rerun this script with the new argument '--zone <zone>' to manually specify the UTM zone (e.g., '15N').", file=sys.stderr)
        sys.exit(1)

    # Extract coordinate and classification data
    xs = las.x
    ys = las.y
    zs = las.z
    cls = las.classification

    # Prepare data rows for batch insertion
    rows = []
    for x, y, z, c in zip(xs, ys, zs, cls):
        label = CLASSIFICATION_MAP.get(c, 'User Definable' if c >= 64 else 'Reserved')
        rows.append((x, y, z, zone_str, label))

    # Connect to SQLite and insert in batches
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode = WAL;")
    c.execute("PRAGMA synchronous = NORMAL;")
    batch_size = 10000
    for i in range(0, len(rows), batch_size):
        c.executemany(
            """
            INSERT INTO ProcessedLiDARPoints (
                easting, northing, altitude, zone, classification
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows[i:i + batch_size]
        )
        conn.commit()

    # Close the database connection.
    conn.close()

######################################
## Metric Computation
######################################

def compute_for_point(c, pid, x0, y0, radius, max_slope_deg, max_rough):
    """
    Compute terrain metrics for a single point using neighboring points
    from an open SQLite connection.

    This function performs a localized plane fit using all neighboring
    points within a specified radius, computes the surface normal, slope,
    roughness (as RMS error from the fit), and curvature (based on PCA),
    then returns a tuple of metrics used to evaluate traversability.

    Parameters
    ----------
    c : sqlite3.Cursor
        SQLite cursor with access to the ProcessedLiDARPoints and its R-tree index.
    pid : int
        Point ID from the database.
    x0, y0 : float
        The easting and northing coordinates of the point.
    radius : float
        Radius (in meters) to search for neighboring points.
    max_slope_deg : float
        Maximum expected slope (in degrees) used for slope normalization.
    max_rough : float
        Maximum expected roughness used for RMS normalization.

    Returns
    -------
    tuple or None
        Tuple of the form:
            (pid, normal_x, normal_y, normal_z, slope_deg, roughness, curvature, trav_score)
        Returns None if not enough neighbors are found for a reliable computation.

    Notes
    -----
    - Uses least-squares plane fitting to compute normal and RMS.
    - Uses eigenvalue decomposition of covariance matrix for curvature.
    - Traversability score is derived as:
        1.0 - (normalized slope + normalized roughness + curvature),
      clamped to [0.0, 1.0].
    - Neighbor query leverages R-tree spatial index for efficiency.
    """

    try:
        # Query neighboring points using the R-Tree index
        c.execute("""
            SELECT easting, northing, altitude
            FROM ProcessedLiDARPoints AS R
            JOIN ProcessedLiDARPoints_idx AS idx ON R.id = idx.id
            WHERE idx.min_x BETWEEN ? AND ?
            AND idx.min_y BETWEEN ? AND ?
        """, (x0 - radius, x0 + radius, y0 - radius, y0 + radius))
        
        # Fetch all neighbors and check if we have enough points.
        nbrs = np.array(c.fetchall())
        if nbrs.shape[0] < 3:
            return None

        # Fit plane: Z = aX + bY + c
        X, Y, Z = nbrs[:,0], nbrs[:,1], nbrs[:,2]
        A = np.vstack((X, Y, np.ones_like(X))).T
        coeffs, *_ = np.linalg.lstsq(A, Z, rcond=None)
        a, b, _ = coeffs

        # Compute normal vector to the fitted plane
        n = np.array([-a, -b, 1.0])
        n /= np.linalg.norm(n)
        nx, ny, nz = n

        # Compute slope in degrees
        slope_deg = np.degrees(np.arctan(np.hypot(a, b)))

        # Compute RMS error (roughness)
        Z_fit = A.dot(coeffs)
        rms = np.sqrt(np.mean((Z - Z_fit)**2))

        # Compute curvature from eigenvalues of covariance matrix
        P = nbrs - nbrs.mean(axis=0)
        eigs = np.linalg.eigvalsh(np.cov(P, rowvar=False))
        curvature = eigs[0] / eigs.sum()

        # Normalize slope and roughness
        s_norm = min(1.0, slope_deg / max_slope_deg)
        r_norm = min(1.0, rms / max_rough)

        # Compute traversability score
        trav = max(0.0, 1.0 - (s_norm + r_norm + curvature))

        # Return the computed metrics as a tuple
        return (pid, nx, ny, nz, slope_deg, rms, curvature, trav)

    # Handle any exceptions that occur during processing
    except Exception as e:
        print(f"Error processing point {pid} at ({x0}, {y0}): {e}", file=sys.stderr)
        return None

def worker_batch(db_path, points, radius, max_slope_deg, max_rough):
    """
    Process a batch of points in a separate worker process.

    Each worker opens its own SQLite database connection to avoid concurrency
    issues, retrieves neighbors for each point using the R-tree index, and
    computes terrain metrics such as normal vector, slope, roughness, curvature,
    and a derived traversability score.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database containing ProcessedLiDARPoints.
    points : list of tuple[int, float, float]
        A batch of (id, easting, northing) tuples representing point IDs and locations.
    radius : float
        Search radius in meters for neighborhood-based calculations.
    max_slope_deg : float
        Maximum slope used for normalizing slope in traversability scoring.
    max_rough : float
        Maximum roughness used for normalizing RMS in traversability scoring.

    Returns
    -------
    list[tuple]
        A list of result tuples, each of the form:
        (id, normal_x, normal_y, normal_z, slope_deg, roughness, curvature, trav_score)

    Notes
    -----
    - Logs skipped points (too few neighbors) and any computation exceptions.
    - Intended for parallel execution via `concurrent.futures.ProcessPoolExecutor`.
    - Avoids sharing database connections across processes to maintain thread safety.
    """

    # Ensure the process has a unique identifier for logging
    pid_str = f"[PID {os.getpid()}]"

    # Initialize results list
    results = []

    try:
        # Open a new SQLite connection for this worker process
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("PRAGMA journal_mode = WAL;")
        c.execute("PRAGMA synchronous = NORMAL;")

        # Process each point in the batch
        for pid, x0, y0 in points:
            try:
                # Compute the metrics for this point
                res = compute_for_point(c, pid, x0, y0, radius, max_slope_deg, max_rough)

                # If the result is valid, append it to results; otherwise, log a skip message
                if res:
                    results.append(res)
                else:
                    print(f"{pid_str} Skipping point {pid} at ({x0:.2f}, {y0:.2f}) - insufficient neighbors.")
            except Exception as e:
                print(f"{pid_str} ERROR at point {pid} ({x0:.2f}, {y0:.2f}): {e}", file=sys.stderr)

    # If any exception occurs during the worker's execution, log it with the process ID for easier debugging.
    except Exception as outer:
        print(f"{pid_str} ERROR initializing database connection: {outer}", file=sys.stderr)

    # Ensure the connection is closed properly, even if an error occurs.
    finally:
        conn.close()

    # Return the results collected by this worker
    return results

def compute_metrics(db_path,
                    radius=1.0,
                    max_slope_deg=90.0,
                    max_rough=1.0,
                    workers=20,
                    chunk_size=100,
                    print_rate=0.005):
    """
    Compute terrain metrics (normal vector, slope, roughness, curvature, traversability)
    for all LiDAR points in the database using parallel processing.

    This function:
    - Fetches all point positions from the database.
    - Partitions the workload into chunks.
    - Spawns multiple processes to compute metrics for each point using local neighborhoods.
    - Performs batched database updates for efficiency.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database containing ProcessedLiDARPoints.
    radius : float, optional
        Radius in meters to search for neighbors (default is 1.0).
    max_slope_deg : float, optional
        Maximum slope in degrees to normalize slope scoring (default is 90.0).
    max_rough : float, optional
        Maximum RMS value to normalize roughness scoring (default is 1.0).
    workers : int, optional
        Number of parallel worker processes (default is 20).
    chunk_size : int, optional
        Number of points to assign per worker batch (default is 100).
    print_rate : float, optional
        Proportion of progress (0 to 1) at which to print status updates (default is 0.005, or every 0.5%).

    Returns
    -------
    None

    Notes
    -----
    - Uses SQLite WAL mode for better concurrency.
    - Avoids loading full point data into memory; only coordinates are fetched initially.
    - Metrics are computed via `compute_for_point`, which performs neighborhood plane fitting and scoring.
    - Suitable for large datasets with millions of points.
    """

    # Open connection to count total points and fetch them all
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode = WAL;")
    c.execute("PRAGMA synchronous = NORMAL;")

    # Count total points for progress tracking
    c.execute("SELECT COUNT(*) FROM ProcessedLiDARPoints")
    total = c.fetchone()[0]
    print(f"  Computing metrics for {total} points using {workers or os.cpu_count()} workers…")

    # Fetch all point coordinates (ID, easting, northing)
    fetchAllSTime = time.time()
    c.execute("SELECT id, easting, northing FROM ProcessedLiDARPoints")
    all_points = c.fetchall()
    conn.close()
    fetchAllTime = time.time() - fetchAllSTime
    print(f"  Fetch All completed in {fetchAllTime:.2f} seconds.")

    # Split all points into chunks for parallel processing
    chunks = [all_points[i:i+chunk_size] for i in range(0, len(all_points), chunk_size)]

    # Initialize variables for progress tracking and updates
    updates = []
    done = 0
    next_print = print_rate
    start = time.time()

    # Print initial status and compute initial ETA
    elapsed = time.time() - start
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else float('inf')
    print(f"  Processing {len(chunks)} chunks of {chunk_size} points each...")
    print(f"    [{done:>10,d}/{total:,}] {done/total:>5.1%} "
                            f"elapsed {elapsed:>6.1f}s ETA {eta:>6.1f}s")

    # Open write connection to apply updates
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("PRAGMA journal_mode = WAL;")
        c.execute("PRAGMA synchronous = NORMAL;")

        # Use a process pool for parallel computation
        with ProcessPoolExecutor(max_workers=workers) as exe:
            chunk_queue = deque(chunks) # Queue of work units
            futures = {}                # Track futures and their input chunks

            # Submit initial chunks (prefill pipeline)
            for _ in range(min(len(chunk_queue), workers * 4)):
                chunk = chunk_queue.popleft()
                future = exe.submit(worker_batch, db_path, chunk, radius, max_slope_deg, max_rough)
                futures[future] = chunk

            # As futures complete...
            while futures:
                for fut in as_completed(futures):
                    try:
                        # Retrieve computed results
                        results = fut.result()
                        updates.extend(results)
                        done += len(results)
                        
                    # Log any failed chunks with traceback
                    except Exception as e:
                        import traceback
                        print(f"    [ERROR] Future failed for chunk: {futures[fut][:2]}")
                        traceback.print_exc()

                    # Progress output based on percent complete
                    if done / total >= next_print:
                        elapsed = time.time() - start
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (total - done) / rate if rate > 0 else float('inf')
                        print(f"    [{done:>10,d}/{total:,}] {done/total:>5.1%} "
                            f"elapsed {elapsed:>6.1f}s ETA {eta:>6.1f}s")
                        next_print += print_rate

                    # Launch next chunk if any remain
                    if chunk_queue:
                        new_chunk = chunk_queue.popleft()
                        new_future = exe.submit(worker_batch, db_path, new_chunk, radius, max_slope_deg, max_rough)
                        futures[new_future] = new_chunk

                    # Remove completed future
                    del futures[fut]

                    # Periodically flush updates to DB in batches
                    if len(updates) >= 1000:
                        c.execute("BEGIN")
                        c.executemany("""
                            UPDATE ProcessedLiDARPoints
                            SET normal_x    = ?, normal_y    = ?, normal_z    = ?,
                                slope       = ?, rough       = ?, curvature   = ?, trav_score = ?
                            WHERE id = ?
                        """, [(nx,ny,nz,slope,r,curv,trav,pid) for (pid,nx,ny,nz,slope,r,curv,trav) in updates])
                        conn.commit()
                        updates.clear()

        # Final flush of any remaining updates
        if updates:
            c.execute("BEGIN")
            c.executemany("""
                UPDATE ProcessedLiDARPoints
                   SET normal_x    = ?, normal_y    = ?, normal_z    = ?,
                       slope       = ?, rough       = ?, curvature   = ?, trav_score = ?
                 WHERE id = ?
            """, [(nx,ny,nz,slope,r,curv,trav,pid) for (pid,nx,ny,nz,slope,r,curv,trav) in updates])
            conn.commit()

    # Print final summary of processing
    print(f"  All done in {time.time() - start:.1f}s")

######################################
## Main Function
######################################

def main():
    """
    Main driver function to ingest one or more LAS files into a SQLite database,
    compute terrain metrics, and set up spatial and attribute indexes for querying.

    Workflow:
    1. Validates the input path (file or directory).
    2. Creates the database schema if needed.
    3. Loads LAS point data into `ProcessedLiDARPoints`.
    4. Builds an R-tree index for spatial queries.
    5. Computes metrics for each point (normal vector, slope, roughness, etc.).
    6. Adds R-tree triggers to keep the index in sync.
    7. Adds B-tree indexes to optimize common queries.

    Command-line arguments:
    -----------------------
    input  : Path to a `.las` file or a directory containing `.las` files.
    output : Path to the output SQLite database file.
    """

    # Set up argument parser for command line arguments.
    parser = argparse.ArgumentParser(description='Load LAS files into a spatial SQLite database with metrics')
    parser.add_argument('input', help='Input LAS file or directory')
    parser.add_argument('output', help='Output SQLite database file')
    parser.add_argument('--workers', type=int, help='Number of parallel workers for metric computation', default=8)
    parser.add_argument('--radius', type=float, help='Radius (m) for neighborhood searches during metric computation', default=5.0)
    parser.add_argument('--zone', help='Manually specify UTM zone and hemisphere (e.g., 15N)', default=None)
    args = parser.parse_args()

    # Validate input arguments.
    if not os.path.exists(args.input):
        print(f"Path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, '*.las'))
        if not files:
            print(f"No .las files found in directory: {args.input}", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.input.lower().endswith('.las'):
            print(f"Not a .las file: {args.input}", file=sys.stderr)
            sys.exit(1)
        files = [args.input]

    # Step 0: Start overall time tracking.
    print("[Step 0] - Starting time tracking...")
    overall_start = time.time()

    # Step 1: Create the database schema
    print("[Step 1] - Creating database schema...")
    create_db(args.output)

    # Step 2: Process each LAS file and insert data into database
    print("[Step 2] - Processing LAS files...")
    processLasStart = time.time()
    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] Processing: {f} ...")
        start = time.time()
        try:
            # Verify the zone is valid if specified
            if args.zone and not re.match(r'^\d{1,2}[NS]$', args.zone):
                print(f"Invalid UTM zone format: {args.zone}. Expected format is '15N' or '33S'.", file=sys.stderr)
                sys.exit(1)

            process_file(f, args.output, args.zone)
        except Exception as e:
            print(f"    Error processing {f}: {e}", file=sys.stderr)
            continue
        print(f"    Done in {time.time() - start:.2f} seconds")
    print(f"  All LAS files processed successfully in {time.time() - processLasStart:.2f} seconds.")

    # Step 3: Build R-tree index on spatial coordinates
    print("[Step 3] - Creating R-Tree index...")
    create_rtree_index(args.output)

    # Step 4: Compute terrain metrics in parallel
    print("[Step 4] - Computing terrain metrics (this may take a while)...")
    start = time.time()
    compute_metrics(args.output, radius=args.radius, workers=args.workers)
    print(f"  Metrics computed in {time.time() - start:.2f} seconds.")

    # Step 5: Add triggers to auto-update R-tree when base table is modified
    print("[Step 5] - Adding R-tree triggers...")
    create_rtree_triggers(args.output)

    # Step 6: Add B-tree indexes for common attribute queries
    print("[Step 6] - Creating attribute indexes...")
    create_indexes(args.output)

    # Summary
    total = time.time() - overall_start
    print("\n[Summary] - Database setup complete.")
    print(f"  Processed {len(files)} LAS files into {args.output}.")
    print(f"  Total time: {total:.2f} seconds.")
    print("  R-tree index, triggers, and attribute indexes created successfully.")
    print("  You can now query `ProcessedLiDARPoints` and `ProcessedLiDARPoints_idx` for spatial analysis.")

if __name__ == '__main__':
    main()
