#!/usr/bin/env python3
"""
lidar_rf_realtime.py

LiDAR -> 3D viewer + RF propagation demo.

Revisions included here:
- Antenna pattern improvements and configs directory.
- Streaming full-resolution pointcloud (NDJSON) via /pointcloud_full.
- Server-side reservoir sampling for fast loading and KD-tree building.
- Worker initializer receives points and grid arrays to avoid per-task pickling.
- Simulation progress broadcasting: server sends {"type":"sim_progress","percent":N} over websockets.
- Client HTML/JS updated to progressively stream the point cloud and avoid flashing by keeping
  the previous display until the streamed replacement has reached a threshold.
"""
import argparse
import asyncio
import json
import math
import os
import random
import sqlite3
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

# KD-tree
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    from scipy.spatial import KDTree  # fallback

# ------------------------
# Antenna config loader + default (uses datasheet info)
# ------------------------
CONFIG_DIR = Path("./configs")

def ensure_default_configs():
    """
    Ensure configs/ exists and contains at least one example config.
    Creates:
      - AMY-9M16.json (900 MHz Yagi defaults)
      - AM-2G16-90.json (2.3-2.7 GHz sector)
      - AM-5G20-90.json (5.15-5.85 GHz sector)
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- AMY-9M16 (existing default) ---
    default_path = CONFIG_DIR / "AMY-9M16.json"
    if not default_path.exists():
        gain = 16.0
        hp_bw = 31.5
        vp_bw = 31.5
        fbr = 20.0
        back_gain = gain - fbr
        azimuth_pattern = [
            [0.0, gain],
            [5.0, gain - 0.5],
            [10.0, gain - 2.0],
            [15.0, gain - 3.0],
            [20.0, gain - 6.0],
            [30.0, gain - 10.0],
            [60.0, gain - 20.0],
            [120.0, back_gain - 1.0],
            [180.0, back_gain],
            [240.0, back_gain - 1.0],
            [300.0, gain - 20.0],
            [330.0, gain - 6.0]
        ]
        elevation_pattern = [
            [-30.0, -18.0],
            [-10.0, -3.0],
            [0.0, 0.0],
            [10.0, -3.0],
            [30.0, -12.0]
        ]
        default = {
            "name": "AMY-9M16",
            "model": "airMAX Yagi AMY-9M16",
            "type": "yagi",
            "gain_db": gain,
            "hp_bw_deg": hp_bw,
            "vp_bw_deg": vp_bw,
            "fwd_back_db": fbr,
            "freq_hz_min": 902e6,
            "freq_hz_max": 928e6,
            "sidelobe_db": -20.0,
            "hard_null": False,
            "azimuth_pattern": azimuth_pattern,
            "elevation_pattern": elevation_pattern,
            "notes": "Derived defaults & sample patterns from uploaded AMY-9M16 datasheet."
        }
        with open(default_path, "w") as f:
            json.dump(default, f, indent=2)
        print(f"[CONFIG] Created default antenna config at {default_path}")

    # --- AM-2G16-90 (2.4 GHz sector) ---
    sector_path = CONFIG_DIR / "AM-2G16-90.json"
    if not sector_path.exists():
        sector = {
            "name": "AM-2G16-90",
            "model": "AM-2G16-90 Sector (2.3-2.7 GHz)",
            "type": "sector",
            "gain_db": 16.0,
            "hp_bw_deg": 91.0,
            "vp_bw_deg": 90.0,
            "fwd_back_db": 20.0,
            "freq_hz_min": 2.3e9,
            "freq_hz_max": 2.7e9,
            "sidelobe_db": -18.0,
            "hard_null": False,
            "notes": "Approximated azimuth & elevation patterns digitized from datasheet polar plots. HPOL ~91° (6 dB).",
            "azimuth_pattern": [
                [0.0, 16.0],
                [10.0, 15.0],
                [20.0, 13.0],
                [30.0, 10.0],
                [40.0, 6.0],
                [45.5, 4.0],
                [60.0, -2.0],
                [90.0, -12.0],
                [120.0, -18.0],
                [150.0, -20.0],
                [180.0, -20.0],
                [210.0, -20.0],
                [240.0, -18.0],
                [270.0, -12.0],
                [300.0, -2.0],
                [319.0, 6.0],
                [330.0, 10.0],
                [340.0, 13.0]
            ],
            "elevation_pattern": [
                [-30.0, -18.0],
                [-10.0, -4.0],
                [0.0, 0.0],
                [5.0, -1.0],
                [10.0, -3.0],
                [20.0, -9.0],
                [30.0, -16.0]
            ]
        }
        with open(sector_path, "w") as f:
            json.dump(sector, f, indent=2)
        print(f"[CONFIG] Created sector antenna config at {sector_path}")

    # AM-5G20-90 (5 GHz sector)
    sector5g_path = CONFIG_DIR / "AM-5G20-90.json"
    if not sector5g_path.exists():
        sector5g = {
            "name": "AM-5G20-90",
            "model": "AM-5G20-90 Sector (5.15-5.85 GHz)",
            "type": "sector",
            "gain_db": 20.0,
            "hp_bw_deg": 91.0,
            "vp_bw_deg": 85.0,
            "fwd_back_db": 28.0,
            "freq_hz_min": 5.15e9,
            "freq_hz_max": 5.85e9,
            "sidelobe_db": -18.0,
            "hard_null": False,
            "notes": "Approximate azimuth & elevation patterns digitized from datasheet polar plots. HPOL ~91° (6 dB). Gain ~19.5-20.3 dBi range.",
            "azimuth_pattern": [
                [0.0, 20.0],
                [6.0, 19.2],
                [12.0, 17.8],
                [20.0, 15.0],
                [30.0, 10.0],
                [40.0, 3.0],
                [45.5, 0.0],
                [60.0, -6.0],
                [90.0, -12.0],
                [120.0, -18.0],
                [150.0, -20.0],
                [180.0, -20.0],
                [210.0, -20.0],
                [240.0, -18.0],
                [270.0, -12.0],
                [300.0, -6.0],
                [319.0, 3.0],
                [330.0, 10.0],
                [340.0, 15.0]
            ],
            "elevation_pattern": [
                [-30.0, -18.0],
                [-10.0, -5.0],
                [0.0, 0.0],
                [5.0, -1.5],
                [10.0, -4.0],
                [20.0, -10.0],
                [30.0, -18.0]
            ]
        }
        with open(sector5g_path, "w") as f:
            json.dump(sector5g, f, indent=2)
        print(f"[CONFIG] Created sector antenna config at {sector5g_path}")

def load_antenna_configs() -> Dict[str, Dict]:
    """
    Load all .json files from configs/ into a dict keyed by config name.
    Returns mapping name -> spec dict.
    """
    ensure_default_configs()
    specs = {}
    for fn in sorted(CONFIG_DIR.glob("*.json")):
        try:
            with open(fn, "r") as f:
                spec = json.load(f)
            name = spec.get("name") or fn.stem
            spec["__filename"] = str(fn)
            specs[name] = spec
            print(f"[CONFIG] Loaded antenna config '{name}' from {fn}")
        except Exception as e:
            print(f"[CONFIG] Failed to load {fn}: {e}")
    return specs

# ------------------------
# RF helpers
# ------------------------
C = 299792458.0  # speed of light m/s

def fspl_db(dist_m: float, freq_hz: float) -> float:
    """Free-space path loss in dB (distance in meters, freq in Hz)."""
    if dist_m <= 0.001:
        return 0.0
    return 20.0 * math.log10(4.0 * math.pi * dist_m * freq_hz / C)

def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest difference between two angles in degrees (0..180)."""
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d

def antenna_pattern_gain_db(tx: Dict, rx_x: float, rx_y: float) -> float:
    """
    Improved antenna pattern:
      - If antenna_spec contains 'azimuth_pattern' (list of [angle_deg, gain_db]),
        use it (interpolated with wrap-around).
      - Else use a Gaussian HPBW model (-3 dB at diff = HPBW/2).
      - Apply vertical taper using 'elevation_pattern' or 'vp_bw_deg'.
      - Respect fwd_back_db, sidelobe_db, and hard_null.
    tx may include 'tx_ground_z' and 'rx_ground_z' to make elevation computation accurate.
    """
    spec = tx.get("antenna_spec") if "antenna_spec" in tx else None

    def angdiff(a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)

    # compute bearing (0=east, 90=north), convert user heading (0=north) -> east-zero
    dx = float(rx_x) - float(tx["x"])
    dy = float(rx_y) - float(tx["y"])
    bearing = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    heading_user = float(tx.get("heading_deg", 0.0)) % 360.0
    heading_east0 = (heading_user + 90.0) % 360.0
    az_diff = angdiff(bearing, heading_east0)
    az_abs = bearing  # absolute azimuth to interpolate measured azimuth patterns

    # fallback legacy if no spec provided
    if not spec:
        ttype = str(tx.get("ant_type", "omni")).lower()
        peak_db = float(tx.get("antenna_gain_db", 0.0))
        if ttype == "omni":
            return peak_db
        beam = float(tx.get("beamwidth_deg", 90.0))
        half = max(1e-3, beam / 2.0)
        sidelobe_db = float(tx.get("sidelobe_db", -80.0))
        alpha = float(tx.get("pattern_sharpness", 2.5))
        if az_diff > half:
            if tx.get("hard_null", False):
                return peak_db - 120.0
            return peak_db + sidelobe_db
        theta = math.pi * az_diff / (2.0 * half)
        rel = math.cos(theta)
        rel = max(0.0, rel) ** alpha
        rel_linear = max(rel, 1e-12)
        rel_db = 10.0 * math.log10(rel_linear)
        return peak_db + rel_db

    # -----------------------
    # AZIMUTH (horizontal) GAIN
    # -----------------------
    peak_db = float(spec.get("gain_db", tx.get("antenna_gain_db", 0.0)))
    sidelobe_db = float(spec.get("sidelobe_db", -80.0))
    fwd_back_db = float(spec.get("fwd_back_db", spec.get("fbr_db", 20.0)))
    hard_null = bool(spec.get("hard_null", False))

    az_gain_db = None
    azpat = spec.get("azimuth_pattern")
    if azpat:
        # Accept either dict {"angles":[..],"gains":[..]} or list[[angle,gain], ...]
        if isinstance(azpat, dict) and "angles" in azpat and "gains" in azpat:
            angles = np.array(azpat["angles"], dtype=float) % 360.0
            gains = np.array(azpat["gains"], dtype=float)
        else:
            angles = []
            gains = []
            for p in azpat:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    angles.append(float(p[0]) % 360.0)
                    gains.append(float(p[1]))
            angles = np.array(angles, dtype=float)
            gains = np.array(gains, dtype=float)

        if angles.size >= 2:
            order = np.argsort(angles)
            angs = angles[order]
            g = gains[order]
            # extend for wrap-around
            angs_ext = np.concatenate([angs - 360.0, angs, angs + 360.0])
            g_ext = np.concatenate([g, g, g])
            az_gain_db = float(np.interp(az_abs, angs_ext, g_ext))
        else:
            az_gain_db = peak_db
    else:
        # Gaussian horizontal model (-3 dB at HPBW/2)
        hp_bw = float(spec.get("hp_bw_deg", spec.get("hp_bw", 90.0)))
        half = max(1e-3, hp_bw / 2.0)
        rel_linear = math.exp(-math.log(2.0) * (az_diff / half) ** 2)
        rel_linear = max(rel_linear, 1e-12)
        rel_db = 10.0 * math.log10(rel_linear)
        az_gain_db = peak_db + rel_db

    # Hard null
    hp_bw = float(spec.get("hp_bw_deg", spec.get("hp_bw", 90.0)))
    half = max(1e-3, hp_bw / 2.0)
    if hard_null and az_diff > half:
        return peak_db - 120.0

    # enforce floor based on forward/back or sidelobe spec
    min_allowed_db = peak_db - abs(fwd_back_db)
    sidelobe_floor_db = peak_db + sidelobe_db
    az_gain_db = max(az_gain_db, min_allowed_db, sidelobe_floor_db)

    # -----------------------
    # VERTICAL / ELEVATION TAPER
    # -----------------------
    # Use tx_ground_z and rx_ground_z if provided (set by caller)
    tx_ground_z = float(tx.get("tx_ground_z", 0.0))
    rx_ground_z = float(tx.get("rx_ground_z", 0.0))
    tx_h = float(tx.get("h_m", 2.0))
    tx_z_abs = tx_ground_z + tx_h
    # approximate rx z
    rx_z_abs = float(tx.get("rx_abs_z", rx_ground_z))
    dz = rx_z_abs - tx_z_abs
    horiz = math.hypot(dx, dy)
    if horiz <= 1e-6:
        elev_deg = 0.0
    else:
        elev_deg = math.degrees(math.atan2(dz, horiz))

    vert_rel_db = 0.0
    vpat = spec.get("elevation_pattern")
    if vpat:
        if isinstance(vpat, dict) and "angles" in vpat and "gains" in vpat:
            v_angles = np.array(vpat["angles"], dtype=float)
            v_gains = np.array(vpat["gains"], dtype=float)
        else:
            v_angles = []
            v_gains = []
            for p in vpat:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    v_angles.append(float(p[0]))
                    v_gains.append(float(p[1]))
            v_angles = np.array(v_angles, dtype=float) if v_angles else np.array([])
            v_gains = np.array(v_gains, dtype=float) if v_gains else np.array([])

        if v_angles.size >= 2:
            vert_val_db = float(np.interp(elev_deg, v_angles, v_gains))
            boresight_db = float(np.interp(0.0, v_angles, v_gains))
            vert_rel_db = vert_val_db - boresight_db
        else:
            vert_rel_db = 0.0
    else:
        vp_bw = float(spec.get("vp_bw_deg", spec.get("vp_bw", 90.0)))
        half_v = max(1e-3, vp_bw / 2.0)
        rel_linear_v = math.exp(-math.log(2.0) * (elev_deg / half_v) ** 2)
        rel_linear_v = max(rel_linear_v, 1e-12)
        vert_rel_db = 10.0 * math.log10(rel_linear_v)

    # -----------------------
    # COMBINE
    # -----------------------
    total_gain_db = az_gain_db + vert_rel_db
    return float(total_gain_db)

# ------------------------
# DB / point cloud loader (streaming + reservoir sampling for server-side)
# ------------------------
def load_pointcloud_from_db(db_path: str,
                           bbox: Optional[Tuple[float,float,float,float]] = None,
                           max_points: Optional[int] = None,
                           max_server_points: int = 1500000) -> Tuple[np.ndarray, List[Dict]]:
    """
    Stream rows from the DB and optionally reservoir-sample to max_points for client
    or to max_server_points for server use.
    - If max_points is provided, returns at most max_points rows (sampled).
    - Otherwise caps server-side returned points to max_server_points.
    Returns (points_array Nx3, metadata list) where points_array are the sampled points.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = None
    # speed up pragmas for bulk read (safe for read-only usage)
    try:
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA mmap_size = 30000000000;")
        conn.execute("PRAGMA journal_mode = OFF;")
    except Exception:
        pass

    c = conn.cursor()
    if bbox:
        min_x, min_y, max_x, max_y = bbox
        q = """
            SELECT p.id, p.easting, p.northing, p.altitude, p.classification
            FROM ProcessedLiDARPoints_idx AS idx
            JOIN ProcessedLiDARPoints AS p ON p.id = idx.id
            WHERE idx.min_x BETWEEN ? AND ? AND idx.min_y BETWEEN ? AND ?
        """
        params = (min_x, max_x, min_y, max_y)
    else:
        q = "SELECT id, easting, northing, altitude, classification FROM ProcessedLiDARPoints"
        params = ()

    try:
        c.execute(q, params)
    except Exception as e:
        conn.close()
        raise

    sample_limit = max_points if (max_points is not None) else max_server_points
    sampled_pts = []
    sampled_meta = []
    total_seen = 0
    fetch_size = 10000

    while True:
        rows = c.fetchmany(fetch_size)
        if not rows:
            break
        for r in rows:
            total_seen += 1
            pid, x, y, z, cls = r
            pt = (float(x), float(y), float(z))
            meta = {"id": int(pid), "class": cls}
            if len(sampled_pts) < sample_limit:
                sampled_pts.append(pt)
                sampled_meta.append(meta)
            else:
                j = random.randrange(total_seen)
                if j < sample_limit:
                    sampled_pts[j] = pt
                    sampled_meta[j] = meta

    conn.close()

    if not sampled_pts:
        return np.zeros((0,3)), []

    pts = np.array(sampled_pts, dtype=float)
    return pts, sampled_meta

# ------------------------
# Full-resolution stream endpoint generator (NDJSON)
# ------------------------
def stream_full_points_generator(db_path: str, bbox: Tuple[float,float,float,float], fetch_size: int = 10000):
    """
    Yield newline-delimited JSON lines: one point per line: {"x": x, "y": y, "z": z}
    Final line is metadata: {"_meta": {"centroid": [...], "bounds":[minx,maxx,miny,maxy,minz,maxz], "count": N}}
    """
    minx, miny, maxx, maxy = bbox
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    q = """
        SELECT p.easting, p.northing, p.altitude
        FROM ProcessedLiDARPoints_idx AS idx
        JOIN ProcessedLiDARPoints AS p ON p.id = idx.id
        WHERE idx.min_x BETWEEN ? AND ? AND idx.min_y BETWEEN ? AND ?
    """
    params = (minx, maxx, miny, maxy)
    try:
        c.execute(q, params)
    except Exception as e:
        conn.close()
        # yield no rows: error will be handled by response
        return

    total = 0
    sx = sy = sz = 0.0
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")

    while True:
        rows = c.fetchmany(fetch_size)
        if not rows:
            break
        for r in rows:
            x = float(r[0]); y = float(r[1]); z = float(r[2])
            total += 1
            sx += x; sy += y; sz += z
            xmin = min(xmin, x); ymin = min(ymin, y); zmin = min(zmin, z)
            xmax = max(xmax, x); ymax = max(ymax, y); zmax = max(zmax, z)
            yield (json.dumps({"x": x, "y": y, "z": z}) + "\n").encode('utf-8')

    if total > 0:
        centroid = [sx / total, sy / total, sz / total]
        bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
    else:
        centroid = [0.0, 0.0, 0.0]
        bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    yield (json.dumps({"_meta": {"centroid": centroid, "bounds": bounds, "count": total}}) + "\n").encode('utf-8')
    conn.close()

# ------------------------
# Occlusion test utilities (worker-side)
# ------------------------
def estimate_obstruction_loss_local(tx: Tuple[float,float,float],
                                    rx: Tuple[float,float,float],
                                    kdtree,
                                    pts_local: np.ndarray,
                                    sample_step_m: float = 1.0,
                                    blocker_radius_m: float = 0.5,
                                    per_blocker_db: float = 12.0) -> float:
    tx = np.array(tx)
    rx = np.array(rx)
    vec = rx - tx
    dist = np.linalg.norm(vec)
    if dist < 1e-3:
        return 0.0
    dirv = vec / dist
    n_samples = max(1, int(math.ceil(dist / sample_step_m)))
    blocker_count = 0
    for i in range(1, n_samples):
        t = i / float(n_samples)
        sample_pos = tx + dirv * (t * dist)
        expected_height = tx[2] + (rx[2] - tx[2]) * t
        idxs = kdtree.query_ball_point(sample_pos[:2], blocker_radius_m)
        if not idxs:
            continue
        heights = pts_local[idxs][:,2]
        if np.any(heights > expected_height + 0.25):
            blocker_count += 1
    if blocker_count == 0:
        return 0.0
    return per_blocker_db * math.sqrt(blocker_count)

# ------------------------
# Worker globals & initializer (updated to also accept grid arrays)
# ------------------------
_WORKER_PTS = None
_WORKER_KDTREE = None
_WORKER_GRID_X = None
_WORKER_GRID_Y = None

def _worker_init(pts: np.ndarray, grid_x=None, grid_y=None):
    """
    Worker initializer: each worker gets its own local copy reference and builds KD-tree.
    grid_x, grid_y are optional lists/arrays of grid coordinates so workers can map
    ix/iy -> world coordinates without receiving them per-task.
    """
    global _WORKER_PTS, _WORKER_KDTREE, _WORKER_GRID_X, _WORKER_GRID_Y
    _WORKER_PTS = pts
    _WORKER_GRID_X = None
    _WORKER_GRID_Y = None
    if grid_x is not None:
        # convert to Python list for safe pickling / memory sharing
        try:
            _WORKER_GRID_X = list(grid_x)
            _WORKER_GRID_Y = list(grid_y)
        except Exception:
            # fallback: convert via numpy
            _WORKER_GRID_X = list(np.array(grid_x).tolist())
            _WORKER_GRID_Y = list(np.array(grid_y).tolist())
    if pts is None or pts.size == 0:
        _WORKER_KDTREE = None
    else:
        xy = pts[:, :2]
        try:
            _WORKER_KDTREE = KDTree(xy)
        except Exception:
            _WORKER_KDTREE = KDTree(xy.tolist())

# ------------------------
# Worker computation (adjusted to use worker-global grid arrays)
# ------------------------
def _compute_rssi_chunk(chunk_args):
    (txs, nx, ny, idx_start, idx_end, default_freq_hz, tx_power_dbm, sample_step_m) = chunk_args
    pts_local = _WORKER_PTS
    kdtree_local = _WORKER_KDTREE
    grid_x_local = _WORKER_GRID_X
    grid_y_local = _WORKER_GRID_Y

    if pts_local is None or pts_local.shape[0] == 0 or kdtree_local is None:
        return (idx_start, [-999.0] * (idx_end - idx_start))
    if grid_x_local is None or grid_y_local is None:
        return (idx_start, [-999.0] * (idx_end - idx_start))

    rssi_out = []
    for flat_idx in range(idx_start, idx_end):
        iy = flat_idx // nx
        ix = flat_idx % nx
        # get world coords for this grid cell from worker-local lists
        try:
            gx = float(grid_x_local[ix])
            gy = float(grid_y_local[iy])
        except Exception:
            # defensive fallback to zeros if indexing fails
            gx = 0.0
            gy = 0.0

        try:
            _, nn = kdtree_local.query([gx, gy], k=1)
            z_ground = float(pts_local[int(nn)][2])
        except Exception:
            z_ground = float(np.median(pts_local[:,2]))
        best_rssi = -999.0
        for tx in txs:
            tx_x = float(tx["x"])
            tx_y = float(tx["y"])
            tx_h = float(tx.get("h_m", 2.0))
            tx_p = float(tx.get("power_dbm", tx_power_dbm))
            freq_hz = float(tx.get("freq_hz", default_freq_hz))
            try:
                _, tnn = kdtree_local.query([tx_x, tx_y], k=1)
                ground_tx_z = float(pts_local[int(tnn)][2])
            except Exception:
                ground_tx_z = float(np.median(pts_local[:,2]))
            tx_z = ground_tx_z + tx_h
            dxy = math.hypot(gx - tx_x, gy - tx_y)
            d3 = math.sqrt(max(1e-6, dxy**2 + (z_ground - tx_z)**2))
            fspl_val = fspl_db(d3, freq_hz)

            # pass ground heights into antenna pattern so elevation is meaningful
            tx_for_pattern = dict(tx)
            tx_for_pattern['tx_ground_z'] = ground_tx_z
            tx_for_pattern['rx_ground_z'] = z_ground
            tx_for_pattern['rx_abs_z'] = z_ground  # receiver assumed ground for grid points

            ant_gain = antenna_pattern_gain_db(tx_for_pattern, gx, gy)
            obs_loss = estimate_obstruction_loss_local((tx_x, tx_y, tx_z), (gx, gy, z_ground), kdtree_local, pts_local,
                                                       sample_step_m=sample_step_m,
                                                       blocker_radius_m=1.0,
                                                       per_blocker_db=12.0)
            rssi_val = tx_p + ant_gain - fspl_val - obs_loss
            if rssi_val > best_rssi:
                best_rssi = rssi_val
        rssi_out.append(float(best_rssi))
    return (idx_start, rssi_out)

# ------------------------
# Parallel orchestrator
# ------------------------
def compute_rssi_grid_parallel(txs: List[Dict],
                               grid_x: np.ndarray,
                               grid_y: np.ndarray,
                               default_freq_hz: float,
                               executor: ProcessPoolExecutor,
                               num_workers: int,
                               tx_power_dbm: float = 20.0,
                               sample_step_m: float = 2.0) -> np.ndarray:
    nx = len(grid_x)
    ny = len(grid_y)
    total = nx * ny
    if total == 0:
        return np.full((ny, nx), -999.0, dtype=float)
    n_tasks = max(num_workers * 2, num_workers)
    chunk_size = max(1, int(math.ceil(total / float(n_tasks))))
    tasks = []
    for i in range(0, total, chunk_size):
        start = i
        end = min(total, i + chunk_size)
        # reduced per-task args (grid arrays are provided in worker initializer)
        tasks.append((txs, nx, ny, start, end, float(default_freq_hz), float(tx_power_dbm), float(sample_step_m)))
    futures = [executor.submit(_compute_rssi_chunk, t) for t in tasks]
    rssi_flat = [-999.0] * total

    completed = 0
    total_futs = len(futures)
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except Exception:
        loop = None

    for fut in as_completed(futures):
        try:
            idx_start, rlist = fut.result()
            rssi_flat[idx_start:idx_start + len(rlist)] = rlist
        except Exception as e:
            print("Worker failed:", e)
        completed += 1
        # compute percent
        try:
            percent = int((completed / float(total_futs)) * 100.0)
        except Exception:
            percent = 0
        # schedule non-blocking broadcast of progress
        try:
            if loop and loop.is_running():
                loop.create_task(broadcast_message(json.dumps({"type": "sim_progress", "percent": percent})))
        except Exception:
            pass

    # final 100% broadcast
    try:
        if loop and loop.is_running():
            loop.create_task(broadcast_message(json.dumps({"type": "sim_progress", "percent": 100})))
    except Exception:
        pass

    rss_grid = np.array(rssi_flat, dtype=float).reshape((ny, nx))
    return rss_grid

# ------------------------
# FastAPI app + websockets (UI unchanged from prior)
# ------------------------
app = FastAPI()
# GZip for large transfers
app.add_middleware(GZipMiddleware, minimum_size=500)

connected_websockets: List[WebSocket] = []

DB_PATH = None

# Full HTML page (client)
# Important: this HTML listens for sim_progress and streams /pointcloud_full progressively.
HTML_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>LiDAR RF Viewer (on-demand + antenna)</title>
  <style>
    body { margin:0; overflow:hidden; background:#0b0b0b; color:#eee; font-family: Arial, sans-serif; }
    #overlay { position:absolute; top:8px; left:8px; z-index:20; background:rgba(255,255,255,0.95); color:#000; padding:10px; border-radius:6px; width:420px; max-height:92vh; overflow:auto; }
    #legend { margin-top:8px; padding:6px; background:#fff; border-radius:4px; font-size:12px; color:#000; }
    .color-bar { height:14px; width:100%; border-radius:4px; background: linear-gradient(to right, rgb(200,0,0), rgb(255,200,0), rgb(0,180,0)); display:block; margin-bottom:6px; }
    .legend-row { display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px; }
    label { display:inline-block; width:160px; font-size:13px; vertical-align: middle; }
    input[type="number"], input[type="text"], select { width:220px; }
    .config-row { margin-bottom:6px; display:flex; align-items:center; }
    .config-row input[type="checkbox"] { width: auto; margin-left:0; margin-right:6px; transform:scale(1.2); }
    .config-note { font-size:11px; color:#333; margin-top:6px; }
    button { margin-top:6px; }
    .tx-group { margin-top:8px; padding:6px; border-radius:6px; background: #f3f3f3; color:#000; }
    .tx-group label { color: #000; }
    #progressBarContainer { width:100%; background:#ddd; border-radius:6px; overflow:hidden; margin-top:6px; display:none; }
    #progressBar { height:12px; width:0%; background:#4caf50; }
  </style>
</head>
<body>
<div id="overlay">
  <div><b>LiDAR RF Viewer — On-demand</b></div>

  <div style="margin-top:8px;">
    <label>UTM Easting</label><input id="utm_e" type="number" step="0.01" value="0"/><br/>
    <label>UTM Northing</label><input id="utm_n" type="number" step="0.01" value="0"/><br/>
    <label>Radius (m)</label><input id="utm_r" type="number" step="1" value="100"/><br/>
    <div style="margin-top:6px;">
      <button id="loadArea">Load area (stream full res)</button>
      <button id="clearTx">Clear TX</button>
    </div>
  </div>

  <hr/>

  <div>
    <label>Grid resolution (m)</label><input id="grid_res" type="number" step="0.1" value="6.0"/>
    <br/>
    <label>Grid margin (m)</label><input id="grid_margin" type="number" step="1" value="40.0"/>
    <div style="margin-top:6px;">
      <button id="setGrid">Set grid</button>
    </div>
  </div>

  <hr/>

  <div>
    <label>Antenna config</label>
    <select id="ant_config" style="width:220px;">
      <option value="__loading__">Loading...</option>
    </select><br/>
    <div id="ant_config_editor" style="margin-top:8px;">
      <!-- Dynamically generated config fields will appear here -->
      <div class="config-note">Select an antenna config to edit its fields (the editor mirrors the config file).</div>
    </div>

    <div class="tx-group">
      <div class="config-row"><label for="txp">TX Power (dBm)</label><input id="txp" type="number" step="0.1" value="20"/></div>
      <div class="config-row"><label for="freq">Frequency</label>
        <select id="freq" style="width:220px;">
          <option value="900e6">900 MHz</option>
          <option value="2.4e9">2.4 GHz</option>
          <option value="5.8e9">5.8 GHz</option>
        </select>
      </div>
      <div class="config-row"><label for="txh">TX Height (m)</label><input id="txh" type="number" step="0.1" value="2"/></div>
      <div class="config-row"><label for="heading">Heading (deg)</label><input id="heading" type="number" step="1" value="0"/></div>
    </div>

    <div style="margin-top:6px;">
      <button id="addTx">Add transmitter (click scene)</button>
      <button id="simulate">Simulate</button>
    </div>
  </div>

  <hr/>

  <div><b>Clients:</b> <span id="clients">0</span></div>
  <div><b>Loaded points:</b> <span id="ptcount">0</span></div>

  <div id="progressBarContainer"><div id="progressBar"></div></div>

  <div id="legend">
    <div class="color-bar"></div>
    <div class="legend-row"><span><b>Green</b> — Strong (≥ -60 dBm)</span><span>-60 dBm</span></div>
    <div class="legend-row"><span><b>Yellow</b> — Questionable (-80 to -60 dBm)</span><span>-80 dBm</span></div>
    <div class="legend-row"><span><b>Red</b> — Poor (-100 to -80 dBm)</span><span>-100 dBm</span></div>
    <div class="legend-row"><span><b>Black</b> — No signal (≤ -100 dBm)</span><span>&lt;= -100</span></div>
  </div>
</div>

<div id="errbox" style="display:none; position:absolute; bottom:8px; left:8px; right:8px; max-height:160px; overflow:auto; background:rgba(0,0,0,0.85); color:#f88; padding:8px; font-family:monospace; z-index:30; border-radius:6px;"></div>

<script src="https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.min.js"></script>

<script>
// Client-side JS with heading 0 -> up (+Y).

function showError(msg){
  const box = document.getElementById('errbox');
  box.style.display = 'block';
  box.innerText = (box.innerText ? (box.innerText + "\n\n") : "") + msg;
  console.error(msg);
}

let scene, camera, renderer, pcPoints, heatPoints;
let pointCloud = [], centroid = [0,0,0], bounds = [0,0,0,0,0,0];
let ws;
let adding = false;
let txVisuals = [];
let antennaSpecsCache = {}; // name -> full spec object

async function loadAntennaConfigs(){
  try{
    const res = await fetch('/antenna_configs');
    if(!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const sel = document.getElementById('ant_config');
    sel.innerHTML = '';
    if(data.length === 0){
      sel.innerHTML = '<option value="__none__">-- no configs --</option>';
      renderAntConfigEditor(null);
      return;
    }
    for(const c of data){
      const opt = document.createElement('option');
      opt.value = c.name;
      opt.innerText = `${c.name} (${c.type || 'unknown'})`;
      sel.appendChild(opt);
    }
    // fetch and cache full specs for select's options
    for(const c of data){
      try{
        const r = await fetch(`/antenna_config/${encodeURIComponent(c.name)}`);
        if(r.ok){
          const spec = await r.json();
          antennaSpecsCache[c.name] = spec;
        } else {
          antennaSpecsCache[c.name] = { name: c.name, model: c.model || '', type: c.type || '', note: 'failed to fetch full config' };
        }
      }catch(e){
        antennaSpecsCache[c.name] = { name: c.name, model: c.model || '', type: c.type || '', note: 'fetch error' };
      }
    }
    // Set selection to first and render editor
    sel.selectedIndex = 0;
    renderAntConfigEditor(antennaSpecsCache[sel.value]);
    sel.onchange = ()=>{
      const name = sel.value;
      renderAntConfigEditor(antennaSpecsCache[name] || null);
    };
  }catch(err){
    console.warn('Failed to load antenna configs:', err);
    const sel = document.getElementById('ant_config');
    sel.innerHTML = '<option value="builtin_omni">builtin_omni</option>';
    renderAntConfigEditor(null);
  }
}

function renderAntConfigEditor(spec){
  const container = document.getElementById('ant_config_editor');
  container.innerHTML = '';
  if(!spec){
    container.innerHTML = '<div class="config-note">No antenna config available. You may add JSON files to the server configs/ directory and restart the server.</div>';
    return;
  }
  const keys = Object.keys(spec).filter(k => k !== '__filename').sort();
  for(const k of keys){
    const val = spec[k];
    const row = document.createElement('div');
    row.className = 'config-row';
    const lbl = document.createElement('label');
    lbl.innerText = k;
    lbl.setAttribute('for', `cfg_${k}`);
    row.appendChild(lbl);

    let input;
    if(typeof val === 'boolean'){
      input = document.createElement('input');
      input.type = 'checkbox';
      input.id = `cfg_${k}`;
      input.checked = !!val;
      input.dataset.type = 'boolean';
      if(k === 'name' || k === 'model' || k === 'type') input.disabled = true;
      const wrap = document.createElement('div');
      wrap.style.display = 'flex';
      wrap.style.alignItems = 'center';
      wrap.appendChild(input);
      row.appendChild(wrap);
    } else if(typeof val === 'number'){
      input = document.createElement('input');
      input.type = 'number';
      input.id = `cfg_${k}`;
      const step = Math.abs(val) >= 1000 ? '1' : (Math.abs(val) >= 1 ? '0.1' : '0.001');
      input.step = step;
      input.value = val;
      input.dataset.type = 'number';
      if(k === 'name' || k === 'model' || k === 'type') input.disabled = true;
      row.appendChild(input);
    } else {
      input = document.createElement('input');
      input.type = 'text';
      input.id = `cfg_${k}`;
      input.value = val === null || val === undefined ? '' : String(val);
      input.dataset.type = 'string';
      if(k === 'name' || k === 'model' || k === 'type') input.disabled = true;
      row.appendChild(input);
    }
    container.appendChild(row);
  }

  const note = document.createElement('div');
  note.className = 'config-note';
  note.innerText = 'Fields above are direct reflections of the config file. Editable values will be merged into antenna_spec for each transmitter.';
  container.appendChild(note);
}

// safe helper to gather editor values (not used directly, but available)
function getEditorValuesForSpec(){
  const container = document.getElementById('ant_config_editor');
  const inputs = container.querySelectorAll('[id^="cfg_"]');
  const out = {};
  inputs.forEach(inp=>{
    const k = inp.id.replace(/^cfg_/, '');
    const t = inp.dataset.type || 'string';
    if(t === 'boolean'){
      out[k] = !!inp.checked;
    } else if(t === 'number'){
      const v = inp.value;
      out[k] = v === '' ? null : parseFloat(v);
    } else {
      out[k] = inp.value;
    }
  });
  return out;
}

function safeGet(url){
  return fetch(url).then(r => {
    if(!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText} for ${url}`);
    return r.json();
  });
}

function rssi_to_rgb(rssi){
  if(rssi <= -100) return [0,0,0];
  const t = Math.min(1, Math.max(0, (rssi + 100) / 40.0));
  let r=0,g=0,b=0;
  if(t < 0.5){
    const u = t / 0.5;
    r = 1.0; g = u; b = 0.0;
  } else {
    const u = (t - 0.5) / 0.5;
    r = 1.0 - u; g = 1.0; b = 0.0;
  }
  return [r,g,b];
}

function createTxViz(tx) {
  // heading convention: 0 => +Y (up). direction vector = (sin(h), cos(h))
  const cx = centroid[0] || 0, cy = centroid[1] || 0;
  const group = new THREE.Group();

  // small marker sphere at the TX location
  const sGeom = new THREE.SphereGeometry(1.2, 12, 8);
  const sMat = new THREE.MeshBasicMaterial({color: 0x00ff00});
  const s = new THREE.Mesh(sGeom, sMat);
  s.position.set(tx.x - cx, tx.y - cy, tx.h_m + 0.5);
  group.add(s);

  // arrow at apex, pointing in heading direction
  const headingRad = (tx.heading_deg || 0) * Math.PI / 180.0;
  const dir = new THREE.Vector3(Math.sin(headingRad), Math.cos(headingRad), 0).normalize();
  const apexX = tx.x - cx;
  const apexY = tx.y - cy;
  const apexZ = tx.h_m + 0.5;
  const origin = new THREE.Vector3(apexX, apexY, apexZ);
  const arrowLen = 40;
  const arrow = new THREE.ArrowHelper(dir, origin, Math.min(arrowLen, 80), 0xff0000, 4, 2);
  group.add(arrow);

  group.userData.tx = tx;
  return group;
}

async function init(){
  try{
    await loadAntennaConfigs();

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0b0b);
    camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 1000000);
    camera.up.set(0,0,1);
    renderer = new THREE.WebGLRenderer({antialias:true});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(innerWidth, innerHeight);
    document.body.appendChild(renderer.domElement);
    window.addEventListener('resize', ()=>{ camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const dl = new THREE.DirectionalLight(0xffffff, 0.2); dl.position.set(1,-1,1); scene.add(dl);

    centroid = [0,0,0];
    bounds = [0,0,0,0,0,0];
    document.getElementById('ptcount').innerText = '0';

    // heat points for simulation output
    heatPoints = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({size: 2, vertexColors:true}));
    scene.add(heatPoints);

    const gridHelper = new THREE.GridHelper(100, 10, 0x222222, 0x151515);
    gridHelper.rotation.x = Math.PI/2;
    scene.add(gridHelper);
    scene.add(new THREE.AxesHelper(5));

    camera.position.set(0, -50, 30);
    camera.lookAt(0,0,0);

    let dragging = false, lastX=0, lastY=0;
    renderer.domElement.addEventListener('mousedown', (e)=>{ dragging=true; lastX=e.clientX; lastY=e.clientY; });
    window.addEventListener('mousemove', (e)=>{ if(!dragging) return; const dx=(e.clientX-lastX)/200; const dy=(e.clientY-lastY)/200; lastX=e.clientX; lastY=e.clientY; camera.position.applyAxisAngle(new THREE.Vector3(0,0,1), dx); camera.position.z += dy*10; camera.lookAt(0,0,0); });
    window.addEventListener('mouseup', ()=>dragging=false);
    renderer.domElement.addEventListener('wheel', (ev)=>{ ev.preventDefault(); const z = camera.position.z * (1 + Math.sign(ev.deltaY)*0.05); camera.position.z = Math.max(1, Math.min(1e7, z)); camera.lookAt(0,0,0); }, {passive:false});

    (function animate(){ requestAnimationFrame(animate); renderer.render(scene, camera); })();

    ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://') + location.host + '/ws');
    ws.onopen = ()=>console.log('WS open');
    ws.onmessage = (evt)=>{ try{ const m = JSON.parse(evt.data);
        if(m.type === 'clients') document.getElementById('clients').innerText = m.count;
        else if(m.type === 'heatmap') renderHeatmap(m.grid);
        else if(m.type === 'sim_progress') updateSimProgress(m.percent);
    }catch(e){ showError('WS parse error: ' + e.message); } };
    ws.onclose = ()=>console.log('WS closed');
    ws.onerror = (e)=> showError('WS error');

    document.getElementById('addTx').onclick = ()=>{ adding=true; document.getElementById('addTx').innerText='Click scene to place TX'; };
    document.getElementById('clearTx').onclick = async ()=>{ try{ await fetch('/clear_txs', {method:'POST'}); txVisuals.forEach(g => scene.remove(g)); txVisuals = []; console.log('Cleared TXs and visuals'); }catch(e){ showError('clear txs failed'); } };

    document.getElementById('loadArea').onclick = async ()=>{
      const e = parseFloat(document.getElementById('utm_e').value || '0');
      const n = parseFloat(document.getElementById('utm_n').value || '0');
      const r = parseFloat(document.getElementById('utm_r').value || '100');
      try{
        document.getElementById('ptcount').innerText = 'loading...';
        // Call load_bbox to prepare server KD-tree and return client sample metadata + cached sample points
        const res = await fetch('/load_bbox', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({cx: e, cy: n, r: r})
        });
        if(!res.ok) throw new Error(`HTTP ${res.status}`);
        const info = await res.json();
        // Start streaming the full-resolution NDJSON
        streamFullPointcloud();
        // if server returned a quick client sample, render it immediately to give instant feedback
        if(info.client_points && Array.isArray(info.client_points) && info.client_points.length > 0){
          await updatePointCloudFromJSON({"points": info.client_points, "centroid": info.centroid, "bounds": info.bounds});
          document.getElementById('ptcount').innerText = info.client_sample_count || info.count || 'loaded';
        } else {
          // leave existing points visible until streaming fills replacement
          document.getElementById('ptcount').innerText = info.count || 'loading';
        }
      }catch(err){
        showError('Load area failed: ' + (err && err.message ? err.message : String(err)));
      }
    };

    document.getElementById('setGrid').onclick = async ()=>{
      const gr = parseFloat(document.getElementById('grid_res').value || '6.0');
      const gm = parseFloat(document.getElementById('grid_margin').value || '40.0');
      try{
        const res = await fetch('/set_grid', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({grid_res:gr, grid_margin:gm})});
        if(!res.ok) throw new Error(`HTTP ${res.status}`);
        const info = await res.json();
        console.log('Grid updated', info);
        heatPoints.geometry = new THREE.BufferGeometry();
      }catch(err){
        showError('Set grid failed: ' + (err && err.message ? err.message : String(err)));
      }
    };

    renderer.domElement.addEventListener('click', async (ev)=>{
      if(!adding) return;
      adding=false;
      document.getElementById('addTx').innerText='Add transmitter (click scene)';
      const rect = renderer.domElement.getBoundingClientRect();
      const mx = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      const my = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      const vec = new THREE.Vector3(mx, my, 0.5).unproject(camera);
      const dir = vec.clone().sub(camera.position).normalize();
      const t = - camera.position.z / dir.z;
      const intersect = camera.position.clone().add(dir.multiplyScalar(t));
      const worldX = intersect.x + (centroid[0] || 0);
      const worldY = intersect.y + (centroid[1] || 0);

      const h = parseFloat(document.getElementById('txh') ? document.getElementById('txh').value : '2') || 2;
      const p = parseFloat(document.getElementById('txp') ? document.getElementById('txp').value : '20') || 20;
      const freqEl = document.getElementById('freq');
      const freq = freqEl ? parseFloat(freqEl.value) : 2.4e9;
      const heading = parseFloat(document.getElementById('heading') ? document.getElementById('heading').value : '0') || 0;

      const sel = document.getElementById('ant_config');
      const ant_config = sel ? sel.value : null;
      let antenna_spec = null;
      if(ant_config && antennaSpecsCache[ant_config]){
        antenna_spec = Object.assign({}, antennaSpecsCache[ant_config]);
        const container = document.getElementById('ant_config_editor');
        const inputs = container.querySelectorAll('[id^="cfg_"]');
        inputs.forEach(inp=>{
          const k = inp.id.replace(/^cfg_/, '');
          if(inp.dataset.type === 'boolean'){
            antenna_spec[k] = !!inp.checked;
          } else if(inp.dataset.type === 'number'){
            const v = inp.value;
            antenna_spec[k] = v === '' ? antenna_spec[k] : parseFloat(v);
          } else {
            antenna_spec[k] = inp.value;
          }
        });
      } else {
        antenna_spec = null;
      }

      // Build TX object including antenna_spec (if any)
      const tx = {
        x: worldX, y: worldY, h_m: h, power_dbm: p, freq_hz: freq,
        ant_config: ant_config,
        heading_deg: heading
      };
      if(antenna_spec) tx["antenna_spec"] = antenna_spec;

      try{
        const res = await fetch('/add_tx', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify(tx)
        });
        const text = await res.text();
        let parsed = null;
        try { parsed = JSON.parse(text); } catch(_) { parsed = null; }

        if(!res.ok){
          const msg = parsed && parsed.error ? parsed.error : (text || `HTTP ${res.status}`);
          showError(`Add TX failed: ${msg}`);
          console.error('Add TX failed', res.status, text);
        } else {
          console.log('Add TX response:', parsed || text);
          const vis = createTxViz(tx);
          scene.add(vis);
          txVisuals.push(vis);
        }
      }catch(e){
        showError('Add TX network / parse error: ' + (e && e.message ? e.message : String(e)));
        console.error('Add TX exception', e);
      }
    });

    document.getElementById('simulate').onclick = async ()=>{
      try{
        document.getElementById('simulate').disabled = true;
        // show progress UI
        document.getElementById('progressBarContainer').style.display = 'block';
        document.getElementById('progressBar').style.width = '0%';
        const res = await fetch('/simulate', {method:'POST'});
        if(!res.ok){
          const txt = await res.text();
          showError('Simulate failed: ' + txt);
          document.getElementById('simulate').disabled = false;
          return;
        }
        const payload = await res.json();
        if(payload && payload.grid) renderHeatmap(payload.grid);
        document.getElementById('simulate').disabled = false;
        // hide progress shortly after completion
        setTimeout(()=>{ document.getElementById('progressBarContainer').style.display = 'none'; }, 1200);
      }catch(err){
        showError('Simulate failed: ' + (err && err.message ? err.message : String(err)));
        document.getElementById('simulate').disabled = false;
        document.getElementById('progressBarContainer').style.display = 'none';
      }
    };

  }catch(err){
    showError('Init error: ' + (err && err.message ? err.message : String(err)));
  }
}

// Update or create point cloud from a JSON payload like /pointcloud (sampled)
async function updatePointCloudFromJSON(json){
  try{
    if(!json || !Array.isArray(json.points)) throw new Error('Invalid /pointcloud response');
    pointCloud = json.points;
    centroid = json.centroid || [0,0,0];
    bounds = json.bounds || [0,0,0,0,0,0];
    const positions = new Float32Array(pointCloud.length * 3);
    for(let i=0;i<pointCloud.length;i++){
      positions[i*3+0] = pointCloud[i].x;
      positions[i*3+1] = pointCloud[i].y;
      positions[i*3+2] = pointCloud[i].z;
    }
    if(pcPoints){
      scene.remove(pcPoints);
      if(pcPoints.geometry) pcPoints.geometry.dispose();
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const spanX = bounds[1] - bounds[0];
    const spanY = bounds[3] - bounds[2];
    const spanZ = bounds[5] - bounds[4];
    const span = Math.max(1, spanX, spanY, spanZ);
    const size = Math.max(0.6, span / 2000);
    const mat = new THREE.PointsMaterial({size:size, sizeAttenuation:true, color:0xffffff});
    pcPoints = new THREE.Points(geom, mat);
    scene.add(pcPoints);
    document.getElementById('ptcount').innerText = pointCloud.length;
  }catch(e){
    showError('updatePointCloudFromJSON error: ' + (e && e.message ? e.message : String(e)));
  }
}

// STREAM full-resolution NDJSON and update display progressively (no flash).
async function streamFullPointcloud(){
  try{
    const resp = await fetch('/pointcloud_full');
    if(!resp.ok){
      const txt = await resp.text();
      showError('/pointcloud_full failed: ' + txt);
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let { value: chunk, done: readerDone } = await reader.read();
    let buffer = '';
    // We'll accumulate new points in arrays, and only replace the visible cloud once we have enough
    const newPositions = [];
    const newMeta = { count: 0, centroid: [0,0,0], bounds: [0,0,0,0,0,0] };
    // create a temporary Points object but don't add it to the scene until it's sizeable
    let tempPoints = null;
    let tempMaterial = null;
    // threshold: wait until we have at least 2000 points before swapping to avoid flashing
    const SWAP_THRESHOLD = 2000;
    const FLUSH_CHUNK = 1000; // how many points to buffer between geometry updates
    let flushCounter = 0;
    // streaming loop
    while(!readerDone){
      if(chunk){
        buffer += decoder.decode(chunk, {stream: true});
        let nl;
        while((nl = buffer.indexOf('\n')) >= 0){
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl+1);
          if(line.length === 0) continue;
          let obj = null;
          try{
            obj = JSON.parse(line);
          }catch(e){
            console.warn('json parse line', e, line);
            continue;
          }
          if(obj._meta){
            // final metadata (centroid/bounds/count)
            newMeta.count = obj._meta.count || newPositions.length;
            newMeta.centroid = obj._meta.centroid || newMeta.centroid;
            newMeta.bounds = obj._meta.bounds || newMeta.bounds;
            // finished streaming - perform final swap if tempPoints exists
            if(tempPoints && newPositions.length > 0){
              // finalize geometry
              const positions = new Float32Array(newPositions.length * 3);
              for(let i=0;i<newPositions.length;i++){
                positions[i*3+0] = newPositions[i][0] - newMeta.centroid[0];
                positions[i*3+1] = newPositions[i][1] - newMeta.centroid[1];
                positions[i*3+2] = newPositions[i][2] - newMeta.centroid[2];
              }
              const geom = new THREE.BufferGeometry();
              geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
              // keep same size calculation as updatePointCloudFromJSON
              const spanX = newMeta.bounds[1] - newMeta.bounds[0];
              const spanY = newMeta.bounds[3] - newMeta.bounds[2];
              const spanZ = newMeta.bounds[5] - newMeta.bounds[4];
              const span = Math.max(1, spanX, spanY, spanZ);
              const size = Math.max(0.6, span / 2000);
              const mat = new THREE.PointsMaterial({size:size, sizeAttenuation:true, color:0xffffff});
              const newPc = new THREE.Points(geom, mat);
              // Add newPc and remove old pcPoints if present
              scene.add(newPc);
              if(pcPoints){
                scene.remove(pcPoints);
                if(pcPoints.geometry) pcPoints.geometry.dispose();
              }
              pcPoints = newPc;
              centroid = newMeta.centroid;
              bounds = newMeta.bounds;
              document.getElementById('ptcount').innerText = newPositions.length;
            }
            // done
            return;
          } else {
            // regular point object
            if(typeof obj.x === 'number' && typeof obj.y === 'number' && typeof obj.z === 'number'){
              newPositions.push([obj.x, obj.y, obj.z]);
              flushCounter++;
            }
            // If this is the first time we've accumulated enough points, create the tempPoints object
            if(tempPoints === null && newPositions.length >= Math.min(SWAP_THRESHOLD, FLUSH_CHUNK)){
              // prepare partial geometry (we'll update it periodically)
              const initialCount = Math.min(newPositions.length, newPositions.length);
              const positions = new Float32Array(initialCount * 3);
              for(let i=0;i<initialCount;i++){
                positions[i*3+0] = newPositions[i][0];
                positions[i*3+1] = newPositions[i][1];
                positions[i*3+2] = newPositions[i][2];
              }
              const geom = new THREE.BufferGeometry();
              geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
              tempMaterial = new THREE.PointsMaterial({size: 0.6, sizeAttenuation:true, color:0xcccccc});
              tempPoints = new THREE.Points(geom, tempMaterial);
              // Do not remove old pcPoints yet — keep both visible to avoid flashing.
              scene.add(tempPoints);
              // center adjustment will be applied on finalization; for now display relative positions anchored to previous centroid (may be slightly off)
            }
            // periodically flush incremental updates to tempPoints to give visual feedback
            if(tempPoints && flushCounter >= FLUSH_CHUNK){
              // expand tempPoints geometry to include newPoints
              const curLen = tempPoints.geometry.getAttribute('position').array.length / 3;
              const addLen = Math.max(0, newPositions.length - curLen);
              if(addLen > 0){
                const newArr = new Float32Array((curLen + addLen) * 3);
                newArr.set(tempPoints.geometry.getAttribute('position').array, 0);
                // write added points
                for(let i=0;i<addLen;i++){
                  const p = newPositions[curLen + i];
                  newArr[(curLen + i)*3 + 0] = p[0];
                  newArr[(curLen + i)*3 + 1] = p[1];
                  newArr[(curLen + i)*3 + 2] = p[2];
                }
                tempPoints.geometry.setAttribute('position', new THREE.BufferAttribute(newArr, 3));
                tempPoints.geometry.attributes.position.needsUpdate = true;
              }
              flushCounter = 0;
              // if we've reached the SWAP_THRESHOLD, finalize swap
              if(newPositions.length >= SWAP_THRESHOLD){
                // Build final geometry and swap
                const positions = new Float32Array(newPositions.length * 3);
                // center to centroid when meta is available; if not yet available, center to first-pass centroid
                let cx = centroid[0] || 0;
                let cy = centroid[1] || 0;
                let cz = centroid[2] || 0;
                for(let i=0;i<newPositions.length;i++){
                  positions[i*3+0] = newPositions[i][0] - cx;
                  positions[i*3+1] = newPositions[i][1] - cy;
                  positions[i*3+2] = newPositions[i][2] - cz;
                }
                const geom = new THREE.BufferGeometry();
                geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                const span = 1;
                const size = Math.max(0.6, span / 2000);
                const mat = new THREE.PointsMaterial({size:size, sizeAttenuation:true, color:0xffffff});
                const newPc = new THREE.Points(geom, mat);
                scene.add(newPc);
                // remove old pcPoints
                if(pcPoints){
                  scene.remove(pcPoints);
                  if(pcPoints.geometry) pcPoints.geometry.dispose();
                }
                // remove tempPoints
                if(tempPoints){
                  scene.remove(tempPoints);
                  if(tempPoints.geometry) tempPoints.geometry.dispose();
                  tempPoints = null;
                  tempMaterial = null;
                }
                pcPoints = newPc;
                // update displayed count
                document.getElementById('ptcount').innerText = newPositions.length;
              }
            }
          }
        }
      }
      ({ value: chunk, done: readerDone } = await reader.read());
    }

    // If we exit loop without meta, finalize best-effort
    if(newPositions.length > 0){
      const positions = new Float32Array(newPositions.length * 3);
      for(let i=0;i<newPositions.length;i++){
        positions[i*3+0] = newPositions[i][0] - (centroid[0] || 0);
        positions[i*3+1] = newPositions[i][1] - (centroid[1] || 0);
        positions[i*3+2] = newPositions[i][2] - (centroid[2] || 0);
      }
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const mat = new THREE.PointsMaterial({size:0.6, sizeAttenuation:true, color:0xffffff});
      const newPc = new THREE.Points(geom, mat);
      scene.add(newPc);
      if(pcPoints){
        scene.remove(pcPoints);
        if(pcPoints.geometry) pcPoints.geometry.dispose();
      }
      pcPoints = newPc;
      document.getElementById('ptcount').innerText = newPositions.length;
    }

  }catch(e){
    showError('streamFullPointcloud failed: ' + (e && e.message ? e.message : String(e)));
  }
}

function updateSimProgress(percent){
  const pct = Math.max(0, Math.min(100, percent||0));
  const bar = document.getElementById('progressBar');
  const container = document.getElementById('progressBarContainer');
  container.style.display = 'block';
  bar.style.width = pct + '%';
  if(pct >= 100){
    setTimeout(()=>{ container.style.display = 'none'; bar.style.width = '0%'; }, 800);
  }
}

function renderHeatmap(grid){
  try{
    const xs = grid.xs, ys = grid.ys, rssi = grid.rssi;
    const nx = xs.length, ny = ys.length;
    const pos = new Float32Array(nx*ny*3);
    const col = new Float32Array(nx*ny*3);
    let k=0;
    for(let j=0;j<ny;j++){
      for(let i=0;i<nx;i++){
        const x = xs[i] - (centroid[0] || 0);
        const y = ys[j] - (centroid[1] || 0);
        pos[k*3+0]=x; pos[k*3+1]=y; pos[k*3+2]=0.15;
        const v = rssi[j*nx + i];
        const rgb = rssi_to_rgb(v);
        col[k*3+0]=rgb[0]; col[k*3+1]=rgb[1]; col[k*3+2]=rgb[2];
        k++;
      }
    }
    if(heatPoints.geometry && heatPoints.geometry.dispose) heatPoints.geometry.dispose();
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
    heatPoints.geometry = geom;
    heatPoints.material.vertexColors = true;
  }catch(e){
    showError('Heatmap render error: ' + e.message);
  }
}

window.addEventListener('error', (ev)=> showError(`Global error: ${ev.message} at ${ev.filename}:${ev.lineno}`));
window.addEventListener('unhandledrejection', (ev)=> showError('Unhandled promise rejection: ' + (ev.reason && ev.reason.message ? ev.reason.message : String(ev.reason))));

init();
</script>
</body>
</html>
"""  # end HTML_PAGE

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.get("/pointcloud")
async def pointcloud():
    """
    Return current server_state point subset (centered for viewer) along with metadata.
    This is a sampled client-friendly payload (fast). For full resolution stream, use /pointcloud_full.
    """
    global server_state
    pts = server_state["pts"]
    if pts is None or pts.shape[0] == 0:
        return JSONResponse({"points": [], "centroid": [0,0,0], "bounds": [0,0,0,0,0,0]})
    xs = pts[:,0]; ys = pts[:,1]; zs = pts[:,2]
    cx = float(xs.mean()); cy = float(ys.mean()); cz = float(zs.mean())
    bounds = [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()), float(zs.min()), float(zs.max())]
    # by default send at most 120k points as sample
    max_client = 120000
    if pts.shape[0] > max_client:
        idxs = np.linspace(0, pts.shape[0]-1, max_client, dtype=int)
        sample = pts[idxs]
    else:
        sample = pts
    centered = [{"x": float(x - cx), "y": float(y - cy), "z": float(z - cz)} for (x, y, z) in sample]
    return JSONResponse({"points": centered, "centroid": [cx, cy, cz], "bounds": bounds})

@app.get("/pointcloud_full")
async def pointcloud_full():
    """
    Stream the full-resolution point cloud for the last loaded bbox as NDJSON.
    Each line: {"x":..., "y":..., "z":...}
    Final line: {"_meta": {"centroid": [...], "bounds":[minx,maxx,miny,maxy,minz,maxz], "count": N}}
    """
    global server_state, DB_PATH
    bbox = server_state.get("last_bbox")
    if not bbox:
        return JSONResponse({"error": "no bbox loaded - call /load_bbox first"}, status_code=400)
    gen = stream_full_points_generator(DB_PATH, bbox)
    return StreamingResponse(gen, media_type="application/x-ndjson")

# --- endpoints for antenna configs and reload ---
@app.get("/antenna_configs")
async def antenna_configs():
    """
    Return a summary list of available antenna configs for the client UI.
    """
    global server_state
    out = []
    for name, spec in server_state.get("antenna_configs", {}).items():
        out.append({
            "name": name,
            "model": spec.get("model", ""),
            "type": spec.get("type", ""),
            "gain_db": spec.get("gain_db"),
            "hp_bw_deg": spec.get("hp_bw_deg"),
            "vp_bw_deg": spec.get("vp_bw_deg"),
            "fwd_back_db": spec.get("fwd_back_db"),
            "freq_hz_min": spec.get("freq_hz_min"),
            "freq_hz_max": spec.get("freq_hz_max"),
        })
    return JSONResponse(out)

@app.get("/antenna_config/{name}")
async def antenna_config(name: str):
    """
    Return full antenna config dict for a given name (if exists).
    """
    global server_state
    spec = server_state.get("antenna_configs", {}).get(name)
    if spec is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(spec)

@app.post("/reload_configs")
async def reload_configs():
    """
    Hot-reload configs/ JSON files into server_state (no restart required).
    """
    global server_state
    try:
        server_state["antenna_configs"] = load_antenna_configs()
        server_state["client_sample"] = None
        return JSONResponse({"ok": True, "count": len(server_state["antenna_configs"])})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/load_bbox")
async def load_bbox(req: Request):
    """
    Load subset of DB: JSON {cx, cy, r}
    Rebuilds executor and KD-trees for workers.
    Also stores last_bbox in server_state for the /pointcloud_full stream.
    """
    global DB_PATH, server_state
    data = await req.json()
    try:
        cx = float(data.get("cx"))
        cy = float(data.get("cy"))
        r = float(data.get("r"))
    except Exception:
        return JSONResponse({"error": "invalid parameters, require cx, cy, r"}, status_code=400)
    minx = cx - r; maxx = cx + r
    miny = cy - r; maxy = cy + r
    print(f"[LOAD] loading bbox centered {cx},{cy} r={r} -> {minx},{miny} - {maxx},{maxy}")
    server_state["last_bbox"] = (minx, miny, maxx, maxy)

    # Sampled load for server-side KD-tree build (fast)
    pts, meta = load_pointcloud_from_db(DB_PATH, bbox=(minx, miny, maxx, maxy),
                                       max_points=None, max_server_points=1500000)
    if pts.shape[0] == 0:
        return JSONResponse({"count": 0, "message": "no points in area"}, status_code=200)
    xy = pts[:, :2]
    try:
        kdtree = KDTree(xy)
    except Exception:
        kdtree = KDTree(xy.tolist())
    server_state["pts"] = pts
    server_state["kdtree"] = kdtree
    server_state["grid_x"] = None
    server_state["grid_y"] = None
    server_state["client_sample"] = None
    cnt = pts.shape[0]
    xs = pts[:,0]; ys = pts[:,1]; zs = pts[:,2]
    centroid = [float(xs.mean()), float(ys.mean()), float(zs.mean())]
    bounds = [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()), float(zs.min()), float(zs.max())]
    old_exec = server_state.get("executor")
    if old_exec:
        try:
            old_exec.shutdown(wait=False)
        except Exception:
            pass
    num_workers = server_state.get("num_workers", os.cpu_count() or 4)
    executor = ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init, initargs=(pts,))
    server_state["executor"] = executor
    server_state["num_workers"] = num_workers
    print(f"[LOAD] loaded {cnt} points; centroid={centroid}. Created executor with {num_workers} workers.")
    # prepare a small client-friendly sample
    client_pts, _ = load_pointcloud_from_db(DB_PATH, bbox=(minx, miny, maxx, maxy), max_points=120000)
    if client_pts.shape[0] > 0:
        cx_c = float(client_pts[:,0].mean()); cy_c = float(client_pts[:,1].mean()); cz_c = float(client_pts[:,2].mean())
        centered_client = [{"x": float(x - cx_c), "y": float(y - cy_c), "z": float(z - cz_c)} for (x,y,z) in client_pts]
    else:
        centered_client = []
    server_state["client_sample"] = {"points": centered_client, "centroid": centroid, "bounds": bounds}
    print(f"[LOAD] returning client sample {len(centered_client)} pts (client cap 120k)")
    return JSONResponse({"count": cnt, "centroid": centroid, "bounds": bounds, "client_sample_count": len(centered_client), "client_points": centered_client})

@app.post("/set_grid")
async def set_grid(req: Request):
    global server_state
    data = await req.json()
    try:
        gr = float(data.get("grid_res", server_state.get("grid_res", 6.0)))
        gm = float(data.get("grid_margin", server_state.get("grid_margin", 40.0)))
    except Exception:
        return JSONResponse({"error":"invalid parameters"}, status_code=400)
    server_state["grid_res"] = gr
    server_state["grid_margin"] = gm
    server_state["grid_x"] = None
    server_state["grid_y"] = None
    server_state["client_sample"] = None
    print(f"[GRID] updated grid_res={gr} grid_margin={gm}")
    return JSONResponse({"grid_res": gr, "grid_margin": gm})

@app.post("/add_tx")
async def add_tx(req: Request):
    """
    Add transmitter. If the payload contains 'ant_config' matching a loaded config,
    attach a copy of that spec as tx['antenna_spec'].
    """
    global server_state
    try:
        tx = await req.json()
    except Exception as e:
        msg = f"Invalid JSON body: {e}"
        print("[TX] add_tx: invalid json:", e)
        return JSONResponse({"error": msg}, status_code=400)

    # Validate and coerce numeric fields
    try:
        tx["x"] = float(tx.get("x"))
        tx["y"] = float(tx.get("y"))
        tx["h_m"] = float(tx.get("h_m", 2.0))
        tx["power_dbm"] = float(tx.get("power_dbm", 20.0))
        tx["freq_hz"] = float(tx.get("freq_hz", server_state.get("freq_hz", 2.4e9)))
        tx["heading_deg"] = float(tx.get("heading_deg", 0.0))
    except Exception as e:
        msg = f"Invalid transmitter parameter types: {e}"
        print("[TX] bad param types:", e)
        return JSONResponse({"error": msg}, status_code=400)

    # Attach antenna spec if provided (also accept full antenna_spec in payload)
    ant_cfg_name = tx.get("ant_config")
    if ant_cfg_name:
        spec = server_state.get("antenna_configs", {}).get(ant_cfg_name)
        if spec:
            tx_spec = dict(spec)
            provided_spec = tx.get("antenna_spec") or {}
            if isinstance(provided_spec, dict):
                tx_spec.update(provided_spec)
            tx["antenna_spec"] = tx_spec
        else:
            print(f"[TX] requested unknown ant_config '{ant_cfg_name}'")
    else:
        if "antenna_spec" in tx:
            pass

    try:
        server_state["txs"].append(tx)
        print("[TX] added:", tx)
        return JSONResponse({"ok": True, "total_txs": len(server_state["txs"])})
    except Exception as exc:
        tb = traceback.format_exc()
        print("[TX] exception adding tx:", tb)
        return JSONResponse({"error": "server exception when adding transmitter", "detail": str(exc)}, status_code=500)

@app.post("/clear_txs")
async def clear_txs():
    server_state["txs"].clear()
    server_state["client_sample"] = None
    print("[TX] cleared")
    return JSONResponse({"ok": True})

@app.post("/simulate")
async def simulate():
    global server_state
    pts = server_state["pts"]
    if pts is None or pts.shape[0] == 0:
        return JSONResponse({"error": "no points loaded"}, status_code=400)
    txs = server_state["txs"]
    if not txs:
        return JSONResponse({"error": "no transmitters defined"}, status_code=400)
    if server_state["grid_x"] is None:
        xs = pts[:,0]; ys = pts[:,1]
        grid_margin_m = server_state.get("grid_margin", 40.0)
        grid_res_m = server_state.get("grid_res", 6.0)
        minx, maxx = float(xs.min()-grid_margin_m), float(xs.max()+grid_margin_m)
        miny, maxy = float(ys.min()-grid_margin_m), float(ys.max()+grid_margin_m)
        nx = max(8, int(math.ceil((maxx-minx)/grid_res_m)))
        ny = max(8, int(math.ceil((maxy-miny)/grid_res_m)))
        server_state["grid_x"] = np.linspace(minx, maxx, nx)
        server_state["grid_y"] = np.linspace(miny, maxy, ny)
    grid_x = server_state["grid_x"]
    grid_y = server_state["grid_y"]

    # Always create a fresh executor for the simulation pass that initializes workers with the grid arrays.
    num_workers = server_state.get("num_workers", os.cpu_count() or 4)
    start = time.time()
    rss = None
    try:
        # Convert grid to plain Python lists for stable pickling into initializer
        grid_x_list = list(map(float, grid_x))
        grid_y_list = list(map(float, grid_y))

        with ProcessPoolExecutor(max_workers=num_workers,
                                 initializer=_worker_init,
                                 initargs=(pts, grid_x_list, grid_y_list)) as executor:
            rss = compute_rssi_grid_parallel(txs, grid_x, grid_y, server_state.get("freq_hz", 2.4e9),
                                             executor=executor, num_workers=num_workers,
                                             tx_power_dbm=20.0, sample_step_m=3.0)
    except Exception as e:
        tb = traceback.format_exc()
        print("[SIM] simulation exception:", tb)
        return JSONResponse({"error": "simulation failed", "detail": str(e)}, status_code=500)

    elapsed = time.time() - start
    print(f"[SIM] compute finished in {elapsed:.2f}s")
    if rss is None:
        return JSONResponse({"error": "simulation produced no result"}, status_code=500)
    rssi_flat = [float(x) for x in rss.ravel().tolist()]
    payload = {"type":"heatmap", "grid": {"xs": [float(x) for x in grid_x], "ys": [float(y) for y in grid_y], "rssi": rssi_flat}}
    try:
        await broadcast_message(json.dumps(payload))
    except Exception as e:
        print("[SIM] broadcast error:", e)
    return JSONResponse(payload)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        await broadcast_clients_count()
        while True:
            _ = await websocket.receive_text()
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
        await broadcast_clients_count()

async def broadcast_clients_count():
    msg = json.dumps({"type":"clients", "count": len(connected_websockets)})
    await broadcast_message(msg)

async def broadcast_message(msg: str):
    living = []
    for ws in list(connected_websockets):
        try:
            await ws.send_text(msg)
            living.append(ws)
        except Exception:
            try:
                await ws.close()
            except Exception:
                pass
    connected_websockets[:] = living

# ------------------------
# Server global state
# ------------------------
server_state = {
    "pts": np.zeros((0,3)),
    "kdtree": None,
    "txs": [],
    "grid_x": None,
    "grid_y": None,
    "grid_res": 6.0,
    "grid_margin": 40.0,
    "freq_hz": 2.4e9,
    "executor": None,
    "num_workers": (os.cpu_count() or 4),
    "antenna_configs": {},   # filled at startup
    "client_sample": None,
    "last_bbox": None
}

# ------------------------
# Entrypoint
# ------------------------
def main():
    global DB_PATH, server_state
    parser = argparse.ArgumentParser(description="Run LiDAR RF on-demand simulator + viewer")
    parser.add_argument("db", help="Path to SQLite DB created by lidar_preprocess.py")
    parser.add_argument("--workers", type=int, default=(os.cpu_count() or 4), help="Number of worker processes for simulation")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    DB_PATH = args.db
    server_state["num_workers"] = max(1, int(args.workers))
    server_state["pts"] = np.zeros((0,3))
    server_state["kdtree"] = None
    server_state["grid_x"] = None
    server_state["grid_y"] = None
    server_state["txs"] = []
    server_state["executor"] = None
    server_state["client_sample"] = None
    server_state["last_bbox"] = None

    # Load antenna configs at startup
    try:
        server_state["antenna_configs"] = load_antenna_configs()
    except Exception as e:
        print("[CONFIG] failed to load antenna configs:", e)
        server_state["antenna_configs"] = {}

    print("Server starting WITHOUT loading points. Use the web UI 'Load area' to populate a subset (then stream full resolution).")
    print(f"Starting server at http://{args.host}:{args.port} ...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
