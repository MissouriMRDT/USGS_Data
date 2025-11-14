#!/usr/bin/env python3

# ******************************************************************************
#  @file      lidar_rf_realtime.py
#  @brief     Interactive LiDAR Point Cloud Explorer with Real-Time RF Simulation.
# 
#  This script implements a FastAPI web application that allows users to
#  explore LiDAR point cloud data stored in a SQLite database and perform
#  real-time RF coverage simulations using advanced GPU acceleration techniques.
#
# 
#  @author     ClayJay3 (claytonraycowen@gmail.com)
#  @date       2025-05-31
#  @copyright  Copyright Mars Rover Design Team 2025 – All Rights Reserved
# ******************************************************************************

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

# Try to import advanced RF and GPU libraries.
try:
    import cupy as cp
    HAS_CUPY = True
    print("[GPU] CuPy available for GPU acceleration")
except ImportError:
    cp = np
    HAS_CUPY = False
    print("[GPU] CuPy not available, using NumPy")

# KD-tree fallback.
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    from scipy.spatial import KDTree

# Speed of light constant.
C = 299792458.0

# ------------------------
# Antenna config loader + default
# ------------------------
CONFIG_DIR = Path("./configs")

def ensure_default_configs():
    """
    Ensure configs/ exists and contains at least one example config.
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

    # --- AM-5G20-90 (5 GHz sector) ---
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
    Load antenna configurations from JSON files in the configs/ directory.

    Returns:
    --------
    Dict[str, Dict]
        A dictionary mapping antenna names to their configuration dictionaries.
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
# Advanced RF Models with GPU Acceleration
# ------------------------
class GPURFSimulator:
    """
    GPU-accelerated RF coverage simulator using point cloud data.
    Uses CuPy for GPU computations if available, otherwise falls back to NumPy.
    Implements Fresnel reflections, diffraction, and occlusion using trimesh ray tracing.
    """

    def __init__(self):
        """
        Initialize the GPURFSimulator.
        """
        self.initialized = False
        self.mesh = None
        self.original_points = None
        self.mesh_shift = np.zeros(3, dtype=float)

    def initialize(self, points: np.ndarray):
        """
        Initialize the GPU RF simulator with the given point cloud.

        Parameters:
        -----------
        points : np.ndarray
            The input point cloud as an (N,3) array of XYZ coordinates.

        Returns:
        --------
        None
        """
        if points is None or points.size == 0:
            print("[GPU] initialize called with empty points")
            return

        print("[GPU] Initializing GPU-accelerated RF simulation (keeping original coordinates)...")
        # Keep original points. (in same coordinate system as KD-tree and TX coords)
        self.original_points = np.asarray(points, dtype=float)
        
        self.initialized = True
        print("[GPU] GPU simulation initialized")

    def compute_coverage_gpu(self, txs: List[Dict], grid_x: np.ndarray,
                         grid_y: np.ndarray, freq_hz: float,
                         kdtree: any, point_cloud: np.ndarray,
                         coarse_plane_grid_m: float = 10.0) -> np.ndarray:
        """
        Compute RF coverage over a grid using GPU acceleration.

        Parameters:
        -----------
        txs : List[Dict]
            List of transmitter specifications.
        grid_x : np.ndarray
            1D array of grid X coordinates.
        grid_y : np.ndarray
            1D array of grid Y coordinates.
        freq_hz : float
            Center frequency in Hz.
        kdtree : any
            KD-tree built from the point cloud for diffraction fallback.
        point_cloud : np.ndarray
            The input point cloud as an (N,3) array of XYZ coordinates.
        coarse_plane_grid_m : float
            Grid spacing in meters for coarse plane estimation.

        Returns:
        --------
        np.ndarray
            2D array of received signal strength in dBm over the grid.
        """
        # If CuPy is available, use it; otherwise, fall back to NumPy. CuPy has many NumPy-compatible functions that instead run on the GPU.
        xp = cp if HAS_CUPY else np
        use_cupy = HAS_CUPY

        if grid_x is None or grid_y is None:
            raise RuntimeError("Grid not set")

        nx = len(grid_x); ny = len(grid_y); N = nx * ny
        print(f"[GPU_SIM] compute_coverage_gpu: use_cupy={use_cupy} grid {nx}x{ny} => {N} cells")

        # World coordinates. (numpy arrays for CPU tasks, xp arrays for GPU math)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_xy = np.column_stack((gx.ravel(), gy.ravel()))  # (N,2)

        # Ground z per cell. (nearest neighbor from point cloud) - CPU
        if kdtree is not None and point_cloud is not None and point_cloud.shape[0] > 0:
            try:
                _, idxs = kdtree.query(grid_xy, k=1)
                grid_z = point_cloud[np.array(idxs, dtype=int), 2].astype(float)
            except Exception as e:
                print("[GPU_SIM] Warning: kdtree.query failed:", e)
                grid_z = np.zeros((N,), dtype=float)
        else:
            grid_z = np.zeros((N,), dtype=float)

        # Convert arrays to xp for GPU math.
        grid_xp = xp.asarray(grid_xy[:, 0].astype(float))
        grid_yp = xp.asarray(grid_xy[:, 1].astype(float))
        grid_zp = xp.asarray(grid_z.astype(float))

        eps = 1e-12
        DEFAULT_MATERIAL_LOSS_DB = 15.0

        # Helpers. (pattern interpolation remains CPU cause it's pretty cheap)
        def parse_pattern(raw):
            """
            Parse antenna pattern from various formats into sorted list of (angle_deg, gain_db).

            Parameters:
            -----------
            raw : various
                Raw pattern input (dict, list of tuples, list of dicts, list of strings).
            
            Returns:
            --------
            List[Tuple[float, float]]
                Sorted list of (angle_deg, gain_db) tuples.
            """
            out = []
            if not raw:
                return out
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        out.append((float(k) % 360.0, float(v)))
                    except Exception:
                        continue
                out.sort(key=lambda x: x[0]); return out
            if isinstance(raw, (list, tuple)):
                for e in raw:
                    try:
                        if isinstance(e, (list, tuple)) and len(e) >= 2:
                            out.append((float(e[0]) % 360.0, float(e[1]))); continue
                        if isinstance(e, dict):
                            if "angle" in e and "gain" in e:
                                out.append((float(e["angle"]) % 360.0, float(e["gain"]))); continue
                            vals = list(e.values())
                            if len(vals) >= 2:
                                out.append((float(vals[0]) % 360.0, float(vals[1]))); continue
                        if isinstance(e, str):
                            for sep in [',', ':', ' ']:
                                if sep in e:
                                    p = [s for s in e.split(sep) if s != '']
                                    if len(p) >= 2:
                                        out.append((float(p[0]) % 360.0, float(p[1]))); break
                    except Exception:
                        continue
            out.sort(key=lambda x: x[0]); return out

        def interp_pattern_scalar(pattern_list, angle_deg):
            """
            Interpolate antenna pattern gain at given angle in degrees.

            Parameters:
            -----------
            pattern_list : List[Tuple[float, float]]
                Sorted list of (angle_deg, gain_db) tuples.
            angle_deg : float
                Angle in degrees to interpolate gain for.
            
            Returns:
            --------
            float
                Interpolated gain in dB.
            """
            if not pattern_list: return 0.0
            angs = np.array([p[0] for p in pattern_list], dtype=float)
            gains = np.array([p[1] for p in pattern_list], dtype=float)
            angs_ext = np.concatenate((angs, angs[:1] + 360.0))
            gains_ext = np.concatenate((gains, gains[:1]))
            a = float(angle_deg) % 360.0
            idx = np.searchsorted(angs_ext, a, side='right')
            i1 = max(0, idx - 1); i2 = min(len(angs_ext) - 1, idx)
            a1, a2 = angs_ext[i1], angs_ext[i2]; g1, g2 = gains_ext[i1], gains_ext[i2]
            if abs(a2 - a1) < 1e-9: return float(g1)
            frac = (a - a1) / (a2 - a1); return float(g1 + frac * (g2 - g1))

        def knife_edge_loss_db(h, d1, d2, lam):
            """
            Compute knife-edge diffraction loss in dB.

            Parameters:
            -----------
            h : float
                Height of obstruction above line-of-sight in meters.
            d1 : float
                Distance from transmitter to obstruction in meters.
            d2 : float
                Distance from obstruction to receiver in meters.
            lam : float
                Wavelength in meters.
          
            Returns:
            --------
            float
                Diffraction loss in dB.
            """
            if d1 <= 0 or d2 <= 0:
                return 0.0
            v = (h * np.sqrt(2.0 * (d1 + d2) / (lam * d1 * d2)))
            if v <= -0.78:
                return 0.0
            val = 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1)**2 + 1.0) + v - 0.1)
            return max(0.0, float(val))

        def kd_ray_sample_diffraction(origin_world, target_world, kdtree_local, pc_np, lam, n_samples=6, clearance_m=0.6):
            """
            Estimate diffraction loss using KD-tree ray sampling.

            Parameters:
            -----------
            origin_world : np.ndarray
                Transmitter origin point (3,).
            target_world : np.ndarray
                Receiver target points (M,3).
            kdtree_local : any
                KD-tree built from local point cloud.
            pc_np : np.ndarray
                Local point cloud as (N,3) array.
            lam : float
                Wavelength in meters.
            n_samples : int
                Number of samples along the ray.
            clearance_m : float
                Clearance radius in meters for KD-tree queries.
            
            Returns:
            --------
            np.ndarray
                Array of diffraction losses in dB for each target point (M,).
            """
            # Identical to earlier robust fallback. (kept as CPU)
            M = target_world.shape[0]
            losses = np.zeros(M, dtype=float)
            if kdtree_local is None or pc_np is None or pc_np.shape[0] == 0:
                return losses
            vecs = target_world - origin_world[None, :]
            dists = np.linalg.norm(vecs, axis=1)
            nz = dists > 1e-9
            ts = np.linspace(0.1, 0.9, min(n_samples, 8))
            for t in ts:
                pts = origin_world[None, :] + vecs * t
                try:
                    idx_lists = kdtree_local.query_ball_point(pts[:, :2], r=clearance_m)
                except Exception:
                    d2, i2 = kdtree_local.query(pts[:, :2], k=1)
                    idx_lists = [[int(i2[i])] for i, _ in enumerate(d2)]
                for i, idxs in enumerate(idx_lists):
                    if not idxs or not nz[i]:
                        continue
                    ray_z = pts[i, 2]
                    zs = pc_np[np.array(idxs, dtype=int), 2]
                    if zs.size and np.any(zs > ray_z + 0.05):
                        hit_z = float(np.max(zs))
                        d_total = dists[i]
                        d1 = d_total * t
                        d2 = d_total - d1
                        los_z = origin_world[2] + (target_world[i, 2] - origin_world[2]) * (d1 / (d_total + eps))
                        h = hit_z - los_z
                        if h <= 0:
                            continue
                        Ld = knife_edge_loss_db(h, d1, d2, lam)
                        mat_loss = DEFAULT_MATERIAL_LOSS_DB
                        total = Ld + mat_loss
                        if total > losses[i]:
                            losses[i] = total
            return losses

        # Trimesh ray helpers. (CPU)
        mesh = getattr(self, "mesh", None)
        mesh_shift = np.asarray(getattr(self, "mesh_shift", np.zeros(3, dtype=float)), dtype=float)
        has_trimesh = False
        if mesh is not None:
            try:
                import trimesh
                if isinstance(mesh, trimesh.Trimesh):
                    has_trimesh = True
                else:
                    # Try to coerce.
                    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices) - mesh_shift, faces=np.asarray(mesh.faces), process=False)
                    has_trimesh = True
            except Exception as e:
                print("[GPU_SIM] trimesh unavailable for accelerated occlusion:", e)
                has_trimesh = False

        # --- COARSE PLANE CACHE (CPU) ---
        # Build a coarse grid over the domain and estimate a local plane for each coarse cell.
        coarse_grid_m = float(coarse_plane_grid_m)
        # Bounding box of grid.
        minx = float(np.min(grid_xy[:, 0])); maxx = float(np.max(grid_xy[:, 0]))
        miny = float(np.min(grid_xy[:, 1])); maxy = float(np.max(grid_xy[:, 1]))
        nx_coarse = max(1, int(np.ceil((maxx - minx) / coarse_grid_m)))
        ny_coarse = max(1, int(np.ceil((maxy - miny) / coarse_grid_m)))
        coarse_xs = np.linspace(minx, maxx, nx_coarse)
        coarse_ys = np.linspace(miny, maxy, ny_coarse)
        coarse_centers = np.array(np.meshgrid(coarse_xs, coarse_ys)).reshape(2, -1).T  # (M_coarse, 2)

        # CPU function to estimate plane via PCA for one center. (uses KD-tree to find neighbors)
        def estimate_plane_cpu(center_xy, k=48):
            """
            Estimate local plane at center_xy using k nearest neighbors from point_cloud via kdtree.

            Parameters:
            -----------
            center_xy : np.ndarray
                Center XY coordinate (2,).
            k : int
                Number of nearest neighbors to use.
            
            Returns:
            --------
            Tuple[np.ndarray, np.ndarray]
                (centroid (3,), normal (3,)) of estimated plane, or (None, None) on failure.
            """
            if kdtree is None or point_cloud is None or point_cloud.shape[0] == 0:
                return None, None
            try:
                dists, idxs = kdtree.query([center_xy], k=min(k, point_cloud.shape[0]))
                idxs = np.array(idxs[0], dtype=int)
                pts = point_cloud[idxs]
                if pts.shape[0] < 6:
                    return None, None
                cen = np.mean(pts, axis=0)
                cov = (pts - cen).T @ (pts - cen) / float(pts.shape[0])
                w, v = np.linalg.eigh(cov)
                normal = v[:, 0]
                if normal[2] < 0:
                    normal = -normal
                normal = normal / (np.linalg.norm(normal) + eps)
                return cen, normal
            except Exception:
                return None, None

        # Compute coarse planes (CPU), cache results in arrays. (len = M_coarse)
        coarse_pts = []
        coarse_norms = []
        for cc in coarse_centers:
            pt, n = estimate_plane_cpu(cc, k=64)
            if pt is None:
                # Fallback to horizontal at median elevation near center from kdtree nearest.
                if kdtree is not None and point_cloud is not None:
                    try:
                        d, i = kdtree.query([cc], k=1)
                        p = point_cloud[int(i[0])]
                        pt = p
                        n = np.array([0.0, 0.0, 1.0])
                    except Exception:
                        pt = np.array([cc[0], cc[1], 0.0])
                        n = np.array([0.0, 0.0, 1.0])
                else:
                    pt = np.array([cc[0], cc[1], 0.0])
                    n = np.array([0.0, 0.0, 1.0])
            coarse_pts.append(pt)
            coarse_norms.append(n)
        coarse_pts = np.vstack(coarse_pts)  # (M_coarse, 3)
        coarse_norms = np.vstack(coarse_norms)  # (M_coarse, 3)

        # Map each fine grid cell to the nearest coarse center index. (CPU)
        from sklearn.neighbors import KDTree as SkKDTree  # Scikit-learn KDTree is usually available; if not, fallback to numpy.
        try:
            sktree = SkKDTree(coarse_centers)
            _, coarse_idx = sktree.query(grid_xy, k=1)
            coarse_idx = coarse_idx.ravel()
        except Exception:
            # Fallback brute force. (may be slower)
            dists_coarse = np.sum((grid_xy[:, None, :] - coarse_centers[None, :, :])**2, axis=2)
            coarse_idx = np.argmin(dists_coarse, axis=1)

        # Upload coarse plane arrays to GPU. (xp)
        coarse_pts_xp = xp.asarray(coarse_pts.astype(float))
        coarse_norms_xp = xp.asarray(coarse_norms.astype(float))
        coarse_idx_xp = xp.asarray(coarse_idx.astype(int))

        # Batch settings for GPU work.
        BATCH = 8192
        rss_mw = xp.zeros(N, dtype=float)

        # Small helper: Fresnel. (implemented using numpy functions but vectorizable on xp)
        def fresnel_unpolarized_vec(inc_dirs_xp, normals_xp, eps_r):
            """
            Compute unpolarized Fresnel reflection coefficients (magnitude and sign) for vectors.

            Parameters:
            -----------
            inc_dirs_xp : xp.ndarray
                Incident direction vectors (M,3).
            normals_xp : xp.ndarray
                Surface normal vectors (M,3).
            eps_r : float
                Relative permittivity of the surface.
            
            Returns:
            --------
            Tuple[xp.ndarray, xp.ndarray]
                (magnitude (M,), sign (M,)) of reflection coefficients.
            """
            # Ensure unit.
            inc_u = inc_dirs_xp / (xp.linalg.norm(inc_dirs_xp, axis=1)[:, None] + eps)
            n_u = normals_xp / (xp.linalg.norm(normals_xp, axis=1)[:, None] + eps)
            cos_i = xp.clip(-xp.sum(inc_u * n_u, axis=1), -1.0, 1.0)
            sin_i = xp.sqrt(xp.maximum(0.0, 1.0 - cos_i**2))
            n1 = 1.0
            n2 = xp.sqrt(float(eps_r))
            sin_t = n1 / n2 * sin_i
            # Handle total internal reflection cases: sin_t >=1 -> mag=1, sign=-1.
            total_internal = sin_t >= 1.0
            cos_t = xp.sqrt(xp.maximum(0.0, 1.0 - sin_t**2))
            rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t + eps)
            rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t + eps)
            mag = 0.5 * (xp.abs(rs) + xp.abs(rp))
            avg = 0.5 * (rs + rp)
            sign = xp.where(xp.real(avg) < 0.0, -1.0, 1.0)
            mag = xp.where(total_internal, 1.0, mag)
            sign = xp.where(total_internal, -1.0, sign)
            return mag, sign

        # Main per-transmitter loop; heavy arithmetic uses xp and runs on GPU if cupy present.
        for tx_idx, tx in enumerate(txs):
            tx_x = float(tx.get("x", 0.0)); tx_y = float(tx.get("y", 0.0)); tx_h = float(tx.get("h_m", 2.0))
            tx_power_dbm = float(tx.get("power_dbm", 20.0))
            tx_freq = float(tx.get("freq_hz", freq_hz))
            tx_ant_spec = tx.get("antenna_spec") if isinstance(tx.get("antenna_spec"), dict) else (tx.get("antenna_spec") or {})
            tx_heading = float(tx.get("heading_deg", tx.get("heading", 0.0) or 0.0))
            declared_gain = float(tx_ant_spec.get("gain_db", 0.0))
            D = float(tx_ant_spec.get("aperture_m", tx_ant_spec.get("largest_dim_m", tx_ant_spec.get("diameter_m", 0.1))))
            if D <= 0:
                D = 0.1
            fraunhofer_R = 2.0 * (D**2) / (C / tx_freq + eps)

            # Reflection/fresnel defaults.
            eps_r = float(tx.get("ground_eps_r", 6.0))
            roughness = float(tx.get("ground_roughness_m", 0.02))
            enable_reflection = bool(tx.get("enable_ground_reflection", True))
            rx_aperture_m = float(tx.get("rx_aperture_m", 0.0))

            # Pattern CPU precompute (per-cell dBi) -> then send to GPU.
            azpat = parse_pattern(tx_ant_spec.get("azimuth_pattern") or tx_ant_spec.get("az_pattern") or [])
            elpat = parse_pattern(tx_ant_spec.get("elevation_pattern") or tx_ant_spec.get("elevation") or [])
            # Compute az/el CPU arrays. (fast)
            az_abs = np.degrees(np.arctan2(np.asarray(grid_xy[:,1]) - tx_y, np.asarray(grid_xy[:,0]) - tx_x)) % 360.0
            heading_bearing = (90.0 - tx_heading) % 360.0
            rel_az = (az_abs - heading_bearing + 360.0) % 360.0
            el_abs = np.degrees(np.arctan2(np.asarray(grid_z) - tx_h, np.sqrt((np.asarray(grid_xy[:,0]) - tx_x)**2 + (np.asarray(grid_xy[:,1]) - tx_y)**2)))

            def pattern_gains_np(pat):
                if not pat:
                    return None
                angs = np.array([p[0] for p in pat], dtype=float)
                gains = np.array([p[1] for p in pat], dtype=float)
                return angs, gains

            az_parsed = pattern_gains_np(azpat)
            el_parsed = pattern_gains_np(elpat)

            az_vals_np = np.zeros(N, dtype=float)
            if az_parsed is None:
                az_vals_np[:] = declared_gain
            else:
                pat_peak = float(np.max(az_parsed[1]))
                scale_offset = declared_gain - pat_peak
                for i in range(N):
                    az_vals_np[i] = interp_pattern_scalar(azpat, float(rel_az[i]))

                az_vals_np += scale_offset

            el_vals_np = np.zeros(N, dtype=float)
            if el_parsed is None:
                el_vals_np[:] = 0.0
            else:
                pat_peak_e = float(np.max(el_parsed[1]))
                el_scale_offset = 0.0 if abs(pat_peak_e) <= 2.0 else (0.0 - pat_peak_e)
                for i in range(N):
                    el_vals_np[i] = interp_pattern_scalar(elpat, float(el_abs[i]))
                el_vals_np += el_scale_offset

            gain_db_np = az_vals_np + el_vals_np

            # Move gain linear to GPU.
            Gt_lin_xp = xp.asarray((10.0 ** (gain_db_np / 10.0)).astype(float))

            # Frequency averaging.
            def make_freq_list(center_f, tx_opts):
                n = int(tx_opts.get("freq_averaging_n", 5))
                frac = float(tx_opts.get("freq_averaging_frac", 0.005))
                if n <= 1 or frac <= 0.0:
                    return [center_f]
                idx = np.linspace(-1.0, 1.0, n)
                return list(center_f * (1.0 + idx * frac))

            freq_list = make_freq_list(tx_freq, tx)

            # Precompute kd-based diffraction losses (CPU) once per center frequency ideally, but it's not freq-dependent strongly.
            # We'll compute kd_losses once (CPU) and transfer to GPU array for use as amplitude attenuation fallback when mesh unavailable.
            if kdtree is not None and point_cloud is not None and point_cloud.shape[0] > 0:
                # Use center frequency lam for diffraction estimate.
                lam_center = C / float(tx_freq)
                kd_losses_full = kd_ray_sample_diffraction(np.array([tx_x, tx_y, tx_h], dtype=float),
                                                          np.column_stack((np.asarray(grid_xy[:,0]), np.asarray(grid_xy[:,1]), np.asarray(grid_z))),
                                                          kdtree, point_cloud, lam_center, n_samples=8, clearance_m=0.6)
            else:
                kd_losses_full = np.zeros((N,), dtype=float)
            kd_losses_xp = xp.asarray(kd_losses_full.astype(float))

            # Accumulate frequency-averaged power in mW on GPU.
            rss_mw_tx_acc = xp.zeros(N, dtype=float)

            # For reflection: determine per-grid-plane normals from coarse cache (upload to GPU already)
            # coarse_idx_xp maps each fine cell to coarse plane index; coarse_pts_xp, coarse_norms_xp available.
            for f in freq_list:
                lam = C / float(f)
                k = 2.0 * np.pi / lam
                # Batch loop - do heavy math on GPU.
                for s in range(0, N, BATCH):
                    e = min(N, s + BATCH)

                    # Gather batch coords on GPU. (xp arrays)
                    bx = grid_xp[s:e]; by = grid_yp[s:e]; bz = grid_zp[s:e]
                    idxs = xp.arange(s, e, dtype=int)

                    # Direct path geometry. (GPU)
                    dx = bx - float(tx_x); dy = by - float(tx_y); dz = bz - float(tx_h)
                    dist_direct = xp.sqrt(dx*dx + dy*dy + dz*dz) + eps
                    R_eff_direct = xp.maximum(dist_direct, float(fraunhofer_R))

                    # Direct occlusion: prefer trimesh (CPU) check; if trimesh exists we do the per-batch ray test on CPU
                    # but we move its boolean result to GPU as occlusion mask; otherwise use kd_losses_xp slice.
                    if has_trimesh and mesh is not None:
                        # Build CPU ray origins/directions for batch.
                        # NOTE: we perform this CPU ray test once per batch to avoid thousands of small CPU calls.
                        origins = np.tile(np.array([tx_x, tx_y, tx_h]) - mesh_shift, (e - s, 1))
                        dirs = np.vstack((xp.asnumpy(bx) - tx_x, xp.asnumpy(by) - tx_y, xp.asnumpy(bz) - tx_h)).T
                        dir_norms = np.linalg.norm(dirs, axis=1) + eps
                        dirs_unit = dirs / dir_norms[:, None]
                        try:
                            # Intersects_first returns distances array. (numpy)
                            d_first = mesh.ray.intersects_first(origins=origins, directions=dirs_unit)
                            occluded_mask = np.logical_and(np.isfinite(d_first), d_first < (dir_norms - 1e-6))
                            occlusion_loss_db_direct = np.where(occluded_mask, DEFAULT_MATERIAL_LOSS_DB, 0.0).astype(float)
                        except Exception:
                            # Fallback coarse: if any intersection exists mark occluded.
                            try:
                                any_hit = mesh.ray.intersects_any(origins=origins, directions=dirs_unit)
                                occlusion_loss_db_direct = np.where(any_hit, DEFAULT_MATERIAL_LOSS_DB, 0.0).astype(float)
                            except Exception:
                                occlusion_loss_db_direct = kd_losses_full[s:e]
                    else:
                        # Use kd-based losses.
                        occlusion_loss_db_direct = np.asarray(kd_losses_full[s:e]).astype(float)

                    # Move occlusion to GPU.
                    occl_db_direct_xp = xp.asarray(occlusion_loss_db_direct.astype(float))

                    # Complex direct field. (GPU)
                    Pt_w = (10.0 ** (tx_power_dbm / 10.0)) / 1000.0
                    amp_pref_direct = xp.sqrt(Pt_w * Gt_lin_xp[s:e] * 1.0) * (lam / (4.0 * xp.pi * R_eff_direct))
                    amp_factor_direct = 10.0 ** (-occl_db_direct_xp / 20.0)
                    phase_direct = xp.exp(-1j * (2.0 * xp.pi / lam) * dist_direct)
                    E_direct = amp_pref_direct * amp_factor_direct * phase_direct

                    # Reflected path. (GPU math, CPU plane & mesh decisions)
                    if enable_reflection:
                        # Get coarse plane index slice. (GPU -> but coarse_idx_xp is xp already)
                        coarse_ids = coarse_idx_xp[s:e]  # xp int array
                        # Fetch coarse points/normals for batch. (xp index gather)
                        plane_pts_batch = coarse_pts_xp[coarse_ids]  # (batch,3)
                        plane_norms_batch = coarse_norms_xp[coarse_ids]  # (batch,3)

                        # Compute image source coordinates on GPU: image = tx - 2 * dot(v, normal) * normal.
                        tx_origin_xp = xp.asarray(np.array([tx_x, tx_y, tx_h], dtype=float))
                        vtx = tx_origin_xp[None, :] - plane_pts_batch  # (batch,3)
                        proj = xp.sum(vtx * plane_norms_batch, axis=1)[:, None]  # (batch,1)
                        image_tx_batch = tx_origin_xp[None, :] - 2.0 * proj * plane_norms_batch  # (batch,3)

                        # Path from image to receiver. (GPU)
                        rx_pts_batch = xp.stack((bx, by, bz), axis=1)  # (batch,3)
                        vec_img_rx = rx_pts_batch - image_tx_batch
                        dist_img_rx = xp.linalg.norm(vec_img_rx, axis=1) + eps
                        R_eff_refl = xp.maximum(dist_img_rx, float(fraunhofer_R))

                        # For occlusion of reflected ray: we need a CPU mesh ray test per batch because mesh.ray is CPU-only.
                        # We'll prepare per-batch origins/directions on CPU and query mesh.ray for intersections,
                        # then transfer boolean occlusion mask to GPU.
                        if has_trimesh and mesh is not None:
                            # Prepare CPU arrays of origins/directions. (image_tx_batch -> numpy)
                            try:
                                img_origins_cpu = xp.asnumpy(image_tx_batch) + mesh_shift  # Account for mesh_shift invert earlier usage.
                                img_dirs_cpu = xp.asnumpy(vec_img_rx)
                                img_dir_norms = np.linalg.norm(img_dirs_cpu, axis=1) + eps
                                img_dirs_unit_cpu = img_dirs_cpu / img_dir_norms[:, None]
                                # Intersects_first gives first-hit distance per ray. (numpy)
                                try:
                                    d_first_refl = mesh.ray.intersects_first(origins=img_origins_cpu, directions=img_dirs_unit_cpu)
                                    occluded_refl_mask = np.logical_and(np.isfinite(d_first_refl), d_first_refl < (img_dir_norms - 1e-6))
                                    occl_db_refl = np.where(occluded_refl_mask, DEFAULT_MATERIAL_LOSS_DB, 0.0).astype(float)
                                except Exception:
                                    any_hit_refl = mesh.ray.intersects_any(origins=img_origins_cpu, directions=img_dirs_unit_cpu)
                                    occl_db_refl = np.where(any_hit_refl, DEFAULT_MATERIAL_LOSS_DB, 0.0).astype(float)
                            except Exception as ex:
                                # Trimesh per-batch failed: fallback to kd-based loss for reflected path.
                                occl_db_refl = np.asarray(kd_losses_full[s:e]).astype(float)
                        else:
                            occl_db_refl = np.asarray(kd_losses_full[s:e]).astype(float)

                        # Move occlusion to GPU.
                        occl_db_refl_xp = xp.asarray(occl_db_refl.astype(float))

                        # Incidence vectors (GPU): approximate incident vector from tx->plane_pt. (plane_pt = plane_pts_batch)
                        inc_vec = (plane_pts_batch - tx_origin_xp[None, :])  # (batch,3)
                        # Compute Fresnel magnitude & sign on GPU.
                        refl_mag_xp, refl_sign_xp = fresnel_unpolarized_vec(inc_vec, plane_norms_batch, eps_r)

                        # Roughness spec factor (GPU) approximate.
                        cos_i = xp.clip(-xp.sum((inc_vec / (xp.linalg.norm(inc_vec, axis=1)[:, None] + eps)) * plane_norms_batch, axis=1), -1.0, 1.0)
                        sin_i = xp.sqrt(xp.maximum(0.0, 1.0 - cos_i**2))
                        spec_factor_xp = xp.exp(- (4.0 * xp.pi * roughness * sin_i / (lam + eps))**2)

                        # Amplitude prefactor for reflected path (GPU)
                        # and convert per-cell Gt to xp slice.
                        Gt_lin_batch = Gt_lin_xp[s:e]
                        amp_pref_refl = xp.sqrt(Pt_w * Gt_lin_batch * 1.0) * (lam / (4.0 * xp.pi * R_eff_refl))
                        amp_factor_refl = 10.0 ** (-occl_db_refl_xp / 20.0)
                        # Combine reflection amplitude: refl_mag * spec_factor * sign.
                        refl_complex_factor = (refl_mag_xp * spec_factor_xp) * refl_sign_xp
                        # Phase for reflected path uses image->rx distance.
                        phase_refl = xp.exp(-1j * (2.0 * xp.pi / lam) * dist_img_rx)
                        E_refl = amp_pref_refl * amp_factor_refl * phase_refl * refl_complex_factor
                    else:
                        E_refl = xp.zeros(e - s, dtype=complex)

                    # Coherent sum for this batch.
                    E_batch = E_direct + E_refl

                    # Power (W)
                    Pr_batch_w = xp.abs(E_batch) ** 2
                    rss_mw_tx_acc[s:e] += (Pr_batch_w * 1000.0)

            # Frequency average on GPU.
            rss_mw_tx = rss_mw_tx_acc / float(len(freq_list))

            # Optional receiver aperture smoothing: do on GPU by convolving with a small kernel if aperture>0.
            if rx_aperture_m > 0.0:
                # Crude GPU smoothing: perform a per-point average of neighbors whose XY distance <= aperture using grid indexing
                # For simplicity and performance we resample grid to 2D image and run a gaussian blur using FFT if available (cupy has FFT).
                # Only do this if nx,ny sensible and xp supports FFT; otherwise fallback to CPU average.
                try:
                    # Reshape to 2D (ny,nx)
                    img = rss_mw_tx.reshape((ny, nx))
                    # Compute gaussian kernel sigma in pixels approximated by aperture / grid spacing.
                    dx = (grid_x[1] - grid_x[0]) if nx > 1 else 1.0
                    dy = (grid_y[1] - grid_y[0]) if ny > 1 else 1.0
                    sigma_x = max(1.0, rx_aperture_m / (dx + eps))
                    sigma_y = max(1.0, rx_aperture_m / (dy + eps))
                    # Separable gaussian using FFT: create gaussian kernel in freq domain.
                    # Create coords.
                    ax = xp.fft.fftfreq(nx)[:, None]
                    ay = xp.fft.fftfreq(ny)[None, :]
                    # Build gaussian in frequency domain. (approx)
                    # NOTE: simpler: use cupyx.scipy.ndimage.gaussian_filter if cupy + cupyx installed; else fallback.
                    try:
                        import cupyx.scipy.ndimage as cnd
                        img_sm = cnd.gaussian_filter(img, sigma=(sigma_y, sigma_x))
                    except Exception:
                        # Fallback: no cupyx gaussian; do a naive uniform blur with small kernel. (slower)
                        kx = max(1, int(round(rx_aperture_m / (dx + eps))))
                        ky = max(1, int(round(rx_aperture_m / (dy + eps))))
                        kernel = xp.ones((2*ky+1, 2*kx+1), dtype=float)
                        kernel = kernel / xp.sum(kernel)
                        # Separable conv via fft. (if xp supports fft)
                        try:
                            img_padded = img
                            # Naive convolution: apply via scipy equivalent not available -> skip heavy smoothing.
                            img_sm = img  # Fallback: no smoothing.
                        except Exception:
                            img_sm = img
                    rss_mw_tx = img_sm.ravel()
                except Exception:
                    # Fallback: no GPU smoothing available.
                    pass

            # Accumulate over transmitters. (incoherent)
            rss_mw += rss_mw_tx

        # Finalize: to dBm and back to numpy.
        rss_mw = xp.maximum(rss_mw, 1e-12)
        rss_dbm = 10.0 * xp.log10(rss_mw)
        if use_cupy:
            out = xp.asnumpy(rss_dbm).reshape((ny, nx))
        else:
            out = rss_dbm.reshape((ny, nx))

        print("[GPU_SIM] compute_coverage_gpu done, shape=", out.shape)
        return out


# ------------------------
# DB / point cloud loader
# ------------------------
def load_pointcloud_from_db(db_path: str,
                           bbox: Optional[Tuple[float,float,float,float]] = None,
                           max_points: Optional[int] = None,
                           max_server_points: int = 1500000) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load and optionally sample point cloud from SQLite database within a bounding box.

    Parameters:
    -----------
    db_path : str
        Path to the SQLite database file.
    bbox : Optional[Tuple[float,float,float,float]]
        Bounding box (min_x, min_y, max_x, max_y) to filter points
        from. If None, load all points.
    max_points : Optional[int]
        Maximum number of points to sample and return.
    max_server_points : int
        Maximum number of points to sample if max_points is None.
    
    Returns:
    --------
    Tuple[np.ndarray, List[Dict]]
        A tuple containing:
        - pts : np.ndarray
            Array of shape (M, 3) with sampled point coordinates.
        - meta : List[Dict]
            List of metadata dictionaries for each sampled point.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = None
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
# Full-resolution stream endpoint generator (NDJSON) - preserved
# ------------------------
def stream_full_points_generator(db_path: str, bbox: Tuple[float,float,float,float], fetch_size: int = 10000):
    """
    Generator that streams full-resolution LiDAR points from the database within a bounding box as NDJSON.

    Parameters:
    -----------
    db_path : str
        Path to the SQLite database file.
    bbox : Tuple[float,float,float,float]
        Bounding box (min_x, min_y, max_x, max_y) to filter points
    fetch_size : int
        Number of rows to fetch per database query.
    
    Yields:
    -------
    bytes
        Encoded NDJSON lines for each point and a final metadata line.
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
# FastAPI app + websockets (ORIGINAL GUI FULLY PRESERVED)
# ------------------------
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=500)

connected_websockets: List[WebSocket] = []

DB_PATH = None

# Initialize GPU simulator
gpu_simulator = GPURFSimulator()

# Full HTML page (client) - ORIGINAL PRESERVED
HTML_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>LiDAR RF Viewer</title>
  <style>
    body { margin:0; overflow:hidden; background:#0b0b0b; color:#eee; font-family: Arial, sans-serif; }
    #overlay { position:absolute; top:8px; left:8px; z-index:20; background:rgba(255,255,255,0.95); color:#000; padding:10px; border-radius:6px; width:420px; max-height:92vh; overflow:auto; }
    #legend { margin-top:8px; padding:6px; background:#fff; border-radius:4px; font-size:12px; color:#000; }
    .color-bar {height: 14px; width: 100%; border-radius: 4px; background: linear-gradient(to right, rgb(0, 0, 0), rgb(255, 0, 0),rgb(255, 255, 0),rgb(0, 255, 0),rgb(0, 0, 255)); display: block; margin-bottom: 6px; }    .legend-row { display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px; }
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
    #gpuStatus { color: #0066cc; font-weight: bold; }
    .status-good { color: #00aa00; }
    .status-warn { color: #ff8800; }
  </style>
</head>
<body>
<div id="overlay">
  <div><b>LiDAR RF Viewer</b> <span id="gpuStatus">(Checking GPU...)</span></div>

  <div style="margin-top:8px;">
    <label>UTM Easting</label><input id="utm_e" type="number" step="0.01" value="606853.65"/><br/>
    <label>UTM Northing</label><input id="utm_n" type="number" step="0.01" value="4200936.40"/><br/>
    <label>Radius (m)</label><input id="utm_r" type="number" step="1" value="100"/><br/>
    <div style="margin-top:6px;">
      <button id="loadArea">Load area (stream full res)</button>
      <button id="clearTx">Clear TX</button>
    </div>
  </div>

  <hr/>

  <div>
    <label>Grid resolution (m)</label><input id="grid_res" type="number" step="0.1" value="2.0"/>
    <br/>
    <label>Grid margin (m)</label><input id="grid_margin" type="number" step="1" value="20.0"/>
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
      <button id="simulate">Simulate (with Occlusion)</button>
    </div>
  </div>

  <hr/>

  <div><b>Clients:</b> <span id="clients">0</span></div>
  <div><b>Loaded points:</b> <span id="ptcount">0</span></div>
  <div><b>Simulation status:</b> <span id="simStatus">Ready</span></div>

  <div id="progressBarContainer"><div id="progressBar"></div></div>

  <div id="legend">
    <div class="color-bar"></div>
    <div class="legend-row"><span><b>Blue</b> - Excellent (≥ -55 dBm)</span><span>-55 dBm</span></div>
    <div class="legend-row"><span><b>Green</b> - Good (-65 to -56 dBm)</span><span>-65 dBm</span></div>
    <div class="legend-row"><span><b>Yellow</b> - Fair (-75 to -66 dBm)</span><span>-75 dBm</span></div>
    <div class="legend-row"><span><b>Red</b> - Poor (-85 to -76 dBm)</span><span>-85 dBm</span></div>
    <div class="legend-row"><span><b>Black</b> - Unusable (≤ -86 dBm)</span><span>&lt;= -86</span></div>
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
let antennaSpecsCache = {};

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

function rssi_to_rgb(rssi) {
  // Define RSSI breakpoints and corresponding RGB colors
  const levels = [
    { rssi: -100, color: [0, 0, 0] },     // Black (Unusable)
    { rssi: -85,  color: [1, 0, 0] },     // Red (Poor)
    { rssi: -75,  color: [1, 1, 0] },     // Yellow (Fair)
    { rssi: -65,  color: [0, 1, 0] },     // Green (Good)
    { rssi: -55,  color: [0, 0, 1] }      // Blue (Excellent)
  ];

  // Clamp RSSI to range
  if (rssi <= levels[0].rssi) return levels[0].color;
  if (rssi >= levels[levels.length - 1].rssi) return levels[levels.length - 1].color;

  // Find the segment rssi falls into
  for (let i = 0; i < levels.length - 1; i++) {
    const a = levels[i];
    const b = levels[i + 1];
    if (rssi >= a.rssi && rssi <= b.rssi) {
      // Linear interpolation between colors
      const t = (rssi - a.rssi) / (b.rssi - a.rssi);
      const r = a.color[0] + t * (b.color[0] - a.color[0]);
      const g = a.color[1] + t * (b.color[1] - a.color[1]);
      const b_ = a.color[2] + t * (b.color[2] - a.color[2]);
      return [r, g, b_];
    }
  }
}


function createTxViz(tx) {
  const cx = centroid[0] || 0, cy = centroid[1] || 0, cz = centroid[2] || 0;
  const group = new THREE.Group();

  const sGeom = new THREE.SphereGeometry(1.2, 12, 8);
  const sMat = new THREE.MeshBasicMaterial({color: 0x00ff00});
  const s = new THREE.Mesh(sGeom, sMat);
  // Viewer Z coordinates are point-cloud-relative, so subtract centroid Z from absolute tx.h_m
  s.position.set(tx.x - cx, tx.y - cy, (tx.h_m - cz) + 0.5);
  group.add(s);

  const headingRad = (tx.heading_deg || 0) * Math.PI / 180.0;
  const dir = new THREE.Vector3(Math.sin(headingRad), Math.cos(headingRad), 0).normalize();
  const apexX = tx.x - cx;
  const apexY = tx.y - cy;
  const apexZ = (tx.h_m - cz) + 0.5;
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

    // Check GPU status
    const gpuStatus = await fetch('/gpu_status').then(r => r.json());
    const statusEl = document.getElementById('gpuStatus');
    statusEl.innerText = `(${gpuStatus.status})`;
    if(gpuStatus.status.includes('GPU')) {
      statusEl.className = 'status-good';
    } else {
      statusEl.className = 'status-warn';
    }

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
        else if(m.type === 'heatmap') {
          renderHeatmap(m.grid);
          if(m.simulation_method) {
            document.getElementById('simStatus').innerHTML = `<span class="status-good">Complete (${m.simulation_method}, ${m.simulation_time?.toFixed(1)}s)</span>`;
          }
        }
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
        const res = await fetch('/load_bbox', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({cx: e, cy: n, r: r})
        });
        if(!res.ok) throw new Error(`HTTP ${res.status}`);
        const info = await res.json();
        streamFullPointcloud();
        if(info.client_points && Array.isArray(info.client_points) && info.client_points.length > 0){
          await updatePointCloudFromJSON({"points": info.client_points, "centroid": info.centroid, "bounds": info.bounds});
          document.getElementById('ptcount').innerText = info.client_sample_count || info.count || 'loaded';
        } else {
          document.getElementById('ptcount').innerText = info.count || 'loading';
        }
      }catch(err){
        showError('Load area failed: ' + (err && err.message ? err.message : String(err)));
      }
    };

    document.getElementById('setGrid').onclick = async ()=>{
      const gr = parseFloat(document.getElementById('grid_res').value || '2.0');
      const gm = parseFloat(document.getElementById('grid_margin').value || '20.0');
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
          // Use server-returned final_h (absolute elevation) for the visual so it matches the simulator.
          if(parsed && parsed.final_h !== undefined){
            tx.h_m = parseFloat(parsed.final_h);
            if(parsed.ground_z !== undefined) tx.ground_z = parseFloat(parsed.ground_z);
          }
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
        document.getElementById('simStatus').innerHTML = '<span class="status-warn">Running...</span>';
        document.getElementById('progressBarContainer').style.display = 'block';
        document.getElementById('progressBar').style.width = '0%';
        const res = await fetch('/simulate', {method:'POST'});
        if(!res.ok){
          const txt = await res.text();
          showError('Simulate failed: ' + txt);
          document.getElementById('simulate').disabled = false;
          document.getElementById('simStatus').innerText = 'Failed';
          return;
        }
        const payload = await res.json();
        if(payload && payload.grid) renderHeatmap(payload.grid);
        if(payload.simulation_method) {
          document.getElementById('simStatus').innerHTML = `<span class="status-good">Complete (${payload.simulation_method}, ${payload.simulation_time?.toFixed(1)}s)</span>`;
        }
        document.getElementById('simulate').disabled = false;
        setTimeout(()=>{ document.getElementById('progressBarContainer').style.display = 'none'; }, 1200);
      }catch(err){
        showError('Simulate failed: ' + (err && err.message ? err.message : String(err)));
        document.getElementById('simulate').disabled = false;
        document.getElementById('simStatus').innerText = 'Failed';
        document.getElementById('progressBarContainer').style.display = 'none';
      }
    };

  }catch(err){
    showError('Init error: ' + (err && err.message ? err.message : String(err)));
  }
}

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
    const newPositions = [];
    const newMeta = { count: 0, centroid: [0,0,0], bounds: [0,0,0,0,0,0] };
    let tempPoints = null;
    let tempMaterial = null;
    const SWAP_THRESHOLD = 2000;
    const FLUSH_CHUNK = 1000;
    let flushCounter = 0;

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
            newMeta.count = obj._meta.count || newPositions.length;
            newMeta.centroid = obj._meta.centroid || newMeta.centroid;
            newMeta.bounds = obj._meta.bounds || newMeta.bounds;
            if(tempPoints && newPositions.length > 0){
              const positions = new Float32Array(newPositions.length * 3);
              for(let i=0;i<newPositions.length;i++){
                positions[i*3+0] = newPositions[i][0] - newMeta.centroid[0];
                positions[i*3+1] = newPositions[i][1] - newMeta.centroid[1];
                positions[i*3+2] = newPositions[i][2] - newMeta.centroid[2];
              }
              const geom = new THREE.BufferGeometry();
              geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
              const spanX = newMeta.bounds[1] - newMeta.bounds[0];
              const spanY = newMeta.bounds[3] - newMeta.bounds[2];
              const spanZ = newMeta.bounds[5] - newMeta.bounds[4];
              const span = Math.max(1, spanX, spanY, spanZ);
              const size = Math.max(0.6, span / 2000);
              const mat = new THREE.PointsMaterial({size:size, sizeAttenuation:true, color:0xffffff});
              const newPc = new THREE.Points(geom, mat);
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
            return;
          } else {
            if(typeof obj.x === 'number' && typeof obj.y === 'number' && typeof obj.z === 'number'){
              newPositions.push([obj.x, obj.y, obj.z]);
              flushCounter++;
            }
            if(tempPoints === null && newPositions.length >= Math.min(SWAP_THRESHOLD, FLUSH_CHUNK)){
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
              scene.add(tempPoints);
            }
            if(tempPoints && flushCounter >= FLUSH_CHUNK){
              const curLen = tempPoints.geometry.getAttribute('position').array.length / 3;
              const addLen = Math.max(0, newPositions.length - curLen);
              if(addLen > 0){
                const newArr = new Float32Array((curLen + addLen) * 3);
                newArr.set(tempPoints.geometry.getAttribute('position').array, 0);
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
              if(newPositions.length >= SWAP_THRESHOLD){
                const positions = new Float32Array(newPositions.length * 3);
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
                if(pcPoints){
                  scene.remove(pcPoints);
                  if(pcPoints.geometry) pcPoints.geometry.dispose();
                }
                if(tempPoints){
                  scene.remove(tempPoints);
                  if(tempPoints.geometry) tempPoints.geometry.dispose();
                  tempPoints = null;
                  tempMaterial = null;
                }
                pcPoints = newPc;
                document.getElementById('ptcount').innerText = newPositions.length;
              }
            }
          }
        }
      }
      ({ value: chunk, done: readerDone } = await reader.read());
    }

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
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Serve the main HTML page.
    """
    return HTML_PAGE

@app.get("/gpu_status")
async def gpu_status():
    """
    Check and report GPU acceleration status.

    global server_state
    
    Returns:
    --------
    JSONResponse: A JSON response indicating GPU status.
    """
    status = {
        "cupy": HAS_CUPY,
        "gpu_initialized": server_state.get("gpu_initialized", False)
    }
    
    if HAS_CUPY:
        status["status"] = "GPU Acceleration Available"
    else:
        status["status"] = "NO GPU!"
    
    return JSONResponse(status)

@app.get("/pointcloud")
async def pointcloud():
    """
    Serve a sampled and centered point cloud for client visualization.

    Returns:
    --------
    JSONResponse: A JSON response containing centered point cloud data.
    """
    global server_state
    pts = server_state["pts"]
    if pts is None or pts.shape[0] == 0:
        return JSONResponse({"points": [], "centroid": [0,0,0], "bounds": [0,0,0,0,0,0]})
    xs = pts[:,0]; ys = pts[:,1]; zs = pts[:,2]
    cx = float(xs.mean()); cy = float(ys.mean()); cz = float(zs.mean())
    bounds = [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()), float(zs.min()), float(zs.max())]
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
    Stream the full point cloud within the last loaded bounding box.

    Returns:
    --------
    StreamingResponse: A streaming response yielding point cloud data in NDJSON format.
    """
    global server_state, DB_PATH
    bbox = server_state.get("last_bbox")
    if not bbox:
        return JSONResponse({"error": "no bbox loaded - call /load_bbox first"}, status_code=400)
    gen = stream_full_points_generator(DB_PATH, bbox)
    return StreamingResponse(gen, media_type="application/x-ndjson")

@app.get("/antenna_configs")
async def antenna_configs():
    """
    List available antenna configurations with key specs.

    Returns:
    --------
    JSONResponse: A JSON response containing a list of antenna configurations.
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
    Get full specification for a named antenna configuration.

    Parameters:
    -----------
    name (str): The name of the antenna configuration.

    Returns:
    --------
    JSONResponse: A JSON response containing the antenna configuration specification.
    """
    global server_state
    spec = server_state.get("antenna_configs", {}).get(name)
    if spec is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(spec)

@app.post("/reload_configs")
async def reload_configs():
    """
    Reload antenna configurations from disk.

    Returns:
    --------
    JSONResponse: A JSON response indicating success and count of configurations.
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
    Load point cloud data within a specified bounding box and initialize GPU simulation.

    Parameters:
    -----------
    req (Request): The incoming request containing JSON body with 'cx', 'cy', and 'r'.

    Returns:
    --------
    JSONResponse: A JSON response containing load information and client sample points.
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

    # Sampled load for server-side KD-tree build.
    pts, meta = load_pointcloud_from_db(DB_PATH, bbox=(minx, miny, maxx, maxy),
                                       max_points=None, max_server_points=1500000)
    if pts.shape[0] == 0:
        return JSONResponse({"count": 0, "message": "no points in area"}, status_code=200)
    
    # Initialize GPU simulation with the loaded points.
    gpu_simulator.initialize(pts)
    server_state["gpu_initialized"] = True
    
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
    
    # Prepare client sample.
    client_pts, _ = load_pointcloud_from_db(DB_PATH, bbox=(minx, miny, maxx, maxy), max_points=120000)
    if client_pts.shape[0] > 0:
        cx_c = float(client_pts[:,0].mean()); cy_c = float(client_pts[:,1].mean()); cz_c = float(client_pts[:,2].mean())
        centered_client = [{"x": float(x - cx_c), "y": float(y - cy_c), "z": float(z - cz_c)} for (x,y,z) in client_pts]
    else:
        centered_client = []
    server_state["client_sample"] = {"points": centered_client, "centroid": centroid, "bounds": bounds}
    
    print(f"[LOAD] GPU initialized with {cnt} points, KD-tree built, returning {len(centered_client)} client sample points")
    return JSONResponse({"count": cnt, "centroid": centroid, "bounds": bounds, "client_sample_count": len(centered_client), "client_points": centered_client})

@app.post("/set_grid")
async def set_grid(req: Request):
    """
    Set grid resolution and margin for simulations.

    Parameters:
    -----------
    req (Request): The incoming request containing JSON body with 'grid_res' and 'grid_margin'.

    Returns:
    --------
    JSONResponse: A JSON response confirming the updated grid parameters.
    """
    global server_state
    data = await req.json()
    try:
        gr = float(data.get("grid_res", server_state.get("grid_res", 2.0)))
        gm = float(data.get("grid_margin", server_state.get("grid_margin", 20.0)))
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
    Add a transmitter with automatic ground elevation detection.

    Parameters:
    -----------
    req (Request): The incoming request containing JSON body with transmitter parameters.

    Returns:
    --------
    JSONResponse: A JSON response indicating success or failure of adding the transmitter.
    """
    global server_state
    try:
        tx = await req.json()
    except Exception as e:
        msg = f"Invalid JSON body: {e}"
        print("[TX] add_tx: invalid json:", e)
        return JSONResponse({"error": msg}, status_code=400)

    # Pull user-provided XY and requested height. (interpreted as AGL)
    try:
        tx_x = float(tx.get("x"))
        tx_y = float(tx.get("y"))
    except Exception as e:
        msg = f"Invalid TX X/Y: {e}"
        print("[TX] bad coords:", e)
        return JSONResponse({"error": msg}, status_code=400)

    # Default user-provided AGL (height field in UI) - treat this as height above ground.
    try:
        user_h_agl = float(tx.get("h_m", tx.get("height_agl", 2.0)))
    except Exception:
        user_h_agl = 2.0

    # Find ground elevation at (tx_x, tx_y).
    ground_z = 0.0
    used_method = "none"
    try:
        pts = server_state.get("pts")
        kdtree = server_state.get("kdtree")
        # Prefer KD-tree on point cloud (kdtree was built on pts[:, :2]).
        if kdtree is not None and pts is not None and pts.shape[0] > 0:
            # Scipy cKDTree.query returns (dist, idx).
            try:
                d, idx = kdtree.query([tx_x, tx_y], k=1)
                # idx may be array-like
                if hasattr(idx, "__len__"):
                    idx0 = int(idx[0])
                else:
                    idx0 = int(idx)
                ground_z = float(pts[idx0, 2])
                used_method = "kdtree_pointcloud"
            except Exception:
                # Some KDTree implementations expect shape (N,2) input; try alternative.
                try:
                    d, idx = kdtree.query(np.array([[tx_x, tx_y]]), k=1)
                    idx0 = int(np.asarray(idx).ravel()[0])
                    ground_z = float(pts[idx0, 2])
                    used_method = "kdtree_pointcloud"
                except Exception:
                    used_method = "kdtree_failed"
                    ground_z = 0.0
    except Exception as e:
        print("[TX] ground_z lookup exception:", e)
        ground_z = 0.0
        used_method = "exception"

    # Compose final TX height = ground_z + user-provided AGL.
    final_h = float(ground_z) + float(user_h_agl)

    # Normalize and sanitize TX params.
    try:
        tx["x"] = float(tx_x)
        tx["y"] = float(tx_y)
        tx["h_m"] = float(final_h)           # Absolute elevation used by simulator.
        tx["height_agl"] = float(user_h_agl) # User-specified AGL.
        tx["ground_z"] = float(ground_z)     # Detected ground elevation at XY.
        tx["power_dbm"] = float(tx.get("power_dbm", 20.0))
        tx["freq_hz"] = float(tx.get("freq_hz", server_state.get("freq_hz", 2.4e9)))
        tx["heading_deg"] = float(tx.get("heading_deg", tx.get("heading", 0.0) or 0.0))
    except Exception as e:
        msg = f"Invalid transmitter parameter types: {e}"
        print("[TX] parameter conversion error:", e)
        return JSONResponse({"error": msg}, status_code=400)

    # Merge antenna config if present. (keeps previous behavior)
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

    # Save and print debug.
    try:
        server_state["txs"].append(tx)
        print(f"[TX] added: x={tx['x']:.2f} y={tx['y']:.2f} ground_z={ground_z:.3f} user_agl={user_h_agl:.3f} final_h={final_h:.3f} method={used_method}")
        return JSONResponse({"ok": True, "total_txs": len(server_state["txs"]), "ground_z": ground_z, "final_h": final_h, "used_method": used_method})
    except Exception as exc:
        tb = traceback.format_exc()
        print("[TX] exception adding tx:", tb)
        return JSONResponse({"error": "server exception when adding transmitter", "detail": str(exc)}, status_code=500)

@app.post("/clear_txs")
async def clear_txs():
    """
    Clear all defined transmitters.

    Returns:
    --------
    JSONResponse: A JSON response indicating success.
    """
    server_state["txs"].clear()
    server_state["client_sample"] = None
    print("[TX] cleared")
    return JSONResponse({"ok": True})

@app.post("/simulate")
async def simulate():
    """
    Run the RF coverage simulation using GPU acceleration with occlusion.

    Returns:
    --------
    JSONResponse: A JSON response containing the simulation results or error message.
    """
    global server_state, gpu_simulator
    pts = server_state["pts"]
    kdtree = server_state.get("kdtree")
    
    if pts is None or pts.shape[0] == 0:
        return JSONResponse({"error": "no points loaded - use 'Load area' first"}, status_code=400)
    if kdtree is None:
        return JSONResponse({"error": "KD-tree not built"}, status_code=400)
        
    txs = server_state["txs"]
    if not txs:
        return JSONResponse({"error": "no transmitters defined - click 'Add transmitter' and place in scene"}, status_code=400)
        
    if server_state["grid_x"] is None:
        xs = pts[:,0]; ys = pts[:,1]
        grid_margin_m = server_state.get("grid_margin", 20.0)
        grid_res_m = server_state.get("grid_res", 2.0)
        minx, maxx = float(xs.min()-grid_margin_m), float(xs.max()+grid_margin_m)
        miny, maxy = float(ys.min()-grid_margin_m), float(ys.max()+grid_margin_m)
        nx = max(8, int(math.ceil((maxx-minx)/grid_res_m)))
        ny = max(8, int(math.ceil((maxy-miny)/grid_res_m)))
        server_state["grid_x"] = np.linspace(minx, maxx, nx)
        server_state["grid_y"] = np.linspace(miny, maxy, ny)
    
    grid_x = server_state["grid_x"]
    grid_y = server_state["grid_y"]
    freq_hz = server_state.get("freq_hz", 2.4e9)

    start = time.time()
    print(f"[SIM] Starting simulation with {len(txs)} transmitters, grid {len(grid_x)}x{len(grid_y)}")
    
    try:
        if HAS_CUPY:
            method = "GPU" 
        else:
            raise RuntimeError("Cannot run GPU simulation without CuPy")
        
        rss = gpu_simulator.compute_coverage_gpu(txs, grid_x, grid_y, freq_hz, kdtree, pts)
    except Exception as e:
        print(f"[SIM] Simulation failed: {e}")
        traceback.print_exc()
        return JSONResponse({"error": f"Simulation failed: {str(e)}"}, status_code=500)

    elapsed = time.time() - start
    print(f"[SIM] {method} compute with occlusion finished in {elapsed:.2f}s")
    
    if rss is None:
        return JSONResponse({"error": "simulation produced no result"}, status_code=500)
        
    rssi_flat = [float(x) for x in rss.ravel().tolist()]
    payload = {
        "type": "heatmap", 
        "grid": {
            "xs": [float(x) for x in grid_x], 
            "ys": [float(y) for y in grid_y], 
            "rssi": rssi_flat
        },
        "simulation_method": method + " with Occlusion",
        "simulation_time": elapsed
    }
    
    try:
        await broadcast_message(json.dumps(payload))
    except Exception as e:
        print("[SIM] broadcast error:", e)
        
    return JSONResponse(payload)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time client updates.
    
    Parameters:
    -----------
    websocket (WebSocket): The WebSocket connection instance.
    
    Returns:
    --------
    None
    """
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
    """
    Broadcast the current count of connected WebSocket clients.

    Returns:
    --------
    None
    """
    msg = json.dumps({"type":"clients", "count": len(connected_websockets)})
    await broadcast_message(msg)

async def broadcast_message(msg: str):
    """
    Broadcast a message to all connected WebSocket clients.

    Parameters:
    -----------
    msg (str): The message to broadcast.

    Returns:
    --------
    None
    """
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
    "grid_res": 2.0,
    "grid_margin": 20.0,
    "freq_hz": 2.4e9,
    "executor": None,
    "num_workers": (os.cpu_count() or 4),
    "antenna_configs": {},
    "client_sample": None,
    "last_bbox": None,
    "gpu_initialized": False
}

# ------------------------
# Entrypoint
# ------------------------
def main():
    global DB_PATH, server_state
    parser = argparse.ArgumentParser(description="Run LiDAR RF on-demand simulator + viewer with FIXED GPU acceleration and occlusion")
    parser.add_argument("db", help="Path to SQLite DB created by lidar_preprocess.py")
    parser.add_argument("--workers", type=int, default=(os.cpu_count() or 4), help="Number of worker processes for simulation")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    DB_PATH = args.db
    server_state["num_workers"] = max(1, int(args.workers))
    server_state["pts"] = np.zeros((0,3))

    # Load antenna configs at startup.
    try:
        server_state["antenna_configs"] = load_antenna_configs()
    except Exception as e:
        print("[CONFIG] failed to load antenna configs:", e)
        server_state["antenna_configs"] = {}

    print("Use the web UI 'Load area' to populate a subset and initialize GPU simulation")
    print(f"Starting server at http://{args.host}:{args.port} ...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()