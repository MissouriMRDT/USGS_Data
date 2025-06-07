#!/usr/bin/env python3

# ******************************************************************************
#  @file      plan_and_plot.py
#  @brief     Plan a path over LiDAR terrain, heavily biasing for high trav_score.
# 
#  This script loads LiDAR points from a SQLite database, applies Dijkstra's algorithm
#  to find a path from a start to an end point, heavily penalizing low trav_score.
#  It then plots the terrain in 3D, coloring points by trav_score and overlaying the path.
# 
#  Example usage:
#      $ python plan_and_plot.py /path/to/lidar.db 1000.0 2000.0 1500.0 2500.0 --margin 50.0 --score_thresh 0.2 --k 8 --beta 5.0 --output path_plot.png
# 
#  @author     ClayJay3 (claytonraycowen@gmail.com)
#  @date       2025-05-31
#  @copyright  Copyright Mars Rover Design Team 2025 – All Rights Reserved
# ******************************************************************************

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import KDTree
import heapq
import argparse

def load_points(db_path, xmin, xmax, ymin, ymax, score_thresh=0.0):
    """
    Load points within bounding box [xmin, xmax] × [ymin, ymax]
    with trav_score >= score_thresh. Returns:
      - coords: (N×3) array of (easting, northing, altitude)
      - scores: (N) array of trav_score

    Args:
    -----
    db_path: str, path to the SQLite database file
    xmin, xmax: float, bounding box limits in easting
    ymin, ymax: float, bounding box limits in northing
    score_thresh: float, minimum trav_score to include (default 0.0)
    
    Returns:
    --------
    coords: np.ndarray, shape (N, 3) with (easting, northing, altitude)
    scores: np.ndarray, shape (N,) with trav_score values
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT p.easting, p.northing, p.altitude, p.trav_score
          FROM ProcessedLiDARPoints_idx AS idx
          JOIN ProcessedLiDARPoints AS p ON p.id = idx.id
         WHERE idx.min_x BETWEEN ? AND ?
           AND idx.min_y BETWEEN ? AND ?
           AND p.trav_score >= ?
    """, (xmin, xmax, ymin, ymax, score_thresh))
    rows = c.fetchall()
    conn.close()

    # If no points found, return empty arrays.
    if not rows:
        return np.zeros((0, 3)), np.array([])

    # Convert rows to numpy arrays.
    arr = np.array(rows, dtype=float)
    coords = arr[:, 0:3]
    scores = arr[:, 3]
    return coords, scores

def dijkstra_biased(coords_xy, scores, start_idx, end_idx, k=8, 
                    penalty_factor=5.0):
    """
    Run Dijkstra's algorithm on a graph of LiDAR points, where edge cost
    = distance(i,j) * (1 + penalty_factor*(1 - scores[j])).
    
    Args:
    -----
    coords_xy: (Nx2) array of (easting, northing)
    scores:   (N)     array of trav_score in [0,1]
    start_idx, end_idx: indices into coords_xy
    k: number of nearest neighbors per node
    penalty_factor: β in the cost formula—higher β strongly penalizes low trav_score.
    
    Returns: 
    --------
    list of vertex‐indices forming the path, or [] if unreachable.
    """
    N = coords_xy.shape[0]
    if N == 0 or start_idx >= N or end_idx >= N:
        return []

    # Build KDTree once.
    tree = KDTree(coords_xy)
    # Query (k+1) neighbors: the first neighbor is the point itself (distance=0).
    dists, neighs = tree.query(coords_xy, k=k+1)

    visited = [False]*N
    dist_to = [np.inf]*N
    prev = [-1]*N
    dist_to[start_idx] = 0.0

    pq = [(0.0, start_idx)]
    while pq:
        cur_cost, i = heapq.heappop(pq)
        if visited[i]:
            continue
        visited[i] = True
        if i == end_idx:
            break

        # For each neighbor j = neighs[i][1..k]:
        for m in range(1, k+1):
            j = neighs[i][m]
            if visited[j]:
                continue
            dij = dists[i][m]  # Euclidean distance(i,j)
            # Biased cost: distance * (1 + β*(1 - trav_score[j]))
            penalty = 1.0 + penalty_factor * (1.0 - scores[j])
            new_cost = cur_cost + dij * penalty
            if new_cost < dist_to[j]:
                dist_to[j] = new_cost
                prev[j] = i
                heapq.heappush(pq, (new_cost, j))

    # If end_idx was never reached, return empty path.
    if not visited[end_idx]:
        return []

    # Reconstruct path.
    path = []
    u = end_idx
    while u != -1:
        path.append(u)
        u = prev[u]
    path.reverse()
    return path

def find_nearest_index(coords_xy, point_xy):
    """
    Return the index of the point in coords_xy nearest to (point_xy).

    Args:
    -----
    coords_xy: (Nx2) array of (easting, northing)
    point_xy: tuple of (easting, northing) to find nearest to

    Returns:
    -------
    int: index of the nearest point in coords_xy
    """
    tree = KDTree(coords_xy)
    _, idx = tree.query(point_xy)
    return idx

def plot_terrain_and_path(coords, scores, path_indices, margin=None, score_thresh=None, k=None, beta=None, output_file=None):
    """
    3D plot of terrain (colored by trav_score) and overlay the path in cyan.

    Args:
    -----
    coords: (Nx3) array of (easting, northing, altitude)
    scores: (N) array of trav_score values
    path_indices: list of indices forming the path
    margin: float, margin around start/end points (for info box)
    score_thresh: float, minimum trav_score to include in plot
    k: int, number of nearest neighbors used in Dijkstra
    beta: float, penalty factor β for 1 - trav_score
    output_file: str, if set, save the plot to this PNG file

    Returns:
    -------
    None, displays the plot or saves it to output_file.
    """
    xs, ys, zs = coords[:,0], coords[:,1], coords[:,2]
    # Normalize trav_score for colormap.
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    cmap = LinearSegmentedColormap.from_list("GreenRed", ["red","green"])

    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection='3d')

    # Plot point cloud with alpha to let the path stand out.
    sc = ax.scatter(
        xs, ys, zs,
        c=norm_scores,
        cmap=cmap,
        s=2,
        alpha=0.4,
        depthshade=True
    )

    # Overlay the path in bright cyan.
    if path_indices:
        path_coords = coords[path_indices]
        ax.plot(
            path_coords[:,0], path_coords[:,1], path_coords[:,2],
            color='cyan',
            linewidth=3,
            solid_capstyle='round',
            zorder=10,
            label='Planned Path'
        )
        # Draw the waypoints as larger cyan dots with black edge.
        ax.scatter(
            path_coords[:,0], path_coords[:,1], path_coords[:,2],
            color='cyan',
            s=20,
            edgecolor='k',
            zorder=11
        )

        start_coord = path_coords[0]
        end_coord = path_coords[-1]
        
        # START
        ax.plot(
            [start_coord[0]], [start_coord[1]], [start_coord[2]],
            marker='o',
            color='gold',
            markersize=5,
            markeredgecolor='black',
            markeredgewidth=1.5,
            zorder=30,
            label=f'Start: ({start_coord[0]:.2f}, {start_coord[1]:.2f})'
        )

        # END
        ax.plot(
            [end_coord[0]], [end_coord[1]], [end_coord[2]],
            marker='o',
            color='magenta',
            markersize=5,
            markeredgecolor='black',
            markeredgewidth=1.5,
            zorder=30,
            label=f'End: ({end_coord[0]:.2f}, {end_coord[1]:.2f})'
        )


    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('Terrain (colored by trav_score) + Planned Path')
    plt.colorbar(sc, shrink=0.6, label='Normalized trav_score')

    # Equalize axis scaling so altitude does not look exaggerated.
    x_span = xs.max() - xs.min()
    y_span = ys.max() - ys.min()
    z_span = zs.max() - zs.min()
    try:
        ax.set_box_aspect((x_span, y_span, z_span))
    except AttributeError:
        # Fallback for older Matplotlib versions.
        max_span = max(x_span, y_span, z_span)
        x_mid = (xs.max() + xs.min()) / 2
        y_mid = (ys.max() + ys.min()) / 2
        z_mid = (zs.max() + zs.min()) / 2
        ax.set_xlim(x_mid - max_span/2, x_mid + max_span/2)
        ax.set_ylim(y_mid - max_span/2, y_mid + max_span/2)
        ax.set_zlim(z_mid - max_span/2, z_mid + max_span/2)

    ax.legend()

    # Create an annotation box with parameters
    info_text = (
        f"Margin: {margin:.1f} m\n"
        f"Score Threshold: {score_thresh:.2f}\n"
        f"k (Neighbors): {k}\n"
        f"β (Penalty): {beta:.2f}"
    )

    ax.text2D(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.7)
    )

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Saved 3D plot to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Plan a path over LiDAR terrain, heavily biasing for high trav_score'
    )
    parser.add_argument('db_path',    help='Path to the SQLite DB file')
    parser.add_argument('start_x',    type=float, help='Start easting (m)')
    parser.add_argument('start_y',    type=float, help='Start northing (m)')
    parser.add_argument('end_x',      type=float, help='End easting (m)')
    parser.add_argument('end_y',      type=float, help='End northing (m)')
    parser.add_argument('--margin',    type=float, default=50.0,
                        help='Bounding-box margin around start/end (m)')
    parser.add_argument('--score_thresh', type=float, default=0.0,
                        help='Minimum trav_score to include (default 0.0)')
    parser.add_argument('--k',         type=int,   default=8,
                        help='Number of nearest neighbors per node (default 8)')
    parser.add_argument('--beta',      type=float, default=5.0,
                        help='Penalty factor β for 1 - trav_score (default 5.0)')
    parser.add_argument('--output',    help='If set, save plot to this PNG')
    args = parser.parse_args()

    # Define bounding box around start/end + margin.
    xmin = min(args.start_x, args.end_x) - args.margin
    xmax = max(args.start_x, args.end_x) + args.margin
    ymin = min(args.start_y, args.end_y) - args.margin
    ymax = max(args.start_y, args.end_y) + args.margin

    print(f"\nLoading points in X∈[{xmin:.2f},{xmax:.2f}], Y∈[{ymin:.2f},{ymax:.2f}] "
          f"with trav_score ≥ {args.score_thresh:.2f}\n")
    coords3d, scores = load_points(
        args.db_path, xmin, xmax, ymin, ymax, args.score_thresh
    )
    if coords3d.shape[0] == 0:
        print("No points found in this bounding box. Exiting.")
        return

    coords_xy = coords3d[:, :2]  # only (easting, northing) for graph

    # Find nearest indices (in the subset) to the exact start/end coordinates.
    start_idx = find_nearest_index(coords_xy, (args.start_x, args.start_y))
    end_idx   = find_nearest_index(coords_xy, (args.end_x,   args.end_y))
    print(f"Found start → index {start_idx}, end → index {end_idx}")

    print("\nRunning biased Dijkstra (β = {:.2f}) …".format(args.beta))
    path_indices = dijkstra_biased(
        coords_xy, scores, start_idx, end_idx,
        k=args.k, penalty_factor=args.beta
    )
    if not path_indices:
        print("No feasible path found. Exiting.")
        return
    print(f"Path found with {len(path_indices)} waypoints.\n")

    # Plot the terrain + path.
    plot_terrain_and_path(
        coords3d, scores, path_indices,
        margin=args.margin, score_thresh=args.score_thresh, k=args.k, beta=args.beta,
        output_file=args.output
    )

if __name__ == '__main__':
    main()
