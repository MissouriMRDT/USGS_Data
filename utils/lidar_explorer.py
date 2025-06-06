#!/usr/bin/env python3

# ******************************************************************************
#  @file      lidar_explorer.py
#  @brief     Interactive 3D viewer for LiDAR point clouds from SQLite database.
# 
#  This script provides an interactive 3D viewer for point cloud data, with
#  controls for loading different regions and filtering by traversability score.
#
#  Example usage:
#      $ python lidar_explorer.py /path/to/lidar.db --region 1000 2000 1200 2200
# 
#  @author     ClayJay3 (claytonraycowen@gmail.com)
#  @date       2025-05-31
#  @copyright  Copyright Mars Rover Design Team 2025 – All Rights Reserved
# ******************************************************************************

import sqlite3
import numpy as np
import open3d as o3d
import argparse
import sys
import tkinter as tk
from tkinter import ttk
import threading
import time
import queue  # Add queue for thread-safe communication.

class PointCloudViewer:
    """
    Interactive 3D viewer for LiDAR point clouds.
    """

    def __init__(self, db_path, initial_region=None, score_thresh=0.0):
        """
        Initialize the point cloud viewer.

        Args:
        -----
        db_path (str): Path to the SQLite database containing LiDAR points.
        initial_region (list, optional): Initial region to load as [xmin, xmax, ymin, ymax].
        score_thresh (float): Minimum traversability score to display points.
        If None, defaults to [0, 100, 0, 100].

        Returns:
        --------
        None
        """
        self.db_path = db_path
        self.score_thresh = score_thresh
        if initial_region is None:
            self.region = [0, 100, 0, 100]
        else:
            self.region = initial_region
        self.region_size = 100
        self.vis = None
        self.pcd = None
        self.point_size = 2
        self.is_running = True
        
        # Command queue for thread-safe communication.
        self.command_queue = queue.Queue()
        
        # Set up the GUI first.
        self.setup_gui()
        
        # Then load the point cloud in a separate thread.
        threading.Thread(target=self.initialize_visualization, daemon=True).start()
    
    def setup_gui(self):
        """
        Set up the Tkinter GUI for controls.

        This method creates the main window, region controls, score threshold slider,
        point size controls, and status messages. It also binds the necessary events
        for user interactions.

        Args:
        -----
        None

        Returns:
        --------
        None
        """
        self.root = tk.Tk()
        self.root.title("LiDAR Explorer Controls")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Main frame.
        mainframe = ttk.Frame(self.root, padding="10")
        mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Region control frame.
        region_frame = ttk.LabelFrame(mainframe, text="Region Controls")
        region_frame.grid(column=0, row=0, sticky=(tk.W, tk.E))
        
        # X coordinate.
        ttk.Label(region_frame, text="X:").grid(column=0, row=0, sticky=tk.W)
        self.x_var = tk.DoubleVar(value=self.region[0] + (self.region[1] - self.region[0]) / 2)
        x_entry = ttk.Entry(region_frame, width=10, textvariable=self.x_var)
        x_entry.grid(column=1, row=0, sticky=(tk.W, tk.E))
        
        # Y coordinate.
        ttk.Label(region_frame, text="Y:").grid(column=2, row=0, sticky=tk.W)
        self.y_var = tk.DoubleVar(value=self.region[2] + (self.region[3] - self.region[2]) / 2)
        y_entry = ttk.Entry(region_frame, width=10, textvariable=self.y_var)
        y_entry.grid(column=3, row=0, sticky=(tk.W, tk.E))
        
        # Size.
        ttk.Label(region_frame, text="Size:").grid(column=4, row=0, sticky=tk.W)
        self.size_var = tk.DoubleVar(value=self.region_size)
        size_entry = ttk.Entry(region_frame, width=10, textvariable=self.size_var)
        size_entry.grid(column=5, row=0, sticky=(tk.W, tk.E))
        
        # Load button.
        load_button = ttk.Button(region_frame, text="Load Region", command=self.on_load_button)
        load_button.grid(column=6, row=0, sticky=tk.W)
        
        # Score threshold frame.
        score_frame = ttk.LabelFrame(mainframe, text="Traversability Score")
        score_frame.grid(column=0, row=1, sticky=(tk.W, tk.E))
        
        # Score slider.
        ttk.Label(score_frame, text="Min Score:").grid(column=0, row=0, sticky=tk.W)
        self.score_var = tk.DoubleVar(value=self.score_thresh)
        score_slider = ttk.Scale(score_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                variable=self.score_var, length=200)
        score_slider.grid(column=1, row=0, sticky=(tk.W, tk.E))
        self.score_label = ttk.Label(score_frame, text=f"{self.score_thresh:.2f}")
        self.score_label.grid(column=2, row=0, sticky=tk.W)
        score_slider.bind("<Motion>", self.on_score_change)
        
        # Apply button.
        apply_score = ttk.Button(score_frame, text="Apply", command=self.on_apply_score)
        apply_score.grid(column=3, row=0, sticky=tk.W)
        
        # Point size control.
        size_frame = ttk.LabelFrame(mainframe, text="Point Size")
        size_frame.grid(column=0, row=2, sticky=(tk.W, tk.E))
        
        ttk.Label(size_frame, text="Size:").grid(column=0, row=0, sticky=tk.W)
        self.point_size_var = tk.IntVar(value=self.point_size)
        point_size_slider = ttk.Scale(size_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                      variable=self.point_size_var, length=200)
        point_size_slider.grid(column=1, row=0, sticky=(tk.W, tk.E))
        self.point_size_label = ttk.Label(size_frame, text=f"{self.point_size}")
        self.point_size_label.grid(column=2, row=0, sticky=tk.W)
        point_size_slider.bind("<Motion>", self.on_point_size_change)
        
        # Apply button for point size.
        apply_size = ttk.Button(size_frame, text="Apply", command=self.on_apply_point_size)
        apply_size.grid(column=3, row=0, sticky=tk.W)
        
        # Status frame.
        status_frame = ttk.Frame(mainframe)
        status_frame.grid(column=0, row=3, sticky=(tk.W, tk.E))
        
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.grid(column=0, row=0, sticky=tk.W)
        
        # Add padding to all children.
        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)
            if isinstance(child, ttk.LabelFrame):
                for grandchild in child.winfo_children():
                    grandchild.grid_configure(padx=5, pady=5)
    
    def initialize_visualization(self):
        """
        Initialize the Open3D visualizer in a separate thread.
        This method sets up the Open3D visualizer, loads the initial point cloud,
        and starts the main loop for rendering.
        This method runs in a separate thread to avoid blocking the Tkinter main loop.
        
        Args:
        -----
        None

        Returns:
        --------
        None
        """
        try:
            self.status_var.set("Loading initial point cloud...")
            # Create a visualizer object.
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="LiDAR Point Cloud Viewer", width=1024, height=768)
            
            # Set render options.
            opt = self.vis.get_render_option()
            opt.background_color = np.array([0.1, 0.1, 0.1])
            opt.point_size = self.point_size
            
            # Initial point cloud load.
            self._load_and_display_region(self.region, self.score_thresh)
            
            # NON-BLOCKING: Use polling approach with command processing.
            while self.is_running:
                # Process any pending commands from the UI thread.
                self.process_commands()
                
                if not self.vis.poll_events():
                    break
                self.vis.update_renderer()
                time.sleep(0.01)  # Short sleep to prevent high CPU usage.
                
            # When the visualizer is closed.
            self.is_running = False
            if self.root.winfo_exists():
                self.root.destroy()
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Visualization error: {str(e)}")
    
    def process_commands(self):
        """
        Process commands from the command queue in a thread-safe manner.
        This method checks the command queue for any pending commands and executes them.
        It handles commands like loading a region or updating the point size.
        This method is called periodically from the visualization thread to ensure
        that commands from the UI are processed without blocking the main loop.

        Args:
        -----
        None

        Returns:
        --------
        None
        """
        try:
            # Process all available commands.
            while not self.command_queue.empty():
                cmd, args = self.command_queue.get_nowait()
                if cmd == 'load_region':
                    region, score_thresh = args
                    self._load_and_display_region(region, score_thresh)
                elif cmd == 'update_point_size':
                    point_size = args
                    opt = self.vis.get_render_option()
                    opt.point_size = float(point_size)
                    self.vis.update_renderer()
                self.command_queue.task_done()
        except Exception as e:
            self.status_var.set(f"Error processing command: {str(e)}")
            print(f"Command processing error: {str(e)}")
    
    def _load_and_display_region(self, region, score_thresh):
        """
        Load and display points in the specified region with the given score threshold.
        This method retrieves points from the database within the specified region
        and score threshold, creates a point cloud, and updates the Open3D visualizer.
        
        Args:
        -----
        region (list): The region to load points from as [xmin, xmax, ymin, ymax].
        score_thresh (float): Minimum traversability score to display points.

        Returns:
        --------
        None
        """
        if not self.vis:
            return
        
        xmin, xmax, ymin, ymax = region
        coords, scores = self.load_points(xmin, xmax, ymin, ymax, score_thresh)
        
        if coords.shape[0] == 0:
            return
        
        # Create a new point cloud.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        
        # Color by traversability score. (red=0, green=1)
        colors = np.zeros((len(scores), 3))
        colors[:, 0] = 1.0 - scores  # Red channel.
        colors[:, 1] = scores        # Green channel.
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Update the point cloud.
        self.vis.clear_geometries()
        self.vis.add_geometry(pcd)
        self.vis.update_geometry(pcd)
        
        # Reset view to see all points.
        self.vis.reset_view_point(True)
        
        # Store the point cloud.
        self.pcd = pcd

        # Update render options.
        opt = self.vis.get_render_option()
        opt.point_size = self.point_size
        self.vis.update_renderer()
    
    def load_points(self, xmin, xmax, ymin, ymax, score_thresh=0.0):
        """
        Load points from the SQLite database within the specified region and score threshold.
        This method connects to the database, retrieves points that fall within the 
        specified region and have a traversability score above the threshold.

        Args:
        -----
        xmin (float): Minimum X coordinate of the region.
        xmax (float): Maximum X coordinate of the region.
        ymin (float): Minimum Y coordinate of the region.
        ymax (float): Maximum Y coordinate of the region.
        score_thresh (float): Minimum traversability score to include points.

        Returns:
        --------
        tuple: A tuple containing:
            - coords (np.ndarray): Array of point coordinates (N, 3).
            - scores (np.ndarray): Array of traversability scores (N,).
        If no points are found, returns empty arrays.
        If an error occurs, returns zeros and logs the error.
        """
        self.status_var.set(f"Loading points... {xmin:.1f},{ymin:.1f} to {xmax:.1f},{ymax:.1f}")
        try:
            conn = sqlite3.connect(self.db_path)
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
            if not rows:
                self.status_var.set("No points found in this region")
                return np.zeros((0, 3)), np.array([])
            
            arr = np.array(rows, dtype=float)
            coords = arr[:, 0:3]
            scores = arr[:, 3]
            self.status_var.set(f"Loaded {len(rows)} points")
            return coords, scores
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Database error: {str(e)}")
            return np.zeros((0, 3)), np.array([])
    
    def on_load_button(self):
        """
        Handle the Load Region button click.
        This method retrieves the coordinates and size from the input fields,
        calculates the region to load, and queues the command to load the points.

        Args:
        -----
        None

        Returns:
        --------
        None
        """
        try:
            x = self.x_var.get()
            y = self.y_var.get()
            size = self.size_var.get()
            half_size = size / 2
            region = [x - half_size, x + half_size, y - half_size, y + half_size]
            self.region = region  # Update the region.
            self.region_size = size
            
            # Queue the command instead of directly calling.
            self.command_queue.put(('load_region', (region, self.score_thresh)))
            self.status_var.set(f"Loading region around ({x}, {y}) with size {size}...")
        except Exception as e:
            self.status_var.set(f"Error loading region: {str(e)}")
            print(f"Load region error: {str(e)}")
    
    def on_score_change(self, event):
        """
        Handle score slider changes.
        This method updates the score label when the slider is moved.
        
        Args:
        -----
        event: The event triggered by the slider movement.

        Returns:
        --------
        None
        """
        self.score_label.config(text=f"{self.score_var.get():.2f}")
    
    def on_apply_score(self):
        """
        Apply the score threshold.
        This method retrieves the score threshold from the slider,
        queues the command to load points with the new score threshold,
        and updates the status message.

        Args:
        -----
        None

        Returns:
        --------
        None
        """
        try:
            self.score_thresh = self.score_var.get()
            # Queue the command
            self.command_queue.put(('load_region', (self.region, self.score_thresh)))
            self.status_var.set(f"Applying score threshold: {self.score_thresh:.2f}...")
        except Exception as e:
            self.status_var.set(f"Error applying score: {str(e)}")
            print(f"Apply score error: {str(e)}")
    
    def on_point_size_change(self, event):
        """
        Handle point size slider changes.
        This method updates the point size label when the slider is moved.

        Args:
        -----
        event: The event triggered by the slider movement.

        Returns:
        --------
        None
        """
        self.point_size_label.config(text=f"{self.point_size_var.get()}")
    
    def on_apply_point_size(self):
        """
        Apply the point size setting.
        This method retrieves the point size from the slider,
        queues the command to update the point size in the visualizer,
        and updates the status message.

        Args:
        -----
        None

        Returns:
        --------
        None
        """
        try:
            self.point_size = self.point_size_var.get()
            # Queue the command.
            self.command_queue.put(('update_point_size', self.point_size))
            self.status_var.set(f"Updating point size to {self.point_size}...")
        except Exception as e:
            self.status_var.set(f"Error applying point size: {str(e)}")
            print(f"Apply point size error: {str(e)}")
    
    def on_closing(self):
        """
        Handle window closing.
        This method is called when the user closes the Tkinter window.
        It stops the visualization thread and closes the Open3D visualizer.

        Args:
        -----
        None

        Returns:
        --------
        None
        """
        self.is_running = False
        if self.vis:
            self.vis.destroy_window()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        # Start the Tkinter event loop.
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(
        description='Interactive 3D viewer for LiDAR point clouds'
    )
    parser.add_argument('db_path', help='Path to the SQLite DB file')
    parser.add_argument('--region', nargs=4, type=float, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                      help='Initial region to load (xmin xmax ymin ymax)')
    parser.add_argument('--score_thresh', type=float, default=0.0,
                      help='Minimum traversability score to display (default: 0.0)')
    
    args = parser.parse_args()
    
    viewer = PointCloudViewer(
        args.db_path,
        initial_region=args.region,
        score_thresh=args.score_thresh
    )
    viewer.run()

if __name__ == '__main__':
    main()