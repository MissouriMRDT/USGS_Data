# USGS_Data  
*A LiDAR Preprocessing Tool by the Missouri S&T Mars Rover Design Team*

![LiDAR Point Cloud Visualization](res/Rolla_Fugitive2.png)

This repository provides Python utilities for converting raw USGS **LAS 1.4** LiDAR files into a structured **SQLite database**. It is designed to support autonomous navigation, terrain simulation, and mission planning tasks by making point cloud data queryable and region-aware.

---

## 📁 Repository Structure

```

USGS_Data/
├── data/
│   ├── las/            # Raw USGS .las files organized by region
│   │   ├── MDRS/       # Example: Mars Desert Research Station laz files
|   |   ├── Fugitive/   # Example: Fugitive Beach laz files
|   |   └── SDELC/      # Example: General Rolla laz files (Golf Course, Turf Fields, SDELC, etc)
│   └── sqlite/         # Output SQLite databases (e.g., MDRS.db)
├── utils/
|   ├── plan_and_plot.py # Load a created database and plan a path using the trav_score
│   ├── lidar_preprocess.py  # LAS → SQLite core logic
|   ├── lidar_explorer.py # Open3D viewer of lidar data from the created database
│   └── main.py        # Command-line interface entry point
├── res/                # Directory for images and resources
│   └── *.png # Example image file
├── .gitattributes
├── .gitignore
└── README.md
````

---

## 🚀 Getting Started

### 🔧 Build the Tool

Make sure you have `python3` and `libsqlite3-dev` installed:

```bash
sudo apt update
sudo apt install python3 libsqlite3-dev
````

Then install the required Python packages from the Pipfile:

```bash
pipenv install
```

---

## 🧠 Features

* ✔️ Supports LAS 1.4 Format 6 binary files
* ✔️ Extracts scaled XYZ + UTM zone from headers and VLR
* ✔️ Outputs to a normalized SQLite table with:

  * Easting, Northing, Altitude
  * UTM Zone
  * Point Classification (human-readable)
* ✔️ Automatically creates indexes and batches inserts for performance

---

## 💡 Best Practices

* Store LAS files in `data/las/<region>/`
* Store DBs in `data/sqlite/<region>.db`
* Commit both LAS and DB files to Git for version tracking
* Avoid running this tool on non-USGS or non-LAS 1.4 data

---

## 🤖 Use Cases

* Preloading LiDAR terrain data into simulation
* Spatial queries in C++/Python for autonomous navigation
* Region-aware map segmentation and classification

---

## 📜 License

* Source code © 2025 Missouri S\&T Mars Rover Design Team
* USGS data is public domain