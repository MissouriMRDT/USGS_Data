# USGS_Data  
*A LiDAR Preprocessing Tool by the Missouri S&T Mars Rover Design Team*

This repository provides a C++ utility for converting raw USGS **LAS 1.4** LiDAR files into a structured **SQLite database**. It is designed to support autonomous navigation, terrain simulation, and mission planning tasks by making point cloud data queryable and region-aware.

---

## 📁 Repository Structure

```

USGS\_Data/
├── data/
│   ├── las/            # Raw USGS .las files organized by region
│   │   └── MDRS/       # Example: Mars Desert Research Station dataset
│   └── sqlite/         # Output SQLite databases (e.g., MDRS.db)
├── utils/
│   ├── LiDARLoader.cpp # LAS → SQLite core logic
│   ├── LiDARLoader.h   # Class definition and LAS structures
│   └── main.cpp        # Command-line interface entry point
├── .gitattributes
├── .gitignore
└── README.md

````

---

## 🚀 Getting Started

### 🔧 Build the Tool

Make sure you have `g++` and `libsqlite3-dev` installed:

```bash
sudo apt update
sudo apt install g++ libsqlite3-dev
````

Then compile the loader:

```bash
g++ -std=c++17 -O2 \
    utils/main.cpp utils/LiDARLoader.cpp \
    -lsqlite3 -o utils/lidar_loader
```

---

## ▶️ Running the Tool

You must provide:

1. A path to a single `.las` file **or** a directory containing multiple `.las` files
2. A path to the output `.db` file

```bash
./utils/lidar_loader <input.las | input_directory/> <output_database.db>
```

### ✅ Example

```bash
./utils/lidar_loader data/las/MDRS/ data/sqlite/MDRS.db
```

All point data will be loaded into the `RawPoints` table inside `MDRS.db`.

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

## 🗃️ Table Schema: `RawPoints`

| Column           | Type    | Description                           |
| ---------------- | ------- | ------------------------------------- |
| `id`             | INTEGER | Auto-increment primary key            |
| `Easting`        | REAL    | Scaled X position                     |
| `Northing`       | REAL    | Scaled Y position                     |
| `Altitude`       | REAL    | Scaled Z position                     |
| `Zone`           | TEXT    | UTM zone (e.g. `12N`)                 |
| `Classification` | TEXT    | Point class (e.g. Ground, Water, etc) |

An index on `(Easting, Northing)` is automatically created.

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