/******************************************************************************
 * @brief Declaration of the LiDARLoader class for parsing and inserting LAS 1.4 data.
 *
 * This header defines the interface for the LiDARLoader class, which provides
 * functionality for:
 *  - Reading minimal header and metadata from LAS 1.4 files (Format 6)
 *  - Extracting UTM zone and hemisphere from the first Variable Length Record (VLR)
 *  - Scaling raw point data (X, Y, Z) into real-world coordinates
 *  - Inserting parsed LiDAR points into a SQLite database in a normalized schema
 *
 * The class is intended for offline preprocessing of LiDAR datasets before use
 * in autonomous navigation and perception systems. It assumes a fixed LAS 1.4
 * format and a database schema containing a `RawPoints` table.
 *
 * Example usage:
 * @code
 *   LiDARLoader loader;
 *   loader.LoadAndExportData("path/to/file.las", "path/to/database.db");
 * @endcode
 *
 * @file LiDARLoader.h
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 *
 * @copyright Copyright Mars Rover Design Team 2025 - All Rights Reserved
 ******************************************************************************/

#ifndef LIDARLOADER_H
#define LIDARLOADER_H

/// \cond
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <regex>
#include <sqlite3.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

/// \endcond

/******************************************************************************
 * @brief Class for loading and processing LAS 1.4 LiDAR files into SQLite.
 *
 * The LiDARLoader class reads LAS 1.4 binary files, extracts and scales point
 * data, and inserts the results into a structured SQLite database for use in
 * autonomous systems. It supports:
 *  - Minimal LAS header parsing
 *  - UTM zone extraction via VLR text
 *  - Efficient record collection and batched SQLite insertion
 *
 * This class is intended to be used offline as a preprocessing utility.
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
class LiDARLoader
{
    public:
        ////////////////////////////////////
        // Structures for LAS 1.4
        ////////////////////////////////////

        /******************************************************************************
         * @brief Structure representing a single parsed LiDAR point.
         *
         * Contains easting, northing, altitude, UTM zone, and classification label.
         * Used as the internal representation for inserting into the SQLite database.
         *
         * @author Eli Byrd (edbgkk@mst.edu)
         * @date 2025-05-20
         ******************************************************************************/
        struct PointRow
        {
                double dEasting;
                double dNorthing;
                double dAltitude;
                std::pair<int, char> stdZone;
                std::string szClassification;
        };

        ////////////////////////////////////
        // Predefined Enumerations
        ////////////////////////////////////
        enum class PointClassification : uint8_t;
        enum class LASHeaderOffset : size_t;

        ////////////////////////////////////
        // Constructors and Destructors
        ////////////////////////////////////

        /******************************************************************************
         * @brief Construct a new LiDAR Loader object.
         *
         * @author Eli Byrd (edbgkk@mst.edu)
         * @date 2025-05-20
         ******************************************************************************/
        LiDARLoader() = default;

        /******************************************************************************
         * @brief Destroy the LiDAR Loader object.
         *
         * @author Eli Byrd (edbgkk@mst.edu)
         * @date 2025-05-20
         ******************************************************************************/
        ~LiDARLoader() = default;

        ////////////////////////////////////
        // Public Methods
        ////////////////////////////////////
        unsigned long long LoadAndExportData(const std::string& szFilename, const std::string& szDBPath);
        void ComputeGridRoughness(const std::string& dbPath, double gridSize, double globalRadius);

    private:
        ////////////////////////////////////
        // Predefined Structures for LAS 1.4
        ////////////////////////////////////
        struct MinimalLASHeader;
        struct VLRHeader;

        ////////////////////////////////////
        // Private Methods
        ////////////////////////////////////
        MinimalLASHeader ReadMinimalHeader(std::ifstream& fInput);
        std::pair<int, char> ExtractUTMZoneFromVLR1(std::ifstream& fInput, const MinimalLASHeader& stHeader);
        std::vector<PointRow> CollectPointRecords(std::ifstream& fInput, const MinimalLASHeader& stHeader, std::pair<int, char> stdUTMZone);
        bool CreateDB(const std::string& szDBPath);
        int PopulateSQL(const std::vector<PointRow>& vPointRows, const std::string& szDBPath);
        std::vector<std::array<double, 3>> GetNearbyPoints(int nCx,
                                                           int nCy,
                                                           const std::map<std::pair<int, int>, std::vector<std::array<double, 3>>>& mGridBuckets,
                                                           double dGridSize,
                                                           double dRadius);
        double ComputePlaneFitRMSError(const std::vector<std::array<double, 3>>& vPoints);
};

#endif
