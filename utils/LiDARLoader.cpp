/******************************************************************************
 * @brief Implementation file for LAS 1.4 LiDAR data loading and database insertion.
 *
 * This module defines the LiDARLoader class, which handles reading LAS 1.4 files,
 * parsing and scaling point data, and storing the results in a SQLite database.
 * It supports:
 *  - Minimal LAS header parsing
 *  - UTM zone extraction from the first VLR
 *  - Scaled point record extraction (Format 6 only)
 *  - Efficient batched insertion into a normalized RawPoints table
 *
 * This file is intended to be used in the preprocessing stage of autonomous
 * navigation, populating a spatial database for later query by runtime systems.
 *
 * @file LiDARLoader.cpp
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 *
 * @copyright Copyright Mars Rover Design Team 2025 - All Rights Reserved
 ******************************************************************************/

/// \cond
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sqlite3.h>
#include <sstream>
#include <stdexcept>

/// \endcond

#include "LiDARLoader.h"

/******************************************************************************
 * @brief Minimal LAS header structure containing only required fields.
 *
 * This struct holds a simplified subset of the LAS 1.4 file header, capturing
 * only the fields necessary for reading and interpreting point data for this
 * application.
 *
 * Fields:
 * - unPointDataFormat: Format ID for point record structure (e.g., 6 for LAS 1.4)
 * - dXScale, dYScale, dZScale: Scale factors for converting raw X/Y/Z integers
 * - dXOffset, dYOffset, dZOffset: Offsets added to scaled X/Y/Z values
 * - dMaxX, dMaxY, dMaxZ: Bounding box maximum coordinates
 * - dMinX, dMinY, dMinZ: Bounding box minimum coordinates
 * - unHeaderSize: Total size of the LAS file header (typically 375 bytes for LAS 1.4)
 * - unNumberOfPointRecords: Total number of point records in the file
 * - unNumVLRs: Number of Variable Length Records (VLRs) following the header
 * - unOffsetToPointData: Byte offset in the file where point data begins
 * - unPointRecordLength: Size in bytes of a single point record
 *
 * @note This structure is read directly from the LAS header using `memcpy`.
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
struct LiDARLoader::MinimalLASHeader
{
        uint8_t unPointDataFormat;
        double dXScale, dYScale, dZScale;
        double dXOffset, dYOffset, dZOffset;
        double dMaxX, dMaxY, dMaxZ;
        double dMinX, dMinY, dMinZ;
        uint16_t unHeaderSize;
        uint64_t unNumberOfPointRecords;
        uint32_t unNumVLRs;
        uint32_t unOffsetToPointData;
        uint16_t unPointRecordLength;
};

#pragma pack(push, 1)

/******************************************************************************
 * @brief Represents the header of a Variable Length Record (VLR) in a LAS file.
 *
 * A VLR contains metadata such as projection information, coordinate system
 * details, or user-defined data. This structure matches the fixed 54-byte layout
 * defined in the ASPRS LAS 1.4 specification.
 *
 * @note This structure is packed to avoid padding between fields.
 *
 * Fields:
 * - reserved: Always 0
 * - userID: Identifier of the VLR creator (e.g., "LASF_Projection")
 * - recordID: VLR record type code
 * - recordLengthAfterHeader: Length of the VLR payload that follows
 * - description: Human-readable description of the VLR
 *
 * @see https://www.asprs.org/wp-content/uploads/2019/03/LAS_1_4_r14.pdf
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
struct LiDARLoader::VLRHeader
{
        uint16_t unReserved;
        char szUserID[16];
        uint16_t unRecordID;
        uint16_t unRecordLengthAfterHeader;
        char szDescription[32];
};

#pragma pack(pop)

/******************************************************************************
 * @brief Enumeration for ASPRS LAS 1.4 point classifications.
 *
 * Represents the classification of each LiDAR point according to the LAS 1.4
 * specification. These values are used to describe the type of object the point
 * represents (e.g., ground, vegetation, building).
 *
 * @note Values:
 * - 0–22: Standard ASPRS classifications
 * - 23–63: Reserved
 * - 64–255: User-definable by the data producer
 *
 * Used in parsing and categorizing points during LAS file loading.
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
enum class LiDARLoader::PointClassification : uint8_t
{
    unCreatedNeverClassified = 0,     // Created, never classified
    unUnclassified           = 1,     // Unclassified
    unGround                 = 2,     // Ground
    unLowVegetation          = 3,     // Low Vegetation
    unMediumVegetation       = 4,     // Medium Vegetation
    unHighVegetation         = 5,     // High Vegetation
    unBuilding               = 6,     // Building
    unLowPointNoise          = 7,     // Low Point (Noise)
    unReserved8              = 8,     // Reserved
    unWater                  = 9,     // Water
    unRail                   = 10,    // Rail
    unRoadSurface            = 11,    // Road Surface
    unReserved12             = 12,    // Reserved
    unWireGuardShield        = 13,    // Wire – Guard (Shield)
    unWireConductorPhase     = 14,    // Wire – Conductor (Phase)
    unTransmissionTower      = 15,    // Transmission Tower
    unWireStructureConnector = 16,    // Wire-Structure Connector
    unBridgeDeck             = 17,    // Bridge Deck
    unHighNoise              = 18,    // High Noise
    unOverheadStructure      = 19,    // Overhead Structure
    unIgnoredGround          = 20,    // Ignored Ground
    unSnow                   = 21,    // Snow
    unTemporalExclusion      = 22,    // Temporal Exclusion
    unReserved               = 23,    // 23–63
    unUserDefinable          = 64     // 64–255
};

/******************************************************************************
 * @brief Enumerated Byte Offsets for relevant fields in the LAS 1.4 header.
 *
 * These constants define the byte positions of specific fields within the
 * LAS file header according to the ASPRS LAS 1.4 specification (Table 3).
 * They are used to extract only the necessary fields via direct `memcpy`
 * or `seekg` operations, avoiding full parsing of unused fields.
 *
 * @note All offsets are relative to the beginning of the file.
 *
 * @note Only the fields required by the loader are included.
 *       Values are hard-coded for performance and simplicity.
 *
 * @see https://www.asprs.org/wp-content/uploads/2019/03/LAS_1_4_r14.pdf
 * @date 2025-05-20
 * @author Eli Byrd (edbgkk@mst.edu)
 ******************************************************************************/
enum class LiDARLoader::LASHeaderOffset : size_t
{
    siFormat                = 104,    // Point Data Record Format          | unsigned char      | 1 byte
    siXScaleFactor          = 131,    // X Scale Factor                    | double             | 8 bytes
    siYScaleFactor          = 139,    // Y Scale Factor                    | double             | 8 bytes
    siZScaleFactor          = 147,    // Z Scale Factor                    | double             | 8 bytes
    siXOffset               = 155,    // X Offset                          | double             | 8 bytes
    siYOffset               = 163,    // Y Offset                          | double             | 8 bytes
    siZOffset               = 171,    // Z Offset                          | double             | 8 bytes
    siXMax                  = 179,    // X Max                             | double             | 8 bytes
    siYMax                  = 195,    // Y Max                             | double             | 8 bytes
    siZMax                  = 211,    // Z Max                             | double             | 8 bytes
    siXMin                  = 187,    // X Min                             | double             | 8 bytes
    siYMin                  = 203,    // Y Min                             | double             | 8 bytes
    siZMin                  = 219,    // Z Min                             | double             | 8 bytes
    siNumberOfPointRecords  = 247,    // Number of Point Records           | unsigned long long | 8 bytes
    siHeaderSize            = 94,     // Header Size                       | unsigned short     | 2 bytes
    siNumberOfVLRs          = 100,    // Number of Variable Length Records | unsigned long      | 4 bytes
    siOffsetToPointData     = 96,     // Offset to Point Data              | unsigned long      | 4 bytes
    siPointDataRecordLength = 105     // Point Data Record Length          | unsigned short     | 2 bytes
};

/******************************************************************************
 * @brief Converts a raw LAS classification byte into a descriptive string.
 *
 * This function interprets a uint8_t classification value according to the
 * ASPRS LAS 1.4 specification and maps it to a human-readable string label.
 *
 * The input is first converted into a PointClassification enum using the
 * following logic:
 * - Values 0–22 map directly to standard classifications
 * - Values 23–63 are treated as "Reserved"
 * - Values 64–255 are treated as "User Definable"
 *
 * Then, the enum is mapped to a string such as "Ground", "Building", etc.
 *
 * @param unClassification The raw classification value from a point record.
 * @return std::string - The corresponding classification name.
 *         Returns "Unknown" for any unrecognized or unmapped values.
 *
 * @note This is typically used when exporting or displaying classification types
 *       from LAS point records in a readable form.
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
static std::string MakeClassification(uint8_t unClassification)
{
    // 1) Convert the classification value to the enum type
    LiDARLoader::PointClassification eClassification;
    if (unClassification <= 22)
    {
        eClassification = static_cast<LiDARLoader::PointClassification>(unClassification);
    }
    else if (unClassification >= 23 && unClassification <= 63)
    {
        eClassification = LiDARLoader::PointClassification::unHighNoise;
    }
    else
    {
        eClassification = LiDARLoader::PointClassification::unUserDefinable;
    }

    // 2) Map and return enum value to a string
    switch (eClassification)
    {
        case LiDARLoader::PointClassification::unCreatedNeverClassified: return "Created (Never Classified)";
        case LiDARLoader::PointClassification::unUnclassified: return "Unclassified";
        case LiDARLoader::PointClassification::unGround: return "Ground";
        case LiDARLoader::PointClassification::unLowVegetation: return "Low Vegetation";
        case LiDARLoader::PointClassification::unMediumVegetation: return "Medium Vegetation";
        case LiDARLoader::PointClassification::unHighVegetation: return "High Vegetation";
        case LiDARLoader::PointClassification::unBuilding: return "Building";
        case LiDARLoader::PointClassification::unLowPointNoise: return "Low Point (Noise)";
        case LiDARLoader::PointClassification::unReserved8: return "Reserved";
        case LiDARLoader::PointClassification::unWater: return "Water";
        case LiDARLoader::PointClassification::unRail: return "Rail";
        case LiDARLoader::PointClassification::unRoadSurface: return "Road Surface";
        case LiDARLoader::PointClassification::unReserved12: return "Reserved";
        case LiDARLoader::PointClassification::unWireGuardShield: return "Wire – Guard (Shield)";
        case LiDARLoader::PointClassification::unWireConductorPhase: return "Wire – Conductor (Phase)";
        case LiDARLoader::PointClassification::unTransmissionTower: return "Transmission Tower";
        case LiDARLoader::PointClassification::unWireStructureConnector: return "Wire-Structure Connector";
        case LiDARLoader::PointClassification::unBridgeDeck: return "Bridge Deck";
        case LiDARLoader::PointClassification::unHighNoise: return "High Noise";
        case LiDARLoader::PointClassification::unOverheadStructure: return "Overhead Structure";
        case LiDARLoader::PointClassification::unIgnoredGround: return "Ignored Ground";
        case LiDARLoader::PointClassification::unSnow: return "Snow";
        case LiDARLoader::PointClassification::unTemporalExclusion: return "Temporal Exclusion";
        case LiDARLoader::PointClassification::unReserved: return "Reserved";
        case LiDARLoader::PointClassification::unUserDefinable: return "User Definable";
    }
    return "Unknown";
}

/******************************************************************************
 * @brief Reads a 32-bit little-endian integer from a byte buffer.
 *
 * This function interprets 4 consecutive bytes starting at the specified offset
 * in the buffer as a little-endian encoded 32-bit signed integer.
 *
 * @param szBuffer Pointer to the byte buffer.
 * @param siOffset Offset from the start of the buffer where the 4-byte integer begins.
 * @return int32_t - The decoded 32-bit signed integer.
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
static int32_t ReadLE32(const char* szBuffer, size_t siOffset)
{
    // Read the first 4 bytes as unsigned 8-bit integers
    uint8_t unByte0 = static_cast<uint8_t>(szBuffer[siOffset + 0]);
    uint8_t unByte1 = static_cast<uint8_t>(szBuffer[siOffset + 1]);
    uint8_t unByte2 = static_cast<uint8_t>(szBuffer[siOffset + 2]);
    uint8_t unByte3 = static_cast<uint8_t>(szBuffer[siOffset + 3]);

    // Combine the bytes into a single 32-bit signed integer
    return int32_t(unByte0 | (unByte1 << 8) | (unByte2 << 16) | (unByte3 << 24));
}

/******************************************************************************
 * @brief Trims trailing null characters and spaces from a character buffer.
 *
 * This function returns a string that excludes any trailing `\\0` or space (' ')
 * characters from the end of the input buffer, up to the specified length.
 * Commonly used to clean fixed-length LAS fields such as user IDs and descriptions.
 *
 * @param szBuffer Pointer to the input character buffer.
 * @param siLength Length of the buffer to consider for trimming.
 * @return std::string - A trimmed string with trailing nulls and spaces removed.
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
static std::string Trim(const char* szBuffer, size_t siLength)
{
    // Start with the full length of the buffer
    size_t siEnd = siLength;

    // Decrement the length until we find a non-null and non-space character
    while (siEnd > 0 && (szBuffer[siEnd - 1] == '\0' || szBuffer[siEnd - 1] == ' '))
    {
        --siEnd;
    }

    // Return a string constructed from the trimmed buffer
    return std::string(szBuffer, szBuffer + siEnd);
}

/******************************************************************************
 * @brief Reads the minimal LAS 1.4 header from the input stream.
 *
 * This function seeks to specific byte offsets in the file (as defined in the
 * LAS 1.4 specification) and extracts only the necessary fields for parsing and
 * scaling point data. It uses `LASHeaderOffset` to identify key locations in the
 * header, and avoids reading unnecessary metadata.
 *
 * @param fInput Reference to the opened binary input stream for the LAS file.
 * @return LiDARLoader::MinimalLASHeader - A structure containing only the essential
 *         LAS header fields required for downstream processing.
 *
 * @throws std::runtime_error - If any required field cannot be read.
 *
 * @see LASHeaderOffset
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
LiDARLoader::MinimalLASHeader LiDARLoader::ReadMinimalHeader(std::ifstream& fInput)
{
    // Create a MinimalLASHeader object
    MinimalLASHeader stHeader{};

    // Read in the size of the header
    fInput.seekg(static_cast<size_t>(LASHeaderOffset::siHeaderSize), std::ios::beg);
    fInput.read(reinterpret_cast<char*>(&stHeader.unHeaderSize), sizeof(stHeader.unHeaderSize));

    // Read in the number of variable length records
    fInput.seekg(static_cast<size_t>(LASHeaderOffset::siNumberOfVLRs), std::ios::beg);
    fInput.read(reinterpret_cast<char*>(&stHeader.unNumVLRs), sizeof(stHeader.unNumVLRs));

    // Read the entire header into a buffer
    std::vector<char> vBuffer(stHeader.unHeaderSize);
    fInput.seekg(0, std::ios::beg);
    fInput.read(vBuffer.data(), vBuffer.size());

    // Read the relevant fields from the buffer
    std::memcpy(&stHeader.unPointDataFormat, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siFormat), sizeof(stHeader.unPointDataFormat));
    std::memcpy(&stHeader.dXScale, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siXScaleFactor), sizeof(stHeader.dXScale));
    std::memcpy(&stHeader.dYScale, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siYScaleFactor), sizeof(stHeader.dYScale));
    std::memcpy(&stHeader.dZScale, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siZScaleFactor), sizeof(stHeader.dZScale));
    std::memcpy(&stHeader.dXOffset, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siXOffset), sizeof(stHeader.dXOffset));
    std::memcpy(&stHeader.dYOffset, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siYOffset), sizeof(stHeader.dYOffset));
    std::memcpy(&stHeader.dZOffset, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siZOffset), sizeof(stHeader.dZOffset));
    std::memcpy(&stHeader.dMaxX, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siXMax), sizeof(stHeader.dMaxX));
    std::memcpy(&stHeader.dMaxY, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siYMax), sizeof(stHeader.dMaxY));
    std::memcpy(&stHeader.dMaxZ, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siZMax), sizeof(stHeader.dMaxZ));
    std::memcpy(&stHeader.dMinX, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siXMin), sizeof(stHeader.dMinX));
    std::memcpy(&stHeader.dMinY, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siYMin), sizeof(stHeader.dMinY));
    std::memcpy(&stHeader.dMinZ, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siZMin), sizeof(stHeader.dMinZ));
    std::memcpy(&stHeader.unNumberOfPointRecords, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siNumberOfPointRecords), sizeof(stHeader.unNumberOfPointRecords));
    std::memcpy(&stHeader.unOffsetToPointData, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siOffsetToPointData), sizeof(stHeader.unOffsetToPointData));
    std::memcpy(&stHeader.unPointRecordLength, vBuffer.data() + static_cast<size_t>(LASHeaderOffset::siPointDataRecordLength), sizeof(stHeader.unPointRecordLength));

    // Return the simplified header
    return stHeader;
}

/******************************************************************************
 * @brief Extracts the UTM zone and hemisphere from the first VLR in a LAS file.
 *
 * This function reads the first Variable Length Record (VLR) from the LAS file,
 * typically used for projection metadata, and searches for a UTM zone declaration
 * using a regular expression (e.g., "UTM zone 12N").
 *
 * @param fInput Reference to an open binary input stream for the LAS file.
 * @param stHeader  The parsed minimal LAS header providing offset and VLR count.
 * @return std::pair<int, char> - A pair containing:
 *         - The UTM zone number (1–60) or -1 if not found.
 *         - The hemisphere character ('N', 'S', or '?' if unknown).
 *
 * @note This assumes the UTM zone is encoded as plain text in the VLR payload.
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
std::pair<int, char> LiDARLoader::ExtractUTMZoneFromVLR1(std::ifstream& fInput, const MinimalLASHeader& stHeader)
{
    // 1) Seek to the first VLR, which lives at byte offset = headerSize
    fInput.seekg(stHeader.unHeaderSize, std::ios::beg);

    // 2) Read the VLR header
    VLRHeader stVLRHeader;
    fInput.read(reinterpret_cast<char*>(&stVLRHeader), sizeof(stVLRHeader));
    std::vector<char> payload(stVLRHeader.unRecordLengthAfterHeader);
    fInput.read(payload.data(), payload.size());

    // 3) Convert payload to string and search for UTM zone and hemisphere
    std::string szPayload(payload.begin(), payload.end());
    static const std::regex stdRegex(R"(UTM\s+zone\s*([0-9]+)([NS])?)", std::regex::icase);
    std::smatch stdMatch;

    // 3a) If regex matches, extract the zone and hemisphere
    if (std::regex_search(szPayload, stdMatch, stdRegex))
    {
        int nZone  = std::stoi(stdMatch[1].str());
        char cHemi = (stdMatch.size() >= 3 && !stdMatch[2].str().empty()) ? stdMatch[2].str()[0] : '?';

        // 3b) Return the UTM zone and hemisphere
        return {nZone, cHemi};
    }

    // 4) If no match, return -1 for zone and '?' for hemisphere
    return {-1, '?'};
}

/******************************************************************************
 * @brief Reads and converts all point records from the LAS file into usable coordinates.
 *
 * This function iterates over each point in the LAS file using the format specified
 * in the provided LAS header, extracts raw X/Y/Z integer values, applies scaling and
 * offsets, and constructs a list of usable point data entries (PointRow).
 *
 * Each point's classification byte is read and converted into its numeric string
 * representation (as defined by the LAS classification standard).
 *
 * @param fInput         Reference to the binary input stream for the LAS file.
 * @param stHeader          The minimal LAS header containing scale factors, offsets, and counts.
 * @param stdUTMZone    The UTM zone (number and hemisphere) to tag each point with.
 *
 * @return std::vector<PointRow> - A vector containing all scaled point data ready for insertion into the database.
 *
 * @note This function assumes Point Data Format 6 (LAS 1.4).
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
std::vector<LiDARLoader::PointRow> LiDARLoader::CollectPointRecords(std::ifstream& fInput, const MinimalLASHeader& stHeader, std::pair<int, char> stdUTMZone)
{
    // 1) Read the point data record length and reserve space for the rows
    fInput.seekg(stHeader.unOffsetToPointData, std::ios::beg);
    std::vector<char> vBuffer(stHeader.unPointRecordLength);
    std::vector<PointRow> vPointRows;
    vPointRows.reserve(stHeader.unNumberOfPointRecords);

    // 2) Read each point record
    for (uint64_t unIndex = 0; unIndex < stHeader.unNumberOfPointRecords; ++unIndex)
    {
        // 2a) Read the point data into the buffer
        fInput.read(vBuffer.data(), vBuffer.size());
        if (!fInput)
        {
            break;
        }

        // 2b) Extract raw X/Y/Z coordinates from the buffer
        int32_t nRawX = ReadLE32(vBuffer.data(), 0);
        int32_t nRawY = ReadLE32(vBuffer.data(), 4);
        int32_t nRawZ = ReadLE32(vBuffer.data(), 8);

        // 2c) Extract the classification byte
        std::string szCLS = MakeClassification(static_cast<uint8_t>(vBuffer[16]));

        // 2d) Scale and offset the coordinates
        double dEasting  = nRawX * stHeader.dXScale + stHeader.dXOffset;
        double dNorthing = nRawY * stHeader.dYScale + stHeader.dYOffset;
        double dAltitude = nRawZ * stHeader.dZScale + stHeader.dZOffset;

        // 2e) Create a PointRow object and add it to the vector
        vPointRows.push_back({dEasting, dNorthing, dAltitude, stdUTMZone, szCLS});
    }

    // 3) Return the vector of PointRow objects
    return vPointRows;
}

/******************************************************************************
 * @brief Creates the SQLite database and RawPoints table if they do not exist.
 *
 * This function opens (or creates) the SQLite database at the given path,
 * and ensures the required `RawPoints` table and index exist.
 *
 * @param dbPath Filesystem path to the SQLite database.
 * @return bool - true if the database was created or verified successfully, false on error.
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
bool LiDARLoader::CreateDB(const std::string& szDBPath)
{
    // Ensure directory exists
    std::filesystem::path fsPath(szDBPath);
    if (!fsPath.parent_path().empty())
        std::filesystem::create_directories(fsPath.parent_path());

    sqlite3* sqlDatabase = nullptr;
    if (sqlite3_open(szDBPath.c_str(), &sqlDatabase) != SQLITE_OK)
    {
        std::cerr << "Failed to create DB: " << sqlite3_errmsg(sqlDatabase) << "\n";
        return false;
    }

    const char* szCreateRawPointsTableSQL  = R"(
        CREATE TABLE IF NOT EXISTS RawPoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Easting REAL NOT NULL,
            Northing REAL NOT NULL,
            Altitude REAL NOT NULL,
            Zone TEXT NOT NULL,
            Classification TEXT NOT NULL
        );
    )";

    const char* szCreateRawPointsIndexsSQL = R"(
        CREATE INDEX IF NOT EXISTS idx_Easting_Northing ON RawPoints (Easting, Northing);
    )";

    char* szErrMsg                         = nullptr;
    if (sqlite3_exec(sqlDatabase, szCreateRawPointsTableSQL, nullptr, nullptr, &szErrMsg) != SQLITE_OK)
    {
        std::cerr << "Failed to create table: " << szErrMsg << "\n";
        sqlite3_free(szErrMsg);
        sqlite3_close(sqlDatabase);
        return false;
    }

    if (sqlite3_exec(sqlDatabase, szCreateRawPointsIndexsSQL, nullptr, nullptr, &szErrMsg) != SQLITE_OK)
    {
        std::cerr << "Failed to create index: " << szErrMsg << "\n";
        sqlite3_free(szErrMsg);
        sqlite3_close(sqlDatabase);
        return false;
    }

    sqlite3_close(sqlDatabase);
    return true;
}

/******************************************************************************
 * @brief Inserts LiDAR point data into a SQLite database using batched transactions.
 *
 * This function takes a list of pre-parsed and scaled LiDAR points (`PointRow`)
 * and inserts them into the `RawPoints` table within the specified SQLite database.
 * It uses batched transactions (every 10,000 rows) for better performance during bulk loads.
 *
 * Each point record includes:
 * - Easting, Northing, Altitude (in meters)
 * - Zone (e.g., "12N")
 * - Classification (as a string)
 *
 * @param rows          A vector of PointRow entries to insert into the database.
 * @param dbPath        Filesystem path to the SQLite database to be populated.
 *                      Will create parent directories if needed.
 *
 * @return int - Returns 0 on success, -1 on failure (e.g., DB open/prepare error).
 *
 * @note The function handles its own transaction logic and safely finalizes prepared statements.
 * @note Uses `SQLITE_TRANSIENT` to ensure strings are copied before the buffer is deallocated.
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
int LiDARLoader::PopulateSQL(const std::vector<PointRow>& vPointRows, const std::string& szDBPath)
{
    // 1) Check if the database exists and create it if not
    if (!CreateDB(szDBPath))
        return -1;

    // 2) Ensure directory exists
    std::filesystem::path fPath(szDBPath);
    if (!fPath.parent_path().empty())
    {
        std::filesystem::create_directories(fPath.parent_path());
    }

    // 3) Open DB
    sqlite3* sqlDatabase = nullptr;
    if (sqlite3_open(szDBPath.c_str(), &sqlDatabase) != SQLITE_OK)
    {
        std::cerr << "Cannot open DB: " << sqlite3_errmsg(sqlDatabase) << "\n";
        return -1;
    }

    // 4) Begin transaction for speed
    sqlite3_exec(sqlDatabase, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);

    // 5) Prepare INSERT statement
    const char* szCommand      = R"(
        INSERT INTO RawPoints
          (Easting, Northing, Altitude, Zone, Classification)
        VALUES (?, ?, ?, ?, ?);
    )";
    sqlite3_stmt* sqlStatement = nullptr;
    if (sqlite3_prepare_v2(sqlDatabase, szCommand, -1, &sqlStatement, nullptr) != SQLITE_OK)
    {
        std::cerr << "Prepare failed: " << sqlite3_errmsg(sqlDatabase) << "\n";
        sqlite3_close(sqlDatabase);
        return -1;
    }

    // 6) Loop rows
    for (size_t siIndex = 0; siIndex < vPointRows.size(); ++siIndex)
    {
        // 5a) Bind parameters
        const LiDARLoader::PointRow& stPointRow = vPointRows[siIndex];
        sqlite3_bind_double(sqlStatement, 1, stPointRow.dEasting);
        sqlite3_bind_double(sqlStatement, 2, stPointRow.dNorthing);
        sqlite3_bind_double(sqlStatement, 3, stPointRow.dAltitude);

        std::string szZoneStr = std::to_string(stPointRow.stdZone.first) + stPointRow.stdZone.second;
        sqlite3_bind_text(sqlStatement, 4, szZoneStr.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(sqlStatement, 5, stPointRow.szClassification.c_str(), -1, SQLITE_TRANSIENT);

        // 5b) Execute & reset
        sqlite3_step(sqlStatement);
        sqlite3_reset(sqlStatement);

        // 5c) Commit every 10,000 rows
        if ((siIndex + 1) % 10000 == 0)
        {
            sqlite3_exec(sqlDatabase, "END TRANSACTION;", nullptr, nullptr, nullptr);
            sqlite3_exec(sqlDatabase, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);
        }
    }

    // 7) Finalize & commit
    sqlite3_finalize(sqlStatement);
    sqlite3_exec(sqlDatabase, "END TRANSACTION;", nullptr, nullptr, nullptr);
    sqlite3_close(sqlDatabase);

    // 8) Output the number of rows inserted and return success
    std::cout << "Inserted " << vPointRows.size() << " rows into RawPoints\n";
    return 0;
}

/******************************************************************************
 * @brief Loads point data from a LAS file and inserts it into the SQLite database.
 *
 * This function performs the full data loading pipeline for a single LAS file:
 * - Opens the binary LAS file
 * - Reads the minimal LAS 1.4 header
 * - Extracts UTM zone information from the first Variable Length Record (VLR)
 * - Parses and scales all point data records
 * - Inserts the points into the `RawPoints` table using `PopulateSQL()`
 *
 * @param szFilename Path to the LAS file to load.
 * @param szDBPath   Path to the SQLite database where the data will be inserted.
 *
 * @return unsigned long long - Number of point records successfully loaded and inserted.
 *
 * @throws std::runtime_error If the file cannot be opened.
 * @note If UTM zone extraction fails, the zone will be stored as {-1, '?'}.
 * @note Outputs warnings if the number of parsed points is less than expected.
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
unsigned long long LiDARLoader::LoadAndExportData(const std::string& szFilename, const std::string& szDBPath)
{
    // 1) Open the file
    std::ifstream fInput(szFilename, std::ios::binary);
    if (!fInput)
    {
        throw std::runtime_error("Failed to open file: " + szFilename);
    }

    // 2) Read the minimal header
    MinimalLASHeader stHeader = ReadMinimalHeader(fInput);

    // 3) Read the VLRs amd extract UTM zone
    std::pair<int, char> stdUTM = ExtractUTMZoneFromVLR1(fInput, stHeader);
    if (stdUTM.first <= 0)
    {
        std::cerr << "Failed to find UTM zone in VLR #1\n";
    }

    // 4) Collect point records
    std::vector<LiDARLoader::PointRow> vPointRows = CollectPointRecords(fInput, stHeader, {stdUTM.first, stdUTM.second});
    if (vPointRows.size() != stHeader.unNumberOfPointRecords)
    {
        std::cerr << "Warning: Only " << vPointRows.size() << " points out of " << stHeader.unNumberOfPointRecords << "\n";
        return 0;
    }

    // 5) Populate the database
    PopulateSQL(vPointRows, szDBPath);

    // 6) Return the number of points loaded
    return vPointRows.size();
}
