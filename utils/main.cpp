/******************************************************************************
 * @brief Entry point for the LiDAR preprocessing tool.
 *
 * This CLI tool accepts a LAS file or directory and a destination .db file.
 * It parses LAS 1.4 point data and loads the results into the given SQLite DB.
 *
 * @note Timing statistics are printed per file and for the full operation.
 *
 * @file main.cpp
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 *
 * @copyright Copyright Mars Rover Design Team 2025 - All Rights Reserved
 ******************************************************************************/

/// \cond
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

/// \endcond

#include "LiDARLoader.h"

/******************************************************************************
 * @brief Main entry point for loading LAS files into SQLite.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument vector.
 * @return int - Exit code (0 on success, 1 on error).
 *
 * @author Eli Byrd (edbgkk@mst.edu)
 * @date 2025-05-20
 ******************************************************************************/
int main(int argc, char** argv)
{
    // 1) Validate input
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input.las | input_dir/> <output.db>\n";
        return 1;
    }

    // 2) Parse command-line arguments
    const std::filesystem::path fsInputPath(argv[1]);
    const std::string szDBPath(argv[2]);
    if (!std::filesystem::exists(fsInputPath))
    {
        std::cerr << "Path does not exist: " << fsInputPath << "\n";
        return 1;
    }

    // 3) Gather all .las files
    std::vector<std::filesystem::path> vLASFiles;
    if (std::filesystem::is_directory(fsInputPath))
    {
        // 3a) Iterate through the directory and collect .las files
        for (const std::filesystem::directory_entry& fsEntry : std::filesystem::directory_iterator(fsInputPath))
        {
            if (fsEntry.is_regular_file() && fsEntry.path().extension() == ".las")
            {
                vLASFiles.push_back(fsEntry.path());
            }
        }

        // 3b) Check if no .las files were found and exit if so
        if (vLASFiles.empty())
        {
            std::cerr << "No .las files found in directory: " << fsInputPath << "\n";
            return 1;
        }
    }
    else
    {
        // 3c) Check if the input path is a single .las file
        if (fsInputPath.extension() != ".las")
        {
            std::cerr << "Not a .las file: " << fsInputPath << "\n";
            return 1;
        }

        // 3d) Add the .las file to the vector
        vLASFiles.push_back(fsInputPath);
    }

    // 4) Create a LiDARLoader instance
    LiDARLoader pLoader;

    // 5) Start overall timer
    std::chrono::time_point<std::chrono::steady_clock> tOverallStart = std::chrono::steady_clock::now();

    // 6) Iterate through each .las file and process it
    for (const std::filesystem::path& fsLASPath : vLASFiles)
    {
        // 6a) Start per-file timer
        std::cout << "Processing: " << fsLASPath << " ...\n";
        std::chrono::time_point<std::chrono::steady_clock> tFileStart = std::chrono::steady_clock::now();

        // 6b) Load and export data from the LAS file
        try
        {
            pLoader.LoadAndExportData(fsLASPath.string(), szDBPath);
        }
        catch (const std::exception& stdException)
        {
            std::cerr << "Error processing " << fsLASPath << ": " << stdException.what() << "\n";
            continue;
        }

        // 6c) Stop per-file timer and print duration
        std::chrono::time_point<std::chrono::steady_clock> tFileEnd = std::chrono::steady_clock::now();
        std::chrono::duration<double> tFileDuration                 = tFileEnd - tFileStart;
        std::cout << "  Done in " << tFileDuration.count() << " seconds\n\n";
    }

    // 7) Stop overall timer and print total duration
    std::chrono::time_point<std::chrono::steady_clock> tOverallEnd = std::chrono::steady_clock::now();
    std::chrono::duration<double> tOverallDuration                 = tOverallEnd - tOverallStart;
    std::cout << "Total processing time: " << tOverallDuration.count() << " seconds for " << vLASFiles.size() << " file(s)\n";

    // 8) Return success
    return 0;
}
