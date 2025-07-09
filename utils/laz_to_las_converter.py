#!/usr/bin/env python3

# ******************************************************************************
#  @file      laz_to_las_converter.py
#  @brief     Converts LAZ files to LAS format.
#
#  This script provides a utility for converting LAZ (compressed LiDAR) files
#  to LAS (uncompressed LiDAR) format, which can be more easily processed by
#  various tools and libraries.
#
#  Example usage:
#      $ python laz_to_las_converter.py /path/to/file.laz
#
#  @author     ClayJay3 (claytonraycowen@gmail.com)
#  @date       2025-05-31
#  @copyright  Copyright Mars Rover Design Team 2025 – All Rights Reserved
# ******************************************************************************


import os
import laspy
from pathlib import Path
import glob
import argparse

def convert_laz_to_las(laz_file_path, output_las_path=None):
    """
    Convert a .laz file to a .las file using laspy.
    
    Parameters:
    -----------
    laz_file_path (str): Path to the .laz file
    output_las_path (str, optional): Path to save the output .las file. If None, will use the same name
        with .las extension in the same directory.
        
    Returns:
    --------
    str
        Path to the created .las file
    """
    # Validate input path.
    if not os.path.exists(laz_file_path):
        raise FileNotFoundError(f"Input file not found: {laz_file_path}")
    
    # Generate output path if not provided.
    if output_las_path is None:
        output_las_path = str(Path(laz_file_path).with_suffix('.las'))
    
    try:
        # Try to use laspy's automatic backend detection first.
        try:
            print(f"Attempting to read {laz_file_path} with automatic backend detection")
            with laspy.open(laz_file_path, mode="r") as laz_file:
                las_data = laz_file.read()
                print("Successfully read LAZ file with automatic backend")
        except Exception as e:
            print(f"Automatic backend detection failed: {str(e)}")
            
            # Try to determine which backend is available.
            print("Checking available LAZ backends...")
            
            # Check if we're using laspy v2.0+.
            try:
                from laspy.compression import LazBackend
                
                # Try with laszip backend.
                try:
                    import laszip
                    print("Laszip module is available")
                    
                    try:
                        with laspy.open(laz_file_path, mode="r", laz_backend=LazBackend.LasZip) as laz_file:
                            las_data = laz_file.read()
                        print("Successfully read with LasZip backend")
                    except Exception as laszip_e:
                        print(f"Failed to use LasZip backend: {str(laszip_e)}")
                        raise
                        
                except ImportError:
                    print("Laszip module not available")
                    
                    # Try with lazrs backend.
                    try:
                        import lazrs
                        print("Lazrs module is available")
                        
                        try:
                            with laspy.open(laz_file_path, mode="r", laz_backend=LazBackend.Lazrs) as laz_file:
                                las_data = laz_file.read()
                            print("Successfully read with Lazrs backend")
                        except Exception as lazrs_e:
                            print(f"Failed to use Lazrs backend: {str(lazrs_e)}")
                            raise
                            
                    except ImportError:
                        print("Lazrs module not available")
                        raise RuntimeError("Neither laszip nor lazrs backends are available")
                
            except ImportError:
                # For older laspy versions, try a simpler approach.
                print("Using older laspy API for backends")
                with laspy.file.File(laz_file_path, mode="r") as laz_file:
                    las_data = laz_file
                
        # Write the LAS file.
        print(f"Writing LAS data to {output_las_path}")
        try:
            # Method 1: Direct writing. (newer laspy versions)
            las_data.write(output_las_path)
            print("Used direct write method")
        except (AttributeError, TypeError) as e:
            print(f"Direct write failed: {str(e)}, trying alternative methods...")
            try:
                # Method 2: Using write_points.
                with laspy.open(output_las_path, mode="w", header=las_data.header) as las_file:
                    las_file.write_points(las_data.points)
                print("Used write_points method")
            except (AttributeError, TypeError) as e:
                print(f"write_points failed: {str(e)}, trying another method...")
                try:
                    # Method 3: Using write_chunk.
                    with laspy.open(output_las_path, mode="w", header=las_data.header) as las_file:
                        las_file.write_chunk(las_data.points)
                    print("Used write_chunk method")
                except Exception as e:
                    print(f"All writing methods failed: {str(e)}")
                    raise
            
        return output_las_path
    
    except Exception as e:
        raise RuntimeError(f"Error converting {laz_file_path} to LAS: {str(e)}")


def batch_convert_laz_to_las(input_dir, output_dir=None, pattern="*.laz"):
    """
    Batch convert .laz files to .las files.
    
    Parameters:
    -----------
    input_dir (str): Directory containing .laz files
    output_dir (str, optional): Directory to save .las files. If None, will use the same directory.
    pattern (str, optional): Glob pattern to match .laz files. Default is "*.laz"
        
    Returns:
    --------
    list
        List of paths to the created .las files
    """
    # Validate input directory.
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    
    # If output directory not specified, use input directory.
    if output_dir is None:
        output_dir = input_dir
    else:
        # Create output directory if it doesn't exist.
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all .laz files.
    laz_files = glob.glob(os.path.join(input_dir, pattern))
    
    converted_files = []
    for laz_file in laz_files:
        file_name = os.path.basename(laz_file)
        las_file_name = os.path.splitext(file_name)[0] + '.las'
        output_path = os.path.join(output_dir, las_file_name)
        
        try:
            converted_path = convert_laz_to_las(laz_file, output_path)
            converted_files.append(converted_path)
            print(f"Converted: {laz_file} -> {converted_path}")
        except Exception as e:
            print(f"Failed to convert {laz_file}: {str(e)}")
    
    return converted_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LAZ files to LAS format")
    parser.add_argument("input", help="Input LAZ file or directory")
    parser.add_argument("-o", "--output", help="Output LAS file or directory (optional)")
    parser.add_argument("-p", "--pattern", default="*.laz", help="File pattern if input is a directory (default: *.laz)")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        # Batch conversion.
        batch_convert_laz_to_las(args.input, args.output, args.pattern)
    else:
        # Single file conversion.
        try:
            output_path = convert_laz_to_las(args.input, args.output)
            print(f"Converted: {args.input} -> {output_path}")
        except Exception as e:
            print(f"Error: {str(e)}")