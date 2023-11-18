import os
import numpy as np
import cv2
import argparse


def convert_npz_to_png(_npz_dir, _output_dir=None, _inplace=False):
    if not _inplace and _output_dir is not None:
        # Ensure the output directory exists if it's specified
        os.makedirs(_output_dir, exist_ok=True)

    # List all NPZ files in the directory
    npz_files = [f for f in os.listdir(_npz_dir) if f.endswith(".npz")]

    for npz_file in npz_files:
        # Load the NPZ file
        data = np.load(os.path.join(_npz_dir, npz_file))

        # Assuming your NPZ file contains an array named 'data'
        image_array = data["data"]

        if _inplace:
            # Save the image as a PNG file with the same name, overwriting the NPZ file
            png_filename = f"{os.path.splitext(npz_file)[0]}.png"
            png_path = os.path.join(_npz_dir, png_filename)
        else:
            # Save the image as a PNG file in the output directory
            png_filename = f"{os.path.splitext(npz_file)[0]}.png"
            png_path = os.path.join(_output_dir, png_filename)

        cv2.imwrite(png_path, image_array)

        print(f"Converted {npz_file} to {png_filename}")

        if _inplace:
            # Remove the original NPZ file if the inplace option is enabled
            os.remove(os.path.join(_npz_dir, npz_file))
            print(f"Removed original NPZ file: {npz_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPZ files to PNG images.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the directory containing NPZ files.",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to the output directory for PNG files."
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Replace NPZ files with PNG files in the root directory.",
    )
    args = parser.parse_args()

    npz_dir = args.input
    output_dir = args.output
    inplace = args.inplace

    convert_npz_to_png(npz_dir, output_dir, inplace)
