import os
import numpy as np
import json
import cv2
import argparse
from tqdm import tqdm


def resize_and_save_images(input_dir, output_dir, target_size=(512, 512)):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all NPZ files in the input directory
    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]

    # Process each NPZ file and its corresponding JSON metadata
    for npz_file in tqdm(npz_files, desc="Processing NPZ files"):
        # Load the NPZ file
        data = np.load(os.path.join(input_dir, npz_file))
        pixel_array = data["data"]

        # Load the corresponding JSON metadata
        json_file = os.path.splitext(npz_file)[0] + ".json"
        json_path = os.path.join(input_dir, json_file)

        with open(json_path, "r") as json_file:
            metadata = json.load(json_file)

        # Extract the bounding box coordinates and confidence score
        coords = metadata["coords"]
        score = metadata["score"]

        # Resize the pixel array while maintaining aspect ratio
        resized_image = cv2.resize(
            pixel_array, target_size, interpolation=cv2.INTER_LINEAR
        )

        # Scale the coordinates to match the resized image
        scale_x = target_size[1] / pixel_array.shape[1]
        scale_y = target_size[0] / pixel_array.shape[0]
        new_coords = [
            int(coords[0] * scale_x),
            int(coords[1] * scale_y),
            int(coords[2] * scale_x),
            int(coords[3] * scale_y),
        ]

        # Save the resized image as an NPZ file with the same name
        output_npz_file = os.path.join(output_dir, npz_file)
        np.savez_compressed(output_npz_file, data=resized_image)

        # Update the JSON metadata with the scaled coordinates
        metadata["coords"] = new_coords

        # Save the updated metadata as a JSON file
        output_json_file = os.path.join(output_dir, json_file)
        with open(output_json_file, "w") as json_file:
            json.dump(metadata, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images and update metadata.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input directory containing NPZ and JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output directory for resized images and updated JSON files.",
    )
    parser.add_argument(
        "-w", "--width", type=int, default=512, help="Target width for resized images."
    )
    parser.add_argument(
        "-h",
        "--height",
        type=int,
        default=512,
        help="Target height for resized images.",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    target_size = (args.width, args.height)

    resize_and_save_images(input_dir, output_dir, target_size)
