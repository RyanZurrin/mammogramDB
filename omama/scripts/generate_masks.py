import numpy as np
import json
import os
from tqdm import tqdm
import argparse


def generate_masks_from_coords(npz_folder, json_folder, output_folder):
    npz_files = sorted(os.listdir(npz_folder))
    json_files = sorted(os.listdir(json_folder))

    for npz_file, json_file in tqdm(zip(npz_files, json_files), total=len(npz_files)):
        # Load the image from .npz file
        npz_path = os.path.join(npz_folder, npz_file)
        image_data = np.load(npz_path)
        image = image_data["data"]

        # Load bounding box coordinates from .json file
        json_path = os.path.join(json_folder, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)
            coords = data["coords"]

        # Generate a binary mask
        mask = np.zeros(image.shape[:2], dtype=bool)
        x1, y1, x2, y2 = coords
        mask[y1:y2, x1:x2] = 1

        # Save the mask
        mask_filename = npz_file.replace(".npz", "_mask.npz")
        mask_path = os.path.join(output_folder, mask_filename)
        np.savez(mask_path, data=mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate masks from bounding box coordinates."
    )
    parser.add_argument(
        "-n",
        "--npz_folder",
        type=str,
        required=True,
        help="Path to the folder containing .npz image files.",
    )
    parser.add_argument(
        "-j",
        "--json_folder",
        type=str,
        required=True,
        help="Path to the folder containing .json files with coordinates.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder where generated masks will be saved.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    generate_masks_from_coords(args.npz_folder, args.json_folder, args.output_folder)
