import numpy as np
import os
import argparse
from tqdm import tqdm


def convert_npz_to_npy(input_dir, output_dir=None, delete_original=False):
    # If no output directory is specified, use the input directory
    output_dir = output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]

    for npz_file in tqdm(npz_files, desc="Converting"):
        # Load the NPZ file
        loaded = np.load(os.path.join(input_dir, npz_file), allow_pickle=True)

        # Get the name of the saved array (assuming only one array per file)
        array_name = list(loaded.keys())[0]
        array_data = loaded[array_name]

        # Create the corresponding .npy filename and save
        npy_filename = f"{os.path.splitext(npz_file)[0]}.npy"
        np.save(os.path.join(output_dir, npy_filename), array_data)

        # Remove the original .npz file if the delete flag is set
        if delete_original:
            os.remove(os.path.join(input_dir, npz_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npz files to .npy format.")
    parser.add_argument("input_dir", type=str, help="Directory containing .npz files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for .npy files. If not specified, it uses the input directory.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the original .npz files after conversion.",
    )
    args = parser.parse_args()

    convert_npz_to_npy(args.input_dir, args.output_dir, args.delete)
