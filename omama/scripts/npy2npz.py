import numpy as np
import os
import argparse
from tqdm import tqdm


def convert_npy_to_npz(input_dir, output_dir=None, delete_original=False):
    # If no output directory is specified, use the input directory
    output_dir = output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    for npy_file in tqdm(npy_files, desc="Converting"):
        # Load the NPY file
        array_data = np.load(os.path.join(input_dir, npy_file), allow_pickle=True)

        # Create the corresponding .npz filename and save
        npz_filename = f"{os.path.splitext(npy_file)[0]}.npz"
        np.savez_compressed(os.path.join(output_dir, npz_filename), data=array_data)

        # Remove the original .npy file if the delete flag is set
        if delete_original:
            os.remove(os.path.join(input_dir, npy_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy files to .npz format.")
    parser.add_argument("input_dir", type=str, help="Directory containing .npy files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for .npz files. If not specified, it uses the input directory.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the original .npy files after conversion.",
    )
    args = parser.parse_args()

    convert_npy_to_npz(args.input_dir, args.output_dir, args.delete)
