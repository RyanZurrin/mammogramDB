import os
import numpy as np
from tqdm import tqdm


def check_file(file_path):
    try:
        with np.load(file_path) as data:
            # You can add any conditions you need to check here
            if "data" not in data:
                return file_path
        return None
    except Exception as e:
        return file_path


def find_error_files(directory):
    error_files = []
    npz_files = [f for f in os.listdir(directory) if f.endswith(".npz")]
    for file in tqdm(npz_files, desc="Checking files"):
        full_path = os.path.join(directory, file)
        if result := check_file(full_path):
            error_files.append(result)
    return error_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find problematic npz files")
    parser.add_argument(
        "-d", "--directory", required=True, help="Directory containing the npz files"
    )

    args = parser.parse_args()
    directory = args.directory

    if error_files := find_error_files(directory):
        print("Problematic files found:")
        for f in error_files:
            print(f)
    else:
        print("No problematic files found.")
