import os
import json
import numpy as np
import argparse
from tqdm import tqdm


def process_batch(
    _npz_files, _json_files, _images_mm, _labels_mm, _start_idx, _end_idx
):
    _error_files = []

    for i, (npz_filename, json_filename) in enumerate(
        tqdm(
            zip(_npz_files, _json_files),
            total=len(_npz_files),
            desc=f"Processing files [{_start_idx}, {_end_idx}]",
        )
    ):
        npz_filepath = os.path.join(directory, npz_filename)
        json_filepath = os.path.join(directory, json_filename)

        try:
            with np.load(npz_filepath, allow_pickle=True) as img:
                _images_mm[i + _start_idx] = img["data"]

            with open(json_filepath, "r") as label:
                label_data = json.load(label)
                _labels_mm[i + _start_idx] = [label_data["coords"], label_data["score"]]

        except Exception as e:
            print(f"Error processing file {npz_filename}: {e}")
            _error_files.append(npz_filename)

    return _error_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate npz and json files into two numpy arrays"
    )
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Directory containing the npz and json files",
    )
    parser.add_argument(
        "-i",
        "--images_output",
        default="all_images.npy",
        help="Output file name for the images array",
    )
    parser.add_argument(
        "-l",
        "--labels_output",
        default="all_labels.npy",
        help="Output file name for the labels array",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1000,
        help="Number of files to process in each batch. Default is 1000.",
    )

    args = parser.parse_args()

    directory = args.directory
    images_path = os.path.join(directory, args.images_output)
    labels_path = os.path.join(directory, args.labels_output)

    error_files = []

    batch_size = args.batch_size
    npz_files = sorted([f for f in os.listdir(directory) if f.endswith(".npz")])
    json_files = sorted([f for f in os.listdir(directory) if f.endswith(".json")])

    total_files = len(npz_files)
    num_batches = total_files // batch_size + (total_files % batch_size != 0)

    images_mm = np.memmap(images_path, dtype="object", mode="w+", shape=(total_files,))
    labels_mm = np.memmap(
        labels_path, dtype="object", mode="w+", shape=(total_files, 2)
    )

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, total_files)

        batch_error_files = process_batch(
            npz_files[start_idx:end_idx],
            json_files[start_idx:end_idx],
            images_mm,
            labels_mm,
            start_idx,
            end_idx,
        )
        error_files.extend(batch_error_files)

    with open("errors.txt", "w") as f:
        for error_file in error_files:
            f.write(f"{error_file}\n")
